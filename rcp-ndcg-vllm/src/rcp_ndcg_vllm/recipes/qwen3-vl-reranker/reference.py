"""Reference implementation for the ``qwen3-vl-reranker`` family (the ``--tokenizer`` spec the harness
resolves from the variant's recipe names the checkpoint; ONE reference serves every size, decision 34).

The serving referent is the model card's own script, ``scripts/qwen3_vl_reranker.py``, byte-identical at
every variant's pinned revision (2b:
https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B/blob/4bd860ac4f15ad1897a214615cccc700f8f71818/scripts/qwen3_vl_reranker.py,
8b: https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B/blob/b212dc8c91a8164aef1ea2de9c1a867611e75c04/scripts/qwen3_vl_reranker.py),
sha256 bd5d2f5d97fc4a738864d93f6b15d8850243e60da4484f3ea78867a46efdebd6 (a transformers checkpoint, so
``reference.kind: transformers``):

- prompt = ``processor.apply_chat_template([system, user], tokenize=False,
  add_generation_prompt=True)`` through the repo's ``chat_template.jinja``, with one system
  message (the judge instruction) and one user message built as ``<Instruct>: <instruction>``,
  ``<Query>:`` + query parts, ``\\n<Document>:`` + document parts;
- score = ``sigmoid((W_yes - W_no) . h_last)``: a linear built from the tied LM head's rows for
  yes=9693 and no=2152 (the snapshot vocab, equal to the repo's ``1_LogitScore/config.json``),
  left padding, ``MAX_LENGTH = 8192``.

Runs as a subprocess in the reference environment (``reference.in``/``reference.lock`` beside this
file: torch, transformers, qwen-vl-utils) -- never inside the harness process:

    reference.py --mode render --pairs <pairs.jsonl> --out <out.json> --tokenizer <spec> [--device cpu]
    reference.py --mode score  --pairs <pairs.jsonl> --out <out.json> --tokenizer <spec> --device cuda:0

Modes and output JSON (the harness's contract):

- ``render`` -- ``{"rows": [{"index", "shape", "query", "documents"}]}``: the content spans
  the card's model reads per pairs row -- the harness compares spans, so this is the required
  output FORMAT (the frame is the engine's chat template; stage-1 ``template_render_check``
  compares the file and the declared shape). The spans are the card's own: its prompt
  (:func:`render_pair_direct`, the repo chat template's string for the card's messages), its own
  over-cap cut (:func:`truncate_tokens_optimized` on all but the last 5 ids, the 5 re-appended),
  and the query and document text that cut keeps, located by the kept tokens' character offsets
  on the rendered prompt -- verbatim prefixes, never a decode. Nothing here follows the product
  client's cut (its query share, its settle-once rule); where the two cuts differ the recipe
  declares ``over_cap_cut_differs``. Media columns are refused (below).
- ``media`` -- ``{"rows": [{"index", "side", "placement", "media": [{"kind", "width", "height",
  "tokens"}]}]}``: for every pairs row carrying ``media``, per side, what the card's model consumes (the
  parts after the side's prefix in the card's order, each image's size after the card's resize --
  qwen-vl-utils' ``smart_resize`` at factor 32 under the image part's 4096..1310720 px -- and its tokens).
  Needs Pillow only.
- ``score`` -- ``{"rows": [{"index", "scores": [...]}]}`` on the recipe's
  ``reference.score_scale`` (probability: the sigmoid above). Needs torch, transformers and the
  checkpoint's weights; never runs on the CPU stage (stage 2 needs a served engine anyway).

Deviations from the card's script, all declared:

- ``render`` reports the card's spans rather than the card's whole prompt string: the harness's
  contract is spans (the frame is the engine's chat template's job).
- ``render`` always uses the card's default instruction and ignores the pairs row's
  ``instruction`` field: the recipe's endpoint never sends an instruction
  (``client.instruction: none``), so the engine's template default is what the model reads.
- The card's truncation (``truncate_tokens_optimized``: keep every special token and the first
  ``MAX_LENGTH - specials`` non-special tokens of all but the last 5 ids, re-append those 5) runs
  in both modes. It keeps the assistant tail (the anchor) and cuts differently from the client:
  the recipe's declared ``over_cap_cut_differs`` (over-cap pairs are reported non-gating).
- The card's video branch (fps 1 / max_frames 64 containers, ``total_pixels`` frame arrays) is
  refused, not mirrored: the recipe declares ``input: [text, image]``, and the family's ONE video
  policy (64 uniformly spaced frames per clip -- the ``qwen3-vl-embedding`` family's
  ``client.video_policy``) supersedes the card's container sampler. A video-bearing row is a loud
  error in every mode, never a silent sampling at either rule. Image columns ride the card's
  per-document message builder in ``score`` only (``query_image`` / ``documents_images``);
  ``render`` refuses them (its contract is the text spans the client ships).
- ``tokenize`` mirrors the card's vision-wiring fallback: when ``process_vision_info`` raises
  (a media decode failure in score mode), the card re-renders the prompt as a NULL-only user
  turn; the reference copies that behavior verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

MAX_LENGTH = 8192
"""The card's MAX_LENGTH: the reference truncates post-template to this many tokens."""

TAIL_IDS = 5
"""The card's protected tail: ``tokenize`` truncates all but the last 5 ids and re-appends them (the
assistant header the model scores at)."""

SYSTEM_TEXT = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
"""The served template's fixed system line (byte-identical to the engine's qwen3_vl_reranker.jinja)."""

DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
"""The engine template's fallback instruction (= the card script's default); the endpoint never
sends an instruction field, so this text is what every served prompt carries."""


def _special(name: str) -> str:
    """The literal form of one of the tokenizer's added specials, built from its name."""
    return "<|" + name + "|>"


IM_START = _special("im_start")
IM_END = _special("im_end")
VISION_START = _special("vision_start")
VISION_END = _special("vision_end")
IMAGE_PAD = _special("image_pad")
VIDEO_PAD = _special("video_pad")
IMAGE_PLACEHOLDER = VISION_START + IMAGE_PAD + VISION_END
VIDEO_PLACEHOLDER = VISION_START + VIDEO_PAD + VISION_END


def _content_parts(prefix: str, text: str, image: Any = None, video: Any = None) -> list[dict[str, Any]]:
    """The card script's format_mm_content for one side: the prefix text, then image, text.

    A side with no text and no media sends the literal text "NULL" -- the placeholder the recipe's
    ``empty_doc: send_text`` policy also sends. One declared deviation from the card's mirror: the
    card's video branch is refused here (see the module docstring) -- ``input: [text, image]`` and
    the family's one video policy live elsewhere; a video-bearing side is a loud error.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": prefix}]
    if not text and not image and not video:
        content.append({"type": "text", "text": "NULL"})
        return content
    if video:
        raise SystemExit(
            "a side carries a video, and this recipe declares input [text, image] -- video is out "
            "of its serving form. The family's one video policy (64 uniformly spaced frames per "
            "clip, the container as video_url under a pinned engine) lives in the "
            "qwen3-vl-embedding family recipe; drop the video or hold the row for that recipe"
        )
    if image:
        content.append({"type": "image", "image": image, "min_pixels": 4096, "max_pixels": 1310720})
    if text:
        content.append({"type": "text", "text": text})
    return content


def format_mm_instruction(query: dict[str, Any], doc: dict[str, Any], instruction: str) -> list[dict[str, Any]]:
    """The card script's format_mm_instruction: one system message + one user message.

    ``query``/``doc`` carry optional ``text``, ``image`` and ``video`` keys; media parts ride
    inside the content list in the card's order (prefix, video, image, text).
    """
    contents: list[dict[str, Any]] = [{"type": "text", "text": "<Instruct>: " + instruction}]
    contents.extend(_content_parts("<Query>:", str(query.get("text") or ""), query.get("image"), query.get("video")))
    contents.extend(_content_parts("\n<Document>:", str(doc.get("text") or ""), doc.get("image"), doc.get("video")))
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_TEXT}]},
        {"role": "user", "content": contents},
    ]


def _render_part_direct(part: dict[str, Any]) -> str:
    """One content part as the repo chat template renders it: text inline, media as placeholders."""
    kind = part.get("type")
    if kind == "text":
        return str(part.get("text") or "")
    if kind in ("image", "image_url"):
        return IMAGE_PLACEHOLDER
    if kind in ("video", "video_url"):
        return VIDEO_PLACEHOLDER
    raise ValueError(f"unhandled content part type {kind!r}")


def render_pair_direct(query: dict[str, Any], doc: dict[str, Any], instruction: str) -> str:
    """The reference prompt, built by hand (the no-transformers render path).

    The same ChatML string the repo's ``chat_template.jinja`` renders for
    :func:`format_mm_instruction`'s messages with ``add_generation_prompt=True``: the system
    line, the user turn with every content part in order, then the assistant header. The recipe's
    stage-1 template check proves the served template file renders this exact string.
    """
    user_parts = format_mm_instruction(query, doc, instruction)[1]["content"]
    return (
        IM_START
        + "system\n"
        + SYSTEM_TEXT
        + IM_END
        + "\n"
        + IM_START
        + "user\n"
        + "".join(_render_part_direct(part) for part in user_parts)
        + IM_END
        + "\n"
        + IM_START
        + "assistant\n"
    )


def truncate_tokens_optimized(tokens: list[int], max_length: int, special_tokens: set[int]) -> list[int]:
    """The card script's truncation: keep every special token and the first
    ``max_length - num_special`` non-special tokens, so a caller-protected tail can be re-appended.

    Args:
        tokens: The prompt's token ids.
        max_length: The cap, in tokens (the card's MAX_LENGTH = 8192).
        special_tokens: The tokenizer's special ids, kept wherever they sit.

    Returns:
        The kept ids: every special, the head non-specials up to the cap, never the anchor-less
        tail -- the caller re-appends the last ids it protects (the card keeps 5).
    """
    if len(tokens) <= max_length:
        return list(tokens)
    keep = max_length - sum(1 for token in tokens if token in special_tokens)
    out: list[int] = []
    kept = 0
    for token in tokens:
        if token in special_tokens:
            out.append(token)
        elif kept < keep:
            out.append(token)
            kept += 1
    return out


class Qwen3VLRerankerReference:
    """The card script's reranker: ``load(device)`` then ``score(query, docs, instruction)``.

    ``load`` and ``score`` need torch, transformers and the checkpoint's weights (the GPU wave's
    reference environment); ``render`` needs the tokenizer files only.
    """

    def __init__(self, model_name_or_path: str | os.PathLike[str]) -> None:
        self.model_name_or_path = os.fspath(model_name_or_path)
        self.model: Any = None
        self.processor: Any = None
        self.score_linear: Any = None

    def _true_false_ids(self) -> tuple[int, int]:
        """The yes/no token ids from the snapshot's vocab, cross-checked against the repo's
        1_LogitScore/config.json when the file is present (never hardcoded)."""
        if self.processor is None:
            raise RuntimeError("call load(device) first")
        tokenizer = self.processor.tokenizer
        true_id = tokenizer.convert_tokens_to_ids("yes")
        false_id = tokenizer.convert_tokens_to_ids("no")
        config_path = Path(self.model_name_or_path) / "1_LogitScore" / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if int(config["true_token_id"]) != int(true_id) or int(config["false_token_id"]) != int(false_id):
                raise ValueError(
                    f"the snapshot's vocab (yes={true_id}, no={false_id}) disagrees with "
                    f"{config_path} (true={config['true_token_id']}, false={config['false_token_id']})"
                )
        return int(true_id), int(false_id)

    def load(self, device: str) -> Qwen3VLRerankerReference:
        """The card script's load: the backbone in bf16 plus a float32-score linear built from
        the LM head's yes/no rows (W_yes - W_no), on ``device``.

        Args:
            device: ``cpu`` or ``cuda:0`` -- where the weights go.

        Returns:
            self, with :attr:`model` (the backbone), :attr:`processor` (left padding) and
            :attr:`score_linear` (the fp32 head) set.
        """
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        repo, revision = _repo_and_revision(self.model_name_or_path)
        if revision is None:
            language_model = Qwen3VLForConditionalGeneration.from_pretrained(repo, torch_dtype=torch.bfloat16).to(
                device
            )
            self.processor = AutoProcessor.from_pretrained(repo, padding_side="left")
        else:
            language_model = Qwen3VLForConditionalGeneration.from_pretrained(
                repo, revision=revision, torch_dtype=torch.bfloat16
            ).to(device)
            self.processor = AutoProcessor.from_pretrained(repo, revision=revision, padding_side="left")
        self.model = language_model.model
        self.model.eval()
        true_id, false_id = self._true_false_ids()
        weight_yes = language_model.lm_head.weight.data[true_id]
        weight_no = language_model.lm_head.weight.data[false_id]
        hidden = weight_yes.size()[0]
        linear = torch.nn.Linear(hidden, 1, bias=False)
        with torch.no_grad():
            linear.weight[0] = weight_yes - weight_no
        self.score_linear = linear.to(device).to(self.model.dtype)
        self.score_linear.eval()
        return self

    def tokenize(self, pairs: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """The card script's tokenize: render, run the vision wiring, processor-encode, truncate
        the body (all but the last 5 ids) and left-pad to the longest, capped at MAX_LENGTH.

        Args:
            pairs: One formatted message list per document (``format_mm_instruction`` output).

        Returns:
            The processor's batch encoding, left-padded, every input at most MAX_LENGTH + 5
            tokens (the card's own over-cap shape).
        """
        text = self.processor.apply_chat_template(pairs, tokenize=False, add_generation_prompt=True)
        try:
            from qwen_vl_utils import process_vision_info

            images, videos, video_kwargs = process_vision_info(
                pairs, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
            )
        except Exception as error:  # the card's fallback: drop the media, log, and score a NULL user turn
            print(f"error in processing vision info: {error}", file=sys.stderr)
            images, videos = None, None
            video_kwargs = {"do_sample_frames": False}
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": "NULL"}]}],
                add_generation_prompt=True,
                tokenize=False,
            )
        if videos is not None:
            videos, video_metadatas = zip(*videos, strict=True)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = self.processor(
            text=text,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            truncation=False,
            padding=False,
            do_resize=False,
            **video_kwargs,
        )
        special_ids = set(self.processor.tokenizer.all_special_ids)
        ids = [truncate_tokens_optimized(x[:-5], MAX_LENGTH, special_ids) + x[-5:] for x in inputs["input_ids"]]
        padded = self.processor.tokenizer.pad(
            {"input_ids": ids}, padding=True, return_tensors="pt", max_length=MAX_LENGTH
        )
        # The card merges the padded keys back into the processor's full inputs, keeping
        # pixel_values / image_grid_thw for image-bearing rows; returning the pad dict alone
        # would drop them and the backbone call would score garbage or fail.
        for key, value in padded.items():
            inputs[key] = value
        return inputs

    def score(self, query: dict[str, Any], docs: list[dict[str, Any]], instruction: str) -> list[float]:
        """Sigmoid scores for one query against its documents, on the card's scale.

        Args:
            query: ``{"text": ..., "image": ..., "video": ...}`` (media optional).
            docs: One document dict per candidate, same keys.
            instruction: The instruction text (the recipe sends none: pass DEFAULT_INSTRUCTION).

        Returns:
            One probability per document, ``sigmoid((W_yes - W_no) . h_last)`` in [0, 1].
        """
        import torch

        if self.model is None or self.score_linear is None:
            raise RuntimeError(
                "score() needs the checkpoint's weights: call load(device) first (the CPU stage runs "
                "render only; stage 2 scores against the served engine)"
            )
        scores: list[float] = []
        for doc in docs:
            messages = format_mm_instruction(query, doc, instruction)
            inputs = self.tokenize([messages])
            # Move the tensors only: transformers' processors also return list-valued keys (the
            # processor's own metadata), and `.to()` on those crashes (GPU-E1).
            inputs = {
                key: (value.to(self.model.device) if isinstance(value, torch.Tensor) else value)
                for key, value in inputs.items()
            }
            batch_scores = self.model(**inputs).last_hidden_state[:, -1]
            sigmoid_input = self.score_linear(batch_scores)
            scores.append(float(torch.sigmoid(sigmoid_input).squeeze(-1).item()))
        return scores


# ---------------------------------------------------------------------------
# render mode (stage 1): the card's content spans after its own cut, no torch.
# ---------------------------------------------------------------------------


def _raw_tokenizer(spec: str) -> Any:
    """The render path's tokenizer: the ``tokenizers`` library over the spec's ``tokenizer.json``
    (a local directory or file, else the Hub file at ``repo@revision``) -- the same Rust engine the
    product counts with, torch-free."""
    from tokenizers import Tokenizer

    path = Path(spec)
    if path.is_dir():
        return Tokenizer.from_file(str(path / "tokenizer.json"))
    if path.is_file():
        return Tokenizer.from_file(str(path))
    from huggingface_hub import hf_hub_download

    repo, _, revision = spec.partition("@")
    file = hf_hub_download(repo, "tokenizer.json", revision=revision or None)
    return Tokenizer.from_file(file)


def card_spans(tok: Any, query: str, document: str) -> tuple[str, str]:
    """The query and document text the card's model reads for one text pair, after the card's own cut.

    The card renders the pair's prompt (:func:`render_pair_direct`), tokenizes it untruncated and keeps
    :func:`truncate_tokens_optimized` of all but the last :data:`TAIL_IDS` ids plus those ids. That keeps
    every special token and a PREFIX of the body's non-special tokens, so the text it keeps of each content
    span is a verbatim prefix of that span: the span's characters up to the last kept non-special token's
    end offset. Under the cap both spans come back whole.

    Args:
        tok: the ``tokenizers`` tokenizer of the pinned revision.
        query: the query text (non-empty).
        document: the document text (the card's "NULL" already substituted for an empty one).

    Returns:
        ``(query_span, document_span)`` as the card's model reads them.
    """
    head = render_pair_direct({"text": ""}, {"text": ""}, DEFAULT_INSTRUCTION)
    query_at = head.index("<Query>:") + len("<Query>:")
    prompt = render_pair_direct({"text": query}, {"text": document}, DEFAULT_INSTRUCTION)
    document_at = query_at + len(query) + len("\n<Document>:")
    assert prompt[query_at : query_at + len(query)] == query
    assert prompt[document_at : document_at + len(document)] == document
    encoding = tok.encode(prompt, add_special_tokens=False)
    ids, offsets = list(encoding.ids), list(encoding.offsets)
    specials = {token_id for token_id, token in tok.get_added_tokens_decoder().items() if token.special}
    body = ids[:-TAIL_IDS]
    kept_body = truncate_tokens_optimized(body, MAX_LENGTH, specials)
    if len(kept_body) == len(body):
        return query, document
    keep = MAX_LENGTH - sum(1 for token_id in body if token_id in specials)
    kept_non_special = [position for position, token_id in enumerate(body) if token_id not in specials][:keep]
    cut = offsets[kept_non_special[-1]][1] if kept_non_special else 0
    return query[: max(0, min(len(query), cut - query_at))], document[: max(0, min(len(document), cut - document_at))]


def render_rows(pairs: list[dict[str, Any]], tokenizer_spec: str) -> list[dict[str, Any]]:
    """The card's content spans per pairs row: the query span and one document span per document.

    The card's empty-side rule rides with the spans (an empty document is the literal "NULL", which the
    recipe's ``empty_doc: send_text`` also sends); an empty QUERY is refused (the product's
    ``empty_query: refuse`` default) while a whitespace-only query is the card's verbatim text (its
    ``format_mm_content`` keeps any non-empty text), and a media column is refused loudly -- the pairs contract is text,
    images ride score mode's named columns, and video is out of this recipe's serving form (see the module
    docstring). Nothing is silently dropped or defaulted.
    """
    tok = _raw_tokenizer(tokenizer_spec)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(pairs):
        carried = sorted(
            key for key in ("query_image", "query_video", "documents_images", "documents_videos") if row.get(key)
        )
        if carried:
            raise SystemExit(
                f"pairs row {index} carries media columns {carried}, and --mode render's contract "
                "is the text spans: score mode takes query_image/documents_images, "
                "and video is out of this recipe's serving form"
            )
        query = str(row["query"])
        if not query:
            # The card renders an empty side as "NULL"; the endpoint refuses an empty query instead (empty_query:
            # refuse, the declared policy), so no such request exists to compare. A whitespace-only query is not
            # empty to the card (format_mm_content's ``if text`` keeps it verbatim), and the client sends it.
            raise SystemExit(
                f"pairs row {index} carries an empty query, and the endpoint refuses one "
                "(empty_query: refuse, the declared policy): drop the row, nothing is defaulted"
            )
        documents = [str(document) if str(document) else "NULL" for document in row["documents"]]
        spans = [card_spans(tok, query, document) for document in documents]
        rows.append({"index": index, "shape": "pair", "query": spans[0][0], "documents": [span[1] for span in spans]})
    return rows


def card_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """``qwen_vl_utils.vision_process.smart_resize`` (qwen-vl-utils 0.0.14), as the card's
    ``process_vision_info(..., image_patch_size=16)`` calls it per image (factor 32, the image part's own
    min_pixels/max_pixels): each edge rounded to the factor (at least one factor), the area floored into
    ``max_pixels`` or ceiled up to ``min_pixels``; an aspect ratio over 200 is refused."""
    import math

    if max(height, width) / min(height, width) > 200:
        raise SystemExit(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


IMAGE_FACTOR = 32
"""The card's image factor: ``process_vision_info(..., image_patch_size=16)`` times the spatial merge 2."""


def media_rows(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The media stage's reference side: per pairs row and side carrying media, what the card's model
    consumes -- :func:`_content_parts`' order after the side's prefix (video, image, text; one image per side,
    video refused), each image resized by :func:`card_resize` under the image part's own min/max pixels and
    costing its merged patches plus the vision start and end markers."""
    import base64
    import io

    from PIL import Image

    out: list[dict[str, Any]] = []
    for index, row in enumerate(pairs):
        media = row.get("media") or {}
        sides = [("query", str(row["query"]), list(media.get("query") or []))]
        sides += [
            (f"document {position}", str(row["documents"][position]), list(entries or []))
            for position, entries in enumerate(media.get("documents") or [])
        ]
        for side, text, entries in sides:
            if not entries:
                continue
            if any(entry.get("kind") == "video" for entry in entries) or len(entries) > 1:
                out.append({"index": index, "side": side, "refused": "one image per side; video is out of scope"})
                continue
            payload = base64.b64decode(str(entries[0]["uri"]).split(",", 1)[1])
            with Image.open(io.BytesIO(payload)) as handle:
                width, height = handle.size
            part = _content_parts("", text, image="image")[1]
            resized_h, resized_w = card_resize(height, width, IMAGE_FACTOR, part["min_pixels"], part["max_pixels"])
            tokens = (resized_h // IMAGE_FACTOR) * (resized_w // IMAGE_FACTOR) + 2
            out.append(
                {
                    "index": index,
                    "side": side,
                    "placement": ["image"] + (["text"] if text else []),
                    "media": [{"kind": "image", "width": resized_w, "height": resized_h, "tokens": tokens}],
                }
            )
    return out


def _repo_and_revision(spec: str) -> tuple[str, str | None]:
    """A tokenizer spec (``repo@revision`` or a local path) as ``(repo_or_dir, revision)``.

    Hugging Face repo ids reject ``@``, so the harness's ``<repo>@<commit>`` spec must be split
    before any ``from_pretrained`` call; a local snapshot directory carries no revision.
    """
    repo, _, revision = spec.partition("@")
    if Path(repo).exists():
        return repo, None
    return repo, revision or None


def _check_recipe_variant(recipe_path: str, tokenizer_spec: str) -> None:
    """The resolved recipe names the same checkpoint the tokenizer spec pins.

    The checkpoint loads from the resolved recipe's ``model``/``revision`` (the harness's one variant
    contract); the tokenizer spec is the variant's ``client.tokenizer``, so a mismatch means the harness
    resolved a different variant than this reference would serve.  A local tokenizer path (stage 1)
    carries no repository identity: nothing to compare.
    """
    candidate = Path(tokenizer_spec).expanduser()
    if candidate.exists() or tokenizer_spec.startswith(("/", "./", "../", "~")) or tokenizer_spec.endswith(".json"):
        return
    recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    expected = f"{recipe['model']}@{recipe['revision']}"
    if tokenizer_spec != expected:
        raise SystemExit(
            f"the resolved recipe names {expected}, but the tokenizer spec is {tokenizer_spec!r}: the "
            "reference would load a different checkpoint than the variant it serves"
        )


def main() -> int:
    """The subprocess CLI the harness launches (``--mode render|score``)."""
    parser = argparse.ArgumentParser(description="the qwen3-vl-reranker family reference")
    parser.add_argument("--mode", required=True, choices=["render", "score", "media"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument(
        "--recipe",
        required=True,
        help="the resolved recipe JSON the harness passed (the variant's id, model and revision)",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    _check_recipe_variant(args.recipe, args.tokenizer)
    # The checkpoint comes from the resolved recipe (the harness's one variant contract), never from
    # the tokenizer spec: the spec may be a local render-only path, and the recipe is the variant's
    # identity (model + revision).
    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    model_spec = f"{recipe['model']}@{recipe['revision']}"

    rows_raw = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    document: dict[str, Any] = {}
    if args.mode == "render":
        # The harness compares content spans; these are the card's own (its prompt, its cut). The
        # instruction is the recipe's declared none: the card's default instruction, always.
        document = {"rows": render_rows(rows_raw, args.tokenizer), "render_path": "card-content-spans"}
    elif args.mode == "media":
        document = {"rows": media_rows(rows_raw)}
    else:
        reference = Qwen3VLRerankerReference(model_spec).load(args.device)
        for index, row in enumerate(rows_raw):
            if row.get("query_video") or row.get("documents_videos"):
                raise SystemExit(
                    f"pairs row {index} carries a video column, and this recipe declares input "
                    "[text, image]: the family's one video policy lives in the "
                    "qwen3-vl-embedding family recipe (64 uniformly spaced frames per clip)"
                )
            query = {"text": str(row["query"]), "image": row.get("query_image")}
            doc_images = row.get("documents_images") or []
            documents = [
                {"text": str(document_text), "image": doc_images[doc_index] if doc_index < len(doc_images) else None}
                for doc_index, document_text in enumerate(row["documents"])
            ]
            scores = reference.score(query, documents, DEFAULT_INSTRUCTION)
            rows.append({"index": index, "scores": scores})
        document = {"rows": rows}
    Path(args.out).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
