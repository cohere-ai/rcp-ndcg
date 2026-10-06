"""Reference implementation for recipe ``qwen3-vl-reranker-2b`` (Qwen/Qwen3-VL-Reranker-2B).

The serving referent is the model card's own script, ``scripts/qwen3_vl_reranker.py`` at
revision 4bd860ac4f15ad1897a214615cccc700f8f71818 (a transformers checkpoint, so
``reference.kind: transformers``):

- prompt = ``processor.apply_chat_template([system, user], tokenize=False,
  add_generation_prompt=True)`` through the repo's ``chat_template.jinja``, with one system
  message (the judge instruction) and one user message built as ``<Instruct>: <instruction>``,
  ``<Query>:`` + query parts, ``\\n<Document>:`` + document parts;
- score = ``sigmoid((W_yes - W_no) . h_last)``: a linear built from the tied LM head's rows for
  yes=9693 and no=2152 (the snapshot vocab, equal to the repo's ``1_LogitScore/config.json``),
  left padding, ``MAX_LENGTH = 8192``.

Runs as a subprocess in the reference environment (``requirements-reference.txt`` beside this
file: torch, transformers, qwen-vl-utils) -- never inside the harness process:

    reference.py --mode render --pairs <pairs.jsonl> --out <out.json> --tokenizer <spec> [--device cpu]
    reference.py --mode score  --pairs <pairs.jsonl> --out <out.json> --tokenizer <spec> --device cuda:0

Modes and output JSON (the harness's contract):

- ``render`` -- ``{"rows": [{"index", "shape", "text"}]}``, the prompt text per pairs row (the
  pair shape). Two paths, both declared here: the card's own path (``transformers``'
  ``apply_chat_template`` over the repo's ``chat_template.jinja``) when transformers imports,
  otherwise a direct construction of the same ChatML string (same message builder, the template
  rendered by hand). The direct path exists so stage 1 runs on CPU with tokenizer files only;
  the served-side template check proves the shipped template file renders the same string.
- ``score`` -- ``{"rows": [{"index", "scores": [...]}]}`` on the recipe's
  ``reference.score_scale`` (probability: the sigmoid above). Needs torch, transformers and the
  ~4.0 GB weights; never runs on the CPU stage (stage 2 needs a served engine anyway).

Deviations from the card's script, all declared:

- ``render`` uses ``AutoTokenizer.apply_chat_template``; the card calls the processor's. Both
  resolve the same repo file ``chat_template.jinja`` (the processor delegates to its tokenizer).
- ``render`` always uses the card's default instruction and ignores the pairs row's
  ``instruction`` field: the recipe's endpoint never sends an instruction
  (``client.instruction: none``), so the engine's template default is what the model reads.
- The card's truncation (``truncate_tokens_optimized``: keep every special token and the first
  ``MAX_LENGTH - specials`` non-special tokens, re-append the last 5 ids) is kept for ``score``
  unchanged; it is the recipe's declared ``anchor_drop_over_cap`` deviation (over-cap pairs are
  reported non-gating), not something the served path copies.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

MAX_LENGTH = 8192
"""The card's MAX_LENGTH: the reference truncates post-template to this many tokens."""

SYSTEM_TEXT = (
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
"""The served template's fixed system line (byte-identical to the engine's qwen3_vl_reranker.jinja)."""

DEFAULT_INSTRUCTION = "Given a search query, retrieve relevant candidates that answer the query."
"""The engine template's fallback instruction (= the card script's default); the endpoint never
sends an instruction field, so this text is what every served prompt carries."""

REVISION = "4bd860ac4f15ad1897a214615cccc700f8f71818"
"""The checkpoint commit the recipe pins."""


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
    """The card script's format_mm_content for one side: the prefix text, then video, image, text.

    A side with no text and no media sends the literal text "NULL" -- the placeholder the recipe's
    ``empty_doc: send_text`` policy also sends.
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": prefix}]
    if not text and not image and not video:
        content.append({"type": "text", "text": "NULL"})
        return content
    if video:
        if isinstance(video, str):
            content.append({"type": "video", "video": video, "fps": 1.0, "max_frames": 64})
        else:
            content.append({"type": "video", "video": video, "total_pixels": 4 * 2 * 1310720})
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

    ``load`` and ``score`` need torch, transformers and the ~4.0 GB weights (the GPU wave's
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

        language_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path, torch_dtype=torch.bfloat16
        ).to(device)
        self.model = language_model.model
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, padding_side="left")
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
        except Exception:  # text-only input, or qwen-vl-utils unavailable: the card's fallback
            images, videos = None, None
            video_kwargs = {"do_sample_frames": False}
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
        return padded

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
                "score() needs the ~4.0 GB weights: call load(device) first (the CPU stage runs "
                "render only; stage 2 scores against the served engine)"
            )
        scores: list[float] = []
        for doc in docs:
            messages = format_mm_instruction(query, doc, instruction)
            inputs = self.tokenize([messages])
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            batch_scores = self.model(**inputs).last_hidden_state[:, -1]
            sigmoid_input = self.score_linear(batch_scores)
            scores.append(float(torch.sigmoid(sigmoid_input).squeeze(-1).item()))
        return scores


def _render_messages_transformers(messages: list[dict[str, Any]], tokenizer: Any) -> str:
    """The card's render path: the repo chat template applied by transformers (tokenize off)."""
    return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


def render_pair(query: dict[str, Any], doc: dict[str, Any], instruction: str, tokenizer: Any) -> str:
    """The reference prompt text for one pair: the card's messages through the repo chat template.

    With a transformers tokenizer the card's own path runs; without one (the CPU stage-1
    environment carries tokenizer files only) :func:`render_pair_direct` builds the same string.
    """
    messages = format_mm_instruction(query, doc, instruction)
    if tokenizer is not None:
        return _render_messages_transformers(messages, tokenizer)
    return render_pair_direct(query, doc, instruction)


def _load_tokenizer_for_render(spec: str) -> Any:
    """The tokenizer the card's render path needs, from a snapshot dir or a repo@revision spec.

    Returns ``None`` when transformers is unavailable -- the direct render path takes over, and
    the output records which path ran.
    """
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError:
        return None
    repo, _, revision = spec.partition("@")
    if Path(repo).exists():
        return AutoTokenizer.from_pretrained(repo)
    return AutoTokenizer.from_pretrained(repo, revision=revision or None)


def main() -> int:
    """The subprocess CLI the harness launches (``--mode render|score``)."""
    parser = argparse.ArgumentParser(description="the qwen3-vl-reranker-2b reference")
    parser.add_argument("--mode", required=True, choices=["render", "score"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows_raw = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    document: dict[str, Any] = {}
    if args.mode == "render":
        tokenizer = _load_tokenizer_for_render(args.tokenizer)
        for index, row in enumerate(rows_raw):
            # The endpoint never sends an instruction (client.instruction: none), so the render
            # is the card's default-instruction prompt; the row's instruction field is ignored.
            text = render_pair(
                {"text": str(row["query"])},
                {"text": str(row["documents"][0])},
                DEFAULT_INSTRUCTION,
                tokenizer,
            )
            rows.append({"index": index, "shape": "pair", "text": text})
        document = {
            "rows": rows,
            "render_path": "transformers-apply_chat_template" if tokenizer is not None else "direct-chatml-mirror",
        }
    else:
        reference = Qwen3VLRerankerReference(_model_path(args.tokenizer)).load(args.device)
        for index, row in enumerate(rows_raw):
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


def _model_path(tokenizer_spec: str) -> str:
    """The weights location the score mode loads: the snapshot dir, or the repo id@revision."""
    return tokenizer_spec


if __name__ == "__main__":
    raise SystemExit(main())
