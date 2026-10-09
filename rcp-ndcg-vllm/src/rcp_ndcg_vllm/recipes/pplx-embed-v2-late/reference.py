"""Reference implementation for ``perplexity-ai/pplx-embed-v2-late-0.6b`` -- the model card's own code path.

The model is a multimodal late-interaction (multi-vector) retriever on a Qwen3.5-0.6B
backbone: one L2-normalized 128-dim vector per kept token, retrieval score = fp32 MaxSim.
This script is the runnable ground truth the equivalence harness compares the served
engine against; it runs as a subprocess in its own environment (never inside the harness,
which holds no torch).

Published code path: **sentence-transformers**, and the checkpoint needs no custom Python
code at all: the card's usage is ``sentence_transformers.MultiVectorEncoder`` (the export
is native ST modules -- Transformer, Dense, MultiVectorMask, Normalize), used exactly as
the card prescribes: ``encode_query`` / ``encode_document`` and ``model.similarity``
(MaxSim). Nothing here re-implements the model: the prompts are the checkpoint's own
(``config_sentence_transformers.json``: ``"[Q] "`` / ``"[D] "``, prepended verbatim --
the card: "PyLate inserts Q/D markers at the second position; this model expects them
first"), the per-task caps are the checkpoint's own (``sentence_bert_config.json``:
query_length 1024, document_length 4096, applied as sentence-transformers'
``truncation="longest_first"`` right cut of the whole rendered prompt) and the
document-side token filter is the checkpoint's own ``MultiVectorMask`` (the 32 ASCII
punctuation token positions dropped from the DOCUMENT scoring mask; queries keep
everything). The recipe declares both mechanisms
(``client.query_max_tokens``/``max_tokens``/``document_skip_token_ids``) and its
``over_cap_cut_differs`` deviation covers the one corner where the card's id cut and a
text cut differ.

Reference environment (this file's own python; the checkpoint's ``requirements.txt`` at
the pinned revision, installed over the engine image's torch per the node's bootstrap):
sentence-transformers>=6.0.0,<7.0.0, transformers>=5.4.0,<6.0.0 -- a GPU host for the
embed mode (the fp32 checkpoint is 2,377,321,184 bytes; CPU also works but the wave runs
cuda). ``--mode render`` needs only ``huggingface_hub`` and ``tokenizers``: it reads the
pinned ``config_sentence_transformers.json``, ``sentence_bert_config.json`` and
``tokenizer.json`` and never imports torch.

Subprocess contract (``rcp_ndcg_vllm.equivalence.reference.run_reference``):

    reference.py --mode <render|embed|media> --pairs <file> --out <file> --tokenizer <repo>@<rev> \
                 --recipe <resolved-recipe.json> [--device <d>]

- ``render``: ``{"rows": [{"index", "shape", "text"}]}`` -- the prompt text the card's
  model reads, one render per declared shape per row (the row's query and its first
  document, which is what the harness samples per shape): the sentence-transformers
  prompt prepended verbatim (no strip -- ``prepend_prompt_to_texts``), then the
  pipeline's own cut (the processor's ``truncation="longest_first"`` at
  ``max_length`` = query_length 1024 or document_length 4096 from
  ``sentence_bert_config.json``; a right cut of the whole rendered prompt's ids, the
  fixed head kept). The cut runs in the ``tokenizers`` library the checkpoint's fast
  tokenizer drives, and the kept text is located by the kept tokens' character offsets
  on the prompt, never a decode. One corner has no exact text: when the cut ends inside
  a character that spans several byte-level tokens (an emoji, say), the card reads the
  character's leading bytes; the text keeps whole tokens of whole characters, so its
  ids are a strict prefix of the card's. Everywhere else the text runs to the first
  dropped token's start offset (so a combining mark the NFC normaliser leaves out of a
  token's offsets is kept) and its ids equal the card's -- measured on English,
  punctuation, CJK, emoji and NFD inputs; any other input class is unmeasured. Nothing
  here follows the product client's cut; where the two differ the recipe declares
  ``over_cap_cut_differs``.
- ``embed``: ``{"rows": [{"index", "query_vectors": [[...]], "document_vectors": [[[...]]]}]}``
  -- fp16 per-token matrices (n_kept, 128), one per query and one per document, exactly
  as the pipeline returns them (MultiVectorMask-filtered, L2-normalized by the
  checkpoint's own 3_Normalize module). The checkpoint loaded is the one the harness
  passes in ``--recipe`` (the resolved variant's model and revision): one reference
  serves the whole family, and the 9b's stage-2 comparison must not run against the
  0.6b checkpoint.
- ``media``: ``{"rows": [{"index", "side", "placement", "media": [{"kind", "width", "height", "tokens"}]}]}``
  -- for every pairs row carrying ``media``, what the card's path consumes per side: the
  image document's own resize (the shipped processor's ``smart_resize`` bounds, read
  from the pinned ``processor_config.json``) and its media tokens (the vision start and
  end markers around the merged patches -- the media item's own count, which the client's
  ``content_media_tokens`` and the engine's with/without-media prompt difference both
  use; the ``[D] `` prompt token is the document's text and is counted in the text
  budget, not here); a side the card's path cannot encode -- an
  image query (the card's usage encodes queries as text), an image beside text or
  several images ("Mixed text+image inputs are not supported") -- is
  ``{"index", "side", "refused"}``. Needs ``huggingface_hub`` and Pillow only.

The environment pins live in the recipe's ``notes`` (one home per concept); this
docstring states the code path and the modes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODEL = "perplexity-ai/pplx-embed-v2-late-0.6b"
REVISION = "8fc2de24534aa3610d85fa59c463313a5f096455"
OUTPUT_DIM = 128  # 1_Dense/config.json out_features at the pinned revision; asserted against the model


def _checkpoint_file(tokenizer_spec: str, filename: str) -> Any:
    """One small JSON file of the pinned checkpoint (``config_sentence_transformers.json``, ...).

    The spec carries the same ``repo@revision`` the recipe's client declares; the revision
    pins the file, so the reference reads exactly what the recipe serves.
    """
    from huggingface_hub import hf_hub_download

    repo, _, revision = tokenizer_spec.partition("@")
    path = hf_hub_download(repo or MODEL, filename, revision=revision or REVISION)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_query(prompt: str, text: str) -> str:
    """The query prompt, exactly as sentence-transformers prepends it: verbatim, no strip."""
    return prompt + (text or "")


def format_document(prompt: str, text: str) -> str:
    """The document prompt, exactly as sentence-transformers prepends it: verbatim, no strip.

    An empty document renders the bare prompt (the reference's empty-document branch:
    one kept token, the prefix id, which is no skiplist word).
    """
    return prompt + (text or "")


def prompts(config_st: dict[str, Any]) -> dict[str, str]:
    """The checkpoint's own prompts (config_sentence_transformers.json): ``query`` / ``document``."""
    return {name: str(config_st["prompts"][name]) for name in ("query", "document")}


def card_cut(prompt: str, backend: Any, cap: int) -> str:
    """The pipeline's cut of one rendered prompt, as the text the model reads (see the module docstring).

    ``preprocess`` tokenizes the rendered prompt with ``truncation="longest_first"`` and
    ``max_length=cap``: a right cut of the prompt's ids (the post-processor's tokens
    included) to ``cap``. The kept text is the prompt up to the first dropped token's
    START offset, so it holds everything the kept tokens cover -- including characters
    the tokenizer's NFC normaliser leaves out of a token's reported offsets (a
    decomposed combining mark). When the first dropped token shares its character with a
    kept one (a character spanning several byte-level tokens), the text stops before
    every kept token reaching into that character.
    """
    backend.no_truncation()
    if len(backend.encode(prompt, add_special_tokens=True).ids) <= cap:
        return prompt
    whole = backend.encode(prompt, add_special_tokens=True)
    backend.enable_truncation(max_length=cap, strategy="longest_first", direction="right")
    try:
        kept = backend.encode(prompt, add_special_tokens=True)
    finally:
        backend.no_truncation()
    end = max(offset_end for _, offset_end in kept.offsets)
    first_dropped_start = whole.offsets[len(kept.ids)][0] if len(whole.ids) > len(kept.ids) else len(prompt)
    if first_dropped_start >= end:
        end = first_dropped_start  # no character split: keep everything up to the next token
    else:
        # The first dropped token belongs to a character a kept token also covers: drop every kept token
        # that reaches into it, so the text holds whole tokens of whole characters (a prefix of the ids).
        end = first_dropped_start
        for start, token_end in reversed(kept.offsets):
            if token_end > end:
                end = min(end, start)
    return prompt[:end]


def _backend(tokenizer_spec: str) -> Any:
    """The ``tokenizers`` library over the spec's ``tokenizer.json`` (a local file or directory, else the
    Hub file at the spec's revision) -- the Rust tokenizer the pipeline's processor runs."""
    from tokenizers import Tokenizer

    path = Path(tokenizer_spec).expanduser()
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        repo, _, revision = tokenizer_spec.partition("@")
        path = Path(hf_hub_download(repo or MODEL, "tokenizer.json", revision=revision or REVISION))
    backend = Tokenizer.from_file(str(path))
    backend.no_truncation()
    backend.no_padding()
    return backend


def caps(sentence_bert: dict[str, Any]) -> dict[str, int]:
    """The pipeline's caps per shape: query_length / document_length (sentence_bert_config.json).

    sentence-transformers passes these directly as the per-task ``max_length``; the text
    config's ``max_position_embeddings`` (262,144) is orders of magnitude above both and
    caps nothing.
    """
    return {
        "query": int(sentence_bert["query_length"]),
        "document": int(sentence_bert["document_length"]),
    }


def render(pairs: list[dict[str, Any]], backend: Any, cap: dict[str, int], prompt: dict[str, str]) -> dict[str, Any]:
    """The prompt the card's model reads per declared shape for every pairs row (query and first document):
    the sentence-transformers prompt prepended verbatim, then the pipeline's own cut."""
    out_rows: list[dict[str, Any]] = []
    for index, row in enumerate(pairs):
        query = format_query(prompt["query"], str(row["query"]))
        document = format_document(prompt["document"], str(row["documents"][0]))
        out_rows.append({"index": index, "shape": "query", "text": card_cut(query, backend, cap["query"])})
        out_rows.append({"index": index, "shape": "document", "text": card_cut(document, backend, cap["document"])})
    return {"rows": out_rows}


def _load_reference(model: str, revision: str, device: str) -> Any:
    """Load the resolved variant's MultiVectorEncoder and assert the surface this file relies on.

    The model and revision are the harness's ``--recipe`` facts (the variant the engine serves),
    never this file's 0.6b defaults; the assertion guards an unmeasured dependency surface: a
    sentence-transformers change that removes ``tokenizer``/``encode_query``/``encode_document``
    fails here, on the first line, instead of mid-check.
    """
    from sentence_transformers import MultiVectorEncoder

    reference = MultiVectorEncoder(
        model,
        revision=revision,
        device=device,
    )
    missing = [attr for attr in ("tokenizer", "encode_query", "encode_document") if not hasattr(reference, attr)]
    if missing:
        raise AttributeError(
            f"sentence-transformers MultiVectorEncoder surface changed; missing: {missing} "
            "(reference.py relies on tokenizer/encode_query/encode_document)"
        )
    return reference


def embed(rows: list[dict[str, Any]], device: str, model: str, revision: str) -> dict[str, Any]:
    """Per-token fp16 matrices through the card's own ``encode_query``/``encode_document``.

    The checkpoint loads at its config dtype (float32): the card's usage passes no dtype
    override, so the reference runs the checkpoint's own fp32 -- the served-vs-reference
    cast is the recipe's declared served-dtype deviation.
    """
    reference = _load_reference(model, revision, device)
    out_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        query_out = reference.encode_query([str(row["query"])])
        document_out = reference.encode_document([str(document) for document in row["documents"]])
        width = int(query_out[0].shape[-1])
        if width != OUTPUT_DIM:
            raise ValueError(f"the checkpoint's output width changed: expected {OUTPUT_DIM}, got {width}")
        out_rows.append(
            {
                "index": index,
                "query_vectors": _fp16_lists(query_out[0]),
                "document_vectors": [_fp16_lists(document) for document in document_out],
            }
        )
    return {"rows": out_rows}


def _fp16_lists(matrix: Any) -> list[list[float]]:
    """One fp16 token-vector matrix as nested floats."""
    import numpy as np

    return [[float(value) for value in vector] for vector in np.asarray(matrix, dtype=np.float16)]


def card_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """The resize the shipped processor runs (transformers' Qwen2-VL image processing): each edge
    rounded to the factor, the area floored into the budget (at least one factor) or ceiled up to
    the floor; an aspect ratio over 200 is refused."""
    import math

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar, w_bar = round(height / factor) * factor, round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def media(rows: list[dict[str, Any]], tokenizer_spec: str) -> dict[str, Any]:
    """The media stage's reference side: what the card's path consumes for every side carrying media.

    An image document is encoded ALONE (the card: "Use separate encoding calls for
    text-only and image-only batches. Mixed text+image inputs are not supported."); its
    render is the ``[D] `` system prompt (the checkpoint's own chat template emits it
    first) followed by the vision start and end markers around the image patches, so the
    document's prompt costs the merged patches plus three tokens -- but the media ITEM's
    count (what this mode reports, what the client's ``content_media_tokens`` counts and
    what the engine's with/without-media prompt-token difference measures) is the merged
    patches plus the two vision markers: the ``[D] `` prompt token is the document's text.
    The resize is the shipped
    processor's own (``processor_config.json``: ``min_pixels``/``max_pixels`` are its
    effective bounds, ``patch_size`` x ``merge_size`` the factor). A side the card's path
    cannot encode -- an image query (the card's usage encodes queries as text), an image
    beside a text, several images -- is reported refused, with the card's reason.
    """
    import base64
    import io

    from PIL import Image

    processor = _checkpoint_file(tokenizer_spec, "processor_config.json")["image_processor"]
    factor = int(processor["patch_size"]) * int(processor["merge_size"])
    min_pixels = int(processor["min_pixels"])
    max_pixels = int(processor["max_pixels"])
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        entries = row.get("media") or {}
        if entries.get("query"):
            out.append(
                {"index": index, "side": "query", "refused": "Queries must be text; images are supported as documents."}
            )
        for position, items in enumerate(entries.get("documents") or []):
            if not items:
                continue
            side = f"document {position}"
            if len(items) > 1 or str(row["documents"][position]):
                out.append(
                    {
                        "index": index,
                        "side": side,
                        "refused": "Mixed text+image inputs are not supported; one image per document.",
                    }
                )
                continue
            payload = base64.b64decode(str(items[0]["uri"]).split(",", 1)[1])
            with Image.open(io.BytesIO(payload)) as handle:
                width, height = handle.size
            resized_h, resized_w = card_resize(height, width, factor, min_pixels, max_pixels)
            tokens = (resized_h // factor) * (resized_w // factor) + 2
            out.append(
                {
                    "index": index,
                    "side": side,
                    "placement": ["image"],
                    "media": [{"kind": "image", "width": resized_w, "height": resized_h, "tokens": tokens}],
                }
            )
    return {"rows": out}


def _recipe_facts(recipe_path: str | None) -> tuple[str, str]:
    """The variant's ``(model, revision)`` from the resolved recipe JSON the harness passes.

    The harness always passes it; the 0.6b constants are the standalone-run fallback only.
    """
    if not recipe_path:
        return MODEL, REVISION
    recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    return str(recipe["model"]), str(recipe["revision"])


def main() -> int:
    parser = argparse.ArgumentParser(description="the pplx-embed-v2-late family reference (the model card's path)")
    parser.add_argument("--mode", required=True, choices=["render", "embed", "media"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True, help="the recipe's tokenizer spec (repo@revision)")
    parser.add_argument(
        "--recipe",
        required=True,
        help="the resolved recipe JSON the harness passed (the variant's id, model and revision)",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        prompt = prompts(_checkpoint_file(args.tokenizer, "config_sentence_transformers.json"))
        cap = caps(_checkpoint_file(args.tokenizer, "sentence_bert_config.json"))
        document = render(rows, backend=_backend(args.tokenizer), cap=cap, prompt=prompt)
    elif args.mode == "media":
        document = media(rows, args.tokenizer)
    else:
        model, revision = _recipe_facts(args.recipe)
        document = embed(rows, args.device, model, revision)
    Path(args.out).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
