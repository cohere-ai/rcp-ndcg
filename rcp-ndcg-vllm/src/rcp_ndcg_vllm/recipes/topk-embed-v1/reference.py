"""The topk-embed-v1 family's one reference -- the model card's own code path.

The model is a multimodal late-interaction (multi-vector) retriever on a Qwen3.5 backbone: one
L2-normalized ``dim``-wide vector per kept token (2048 for -small, 1024 for -xsmall), retrieval
score = fp32 MaxSim. This script is the runnable ground truth the equivalence harness compares the
served engine against; it runs as a subprocess in its own environment (never inside the harness,
which holds no torch). One file serves every variant of the family (decision 34): the variant
travels with the invocation, in the resolved recipe the harness passes as ``--recipe``;
:data:`MODEL` and :data:`REVISION` name the family's -small checkpoint as the standalone defaults,
and the CLI reads the variant's own ``model``/``revision``/``client.dim`` from the recipe.

Published code path: **sentence-transformers** -- the checkpoint's ``config_sentence_transformers.json``
declares ``model_type: MultiVectorEncoder`` and its ``modules.json`` the remote ``topk_embed_st``
wrapper, so the model card's usage is ``sentence_transformers.MultiVectorEncoder`` with
``trust_remote_code=True``. Nothing here re-implements the model: ``encode_query``/``encode_document``
are the shipped wrapper's own calls, including its prompts (``Query: `` + text.strip() and
``Document: `` + text, stripped -- from the checkpoint's own ``config.json``, never hardcoded), its
document-side keep-mask (drop ``scoring_skip_ids`` positions; queries keep everything; image documents
keep only image-patch positions) and its right truncation at query_length 1024 / document_length 8192.

Reference environment (this file's own python; the checkpoint's ``requirements.txt`` at the pinned
revision): torch==2.11.0, torchvision==0.26.0, transformers==5.9.0, sentence-transformers==6.0.1,
flash-linear-attention==0.5.1, kernels==0.14.1, safetensors>=0.7.0, Pillow>=12.0 -- a GPU for the
embed mode (about 4.4 GB of weights). ``--mode render`` needs only ``huggingface_hub`` and
``tokenizers``: it reads the pinned ``config.json``, ``tokenizer_config.json``,
``sentence_bert_config.json`` and ``tokenizer.json`` and never imports torch.

Subprocess contract (``rcp_ndcg_test.equivalence.reference.run_reference``):

    reference.py --mode <render|embed|media> --pairs <file> --out <file> --tokenizer <repo>@<rev> [--device <d>]

- ``render``: ``{"rows": [{"index", "shape", "text"}]}`` -- the prompt text the card's model reads, one
  render per declared shape per row (the row's query and its first document, which is what the harness
  samples per shape): the wrapper's ``format_query``/``format_document``, then the wrapper's own cut
  (topk_embed_st.py:75-77: the processor's ``truncation=True`` at ``max_length`` = query_length 1024 or
  document_length 8192 from ``sentence_bert_config.json``, capped by the text config's
  ``max_position_embeddings``; a right cut of the whole prompt's ids, the fixed head kept). The cut runs
  in the ``tokenizers`` library the processor's fast tokenizer drives, and the kept text is located by
  the kept tokens' character offsets on the prompt, never a decode. One corner has no exact text: when
  the cut ends inside a character that spans several byte-level tokens (an emoji, say), the card reads
  the character's leading bytes; the text keeps whole tokens of whole characters, so its ids are a strict
  prefix of the card's. Everywhere else the text runs to the first dropped token's start offset (so a
  combining mark the NFC normaliser leaves out of a token's offsets is kept) and its ids equal the card's
  -- measured on English, punctuation, CJK, emoji and NFD inputs; any other input class is
  unmeasured. Nothing here follows the product client's cut; where the two differ the recipe declares
  ``over_cap_cut_differs``.
- ``embed``: ``{"rows": [{"index", "query_vectors": [[...]], "document_vectors": [[[...]]]}]}`` -- fp16
  per-token matrices (n_kept, the variant's width: 2048 for -small, 1024 for -xsmall), one per
  query and one per document, exactly as the wrapper
  returns them (keep-masked).
- ``media``: ``{"rows": [{"index", "side", "placement", "media": [{"kind", "width", "height", "tokens"}]}]}``
  -- for every pairs row carrying ``media``, what the wrapper consumes per image document (its own resize
  and token count, read from the pinned ``config.json`` and ``processor_config.json``); a side it cannot
  encode is ``{"index", "side", "refused"}``. Needs ``huggingface_hub`` and Pillow only.

The environment pins live in the recipe's ``notes`` (one home per concept); this docstring states the
code path and the modes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODEL = "topk-io/topk-embed-v1-small"
REVISION = "e54485ebab921f2c18c4d092b3f4c40dcca26781"
OUTPUT_DIM = 2048  # the -small checkpoint's config.json dim / output_dim; the variant's own dim arrives in --recipe
#: The two constants above are the family's -small checkpoint and the standalone defaults; every
#: served variant is loaded through the resolved recipe's own pair.


def _checkpoint_file(tokenizer_spec: str, filename: str) -> Any:
    """One small JSON file of the pinned checkpoint (``config.json``, ``tokenizer_config.json``).

    The spec carries the same ``repo@revision`` the recipe's client declares; the revision pins the
    file, so the reference reads exactly what the recipe serves.
    """
    import json

    from huggingface_hub import hf_hub_download

    repo, _, revision = tokenizer_spec.partition("@")
    path = hf_hub_download(repo or MODEL, filename, revision=revision or REVISION)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_query(config: dict[str, Any], text: str) -> str:
    """The query prompt, exactly as the shipped wrapper builds it (topk_embed_st.py:56-57)."""
    return str(config["query_template"]) + (text or "").strip()


def format_document(config: dict[str, Any], text: str, eos_token: str) -> str:
    """The document prompt, exactly as the shipped wrapper builds it (topk_embed_st.py:58-60).

    The whole render is stripped; an empty document falls back to the tokenizer's eos TEXT (itself in
    ``scoring_skip_ids``, so it keeps no vector), then to ``"."``.
    """
    return (str(config["document_prompt"]) + (text or "")).strip() or eos_token or "."


def card_cut(prompt: str, backend: Any, cap: int) -> str:
    """The wrapper's cut of one formatted prompt, as the text the model reads (see the module docstring).

    ``self.processor(text=..., truncation=True, max_length=cap)`` right-cuts the prompt's ids (the
    post-processor's tokens included) to ``cap``. The kept text is the prompt up to the first dropped
    token's START offset, so it holds everything the kept tokens cover -- including characters the
    tokenizer's NFC normaliser leaves out of a token's reported offsets (a decomposed combining mark).
    When the first dropped token shares its character with a kept one (a character spanning several
    byte-level tokens), the text stops before every kept token reaching into that character.
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
    Hub file at the spec's revision) -- the Rust tokenizer the wrapper's processor runs. The file's
    embedded truncation is reset: the wrapper passes its own ``max_length`` on every call."""
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


def caps(config: dict[str, Any], sentence_bert: dict[str, Any]) -> dict[str, int]:
    """The wrapper's caps per shape: query_length / document_length, each capped by the text config's
    ``max_position_embeddings`` (topk_embed_st.py:75-76)."""
    ceiling = int(config["text_config"]["max_position_embeddings"])
    return {
        "query": min(int(sentence_bert["query_length"]), ceiling),
        "document": min(int(sentence_bert["document_length"]), ceiling),
    }


def render(
    config: dict[str, Any], eos_token: str, rows: list[dict[str, Any]], *, backend: Any, cap: dict[str, int]
) -> dict[str, Any]:
    """The prompt the card's model reads per declared shape for every pairs row (query and first document):
    the wrapper's formatting, then the wrapper's own cut."""
    out_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        query = format_query(config, str(row["query"]))
        document = format_document(config, str(row["documents"][0]), eos_token)
        out_rows.append({"index": index, "shape": "query", "text": card_cut(query, backend, cap["query"])})
        out_rows.append({"index": index, "shape": "document", "text": card_cut(document, backend, cap["document"])})
    return {"rows": out_rows}


def _alias_qwen3_5_layer_type(layer_cls: Any = None) -> None:
    """Alias transformers' renamed ``layer_type`` attribute for the checkpoint's remote code.

    The checkpoint's ``hf_backbone.patch_packing`` reads ``layer.layer_type`` (the transformers 5.9
    attribute its own ``requirements.txt`` pins); transformers 5.17's ``Qwen3_5DecoderLayer`` renamed it
    to ``block_type``. When only the new name exists, the old one is aliased to it -- a declared shim, so
    the reference runs on the image's transformers stack (GPU-E1: the unshimmed load crashed at
    ``hf_backbone.py:178`` with ``AttributeError: 'Qwen3_5DecoderLayer' object has no attribute
    'layer_type'``). Nothing else is patched: a checkpoint whose remote code needs more of 5.9 fails
    loudly at its own call.

    Args:
        layer_cls: The decoder-layer class to shim; ``None`` imports transformers' own. A seam for the
            CPU test, which runs without transformers.
    """
    if layer_cls is None:
        try:
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer as layer_cls
        except ImportError:  # a transformers without the class: the remote code will say so itself
            return
    if not hasattr(layer_cls, "layer_type") and hasattr(layer_cls, "block_type"):
        layer_cls.layer_type = property(lambda self: self.block_type)  # type: ignore[attr-defined]


def _load_reference(device: str, model: str, revision: str, expected_dim: int) -> Any:
    """Load the model card's MultiVectorEncoder and assert the surface this file relies on.

    The assertion guards an unmeasured dependency surface: a sentence-transformers change that
    removes ``tokenizer``/``config``/``encode_query``/``encode_document`` fails here, on the first
    line, instead of mid-check. ``model``/``revision`` are the variant's own pair and
    ``expected_dim`` its declared ``client.dim``, both from the resolved recipe.
    """
    from sentence_transformers import MultiVectorEncoder

    _alias_qwen3_5_layer_type()
    model_obj = MultiVectorEncoder(
        model,
        revision=revision,
        trust_remote_code=True,
        device=device,
        model_kwargs={"dtype": "bfloat16"},
    )
    missing = [
        attr for attr in ("tokenizer", "config", "encode_query", "encode_document") if not hasattr(model_obj, attr)
    ]
    if missing:
        raise AttributeError(
            f"sentence-transformers MultiVectorEncoder surface changed; missing: {missing} "
            "(reference.py relies on tokenizer/config/encode_query/encode_document)"
        )
    width = int(model_obj.config.output_dim or model_obj.config.dim)
    if width != expected_dim:
        raise ValueError(f"the checkpoint's output width changed: expected {expected_dim}, got {width}")
    return model_obj


def embed(rows: list[dict[str, Any]], device: str, model: str, revision: str, expected_dim: int) -> dict[str, Any]:
    """Per-token fp16 matrices through the card's own ``encode_query``/``encode_document``."""
    model_obj = _load_reference(device, model, revision, expected_dim)
    out_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        query_out = model_obj.encode_query([str(row["query"])])
        document_out = model_obj.encode_document([str(document) for document in row["documents"]])
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
    """The resize the wrapper's ``_image_row`` runs (topk_embed_st.py:117-126): the image processor module's
    own ``smart_resize`` (transformers' Qwen2-VL image processing) with factor patch x merge, min_pixels the
    processor's ``size.shortest_edge`` and max_pixels ``image_token_budget`` x factor^2 -- each edge rounded to
    the factor, the area floored into the budget (at least one factor) or ceiled up to the floor; an aspect
    ratio over 200 is refused."""
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
    """The media stage's reference side: what the wrapper's model consumes for every side carrying media.

    The wrapper encodes an image document ALONE (``preprocess``: "Encode text or images in separate batches";
    an image query is refused, topk_embed_st.py:71): its prompt is the chat template's user turn around one
    image, the image resized by :func:`card_resize` and costing its merged patches plus the vision start and
    end markers (the template's prefix and suffix around the image pads).  A side the wrapper cannot encode --
    an image query, an image beside a text, several images -- is reported refused, with the wrapper's reason.
    """
    import base64
    import io

    from PIL import Image

    config = _checkpoint_file(tokenizer_spec, "config.json")
    processor = _checkpoint_file(tokenizer_spec, "processor_config.json")["image_processor"]
    factor = int(processor["patch_size"]) * int(processor["merge_size"])
    min_pixels = int(processor["size"]["shortest_edge"])
    max_pixels = int(config["image_token_budget"]) * factor**2
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
                out.append({"index": index, "side": side, "refused": "Encode text or images in separate batches."})
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


def _check_recipe_variant(recipe_path: str, tokenizer_spec: str) -> None:
    """The resolved recipe names the same checkpoint the tokenizer spec pins.

    The checkpoint loads from the tokenizer spec's repository (the variant's ``client.tokenizer``);
    the resolved recipe is the variant's identity, so a mismatch means the harness resolved a
    different variant than this reference would serve.  A local tokenizer path (stage 1) carries no
    repository identity: nothing to compare.
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
    parser = argparse.ArgumentParser(description="the topk-embed-v1-small reference (the model card's path)")
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
    _check_recipe_variant(args.recipe, args.tokenizer)

    rows = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    variant_model, variant_revision = str(recipe["model"]), str(recipe["revision"])
    variant_dim = int((recipe.get("client") or {})["dim"])
    if args.mode == "render":
        config = _checkpoint_file(args.tokenizer, "config.json")
        eos_token = str(_checkpoint_file(args.tokenizer, "tokenizer_config.json").get("eos_token") or "")
        cap = caps(config, _checkpoint_file(args.tokenizer, "sentence_bert_config.json"))
        document = render(config, eos_token, rows, backend=_backend(args.tokenizer), cap=cap)
    elif args.mode == "media":
        document = media(rows, args.tokenizer)
    else:
        document = embed(rows, args.device, variant_model, variant_revision, variant_dim)
    Path(args.out).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
