"""Reference implementation for ``topk-io/topk-embed-v1-small`` -- the model card's own code path.

The model is a multimodal late-interaction (multi-vector) retriever on a Qwen3.5-2B backbone: one
L2-normalized 2048-dim vector per kept token, retrieval score = fp32 MaxSim. This script is the
runnable ground truth the equivalence harness compares the served engine against; it runs as a
subprocess in its own environment (never inside the harness, which holds no torch).

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

Subprocess contract (``rcp_ndcg_vllm.equivalence.reference.run_reference``):

    reference.py --mode <render|embed> --pairs <file> --out <file> --tokenizer <repo>@<rev> [--device <d>]

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
  per-token matrices (n_kept, 2048), one per query and one per document, exactly as the wrapper
  returns them (keep-masked).

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
OUTPUT_DIM = 2048  # config.json dim / output_dim at the pinned revision; asserted against the model


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


def _load_reference(device: str) -> Any:
    """Load the model card's MultiVectorEncoder and assert the surface this file relies on.

    The assertion guards an unmeasured dependency surface: a sentence-transformers change that
    removes ``tokenizer``/``config``/``encode_query``/``encode_document`` fails here, on the first
    line, instead of mid-check.
    """
    from sentence_transformers import MultiVectorEncoder

    model = MultiVectorEncoder(
        MODEL,
        revision=REVISION,
        trust_remote_code=True,
        device=device,
        model_kwargs={"dtype": "bfloat16"},
    )
    missing = [attr for attr in ("tokenizer", "config", "encode_query", "encode_document") if not hasattr(model, attr)]
    if missing:
        raise AttributeError(
            f"sentence-transformers MultiVectorEncoder surface changed; missing: {missing} "
            "(reference.py relies on tokenizer/config/encode_query/encode_document)"
        )
    width = int(model.config.output_dim or model.config.dim)
    if width != OUTPUT_DIM:
        raise ValueError(f"the checkpoint's output width changed: expected {OUTPUT_DIM}, got {width}")
    return model


def embed(rows: list[dict[str, Any]], device: str) -> dict[str, Any]:
    """Per-token fp16 matrices through the card's own ``encode_query``/``encode_document``."""
    model = _load_reference(device)
    out_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        query_out = model.encode_query([str(row["query"])])
        document_out = model.encode_document([str(document) for document in row["documents"]])
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


def main() -> int:
    parser = argparse.ArgumentParser(description="the topk-embed-v1-small reference (the model card's path)")
    parser.add_argument("--mode", required=True, choices=["render", "embed"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True, help="the recipe's tokenizer spec (repo@revision)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        config = _checkpoint_file(args.tokenizer, "config.json")
        eos_token = str(_checkpoint_file(args.tokenizer, "tokenizer_config.json").get("eos_token") or "")
        cap = caps(config, _checkpoint_file(args.tokenizer, "sentence_bert_config.json"))
        document = render(config, eos_token, rows, backend=_backend(args.tokenizer), cap=cap)
    else:
        document = embed(rows, args.device)
    Path(args.out).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
