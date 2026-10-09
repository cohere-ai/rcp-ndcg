"""Reference implementation for the ``pplx-embed-v1`` family -- the model card's own code path.

The family is a dense text embedder on a diffusion-continued-pretrained Qwen3 backbone
with bidirectional attention: one mean-pooled vector per text (1024 dims at 0.6B, 2560 at
4B), no instruction and no prompt prefix (the card's "Instruction: No"), Matryoshka-capable
(the card's table: MRL Yes), and -- natively -- an int8/binary *storage* view of the
pooled vector (the card: "natively produce *unnormalized* int8-quantized embeddings.
Ensure that you compare them via *cosine similarity*"). This script is the runnable
ground truth the equivalence harness compares the served engine against; it runs as a
subprocess in its own environment (never inside the harness, which holds no torch).

Published code path: **sentence-transformers**, with the checkpoint's own remote code
enabled (the card's usage: ``SentenceTransformer(repo, trust_remote_code=True)``). The
repo carries a native ``modules.json`` pipeline -- ``Transformer`` (the remote
``PPLXQwen3Model``), ``1_Pooling`` (mean tokens) and ``st_quantize.FlexibleQuantizer`` --
and the card's usage is ``model.encode(texts)``. This reference reproduces the card's
pipeline up to, and deliberately NOT including, the trailing ``FlexibleQuantizer``:

- the served engine returns the checkpoint's *float* mean-pooled hidden state (vLLM's
  embed pooler over the Qwen3 backbone); the card's int8 tanh quantisation is a
  storage/transfer format applied by a sentence-transformers module *after* pooling, and
  comparing the served float vectors against a lossy re-encoding of the reference would
  measure the quantiser, not the serving path. The reference therefore loads the card's
  own pipeline and stops before the quantiser, and the recipe's notes declare the
  decision. The quantiser's module class is asserted by name, so a checkpoint revision
  that changes the pipeline fails loudly here instead of silently comparing something
  else.

The per-task caps: the checkpoint ships no ``sentence_bert_config.json``, so
sentence-transformers' ``Transformer`` caps its tokenizer's ``model_max_length``
(``tokenizer_config.json``: 131072) at the model's ``max_position_embeddings``
(``config.json``: 32768) and uses that as ``max_seq_length``; every text is right-cut to
32768 tokens with ``truncation="longest_first"`` before the model runs. The reference
reproduces exactly that cap (the min of the two files' numbers) and renders the card's
cut, never the client's content-only cut; where the two differ (an id cut inside a
multi-token character) the recipe declares ``over_cap_cut_differs``.

Reference environment (this file's own python; installed over the engine image's torch per
the node's bootstrap): sentence-transformers>=6.0.0,<7.0.0 and
transformers>=5.0.0,<6.0.0 (the checkpoint's own ``config.json``
``transformers_version`` floor). ``--mode render`` needs only ``huggingface_hub`` and
``tokenizers``: it reads the pinned ``config.json``, ``tokenizer_config.json`` and
``tokenizer.json`` and never imports torch.

Subprocess contract (``rcp_ndcg_test.equivalence.reference.run_reference``):

    reference.py --mode <render|embed> --pairs <file> --out <file> --tokenizer <repo>@<rev> \
                 --recipe <json> [--device <d>]

- ``render``: ``{"rows": [{"index", "shape", "text"}]}`` -- the text the card's model
  reads, one render per declared shape per row (the row's query and its first document):
  the raw text (this family prepends no prompt, the card's "Instruction: No"), then the
  pipeline's own right cut at 32768 ids. The cut runs in the ``tokenizers`` library the
  checkpoint's fast tokenizer drives, and the kept text is located by the kept tokens'
  character offsets, never a decode. One corner has no exact text: when the cut ends
  inside a character spanning several byte-level tokens (an emoji, say), the card reads
  the character's leading bytes; the text keeps whole tokens of whole characters, so its
  ids are a strict prefix of the card's. Everywhere else the text runs to the first
  dropped token's start offset and its ids equal the card's. Nothing here follows the
  product client's cut; where the two differ the recipe declares ``over_cap_cut_differs``.
- ``embed``: ``{"rows": [{"index", "query_vectors": [...], "document_vectors": [[...], ...]}]}``
  -- one float vector per text, exactly the pooled encoder output the served engine
  returns (mean pooling over the tokens, no quantiser; see the module docstring).

The per-variant facts (the model id and revision) travel through ``--recipe``, the
resolved recipe JSON the harness passes: one reference serves the whole family.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

#: The reference's fallback model/revision, used only when ``--recipe`` is absent (the
#: harness always passes it): the family's first variant. Kept here so the module can be
#: exercised standalone.
DEFAULT_MODEL = "perplexity-ai/pplx-embed-v1-0.6b"
DEFAULT_REVISION = "2c4d510dd4a732063c31a0f70193e35067b51fd8"

#: The module class names the card's ``modules.json`` pipeline must carry, in order
#: (``Transformer`` -> ``Pooling`` -> the storage quantiser this reference stops before).
EXPECTED_MODULES = ("Transformer", "Pooling", "FlexibleQuantizer")


def _recipe_facts(recipe_path: str | None) -> tuple[str, str]:
    """The variant's ``(model, revision)`` from the resolved recipe JSON the harness passes."""
    if not recipe_path:
        return DEFAULT_MODEL, DEFAULT_REVISION
    recipe = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    return str(recipe["model"]), str(recipe["revision"])


def _checkpoint_file(tokenizer_spec: str, filename: str) -> Any:
    """One small JSON file of the pinned checkpoint (``config.json``, ``tokenizer_config.json``).

    The spec carries the same ``repo@revision`` the recipe's client declares; the revision
    pins the file, so the reference reads exactly what the recipe serves.
    """
    from huggingface_hub import hf_hub_download

    repo, _, revision = tokenizer_spec.partition("@")
    path = hf_hub_download(repo or DEFAULT_MODEL, filename, revision=revision or DEFAULT_REVISION)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def caps(tokenizer_spec: str) -> int:
    """The card's cut cap: the min of the tokenizer's ``model_max_length`` and the model's
    ``max_position_embeddings``.

    sentence-transformers' ``Transformer.__init__`` caps ``tokenizer.model_max_length`` at
    ``config.max_position_embeddings``; the checkpoint ships no ``sentence_bert_config.json``,
    so the capped ``model_max_length`` is also ``max_seq_length`` -- the ``max_length`` every
    text is tokenised with (``truncation="longest_first"``). Both numbers are read from the
    pinned files, never hardcoded.
    """
    tokenizer_config = _checkpoint_file(tokenizer_spec, "tokenizer_config.json")
    config = _checkpoint_file(tokenizer_spec, "config.json")
    return min(int(tokenizer_config["model_max_length"]), int(config["max_position_embeddings"]))


def card_cut(text: str, backend: Any, cap: int) -> str:
    """The pipeline's cut of one text, as the text the model reads (see the module docstring).

    ``preprocess`` tokenizes the text with ``truncation="longest_first"`` and
    ``max_length=cap``: a right cut of the text's ids (the post-processor's tokens
    included, a no-op for this tokenizer) to ``cap``. The kept text is the text up to the
    first dropped token's START offset, so it holds everything the kept tokens cover --
    including characters the tokenizer's NFC normaliser leaves out of a token's reported
    offsets (a decomposed combining mark). When the first dropped token shares its
    character with a kept one (a character spanning several byte-level tokens), the text
    stops before every kept token reaching into that character.
    """
    backend.no_truncation()
    if len(backend.encode(text, add_special_tokens=True).ids) <= cap:
        return text
    whole = backend.encode(text, add_special_tokens=True)
    backend.enable_truncation(max_length=cap, strategy="longest_first", direction="right")
    try:
        kept = backend.encode(text, add_special_tokens=True)
    finally:
        backend.no_truncation()
    end = max(offset_end for _, offset_end in kept.offsets)
    first_dropped_start = whole.offsets[len(kept.ids)][0] if len(whole.ids) > len(kept.ids) else len(text)
    if first_dropped_start >= end:
        end = first_dropped_start  # no character split: keep everything up to the next token
    else:
        # The first dropped token belongs to a character a kept token also covers: drop every kept token
        # that reaches into it, so the text holds whole tokens of whole characters (a prefix of the ids).
        end = first_dropped_start
        for start, token_end in reversed(kept.offsets):
            if token_end > end:
                end = min(end, start)
    return text[:end]


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
        path = Path(hf_hub_download(repo or DEFAULT_MODEL, "tokenizer.json", revision=revision or DEFAULT_REVISION))
    backend = Tokenizer.from_file(str(path))
    backend.no_truncation()
    backend.no_padding()
    return backend


def render(pairs: list[dict[str, Any]], backend: Any, cap: int) -> dict[str, Any]:
    """The text the card's model reads per declared shape for every pairs row (query and first document):
    the raw text (no prompt), then the pipeline's own right cut at ``cap``."""
    out_rows: list[dict[str, Any]] = []
    for index, row in enumerate(pairs):
        query = str(row["query"])
        document = str(row["documents"][0])
        out_rows.append({"index": index, "shape": "query", "text": card_cut(query, backend, cap)})
        out_rows.append({"index": index, "shape": "document", "text": card_cut(document, backend, cap)})
    return {"rows": out_rows}


def _load_encoder(model: str, revision: str, device: str) -> Any:
    """The card's ``SentenceTransformer`` pipeline without its trailing storage quantiser.

    The full pipeline is loaded exactly as the card prescribes
    (``trust_remote_code=True``: the repo's ``configuration.py``/``modeling.py`` and
    ``st_quantize.py`` are the checkpoint's own code), its module classes are asserted
    against :data:`EXPECTED_MODULES` (a changed pipeline fails here, loudly), and the
    encoder is rebuilt from the modules before the quantiser. The served engine returns
    the pooled float vector, so the reference compares like with like; the quantiser's
    int8 view is a storage format (see the module docstring).
    """
    from sentence_transformers import SentenceTransformer

    full = SentenceTransformer(model, revision=revision, trust_remote_code=True, device=device)
    names = tuple(module.__class__.__name__ for module in full)
    if names != EXPECTED_MODULES:
        raise ValueError(
            f"the checkpoint's sentence-transformers pipeline changed: expected modules {EXPECTED_MODULES}, "
            f"got {names}; the reference's float-encoder contract (stop before the storage quantiser) "
            "does not match this revision"
        )
    return SentenceTransformer(modules=list(full)[:-1])


def _vector(values: Any) -> list[float]:
    """One pooled embedding as a plain float list (the harness compares vectors by cosine)."""
    import numpy as np

    return [float(value) for value in np.asarray(values, dtype=np.float32)]


def embed(rows: list[dict[str, Any]], device: str, model: str, revision: str) -> dict[str, Any]:
    """One float vector per text through the card's own pipeline (mean pooling, no quantiser).

    The checkpoint loads at its config dtype (float32): the card's usage passes no dtype
    override, so the reference runs the checkpoint's own fp32 -- the served-vs-reference
    cast is the recipe's declared served-dtype deviation.
    """
    encoder = _load_encoder(model, revision, device)
    out_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        query_vectors = encoder.encode([str(row["query"])], normalize_embeddings=False)
        document_vectors = encoder.encode([str(document) for document in row["documents"]], normalize_embeddings=False)
        out_rows.append(
            {
                "index": index,
                "query_vectors": _vector(query_vectors[0]),
                "document_vectors": [_vector(vector) for vector in document_vectors],
            }
        )
    return {"rows": out_rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="the pplx-embed-v1 reference (the model card's path)")
    parser.add_argument("--mode", required=True, choices=["render", "embed"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True, help="the recipe's tokenizer spec (repo@revision)")
    parser.add_argument(
        "--recipe",
        required=False,
        help="the resolved recipe JSON the harness passed (the variant's model and revision)",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        document = render(rows, backend=_backend(args.tokenizer), cap=caps(args.tokenizer))
    else:
        model, revision = _recipe_facts(args.recipe)
        document = embed(rows, args.device, model, revision)
    Path(args.out).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
