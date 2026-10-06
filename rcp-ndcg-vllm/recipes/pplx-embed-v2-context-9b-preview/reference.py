"""The reference implementation of perplexity-ai/pplx-embed-v2-context-9b-preview.

Run as a subprocess by the rcp-ndcg-vllm equivalence harness, never imported by it:

    <reference-python> reference.py --mode <render|embed> --pairs <file> --out <file> \
        --tokenizer <spec> [--device <cpu|cuda:N>]

The model's own published code path is the checkpoint's remote code at the pinned
revision (the ``custom_code`` checkpoint; the model card says to load it with
``trust_remote_code=True``), used exactly as the card prescribes:
``transformers.AutoModel.from_pretrained(..., trust_remote_code=True)`` and the
model's own ``encode_queries``/``encode``. The prompts the model reads are built by
the remote ``prepare_inputs`` (``modeling_pplx_contextual.py`` at the revision):

- a **query** is the ``[Q] `` prefix prepended as one special id (248077) before the
  tokenized query text; the pool is the masked mean over the whole sequence, the
  prefix token included (``modeling_pplx_contextual.py:46-75`` and ``:136-151``).
  Over a text wire the string ``"[Q] " + query`` tokenizes to exactly those ids
  (measured at the revision: server-side added-token parsing yields the same
  ``[248077] + body``), which is what ``--mode render`` reports;
- a **document** is one request holding all of its chunks: the string ``"[D] " +
  chunk0 + "<|chunk_sep|>" + chunk1 + ...`` tokenized with
  ``split_special_tokens=True`` -- the ``[D] `` prefix renders as the two literal
  tokens ``'[D', ']'`` (ids 62724, 60) while every ``<|chunk_sep|>`` stays the one
  added id 248079; the pool is one span mean per chunk, the prefix and the markers
  excluded from every span (``:77-135``). The recipe's ``content`` span carries the
  chunk list already joined by the marker (declared preprocessing), so ``render``
  emits the frame around the joined string and ``embed`` re-splits it on the marker
  for the model call.

Modes and outputs (written to ``--out``):

- ``render``: ``{"rows": [{"index", "shape", "text"}]}`` -- the prompt text per
  declared shape. No cut and no limit check here: the recipe's client applies the
  declared cut policy before this reference is consulted, and the model's own
  over-limit raise (``:128-133``) fires inside its own ``prepare_inputs`` in embed
  mode. Runs on the stdlib plus PyYAML (the recipe's declared shapes), so it runs
  in any interpreter that has both.
- ``embed``: ``{"rows": [{"index", "query_vectors": [[...]], "document_vectors":
  [[...]]}]}`` -- the query's vectors and the FIRST document's vectors as flat
  lists of per-chunk (2048-d) rows, the shape the harness's stage-2 comparison
  pairs against the served answer. Needs torch, transformers >= 5.4 (the
  ``Qwen3_5Model`` base) and a GPU host: the fp32 checkpoint is
  33,604,334,528 bytes, so a ``cpu`` device is refused before anything downloads.

Vectors are returned unnormalised -- the model's raw int8-valued output
(``round(tanh(x) * 127).clamp(-128, 127)``, ``:214``), float16-exact for the
values' range. Cosine comparison is scale-invariant, and the served side of the
equivalence check is the engine's own output, so no normalisation is applied here
(the product's client applies the card's ``normalize_embeddings`` on its own path).

The reference environment is pinned in ``requirements-reference.txt`` beside this
file; the harness passes ``--tokenizer`` for its own bookkeeping and the model
loads its own tokenizer from the checkpoint (the remote code's behaviour).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

__all__ = ["main"]

REPO = "perplexity-ai/pplx-embed-v2-context-9b-preview"
REVISION = "b667039ee8b438a6350fbc91bbcecd86f9d363ba"

# Pinned from the checkpoint's config.json at the revision above (the reference
# constants the remote code reads; the checkpoint's own values override the
# configuration file's 32768 defaults with 262144).
QUERY_PREFIX = "[Q] "
DOCUMENT_PREFIX = "[D] "
BOUNDARY_MARKER = "<|chunk_sep|>"
QUERY_LENGTH = 262144
DOCUMENT_LENGTH = 262144
EMBEDDING_DIM = 2048
WEIGHTS_TOTAL_BYTES = 33_604_334_528  # model.safetensors.index.json metadata.total_size


def split_chunks(document: str) -> list[str]:
    """The model's chunk list: the declared document content split on the boundary marker.

    The recipe declares the document content as the chunk list joined by the
    marker, with the chunker's guarantee that no all-special token of this
    tokenizer (the marker is the declared case) appears inside chunk text; the
    split treats every marker as a separator, exactly as the served plugin does.
    An empty document is one empty chunk, the model's own zero-vector branch.
    """
    return document.split(BOUNDARY_MARKER)


def query_text(query: str) -> str:
    """The rendered query: the ``[Q] `` prefix followed by the query text.

    The remote code prepends the prefix as the single special id 248077; the
    measured server-side parse of this string yields the same ids, which is what
    makes the text render the faithful wire form of the model's prompt.
    """
    return QUERY_PREFIX + query


def document_text(document: str) -> str:
    """The rendered document: the ``[D] `` prefix followed by the joined chunk list."""
    return DOCUMENT_PREFIX + document


def _read_pairs(path: str) -> list[dict]:
    """The pairs file: JSONL rows of ``{"query": str, "documents": [str, ...]}``."""
    pairs: list[dict] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("query"), str) or not isinstance(row.get("documents"), list):
            raise ValueError(f"{path}:{number}: expected a query string and a documents list")
        pairs.append(row)
    if not pairs:
        raise ValueError(f"{path} holds no pairs")
    return pairs


def _declared_shapes() -> tuple[str, ...]:
    """The recipe's declared request shapes, read from the sibling recipe.yaml."""
    import yaml

    client = yaml.safe_load((Path(__file__).resolve().parent / "recipe.yaml").read_text(encoding="utf-8"))["client"]
    shapes = tuple(shape for shape in ("query", "document", "pair") if shape in (client.get("template") or {}))
    return shapes or ("document",)


def render(pairs: list[dict]) -> dict:
    """The model's prompt text per declared shape: ``{"rows": [{"index", "shape", "text"}]}``.

    The document content is taken as the already-joined chunk list the recipe
    declares; the frame is the model's own prompt construction, uncut (the
    client's declared cut happens before this reference is consulted).
    """
    rows: list[dict] = []
    for index, row in enumerate(pairs):
        for shape in _declared_shapes():
            text = query_text(row["query"]) if shape == "query" else document_text(row["documents"][0])
            rows.append({"index": index, "shape": shape, "text": text})
    return {"rows": rows}


def _load(device: str):
    """The checkpoint as the model card prescribes: transformers AutoModel, trust_remote_code=True."""
    if device == "cpu":
        raise RuntimeError(
            "reference embed needs a GPU host: the fp32 checkpoint is "
            f"{WEIGHTS_TOTAL_BYTES / 1e9:.1f} GB and its code path needs transformers>=5.4 "
            "(the Qwen3_5Model base); run stage 2 on the GPU wave with --reference-python "
            "pointing at the reference environment (requirements-reference.txt beside this file)"
        )
    import torch  # noqa: F401, PLC0415  (imported only on the GPU path, never in the harness process)
    from transformers import AutoModel  # noqa: PLC0415

    model = AutoModel.from_pretrained(REPO, revision=REVISION, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model


def embed(pairs: list[dict], device: str) -> dict:
    """The model's own vectors: ``encode_queries`` for the query, ``encode`` for the documents.

    One row of output per pairs row: the query's vectors and the FIRST document's
    per-chunk vectors as flat lists -- the served side of the harness's stage-2
    comparison answers one ragged matrix per text and compares the first, so the
    wave's pairs rows carry one document each for full coverage.
    """
    model = _load(device)
    rows: list[dict] = []
    for index, row in enumerate(pairs):
        query_vectors = model.encode_queries([[row["query"]]], normalize_embeddings=False)
        documents = [split_chunks(document) for document in row["documents"]]
        document_vectors = model.encode(documents, normalize_embeddings=False)
        rows.append(
            {
                "index": index,
                "query_vectors": [[[float(value) for value in vector] for vector in query_vectors[0]]],
                "document_vectors": [[[float(value) for value in vector] for vector in document_vectors[0]]],
            }
        )
    return {"rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description="the pplx-embed-v2-context-9b-preview reference")
    parser.add_argument("--mode", required=True, choices=["render", "embed"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True, help="the recipe's tokenizer spec; the model loads its own")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    pairs = _read_pairs(args.pairs)
    document = render(pairs) if args.mode == "render" else embed(pairs, args.device)
    Path(args.out).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
