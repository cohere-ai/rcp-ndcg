"""Reference implementation for Octen/Octen-Embedding-8B -- the paper's in-process path, unchanged.

This is an exact port of the rcp-ndcg paper's encoder path -- not of the model card. Line numbers
grep-verified against the repository 2026-10-06:

- ``experiments/paper/rerankers/reference/octen.py`` (the paper's encoder today; the line cites
  below name it, with the pre-unification ``src/rcp_ndcg/retrieval/hf_dense.py`` follows kept in
  parentheses where useful): ``:33`` ``MAX_LENGTH = 8192`` ("longer texts are truncated on
  the right", the HF default truncation side); ``:54`` the tokenizer loads with
  ``padding_side="left"``; ``:55`` the model loads with ``dtype=torch.bfloat16``; ``:64`` per-batch
  ``tokenizer(padding=True, truncation=True, max_length=8192)``; ``:67`` the embedding is
  ``outputs.last_hidden_state[:, -1, :].float()`` then L2-normalised in float32 (the repo helper's
  exact expression: norms clamped at 1e-12 at ``:22-30``, so a zero row stays zero).
- ``experiments/paper/rerankers/reference/octen.py:112,122-126,151-152``: the document prefix
  ``"- "`` is
  prepended to documents only (``f"{self.document_prefix}{text}"``, ONE string tokenised once);
  queries are encoded as they are. The paper defines NO query instruction: the
  ``"Instruct: ...\nQuery:"`` prompt in the checkpoint's ``config_sentence_transformers.json`` is
  NOT part of the paper's recipe, and neither is its single-space document prompt.
- ``experiments/paper/retrieval/octen.yaml:9-11``: ``pooling: last``, ``doc_prompt: "- "``,
  ``batch_size: 32``.

Differences from the model card (paper wins; the full list is in the research report):
- the card's sentence-transformers config carries a Qwen3-style query instruction and a ``" "``
  (single space) document prompt; the paper uses no instruction and ``"- "``;
- the card advertises 32,768 / 40,960-token contexts; the paper truncates at 8192 on the right (and
  so does the card's own usage example).

Anchor behaviour over the cap: the tokenizer's post-processor appends the end-of-text token
(added-token name ``endoftext``, id 151643) to every sequence and HF's truncation reserves it, so a
right cut at 8192 keeps the ``"- "`` prefix at the head and the appended anchor at the tail. The
recipe therefore declares ``over_cap_cut_differs``: the render carries the full text over-cap (the
truncation happens at encode time in the paper path), and the paper's cut boundary can differ by a
token from the client's content-boundary cut -- over-cap pairs ride the non-gating table, and the
reference is the paper's path verbatim, never a port of the client's cut
(``docs/how-to/add-a-model.md``).

Runs as a subprocess in its own environment (torch + transformers; see this directory's
``requirements-reference.txt``), never inside the harness:

    <reference-python> reference.py --mode <render|embed|score> --pairs <file> --out <file> \
        --tokenizer "<repo>@<revision>|<path>" [--device cpu|cuda:0]

Modes and the JSON each writes to ``--out``:

- ``render`` -- ``{"rows": [{"index", "shape", "text"}]}``: the prompt TEXT per declared shape (the
  recipe's ``client.template`` shapes, read from ``recipe.yaml`` beside this file). The document
  shape renders ``"- " + text``; the query shape renders the query as it is. Over-cap truncation
  happens at encode time in the paper path, so the render carries the full text; stage 1 compares
  under-budget rows only. A row's ``instruction`` is ignored: the recipe's template declares no
  instruction span, exactly like the served path's fit.
- ``embed`` -- ``{"rows": [{"index", "query_vectors", "document_vectors"}]}``: L2-normalised float32
  vectors, one per text (downloads the ~15 GB checkpoint on first use; a GPU is expected).
- ``score`` -- ``{"rows": [{"index", "scores"}]}``: cosine similarity of the query against each
  document (dot product of the L2-normalised vectors; the checkpoint's own
  ``config_sentence_transformers.json`` declares ``similarity_fn_name: cosine``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

MODEL = "Octen/Octen-Embedding-8B"
REVISION = "5adcfa292e712091dfc30f0e97f0b2282e6cc66c"

MAX_LENGTH = 8192  # tokens per text; longer texts are truncated on the right (reference/octen.py:33)
DOCUMENT_PREFIX = "- "  # prepended to documents only (octen.yaml:10; reference/octen.py:151-152)
DTYPE = "bfloat16"  # reference/octen.py:55; also the checkpoint's config.json "dtype"
PAD_SIDE = "left"  # reference/octen.py:54 -- the last position of every row is a real token
BATCH_SIZE = 32  # octen.yaml:11

#: The request shapes the recipe's template declares, with the role each is embedded as.
SHAPES: dict[str, str] = {"query": "query", "document": "document"}


def load(device: str = "cpu") -> tuple[Any, Any, str]:
    """Load the model and tokenizer exactly as ``hf_dense.py:43-51`` does.

    Returns ``(model, tokenizer, device)``: the checkpoint in bfloat16, eval mode, on ``device``;
    the tokenizer with left padding. Downloads ~15 GB of weights on first use.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side=PAD_SIDE, revision=REVISION)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16, revision=REVISION)
    model = model.to(device).eval()
    return model, tokenizer, device


def _tokenizer_for(spec: str) -> Any:
    """The tokenizer ``--tokenizer`` names: the recipe's ``<repo>@<revision>`` spec or a local path.

    Left padding is set unconditionally (``hf_dense.py:49``). The spec's revision, when present,
    overrides nothing else: the model constants above stay pinned to the paper's revision.
    """
    from transformers import AutoTokenizer

    if spec.startswith(("/", "./", "../", "~")):
        return AutoTokenizer.from_pretrained(spec, padding_side=PAD_SIDE)
    repo, _, revision = spec.partition("@")
    return AutoTokenizer.from_pretrained(repo, revision=revision or None, padding_side=PAD_SIDE)


def render_prompt(text: str, role: str) -> str:
    """The prompt string the paper's path encodes for one text of ``role``.

    ``role`` is ``"query"`` or ``"document"``. Documents get ``DOCUMENT_PREFIX + text`` as ONE
    string (the paper concatenates before tokenising, ``torch_dense.py:70``); queries run as they
    are. Never tokenise the prefix separately and concatenate ids: BPE merges the prefix's space
    into the first word of the content.
    """
    if role not in SHAPES:
        raise ValueError(f"unknown role {role!r}; expected 'query' or 'document'")
    return f"{DOCUMENT_PREFIX}{text}" if role == "document" else text


def render(text: str, role: str, instruction: str | None = None) -> list[int]:
    """Token ids of the exact prompt the paper's encoder reads for one text of ``role``.

    The prompt is built as one string (:func:`render_prompt`) and tokenised once with the paper's
    budget: ``truncation=True`` (right side, HF's default) at ``MAX_LENGTH``, with the tokenizer's
    post-processor tokens (``add_special_tokens=True``) -- which reserves the appended end-of-text
    anchor, so the ids always end with the token the model pools.

    ``instruction`` must be ``None``: the paper's recipe defines no instruction for this model, and
    silently dropping one would change the numbers.
    """
    if instruction is not None:
        raise ValueError(
            "the paper's recipe for Octen-Embedding-8B defines no instruction; "
            "pass instruction=None (documents get the '- ' prefix, queries run as-is)"
        )
    tokenizer = _tokenizer_for(MODEL)
    # truncation=True + the default right truncation side, max_length=MAX_LENGTH (hf_dense.py:59)
    return tokenizer(render_prompt(text, role), truncation=True, max_length=MAX_LENGTH)["input_ids"]


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Every row scaled to unit L2 norm (a zero row stays zero), as float32.

    The same expression as the paper path's ``l2_normalize`` (``hf_dense.py:62`` via
    ``rcp_ndcg.retrieval.encoder``): norms clamped at 1e-12, cast to float32.
    """
    import numpy as np

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


def embed(
    texts: list[str],
    role: str,
    model: Any | None = None,
    tokenizer: Any | None = None,
    *,
    device: str = "cpu",
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Embed ``texts`` of one role as L2-normalised float32 vectors of shape ``(len(texts), 4096)``.

    Exactly ``encode_text_batches`` (``hf_dense.py:65-92``) after the role prefix of
    ``TorchDenseEncoder.encode`` (``torch_dense.py:69-70``): left padding, right truncation at
    8192 tokens, last-token pooling in float32, L2 norm. ``model``/``tokenizer`` come from
    :func:`load`; if omitted the model is downloaded (~15 GB) onto ``device``.
    """
    if role not in SHAPES:
        raise ValueError(f"unknown role {role!r}; expected 'query' or 'document'")
    if model is None or tokenizer is None:
        model, tokenizer, device = load(device)

    import numpy as np
    import torch

    prompts = [render_prompt(text, role) for text in texts]
    out: list[np.ndarray] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # .float() first, then normalise -- the same order as hf_dense.py:62
        out.append(_l2_normalize(outputs.last_hidden_state[:, -1, :].float().cpu().numpy()))
    return np.vstack(out).astype(np.float32, copy=False)


def score(query: str, docs: list[str], instruction: str | None = None) -> list[float]:
    """Cosine similarity of the query against each document (the paper's retrieval score).

    The paper's index/search ranks by dot product of L2-normalised vectors
    (``experiments/paper/retrieval/octen.yaml``; the checkpoint's own
    ``config_sentence_transformers.json`` also declares ``similarity_fn_name: cosine``), so
    dot(normalised, normalised) == cosine.
    """
    if instruction is not None:
        raise ValueError("the paper's recipe defines no instruction for this model")
    q = embed([query], "query")
    d = embed(docs, "document")
    return (q @ d.T).reshape(-1).tolist()


# -- the subprocess CLI the harness calls (rcp_ndcg_vllm.equivalence.reference) ---------------


def _declared_shapes() -> list[str]:
    """The recipe's declared template shapes, in the file's order (the harness fits those)."""
    import yaml

    recipe = yaml.safe_load((Path(__file__).resolve().parent / "recipe.yaml").read_text(encoding="utf-8"))
    template = (recipe.get("client") or {}).get("template") or {}
    shapes = [shape for shape in ("query", "document", "pair") if shape in template]
    return shapes or ["document"]


def main() -> int:
    """The subprocess CLI: read the pairs file, write the mode's JSON to ``--out``."""
    parser = argparse.ArgumentParser(description="the Octen/Octen-Embedding-8B paper reference")
    parser.add_argument("--mode", required=True, choices=["render", "embed", "score"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, row in enumerate(rows, start=1):
        if row.get("instruction"):
            raise ValueError(
                f"pairs line {number} carries an instruction; the recipe's template declares no "
                "instruction span and the paper's path defines none for this model"
            )

    if args.mode == "render":
        out: dict[str, Any] = {"rows": []}
        for index, row in enumerate(rows):
            for shape in _declared_shapes():
                text = render_prompt(str(row["documents"][0]), shape) if shape != "query" else str(row["query"])
                out["rows"].append({"index": index, "shape": shape, "text": text})
    elif args.mode == "embed":
        model, tokenizer, device = load(args.device)
        out = {"rows": []}
        for index, row in enumerate(rows):
            q = embed([str(row["query"])], "query", model, tokenizer, device=device)
            d = embed([str(document) for document in row["documents"]], "document", model, tokenizer, device=device)
            out["rows"].append(
                {
                    "index": index,
                    "query_vectors": q.tolist(),
                    "document_vectors": d.tolist(),
                }
            )
    else:
        model, tokenizer, device = load(args.device)
        out = {"rows": []}
        for index, row in enumerate(rows):
            q = embed([str(row["query"])], "query", model, tokenizer, device=device)
            d = embed([str(document) for document in row["documents"]], "document", model, tokenizer, device=device)
            out["rows"].append({"index": index, "scores": (q @ d.T).reshape(-1).tolist()})

    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
