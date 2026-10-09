"""The jina-embeddings-v5-text family's one reference -- the model card's own published code path,
run as a subprocess (never imported by the harness).

**Which code path:** the checkpoint's published remote code via the model card's transformers
snippet (``README.md:148-178`` at the pinned revision): ``AutoModel.from_pretrained(...,
trust_remote_code=True, dtype=torch.bfloat16)`` -- the remote code (``modeling_jina_embeddings_v5.py``)
builds a :class:`peft.PeftMixedModel` over the checkpoint's backbone (``Qwen3Model`` for -small,
``EuroBertModel`` for -nano) with the four task LoRAs and selects the task adapter per encode
(``set_adapter([task])``, ``modeling_jina_embeddings_v5.py:95``). ``encode`` applies the
``"Query: "`` / ``"Document: "`` prefix to the RAW texts itself (``modeling_jina_embeddings_v5.py:79-80``
-- pre-prefixing here would double-prefix, a defect a review caught), pools the mask's last real
token, slices ``truncate_dim`` and L2-normalises (``:100-112``). This reference therefore passes raw
texts plus ``prompt_name`` and takes the card's defaults for everything else (no
``_attn_implementation`` override: the card marks flash-attention "Recommended but optional", and
its GPU kwargs are unmeasured). One file serves every variant of the family (decision 34): the
variant travels with the invocation, in the resolved recipe the harness passes as ``--recipe``;
:data:`HF_REPO` and :data:`HF_REVISION` name the family's -small checkpoint as the standalone
defaults, and the CLI reads the variant's own ``model``/``revision`` from the recipe.

**Reference environment** (its own python — never the harness's process, never the engine image):
``torch``, ``transformers>=4.57`` (the card snippet's ``dtype=`` kwarg) and ``peft`` (the remote
code is a PeftMixedModel); see ``requirements-reference.txt`` beside this recipe. Weights and the
remote code come from the pinned Hub revision (~1.26 GiB / 1.35 GB); ``--model-path`` points at a
local snapshot instead. The weights are never needed for ``--mode render`` (pure string work), so
stage 1 runs on CPU without them.

Paths of truth (read at revision ``dd76d535f5447ca3897a9c893fb1e612ead98192``):
``config_sentence_transformers.json:7-9`` (prompts ``"Query: "`` / ``"Document: "``),
``config.json:19`` (``max_position_embeddings`` 32768 — ``encode`` truncates the assembled prompt
right at it, ``modeling:85-91``: the same budget as the client's, cut on ids instead of on verbatim
content, so over the cap the boundary can differ -- the declared ``over_cap_cut_differs``),
``modeling_jina_embeddings_v5.py:79-80`` (prefixes), ``:100-108`` (mask-based last-real-token
pooling), ``:110-112`` (slice, then L2).

The harness's contract (``rcp_ndcg_test.equivalence.reference``):

    reference.py --mode <render|embed|score> --pairs <file> --out <file> \\
                 --tokenizer <repo@rev|path> --device <cpu|cuda:N> [--model-path <dir>]

- ``render`` — per pairs row, one row per declared shape: the prompt text the card's code builds
  (``{"rows": [{"index", "shape", "text"}]}``), uncut -- the card truncates ids inside ``encode``,
  and the reference never ports the client's cut. Stage 1 compares every under-cap pairs row
  byte-identically against the client's render; an over-cap row differs by declaration
  (``over_cap_cut_differs``) and rides the harness's non-gating table.
- ``embed`` — ``{"rows": [{"index", "query_vectors": [[...]], "document_vectors": [[...]]}]}``,
  one L2-normalised float32 vector per side (the model is dense; the width is the variant's own:
  1024 for the -small, 768 for the -nano).
- ``score`` — cosine of the embedded query against each document (embed role; kept for
  completeness, the harness's embed stage uses ``--mode embed``).

``--tokenizer`` is accepted for the interface and unused: the prompt is plain text, and the model
loads its own tokenizer with the checkpoint (the card path — the ids equality is stage 1's
zero-tolerance comparison on the harness side, against this same file).

**Pinning:** on a repo-id load the checkpoint's remote code never forwards the ``revision`` to
its inner loads: the four adapters (an unversioned ``snapshot_download``,
``modeling_jina_embeddings_v5.py:37-41``) and the tokenizer (``:57-60``) resolve at Hub HEAD, and
so do the base weights on a cache miss — transformers' commit-hash inheritance only serves a
commit already in the cache and downloads at ``main`` otherwise (measured on transformers 5.17.0
with a cold cache); only the config rides the caller's revision (the auto factory passes it,
``models/auto/auto_factory.py:386-390``). :func:`load` therefore resolves the whole snapshot at
the pinned revision (``snapshot_download(HF_REPO, revision=HF_REVISION)``) and loads from it:
the vendor code's local-dir branch takes the adapters from the snapshot too, so every tensor
comes from the pinned revision whatever the cache holds. ``--mode render`` downloads nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

HF_REPO = "jinaai/jina-embeddings-v5-text-small"
HF_REVISION = "dd76d535f5447ca3897a9c893fb1e612ead98192"
#: The two constants above are the family's -small checkpoint and the standalone defaults; every
#: served variant is loaded through the resolved recipe's own pair.
QUERY_PREFIX = "Query: "  # config_sentence_transformers.json:8
DOCUMENT_PREFIX = "Document: "  # config_sentence_transformers.json:9
DEFAULT_TASK = "retrieval"  # config.json task_names[0]; vLLM's _DEFAULT_TASK
Role = Literal["query", "document"]


def render(text: str, role: Role, instruction: str | None = None) -> str:
    """The exact prompt for one text: ``<prefix><text>`` — no chat frame, no BOS.

    ``instruction`` must be ``None``: the model defines no task instructions (contrast
    Qwen3-Embedding), so an instruction-bearing row is an error with a hint, never a silent
    default that changes numbers.
    """
    if instruction:
        raise ValueError(
            f"jina-embeddings-v5 defines no task instructions (config_sentence_transformers.json:7-9 has "
            f"only the {'query' if role == 'query' else 'document'} prompt); got instruction {instruction!r} "
            "(hint: this recipe's client template folds no instruction; drop the field from the pairs file)"
        )
    prefix = QUERY_PREFIX if role == "query" else DOCUMENT_PREFIX
    return f"{prefix}{text}"


class _CardModel:
    """The loaded card model: the remote-code wrapper, its tokenizer and the selected task."""

    def __init__(self, model, tokenizer, task: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.task = task

    def embed(self, texts: list[str], role: Role) -> list[list[float]]:
        """L2-normalised float32 vectors, one per text, from the card's ``encode``.

        RAW texts plus ``prompt_name`` (the wrapper applies the prefix itself, modeling:79-80);
        ``truncate_dim`` stays ``None`` (the variant's full vector). Padding is mask-pooled
        (``:100-108``), so batch composition changes nothing.
        """
        import numpy as np
        import torch

        embs = self.model.encode(
            texts=list(texts),
            task=self.task,
            prompt_name=role,
        )
        array = embs.float().cpu().numpy() if torch.is_tensor(embs) else np.asarray(embs)
        return [[float(value) for value in row] for row in np.asarray(array, dtype=np.float32)]


def _pinned_snapshot_path(repo: str = HF_REPO, revision: str = HF_REVISION) -> str:
    """The pinned snapshot directory: on a repo-id load the card path leaves the adapters and the
    tokenizer at Hub HEAD and the base weights at HEAD on a cache miss (the vendor code never
    forwards the revision — see the module docstring), so the snapshot is resolved once at the
    variant's pinned revision and everything loads from it."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo,
        revision=revision,
        allow_patterns=["*.json", "*.py", "*.txt", "*.jinja", "*.safetensors"],
    )


def load(
    device: str = "cpu",
    model_path: str | None = None,
    task: str = DEFAULT_TASK,
    *,
    repo: str = HF_REPO,
    revision: str = HF_REVISION,
) -> _CardModel:
    """Load the card model. ``device``: ``cpu`` or ``cuda:N``; bf16 weights (the card snippet).

    ``model_path`` (a local snapshot directory) is used as given; without one the variant's pinned
    Hub revision (``repo``/``revision``, defaulting to the family's -small checkpoint) is resolved
    via :func:`_pinned_snapshot_path` — never Hub HEAD (the vendor remote code would ignore the
    revision on its inner loads)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    name = model_path or _pinned_snapshot_path(repo, revision)
    model = AutoModel.from_pretrained(
        name,
        trust_remote_code=True,  # the card snippet (README:153); config parsing + the remote code
        dtype=torch.bfloat16,  # the card snippet; the checkpoint's own torch_dtype
    ).to(device=device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=True,
    )
    return _CardModel(model, tokenizer, task)


def _rows(pairs_path: str) -> list[dict]:
    """The pairs file: JSONL, ``{"query": str, "documents": [str, ...]}`` (plus optional fields)."""
    return [json.loads(line) for line in Path(pairs_path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _cosine(query_vector: list[float], document_vectors: list[list[float]]) -> list[float]:
    """Cosine of one query vector against each document vector (float32 math, the stored scale)."""
    import numpy as np

    q = np.asarray(query_vector, dtype=np.float32)
    d = np.asarray(document_vectors, dtype=np.float32)
    qn, dn = np.linalg.norm(q), np.linalg.norm(d, axis=1)
    return [float(value) for value in (d @ q) / (dn * qn)]


def main() -> int:
    parser = argparse.ArgumentParser(description="the jina-embeddings-v5-text-small reference (the card path)")
    parser.add_argument("--mode", required=True, choices=["render", "embed", "score"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="accepted for the harness contract; the card path loads the checkpoint's own tokenizer",
    )
    parser.add_argument(
        "--recipe",
        required=True,
        help="the resolved recipe JSON the harness passed (the variant's id, model and revision)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-path", default=None, help="a local snapshot directory instead of the Hub revision")
    parser.add_argument("--task", default=DEFAULT_TASK)
    args = parser.parse_args()

    pairs = _rows(args.pairs)
    recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
    variant_repo, variant_revision = str(recipe["model"]), str(recipe["revision"])
    if args.mode == "render":
        rows = [
            {"index": index, "shape": shape, "text": render(text, shape, row.get("instruction"))}
            for index, row in enumerate(pairs)
            for shape, text in (("query", row["query"]), ("document", row["documents"][0]))
        ]
        document = {"rows": rows}
    else:
        card = load(args.device, args.model_path, args.task, repo=variant_repo, revision=variant_revision)
        rows = []
        for index, row in enumerate(pairs):
            query_vectors = card.embed([row["query"]], "query")
            document_vectors = card.embed(list(row["documents"]), "document")
            entry: dict = {"index": index, "query_vectors": query_vectors, "document_vectors": document_vectors}
            if args.mode == "score":
                entry["scores"] = _cosine(query_vectors[0], document_vectors)
            rows.append(entry)
        document = {"rows": rows}
    Path(args.out).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
