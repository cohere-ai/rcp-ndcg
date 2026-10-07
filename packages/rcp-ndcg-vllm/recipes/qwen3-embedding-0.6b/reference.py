"""Reference implementation for Qwen/Qwen3-Embedding-0.6B, run as a subprocess in its own environment
(never imported by the harness).

Published code path, per mode:

- ``--mode embed`` — the model card's "Transformers Usage" snippet, verbatim mechanics
  (transformers ``AutoModel``/``AutoTokenizer`` at the pinned revision, ``padding_side='left'``,
  ``truncation=True, max_length=8192``, the card's ``last_token_pool`` over the attention mask, then
  ``F.normalize(p=2)``; bf16 tensors cast ``.float()`` before ``.numpy()``). Needs torch and
  transformers (>=4.51 for the Qwen3 architecture; the card's floor) — the per-recipe
  ``requirements-reference.txt`` next to this file (the package-level file names the general env).
- ``--mode render`` — the card's prompt format as text (``get_detailed_instruct``); the tokenizer
  spec is resolved and its pin checked (``huggingface_hub`` only -- no weights, no torch). It emits
  the prompt TEXT the served client must render -- the harness compares the texts byte-exactly and
  tokenises both sides with the same tokenizer; the GPU wave's stage-1 ``/tokenize`` check is the
  authoritative engine-side cross-check. The render is the prompt the card's code hands its tokenizer, uncut: the
  card truncates the ids inside ``tokenizer(..., truncation=True, max_length=8192)`` (``embed``), and
  this reference never reproduces the product client's cut.  Under the budget the render equals the
  served prompt byte for byte; over the budget the two cuts can differ at the content boundary (the
  card cuts ids, the client cuts verbatim text and re-tokenizes), so the recipe declares
  ``over_cap_cut_differs`` and over-cap rows ride the harness's non-gating table.

The anchor: the served route (``/v1/embeddings``, string inputs) tokenizes with the tokenizer's
default ``add_special_tokens=True`` (vllm/renderers/params.py:183 at tag v0.31.0), appending one
endoftext token (id 151643) that last-token pooling reads. The card's truncation keeps it inside the
8192 budget (8191 content+frame ids, then the anchor: measured); a naive first-8192-token cut drops
it and diverges at exactly the pooled token — the over-cap divergence the harness's stage-1 anchor
audit exists to catch.  The card keeps the anchor, so the recipe's declared deviation is
``over_cap_cut_differs`` (the anchors survive on both sides; only the content boundary can differ),
never ``anchor_drop_over_cap``.

The query frame is fixed: ``Instruct: {task}\\nQuery:{query}`` with no space after ``Query:`` — the
byte-exact ``prompts.query`` of ``config_sentence_transformers.json`` at the pinned revision. A
task-specialised instruction rewrites that frame (a different recipe), never a per-row field; any
``instruction`` a pairs row carries is ignored on both sides of the equivalence check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Standalone on purpose: the reference runs in its own environment (torch, transformers, the pinned
# tokenizer file), never inside the harness's process, and imports nothing from rcp-ndcg.

MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

#: The added token named endoftext: appended by the tokenizer's post-processor and read by the
#: last-token pooler. Referred to by name everywhere; never typed out.
END_OF_TEXT_NAME = "endoftext"
END_OF_TEXT_ID = 151643

#: The query frame: the byte-exact ``prompts.query`` of the checkpoint's
#: ``config_sentence_transformers.json`` (the card's ``get_detailed_instruct`` fold point included).
QUERY_PROMPT = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"

#: The card's transformers example budget (the architecture allows 32768).
MAX_LENGTH = 8192


def get_detailed_instruct(query: str) -> str:
    """The card's exact query format: the fixed frame, then the query with no separating space."""
    return f"{QUERY_PROMPT}{query}"


def _load_tokenizer_path(spec: str) -> str:
    """The tokenizer file the ``--tokenizer`` spec names, downloaded once and cached.

    The spec is the pinned repository at the pinned revision (drift is refused loudly -- the
    reference tokenises with the checkpoint's own file, never whatever a newer HEAD carries), or a
    path to the same ``tokenizer.json`` (the harness resolves recipe-relative tokenizer specs into
    local paths, and a recipe that ships its tokenizer file passes one).
    """
    from huggingface_hub import hf_hub_download

    candidate = Path(spec).expanduser()
    if candidate.suffix == ".json" or candidate.exists():
        return str(candidate)  # a local tokenizer.json (a recipe-relative spec the harness resolved)
    repo, _, revision = spec.partition("@")
    if repo and repo != MODEL:
        raise SystemExit(f"the tokenizer spec names {repo!r}; this reference pins {MODEL!r}")
    if revision and revision != REVISION:
        raise SystemExit(f"the tokenizer spec pins revision {revision!r}; this reference pins {REVISION!r}")
    return hf_hub_download(MODEL, "tokenizer.json", revision=REVISION)


def render_rows(pairs: list[dict[str, Any]], tokenizer_spec: str) -> dict[str, Any]:
    """``--mode render``: the prompt text per row per declared shape, as the card's code builds it.

    The query shape carries the instruction frame (``get_detailed_instruct``); the document shape is
    the bare text (the checkpoint's ``prompts.document`` is empty).  Uncut: the card truncates the
    ids at encode (``embed_rows``), never the text, and the reference never ports the client's cut.
    The tokenizer spec is resolved (and its pin checked) so a drifted spec is refused here too.
    """
    _load_tokenizer_path(tokenizer_spec)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(pairs):
        rows.append({"index": index, "shape": "query", "text": get_detailed_instruct(str(row["query"]))})
        rows.append({"index": index, "shape": "document", "text": str(row["documents"][0])})
    return {"rows": rows}


def last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """The card's pooling: the hidden state of the last non-pad token per row, both padding sides."""
    import torch

    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def embed_rows(pairs: list[dict[str, Any]], tokenizer_spec: str, device: str) -> dict[str, Any]:
    """``--mode embed``: the card's Transformers Usage path -- L2-normalised float32 vectors.

    One vector for the row's query (instruction-wrapped) and one per document (bare); documents over
    ``MAX_LENGTH`` tokens are truncated the card's way, which keeps the endoftext anchor in budget.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    repo, _, revision = tokenizer_spec.partition("@")
    if (repo and repo != MODEL) or (revision and revision != REVISION):
        raise SystemExit(f"the tokenizer spec {tokenizer_spec!r} does not pin {MODEL}@{REVISION}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
    model = AutoModel.from_pretrained(MODEL, revision=REVISION).to(device).eval()

    def embed(texts: list[str]) -> list[list[float]]:
        """One L2-normalised float32 vector per text, batched the card's way."""
        out: list[Any] = []
        for offset in range(0, len(texts), 8):
            batch = tokenizer(
                texts[offset : offset + 8],
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                hidden = model(**batch).last_hidden_state
            out.append(last_token_pool(hidden, batch["attention_mask"]))
        embeddings = F.normalize(torch.cat(out, dim=0), p=2, dim=1)
        # bf16 tensors cannot cross into numpy: cast to float32 first (the reference draft's measured fix).
        return embeddings.float().cpu().numpy().astype(np.float32).tolist()

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(pairs):
        query_vectors = embed([get_detailed_instruct(str(row["query"]))])
        document_vectors = embed([str(document) for document in row["documents"]])
        rows.append(
            {
                "index": index,
                "query_vectors": query_vectors,
                "document_vectors": document_vectors,
            }
        )
    return {"rows": rows}


def main() -> int:
    """The reference CLI the harness's runner invokes (see ``equivalence/reference.py``)."""
    parser = argparse.ArgumentParser(description="the Qwen/Qwen3-Embedding-0.6B reference")
    parser.add_argument("--mode", required=True, choices=["render", "embed"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--tokenizer",
        required=True,
        help=f"{MODEL}@{REVISION}, or a local path to the same tokenizer.json (render mode)",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        result = render_rows(pairs, args.tokenizer)
    else:
        result = embed_rows(pairs, args.tokenizer, args.device)
    Path(args.out).write_text(json.dumps(result) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    # The card's own example (queries and documents verbatim) reproduces the card's printed cosines
    # through --mode embed; max |delta| ~0.004 on CPU bf16, with ~1e-3 jitter between processes at
    # the same pins -- the gate is the card's table, not any single run's number.
    raise SystemExit(main())
