"""The reference implementation for jina-reranker-v3: the paper's exact in-process scoring path.

Derived from ``experiments/paper/rerankers/reference/jina.py`` (behaviour unchanged) with the
checkpoint's remote code at revision ``d7d7e73b6ea138ced340b83865931b5dfb6c97aa``:

- ``load(device)`` builds the paper's ``JinaRerank`` (``AutoModel.from_pretrained(..., dtype="auto",
  trust_remote_code=True, revision=...)`` -> the remote ``JinaForRanking``; the paper's factory passed
  only model, device and revision -- every runtime parameter is the remote code's default);
- ``score(query, docs)`` is the paper's harness (``JinaRerank.predict`` in
  ``experiments/paper/rerankers/reference/jina.py:65-76``; the pre-unification
  ``rcp-ndcg/src/rcp_ndcg/retrieval/external_rerankers.py:427-455``): whitespace-only
  documents score exactly 0.0 without a model call, the rest go to the checkpoint's ``model.rerank()``
  (which blocks at 125 documents or a residual token capacity of ``131072 - 2*q_len`` flushed at
  ``<= 2048``, weights blocks by ``max((1+cos)/2)``, block-averages the query embed and scores every
  document against that average in float32), and the ranked results are re-aligned back to input order;
- ``render(query, docs)`` is the prompt-level view, runnable tokenizer-only (no weights, no torch):
  the remote code's pre-templating truncation (documents 2048 tokens, query 512, right, decode-back)
  then ``format_docs_prompts_func`` -- the exact prompt one block sends.  The model is listwise: the
  prompt depends on the whole document set, so ``docs`` may be one document (the 1-vs-1 prompt the
  recipe's declared pair shape mirrors) or a block's list.  ``render_query_and_document`` is the
  1-vs-1 entry point.

This module runs as a SUBPROCESS in its own environment (see ``requirements-reference.txt`` beside
it); the harness process never imports it.  CLI contract (``rcp_ndcg_test.equivalence.reference``):

    reference.py --mode <render|score> --pairs <file> --out <file> --tokenizer <spec> [--device <d>]

- ``render`` -> ``{"rows": [{"index", "shape": "pair", "query", "documents"}]}``: per pairs row,
  the spans fed to the model's own builder -- the checkpoint's per-text truncations (documents 2048
  tokens, query 512, right, decode-back) applied to the raw texts, exactly as ``rerank()`` stages
  them before formatting.  The reference never ports the client's cut (``docs/how-to/add-a-model.md``):
  under the per-text caps the spans are the raw texts and compare byte-identically with the wire's;
  over-cap rows differ from the client's cuts and ride the declared ``over_cap_cut_differs`` table.
- ``score``   -> ``{"rows": [{"index", "scores": [...]}]}``: one cosine in [-1, 1] per document,
  input order, empty documents 0.0.

The instruction is never sent: the checkpoint's ``rerank()`` hard-codes ``instruction=None`` in every
``_compute_single_batch`` call (the paper never sends one), so ``score`` refuses a non-None
instruction and ``render`` ignores the pairs row's instruction field.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

MODEL = "jinaai/jina-reranker-v3"
REVISION = "d7d7e73b6ea138ced340b83865931b5dfb6c97aa"

#: The checkpoint's special tokens and their ids (modeling.py JinaForRanking.__init__).  Documentation
#: constants: the prompt builder inserts the token *strings*; the ids are pinned by the tokenizer files
#: (added_tokens.json) and the vendor sources, and the tokenizer check in stage 1 carries their truth.
DOC_EMBED_TOKEN = "<|" + "embed_token" + "|>"
QUERY_EMBED_TOKEN = "<|" + "rerank_token" + "|>"
IM_START = "<|" + "im_start" + "|>"
IM_END = "<|" + "im_end" + "|>"
THINK = chr(60) + "think" + chr(62)
THINK_END = chr(60) + "/" + "think" + chr(62)
DOC_EMBED_TOKEN_ID = 151670
QUERY_EMBED_TOKEN_ID = 151671

#: rerank()'s blocking defaults (modeling.py rerank signature and body).
BLOCK_SIZE = 125
MAX_DOC_LENGTH = 2048
MAX_QUERY_LENGTH = 512
MODEL_MAX_LENGTH = 131072

__all__ = [
    "BLOCK_SIZE",
    "DOC_EMBED_TOKEN_ID",
    "JinaRerankerV3",
    "MAX_DOC_LENGTH",
    "MAX_QUERY_LENGTH",
    "MODEL",
    "MODEL_MAX_LENGTH",
    "QUERY_EMBED_TOKEN_ID",
    "REVISION",
    "format_docs_prompt",
    "load",
    "render_query_and_document",
    "render_rows",
]


def _sanitize(text: str) -> str:
    """Strip the model's marker tokens from user text (modeling.py sanitize_input)."""
    for token in (DOC_EMBED_TOKEN, QUERY_EMBED_TOKEN):
        text = text.replace(token, "")
    return text


def format_docs_prompt(query: str, docs: Sequence[str], instruction: str | None = None) -> str:
    """Verbatim port of modeling.py ``format_docs_prompts_func`` (no_thinking=True, as rerank() calls it).

    The checkpoint sanitizes the query and every document before formatting; this port does too, so a
    caller that has already truncated its texts gets the model's exact prompt.
    """
    query = _sanitize(query)
    doc_list = [_sanitize(doc) for doc in docs]

    prefix = (
        f"{IM_START}system\n"
        "You are a search relevance expert who can determine a ranking of the passages based on how "
        "relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining the ranking."
        f"{IM_END}\n{IM_START}user\n"
    )
    suffix = f"{IM_END}\n{IM_START}assistant\n{THINK}\n\n{THINK_END}\n\n"

    prompt = (
        f"I will provide you with {len(doc_list)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )
    if instruction:
        prompt += f"<instruct>\n{_sanitize(instruction)}\n</instruct>\n"
    doc_prompts = [f'<passage id="{i}">\n{doc}{DOC_EMBED_TOKEN}\n</passage>' for i, doc in enumerate(doc_list)]
    prompt += "\n".join(doc_prompts) + "\n"
    prompt += f"<query>\n{query}{QUERY_EMBED_TOKEN}\n</query>"
    return prefix + prompt + suffix


def _product_tokenizer(spec: str):
    """The recipe tokenizer through the product's loader (the same file stage 1's fits count with)."""
    from rcp_ndcg.data.tokenizer import load_tokenizer

    return load_tokenizer(spec)


def _truncate_text(text: str, backend: object, max_length: int) -> str:
    """modeling.py ``_truncate_texts``' per-text rule: right-truncate the RAW text before templating; a
    text that reaches ``max_length`` tokens is ``decode(ids[:max_length])`` back to text.

    ``tokenizer(text, truncation=True, max_length=N)`` keeps the first N ids here (this tokenizer's
    post-processor adds no special tokens), and HF's ``decode`` keeps special tokens
    (``skip_special_tokens=False``, its default).  The shared backend is never reconfigured (no
    ``enable_truncation``): the product's ``load_tokenizer`` caches it, so a mutation would cap every
    later count in the same process."""
    ids = list(backend.encode(text, add_special_tokens=True).ids)  # type: ignore[attr-defined]
    if len(ids) >= max_length:
        text = backend.decode(ids[:max_length], skip_special_tokens=False)  # type: ignore[attr-defined]
    return text


def render_query_and_document(query: str, document: str, tokenizer_spec: str, instruction: str | None = None) -> str:
    """The exact 1-vs-1 prompt text for one (query, document) pair, tokenizer-only.

    Applies the remote code's pre-templating truncation (2048/512 tokens, right, decode-back) exactly as
    ``rerank()`` does, then formats.  This is the text the recipe's declared pair shape must render to
    byte for byte (the prompt-level view; stage 1's render check compares the wire's SPANS instead).
    """
    tokenizer = _product_tokenizer(tokenizer_spec)
    backend = tokenizer.backend
    truncated_query = _truncate_text(query, backend, MAX_QUERY_LENGTH)
    truncated_doc = _truncate_text(document, backend, MAX_DOC_LENGTH)
    return format_docs_prompt(truncated_query, [truncated_doc], instruction=instruction)


def render_rows(pairs: list[dict], tokenizer_spec: str) -> list[dict]:
    """Stage 1's rerank render contract: the spans fed to the model's own builder, per pairs row.

    ``{"index", "shape": "pair", "query": <query span>, "documents": [<doc span>, ...]}`` -- the
    checkpoint's per-text pre-templating truncation applied to the raw texts (``_truncate_text``:
    512/2048 tokens, right, decode-back), exactly as ``rerank()`` stages them before formatting.
    The reference never ports the client's cut (``docs/how-to/add-a-model.md``): under the per-text caps
    the spans are the raw texts and compare byte-identically with the wire's spans; over-cap rows
    differ from the client's cuts and ride the declared ``over_cap_cut_differs`` table.
    """
    tokenizer = _product_tokenizer(tokenizer_spec)
    backend = tokenizer.backend
    rows: list[dict] = []
    for index, row in enumerate(pairs):
        query = str(row["query"])
        documents = [str(document) for document in row["documents"]] or [""]
        rows.append(
            {
                "index": index,
                "shape": "pair",
                "query": _truncate_text(query, backend, MAX_QUERY_LENGTH),
                "documents": [_truncate_text(document, backend, MAX_DOC_LENGTH) for document in documents],
            }
        )
    return rows


class JinaRerankerV3:
    """The paper's in-process reranker: the checkpoint's own ``rerank()`` over trust_remote_code.

    Units: :meth:`score` returns one cosine similarity in [-1, 1] per document, aligned with *docs*
    (input order, not the checkpoint's score-sorted order); empty documents score exactly 0.0.
    """

    def __init__(
        self, model_name_or_path: str = MODEL, revision: str | None = REVISION, device: str | None = None
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name_or_path
        self.revision = revision
        # The paper's harness (external_rerankers.py:411-420): exactly these arguments; dtype "auto"
        # resolves to the config's torch_dtype (bfloat16).  JinaRerank.is_v3 routes to AutoModel.
        self.model = AutoModel.from_pretrained(
            model_name_or_path, dtype="auto", trust_remote_code=True, revision=revision
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, revision=revision)
        self.model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def to(self, device: str) -> JinaRerankerV3:
        """Move the model to ``device`` (the paper harness's ``.to``)."""
        self.model.to(device)
        self.device = device
        return self

    def predict(self, query: str, docs: Sequence[str]) -> list[float]:
        """The paper's ``predict()`` (external_rerankers.py:427-455): empty docs 0.0, ``rerank()``, re-align."""
        non_empty = [(i, d) for i, d in enumerate(docs) if d.strip()]
        if not non_empty:
            return [0.0] * len(docs)

        indices, filtered_docs = zip(*non_empty, strict=True)
        results = self.model.rerank(query, documents=list(filtered_docs))
        scored = [0.0] * len(filtered_docs)
        for result in results:  # rerank() returns every document (top_n=None), sorted by score desc
            scored[result["index"]] = float(result["relevance_score"])

        output = [0.0] * len(docs)
        for original_index, score in zip(indices, scored, strict=True):
            output[original_index] = score
        return output

    def score(self, query: str, docs: Sequence[str], instruction: str | None = None) -> list[float]:
        """Paper-equivalent scores in input order; the paper's path never passes an instruction."""
        if instruction is not None:
            raise ValueError(
                "the paper's path never passes an instruction: the checkpoint's rerank() hard-codes "
                "instruction=None in every _compute_single_batch call"
            )
        return self.predict(query, docs)

    def render(self, query: str, docs: str | Sequence[str], instruction: str | None = None) -> list[int]:
        """Token ids of the exact prompt for one block (the checkpoint tokenizer's own tokenization).

        The model is listwise: *docs* may be one document (the 1-vs-1 prompt) or a list (one block's
        listwise prompt).  Applies the remote code's pre-templating truncation exactly as ``rerank()``
        does, then formats and tokenizes.  For the paper's multi-block calls the per-block prompts are
        ``render(query, block_docs)`` for each block of the blocking loop documented in the module
        docstring.
        """
        doc_list = [docs] if isinstance(docs, str) else list(docs)
        backend = self.tokenizer.backend_tokenizer
        truncated_docs = [_truncate_text(doc, backend, MAX_DOC_LENGTH) for doc in doc_list]
        truncated_query = _truncate_text(query, backend, MAX_QUERY_LENGTH)
        prompt = format_docs_prompt(truncated_query, truncated_docs, instruction=instruction)
        backend.no_truncation()
        return list(backend.encode(prompt, add_special_tokens=True).ids)  # type: ignore[attr-defined]


def load(device: str | None = None) -> JinaRerankerV3:
    """Load the paper's reranker (transformers + trust_remote_code; weights ~1.19 GB bf16)."""
    return JinaRerankerV3(device=device)


def _rows_of(path: str | Path) -> list[dict]:
    import json

    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("query"), str)
            or not isinstance(row.get("documents"), list)
        ):
            raise SystemExit(f"{path}:{number}: expected a JSON object with query and documents")
        rows.append(row)
    return rows


def main() -> int:
    """The harness's reference CLI: ``--mode render|score`` over a pairs JSONL file."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="the jina-reranker-v3 reference (the paper's in-process path)")
    parser.add_argument("--mode", required=True, choices=["render", "score"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True, help="the recipe tokenizer: <repo>@<revision> or a local path")
    parser.add_argument(
        "--recipe",
        required=True,
        help="the resolved recipe JSON the harness passed (the variant's id, model and revision)",
    )
    parser.add_argument("--device", default=None, help="score mode only: the device the reference loads on")
    args = parser.parse_args()

    rows = _rows_of(args.pairs)
    if args.mode == "render":
        output = {"rows": render_rows(rows, args.tokenizer)}
    else:
        reranker = load(args.device)
        output = {
            "rows": [
                {"index": index, "scores": reranker.score(str(row["query"]), [str(d) for d in row["documents"]])}
                for index, row in enumerate(rows)
            ]
        }
    Path(args.out).write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
