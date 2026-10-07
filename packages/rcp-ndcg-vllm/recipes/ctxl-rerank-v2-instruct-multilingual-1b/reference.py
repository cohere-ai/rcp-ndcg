"""The reference implementation of recipe ``ctxl-rerank-v2-instruct-multilingual-1b``.

Derived from ``experiments/paper/rerankers/reference/contextual.py`` (the paper's in-process
``ContextualRerank``: the move of the pre-unification ``retrieval/external_rerankers.py:275-389``
behaviour, kept beside the paper configs from the clients-final lineage on), exposed through the
harness's subprocess CLI:

    reference.py --mode <render|score> --pairs <file> --out <file> --tokenizer <repo@rev|path> \
                 [--device <cpu|cuda:0>]

- ``--mode render`` (stage 1): ``{"rows": [{"index", "shape", "pair", "query", "documents"}]}`` —
  the cut content spans the served wire carries per pairs row (the harness's rerank reference
  contract): the query settled once per row, the document spans cut to what remains.  That's an
  independent port of the product's role client for this recipe's declaration (the settle rule and
  the anchor-preserving cut of ``rcp_ndcg.data.preprocess``)—tokenizer and ``tokenizers`` only, no
  weights.  The pair's own prompt (what ``score`` scores) stays the paper's own construction below.
- ``--mode score`` (stage 2, GPU): ``{"rows": [{"index", "scores": [...]}]}`` — one raw relevance
  logit per document (vocabulary position 0 at the final position; no sigmoid/softmax), the paper's
  quantity.

Declared behaviours, none of which touches a score the paper measured:

1. The pairs row's ``instruction`` is ignored on both sides: the recipe declares
   ``instruction: none`` (the paper's served-path mode in
   ``experiments/paper/rerankers/ctxl_rerank_1b.yaml``) and the paper's in-process path never sent an
   instruction either (its ``instruction`` attribute stays ``None``).  No fold, no append.
2. ``score`` keeps the paper's whole-prompt right truncation at 8192 tokens, which for a pair whose
   document alone pushes the prompt over the cap drops the query block and the trailing `` ??``
   anchor the last-position score reads.  The served path never drops an anchor (the client-side cut
   keeps every fixed segment), so the recipe declares
   ``reference.known_deviations: [anchor_drop_over_cap]`` and the harness reports over-cap pairs in
   its non-gating table.
3. A query over the declared 4096-token share is a declared divergence row (the wire's settle rule
   ships it at the share whenever it exceeds it — the merged rerank client settles the shared query
   once per call — while the paper's score path keeps the query whole): the gating pairs keep
   queries within the share.

Paper-exact everywhere else: the two-line prompt (document before query, then `` ??``), left
padding, bfloat16 weights on CUDA and float32 on CPU, flash_attention_2 on CUDA (``None`` on CPU —
the environment fallback the paper never exercised; the card's CPU path is fp32 too), right
truncation of the whole prompt at 8192 tokens, and the paper's batching (length-descending
permutation, a padded-area budget, OOM halving). Where the model card and the paper code disagree
the paper code wins: the truncation budget is 8192 (``MAX_SEQ_LENGTH``, what the paper's factory
passed over the 32768 default), not the card's tokenizer-level 131072.

Runs as a subprocess in the reference environment (``requirements-reference.txt`` beside this file:
torch 2.9.1, transformers 4.57.6, flash-attn 2.8.3 on Linux) — never inside the harness process,
which imports no torch. ``render`` needs the tokenizer only (the ``tokenizers`` library);
``score`` needs the weights at the pinned revision (a ~3.28 GB download) and a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

__all__ = ["CtxlRerankReference", "load", "main", "prompt_text", "render", "served_spans"]

MODEL_ID = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b"
REVISION = "8fd1edf6a98564cb712064f884b8ef7df5c1b876"
MAX_SEQ_LEN = 8192
"""The paper's combined (query, document) prompt budget (the factory's ``MAX_SEQ_LENGTH``)."""
QUERY_MAX_TOKENS = 4096
"""The query's declared share (the paper's ``MAX_QUERY_LENGTH``): the wire settles the shared query
at the share whenever it exceeds it (the merged rerank client's settle rule); the paper's score
path keeps the query whole — an over-share query is a declared divergence row."""
BATCH_SIZE = 32
"""Documents per forward at most (experiments/paper/rerankers/ctxl_rerank_1b.yaml)."""
BATCH_SIZE_TOKENS = 15_000
"""The padded-area budget (documents * max prompt characters) the paper's batching keeps small."""
VOCAB_POSITION = 0
"""The vocabulary position whose logit is the relevance score (the checkpoint's token id 0, "!")."""
ADD_SPECIAL_TOKENS = True
"""The pair shape's post-processor flag (the engine tokenizes the render with it; this
checkpoint's ByteLevel post-processor adds no tokens)."""

_PROMPT_HEAD = "Check whether a given document contains information helpful to answer the query.\n"
_HEAD = f"{_PROMPT_HEAD}<Document> "
_MID = "\n<Query> "
_TAIL = " ??"

_loaded: CtxlRerankReference | None = None


def prompt_text(query: str, doc: str) -> str:
    """The paper's exact prompt string: document before query, bare query, the `` ??`` tail."""
    return f"{_HEAD}{doc}{_MID}{query}{_TAIL}"


def render(query: str, doc: str, instruction: str | None = None) -> str:
    """The paper's prompt for one pair. ``instruction`` is accepted and ignored (the recipe
    declares ``instruction: none``; a pairs row's instruction is never folded nor appended)."""
    del instruction
    return prompt_text(query, doc)


def _tokenizer_file(spec: str) -> Path:
    """The ``tokenizer.json`` file ``spec`` names: a local path as given, else the Hub's pinned copy."""
    candidate = Path(spec).expanduser()
    file = candidate / "tokenizer.json" if candidate.is_dir() else candidate
    if file.is_file():
        return file
    from huggingface_hub import hf_hub_download

    repo, _, revision = spec.partition("@")
    return Path(hf_hub_download(repo, "tokenizer.json", revision=revision or None))


def _load_backend_tokenizer(spec: str):
    """The counting tokenizer: the ``tokenizers`` library over the recipe's ``tokenizer.json``."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(_tokenizer_file(spec)))


def _count(text: str, tokenizer, *, add_special_tokens: bool = False) -> int:
    """The number of tokens of ``text``, as the engine counts it when ``add_special_tokens``."""
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)


def _offsets(text: str, tokenizer) -> list[tuple[int, int]]:
    """``(start, end)`` character offsets of each token of ``text``, in order."""
    return [(offset[0], offset[1]) for offset in tokenizer.encode(text, add_special_tokens=False).offsets]


def _token_prefix(text, max_tokens, tokenizer, *, rendered=None, add_special_tokens=False) -> str:
    """The longest prefix of ``text`` that ends at one of its first ``max_tokens`` token boundaries
    and counts at most ``max_tokens`` as the engine reads it (``rendered(piece)`` when given).

        The cut is located on the original text's offset mapping, so the result is a verbatim prefix
        (never tokens decoded back to text — NFC-safe).  Port of ``rcp_ndcg.data.preprocess``
        ``token_prefix``: the same galloping-then-binary search over token boundaries.
    """

    def count(piece: str) -> int:
        shown = rendered(piece) if rendered is not None else piece
        return _count(shown, tokenizer, add_special_tokens=add_special_tokens)

    if count(text) <= max_tokens:
        return text
    offsets = _offsets(text, tokenizer)

    def prefix(tokens: int) -> str:
        return text[: offsets[tokens - 1][1]] if tokens > 0 else ""

    def fits(tokens: int) -> bool:
        return count(prefix(tokens)) <= max_tokens

    over = min(max_tokens, len(offsets))
    if fits(over):
        return prefix(over)
    fitting, step = over - 1, 1
    while fitting > 0 and not fits(fitting):
        over, fitting, step = fitting, max(fitting - step, 0), step * 2
    while over - fitting > 1:
        middle = (over + fitting) // 2
        fitting, over = (middle, over) if fits(middle) else (fitting, middle)
    return prefix(fitting)


def served_spans(tokenizer, query: str, document: str) -> tuple[str, str]:
    """The cut content spans (query, document) the served wire carries for one pair.

    Port of the product's role client for this recipe's declaration: the query's span settles once
    per request — to its declared share (``QUERY_MAX_TOKENS``) whenever it exceeds it (the settle
    rule the wire carries; a bare pair fit binds the share on overflow only) — then through fit's
    probe pair (the query with an empty document) that guarantees the frame and its anchor fit even
    alone, and each document gets what remains.  Every cut is a verbatim token-boundary prefix; the
    frame and its trailing anchor are re-attached by the engine around these spans.
    """

    def assemble(q: str, d: str) -> str:
        return f"{_HEAD}{d}{_MID}{q}{_TAIL}"

    cap = MAX_SEQ_LEN
    if _count(query, tokenizer) > QUERY_MAX_TOKENS:
        q_final = _token_prefix(query, QUERY_MAX_TOKENS, tokenizer)
    else:
        q_final = query
    # fit's probe pair (the query with an empty document): the query keeps the frame room.
    q_final = _token_prefix(
        q_final, cap, tokenizer, rendered=lambda piece: assemble(piece, ""), add_special_tokens=ADD_SPECIAL_TOKENS
    )
    if (
        _count(assemble(q_final, ""), tokenizer, add_special_tokens=ADD_SPECIAL_TOKENS) >= cap
        and _count(document, tokenizer) > 0
    ):
        raise SystemExit(
            f"the query fills the pair budget of {cap} tokens and leaves the document nothing; "
            "lower QUERY_MAX_TOKENS (or raise MAX_SEQ_LEN), so the document keeps a share"
        )
    d_final = _token_prefix(
        document, cap, tokenizer, rendered=lambda piece: assemble(q_final, piece), add_special_tokens=ADD_SPECIAL_TOKENS
    )
    return q_final, d_final


class CtxlRerankReference:
    """The paper's ``ContextualRerank`` for one checkpoint, in the harness's load/score shape."""

    def __init__(self, tokenizer_spec: str) -> None:
        self.tokenizer_spec = tokenizer_spec
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: str | None = None

    def load(self, device: str | None = None) -> CtxlRerankReference:
        """Load the tokenizer and the causal LM (idempotent; completes a partially built instance).

        bfloat16 weights on CUDA with flash_attention_2 (the paper's exact configuration), float32 on
        CPU without it (the card's CPU path; a declared environment fallback the paper never ran).
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo, _, revision = self.tokenizer_spec.partition("@")
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True, revision=revision or None)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding, so the final position is the last real token of every row (card + paper code).
        self.tokenizer.padding_side = "left"
        if self.model is None:
            on_cuda = torch.cuda.is_available()
            model_kwargs: dict[str, Any] = {
                "dtype": torch.bfloat16 if on_cuda else torch.float32,
                "revision": revision or None,
            }
            if on_cuda:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            self.model = AutoModelForCausalLM.from_pretrained(repo, **model_kwargs)
            self.model.eval()
            self.device = device or ("cuda" if on_cuda else "cpu")
            self.model.to(self.device)
        return self

    def predict(self, query: str, docs: list[str]) -> list[float]:
        """One raw relevance logit per document, aligned with ``docs`` (the paper's batching, unchanged)."""
        prompts = [prompt_text(query, doc) for doc in docs]

        # Length-descending permutation for tight batches; the longest (most OOM-prone) prompts are
        # front-loaded so the backoff triggers early. Documents score independently, so batching and
        # order affect throughput only, never the per-document score.
        permutation = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        sorted_prompts = [prompts[i] for i in permutation]

        batches: list[list[str]] = []
        max_len = 0
        for text in sorted_prompts:
            text_len = len(text)
            over_tokens = bool(batches) and (len(batches[-1]) + 1) * max(max_len, text_len) > BATCH_SIZE_TOKENS
            over_count = bool(batches) and len(batches[-1]) >= BATCH_SIZE
            if not batches or over_tokens or over_count:
                batches.append([])
                max_len = 0
            batches[-1].append(text)
            max_len = max(max_len, text_len)

        sorted_scores: list[float] = []
        for batch in batches:
            sorted_scores.extend(self._forward_scores(batch))

        result = [0.0] * len(docs)
        for original_index, score in zip(permutation, sorted_scores, strict=True):
            result[original_index] = score
        return result

    def _forward_scores(self, batch: list[str]) -> list[float]:
        """The paper's forward: right truncation of the whole prompt at MAX_SEQ_LEN, logit[:, -1, 0]."""
        import torch

        try:
            with torch.no_grad():
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Left-padded, so position -1 is the last real token for every row; vocabulary index 0
                # is ctxl's relevance logit (the checkpoint's 1_LogitScore true_token_id).
                return out.logits[:, -1, VOCAB_POSITION].float().tolist()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            return self._forward_scores(batch[:mid]) + self._forward_scores(batch[mid:])


def load(tokenizer_spec: str, device: str | None = None) -> CtxlRerankReference:
    """The loaded reference (idempotent): the tokenizer from ``--tokenizer``, the weights from the Hub."""
    global _loaded
    if _loaded is None:
        _loaded = CtxlRerankReference(tokenizer_spec)
    if _loaded.model is None:
        _loaded.load(device)
    return _loaded


def main() -> int:
    """The harness's reference CLI: read the pairs file, write the mode's JSON to ``--out``."""
    parser = argparse.ArgumentParser(description="the ctxl-rerank-v2-instruct-multilingual-1b reference")
    parser.add_argument("--mode", required=True, choices=["render", "score"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.mode == "render":
        # The harness's rerank reference contract: the wire's content spans per pairs row (the
        # query settled once, each document cut to what remains). The row's instruction is ignored
        # (the recipe declares instruction: none).
        tokenizer = _load_backend_tokenizer(args.tokenizer)
        rows = []
        for index, row in enumerate(pairs):
            documents = [str(document) for document in row["documents"]]
            query_span, _ = served_spans(tokenizer, row["query"], documents[0] if documents else "")
            document_spans = [served_spans(tokenizer, row["query"], document)[1] for document in documents]
            rows.append({"index": index, "shape": "pair", "query": query_span, "documents": document_spans})
        output: dict[str, Any] = {"rows": rows}
    else:
        reference = load(args.tokenizer, args.device)
        rows = [
            {"index": index, "scores": reference.predict(row["query"], row["documents"])}
            for index, row in enumerate(pairs)
        ]
        output = {"rows": rows}
    Path(args.out).write_text(json.dumps(output) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
