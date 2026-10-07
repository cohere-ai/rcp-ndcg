"""The reference implementation of recipe ``ctxl-rerank-v2-instruct-multilingual-1b``.

Derived from ``experiments/paper/rerankers/reference/contextual.py`` (the paper's in-process
``ContextualRerank``: the move of the pre-unification ``retrieval/external_rerankers.py:275-389``
behaviour, kept beside the paper configs since the inference layer was unified), exposed through the
harness's subprocess CLI:

    reference.py --mode <render|score> --pairs <file> --out <file> --tokenizer <repo@rev|path> \
                 [--device <cpu|cuda:0>]

- ``--mode render`` (stage 1): ``{"rows": [{"index", "shape": "pair", "query", "documents"}]}`` --
  the harness's rerank span format, filled with the spans the paper's prompt builder receives: the
  raw query and the raw documents, uncut (the paper's own cut is the whole-prompt right truncation
  at encode, inside ``score``).  No tokenizer, no weights.  Under the cap and within the query share
  these are exactly the spans the served wire carries; over the cap they differ by declaration
  (``anchor_drop_over_cap``: stage 1 reports such rows in its non-gating table).  The reference never
  reproduces the product client's cut (its settle rule or its anchor-preserving content cut).
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
3. A query over the declared 4096-token share is a declared divergence row (the merged rerank
   client settles the shared query once per call and ships it at the share whenever it exceeds it,
   while the paper's path has no query share and keeps the query whole): the gating pairs keep
   queries within the share.

Paper-exact everywhere else: the two-line prompt (document before query, then `` ??``), left
padding, bfloat16 weights on every device (the paper factory's ``DTYPE``), flash_attention_2 on
CUDA (the default attention on CPU, where flash-attention-2 does not exist), right
truncation of the whole prompt at 8192 tokens, and the paper's batching (length-descending
permutation, a padded-area budget, OOM halving). Where the model card and the paper code disagree
the paper code wins: the truncation budget is 8192 (``MAX_SEQ_LENGTH``, what the paper's factory
passed over the 32768 default), not the card's tokenizer-level 131072.

Runs as a subprocess in the reference environment (``requirements-reference.txt`` beside this file:
torch 2.9.1, transformers 4.57.6, flash-attn 2.8.3 on Linux) — never inside the harness process,
which imports no torch. ``render`` is pure string work (the standard library only);
``score`` needs the weights at the pinned revision (a ~3.28 GB download) and a GPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

__all__ = ["CtxlRerankReference", "load", "main", "paper_spans", "prompt_text", "render"]

MODEL_ID = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b"
REVISION = "8fd1edf6a98564cb712064f884b8ef7df5c1b876"
MAX_SEQ_LEN = 8192
"""The paper's combined (query, document) prompt budget (the factory's ``MAX_SEQ_LENGTH``)."""
BATCH_SIZE = 32
"""Documents per forward at most (experiments/paper/rerankers/ctxl_rerank_1b.yaml)."""
BATCH_SIZE_TOKENS = 15_000
"""The padded-area budget (documents * max prompt characters) the paper's batching keeps small."""
VOCAB_POSITION = 0
"""The vocabulary position whose logit is the relevance score (the checkpoint's token id 0, "!")."""

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


def paper_spans(row: dict[str, Any]) -> dict[str, Any]:
    """One pairs row in the harness's rerank span format: the spans the paper's prompt builder gets.

    The raw query and the raw documents, uncut: the paper builds its prompt from them and cuts only at
    encode (the whole-prompt right truncation of ``_forward_scores``).  The row's instruction is ignored
    (``instruction: none``)."""
    return {"query": str(row["query"]), "documents": [str(document) for document in row["documents"]]}


class CtxlRerankReference:
    """The paper's ``ContextualRerank`` for one checkpoint, in the harness's load/score shape."""

    def __init__(self, tokenizer_spec: str) -> None:
        self.tokenizer_spec = tokenizer_spec
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: str | None = None

    def load(self, device: str | None = None) -> CtxlRerankReference:
        """Load the tokenizer and the causal LM (idempotent; completes a partially built instance).

        bfloat16 weights on every device (the paper factory's ``DTYPE = "bfloat16"``, passed explicitly,
        so the class's own float32-on-CPU default never applied), flash_attention_2 on CUDA and the
        default attention elsewhere (flash-attention-2 does not exist on CPU) -- the same rule in
        the 1b, 2b and 6b references.
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
            target = device or ("cuda" if torch.cuda.is_available() else "cpu")
            model_kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "revision": revision or None}
            if target.startswith("cuda"):
                model_kwargs["attn_implementation"] = "flash_attention_2"
            self.model = AutoModelForCausalLM.from_pretrained(repo, **model_kwargs)
            self.model.eval()
            self.device = target
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
        # The harness's rerank span format, filled with the paper's own spans (uncut; see paper_spans).
        rows = [{"index": index, "shape": "pair", **paper_spans(row)} for index, row in enumerate(pairs)]
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
