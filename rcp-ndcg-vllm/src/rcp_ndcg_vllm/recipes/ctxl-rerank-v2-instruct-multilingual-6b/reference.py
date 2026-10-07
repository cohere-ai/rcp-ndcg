"""Reference implementation for ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b.

Derived with unchanged behaviour from the paper's exact in-process scorer
(``experiments/paper/rerankers/reference/contextual.py``, class ``ContextualRerank``; itself
``rcp-ndcg/src/rcp_ndcg/retrieval/external_rerankers.py:275-389`` until the in-process path left the
package), with the constructor arguments the paper's factory passed for this checkpoint:
``max_seq_len=8192`` (the paper's ``MAX_SEQ_LENGTH``), ``batch_size=8``
(``experiments/paper/rerankers/ctxl_rerank_6b.yaml`` as this worktree pins it; the migrated
served-path config carries the budgets and ``instruction: none``), bfloat16 weights and the
revision the recipe pins. The score is the raw logit of vocabulary position 0 at the final position
of the 2-line prompt (document before query, then " ??"); no sigmoid, no softmax, no temperature.

Deviations kept from the paper's code, declared in recipe.yaml:

- the paper right-truncates the whole rendered prompt at 8192 tokens
  (``truncation=True, max_length=8192``), dropping the tail anchors (the query and the " ??")
  on pairs over the cap -- recipe ``reference.known_deviations: [anchor_drop_over_cap]``; the
  served path cuts the content spans only and re-attaches the frame instead;
- the prompt carries no instruction: the paper's factory never set ``ContextualRerank.instruction``
  and the recipe declares ``instruction: none`` -- the pairs file's per-row instruction field is
  therefore ignored here (the served path folds none).

Interface (the brief's in-process contract; the reference runs as a subprocess, never imported by
the harness, which holds no torch -- the equivalence harness itself drives the CLI below):

    load(device=None)                                            -> the model on ``device``
    render(query, doc, instruction=None) -> list[int]            -> the prompt's token ids
    score(query, docs, instruction=None) -> list[float]          -> one raw logit per document

and the CLI the equivalence harness invokes:

    reference.py --mode render --pairs <file> --out <file> --tokenizer <repo@rev|path> [--device cpu]
    reference.py --mode score  --pairs <file> --out <file> --tokenizer <repo@rev|path> --device cuda:0

``render`` mode is string-only (the paper constructs the prompt text and truncates at encode;
for pairs under the cap the text is byte-identical to the served render, which is what stage 1
compares) and needs neither torch nor transformers; ``score`` mode needs the reference
environment (``requirements-reference.txt`` beside recipe.yaml: torch, transformers, the
flash-attn wheel the paper ran with -- the model is ~14.9 GB stored, a GPU run). ``score`` mode
tokenizes with the ``--tokenizer`` spec the harness passes (the recipe's pinned tokenizer).

This module is not part of the package and is never imported by it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODEL_ID = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b"
REVISION = "f14ca1a2fc204dc0934c84a3d2e278f8ff646b80"
DEFAULT_TOKENIZER_SPEC = f"{MODEL_ID}@{REVISION}"
#: The paper's combined (query, document) token budget (``MAX_SEQ_LENGTH``), not the card's 32768.
MAX_SEQ_LEN = 8192
#: The score reads this vocabulary position at the final position (the paper: logits[:, -1, 0]).
VOCAB_POSITION = 0
#: Docs per forward (the paper's in-process config for this checkpoint, batch_size 8 at the
#: pre-migration experiments/paper/rerankers/ctxl_rerank_6b.yaml).
BATCH_SIZE = 8
#: Padded-area budget (docs * max_char_len) that keeps near-max-length batches from OOMing.
BATCH_SIZE_TOKENS = 15_000


def _prompt_text(query: str, doc: str, instruction: str | None = None) -> str:
    """The exact prompt string: document before query, an optional instruction, then " ??".

    This is the paper's ``ContextualRerank._format_prompts`` body. With ``instruction=None``
    (the recipe's declared mode) it is byte-identical to the served render of the recipe's
    declared pair shape for any pair under the budget.
    """
    instruction_text = f" {instruction}" if instruction else ""
    return (
        "Check whether a given document contains information helpful to answer the query.\n"
        f"<Document> {doc}\n<Query> {query}{instruction_text} ??"
    )


def _split_tokenizer_spec(spec: str) -> tuple[str, str | None]:
    """A ``repo@revision`` spec into (repo id, revision); a local path passes through unchanged."""
    if spec.startswith(("/", "./", "../", "~")) or spec.endswith(".json") or Path(spec).exists():
        return spec, None
    repo, _, revision = spec.partition("@")
    return repo, revision or None


class CtxlRerank:
    """The paper's in-process scorer for this checkpoint, unchanged in behaviour."""

    def __init__(self, tokenizer_spec: str = DEFAULT_TOKENIZER_SPEC) -> None:
        self.tokenizer_spec = tokenizer_spec
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: str | None = None

    def load(self, device: str | None = None) -> CtxlRerank:
        """Load the tokenizer and the causal LM onto ``device``; completes a render()-only instance.

        bfloat16 weights on GPU, float32 on CPU (the card's fallback; the paper ran bfloat16) and
        flash_attention_2, as the paper's code pins. Idempotent (a tokenizer loaded for ``render``
        is kept, the model attaches around it), so render-then-score works in one process.
        """
        import torch
        from transformers import AutoModelForCausalLM

        if self.tokenizer is None:
            self.load_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # position -1 is the last real token of every row
        if self.model is None:
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                dtype=dtype,
                revision=REVISION,
                attn_implementation="flash_attention_2",
            )
            self.model.eval()
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        return self

    def load_tokenizer(self) -> None:
        """Load only the tokenizer, from :attr:`tokenizer_spec` (the recipe's pinned one)."""
        from transformers import AutoTokenizer

        name, revision = _split_tokenizer_spec(self.tokenizer_spec)
        self.tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def render(self, query: str, doc: str, instruction: str | None = None) -> list[int]:
        """Token ids of the exact prompt (the tokenizer's post-processor prepends <s>, as served)."""
        if self.tokenizer is None:
            self.load_tokenizer()
        return self.tokenizer(_prompt_text(query, doc, instruction), add_special_tokens=True)["input_ids"]

    def score(self, query: str, docs: list[str], instruction: str | None = None) -> list[float]:
        """One raw relevance logit per document, aligned with ``docs`` (needs ``load``).

        The paper's batching: length-descending permutation, a padded-area budget
        (docs * max_char_len <= ``BATCH_SIZE_TOKENS``) with the hard cap ``BATCH_SIZE``, and OOM
        halving. Batching affects throughput only, never the per-document score.
        """
        self.load()
        prompts = [_prompt_text(query, doc, instruction) for doc in docs]

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
        for orig_idx, score in zip(permutation, sorted_scores, strict=True):
            result[orig_idx] = score
        return result

    def _forward_scores(self, batch: list[str]) -> list[float]:
        import torch

        try:
            with torch.no_grad():
                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_LEN,  # right truncation of the whole prompt (paper behaviour)
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Left-padded: position -1 is the last real token for every row; vocabulary index 0
                # is ctxl's relevance logit (1_LogitScore/config.json true_token_id 0).
                return out.logits[:, -1, VOCAB_POSITION].float().tolist()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            return self._forward_scores(batch[:mid]) + self._forward_scores(batch[mid:])


_instance: CtxlRerank | None = None


def _reference(tokenizer_spec: str = DEFAULT_TOKENIZER_SPEC) -> CtxlRerank:
    global _instance
    if _instance is None:
        _instance = CtxlRerank(tokenizer_spec)
    return _instance


def load(device: str | None = None) -> CtxlRerank:
    """The harness's ``load(device)``: the reference on ``device``; idempotent, render-then-score safe."""
    return _reference().load(device)


def render(query: str, doc: str, instruction: str | None = None) -> list[int]:
    """The harness's ``render(...)``: the prompt's token ids (tokenizer-only, no weights)."""
    return _reference().render(query, doc, instruction)


def score(query: str, docs: list[str], instruction: str | None = None) -> list[float]:
    """The harness's ``score(...)``: one raw relevance logit per document (needs ``load``)."""
    return _reference().score(query, docs, instruction)


def _main() -> int:
    parser = argparse.ArgumentParser(description="the ctxl-rerank-v2-instruct-multilingual-6b reference")
    parser.add_argument("--mode", required=True, choices=["render", "score"])
    parser.add_argument("--pairs", required=True, help='JSONL: one {"query": str, "documents": [str, ...]} per row')
    parser.add_argument("--out", required=True, help="where the mode's JSON result is written")
    parser.add_argument("--tokenizer", required=True, help="the recipe's tokenizer spec (repo@revision or a path)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]

    if args.mode == "render":
        # String-only: the paper builds the prompt text and truncates ids at encode, so the text
        # needs no tokenizer; for pairs under the cap it is byte-identical to the served render.
        # The recipe declares instruction: none, so the rows' instruction field stays unused.
        output = {
            "rows": [
                {"index": index, "shape": "pair", "text": _prompt_text(row["query"], row["documents"][0])}
                for index, row in enumerate(rows)
            ]
        }
    else:
        reference = _reference(args.tokenizer)
        reference.load(args.device)
        output = {
            "rows": [
                {"index": index, "scores": reference.score(row["query"], list(row["documents"]))}
                for index, row in enumerate(rows)
            ]
        }
    Path(args.out).write_text(json.dumps(output, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
