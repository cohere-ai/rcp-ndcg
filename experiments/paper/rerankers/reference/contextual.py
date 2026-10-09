"""ContextualAI ctxl-rerank 1B / 2B / 6B, scored on the vocabulary-position-0 logit: the paper's
exact in-process implementation.

Moved unchanged in behaviour from ``rcp-ndcg/src/rcp_ndcg/retrieval/external_rerankers.py`` (the ``contextual``
framework of the deleted in-process path) when the package stopped carrying in-process models
(the unified-inference change). The old ``load_external_model`` factory built it with
``max_seq_len=8192`` (the paper's ``MAX_SEQ_LENGTH``), the config's ``batch_size``, bfloat16 and
the config's revision. The multi-GPU ``AccelState`` sharding is not carried: the paper's runs were
single-GPU per model, and a rerun that needs several GPUs shards the queries itself.

This module is not part of the package and must never be imported by it; see the README beside it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EvalCrossEncoder(ABC):
    """Abstract base for ``predict(query, docs)``-style rerankers."""

    @abstractmethod
    def predict(self, query: str, docs: list[str]) -> list[float]:
        """Return one relevance score per document, aligned with ``docs``."""

    def to(self, device: str) -> EvalCrossEncoder:
        """Move the model to ``device``.  Default is a no-op (API backends)."""
        return self


# ---------------------------------------------------------------------------
# ContextualAI ctxl-rerank (vocab-position-0 logit).
# ---------------------------------------------------------------------------


class ContextualRerank(EvalCrossEncoder):
    """ContextualAI ctxl-rerank: causal LM scored on vocab-position-0 logit.

    Based on
    https://huggingface.co/ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b.
    """

    def __init__(
        self,
        model_name_or_path: str = "ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b",
        max_seq_len: int = 32768,
        batch_size: int = 32,
        batch_size_tokens: int = 15_000,
        instruction: str | None = None,
        dtype: Any = None,
        attn_implementation: str | None = "flash_attention_2",
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name_or_path
        self.max_seq_len = max_seq_len
        # ``batch_size`` is a hard cap on docs per forward; ``batch_size_tokens``
        # is a padded-area budget (docs * max_char_len) that keeps long-document
        # batches small.  ctxl is a 32k-context causal LM, so a fixed batch of
        # near-max-length docs OOMs -- the token budget prevents that, and the
        # per-batch OOM backoff in ``predict`` is a final safety net.
        self.batch_size = batch_size
        self.batch_size_tokens = batch_size_tokens
        self.instruction = instruction

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "dtype": dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            "revision": revision,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        self.model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def to(self, device: str) -> ContextualRerank:
        self.model.to(device)
        self.device = device
        return self

    def _format_prompts(self, query: str, docs: list[str]) -> list[str]:
        instruction = f" {self.instruction}" if self.instruction else ""
        return [
            f"Check whether a given document contains information helpful to answer the query.\n"
            f"<Document> {doc}\n<Query> {query}{instruction} ??"
            for doc in docs
        ]

    def _forward_scores(self, batch: list[str]) -> list[float]:
        import torch

        try:
            with torch.no_grad():
                enc = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_len
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                # Left-padded, so the final position is the last real token for
                # every row; vocab index 0 is ctxl's relevance logit.
                return out.logits[:, -1, 0].float().tolist()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            return self._forward_scores(batch[:mid]) + self._forward_scores(batch[mid:])

    def predict(self, query: str, docs: list[str]) -> list[float]:
        prompts = self._format_prompts(query, docs)

        # Length-descending permutation for tight token buckets; longest (most
        # OOM-prone) prompts are front-loaded so the backoff triggers early.
        permutation = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        sorted_prompts = [prompts[i] for i in permutation]

        # Padded-area budget: keep ``docs * max_char_len`` under the token budget
        # and never exceed ``batch_size`` docs.  Docs are scored independently, so
        # batching/order only affect throughput -- not the per-doc score.
        batches: list[list[str]] = []
        max_len = 0
        for text in sorted_prompts:
            text_len = len(text)
            over_tokens = bool(batches) and (len(batches[-1]) + 1) * max(max_len, text_len) > self.batch_size_tokens
            over_count = bool(batches) and len(batches[-1]) >= self.batch_size
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
