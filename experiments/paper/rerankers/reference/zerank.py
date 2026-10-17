"""ZeroEntropy ZeRank 1 / 1-small / 2: chat template (system=query, user=document) plus the
"Yes"-token logit divided by 5, sigmoided: the paper's exact in-process implementation.

Moved unchanged in behaviour from ``rcp-ndcg/src/rcp_ndcg/retrieval/external_rerankers.py`` (the ``zerank``
framework of the deleted in-process path) when the package stopped carrying in-process models
(the unified-inference change). The old ``load_external_model`` factory built it with
``max_seq_len=8192`` (the paper's ``MAX_SEQ_LENGTH``), bfloat16 and the config's revision. The
multi-GPU ``AccelState`` sharding is not carried: the paper's runs were single-GPU per model, and a
rerun that needs several GPUs shards the queries itself.

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
# ZeroEntropy ZeRank (chat template + "Yes" token logit).
# ---------------------------------------------------------------------------


class ZerankRerank(EvalCrossEncoder):
    """ZeRank reranker: chat template (system=query, user=doc) + Yes logit.

    Adapted from https://huggingface.co/zeroentropy/zerank-1 (Apache-2.0, see NOTICE).  The score is
    ``sigmoid(yes_logit / 5)`` at the final token position.
    """

    def __init__(
        self,
        model_name_or_path: str = "zeroentropy/zerank-2",
        max_seq_len: int = 8192,
        batch_size_tokens: int = 15_000,
        dtype: Any = None,
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name_or_path
        self.max_seq_length = max_seq_len
        self.batch_size_tokens = batch_size_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="right", revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, dtype=dtype or torch.bfloat16, revision=revision
        )
        self.model.eval()

        self.yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def to(self, device: str) -> ZerankRerank:
        self.model.to(device)
        self.device = device
        return self

    def _format_inputs(self, query: str, docs: list[str]) -> list[str]:
        texts: list[str] = []
        for doc in docs:
            messages = [
                {"role": "system", "content": query.strip()},
                {"role": "user", "content": doc.strip()},
            ]
            texts.append(self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        return texts

    def _batch_logits(self, batch: list[str]) -> list[float]:
        import torch

        try:
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch, padding=True, return_tensors="pt", truncation=True, max_length=self.max_seq_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, use_cache=False)
                attention_mask = inputs["attention_mask"]
                last_positions = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(outputs.logits.shape[0], device=self.device)
                last_logits = outputs.logits[batch_indices, last_positions]
                yes_logits = last_logits[:, self.yes_token_id]
                return [float(v) / 5.0 for v in yes_logits]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            return self._batch_logits(batch[:mid]) + self._batch_logits(batch[mid:])

    def predict(self, query: str, docs: list[str]) -> list[float]:
        texts = self._format_inputs(query, docs)

        # Length-descending permutation for tight token buckets.
        permutation = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        sorted_texts = [texts[i] for i in permutation]

        batches: list[list[str]] = []
        max_length = 0
        for text in sorted_texts:
            text_len = len(text)
            if not batches or (len(batches[-1]) + 1) * max(max_length, text_len) > self.batch_size_tokens:
                batches.append([])
                max_length = 0
            batches[-1].append(text)
            max_length = max(max_length, text_len)

        all_logits: list[float] = []
        for batch in batches:
            all_logits.extend(self._batch_logits(batch))

        from scipy.special import expit

        scores = expit(all_logits).tolist()

        result = [0.0] * len(docs)
        for orig_idx, score in zip(permutation, scores, strict=True):
            result[orig_idx] = score
        return result
