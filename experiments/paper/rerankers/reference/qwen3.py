"""Qwen3-Reranker 0.6B / 4B / 8B, scored on yes/no token log-softmax: the paper's exact in-process implementation.

Moved unchanged in behaviour from ``src/rcp_ndcg/retrieval/external_rerankers.py`` (the ``qwen_og``
framework of the deleted in-process path) when the package stopped carrying in-process models
(RFC-0001, option 1). The old ``load_external_model`` factory built it with
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
# Qwen3-Reranker (yes/no token scoring).
# ---------------------------------------------------------------------------


class QwenOGRerank(EvalCrossEncoder):
    """Qwen3-Reranker scored on yes/no token log-softmax.

    Reproduces the released Qwen3-Reranker recipe (its model card's usage code, Apache-2.0, see NOTICE): a causal
    LM is prompted with ``<Instruct>/<Query>/<Document>`` and the relevance score is
    ``softmax([no_logit, yes_logit])[yes]`` at the final position.
    """

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-Reranker-8B",
        max_seq_len: int = 8192,
        batch_size: int = 8,
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
        dtype: Any = None,
        attn_implementation: str | None = "flash_attention_2",
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", revision=revision)
        self.max_length = max_seq_len
        self.model_name = model_name_or_path
        self.batch_size = batch_size
        self.instruction = instruction

        model_kwargs: dict[str, Any] = {"dtype": dtype or torch.float16, "revision": revision}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        self.model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
            'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
            "<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

    def to(self, device: str) -> QwenOGRerank:
        self.model.to(device)
        self.device = device
        return self

    def _process_inputs(self, pairs: list[str]) -> dict:
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _compute_scores(self, inputs: dict) -> list[float]:
        import torch

        logits = self.model(**inputs).logits[:, -1, :]
        true_vector = logits[:, self.token_true_id]
        false_vector = logits[:, self.token_false_id]
        stacked = torch.stack([false_vector, true_vector], dim=1)
        scores = torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp()
        return scores.float().tolist()

    def _score_pairs(self, pairs: list[str]) -> list[float]:
        import torch

        try:
            return self._compute_scores(self._process_inputs(pairs))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(pairs) == 1:
                raise
            mid = len(pairs) // 2
            return self._score_pairs(pairs[:mid]) + self._score_pairs(pairs[mid:])

    def predict(self, query: str, docs: list[str]) -> list[float]:
        import torch

        pairs = [f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}" for doc in docs]
        scores: list[float] = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                scores.extend(self._score_pairs(pairs[i : i + self.batch_size]))
        if len(scores) != len(docs):
            raise RuntimeError(f"QwenOGRerank: expected {len(docs)} scores, got {len(scores)}")
        return scores
