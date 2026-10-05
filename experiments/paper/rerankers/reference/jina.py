"""Jina reranker v3 through the checkpoint's own ``model.rerank()``: the paper's exact in-process implementation.

Moved unchanged in behaviour from ``src/rcp_ndcg/retrieval/external_rerankers.py`` (the ``jina_hf``
framework of the deleted in-process path) when the package stopped carrying in-process models
(RFC-0001, option 1). The old ``load_external_model`` factory built it with
``max_seq_len=8192`` (the paper's ``MAX_SEQ_LENGTH``), the config's ``batch_size``, bfloat16 and
the config's revision. The multi-GPU ``AccelState`` sharding is not carried: the paper's runs were
single-GPU per model, and a rerun that needs several GPUs shards the queries itself.

This module is not part of the package and must never be imported by it; see the README beside it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EvalCrossEncoder(ABC):
    """Abstract base for ``predict(query, docs)``-style rerankers."""

    @abstractmethod
    def predict(self, query: str, docs: list[str]) -> list[float]:
        """Return one relevance score per document, aligned with ``docs``."""

    def to(self, device: str) -> EvalCrossEncoder:
        """Move the model to ``device``.  Default is a no-op (API backends)."""
        return self


# ---------------------------------------------------------------------------
# Jina reranker v3 (local model.rerank() API).
# ---------------------------------------------------------------------------


class JinaRerank(EvalCrossEncoder):
    """Jina local reranker (v2 / v3) via the model's built-in ``rerank()``."""

    def __init__(
        self,
        model_name_or_path: str = "jinaai/jina-reranker-v3",
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        self.model_name = model_name_or_path
        self.is_v3 = "v3" in model_name_or_path

        model_cls = AutoModel if self.is_v3 else AutoModelForSequenceClassification
        self.model = model_cls.from_pretrained(
            model_name_or_path, dtype="auto", trust_remote_code=True, revision=revision
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, revision=revision)
        self.model.eval()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def to(self, device: str) -> JinaRerank:
        self.model.to(device)
        self.device = device
        return self

    def predict(self, query: str, docs: list[str]) -> list[float]:
        non_empty = [(i, d) for i, d in enumerate(docs) if d.strip()]
        if not non_empty:
            return [0.0] * len(docs)

        indices, filtered_docs = zip(*non_empty, strict=True)
        results = self.model.rerank(query, documents=list(filtered_docs))
        scored = [0.0] * len(filtered_docs)
        for r in results:
            scored[r["index"]] = r["relevance_score"]

        output = [0.0] * len(docs)
        for orig_idx, score in zip(indices, scored, strict=True):
            output[orig_idx] = score
        return output
