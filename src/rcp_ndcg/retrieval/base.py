"""The sparse retrievers' contract: build an index from a corpus, load it, retrieve the top k for a query.

Dense retrieval goes through an :class:`~rcp_ndcg.retrieval.encoder.Encoder` instead -- see
:mod:`rcp_ndcg.retrieval.encoder` for the contract and :mod:`rcp_ndcg.retrieval.encoders` for the backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rcp_ndcg_core.content import Content


class BaseRetriever(ABC):
    """
    Base class for all retrievers.

    Retrievers are designed for single-query, live search scenarios.
    For batch/offline LLM judging, see :func:`rcp_ndcg.llm.judge`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this retriever."""
        ...

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: Search query string
            k: Number of results to return

        Returns:
            List of result dicts with keys:
                - id: int - Document ID (corpus index)
                - text: str - Document text
                - score: float - Relevance score (higher is better)
        """
        ...

    def retrieve_batch(self, queries: Sequence[str], k: int = 5) -> list[list[dict[str, Any]]]:
        """Retrieve for many queries at once.

        The default loops :meth:`retrieve`, which is right for a local index
        where each call is cheap.  Retrievers whose per-query cost is a network
        round trip override this so a run costs one batched call instead of one
        call per query.
        """
        return [self.retrieve(query, k=k) for query in queries]

    @classmethod
    @abstractmethod
    def build_index(
        cls,
        corpus: Sequence[Content | str],
        dataset_dir: Path,
        **kwargs: Any,
    ) -> None:
        """
        Build and persist retriever-specific index under dataset_dir.

        Args:
            corpus: Documents as content parts. Plain strings are accepted and
                lifted, so text-only callers need no change; a sparse retriever
                narrows this back to text itself, since it has no other option.
            dataset_dir: Directory to save indices
            **kwargs: Retriever-specific parameters
        """
        ...

    @classmethod
    @abstractmethod
    def from_index(cls, dataset_dir: Path, corpus: Sequence[Content | str], **kwargs: Any) -> BaseRetriever:
        """
        Load a retriever from pre-built index.

        Args:
            dataset_dir: Directory containing indices
            corpus: Documents, used to attach bodies to returned results
            **kwargs: Retriever-specific parameters

        Returns:
            Initialized retriever instance
        """
        ...


def to_texts(items: Sequence[Content | str]) -> list[str]:
    """The text-only view, for retrievers that cannot represent anything else.

    Used by the sparse retrievers. Lossy on purpose and only where the loss is
    inherent: BM25 has no notion of a page image, so an image-only document
    correctly reads as empty rather than being rejected -- a fused corpus with
    both text and images is still perfectly indexable on its text.
    """
    return [item if isinstance(item, str) else item.text for item in items]


__all__ = [
    "BaseRetriever",
    "to_texts",
]
