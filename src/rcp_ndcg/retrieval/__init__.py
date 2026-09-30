"""First-stage retrieval, reranking and fusion.

* :data:`RetrieverConfig` (:class:`BM25Config`, :class:`DenseConfig`, :class:`LateInteractionConfig`, by ``kind``)
  with an :data:`EncoderConfig`, and :data:`RerankerConfig`; encoders and rerankers are unions by ``provider``
  (:class:`Local`, the served :class:`OpenAICompatible`, and the vendors :class:`Cohere`, :class:`Voyage`,
  :class:`Gemini`), see :mod:`rcp_ndcg.retrieval.config`.
* :func:`index`, :func:`search`, :func:`retrieve`, :func:`rerank`, :func:`fuse`: a
  :class:`~rcp_ndcg.data.Dataset` in, :class:`~rcp_ndcg.data.Rankings` out.

In-process models (``provider: local``) need the ``[local]`` extra and are imported only when used.
"""

from rcp_ndcg.retrieval._api import Index, fuse, index, load_index, rerank, retrieve, search
from rcp_ndcg.retrieval.config import (
    BM25Config,
    Cohere,
    DenseConfig,
    EncoderConfig,
    Gemini,
    LateInteractionConfig,
    Local,
    LocalEncoder,
    OpenAICompatible,
    OpenAICompatibleEncoder,
    OpenAICompatibleReranker,
    RerankerConfig,
    RetrieverConfig,
    Voyage,
)

__all__ = [
    "BM25Config",
    "Cohere",
    "DenseConfig",
    "EncoderConfig",
    "Gemini",
    "Index",
    "LateInteractionConfig",
    "Local",
    "LocalEncoder",
    "OpenAICompatible",
    "OpenAICompatibleEncoder",
    "OpenAICompatibleReranker",
    "RerankerConfig",
    "RetrieverConfig",
    "Voyage",
    "fuse",
    "index",
    "load_index",
    "rerank",
    "retrieve",
    "search",
]
