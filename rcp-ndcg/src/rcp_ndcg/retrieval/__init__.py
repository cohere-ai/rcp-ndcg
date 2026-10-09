"""First-stage retrieval, reranking and fusion.

* :data:`RetrieverConfig` (:class:`BM25Config`, :class:`DenseConfig`, :class:`LateInteractionConfig`, by ``kind``)
  with an :data:`EncoderConfig`, and :data:`RerankerConfig`; encoders and rerankers are the
  :mod:`rcp_ndcg.inference` role configs by ``api`` (the served :class:`ServedEmbedding`,
  :class:`ServedPooling` and :class:`ServedReranker`, the hosted :class:`CohereEmbedding`, :class:`VoyageEmbedding`,
  :class:`GeminiEmbedding`, :class:`CohereReranker` and :class:`VoyageReranker`), see
  :mod:`rcp_ndcg.retrieval.config`.
* :func:`index`, :func:`search`, :func:`retrieve`, :func:`rerank`, :func:`fuse`: a
  :class:`~rcp_ndcg.data.Dataset` in, :class:`~rcp_ndcg.data.Rankings` out.

Every model is reached over the shared inference transport through its role client; there is no in-process
model code in the package.
"""

from rcp_ndcg.inference.types import Embeddings, EncodeRole, l2_normalize  # noqa: F401  # re-exported
from rcp_ndcg.retrieval._api import Index, fuse, index, load_index, rerank, retrieve, search
from rcp_ndcg.retrieval.config import (
    BM25Config,
    CohereEmbedding,
    CohereReranker,
    DenseConfig,
    EncoderConfig,
    GeminiEmbedding,
    LateInteractionConfig,
    PluginEmbedding,
    PluginPooling,
    PluginReranker,
    RerankerConfig,
    RetrieverConfig,
    ServedEmbedding,
    ServedPooling,
    ServedReranker,
    VoyageEmbedding,
    VoyageReranker,
    validate_reranker,
    validate_retriever,
)

__all__ = [
    "BM25Config",
    "CohereEmbedding",
    "CohereReranker",
    "DenseConfig",
    "Embeddings",
    "EncodeRole",
    "EncoderConfig",
    "GeminiEmbedding",
    "Index",
    "LateInteractionConfig",
    "PluginEmbedding",
    "PluginPooling",
    "PluginReranker",
    "RerankerConfig",
    "RetrieverConfig",
    "ServedEmbedding",
    "ServedPooling",
    "ServedReranker",
    "VoyageEmbedding",
    "VoyageReranker",
    "fuse",
    "index",
    "l2_normalize",
    "load_index",
    "rerank",
    "retrieve",
    "search",
    "validate_reranker",
    "validate_retriever",
]
