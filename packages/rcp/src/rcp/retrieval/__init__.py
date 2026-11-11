"""First-stage retrieval, reranking and fusion.

* :data:`RetrieverConfig` (:class:`BM25Config`, :class:`DenseConfig`, :class:`LateInteractionConfig`, by ``kind``)
  with an :data:`EncoderConfig`, and :data:`RerankerConfig`; encoders and rerankers are the
  :mod:`rcp.inference` role configs by ``api`` (the served :class:`ServedEmbedding`,
  :class:`ServedPooling` and :class:`ServedReranker`, the hosted :class:`CohereEmbedding`, :class:`VoyageEmbedding`,
  :class:`GeminiEmbedding`, :class:`CohereReranker` and :class:`VoyageReranker`), see
  :mod:`rcp.retrieval.config`.
* :func:`index`, :func:`search`, :func:`retrieve`, :func:`rerank`, :func:`fuse`: a
  :class:`~rcp.data.Dataset` in, :class:`~rcp.data.Rankings` out.
* :func:`build_store`, :func:`load_store`, :func:`sweep`: the full-width embedding store (corpus and query
  vectors from one forward pass) and the ex-post Matryoshka sweep -- per declared ``k``, apply the head
  and score, so every output dimension is evaluated without another model call.

Every model is reached over the shared inference transport through its role client; there is no in-process
model code in the package.
"""

from rcp.inference.types import Embeddings, EncodeRole, l2_normalize  # noqa: F401  # re-exported
from rcp.retrieval._api import (
    INDEX_BEHAVIOUR_VERSION,
    RERANK_BEHAVIOUR_VERSION,
    RETRIEVE_BEHAVIOUR_VERSION,
    Index,
    fuse,
    index,
    load_index,
    rerank,
    retrieve,
    search,
)
from rcp.retrieval.config import (
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
from rcp.retrieval.store import build_store, load_store, sweep

__all__ = [
    "BM25Config",
    "CohereEmbedding",
    "CohereReranker",
    "DenseConfig",
    "Embeddings",
    "EncodeRole",
    "EncoderConfig",
    "GeminiEmbedding",
    "INDEX_BEHAVIOUR_VERSION",
    "Index",
    "LateInteractionConfig",
    "PluginEmbedding",
    "PluginPooling",
    "PluginReranker",
    "RERANK_BEHAVIOUR_VERSION",
    "RETRIEVE_BEHAVIOUR_VERSION",
    "RerankerConfig",
    "RetrieverConfig",
    "ServedEmbedding",
    "ServedPooling",
    "ServedReranker",
    "VoyageEmbedding",
    "VoyageReranker",
    "build_store",
    "fuse",
    "index",
    "l2_normalize",
    "load_index",
    "load_store",
    "rerank",
    "retrieve",
    "search",
    "sweep",
    "validate_reranker",
    "validate_retriever",
]
