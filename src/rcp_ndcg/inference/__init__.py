"""The inference layer: endpoints, wire types, the adapter seam and the transport's interface, shared by every
role (the judge, the encoders, the rerankers).

The layer sits between :mod:`rcp_ndcg.data` and :mod:`rcp_ndcg.retrieval`, so both the judge and the retrieval
paths send their requests through one set of names:

* :class:`Endpoint` -- where a model is served, which checkpoint, and how the client talks to it
  (:mod:`rcp_ndcg.inference.endpoint`);
* the wire types -- :class:`Call`, :class:`Reply`, :class:`TokenCount`, :class:`Usage`, :class:`EngineInfo`, and
  the per-role request and result types (the judge's :class:`CompletionInput` and :class:`Completion`; the
  encoders' :class:`EncodeRole`, :class:`EmbedRequest`, :class:`PoolRequest`, :class:`Embeddings` and
  :func:`l2_normalize`; the rerankers' :class:`RerankRequest` and :class:`RerankResult`)
  (:mod:`rcp_ndcg.inference.types`);
* :class:`Adapter` and its role-scoped registry -- the one seam a third party implements (C2), selected from a
  config with ``api: <name>`` within the config's role (:mod:`rcp_ndcg.inference.adapters.base`); the shipped
  embed adapters (`openai_embeddings` and the hosted `cohere`, `voyage`, `gemini` profiles) and rerank adapters
  (the served Cohere-shaped wire and its hosted `cohere` and `voyage` profiles) register at import
  (:mod:`rcp_ndcg.inference.adapters.embeddings`, :mod:`rcp_ndcg.inference.adapters.rerank`);
* :class:`Sender` and :class:`Transport` -- the transport's frozen interface; behaviour arrives with the transport work
  (:mod:`rcp_ndcg.inference.transport`);
* the role clients -- the content decisions above the wire: :class:`EmbeddingClient` (dense embeddings;
  :mod:`rcp_ndcg.inference.clients`);
* the role endpoint configs -- :class:`EmbeddingEndpoint`, :class:`PoolingEndpoint`, :class:`RerankEndpoint`
  (:mod:`rcp_ndcg.inference.config`);
* the role clients -- :class:`RerankClient` (:mod:`rcp_ndcg.inference.clients`);
* :data:`FAKE_SCHEME` -- the offline fakes' URL scheme (:mod:`rcp_ndcg.inference.fake`).
"""

from rcp_ndcg.inference.adapters.base import (
    ADAPTER_ENTRY_POINTS,
    ROLES,
    Adapter,
    AdapterRole,
    get_adapter,
    known_adapters,
    register_adapter,
)
from rcp_ndcg.inference.adapters.rerank import CohereRerankAdapter, RerankAdapter, VoyageRerankAdapter
from rcp_ndcg.inference.clients import EmbeddingClient, RerankClient
from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.inference.fake import FAKE_SCHEME
from rcp_ndcg.inference.transport import Sender, Transport
from rcp_ndcg.inference.types import (
    Call,
    Completion,
    CompletionInput,
    Embeddings,
    EmbedRequest,
    EncodeRole,
    EngineInfo,
    PoolRequest,
    Reply,
    RerankRequest,
    RerankResult,
    TokenCount,
    Usage,
    l2_normalize,
)

__all__ = [
    "ADAPTER_ENTRY_POINTS",
    "Adapter",
    "AdapterRole",
    "Call",
    "CohereRerankAdapter",
    "Completion",
    "CompletionInput",
    "EmbedRequest",
    "Embeddings",
    "EmbeddingClient",
    "EmbeddingEndpoint",
    "EncodeRole",
    "EngineInfo",
    "Endpoint",
    "FAKE_SCHEME",
    "PoolRequest",
    "PoolingEndpoint",
    "ROLES",
    "RerankAdapter",
    "RerankClient",
    "RerankEndpoint",
    "RerankRequest",
    "RerankResult",
    "Reply",
    "Sender",
    "TokenCount",
    "Transport",
    "Usage",
    "VoyageRerankAdapter",
    "get_adapter",
    "known_adapters",
    "l2_normalize",
    "register_adapter",
]
