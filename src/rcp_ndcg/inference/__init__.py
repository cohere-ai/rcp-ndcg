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
* :class:`Adapter` and its registry -- the one seam a third party implements (C2), selected from a config with
  ``api: <name>`` (:mod:`rcp_ndcg.inference.adapters.base`);
* :class:`Sender` and :class:`Transport` -- the transport every role sends through: replicas, retries, parking,
  the sync bridge, usage (:mod:`rcp_ndcg.inference.transport`);
* the role endpoint configs -- :class:`EmbeddingEndpoint`, :class:`PoolingEndpoint`, :class:`RerankEndpoint`
  (:mod:`rcp_ndcg.inference.config`);
* :data:`FAKE_SCHEME` -- the offline fakes' URL scheme, and :func:`register_fake_route` for their extra routes
  (:mod:`rcp_ndcg.inference.fake`).
"""

from rcp_ndcg.inference.adapters.base import (
    ADAPTER_ENTRY_POINTS,
    Adapter,
    AdapterRole,
    get_adapter,
    known_adapters,
    register_adapter,
)
from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.inference.fake import FAKE_SCHEME, FakeEndpoint, register_fake_route
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
    "Completion",
    "CompletionInput",
    "EmbedRequest",
    "Embeddings",
    "EmbeddingEndpoint",
    "EncodeRole",
    "EngineInfo",
    "Endpoint",
    "FAKE_SCHEME",
    "FakeEndpoint",
    "PoolRequest",
    "PoolingEndpoint",
    "RerankEndpoint",
    "RerankRequest",
    "RerankResult",
    "Reply",
    "Sender",
    "TokenCount",
    "Transport",
    "Usage",
    "get_adapter",
    "known_adapters",
    "l2_normalize",
    "register_adapter",
    "register_fake_route",
]
