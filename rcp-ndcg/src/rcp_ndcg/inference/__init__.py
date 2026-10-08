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
  adapters register when :mod:`rcp_ndcg.inference.adapters` imports: the judge role's `openai_chat`, the
  embed role's `openai_embeddings` and its hosted `cohere`, `voyage`, `gemini` profiles, the rerank role's
  served Cohere-shaped wire and its hosted `cohere` and `voyage` profiles, and the multi-vector role's
  `vllm_pooling`;
* :class:`Sender` and :class:`Transport` -- the transport every role sends through: replicas, retries, parking,
  the sync bridge, the credentials (the adapter profiles' key variables and headers, R6), usage
  (:mod:`rcp_ndcg.inference.transport`);
* the role clients -- the content decisions above the wire: the one base
  :class:`RoleClient` (:mod:`rcp_ndcg.inference.clients._base`) and the role clients
  :class:`EmbeddingClient`, :class:`RerankClient` and :class:`PoolingClient`
  (:mod:`rcp_ndcg.inference.clients`);
* the role endpoint configs -- :class:`EmbeddingEndpoint`, :class:`PoolingEndpoint`, :class:`RerankEndpoint`
  (:mod:`rcp_ndcg.inference.config`);
* :data:`FAKE_SCHEME` -- the offline fakes' URL scheme, and :func:`register_fake_route` for their extra routes
  (:mod:`rcp_ndcg.inference.fake`).
"""

from rcp_ndcg.inference.adapters.base import (
    ADAPTER_ENTRY_POINTS,
    ENGINE_ADAPTER_ROLES,
    ROLES,
    Adapter,
    AdapterRole,
    adapter_roles_of,
    check_engine_api,
    get_adapter,
    known_adapters,
    register_adapter,
)
from rcp_ndcg.inference.adapters.chat import OpenAIChat
from rcp_ndcg.inference.adapters.pooling import VllmPooling
from rcp_ndcg.inference.adapters.rerank import CohereRerankAdapter, RerankAdapter, VoyageRerankAdapter
from rcp_ndcg.inference.clients import EmbeddingClient, PoolingClient, RerankClient, RoleClient
from rcp_ndcg.inference.config import SELF_HOSTED_APIS, EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.inference.fake import FAKE_SCHEME, FakeEndpoint, register_fake_route
from rcp_ndcg.inference.recipes import (
    RECIPE_SCHEMA_VERSIONS,
    available_recipe_ids,
    expand_role_recipe,
    recipe_client_data,
    recipe_role,
    shorthand_config,
)
from rcp_ndcg.inference.transport import AuthProfile, Sender, Transport
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
    "RECIPE_SCHEMA_VERSIONS",
    "ENGINE_ADAPTER_ROLES",
    "AuthProfile",
    "Adapter",
    "AdapterRole",
    "SELF_HOSTED_APIS",
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
    "FakeEndpoint",
    "OpenAIChat",
    "PoolRequest",
    "PoolingClient",
    "PoolingEndpoint",
    "ROLES",
    "RerankAdapter",
    "RerankClient",
    "RerankEndpoint",
    "RerankRequest",
    "RerankResult",
    "Reply",
    "RoleClient",
    "Sender",
    "TokenCount",
    "Transport",
    "Usage",
    "VllmPooling",
    "VoyageRerankAdapter",
    "adapter_roles_of",
    "available_recipe_ids",
    "check_engine_api",
    "expand_role_recipe",
    "get_adapter",
    "known_adapters",
    "l2_normalize",
    "recipe_client_data",
    "recipe_role",
    "register_adapter",
    "register_fake_route",
    "shorthand_config",
]
