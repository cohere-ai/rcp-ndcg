"""The adapter seam: the :class:`Adapter` protocol, its role-scoped registry and the entry-point group (C2).

The shipped adapters register at import: the embed role's OpenAI shape and its hosted profiles
(:mod:`rcp_ndcg.inference.adapters.embeddings`) and the rerank role's Cohere-shaped wire and its hosted
profiles (:mod:`rcp_ndcg.inference.adapters.rerank`). The rest arrive with their lanes (the judge's
``openai_chat`` with lane L2; the multi-vector ``vllm_pooling`` with the pooling lane), each under its role's
namespace.
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
from rcp_ndcg.inference.adapters.rerank import (
    CohereRerankAdapter,
    RerankAdapter,
    RerankWire,
    VoyageRerankAdapter,
)

__all__ = [
    "ADAPTER_ENTRY_POINTS",
    "ROLES",
    "Adapter",
    "AdapterRole",
    "CohereRerankAdapter",
    "RerankAdapter",
    "RerankWire",
    "VoyageRerankAdapter",
    "get_adapter",
    "known_adapters",
    "register_adapter",
]
