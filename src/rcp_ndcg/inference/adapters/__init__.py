"""The adapter seam: the :class:`Adapter` protocol, its role-scoped registry and the entry-point group (C2).

The shipped adapters register at import of this package, each under its role's namespace: the embed role's OpenAI
shape and its hosted profiles (:mod:`rcp_ndcg.inference.adapters.embeddings`), the rerank role's Cohere-shaped wire
and its hosted profiles (:mod:`rcp_ndcg.inference.adapters.rerank`), and the multi-vector role's ``vllm_pooling``
(:mod:`rcp_ndcg.inference.adapters.pooling`). The judge's ``openai_chat`` arrives with the judge port.
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
from rcp_ndcg.inference.adapters.pooling import VllmPooling
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
    "VllmPooling",
    "get_adapter",
    "known_adapters",
    "register_adapter",
]
