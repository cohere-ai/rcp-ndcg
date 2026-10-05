"""The adapter seam: the :class:`Adapter` protocol, its registry, the entry-point group (C2) and the shipped
adapters, which register at import of this package (the embedding, rerank and multi-vector pooling wires)."""

from rcp_ndcg.inference.adapters.base import (
    ADAPTER_ENTRY_POINTS,
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
