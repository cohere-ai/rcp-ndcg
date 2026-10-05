"""The adapter seam: the :class:`Adapter` protocol, its registry, the entry-point group (C2) and the shipped
adapters, which register at import of this package."""

from rcp_ndcg.inference.adapters.base import (
    ADAPTER_ENTRY_POINTS,
    Adapter,
    AdapterRole,
    get_adapter,
    known_adapters,
    register_adapter,
)
from rcp_ndcg.inference.adapters.pooling import VllmPooling

__all__ = [
    "ADAPTER_ENTRY_POINTS",
    "Adapter",
    "AdapterRole",
    "VllmPooling",
    "get_adapter",
    "known_adapters",
    "register_adapter",
]
