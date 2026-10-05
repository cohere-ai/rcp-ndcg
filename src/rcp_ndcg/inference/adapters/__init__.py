"""The adapter seam: the :class:`Adapter` protocol, its registry and the entry-point group (C2)."""

from rcp_ndcg.inference.adapters.base import (
    ADAPTER_ENTRY_POINTS,
    Adapter,
    AdapterRole,
    get_adapter,
    known_adapters,
    register_adapter,
)

__all__ = [
    "ADAPTER_ENTRY_POINTS",
    "Adapter",
    "AdapterRole",
    "get_adapter",
    "known_adapters",
    "register_adapter",
]
