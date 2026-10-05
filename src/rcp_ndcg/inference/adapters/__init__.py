"""The adapter seam: the :class:`Adapter` protocol, its registry and the entry-point group (C2).

The shipped adapters register at import: the rerank role's Cohere-shaped wire and its hosted profiles
(:mod:`rcp_ndcg.inference.adapters.rerank`). The rest arrive with their lanes (the judge and the encoders with
the transport work).
"""

from rcp_ndcg.inference.adapters.base import (
    ADAPTER_ENTRY_POINTS,
    Adapter,
    AdapterRole,
    get_adapter,
    known_adapters,
    register_adapter,
)
from rcp_ndcg.inference.adapters.rerank import CohereRerankAdapter, RerankAdapter, VoyageRerankAdapter

__all__ = [
    "ADAPTER_ENTRY_POINTS",
    "Adapter",
    "AdapterRole",
    "CohereRerankAdapter",
    "RerankAdapter",
    "VoyageRerankAdapter",
    "get_adapter",
    "known_adapters",
    "register_adapter",
]
