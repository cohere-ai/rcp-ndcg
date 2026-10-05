"""The role clients: one client per role over the transport, each the synchronous API its steps call.

A client holds its role config (with a hosted profile's public API root filled in), the adapter the config's
``api`` selected, and one sender (its endpoint's transport when none is given). It owns the decisions every
path of its role must make the same way -- for the reranker, the query text and the preparation seam -- and
leaves the wire to the adapter and the sending to the transport.
"""

from rcp_ndcg.inference.clients.rerank import Checkpoint, RerankClient

__all__ = ["Checkpoint", "RerankClient"]
