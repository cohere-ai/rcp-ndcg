"""The role clients: the content decisions above the wire -- prompts, normalisation, batching, concurrency.

One module per role, one client per module. A client holds its role config (with a hosted profile's public API
root filled in), the wire adapter the config's ``api`` selected, and a
:class:`~rcp_ndcg.inference.transport.Sender` (its endpoint's :class:`~rcp_ndcg.inference.transport.Transport` unless
one is supplied). It owns the decisions every path of its role must make the same way -- the prompts and the query
text, the preparation seam -- and leaves the wire to the adapter and the sending to the transport.

* :class:`rcp_ndcg.inference.clients.embed.EmbeddingClient` -- dense embeddings (``role: "embed"``).
* :class:`rcp_ndcg.inference.clients.rerank.RerankClient` -- reranking (``role: "rerank"``).
* :class:`rcp_ndcg.inference.clients.pool.PoolingClient` -- late interaction (``role: "multi_vector"``).
"""

from rcp_ndcg.inference.clients.embed import EmbeddingClient
from rcp_ndcg.inference.clients.pool import PoolingClient
from rcp_ndcg.inference.clients.rerank import Checkpoint, RerankClient

__all__ = ["Checkpoint", "EmbeddingClient", "PoolingClient", "RerankClient"]
