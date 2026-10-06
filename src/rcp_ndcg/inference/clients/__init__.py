"""The role clients: the content decisions above the wire -- prompts, the text budget, batching, concurrency.

One base every client derives from (:mod:`rcp_ndcg.inference.clients._base`: the adapter lookup within the
client's role, the hosted profile's default base URL, the transport or an injected ``Sender``, the sync
bridge -- a ``Transport``'s ``run`` or the sender's own -- and the lifecycle: ``close()`` sync,
``async aclose()`` awaited, both context managers), one module per role, one client per module. A client
holds its role config (as given; a hosted profile's public root fills :attr:`RoleClient.endpoint`), the
wire adapter the config's ``api`` selected, and a :class:`~rcp_ndcg.inference.transport.Sender`. It owns the
decisions every path of its role must make the same way -- the prompts and the query text, the fit of every
request into the declared text budget, the preparation seam -- and leaves the wire to the adapter, the
sending (and the credentials, R6) to the transport.

* :class:`rcp_ndcg.inference.clients._base.RoleClient` -- the one base (R5, R14, R15).
* :class:`rcp_ndcg.inference.clients.embed.EmbeddingClient` -- dense embeddings (``role: "embed"``).
* :class:`rcp_ndcg.inference.clients.rerank.RerankClient` -- reranking (``role: "rerank"``).
* :class:`rcp_ndcg.inference.clients.pool.PoolingClient` -- late interaction (``role: "multi_vector"``).
"""

from rcp_ndcg.inference.clients._base import RoleClient
from rcp_ndcg.inference.clients.embed import EmbeddingClient
from rcp_ndcg.inference.clients.pool import PoolingClient
from rcp_ndcg.inference.clients.rerank import Checkpoint, RerankClient

__all__ = ["Checkpoint", "EmbeddingClient", "PoolingClient", "RerankClient", "RoleClient"]
