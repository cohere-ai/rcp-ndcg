"""The role clients: the content decisions above the wire -- prompts, normalisation, batching, concurrency.

One module per role, one client per module. A client holds the role's wire adapter (selected from the config's
``api``) and a :class:`~rcp_ndcg.inference.transport.Sender` (a :class:`~rcp_ndcg.inference.transport.Transport`
unless one is supplied), and turns content into the role's request type.

* :class:`rcp_ndcg.inference.clients.embed.EmbeddingClient` -- dense embeddings (``role: "embed"``).
"""

from rcp_ndcg.inference.clients.embed import EmbeddingClient

__all__ = ["EmbeddingClient"]
