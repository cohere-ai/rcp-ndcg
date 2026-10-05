"""The role clients: one endpoint config plus a sender, with every content decision applied.

A client owns what the *config* decides about a request's content -- the per-role prompts, the transfer
precision, the normalisation -- and leaves everything around the wire (routing, retries, parking, accounting)
to its :class:`~rcp_ndcg.inference.transport.Sender`. The role clients live one per file and share this
package; a config's ``api`` field picks the wire adapter each client speaks.
"""

from rcp_ndcg.inference.clients.pool import PoolingClient

__all__ = ["PoolingClient"]
