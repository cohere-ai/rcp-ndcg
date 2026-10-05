"""The pooling role client: late-interaction (multi-vector) encoding with every content decision applied.

A :class:`PoolingClient` turns :class:`~rcp_ndcg_core.content.Content` into the ragged
:class:`~rcp_ndcg.inference.types.Embeddings` of a multi-vector endpoint, one
:class:`~rcp_ndcg.inference.types.PoolRequest` per ``batch_size`` items sent through its
:class:`~rcp_ndcg.inference.transport.Sender`:

* **Prompts per role** -- the config's ``query_prompt``/``doc_prompt`` is prepended to the side it names
  (:meth:`_prepare`, the one seam where prompts, the instruction mode and, once wired, the text budget
  apply). Until that mechanism exists the client cuts nothing and sends every content as given: a config
  that sets ``max_tokens`` is refused at construction rather than silently ignored.
* **Wire precision** -- the config's ``embed_dtype`` (``float16`` by the owner's decision, ``float32``
  opt-in) travels on every request and survives to the result: the ragged buffer keeps its transfer dtype
  end to end, so an index built from float16 vectors stores float16 (2 bytes per token vector, against 4
  for float32). MaxSim computes in float32 either way.
* **Normalisation** -- when ``normalize`` is set (the default), every token vector is L2-normalised in
  float32 and stored back in the transfer dtype (the engine's own token_embed pooling already normalises;
  normalising twice is harmless).
* **Concurrency** -- at most ``concurrency`` batch requests in flight, reassembled in input order; a
  transport bounds the same number again across everything it sends.

The sync :meth:`encode` runs the async path on the sender's own event-loop bridge when it has one (a
:class:`~rcp_ndcg.inference.transport.Transport`'s ``run``, which also keeps the pool on one loop) and on a
fresh loop otherwise, the way judging already bridges to asyncio.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import cast

from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import ConfigError, ProviderError
from rcp_ndcg.inference.adapters.base import Adapter, get_adapter
from rcp_ndcg.inference.config import PoolingEndpoint
from rcp_ndcg.inference.transport import Sender, Transport
from rcp_ndcg.inference.types import Embeddings, EncodeRole, PoolRequest


class PoolingClient:
    """A served late-interaction encoder: content in, ragged token vectors out.

    The role client of a :class:`~rcp_ndcg.inference.config.PoolingEndpoint`: it applies the role's prompt
    (:meth:`_prepare`), splits the batch, sends ``batch_size``-sized pooling requests ``concurrency`` at a
    time through the wire adapter the config's ``api`` names, and reassembles the ragged vectors in input
    order.

    Args:
        config: The pooling endpoint: where the model is served, its wire adapter (``vllm_pooling``), the
            prompts, ``embed_dtype``, ``dim`` and the batching.
        sender: What sends the calls. ``None`` builds a
            :class:`~rcp_ndcg.inference.transport.Transport` for the endpoint (which the transport work
            implements; until then a bare ``Transport`` raises ``NotImplementedError`` naming it, so offline
            callers pass their own sender).

    Raises:
        ConfigError: The config sets ``max_tokens``: cutting is the text-budget mechanism's job, which is
            not wired yet, and a budget silently ignored would change the vectors.
    """

    def __init__(self, config: PoolingEndpoint, *, sender: Sender | None = None) -> None:
        if config.max_tokens is not None:
            raise ConfigError("max_tokens needs the text-budget mechanism, which is not wired yet")
        self._config = config
        self._adapter: Adapter[PoolRequest, Embeddings] = get_adapter(config.api)()
        self._sender: Sender = sender if sender is not None else Transport(config)

    @property
    def config(self) -> PoolingEndpoint:
        """The endpoint the client encodes against."""
        return self._config

    # -- encoding ----------------------------------------------------------
    def encode(
        self,
        contents: Sequence[Content],
        role: EncodeRole,
        *,
        batch_size: int | None = None,
    ) -> Embeddings:
        """Ragged token vectors for ``contents``, in order (the synchronous form).

        Runs :meth:`aencode` to completion: on the sender's own sync bridge when it has one (a
        :class:`~rcp_ndcg.inference.transport.Transport`, whose pool stays on one loop), else on a fresh
        event loop, the way judging bridges to asyncio today.

        Args:
            contents: The queries or documents as content parts, in order.
            role: Which side of the retrieval pair the batch is; the prompts depend on it.
            batch_size: Items per pooling request; the config's ``batch_size`` when ``None``.

        Returns:
            Ragged embeddings in the transfer dtype (one slice of vectors per item), or single-vector
            embeddings when the served task pooled instead and the reply reported no usage.
        """
        bridge = getattr(self._sender, "run", None)
        if callable(bridge):
            # A transport's sync bridge returns the coroutine's result (Transport.run); a bare Sender has
            # none and a fresh loop bridges instead.
            return cast(Embeddings, bridge(self.aencode(contents, role, batch_size=batch_size)))
        return asyncio.run(self.aencode(contents, role, batch_size=batch_size))

    async def aencode(
        self,
        contents: Sequence[Content],
        role: EncodeRole,
        *,
        batch_size: int | None = None,
    ) -> Embeddings:
        """Ragged token vectors for ``contents``, in order (the asynchronous form).

        Args:
            contents: The queries or documents as content parts, in order.
            role: Which side of the retrieval pair the batch is; the prompts depend on it.
            batch_size: Items per pooling request; the config's ``batch_size`` when ``None``.

        Returns:
            Ragged embeddings in the transfer dtype, in input order. An empty batch is the zero-item value
            and sends nothing. A served task that pooled instead of token-embedding shows up as one vector
            per item, which the adapter refuses when the reply reports usage.
        """
        prepared = self._prepare(contents, role)
        if not prepared:
            return Embeddings.empty(0, multi_vector=True, dtype=self._config.embed_dtype)
        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        size = batch_size or self._config.batch_size
        batches = [prepared[start : start + size] for start in range(0, len(prepared), size)]
        gate = asyncio.Semaphore(self._config.concurrency)

        async def one(batch: list[Content]) -> Embeddings:
            async with gate:
                return await self._encode_batch(batch, role)

        chunks = await asyncio.gather(*(one(batch) for batch in batches))
        return _concat_all(chunks)

    def _prepare(self, contents: Sequence[Content], role: EncodeRole) -> list[Content]:
        """The contents as they are sent: the role's prompt prepended, everything else untouched.

        This is the one place a content decision applies -- the role's prompt today; the text budget, once
        the mechanism exists, cuts the content spans here and re-attaches the template. The client cuts
        nothing and transforms nothing else: a model-side change without a config field is a silent change
        to the vectors.
        """
        prefix = self._config.query_prompt if role is EncodeRole.QUERY else self._config.doc_prompt
        if not prefix:
            return list(contents)
        return [content.with_text_prefix(prefix) for content in contents]

    async def _encode_batch(self, contents: Sequence[Content], role: EncodeRole) -> Embeddings:
        """One batch: a pooling request through the adapter and the sender, checked for alignment."""
        request = PoolRequest(
            contents=tuple(contents),
            role=role,
            embed_dtype=self._config.embed_dtype,
            dim=self._config.dim,
        )
        calls = self._adapter.calls(request, model=self._config.model)
        replies = await self._sender.send(calls)
        embeddings = self._adapter.interpret(request, replies)
        if embeddings.num_items != len(contents):
            raise ProviderError(
                f"the pooling endpoint returned {embeddings.num_items} item(s) for {len(contents)} input(s); "
                "refusing to return misaligned vectors"
            )
        if self._config.normalize:
            return embeddings.l2_normalized()
        return embeddings


def _concat_all(chunks: Sequence[Embeddings]) -> Embeddings:
    """Append every chunk's items in order (one chunk needs no copy)."""
    result = chunks[0]
    for chunk in chunks[1:]:
        result = result.concat(chunk)
    return result


__all__ = ["PoolingClient"]
