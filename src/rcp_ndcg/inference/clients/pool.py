"""The pooling role client: late-interaction (multi-vector) encoding with every content decision applied.

A :class:`PoolingClient` turns :class:`~rcp_ndcg_core.content.Content` into the ragged
:class:`~rcp_ndcg.inference.types.Embeddings` of a multi-vector endpoint, one
:class:`~rcp_ndcg.inference.types.PoolRequest` per ``batch_size`` items sent through its
:class:`~rcp_ndcg.inference.transport.Sender`:

* **Prompts per role** -- the config's ``query_prompt``/``doc_prompt`` is prepended to the side it names;
* **the text budget** -- when the config declares one (``max_tokens``), every item's text is fitted through
  :func:`rcp_ndcg.data.preprocess.fit` (the side's shape): only content spans cut, the template re-attached
  with the anchor kept, every cut recorded in the census under ``text_budget``. A hosted profile with only
  ``max_tokens`` sends content uncut (the vendor path). ``on_overflow: chunk`` is refused: chunked documents
  pool *scores* by maximum, and token vectors are not scores -- a late-interaction document's chunking
  happens at the corpus layer (the retrieval index keeps one slice per chunk). Media items are prepared
  with the request (``prepare_request``), their tokens counted and reserved whole beside the fitted text;
* **Wire precision** -- the config's ``embed_dtype`` (``float16`` by the owner's decision, ``float32``
  opt-in) travels on every request and survives to the result: the ragged buffer keeps its transfer dtype
  end to end, so an index built from float16 vectors stores float16 (2 bytes per token vector, against 4
  for float32). MaxSim computes in float32 either way.
* **Normalisation** -- when ``normalize`` is set (the default), every token vector is L2-normalised in
  float32 and stored back in the transfer dtype (the engine's own token_embed pooling already normalises;
  normalising twice is harmless).
* **Concurrency** -- at most ``concurrency`` batch requests in flight under one :class:`asyncio.TaskGroup`
  (a failing request cancels its siblings, R7), reassembled in input order.

The sync :meth:`encode` runs the async path on the sender's sync bridge (the base's one rule: a
:class:`~rcp_ndcg.inference.transport.Transport`'s ``run``, or the sender's own).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import numpy as np
from rcp_ndcg_core.content import Content

from rcp_ndcg.data.prepare import MediaCensus
from rcp_ndcg.data.preprocess import TextTruncationCensus
from rcp_ndcg.errors import ConfigError, ProviderError
from rcp_ndcg.inference.adapters.base import Adapter
from rcp_ndcg.inference.clients._base import PreparedItems, RoleClient
from rcp_ndcg.inference.config import PoolingEndpoint
from rcp_ndcg.inference.transport import Sender
from rcp_ndcg.inference.types import Call, Embeddings, EncodeRole, PoolRequest, Reply, TokenCount


class PoolingClient(RoleClient):
    """A served late-interaction encoder: content in, ragged token vectors out.

    The role client of a :class:`~rcp_ndcg.inference.config.PoolingEndpoint`: it applies the role's prompt
    and the text budget (:meth:`_prepare`), splits the batch, sends ``batch_size``-sized pooling requests
    ``concurrency`` at a time through the wire adapter the config's ``api`` names, and reassembles the
    ragged vectors in input order.

    Args:
        config: The pooling endpoint: where the model is served, its wire adapter (``vllm_pooling``), the
            prompts, ``embed_dtype``, ``dim``, the text budget and the batching.
        sender: What sends the calls. ``None`` builds a :class:`~rcp_ndcg.inference.transport.Transport`
            for the endpoint; anything else must provide the sync bridge (``run``).
        census: Where the text-budget cuts are recorded; ``None`` gives the client a fresh in-memory census.

    Raises:
        ConfigError: ``dim`` is not set (the base64 frame of ``/pooling`` is flat and carries no shape; a
            refusal at construction keeps the GPU idle-time free, R13), ``on_overflow: chunk`` is declared
            (vector roles do not pool chunks), ``request_shape`` is declared but the wire sends text,
            ``batch_size < 1``, or ``api`` names no adapter of the multi_vector role.
    """

    ROLE = "multi_vector"

    def __init__(
        self,
        config: PoolingEndpoint,
        *,
        sender: Sender | None = None,
        census: TextTruncationCensus | None = None,
        media_census: MediaCensus | None = None,
    ) -> None:
        if config.dim is None:
            raise ConfigError(
                "the pooling endpoint's encoding needs dim: the base64 frame of /pooling is flat and carries "
                "no shape, so the config's dim rebuilds (tokens, dim) client-side",
                hint="set dim to the checkpoint's token-vector width (a ColBERT-style checkpoint projects to "
                "a fixed width, e.g. 128); the bytes encoding carries its shape, but the adapter asks for "
                "base64",
            )
        if config.on_overflow == "chunk":
            raise ConfigError(
                "on_overflow 'chunk' pools scores by max, and token vectors have none to pool: the declared "
                "aggregation would be reinterpreted, so it is refused instead",
                hint="use on_overflow: cut (the content is cut to the budget), or chunk the corpus at load "
                "(the retrieval index keeps one slice per chunk)",
            )
        if config.request_shape != "text":
            raise ConfigError(
                f"request_shape {config.request_shape!r} is declared, but this wire sends rendered text",
                hint="the adapters implement text today; drop request_shape (the default) until the "
                "messages and token_ids routes land",
            )
        super().__init__(config, sender=sender, census=census, media_census=media_census)
        self._adapter: Adapter[PoolRequest, Embeddings] = self._adapter_cls()

    # -- encoding ----------------------------------------------------------
    def encode(
        self,
        contents: Sequence[Content],
        role: EncodeRole,
        *,
        batch_size: int | None = None,
    ) -> Embeddings:
        """Ragged token vectors for ``contents``, in order (the synchronous form).

        Runs :meth:`aencode` to completion on the sender's sync bridge (a transport's ``run`` keeps the
        pool on one loop; an injected sender's own ``run`` is used).

        Args:
            contents: The queries or documents as content parts, in order.
            role: Which side of the retrieval pair the batch is; the prompts depend on it.
            batch_size: Items per pooling request; the config's ``batch_size`` when ``None``.

        Returns:
            Ragged embeddings in the transfer dtype (one slice of vectors per item), or single-vector
            embeddings when the served task pooled instead and the reply reported no usage.
        """
        return self._run(self.aencode(contents, role, batch_size=batch_size))

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
        if not prepared.items:
            if not contents:
                return Embeddings.empty(0, multi_vector=True, dtype=self.config.embed_dtype)
            # Every input was omitted (empty_doc: omit_zero): no request goes out; the result is one empty
            # slice per input -- the MaxSim score an omitted document contributes.
            return Embeddings.ragged(
                [np.zeros((0, 0), dtype=self.config.embed_dtype)] * len(contents),
                dtype=self.config.embed_dtype,
            )
        if batch_size is not None and batch_size < 1:
            raise ConfigError(f"batch_size must be at least 1, got {batch_size}")
        size = batch_size or self.config.batch_size
        max_batch = getattr(self._adapter_cls, "MAX_BATCH", None)
        if max_batch is not None and size > max_batch:
            raise ConfigError(
                f"the {self.config.api} pooling API takes at most {max_batch} items per request; batch_size is {size}",
                hint=f"set batch_size to {max_batch} or less, or leave it unset",
            )
        batches = [prepared.items[start : start + size] for start in range(0, len(prepared.items), size)]
        gate = asyncio.Semaphore(self.config.concurrency)

        async def one(batch: list[Content]) -> Embeddings:
            async with gate:
                return await self._encode_batch(batch, role)

        chunks = await RoleClient.gather([one(list(batch)) for batch in batches])
        if not prepared.omitted:
            return _concat_all(chunks)
        # Some inputs were omitted (empty_doc: omit_zero): rebuild the buffer with an empty slice (a zero
        # MaxSim score) at each omission, the gathered slices in their places, in input order.
        per_item: list[np.ndarray] = []
        for chunk in chunks:
            offsets = chunk.offsets
            assert offsets is not None
            per_item.extend(chunk.vectors[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True))
        sent = iter(per_item)
        slices: list[np.ndarray] = []
        for index in range(len(contents)):
            if index in set(prepared.omitted):
                slices.append(np.zeros((0, 0), dtype=self.config.embed_dtype))
            else:
                slices.append(next(sent))
        return Embeddings.ragged(slices, dtype=self.config.embed_dtype)

    def _prepare(self, contents: Sequence[Content], role: EncodeRole) -> PreparedItems:
        """The contents as they are sent: the role's prompt prepended, the media prepared, then the budget.

        This is the one place a content decision applies -- the role's prompt, the one media preparation
        call (:meth:`RoleClient._prepare_request`), the budget's media fit per wire request with every drop
        recorded (:meth:`RoleClient._fit_media_for_request`), and the text fit: only the text's content span
        is cut (the template re-attached, every cut recorded), and media tokens are reserved whole and
        never cut.
        content span is cut (the template re-attached, every cut recorded), media tokens reserved whole and
        never cut. The client cuts nothing else: a model-side change without a config field is a silent
        change to the vectors.
        """
        prefix = self.config.query_prompt if role is EncodeRole.QUERY else self.config.doc_prompt
        prompted = [content.with_text_prefix(prefix) for content in contents]
        request = self._prepare_request(prompted)
        # The media fit runs per wire request: the pooling wire sends one media item per call, so one
        # item's fit bounds that item's media (drops recorded under the input's position).
        prepared_pairs = [
            self._fit_media_for_request([content], doc_ids=[str(index)])
            for index, content in enumerate(request.contents)
        ]
        fitted = [pair[0][0] for pair in prepared_pairs]
        media_tokens = [pair[1] for pair in prepared_pairs]
        kept, omitted = self._apply_empty_documents(fitted)
        positions = [index for index in range(len(fitted)) if index not in set(omitted)]
        if self._budget is None or not kept:
            texts = [content.text for content in kept]
        else:
            result = self._fit(
                [content.text for content in kept],
                "query" if role is EncodeRole.QUERY else "document",
                media_tokens=[media_tokens[position] for position in positions],
            )
            texts = result.texts
        return PreparedItems(
            items=tuple(self._with_text(content, text) for content, text in zip(kept, texts, strict=True)),
            positions=tuple(positions),
            omitted=tuple(omitted),
        )

    def _probe_calls(self, content: Content) -> Sequence[Call]:
        """The pooling calls one prepared probe item is sent as."""
        request = PoolRequest(
            contents=(content,),
            role=EncodeRole.DOCUMENT,
            embed_dtype=self.config.embed_dtype,
            dim=self.config.dim,
        )
        return self._adapter.calls(request, model=self.config.model)

    async def probe(self) -> Any:
        """The role's startup probe: the transport's replica probe, plus -- when the config declares an
        ``image_processor`` -- the engine media check (one prepared probe image, the engine's prompt-token
        report compared with the counted ones; never silent)."""
        probe: Any = await self._sender.probe()
        await self.check_engine_media()
        return probe

    def _probe_usage(self, reply: Reply) -> TokenCount | None:
        """The reply's prompt-token report (the pooling adapter's, ``None`` when it reported none)."""
        return self._adapter.usage(reply)

    async def _encode_batch(self, contents: Sequence[Content], role: EncodeRole) -> Embeddings:
        """One batch: a pooling request through the adapter and the sender, checked for alignment."""
        request = PoolRequest(
            contents=tuple(contents),
            role=role,
            embed_dtype=self.config.embed_dtype,
            dim=self.config.dim,
        )
        calls = self._adapter.calls(request, model=self.config.model)
        self._gate_media_calls(calls)
        replies = await self._sender.send(calls)
        embeddings = self._adapter.interpret(request, replies)
        if embeddings.num_items != len(contents):
            raise ProviderError(
                f"the pooling endpoint returned {embeddings.num_items} item(s) for {len(contents)} input(s); "
                "refusing to return misaligned vectors"
            )
        if self.config.normalize:
            embeddings = embeddings.l2_normalized()
        return embeddings


def _concat_all(chunks: Sequence[Embeddings]) -> Embeddings:
    """Append every chunk's items in order (one chunk needs no copy)."""
    result = chunks[0]
    for chunk in chunks[1:]:
        result = result.concat(chunk)
    return result


__all__ = ["PoolingClient"]
