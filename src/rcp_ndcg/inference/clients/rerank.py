"""The rerank role client: one query's whole candidate set per request, checkpointed per query.

:class:`RerankClient` turns the config's knobs into requests over a :class:`~rcp_ndcg.inference.transport.Sender`
and hands the answers to :meth:`~rcp_ndcg.inference.adapters.rerank.RerankAdapter.interpret`. It owns the two
decisions every rerank path must make the same way:

* **the query text** -- the config's ``instruction`` mode decides how the instruction reaches the model, and
  one rule covers the served and the hosted path alike: ``fold`` sends ``Task: <instruction>\\nQuery: <text>``
  exactly as today's served path (:meth:`rcp_ndcg_core._records.Query.format_content`), ``field`` sends the
  bare query plus the engine's ``instruction`` request field (served vLLM only), ``none`` sends the bare
  query. This removes the divergence RFC-0001 section 2.1 records, where the served path read a folded query
  and the in-process path the bare one.
* **preparation** -- every input goes through :meth:`RerankClient._prepare`, the one seam where the
  instruction mode and, once wired, the text-budget mechanism apply. Until that mechanism lands the client
  cuts nothing: contents are sent as given, and a config that sets ``max_tokens`` is refused rather than
  silently ignored.

The hosted profiles' request caps, splits and pauses come from the adapter
(:mod:`rcp_ndcg.inference.adapters.rerank`); the client runs ``concurrency`` queries in flight and calls the
``checkpoint`` callable once per query, so a crash costs at most the queries in flight -- the per-query
checkpoint of today's served path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
from typing import TYPE_CHECKING, Any, TypeVar, cast

from rcp_ndcg_core._records import Query, RankingExample
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.inference.adapters.base import get_adapter
from rcp_ndcg.inference.config import RerankEndpoint
from rcp_ndcg.inference.transport import Sender, Transport
from rcp_ndcg.inference.types import Call, EncodeRole, Reply, RerankRequest, RerankResult

if TYPE_CHECKING:
    from rcp_ndcg.inference.adapters.rerank import RerankWire

T = TypeVar("T")
"""The result type of a coroutine the sync bridge runs."""

#: The checkpoint callable: called once per scored query with its id and its scores, aligned to the query's
#: documents in the order they were given. The caller owns the file (today's served path writes one record per
#: query and fsyncs it); the client only calls at the right times.
Checkpoint = Callable[[str, tuple[float, ...]], None]


class RerankClient:
    """One :class:`~rcp_ndcg.inference.config.RerankEndpoint`'s reranking, over one sender.

    The client is the synchronous API the retrieval steps call (each call runs its requests on one event loop
    through :meth:`Transport.run`, as judging does) with an async core (:meth:`arerank`) beside it.

    Attributes:
        config: The role config, with the hosted profiles' public API root filled in when the config set no
            ``base_url``.
    """

    def __init__(self, config: RerankEndpoint, *, sender: Sender | None = None) -> None:
        """A client for ``config``, sending over ``sender`` (a :class:`~rcp_ndcg.inference.transport.Transport`
        when none is given).

        Args:
            config: The rerank endpoint; a hosted profile (``api: cohere`` or ``api: voyage``) without a
                ``base_url`` uses its public API root, a served one requires the config's.
            sender: What sends the calls; ``None`` builds the endpoint's own transport. A test fake or a
                third-party sender is any :class:`~rcp_ndcg.inference.transport.Sender`.

        Raises:
            ConfigError: The config sets ``max_tokens`` (the text-budget mechanism that applies client-side
                budgets is not wired yet, and a budget is never silently ignored), a hosted profile with
                ``instruction: field`` or ``use_activation`` (neither exists on their wire), or a served
                endpoint without a ``base_url``.
        """
        if config.max_tokens is not None:
            raise ConfigError(
                "max_tokens needs the text-budget mechanism, which is not wired yet",
                hint="leave max_tokens unset; until the mechanism lands the client cuts nothing, and a "
                "budget would otherwise be silently ignored",
            )
        adapter_cls = cast("type[RerankWire]", get_adapter(config.api))
        if config.base_url is None:
            default = adapter_cls.DEFAULT_BASE_URL
            if default is None:
                raise ConfigError(
                    f"the {config.api!r} rerank endpoint needs base_url",
                    hint="a served engine has no public root: set base_url to the engine's URL (e.g. http://127.0.0.1:8000/v1)",
                )
            config = config.model_copy(update={"base_url": default})
        self.config = config
        self._adapter = adapter_cls(config)
        self._sender = sender
        self._transport = Transport(config) if sender is None else None

    # -- the synchronous API -------------------------------------------------
    def rerank(
        self, query: str | Content, documents: Sequence[str | Content], *, instruction: str | None = None
    ) -> RerankResult:
        """Relevance scores for *documents* against *query*, in the order the documents were given.

        Args:
            query: The query, as text or content parts.
            documents: The candidates, as text or content parts; empty documents are sent as they are and
                score whatever the server returns.
            instruction: The task instruction, folded or sent per the config's ``instruction`` mode.

        Returns:
            One relevance score per document, aligned to the input order (never the server's ranking order).

        Raises:
            ConfigError: The config cannot serve this request (see :meth:`__init__`).
            CapabilityError: The endpoint refused the request as too long.
            RequestRejectedError: The endpoint refused this one request.
            ProviderError: The endpoint failed after its retries, or its answer was unusable.
        """
        return self._run(self.arerank(query, documents, instruction=instruction))

    def rerank_many(
        self, examples: Sequence[RankingExample], *, checkpoint: Checkpoint | None = None
    ) -> list[RerankResult]:
        """Score every example, ``concurrency`` queries in flight, and return the results in input order.

        Each query is one request (or one per cap-sized chunk of its candidate set, merged), as in today's
        served path: the engine reuses the query's prefix across the documents, and a listwise model needs the
        whole set together. An example with no documents is checkpointed with no scores and makes no request,
        exactly as the served path does. The query is sent through the config's instruction mode, so the
        example's raw query and instruction go in -- never the already-folded
        :meth:`~rcp_ndcg_core._records.Query.format_content` text, which would fold twice.

        Args:
            examples: The ranking examples to score; documents must be populated (``docs`` or ``contents``).
            checkpoint: Called once per scored query with its id and its scores (aligned to the example's
                ``doc_ids``), as each query finishes -- the per-query checkpoint of today's served path: write
                the record and flush here, and a crash costs at most the queries in flight.

        Returns:
            One :class:`~rcp_ndcg.inference.types.RerankResult` per example, in the input order.
        """
        return self._run(self.arerank_many(examples, checkpoint=checkpoint))

    def close(self) -> None:
        """Close the transport's client and pool, when the client built the transport; safe to call twice."""
        if self._transport is not None:
            self._transport.aclose()

    # -- the async core ------------------------------------------------------
    async def arerank(
        self, query: str | Content, documents: Sequence[str | Content], *, instruction: str | None = None
    ) -> RerankResult:
        """The async half of :meth:`rerank`: prepare, send, and read the scores back aligned."""
        prepared_query = self._prepare([query], EncodeRole.QUERY, instruction=instruction)[0]
        prepared_documents = self._prepare(documents, EncodeRole.DOCUMENT)
        if not prepared_documents:
            return RerankResult(scores=())  # an empty candidate set is not a request (as on the served path)
        request = RerankRequest(
            query=prepared_query,
            documents=prepared_documents,
            instruction=instruction if self.config.instruction == "field" else None,
        )
        replies = await self._send(self._adapter.calls(request, model=self.config.model))
        return self._adapter.interpret(request, replies)

    async def arerank_many(
        self, examples: Sequence[RankingExample], *, checkpoint: Checkpoint | None = None
    ) -> list[RerankResult]:
        """The async half of :meth:`rerank_many`: one task per query under a concurrency semaphore, the
        checkpoint called from the event loop as each query lands."""
        limit = asyncio.Semaphore(self.config.concurrency)
        results: list[RerankResult | None] = [None] * len(examples)

        async def score(index: int, example: RankingExample) -> None:
            async with limit:
                result = await self.arerank(example.as_content, example.doc_contents, instruction=example.instruction)
            results[index] = result
            if checkpoint is not None:
                checkpoint(str(example.id), result.scores)

        await asyncio.gather(*(score(index, example) for index, example in enumerate(examples)))
        return [result for result in results if result is not None]

    # -- preparation and sending ---------------------------------------------
    def _prepare(
        self,
        contents: Sequence[str | Content],
        role: EncodeRole,
        *,
        instruction: str | None = None,
    ) -> tuple[Content, ...]:
        """Every input passes through here, and nothing else changes it: the one seam for prompts, the
        instruction mode and, once wired, the text-budget mechanism.

        Today: text is materialised to content parts, and the query's instruction is folded into its text for
        ``instruction: fold`` -- exactly the served path's
        :meth:`~rcp_ndcg_core._records.Query.format_content` render, so the served and the hosted path send
        the same query text. Nothing is cut; a config with ``max_tokens`` is refused in :meth:`__init__`
        rather than silently ignored.

        Args:
            contents: The texts or content parts as the caller gave them.
            role: Which side of the pair this batch is (the instruction applies to the query only).
            instruction: The task instruction, when the caller has one.

        Returns:
            One :class:`~rcp_ndcg_core.content.Content` per input, in order, unchanged unless the instruction
            mode folds it.
        """
        prepared = [content if isinstance(content, Content) else Content.from_text(content) for content in contents]
        if role is not EncodeRole.QUERY or self.config.instruction != "fold" or not instruction:
            return tuple(prepared)
        return tuple(
            Query(query_id="", query=content.text, instruction=instruction, content=content).format_content()
            for content in prepared
        )

    async def _send(self, calls: Sequence[Call]) -> list[Reply]:
        """Send one request's calls and return their replies, one per call, in order.

        A profile with a pause (Voyage) sends its calls one at a time, sleeping before each as today's
        ``VoyageRerank`` does; otherwise the calls go in one ``send``, which the transport routes to one
        replica without interleaving.
        """
        sender = self._transport if self._transport is not None else self._sender
        if sender is None:  # pragma: no cover - __init__ always leaves one of the two set
            raise RuntimeError("rerank client has neither a transport nor a sender")
        pause = self._adapter.PAUSE_S
        if not pause:
            return await sender.send(calls)
        replies: list[Reply] = []
        for call in calls:
            await asyncio.sleep(pause)
            replies.extend(await sender.send([call]))
        return replies

    def _run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        """Run one synchronous call's coroutine on one event loop: the transport's own when the client built
        it (the pool is bound to it), a fresh one for an injected sender."""
        if self._transport is not None:
            return self._transport.run(coroutine)
        if isinstance(self._sender, Transport):
            return self._sender.run(coroutine)
        return asyncio.run(coroutine)


__all__ = ["Checkpoint", "RerankClient"]
