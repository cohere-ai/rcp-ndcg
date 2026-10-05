"""The rerank role client: one query's whole candidate set per request, checkpointed per query.

:class:`RerankClient` turns the config's knobs into requests over a :class:`~rcp_ndcg.inference.transport.Sender`
and hands the answers to :meth:`~rcp_ndcg.inference.adapters.rerank.RerankAdapter.interpret`. It owns the two
decisions every rerank path must make the same way:

* **the query text** -- the config's ``instruction`` mode decides how the instruction reaches the model, and
  one rule covers the served and the hosted path alike: ``fold`` sends ``Task: <instruction>\\nQuery: <text>``
  exactly as today's served path (:meth:`rcp_ndcg_core._records.Query.format_content`), ``field`` sends the
  bare query plus the engine's ``instruction`` request field (served vLLM only), ``none`` sends the bare
  query.
* **the pair budget** -- every request is fitted through :func:`rcp_ndcg.data.preprocess.fit` as the
  ``pair`` shape: the (query, document) pairs are cut span by span within the declared budget (the query to
  ``query_max_tokens`` when it is set), the template's fixed segments re-attached around the cuts (the
  anchors a pointwise reranker reads its score from always survive), every cut recorded in the census under
  ``text_budget``, and a chunked document sent as one request per chunk with the chunks' scores pooled back
  onto the document by ``max`` (:func:`rcp_ndcg.data.preprocess.max_pool_scores_by_document`). The wire
  carries the cut spans -- the engine renders the template itself -- and no ``truncate_prompt_tokens``,
  ``max_tokens_per_query`` or ``max_tokens_per_doc`` is ever sent: the client cut already, so there is
  nothing left for the engine to truncate. A hosted profile that declares only ``max_tokens`` (no
  tokenizer) sends content uncut and records the vendor's documented limit.

The hosted profiles' request caps, splits, pauses and credential facts come from the adapter
(:mod:`rcp_ndcg.inference.adapters.rerank`); the client runs ``concurrency`` queries in flight under one
:class:`asyncio.TaskGroup` (a failing query cancels its siblings, R7) and calls the ``checkpoint`` callable
once per query, so a crash costs at most the queries in flight -- the per-query checkpoint of today's
served path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from rcp_ndcg_core._records import Query, RankingExample
from rcp_ndcg_core.content import Content

from rcp_ndcg.data.preprocess import FitResult, TextTruncationCensus, max_pool_scores_by_document
from rcp_ndcg.inference.clients._base import RoleClient
from rcp_ndcg.inference.config import RerankEndpoint
from rcp_ndcg.inference.transport import Sender
from rcp_ndcg.inference.types import Call, EncodeRole, RerankRequest, RerankResult

if TYPE_CHECKING:
    from rcp_ndcg.inference.adapters.rerank import RerankWire


#: The checkpoint callable: called once per scored query with its id and its scores, aligned to the query's
#: documents in the order they were given. The caller owns the file (today's served path writes one record per
#: query and fsyncs it); the client only calls at the right times.
Checkpoint = Callable[[str, tuple[float, ...]], None]


class RerankClient(RoleClient):
    """One :class:`~rcp_ndcg.inference.config.RerankEndpoint`'s reranking, over one sender.

    The client is the synchronous API the retrieval steps call (each call runs its requests on one event
    loop through the sender's sync bridge, as judging does) with an async core (:meth:`arerank`) beside it.

    Attributes:
        config: The role config, as it was given (a hosted profile's public root fills :attr:`endpoint`).
    """

    ROLE = "rerank"

    def __init__(
        self, config: RerankEndpoint, *, sender: Sender | None = None, census: TextTruncationCensus | None = None
    ) -> None:
        """A client for ``config``, sending over ``sender`` (a :class:`~rcp_ndcg.inference.transport.Transport`
        when none is given).

        Args:
            config: The rerank endpoint; a hosted profile (``api: cohere`` or ``api: voyage``) without a
                ``base_url`` uses its public API root, a served one requires the config's.
            sender: What sends the calls; ``None`` builds the endpoint's own transport. A test fake or a
                third-party sender is any :class:`~rcp_ndcg.inference.transport.Sender` with a sync bridge.
            census: Where the text-budget cuts are recorded; ``None`` gives the client a fresh in-memory
                census (:attr:`census`).

        Raises:
            ConfigError: ``api`` names no registered adapter of the rerank role (the hint lists that role's
                names), a hosted profile with ``instruction: field`` or ``use_activation`` (neither exists on
                their wire), or a served endpoint without a ``base_url``.
        """
        super().__init__(config, sender=sender, census=census)
        self._adapter: RerankWire = self._adapter_cls(self.endpoint)

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
            TextBudgetExceededError: ``on_overflow: fail`` and a pair over budget, or a query that fills the
                budget with no split declared.
            CapabilityError: The endpoint refused the request as too long.
            RequestRejectedError: The endpoint refused this one request.
            ProviderError: The endpoint failed after its retries, or its answer was unusable.
        """
        return self._run(self.arerank(query, documents, instruction=instruction))

    def rerank_many(
        self, examples: Sequence[RankingExample], *, checkpoint: Checkpoint | None = None
    ) -> list[RerankResult]:
        """Score every example, ``concurrency`` queries in flight, and return the results in input order.

        Each query is one request (or one per chunk of a budget-split document, scores pooled by ``max``),
        as in today's served path: the engine reuses the query's prefix across the documents, and a listwise
        model needs the whole set together. An example with no documents is checkpointed with no scores and
        makes no request, exactly as the served path does. The query is sent through the config's
        instruction mode, so the example's raw query and instruction go in -- never the already-folded
        :meth:`~rcp_ndcg_core._records.Query.format_content` text, which would fold twice.

        Args:
            examples: The ranking examples to score; documents must be populated (``docs`` or ``contents``).
            checkpoint: Called once per scored query with its id and its (pooled) scores (aligned to the
                example's ``doc_ids``), as each query finishes -- the per-query checkpoint of today's served
                path: write the record and flush here, and a crash costs at most the queries in flight. A
                failing query cancels its siblings, and no checkpoint lands after the failure (R7).

        Returns:
            One :class:`~rcp_ndcg.inference.types.RerankResult` per example, in the input order.
        """
        return self._run(self.arerank_many(examples, checkpoint=checkpoint))

    # -- the async core ------------------------------------------------------
    async def arerank(
        self, query: str | Content, documents: Sequence[str | Content], *, instruction: str | None = None
    ) -> RerankResult:
        """The async half of :meth:`rerank`: prepare, fit to the budget, send, and read the scores back
        aligned to the documents."""
        prepared_query = self._prepare([query], EncodeRole.QUERY, instruction=instruction)[0]
        prepared_documents = self._prepare(documents, EncodeRole.DOCUMENT)
        if not prepared_documents:
            return RerankResult(scores=())  # an empty candidate set is not a request (as on the served path)
        wire_query, wire_documents, fitted = self._fit_pair(prepared_query, prepared_documents)
        request = RerankRequest(
            query=wire_query,
            documents=tuple(wire_documents),
            instruction=instruction if self.config.instruction == "field" else None,
        )
        replies = await self._send(self._adapter.calls(request, model=self.config.model))
        return self._pooled(fitted, self._adapter.interpret(request, replies), len(prepared_documents))

    async def arerank_many(
        self, examples: Sequence[RankingExample], *, checkpoint: Checkpoint | None = None
    ) -> list[RerankResult]:
        """The async half of :meth:`rerank_many`: one task per query under a concurrency semaphore, in one
        :class:`asyncio.TaskGroup` (a failing query cancels its siblings and no checkpoint lands after the
        failure, R7); the checkpoint called from the event loop as each query lands."""
        limit = asyncio.Semaphore(self.config.concurrency)
        results: list[RerankResult | None] = [None] * len(examples)

        async def score(index: int, example: RankingExample) -> None:
            async with limit:
                result = await self.arerank(example.as_content, example.doc_contents, instruction=example.instruction)
            results[index] = result
            if checkpoint is not None:
                checkpoint(str(example.id), result.scores)

        await RoleClient.gather([score(index, example) for index, example in enumerate(examples)])
        return [result for result in results if result is not None]

    # -- preparation and fitting ----------------------------------------------
    def _prepare(
        self,
        contents: Sequence[str | Content],
        role: EncodeRole,
        *,
        instruction: str | None = None,
    ) -> tuple[Content, ...]:
        """The one seam every input passes through: text materialised to content parts, and the query's
        instruction folded into its text for ``instruction: fold`` -- exactly the served path's
        :meth:`~rcp_ndcg_core._records.Query.format_content` render, so the served and the hosted path send
        the same query text. The budget fit happens on the pairs, in :meth:`_fit_pair`.

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

    def _fit_pair(self, query: Content, documents: Sequence[Content]) -> tuple[Content, list[Content], FitResult]:
        """The query and its candidates as the wire carries them, fitted into the pair budget.

        With a budget declared, :meth:`RoleClient._fit` runs the shared mechanism over the ``(query,
        document)`` pairs (shape ``pair``): the query span is settled first (to ``query_max_tokens``), each
        document gets what remains, a chunked document comes back as one output per chunk (``<id>#<k>``)
        with the full template around it. The wire takes the cut spans (the engine renders the template
        itself); the chunks' ``max`` pooling is the caller's, through the fit result.

        Returns:
            ``(wire_query, wire_documents, fitted)``: the query content (cut) and the document contents
            (one per fit output, in fit's order), and the fit result -- its ``ids`` align to the wire
            documents, its ``chunk_mapping`` carries each chunk back to its input.

        Raises:
            TextBudgetExceededError: the declared overflow policy refuses to shorten a pair.
        """
        if self._budget is None:
            return (
                query,
                list(documents),
                FitResult(
                    shape="pair", texts=(), contents=(), ids=tuple(str(index) for index in range(len(documents)))
                ),
            )  # noqa: E501
        pairs = [(query.text, document.text) for document in documents]
        result = self._fit(pairs, "pair", media_tokens=self._media_tokens(documents))
        contents = [pair if isinstance(pair, tuple) else (pair, "") for pair in result.contents]
        # One query's pairs share one query text, so every output's query span is the same cut. A chunked
        # document is one wire document per chunk, each carrying its input's media parts beside the piece.
        mapping = result.chunk_mapping or {}
        origin = [int(mapping.get(out_id, out_id)) for out_id in result.ids]
        wire_query = self._with_text(query, contents[0][0])
        wire_documents = [
            self._with_text(documents[source], document_text)
            for source, (_, document_text) in zip(origin, contents, strict=True)
        ]
        return wire_query, wire_documents, result

    def _pooled(self, fitted: FitResult, result: RerankResult, documents: int) -> RerankResult:
        """The scores of one query, pooled back onto its documents when the budget chunked them.

        Without a chunk mapping the scores pass through. With one, the outputs' scores (one per chunk, in
        fit's order, aligned to :attr:`FitResult.ids`) pool onto their documents by
        :func:`rcp_ndcg.data.preprocess.max_pool_scores_by_document` (``aggregation: max``), and the result
        is realigned to the original document order.
        """
        if fitted.chunk_mapping is None:
            return result
        per_chunk = dict(zip(fitted.ids, result.scores, strict=True))
        pooled = max_pool_scores_by_document(per_chunk, fitted.chunk_mapping)
        return RerankResult(scores=tuple(pooled[str(index)] for index in range(documents)))

    async def _send(self, calls: Sequence[Call]) -> list[Any]:
        """Send one request's calls and return their replies, one per call, in order.

        A profile with a pause (Voyage) sends its calls one at a time, sleeping before each as today's
        ``VoyageRerank`` does; otherwise the calls go in one ``send``, which the transport routes to one
        replica without interleaving.
        """
        sender = self._sender
        pause = getattr(self._adapter, "PAUSE_S", 0.0)
        if not pause:
            return await sender.send(calls)
        replies: list[Any] = []
        for call in calls:
            await asyncio.sleep(pause)
            replies.extend(await sender.send([call]))
        return replies


__all__ = ["Checkpoint", "RerankClient"]
