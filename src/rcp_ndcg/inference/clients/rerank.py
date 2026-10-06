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
  ``text_budget``, and a chunked document sent as one candidate-set row per chunk, scored in the query's
  request(s), with the chunks' scores pooled back
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

from rcp_ndcg.data.prepare import MediaCensus
from rcp_ndcg.data.preprocess import (
    CHUNK_ID_SEPARATOR,
    DataError,
    FitResult,
    TextTruncationCensus,
    max_pool_scores_by_document,
    token_prefix,
)
from rcp_ndcg.inference.clients._base import RoleClient
from rcp_ndcg.inference.config import RerankEndpoint
from rcp_ndcg.inference.transport import Sender
from rcp_ndcg.inference.types import Call, EncodeRole, Reply, RerankRequest, RerankResult, TokenCount

if TYPE_CHECKING:
    from rcp_ndcg.inference.adapters.rerank import RerankWire


#: The checkpoint callable: called once per scored query with its id and its scores, aligned to the query's
#: documents in the order they were given. The caller owns the file (today's served path writes one record per
#: query and fsyncs it); the client only calls at the right times.
Checkpoint = Callable[[str, tuple[float, ...]], None]

QUERY_DOC_ID = "<query>"
"""The ``doc_id`` of the census row a shared query's settlement is recorded under (one row per rerank call,
when the query settles below its input: over its declared ``query_max_tokens``, or when it alone fills the
budget) -- the sibling of the fit census' ``BUDGET_DOC_ID``; the pair rows keep their positional ids."""


class RerankClient(RoleClient):
    """One :class:`~rcp_ndcg.inference.config.RerankEndpoint`'s reranking, over one sender.

    The client is the synchronous API the retrieval steps call (each call runs its requests on one event
    loop through the sender's sync bridge, as judging does) with an async core (:meth:`arerank`) beside it.

    Attributes:
        config: The role config, as it was given (a hosted profile's public root fills :attr:`endpoint`).
    """

    ROLE = "rerank"

    def __init__(
        self,
        config: RerankEndpoint,
        *,
        sender: Sender | None = None,
        census: TextTruncationCensus | None = None,
        media_census: MediaCensus | None = None,
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
        super().__init__(config, sender=sender, census=census, media_census=media_census)
        self._adapter: RerankWire = self._adapter_cls(self.endpoint)

    # -- the synchronous API -------------------------------------------------
    def rerank(
        self,
        query: str | Content,
        documents: Sequence[str | Content],
        *,
        instruction: str | None = None,
        query_id: str = "",
    ) -> RerankResult:
        """Relevance scores for *documents* against *query*, in the order the documents were given.

        Args:
            query: The query, as text or content parts.
            documents: The candidates, as text or content parts. An empty document follows the config's
                ``empty_doc`` policy (``send`` sends the empty string as it is and scores whatever the
                server returns; ``omit_zero`` never sends it and scores 0.0; ``send_text`` sends the
                placeholder).
            instruction: The task instruction, folded or sent per the config's ``instruction`` mode.
            query_id: The query's id, for the empty-query refusal's message (``arerank_many`` passes the
                example's id); ``""`` names it ``<unnamed>``.

        Returns:
            One relevance score per document, aligned to the input order (never the server's ranking order).

        Raises:
            ConfigError: The config cannot serve this request (see :meth:`__init__`).
            DataError: the query is empty and the config's ``empty_query: refuse`` (the default) declines
                it, naming the query id.
            TextBudgetExceededError: ``on_overflow: fail`` and a pair over budget, or a query that fills the
                budget with no split declared.
            CapabilityError: The endpoint refused the request as too long.
            RequestRejectedError: The endpoint refused this one request.
            ProviderError: The endpoint failed after its retries, or its answer was unusable.
        """
        return self._run(self.arerank(query, documents, instruction=instruction, query_id=query_id))

    def rerank_many(
        self, examples: Sequence[RankingExample], *, checkpoint: Checkpoint | None = None
    ) -> list[RerankResult]:
        """Score every example, ``concurrency`` queries in flight, and return the results in input order.

        Each query is one request (or one row per chunk of a budget-split document, the chunks' scores pooled
        by ``max``),
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
        self,
        query: str | Content,
        documents: Sequence[str | Content],
        *,
        instruction: str | None = None,
        query_id: str = "",
    ) -> RerankResult:
        """The async half of :meth:`rerank`: prepare (media, gates, budget), send, and read the scores back
        aligned to the documents.

        ``empty_doc: omit_zero`` omits an empty document from the request and scores it 0.0: the returned
        scores stay aligned to the documents as given, and a candidate set whose every document is omitted
        makes no request (an empty request never goes out).

        Args:
            query: The query, as text or content parts.
            documents: The candidates, as text or content parts. An empty document follows the config's
                ``empty_doc`` policy (``send`` sends the empty string as it is and scores whatever the
                server returns; ``omit_zero`` never sends it and scores 0.0; ``send_text`` sends the
                placeholder).
            instruction: The task instruction, folded or sent per the config's ``instruction`` mode.
            query_id: The query's id, for the empty-query refusal's message (``arerank_many`` passes the
                example's id); ``""`` names it ``<unnamed>``.

        Returns:
            One relevance score per document, aligned to the input order (never the server's ranking order).

        Raises:
            ConfigError: The config cannot serve this request (see :meth:`__init__`).
            DataError: the query is empty and the config's ``empty_query: refuse`` (the default) declines
                it, naming the query id.
            TextBudgetExceededError: ``on_overflow: fail`` and a pair over budget, or a query that fills the
                budget with no split declared.
            CapabilityError: The endpoint refused the request as too long, or media rides a side the
                config's ``media_sides`` does not allow.
            RequestRejectedError: The endpoint refused this one request.
            ProviderError: The endpoint failed after its retries, or its answer was unusable.
        """
        prepared_query = self._prepare([query], EncodeRole.QUERY, instruction=instruction)[0]
        if (
            getattr(self.config, "empty_query", "send") == "refuse"
            and not prepared_query.text
            and not prepared_query.has_media
        ):
            raise DataError(
                f"the query {query_id or '<unnamed>'!r} is empty, and the config refuses an empty query "
                "(empty_query: refuse): scoring an empty query against every candidate would rank by nothing",
                hint="declare empty_query: send on the rerank config, or drop the empty query from the run",
            )
        prepared_documents = self._prepare(documents, EncodeRole.DOCUMENT)
        if not documents:
            return RerankResult(scores=())  # an empty candidate set is not a request (as on the served path)
        wire_query, wire_documents, fitted, omitted, kept_positions = self._fit_pair(
            prepared_query, prepared_documents, instruction=instruction
        )
        if not wire_documents:
            # Every document omitted (empty_doc: omit_zero): nothing to score, no request.
            return RerankResult(scores=tuple(0.0 for _ in documents))
        request = RerankRequest(
            query=wire_query,
            documents=tuple(wire_documents),
            instruction=instruction if self.config.instruction == "field" else None,
        )
        calls = self._adapter.calls(request, model=self.config.model)
        self._gate_media_calls(calls)
        replies = await self._send(calls)
        # The scores pool back through `origin` (fit output -> original document index), so chunking and
        # `empty_doc: omit_zero` compose: the pooled score lands on the document it was scored for, and an
        # omitted document scores 0.0 at its position.
        return self._pooled(fitted, self._adapter.interpret(request, replies), len(documents), omitted, kept_positions)

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
                result = await self.arerank(
                    example.as_content,
                    example.doc_contents,
                    instruction=example.instruction,
                    query_id=str(example.id),
                )
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

    def _fit_pair(
        self, query: Content, documents: Sequence[Content], *, instruction: str | None = None
    ) -> tuple[Content, list[Content], FitResult, tuple[int, ...], list[int]]:
        """The query and its candidates as the wire carries them, fitted into the pair budget.

        With a budget declared, :meth:`RoleClient._fit` runs the shared mechanism over the ``(query,
        document)`` pairs (shape ``pair``): the query's span is settled first -- once for the batch, through
        fit's own settlement on a probe pair, because one query rides per request while fit settles a pair's
        query only when that pair overflows -- each document gets what remains, and a chunked document comes
        back as one output per chunk (``<id>#<k>``) with the full template around it. The settlement (a
        query over its declared ``query_max_tokens``, or one that alone fills the budget) is recorded in the
        census under :data:`QUERY_DOC_ID`. The vendor path (no tokenizer) settles nothing: fit sends the
        pairs uncut and records the documented limit. The wire takes the cut spans (the engine renders the
        template itself); the chunks' ``max`` pooling is the caller's, through the fit result.

        Returns:
            ``(wire_query, wire_documents, fitted, omitted, kept_positions)``: the query content (cut), the
            document contents (one per fit output, in fit's order), the fit result (its ``ids`` align to the
            wire documents; its ``chunk_mapping`` carries each chunk back to its input), the document
            indices ``empty_doc: omit_zero`` never sends, and per kept document its original index.

        Raises:
            TextBudgetExceededError: the declared overflow policy refuses to shorten a pair.
        """
        # The request's media, per wire request: the pooling of one query's candidate set is one
        # RerankRequest (the adapter splits pointwise by batch_size, each call still one query's
        # documents), so the fit runs per (query, document) pair -- the query's media reserved with the
        # document's on every pair (the query rides every pair). The gates run per pair. Media on a side
        # the config does not allow (2b) is refused here, before anything is prepared.
        self._refuse_media_off_its_side("query", [query])
        document_contents = [
            document if isinstance(document, Content) else Content.from_text(document) for document in documents
        ]
        self._refuse_media_off_its_side("document", document_contents)
        request_prepared = self._prepare_request([query, *documents])
        query = request_prepared.contents[0]
        documents = list(request_prepared.contents[1:])
        # The media fit runs per (query, document) pair -- the wire request the engine sees -- and the
        # fitted contents (both images possibly shrunk to the policy minimum) are what ships. The query
        # rides every pair, so its media is reserved on every pair and the FITTED query is what the wire
        # carries (the pairs agree on the query's fit while no pair drops query media; a query-item drop
        # ships fewer tokens than budgeted and is recorded under QUERY_DOC_ID).
        pairs_after_media = [
            self._fit_media_for_request([query, document], doc_ids=[QUERY_DOC_ID, str(original_index)])
            for original_index, document in enumerate(documents)
        ]
        query = pairs_after_media[0][0][0]
        documents = [pair[0][1] for pair in pairs_after_media]
        pair_media = [pair[1] for pair in pairs_after_media]
        kept_documents, omitted = self._apply_empty_documents(documents)
        kept_positions = [index for index in range(len(documents)) if index not in set(omitted)]
        query_media = self._media_counts_of([query])[0].tokens
        if self._budget is None:
            return (
                query,
                kept_documents,
                FitResult(
                    shape="pair",
                    texts=(),
                    contents=(),
                    ids=tuple(str(index) for index in range(len(kept_documents))),
                ),
                tuple(omitted),
                kept_positions,
            )
        # One query rides per request: settle its span first, so every pair of the batch carries the same
        # one. fit settles a pair's query only when that pair overflows (the share binds on overflow only),
        # which would settle differently per document -- an under-budget pair keeps the whole query while an
        # overflowing one cuts it to its share. So the client settles it once, exactly as fit would: to the
        # declared share when the query exceeds it, then through fit's own probe pair (the query with an
        # empty document, reserving the documents' maximum media count so the settled span matches what
        # ships) for the empty-render verification.
        original_query = query.text
        query_text = original_query
        if not kept_documents:
            # Every document omitted (empty_doc: omit_zero): no request, no settlement, an empty result.
            return (
                query,
                [],
                FitResult(shape="pair", texts=(), contents=(), ids=()),
                tuple(omitted),
                [],
            )
        if self._tokenizer is not None:
            share = self._budget.query_max_tokens
            if share is not None and self._tokenizer.count(query_text) > share:
                query_text = token_prefix(query_text, share, self._tokenizer)
            settled = self._fit(
                [(query_text, "")],
                "pair",
                instruction=instruction,
                media_tokens=[query_media],
                record=False,
            ).contents[0][0]
            if settled != original_query:
                self.census.record(
                    corpus=self.ROLE,
                    doc_id=QUERY_DOC_ID,
                    original_chars=len(original_query),
                    kept_chars=len(settled),
                    original_tokens=self._tokenizer.count(original_query),
                    kept_tokens=self._tokenizer.count(settled),
                    mechanism=TextTruncationCensus.TEXT_BUDGET,
                    budget_source="tokenizer",
                    shape="pair",
                    # The row names the budget that bounded the settlement, exactly as fit's pair rows do:
                    # the pair budget (the settled share applies inside it).
                    budget_tokens=self._budget.max_tokens,
                )
            pairs = [(settled, document.text) for document in kept_documents]
            result = self._fit(
                pairs,
                "pair",
                media_tokens=[pair_media[position] for position in kept_positions],
                instruction=instruction,
            )
        else:
            # The vendor path: no tokenizer, so nothing is measured or settled; fit sends the pairs uncut
            # and records the vendor's documented limit.
            settled = query_text
            result = self._fit(
                [(settled, document.text) for document in kept_documents],
                "pair",
                instruction=instruction,
            )
        contents = [pair if isinstance(pair, tuple) else (pair, "") for pair in result.contents]
        # The settled span is the one every output carries: an under-budget pair repeats it and an
        # overflowing one settles to it (or to a shorter cut that the probe pair already applied).
        if contents and len({left for left, _ in contents}) != 1:
            raise DataError(
                f"the pair fit settled the shared query differently across {len(contents)} document(s); "
                "one query rides per request, so the spans must agree",
                hint="this is a bug in the rerank pair fit: report it with the inputs",
            )
        # A chunked document is one wire document per chunk, each carrying its input's media parts beside
        # the piece (the media tokens are reserved per chunk: fit's cap subtracts the pair's media, and
        # every chunk's text is verified against it).
        mapping = result.chunk_mapping or {}
        chunk_origin = [int(mapping.get(out_id, out_id)) for out_id in result.ids]
        wire_query = self._with_text(query, settled)
        wire_documents = [
            self._with_text(kept_documents[source], document_text)
            for source, (_, document_text) in zip(chunk_origin, contents, strict=True)
        ]
        return wire_query, wire_documents, result, tuple(omitted), kept_positions

    def _probe_calls(self, content: Content) -> Sequence[Call]:
        """The rerank request one prepared probe item is sent as (a one-document pair)."""
        request = RerankRequest(
            query=Content.from_text("probe"),
            documents=(content,),
            instruction=None,
        )
        return self._adapter.calls(request, model=self.config.model)

    async def probe(self) -> Any:
        """The role's startup probe: the transport's replica probe, plus -- when the config declares an
        ``image_processor`` -- the engine media check (never silent)."""
        probe: Any = await self._sender.probe()
        await self.check_engine_media()
        return probe

    def _probe_usage(self, reply: Reply) -> TokenCount | None:
        """The reply's prompt-token report (the served rerank wires report OpenAI-style usage)."""
        return self._adapter.usage(reply)

    def _pooled(
        self,
        fitted: FitResult,
        result: RerankResult,
        documents: int,
        omitted: Sequence[int] = (),
        kept_positions: Sequence[int] = (),
    ) -> RerankResult:
        """The scores of one query, realigned to the documents as they were given.

        The fit's inputs are the kept documents (kept-relative ids ``"0"``..; a chunked document's chunks
        are ``<kept id>#<k>``), and :attr:`FitResult.ids` aligns to its outputs. ``kept_positions`` carries,
        per kept document, its original index. The chunked outputs pool onto their documents by
        :func:`rcp_ndcg.data.preprocess.max_pool_scores_by_document` (``aggregation: max``) first; every
        score then lands on its document's ORIGINAL position, in the input's order -- ``empty_doc:
        omit_zero``'s omitted documents score 0.0 at theirs.
        """
        scores = [0.0] * documents
        if fitted.chunk_mapping is None:
            for position, score in zip(kept_positions, result.scores, strict=True):
                scores[position] = score
        else:
            per_chunk = dict(zip(fitted.ids, result.scores, strict=True))
            pooled = max_pool_scores_by_document(per_chunk, fitted.chunk_mapping)
            for kept_id, score in pooled.items():
                original_index = (
                    int(kept_id.rsplit(CHUNK_ID_SEPARATOR, 1)[0]) if CHUNK_ID_SEPARATOR in kept_id else int(kept_id)
                )
                scores[kept_positions[original_index]] = score
        return RerankResult(scores=tuple(scores))

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
