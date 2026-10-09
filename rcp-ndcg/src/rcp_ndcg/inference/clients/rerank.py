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

from rcp_ndcg.data.postprocess import max_pool_scores_by_document
from rcp_ndcg.data.prepare import MediaCensus
from rcp_ndcg.data.text_budget import (
    ChangeMechanism,
    ContentParts,
    CutCause,
    FitResult,
    TextCutRecord,
    TextTruncationCensus,
    rendered_pair_tokens,
)
from rcp_ndcg.data.text_policy import CHUNK_ID_SEPARATOR, token_prefix
from rcp_ndcg.errors import DataError
from rcp_ndcg.inference.clients._base import RoleClient
from rcp_ndcg.inference.config import RerankEndpoint
from rcp_ndcg.inference.transport import Sender
from rcp_ndcg.inference.types import Call, Reply, RerankRequest, RerankResult, TokenCount

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
        prepared_query = self._stage_normalise(
            [query if isinstance(query, Content) else Content.from_text(query)],
            side="query",
            prompt="",
            instruction=instruction,
        )[0]
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
        prepared_documents = self._stage_normalise(
            [document if isinstance(document, Content) else Content.from_text(document) for document in documents],
            side="document",
            prompt="",
        )
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
    def _side_prefix(self, side: str) -> str:
        """The rerank role declares no per-side prompt prefix: the pair template's fixed segments are the
        frame's home, the query's instruction mode is :meth:`_stage_normalise`'s, and the wire takes the
        cut spans (the engine renders the template itself)."""
        return ""

    def _stage_normalise(
        self,
        contents: Sequence[Content],
        *,
        side: str,
        prompt: str,
        instruction: str | None = None,
    ) -> list[Content]:
        """The rerank role's ``normalise`` stage: text materialised to content parts, and the query's
        instruction folded into its text for ``instruction: fold`` -- exactly the served path's
        :meth:`~rcp_ndcg_core._records.Query.format_content` render, so the served and the hosted path send
        the same query text. The budget fit happens on the pairs, in :meth:`_fit_pair`.

        Args:
            contents: The texts or content parts as the caller gave them.
            side: Which side of the pair this batch is (the instruction applies to the query only).
            prompt: The side's prompt prefix (the rerank role has none).
            instruction: The task instruction, when the caller has one.

        Returns:
            One :class:`~rcp_ndcg_core.content.Content` per input, in order, unchanged unless the instruction
            mode folds it.
        """
        prepared = [content if isinstance(content, Content) else Content.from_text(content) for content in contents]
        if side != "query" or self.config.instruction != "fold" or not instruction:
            return prepared
        return [
            Query(query_id="", query=content.text, instruction=instruction, content=content).format_content()
            for content in prepared
        ]

    def _fit_pair(
        self, query: Content, documents: Sequence[Content], *, instruction: str | None = None
    ) -> tuple[Content, list[Content], FitResult, tuple[int, ...], list[int]]:
        """The query and its candidates as the wire carries them, fitted into the pair budget.

        This is the rerank role's composition of the pipeline (:data:`STAGES`, one order for every role):
        the ``normalise`` stage ran in the caller (the instruction fold), this method runs the rest -- the
        ``media`` stage (ONE preparation of the whole request, then the pair's media fits, documents first
        and the query against the heaviest kept, because one query content rides every call), the ``empty``
        stage's re-entry on the media-fitted documents (as-given empty documents carry no media, so the
        as-given decision is a no-op and the policy fires here, on the content as it will be sent), the
        ``render`` + ``budget`` stages (the pair fit: the query's share, the document cap, then the pair
        budget), and the ``lower`` stage (the wire forms below). Nothing re-orders them.

        With a budget declared, :meth:`RoleClient._fit` runs the shared mechanism over the ``(query,
        document)`` pairs (shape ``pair``): the query's span is settled first -- once for the batch, through
        fit's own settlement on a probe pair reserving the documents' maximum media count, because one query
        rides every request while fit settles a pair's query only when that pair overflows -- each document
        gets what remains, and a chunked document comes back as one output per chunk (``<id>#<k>``) with the
        full template around it. The settlement (a query over its declared ``query_max_tokens``, or one that
        alone fills the budget) is recorded in the census under :data:`QUERY_DOC_ID`. The vendor path (no
        tokenizer) settles nothing: fit sends the pairs uncut and records the documented limit. The wire
        takes the cut spans (the engine renders the template itself); the chunks' ``max`` pooling is the
        caller's, through the fit result.

        The media are decided so that one query and one document always fit one wire call: the request is
        prepared ONCE (:meth:`RoleClient._prepare_request`), each document's media is fitted against the
        budget minus the fixed frame, and the query's media -- decided once, against the heaviest document
        media the fit kept, because the same query content rides every call -- against what remains. Every
        pair's reserved media (query plus document) is then what the text fit subtracts, and the span the
        batch fit verified is the span the wire ships: no pair is shipped over the budget, whatever its
        media.

        Returns:
            ``(wire_query, wire_documents, fitted, omitted, kept_positions)``: the query content (cut, its
            media the one uniform decision), the document contents (one per fit output, in fit's order), the
            fit result (its ``ids`` align to the wire documents; its ``chunk_mapping`` carries each chunk
            back to its input), the document indices ``empty_doc: omit_zero`` never sends, and per kept
            document its original index.

        Raises:
            TextBudgetExceededError: the declared overflow policy refuses to shorten a pair.
        """
        # Media on a side the config does not allow (2b) is refused here, before anything is prepared.
        self._refuse_media_off_its_side("query", [query])
        document_contents = [
            document if isinstance(document, Content) else Content.from_text(document) for document in documents
        ]
        self._refuse_media_off_its_side("document", document_contents)
        # ONE preparation of the whole request (the query's and every document's media): the census rows it
        # writes name their own doc id, and the per-pair fits below slice it (no second preparation, whose
        # rows would name data: URIs).
        request_prepared = self._prepare_request(
            [query, *documents], doc_ids=[QUERY_DOC_ID, *(str(index) for index in range(len(documents)))]
        )
        prepared_query = request_prepared.contents[0]
        prepared_documents = list(request_prepared.contents[1:])
        # Per input id (a document's position, the shared query's QUERY_DOC_ID), the changes made before the
        # text fit -- media resized or dropped, empty documents substituted -- for the rows' processing records.
        changes: dict[str, list[ChangeMechanism]] = {}
        cuts: list[TextCutRecord] = []
        pair_media = [0] * len(prepared_documents)
        query_media = 0
        # The span that will ship: the settled query is share-capped first, so the media allowances reserve
        # ITS render -- never the raw over-share text that never rides the wire (reserving that drops media
        # which fit the shipped pair). rendered_pair_tokens measures the assembled render WITH the fixed
        # frame, so the pair's frame cost is subtracted exactly once.
        original_query = prepared_query.text
        query_text = original_query
        if (
            self._tokenizer is not None
            and self._budget is not None
            and self._budget.query_max_tokens is not None
            and self._tokenizer.count(query_text) > self._budget.query_max_tokens
        ):
            query_text = token_prefix(query_text, self._budget.query_max_tokens, self._tokenizer)
        if self._budget is not None and request_prepared.media:
            slices = request_prepared.per_content()
            if self._tokenizer is not None:
                query_render = rendered_pair_tokens(
                    self._budget,
                    self._tokenizer,
                    query=query_text,
                    document="",
                    instruction=instruction or "",
                )
            else:
                query_render = 0
            query_media_raw = request_prepared.content_tokens[0].tokens
            # The documents first: each pair's document media, against the budget minus the settled query's
            # render and the one document-text token fit's empty-query invariant keeps for a text document.
            doc_fits = [
                self._fit_media_for_request(
                    [content],
                    doc_ids=[str(index)],
                    prepared=slices[1 + index],
                    allowance=max(
                        self._budget.max_tokens - query_media_raw - query_render - (1 if content.text else 0), 0
                    ),
                    changes=changes,
                )
                for index, content in enumerate(prepared_documents)
            ]
            prepared_documents = [fit[0][0] for fit in doc_fits]
            pair_media = [fit[1] for fit in doc_fits]
            # The query's media, decided once against the heaviest kept document media: the same query
            # content rides every call, so one pair's decision is the request's (a query-item drop is
            # recorded under QUERY_DOC_ID, exactly like the query's text settlement).
            text_floor = 1 if any(document.text for document in prepared_documents) else 0
            query_fit = self._fit_media_for_request(
                [prepared_query],
                doc_ids=[QUERY_DOC_ID],
                prepared=slices[0],
                allowance=max(self._budget.max_tokens - max(pair_media, default=0) - query_render - text_floor, 0),
                changes=changes,
            )
            prepared_query = query_fit[0][0]
            query_media = query_fit[1]
        query = prepared_query
        documents = prepared_documents
        kept_documents, omitted = self._apply_empty_documents(documents, changes=changes)
        kept_positions = [index for index in range(len(documents)) if index not in set(omitted)]
        if self._budget is None or not kept_documents:
            self._record_processing("pair", changes=changes)
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
        if not kept_documents:
            # Every document omitted (empty_doc: omit_zero): no request, no settlement, an empty result.
            return (
                query,
                [],
                FitResult(shape="pair", texts=(), contents=(), ids=()),
                tuple(omitted),
                [],
            )
        # One query rides per request: settle its span first, so every pair of the batch carries the same
        # one. fit settles a pair's query only when that pair overflows (the share binds on overflow only),
        # which would settle differently per document -- an under-budget pair keeps the whole query while an
        # overflowing one cuts it to its share. So the client settles it once, exactly as fit would: to the
        # declared share when the query exceeds it, then through fit's own probe pair (the query with an
        # empty document, reserving the query's media beside the documents' maximum media count, so the
        # settled span fits every pair's cap -- a pair with less media only has more room).
        kept_pair_media = [query_media + pair_media[position] for position in kept_positions]
        if self._tokenizer is not None:
            settled = self._fit(
                [(query_text, "")],
                "pair",
                instruction=instruction,
                media_tokens=[query_media + max(pair_media, default=0)],
                record=False,
            ).contents[0][0]
            settle_cut = self._record_settlement(
                original_query,
                settled,
                instruction=instruction,
                reserved_media=query_media + max(pair_media, default=0),
            )
            if settle_cut is not None:
                cuts.append(settle_cut)
            pairs = [(settled, document.text) for document in kept_documents]
            result = self._fit(
                pairs,
                "pair",
                media_tokens=kept_pair_media,
                instruction=instruction,
                ids=[str(position) for position in kept_positions],
            )
        else:
            # The vendor path: no tokenizer, so nothing is measured or settled; fit sends the pairs uncut
            # and records the vendor's documented limit.
            settled = query_text
            result = self._fit(
                [(settled, document.text) for document in kept_documents],
                "pair",
                instruction=instruction,
                ids=[str(position) for position in kept_positions],
            )
        contents = [pair if isinstance(pair, tuple) else (pair, "") for pair in result.contents]
        # The settled span is the one every output carries: the probe pair reserved every pair's media, so
        # no pair re-cuts the query. A residual divergence (a re-tokenization corner) is closed by re-fitting
        # every pair at the shortest verified span -- one query rides per request, so the spans must agree.
        if contents and len({left for left, _ in contents}) != 1:
            shortest = min({left for left, _ in contents}, key=len)
            result = self._fit(
                [(shortest, document.text) for document in kept_documents],
                "pair",
                media_tokens=kept_pair_media,
                instruction=instruction,
                ids=[str(position) for position in kept_positions],
            )
            contents = [pair if isinstance(pair, tuple) else (pair, "") for pair in result.contents]
            if self._tokenizer is not None and not any(cut.doc_id == QUERY_DOC_ID for cut in cuts):
                # The re-fit shortened the shared query below the settled span: a settlement like any other.
                settle_cut = self._record_settlement(
                    original_query,
                    shortest,
                    instruction=instruction,
                    reserved_media=query_media + max(pair_media, default=0),
                )
                if settle_cut is not None:
                    cuts.append(settle_cut)
            if len({left for left, _ in contents}) != 1:
                raise DataError(
                    f"the pair fit settled the shared query differently across {len(contents)} document(s); "
                    "one query rides per request, so the spans must agree",
                    hint="this is a bug in the rerank pair fit: report it with the inputs",
                )
        # The rows' processing records: the settlement's and the pair fit's census rows (the last fit's, when
        # a residual divergence re-fitted), and the media and empty-document changes noted above.
        self._record_processing("pair", cuts=[*cuts, *result.cuts], changes=changes)
        # A chunked document is one wire document per chunk, each carrying its input's media parts beside
        # the piece (the media tokens are reserved per chunk: fit's cap subtracts the pair's media, and
        # every chunk's text is verified against it).  The fit ids are the documents' ORIGINAL positions
        # (the census rows name them); the wire document's media comes from the kept document at that
        # original position.
        mapping = result.chunk_mapping or {}
        original_to_kept = {position: kept for kept, position in enumerate(kept_positions)}
        chunk_origin = [original_to_kept[int(mapping.get(out_id, out_id))] for out_id in result.ids]
        shipped_query = contents[0][0]  # the span the fit verified -- what ships, not the probe's alone
        # Each chunk output carries its input's media beside the piece, so the per-output media are the
        # kept pair's (a chunked document's media ride every chunk).
        per_output_media = [kept_pair_media[source] for source in chunk_origin]
        self._assert_pairs_within_budget(shipped_query, contents, per_output_media, instruction)
        wire_query = self._with_text(query, shipped_query)
        wire_documents = [
            self._with_text(kept_documents[source], document_text)
            for source, (_, document_text) in zip(chunk_origin, contents, strict=True)
        ]
        return wire_query, wire_documents, result, tuple(omitted), kept_positions

    def _record_settlement(
        self, original_query: str, span: str, *, instruction: str | None, reserved_media: int
    ) -> TextCutRecord | None:
        """Record the shared query's settlement under :data:`QUERY_DOC_ID` when the span that ships removed
        content from the query, and return the census row (``None``: nothing was removed).

        Like with like: fit returns the span under the template's declared normalisation (strip, lowercase),
        which both sides apply and which is never a change -- the span is compared with the normalised query,
        and only content actually removed (characters or tokens) is a cut. The row names why the query changed
        (its declared share, else the budget the probe pair bounds it by) and the uncut and the kept probe
        request's whole size: the frame with the query and an empty document, plus the media the probe reserved
        -- the settlement rides every pair, so it is recorded even when every pair would fit whole.
        """
        budget, tokenizer = self._budget, self._tokenizer
        assert budget is not None and tokenizer is not None  # the settlement is measured
        template = budget.template
        normalised = (
            template.normalize_text("pair", original_query)
            if template is not None and template.normalisers("pair")
            else original_query
        )
        if span == normalised or (
            len(span) >= len(normalised) and tokenizer.count(span) >= tokenizer.count(normalised)
        ):
            return None
        share = budget.query_max_tokens
        cause: CutCause = (
            "query_share" if share is not None and tokenizer.count(original_query) > share else "budget_cut"
        )

        def probe_tokens(query: str) -> int:
            return (
                rendered_pair_tokens(budget, tokenizer, query=query, document="", instruction=instruction or "")
                + reserved_media
            )

        return self.census.record(
            corpus=self.ROLE,
            doc_id=QUERY_DOC_ID,
            original_chars=len(original_query),
            kept_chars=len(span),
            original_tokens=tokenizer.count(original_query),
            kept_tokens=tokenizer.count(span),
            mechanism=TextTruncationCensus.TEXT_BUDGET,
            budget_source="tokenizer",
            shape="pair",
            # The row names the budget that bounded the settlement, exactly as fit's pair rows do: the pair budget
            # (the settled share applies inside it).
            budget_tokens=budget.max_tokens,
            cause=cause,
            original_request_tokens=probe_tokens(normalised),
            kept_request_tokens=probe_tokens(span),
        )

    def _assert_pairs_within_budget(
        self,
        query_text: str,
        contents: Sequence[ContentParts],
        pair_media: Sequence[int],
        instruction: str | None,
    ) -> None:
        """The shipped pairs are the budget's, before anything is sent: each wire pair's assembled render
        (the same measure :func:`rcp_ndcg.data.preprocess.fit` verified it with) plus the pair's reserved
        media stays within ``max_tokens``. A violation means the shipped spans and the verified ones
        diverged -- a bug in the pair fit, raised as one, never sent to the engine to truncate."""
        budget, tokenizer = self._budget, self._tokenizer
        assert budget is not None  # a pair fit without a budget ships uncut and never asserts one
        if tokenizer is None:
            return  # the vendor path: nothing is measured client-side
        for (query_span, document_span), media_tokens in zip(contents, pair_media, strict=True):
            total = rendered_pair_tokens(
                budget, tokenizer, query=query_span, document=document_span, instruction=instruction or ""
            )
            if total + media_tokens > budget.max_tokens:
                raise DataError(
                    f"the shipped pair is {total + media_tokens} tokens (render {total} + media "
                    f"{media_tokens}) over the budget of {budget.max_tokens}; the fit verified a pair that "
                    f"differs from the one the wire carries",
                    hint="this is a bug in the rerank pair fit: report it with the inputs",
                )

    def _probe_calls(self, content: Content) -> Sequence[Call]:
        """The rerank request one prepared probe item is sent as (a one-document pair)."""
        request = RerankRequest(
            query=Content.from_text("probe"),
            documents=(content,),
            instruction=None,
        )
        return self._adapter.calls(request, model=self.config.model)

    def _probe_baseline_calls(self, content: Content) -> Sequence[Call] | None:
        """The probe request without its media: the same body shape, the document as the plain text it
        carries (the engine's two prompt-token reports differ by the media block alone)."""
        request = RerankRequest(
            query=Content.from_text("probe"),
            documents=(Content.from_text(content.text),),
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
                # The fit ids are the documents' ORIGINAL positions (the census rows name them); a chunk's
                # id carries its origin before the separator, so the pooled score lands on its document.
                original_index = int(kept_id.rsplit(CHUNK_ID_SEPARATOR, 1)[0])
                scores[original_index] = score
        return RerankResult(scores=tuple(scores))

    async def _send(self, calls: Sequence[Call]) -> list[Any]:
        """Send one request's calls and return their replies, one per call, in order.

        A profile with a pause (Voyage) sends its calls one at a time, sleeping before each as today's
        ``VoyageRerank`` does; otherwise the calls go in one ``send``, which the transport routes to one
        replica without interleaving. Every reply's token report is folded into the transport's usage
        (:meth:`RoleClient._record_usage`), whichever way the calls went out.
        """
        sender = self._sender
        pause = getattr(self._adapter, "PAUSE_S", 0.0)
        if not pause:
            replies = await sender.send(calls)
            self._record_usage(replies)
            return replies
        replies: list[Any] = []
        for call in calls:
            await asyncio.sleep(pause)
            replies.extend(await sender.send([call]))
        self._record_usage(replies)
        return replies


__all__ = ["Checkpoint", "RerankClient"]
