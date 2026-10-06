"""Judging: run the tournament (Stage A) or the rubric (Stage B) over candidate pools.

:func:`judge` is the one judging path. It schedules the windows of every query
(:mod:`rcp_ndcg.llm.schedule`), renders each window with the stage's prompt,
asks the judge, parses the answer into a :class:`~rcp_ndcg_core.schemas.Judgement`
and appends it to the judgement store (:mod:`rcp_ndcg.llm.store`) the moment it
lands. Because every window has a stable ``record_id``, running :func:`judge`
again over the same store asks only for the windows that are missing: resuming
an interrupted pass and re-judging a subset of documents (``docs=``) are the
same call.

Windows of one phase are asked concurrently (the client bounds the calls in
flight) and their answers enter the scheduler's estimators in window order, so a
pass is deterministic for a deterministic judge whatever order the answers
arrive in.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rcp_ndcg_core._hashing import hash_payload, short
from rcp_ndcg_core._records import RankingExample
from rcp_ndcg_core.content import Content, ImagePart, VideoPart
from rcp_ndcg_core.irt import Priors
from rcp_ndcg_core.schemas import (
    Family,
    InvalidCategory,
    Judgement,
    JudgementSet,
    Phase,
    Placement,
    Stage,
    judgement_record_id,
)

from rcp_ndcg.data.prepare import MediaCensus
from rcp_ndcg.data.preprocess import (
    Preprocessing,
    TextTruncationCensus,
    chunk_ranking_example,
    document_id_for_chunk,
    document_ids_from_chunks,
    max_pool_rubric_window_by_document,
    needs_tokenizer,
    require_tokenizer,
    token_prefix,
)
from rcp_ndcg.errors import CapabilityError, ConfigError, DataError
from rcp_ndcg.llm._parsing.common import PARSE_VERSION, UnparseableAnswer
from rcp_ndcg.llm._parsing.schema import response_format as _response_format
from rcp_ndcg.llm.client import Completion, CompletionInput, JudgeClient, JudgeConfig, RequestRejectedError
from rcp_ndcg.llm.prompts import PROMPT_FILES, Prompt, load_prompt, shipped_prompt_name
from rcp_ndcg.llm.schedule import (
    Modality,
    RubricSchedule,
    TournamentSchedule,
    _balanced_groups,
    _canonical_pair,
    _compute_boundary_values,
    _greedy_select_windows,
    _stratified_groups,
    query_rng,
    schedule_for,
    schedule_key,
)
from rcp_ndcg.llm.store import JudgementStore
from rcp_ndcg.storage import local_dir
from rcp_ndcg.support.logging import get_logger

if TYPE_CHECKING:
    from rcp_ndcg.data.tokenizer import TextTokenizer

logger = get_logger(__name__)

#: Attempts per window before an unparseable answer or a refused request is recorded as an invalid judgement.
MAX_ATTEMPTS = 3

#: The preprocessing record a judging pass writes next to its judgements: every text cut and every media item sent
#: (:class:`~rcp_ndcg.data.preprocess.TextTruncationCensus` and :class:`~rcp_ndcg.data.prepare.MediaCensus` rows).
PREPROCESSING_RECORD = "preprocessing.jsonl"

#: Tokens kept, per request, for what the tokenizer file does not describe and no media occupies: the chat
#: template's role markers, its generation prompt, and a system message some templates insert (gpt-oss's
#: harmony format does). A fixed reserve for what is not media: the media themselves are charged exactly (see
#: :func:`media_marker_tokens` and :func:`~rcp_ndcg.data.resolution.content_media_tokens`).
CHAT_TEMPLATE_TOKENS = 256


def media_marker_tokens(tokenizer: TextTokenizer) -> int:
    """The tokens of one media marker as the stage's template renders it, counted with the judge's tokenizer.

    The window's template renders each media part as a :data:`~rcp_ndcg.llm._templates.MEDIA_MARKER`
    placeholder; the window budget charges its tokens per media part -- a declared reserve, measured from
    the template and never guessed. The engine's payload replaces the marker with the media part (so the
    charge is a few tokens above what ``usage.prompt_tokens`` reports); the reserve keeps the budget honest
    for the text the template renders around the media, and errs in the safe direction.
    """
    from rcp_ndcg.llm._templates import MEDIA_MARKER

    return tokenizer.count(MEDIA_MARKER.format(index=0))


def prompt_overhead_tokens(prompt: Prompt, stage: Stage, query: str, window: int, tokenizer: TextTokenizer) -> int:
    """The tokens of one window's prompt besides its documents' text, counted with the judge's tokenizer.

    The stage's template rendered with the query and ``window`` empty documents (the instructions, the query and
    the document markup), plus :data:`CHAT_TEMPLATE_TOKENS`.
    """
    template = prompt.template(with_num_documents=stage == "tournament")
    rendered = template.resolve(query=query, documents=[Content.from_text("")] * window)
    return tokenizer.count(rendered) + CHAT_TEMPLATE_TOKENS


def window_tokens(
    config: JudgeConfig, window: int, *, overhead_tokens: int, media_tokens_per_doc: int = 0
) -> int | None:
    """The tokens of text each document of a window may carry so the window fits the judge's context.

    The context (``context_tokens``) less the prompt's own tokens (``overhead_tokens``,
    :func:`prompt_overhead_tokens`), less the documents' media charge (``media_tokens_per_doc``: each image
    and video its vision block -- the processor's vision start and end plus its patch tokens, a container its
    temporal grid -- plus the template's media marker per media part), less the completion reserve
    (``max_output_tokens``, at most half of what is left), shared equally by ``window`` documents.

    The marker is a declared reserve, not an engine count: the payload builder replaces each marker with the
    media part, so the engine's prompt carries the vision block the charge already covers and never the
    marker itself -- the charge keeps the budget honest for the text the template renders around the media,
    and errs a few tokens high per media part, never low. ``None`` when the judge declares no context:
    documents are sent whole.

    Raises:
        CapabilityError: the prompt and the media alone do not fit.
    """
    context = config.context_tokens
    if context is None:
        return None
    usable = context - overhead_tokens - window * media_tokens_per_doc
    if usable <= 0:
        raise CapabilityError(
            f"a window of {window} documents does not fit the judge's context of {context:,} tokens: its prompt "
            f"takes {overhead_tokens:,} tokens and its images {window * media_tokens_per_doc:,} "
            f"({media_tokens_per_doc:,} per document)",
            hint="use a smaller window, or a lower pixel budget in the resolution policy",
        )
    usable -= min(max(config.max_output_tokens or 0, 0), usable * 0.5)
    return int(usable // window)


def _media_tokens(
    contents: Iterable[Content], preprocessing: Preprocessing, *, strict: bool = True, marker_tokens: int = 0
) -> int | None:
    """The most media tokens any of ``contents`` charges a window's text budget, under the pass's policies.

    Each document is charged its media as the engine counts it
    (:func:`~rcp_ndcg.data.resolution.content_media_tokens`: every image and sampled frame its vision block,
    a container its temporal grid) plus ``marker_tokens`` per media part -- the template's media markers,
    :func:`media_marker_tokens` measures them with the judge's tokenizer.

    Args:
        contents: The documents of a query.
        preprocessing: The effective preprocessing (its ``image`` policy, native when unset, and ``video``).
        strict: Raise when the media cannot be counted; otherwise return ``None`` for them.
        marker_tokens: The template's per-part media marker, measured with the judge's tokenizer; 0 where no
            tokenizer is at hand (no text budget is computed then, either).

    Raises:
        ConfigError: ``strict``, and the media cannot be counted (a native policy, or no ``image_processor``).
    """
    from rcp_ndcg.data.resolution import ImagePolicy, content_media_tokens

    image = preprocessing.image or ImagePolicy.native()
    try:
        return max(
            (
                content_media_tokens(content, image, preprocessing.video).tokens
                + marker_tokens * sum(isinstance(part, ImagePart | VideoPart) for part in content.parts)
                for content in contents
            ),
            default=0,
        )
    except ConfigError:
        if strict:
            raise
        return None


def _judge_tokenizer(config: JudgeConfig) -> TextTokenizer | None:
    """The judge's tokenizer (:attr:`JudgeConfig.tokenizer`, loaded once per process), or ``None``."""
    from rcp_ndcg.data.tokenizer import load_tokenizer

    return load_tokenizer(config.tokenizer) if config.tokenizer is not None else None


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class _Query:
    """One query's judging input: the ids the judge sees (chunk ids when chunked) and their bodies."""

    dataset: str
    query_id: str
    text: str
    units: list[str]
    contents: dict[str, Content]
    chunk_mapping: dict[str, str] | None

    def doc_of(self, unit_id: str) -> str:
        return document_id_for_chunk(unit_id, self.chunk_mapping)


def _rows(dataset: Any, candidates: Mapping[str, Sequence[str]] | None) -> tuple[str, list[RankingExample], Any]:
    """``(name, rows, source)``: the rows to judge and, for a :class:`~rcp_ndcg.data.Dataset`, the dataset.

    A dataset's rows are its queries with their candidate pools (``candidates``, else the dataset's own pools,
    else its judged documents) and the documents' bodies from its corpus.
    """
    from rcp_ndcg.data.dataset import Dataset

    if isinstance(dataset, Dataset):
        return dataset.name, _dataset_rows(dataset, candidates), dataset
    if isinstance(dataset, Sequence) and all(isinstance(row, RankingExample) for row in dataset):
        return "dataset", list(dataset), None
    raise ConfigError(
        f"judge() takes a Dataset (rcp_ndcg.load_dataset) or a sequence of ranking rows, got {type(dataset).__name__}"
    )


def _dataset_rows(dataset: Any, candidates: Mapping[str, Sequence[str]] | None) -> list[RankingExample]:
    if dataset.subsets:
        raise ConfigError(f"{dataset.name!r} is a suite; judge one subset at a time (Dataset.subsets)")
    pools = candidates if candidates is not None else dataset.candidates
    if pools is None:
        pools = {query: list(docs) for query, docs in dataset.qrels.items()}
    queries, corpus = dataset.queries, dataset.corpus
    rows = []
    for query_id, pool in pools.items():
        query = queries.get(str(query_id))
        if query is None:
            raise DataError(f"query {query_id!r} of the candidates is not a query of {dataset.name!r}")
        absent = [doc for doc in pool if doc not in corpus]
        if absent:
            raise DataError(
                f"query {query_id!r}: {len(absent)} candidates are not in the corpus of {dataset.name!r}, "
                f"e.g. {absent[:3]}"
            )
        rows.append(
            RankingExample(
                query_id=str(query_id),
                query=query.text,
                instruction=query.instruction,
                content=query.content,
                doc_ids=list(pool),
                contents=[corpus[doc].as_content for doc in pool],
                qrels=dataset.qrels.get(str(query_id)) or None,
            )
        )
    return rows


def _dataset_identity(name: str, source: Any, rows: Any = None) -> dict[str, Any]:
    """The dataset part of a pass's identity: its name, and for a loaded dataset its URI and resolved revision.

    A local URI (a reader scheme whose location is a path, not ``scheme://``) enters with its path absolute and
    normalised, so the same file named from another directory or as ``./rows.jsonl`` is the same dataset. The
    query selection is left out: judging more queries of the same corpus extends the store.

    A row-sequence input (no dataset object) has no name or revision to give: it is named by the SHA-256 of
    the rows it judges, so two passes over different row corpora never share a store identity (and, through
    :func:`~rcp_ndcg_core.schemas.judgement_record_id`, never share record ids).
    """
    if source is None:
        if rows is None:
            return {"name": name}
        return {"name": name, "rows_sha256": hash_payload({"rows": [row.model_dump(mode="json") for row in rows]})}
    from rcp_ndcg.data.revisions import dataset_uri_revision

    return {
        "name": name,
        "uri": _normalised_uri(source.uri),
        "revision": dataset_uri_revision(source.uri, source.revision),
    }


def _normalised_uri(uri: str) -> str:
    """``uri`` with a local location made absolute and normalised; a remote one (``hf://``, ``suite:``) as it is."""
    scheme, sep, location = uri.partition(":")
    if not sep or scheme == "suite" or "://" in uri or not location:
        return uri
    return f"{scheme}:{Path(location).expanduser().resolve()}"


def _effective_preprocessing(preprocessing: Preprocessing | None, judge: JudgeConfig) -> Preprocessing:
    """The preprocessing in effect: the requested one, else no load-time policy (the window budget still applies).

    Its image policy is taken under the judge's ``image_processor`` (:meth:`ImagePolicy.for_processor`), so the
    record and the family key name the processor family whose resize the client applies.

    Raises:
        ConfigError: the image policy does not fit the judge's processor.
    """
    effective = preprocessing if preprocessing is not None else Preprocessing()
    if effective.image is None:
        return effective
    image = effective.image.for_processor(judge.image_processor)
    if not image.is_native and image.processor is None:
        logger.warning(
            "the judge %s declares no image_processor: images and frames are sent unchanged, as stored, and the "
            "declared pixel budget %s is not applied (recorded as processor: null)",
            judge.model,
            image.descriptor,
        )
    return effective.model_copy(update={"image": image})


def _capped(
    row: RankingExample, preprocessing: Preprocessing, name: str, census: Any, tokenizer: TextTokenizer | None
) -> RankingExample:
    """``row`` with the text policy applied to every document's text (media is never shortened)."""
    from rcp_ndcg.data.preprocess import apply_text_policy

    policy = preprocessing.text
    if policy.on_overflow in ("keep", "chunk") or not row.has_bodies:
        return row
    capped = []
    for doc_id, content in zip(row.doc_ids, row.doc_contents, strict=True):
        cut = apply_text_policy(
            content.text, doc_id=doc_id, policy=policy, tokenizer=tokenizer, corpus=name, census=census
        )
        capped.append(content if cut == content.text else content.truncated(len(cut)))
    return row.model_copy(update={"contents": capped, "docs": [content.text for content in capped]})


def _queries(
    dataset: Any,
    candidates: Mapping[str, Sequence[str]] | None,
    docs: Mapping[str, Sequence[str]] | None,
    preprocessing: Preprocessing,
    census: Any = None,
    tokenizer: TextTokenizer | None = None,
) -> tuple[str, list[_Query], Any]:
    """``(name, queries, source)``: the queries to judge, and the dataset they come from (see :func:`_rows`).

    Raises:
        ConfigError: the preprocessing cuts or chunks text and the judge names no tokenizer.
    """
    if needs_tokenizer(preprocessing):
        what = (
            f"chunking ({preprocessing.chunk.descriptor})"
            if preprocessing.chunk is not None
            else f"text policy {preprocessing.text.on_overflow!r}"
        )
        require_tokenizer(tokenizer, what)
    name, rows, source = _rows(dataset, candidates)
    queries: list[_Query] = []
    for row in rows:
        if candidates is not None and row.id not in candidates:
            continue
        if docs is not None and row.id not in docs:
            continue
        if not row.has_bodies:
            raise DataError(f"query {row.id!r} of {name} has no document bodies; load the dataset with its corpus")
        row = _capped(row, preprocessing, name, census, tokenizer)
        if preprocessing.chunk is not None and not row.chunk_mapping:
            row = chunk_ranking_example(row, preprocessing.chunk, require_tokenizer(tokenizer, "chunking"))
        mapping = row.chunk_mapping or None
        row_docs = document_ids_from_chunks(row.doc_ids, mapping)
        pool = list(candidates[row.id]) if candidates is not None else row_docs
        missing = sorted(set(pool) - set(row_docs))
        if missing:
            raise DataError(f"query {row.id!r}: candidates {missing[:5]} are not among its documents")
        chosen = pool
        if docs is not None:
            wanted = set(docs[row.id])
            absent = sorted(wanted - set(pool))
            if absent:
                raise DataError(f"query {row.id!r}: documents {absent[:5]} to judge are not among its candidates")
            chosen = [doc for doc in pool if doc in wanted]
        rank = {doc: position for position, doc in enumerate(chosen)}
        units = sorted(
            (unit for unit in row.doc_ids if document_id_for_chunk(unit, mapping) in rank),
            key=lambda unit: rank[document_id_for_chunk(unit, mapping)],
        )
        queries.append(
            _Query(
                dataset=name,
                query_id=str(row.id),
                text=row.format_query(),
                units=units,
                contents={unit: content for unit, content in row.doc_id2content.items() if unit in set(units)},
                chunk_mapping=mapping,
            )
        )
    if not queries:
        raise DataError(f"no query of {name} has candidates to judge")
    return name, queries, source


def _store_census(root: Path, loaded: TextTruncationCensus) -> TextTruncationCensus:
    """The census of a pass, appending to the store's ``preprocessing.jsonl``.

    ``loaded`` holds the load-time (``doc_policy``) cuts made while the queries
    were built, before the store directory existed. Each distinct load-time cut of
    a document (its kept tokens and characters) is written once per store: a
    resumed pass cuts the same documents the same way, and those cuts are already
    on record, while another stage or policy that cuts a document differently adds
    its cut. Window-budget cuts are recorded as the pass renders windows, so only
    windows actually asked are recorded.
    """
    sink = root / PREPROCESSING_RECORD
    on_record: set[tuple[Any, ...]] = set()
    from rcp_ndcg.data.preprocess import read_census_rows

    for row in read_census_rows(sink):
        if row["mechanism"] == TextTruncationCensus.DOC_POLICY:
            on_record.add((row["corpus"], row["doc_id"], row.get("kept_tokens"), row["kept_chars"]))
    census = TextTruncationCensus(sink=sink)
    for cut in loaded.cuts(mechanism=TextTruncationCensus.DOC_POLICY):
        cut_key = (cut.corpus, cut.doc_id, cut.kept_tokens, cut.kept_chars)
        if cut_key in on_record:
            continue
        on_record.add(cut_key)
        census.record(
            corpus=cut.corpus,
            doc_id=cut.doc_id,
            original_chars=cut.original_chars,
            kept_chars=cut.kept_chars,
            original_tokens=cut.original_tokens,
            kept_tokens=cut.kept_tokens,
            mechanism=cut.mechanism,
            query_id=cut.query_id,
        )
    return census


def _modality(queries: Sequence[_Query]) -> Modality:
    contents = [content for query in queries for content in query.contents.values()]
    if any(isinstance(part, VideoPart) for content in contents for part in content.parts):
        return "video"
    if any(content.has_media for content in contents):
        return "image"
    return "text"


# ---------------------------------------------------------------------------
# One window's answer and record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowAnswer:
    """A parsed answer of one window: scores and the stated ranking (tournament) or criteria (rubric), by unit."""

    scores: dict[str, float] | None = None
    ranking: tuple[int, ...] | None = None
    criteria: dict[str, dict[str, int]] | None = None


def parse_window(
    stage: Stage, query_id: str, completion: Completion, units: Sequence[str], num_criteria: int
) -> WindowAnswer:
    """Parse one answer of ``stage`` for a window showing ``units`` (in prompt order).

    Raises:
        UnparseableAnswer: the answer is not a complete observation of the window (with its category).
    """
    if stage == "tournament":
        from rcp_ndcg.llm._parsing.listwise import parse_calibrated_listwise

        ranking, scores = parse_calibrated_listwise(query_id, completion, list(units))
        position = {unit: index + 1 for index, unit in enumerate(units)}
        return WindowAnswer(scores=scores, ranking=tuple(position[unit] for unit in ranking))
    from rcp_ndcg.llm._parsing.rubric import parse_rubric_criteria

    return WindowAnswer(criteria=parse_rubric_criteria(query_id, completion, list(units), num_criteria))


def window_record(
    *,
    record_id: str,
    dataset: str,
    query_id: str,
    stage: Stage,
    family: Family,
    window_seq: int | None,
    phase: Phase | None,
    placements: Sequence[tuple[str, str]],
    answer: WindowAnswer | None,
    completion: Completion | None,
    failure: tuple[str, InvalidCategory] | None,
    recorded_at: datetime,
) -> Judgement:
    """The :class:`Judgement` of one window: valid with ``answer``, else invalid with ``failure`` (reason, category).

    ``placements`` are ``(unit_id, doc_id)`` in prompt order.
    """
    rows = []
    for index, (unit, doc) in enumerate(placements):
        rows.append(
            Placement(
                position=index + 1,
                doc_id=doc,
                chunk_id=unit if unit != doc else None,
                score=answer.scores[unit] if answer is not None and answer.scores is not None else None,
                criteria=answer.criteria[unit] if answer is not None and answer.criteria is not None else None,
            )
        )
    reason, category = failure if answer is None and failure is not None else ("no answer", "refused")
    return Judgement(
        record_id=record_id,
        dataset=dataset,
        query_id=query_id,
        stage=stage,
        family_key=family.key,
        window_seq=window_seq,
        phase=phase,
        placements=tuple(rows),
        ranking=answer.ranking if answer is not None else None,
        response=completion.response if completion is not None else None,
        finish_reason=completion.finish_reason if completion is not None else None,
        valid=answer is not None,
        invalid_reason=None if answer is not None else reason[:2000],
        invalid_category=None if answer is not None else category,
        input_tokens=completion.input_tokens if completion is not None else None,
        output_tokens=completion.output_tokens if completion is not None else None,
        recorded_at=recorded_at,
    )


# ---------------------------------------------------------------------------
# One pass
# ---------------------------------------------------------------------------


@dataclass
class _Pass:
    """Everything the windows of one judging pass share."""

    stage: Stage
    client: JudgeClient
    prompt: Prompt
    family: Family
    store: JudgementStore
    existing: dict[str, Judgement]
    preprocessing: Preprocessing
    census: TextTruncationCensus
    media_census: MediaCensus
    tokenizer: TextTokenizer | None
    #: :func:`~rcp_ndcg.llm.schedule.schedule_key` of the pass's schedule: planned windows are keyed by it.
    schedule_key: str
    #: The dataset's identity key (the digest of the store identity's ``dataset`` entry): every record id
    #: names the corpus by it, so two corpora that share query and document ids never share a record id.
    dataset_key: str
    asked: int = 0
    reused: int = 0
    #: ``(query, unit, budget) -> (kept text, original tokens, kept tokens)``: a document's cut, computed once.
    cuts: dict[tuple[str, str, int], tuple[str, int, int]] = field(default_factory=dict)

    @property
    def criteria(self) -> tuple[str, ...]:
        return self.family.criteria

    # -- one window ----------------------------------------------------------

    def window_tokens(self, query: _Query, window: int) -> int | None:
        """The per-document text budget of a window of ``window`` documents of this query, in tokens.

        ``None`` when the judge declares no context, or names no tokenizer: text is never cut by characters, so
        documents are sent whole (media whose token cost is known must still fit the context).

        Raises:
            ConfigError: the judge declares a context and a tokenizer, but the documents' media tokens cannot
                be counted (no pixel budget, or no ``image_processor``): a text budget is not computed on a
                guess.
            CapabilityError: the prompt and the media alone do not fit the context.
        """
        config = self.client.config
        if config.context_tokens is None:
            return None
        marker = media_marker_tokens(self.tokenizer) if self.tokenizer is not None else 0
        if self.tokenizer is None:
            media = _media_tokens(query.contents.values(), self.preprocessing, strict=False)
            if media is not None:
                window_tokens(config, window, overhead_tokens=0, media_tokens_per_doc=media)
            return None
        media = _media_tokens(query.contents.values(), self.preprocessing, marker_tokens=marker) or 0
        overhead = prompt_overhead_tokens(self.prompt, self.stage, query.text, window, self.tokenizer)
        return window_tokens(config, window, overhead_tokens=overhead, media_tokens_per_doc=media)

    def _cut(self, query: _Query, unit: str, text: str, max_tokens: int) -> tuple[str, int, int]:
        """``(kept text, original tokens, kept tokens)`` of one document under a window budget (memoised)."""
        key = (query.query_id, unit, max_tokens)
        if key not in self.cuts:
            from rcp_ndcg.llm._templates import rendered_text

            tokenizer = require_tokenizer(self.tokenizer, "the window budget")
            kept = token_prefix(text, max_tokens, tokenizer, rendered=rendered_text)
            original = tokenizer.count(text)
            self.cuts[key] = (kept, original, original if kept == text else tokenizer.count(kept))
        return self.cuts[key]

    def render(self, query: _Query, units: Sequence[str], max_tokens: int | None) -> CompletionInput:
        """The prompt of one window, with its media prepared, each document's text cut to ``max_tokens`` (as the
        prompt carries it) at a token boundary, and the stage's answer schema when the family decodes to it."""
        from rcp_ndcg.data.prepare import prepare_content

        contents = []
        for unit in units:
            prepared = prepare_content(query.contents[unit], self.preprocessing.image, self.preprocessing.video)
            self.media_census.record(corpus=query.dataset, doc_id=unit, media=prepared.media)
            content = prepared.content
            if max_tokens is not None and content.has_text:
                before = content.text
                kept, original_tokens, kept_tokens = self._cut(query, unit, before, max_tokens)
                if kept != before:
                    content = content.truncated(len(kept))
                    self.census.record(
                        corpus=query.dataset,
                        doc_id=unit,
                        original_chars=len(before),
                        kept_chars=len(content.text),
                        original_tokens=original_tokens,
                        kept_tokens=kept_tokens,
                        mechanism=TextTruncationCensus.WINDOW_BUDGET,
                        query_id=query.query_id,
                    )
            contents.append(content)
        template = self.prompt.template(with_num_documents=self.stage == "tournament")
        resolved = template.resolve_prompt(query=query.text, documents=contents)
        response_format = (
            _response_format(self.stage, len(units), self.criteria) if self.family.decoding == "json_schema" else None
        )
        return CompletionInput(
            user_prompt=resolved.text, user_content=resolved.content, response_format=response_format
        )

    def _record(
        self,
        query: _Query,
        seq: int | None,
        phase: Phase | None,
        units: Sequence[str],
        record_id: str,
        *,
        answer: WindowAnswer | None,
        completion: Completion | None,
        failure: tuple[str, InvalidCategory] | None,
    ) -> Judgement:
        return window_record(
            record_id=record_id,
            dataset=query.dataset,
            query_id=query.query_id,
            stage=self.stage,
            family=self.family,
            window_seq=seq,
            phase=phase,
            placements=[(unit, query.doc_of(unit)) for unit in units],
            answer=answer,
            completion=completion,
            failure=failure,
            recorded_at=datetime.now(UTC),
        )

    async def ask(
        self, query: _Query, seq: int | None, units: Sequence[str], max_tokens: int | None, phase: Phase | None
    ) -> Judgement:
        """One window: the stored judgement when it exists, else the judge's parsed answer, appended to the store.

        An answer that does not parse, or a request the endpoint refuses, is asked
        again up to :data:`MAX_ATTEMPTS` times and then stored as an invalid
        judgement. A stored invalid judgement without an answer (the endpoint
        refused every attempt) is asked again when the pass is resumed; an
        unparseable answer is the judge's answer and is kept. Any other exception
        propagates. ``seq`` is the window's place in the query's schedule, or ``None`` for a planned window, which
        is keyed by its documents alone.
        """
        record_id = judgement_record_id(
            self.family.key,
            query.query_id,
            self.stage,
            seq,
            list(units),
            dataset=self.dataset_key,
            schedule_key=self.schedule_key,
        )
        present = self.existing.get(record_id)
        if present is not None and (present.valid or present.response is not None):
            self.reused += 1
            return present
        request = self.render(query, units, max_tokens)
        completion: Completion | None = None
        failure: tuple[str, InvalidCategory] | None = None
        answer: WindowAnswer | None = None
        for _attempt in range(MAX_ATTEMPTS):
            completion = None
            try:
                completion = await self.client.complete(request)
                answer = parse_window(self.stage, query.query_id, completion, units, len(self.criteria))
                failure = None
                break
            except UnparseableAnswer as exc:
                failure = (f"{type(exc).__name__}: {exc.reason}", exc.category)
            except RequestRejectedError as exc:
                text = str(exc).strip()
                failure = (f"{type(exc).__name__}: {text.splitlines()[-1] if text else repr(exc)}", "refused")
        self.asked += 1
        judgement = self._record(
            query, seq, phase, units, record_id, answer=answer, completion=completion, failure=failure
        )
        self.store.append(judgement)
        self.existing[record_id] = judgement
        return judgement

    async def ask_all(
        self,
        query: _Query,
        start: int | None,
        windows: Sequence[Sequence[str]],
        max_tokens: int | None,
        phase: Phase | None,
    ) -> list[Judgement]:
        """Ask ``windows`` concurrently; return their judgements in window order.

        The windows take the schedule's sequence numbers from ``start``; ``None`` asks planned windows (no
        sequence number). Every window runs to completion before an outage stops the query, so each answer that
        arrived is stored.
        """
        results = await asyncio.gather(
            *(
                self.ask(query, None if start is None else start + index, units, max_tokens, phase)
                for index, units in enumerate(windows)
            ),
            return_exceptions=True,
        )
        judgements: list[Judgement] = []
        for result in results:
            if isinstance(result, BaseException):
                raise result
            judgements.append(result)
        return judgements


# ---------------------------------------------------------------------------
# The two stages, per query
# ---------------------------------------------------------------------------


def _ingest_tournament(judgement: Judgement, bt: Any, obs_counts: Counter[tuple[str, str]]) -> None:
    """Feed one tournament window to the Bradley-Terry estimator (the tournament's own grammar)."""
    from rcp_ndcg.llm._parsing.listwise import judgement_comparisons

    for winner, loser, weight, prob in judgement_comparisons(judgement):
        bt.add_comparison(winner, loser, soft_label=prob, weight=weight)
        obs_counts[_canonical_pair(winner, loser)] += 1


async def run_tournament(query: _Query, schedule: TournamentSchedule, run: _Pass) -> None:
    """Stage A for one query: random, stratified and adaptive windows (see :mod:`rcp_ndcg.llm.schedule`)."""
    from rcp_ndcg_core.irt._bradley_terry import BradleyTerryEstimator

    units = query.units
    n = len(units)
    if n < 2:
        return
    rng = query_rng(schedule.seed, query.dataset, query.query_id)
    n_random, n_stratified, per_batch = schedule.windows_for(n)
    bt = BradleyTerryEstimator(doc_ids=units, l2_reg=Priors().bt_l2)
    obs_counts: Counter[tuple[str, str]] = Counter()
    seq = 0

    async def phase(windows: list[list[str]], max_tokens: int | None, name: Phase) -> None:
        nonlocal seq
        asked = [window for group in windows for window in ([group, group[::-1]] if schedule.mirror else [group])]
        for judgement in await run.ask_all(query, seq, asked, max_tokens, name):
            _ingest_tournament(judgement, bt, obs_counts)
        seq += len(asked)

    w = min(schedule.window, n)
    max_tokens = run.window_tokens(query, w)
    await phase(
        [[units[i] for i in group] for group in _balanced_groups(n, w, n_random, rng)],
        max_tokens,
        "random",
    )
    bt.fit_lbfgs()
    theta_prelim = bt.get_scores() or {}
    if n_stratified > 0 and theta_prelim:
        await phase(_stratified_groups(units, theta_prelim, w, n_stratified, rng), max_tokens, "stratified")
    bt.fit_lbfgs()
    theta = bt.get_scores() or {}

    aw = min(schedule.adaptive_window, n)
    max_tokens_adaptive = run.window_tokens(query, aw)
    for _batch in range(schedule.adaptive_batches_for(n) if per_batch > 0 else 0):
        boundaries = _compute_boundary_values(theta, obs_counts, top_k=schedule.adaptive_depth)
        windows = _greedy_select_windows(
            theta,
            boundaries,
            aw,
            num_windows=per_batch,
            overlap_discount=schedule.overlap_discount,
        )
        if not windows:
            break
        for judgement in await run.ask_all(query, seq, windows, max_tokens_adaptive, "adaptive"):
            _ingest_tournament(judgement, bt, obs_counts)
        seq += len(windows)
        bt.fit_lbfgs()
        theta = bt.get_scores() or {}


async def run_rubric(query: _Query, schedule: RubricSchedule, run: _Pass) -> None:
    """Stage B for one query: balanced random windows, then windows stratified by a preliminary Rasch ability."""
    from rcp_ndcg_core.irt._rasch import RaschEstimator

    units = query.units
    n = len(units)
    if n < 1:
        return
    rng = query_rng(schedule.seed, query.dataset, query.query_id)
    documents = document_ids_from_chunks(units, query.chunk_mapping)
    rasch = RaschEstimator(doc_ids=documents, num_criteria=len(run.criteria))
    w = min(schedule.window, n)
    max_tokens = run.window_tokens(query, w)
    n_random, n_stratified = schedule.windows_for(len(documents), n_units=n)

    def ingest(judgement: Judgement) -> None:
        if not judgement.valid:
            return
        pooled = max_pool_rubric_window_by_document(
            {p.unit_id: p.criteria for p in judgement.placements if p.criteria}, query.chunk_mapping
        )
        for doc, criteria in pooled.items():
            rasch.add_criteria(doc, criteria)

    random_windows = [[units[i] for i in group] for group in _balanced_groups(n, w, n_random, rng)]
    for judgement in await run.ask_all(query, 0, random_windows, max_tokens, "random"):
        ingest(judgement)
    rasch.fit_lbfgs()
    theta_prelim = rasch.get_scores() or {}
    if n_stratified > 0 and theta_prelim:
        unit_theta = {unit: theta_prelim.get(query.doc_of(unit), 0.0) for unit in units}
        stratified = _stratified_groups(units, unit_theta, w, n_stratified, rng)
        for judgement in await run.ask_all(query, len(random_windows), stratified, max_tokens, "stratified"):
            ingest(judgement)


async def run_planned(query: _Query, windows: Sequence[Sequence[str]], run: _Pass, *, mirror: bool) -> None:
    """Exactly ``windows`` of one query (each also reversed when ``mirror``), outside the schedule's phases.

    The windows are recorded with ``phase`` and ``window_seq`` ``None`` and keyed by their documents in order (and
    the schedule), not by their place in the command: asking the same window again, in any plan or grouping,
    reuses it.
    """
    asked = [list(order) for window in windows for order in ((window, window[::-1]) if mirror else (window,))]
    asked = list({tuple(window): window for window in asked}.values())  # a window listed twice is asked once
    max_tokens = run.window_tokens(query, max(len(window) for window in asked))
    await run.ask_all(query, None, asked, max_tokens, None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def judge(
    dataset: Any,
    candidates: Mapping[str, Sequence[str]] | None,
    judge_cfg: JudgeConfig | JudgeClient,
    *,
    stage: Literal["tournament", "rubric"],
    out: str | Path,
    docs: Mapping[str, Sequence[str]] | None = None,
    schedule: TournamentSchedule | RubricSchedule | None = None,
    preprocessing: Preprocessing | None = None,
    force: bool = False,
    windows: Mapping[str, Sequence[Sequence[str]]] | None = None,
) -> JudgementSet:
    """Judge candidate pools with one stage and write the judgements to the store at ``out``.

    Args:
        dataset: The dataset (:func:`rcp_ndcg.data.load_dataset`): its queries and corpus.
        candidates: ``{query_id: [doc_id, ...]}`` -- each query's pool in rank order
            (document ids); ``None`` judges the dataset's own pools (``Dataset.candidates``, else its
            judged documents).
        judge_cfg: The judge (:class:`JudgeConfig`, or a ready :class:`JudgeClient`).
        stage: ``"tournament"`` (Stage A) or ``"rubric"`` (Stage B, criteria C1..C5).
        out: The judgement store directory (:mod:`rcp_ndcg.llm.store`).
        docs: ``{query_id: [doc_id, ...]}`` -- judge only these documents of each pool, on their own
            (re-judging a subset). The schedule's window counts follow the subset's size, as for any pool
            (its placements per document); a new document is judged against chosen opponents with ``windows=``.
        schedule: The window schedule; ``None`` is the paper's for the corpus's modality.
        preprocessing: The text policy, chunking and image policy to apply; ``None`` applies no load-time
            policy (each window still clips a document to its share of the judge's ``context_tokens``, counted
            with the judge's ``tokenizer``).
        force: Supersede judgements of another identity in ``out`` instead of refusing.
        windows: ``{query_id: [[doc_id, ...], ...]}`` -- ask exactly these windows instead of the schedule's,
            each also reversed when the tournament schedule mirrors (e.g. the windows
            :func:`~rcp_ndcg.calibration.select_opponents` plans for a new document). They are recorded under the
            schedule's identity with ``phase`` ``None``. Not with ``docs``, nor with documents judged in chunks.

    Returns:
        The store's judgements of this stage for the judged queries.

    Raises:
        IdentityError: ``out`` holds judgements of this stage under another identity.
        BackendUnavailableError: the endpoint stayed down; stored answers are kept.
        ConfigError: inconsistent settings (a rubric prompt without criteria, a preprocessing that cuts or chunks
            text without a judge tokenizer), or a remote ``out`` (a store is written locally; mirror it with
            :mod:`rcp_ndcg.runs.mirror`).
    """
    return asyncio.run(
        ajudge(
            dataset,
            candidates,
            judge_cfg,
            stage=stage,
            out=out,
            docs=docs,
            schedule=schedule,
            preprocessing=preprocessing,
            force=force,
            windows=windows,
        )
    )


@dataclass
class _Plan:
    """What a judging pass decides before it asks the judge: its queries, schedule, prompt, family and identity."""

    client: JudgeClient
    preprocessing: Preprocessing
    loaded: TextTruncationCensus
    name: str
    queries: list[_Query]
    schedule: TournamentSchedule | RubricSchedule
    prompt: Prompt
    family: Family
    identity: dict[str, Any]
    #: How the pass named its inputs (the prompt, the schedule's prompt, the tokenizer): runtime, never compared.
    sources: dict[str, Any]
    tokenizer: Any
    #: The dataset's identity key (the digest of ``identity["dataset"]``): what a record id names the corpus by.
    dataset_key: str


def _plan(
    dataset: Any,
    candidates: Mapping[str, Sequence[str]] | None,
    judge_cfg: JudgeConfig | JudgeClient,
    *,
    stage: Literal["tournament", "rubric"],
    out: str | Path,
    docs: Mapping[str, Sequence[str]] | None,
    schedule: TournamentSchedule | RubricSchedule | None,
    preprocessing: Preprocessing | None,
    windows: Mapping[str, Sequence[Sequence[str]]] | None = None,
) -> _Plan:
    """Every check a pass makes before it asks the judge, and what it would write under (nothing is written)."""
    local_dir(out, "a judgement store")
    if stage not in ("tournament", "rubric"):
        raise ConfigError(f"stage must be 'tournament' or 'rubric', got {stage!r}")
    if windows is not None:
        if docs is not None:
            raise ConfigError("pass docs= or windows=, not both: windows name their documents")
        least = 2 if stage == "tournament" else 1
        bad = [window for rows in windows.values() for window in rows if len(set(window)) != len(window)]
        bad += [window for rows in windows.values() for window in rows if len(window) < least]
        empty = sorted(query for query, rows in windows.items() if not rows)
        if bad or empty or not any(windows.values()):
            raise ConfigError(
                f"each window needs at least {least} distinct documents, got {bad[:3] or 'no window'}"
                if bad or not any(windows.values())
                else f"the plan names no windows for {', '.join(map(repr, empty[:3]))}: a query with an empty "
                "window list has nothing to ask and crashes the pass mid-flight",
                hint="plan the windows with select_opponents(..., window=)",
                cli_hint="plan the windows with `rcp-ndcg calibration insert --plan`",
            )
        docs = {query: list(dict.fromkeys(doc for window in rows for doc in window)) for query, rows in windows.items()}
    client = judge_cfg if isinstance(judge_cfg, JudgeClient) else JudgeClient.from_config(judge_cfg)
    effective = _effective_preprocessing(preprocessing, client.config)
    # The pass's effective pixel policy, for the client's engine media check: the probe runs when the pass
    # declares one (and the judge declares an image_processor), with exactly the policy the windows send.
    client.image_policy = effective.image
    tokenizer = _judge_tokenizer(client.config)
    if tokenizer is None and client.config.context_tokens is not None:
        logger.warning(
            "the judge %s declares context_tokens but no tokenizer: the per-window text budget cannot be counted, "
            "so documents are sent whole, and a window over the context is refused by the endpoint and recorded "
            "invalid; set judge.tokenizer to size the windows",
            client.model,
        )
    loaded = TextTruncationCensus()
    loaded.sink = None  # held in memory until the store exists; see _store_census
    name, queries, source = _queries(dataset, candidates, docs, effective, loaded, tokenizer)
    if windows is not None and any(query.chunk_mapping for query in queries):
        raise ConfigError("planned windows show whole documents; these documents are judged in chunks")
    modality = _modality(queries)
    if schedule is None:
        schedule = schedule_for(stage, modality)
    expected = TournamentSchedule if stage == "tournament" else RubricSchedule
    if not isinstance(schedule, expected):
        raise ConfigError(f"stage {stage!r} takes a {expected.__name__}, got {type(schedule).__name__}")
    shipped = shipped_prompt_name(stage, modality)
    if schedule.prompt in PROMPT_FILES and schedule.prompt != shipped:
        raise ConfigError(
            f"the schedule names the prompt {schedule.prompt!r}, but the candidates are {modality} documents, "
            f"which the shipped {shipped!r} prompt is written for; a rubric written for another kind of document "
            "is answered anyway and means nothing",
            hint=f"leave schedule.prompt unset (the {shipped!r} prompt), or check the corpus",
        )
    prompt = load_prompt(schedule.prompt or shipped)
    if stage == "rubric" and not prompt.criteria:
        raise ConfigError(f"the rubric prompt {prompt.name!r} names no criteria C1..Cn")
    family = Family(
        stage=stage,
        judge_model=client.model,
        judge_revision=client.config.revision,
        prompt_hash=prompt.sha256,
        criteria=prompt.criteria if stage == "rubric" else (),
        parse_version=PARSE_VERSION,
        decoding=client.config.decoding,
        preprocessing=effective.key,
        tokenizer=tokenizer.sha256 if tokenizer is not None else None,
        # The declared-CONTENT judge settings: the family digest carries each only when it differs from the
        # default (the schema's rule), so a family judged under the defaults keeps its key and one judged under
        # a declared value never pools with it, whatever the store gate does.
        temperature=client.config.temperature,
        max_output_tokens=client.config.max_output_tokens,
        context_tokens=client.config.context_tokens,
        extra_body=client.config.extra_body or None,
        api=client.config.api,
    )
    dataset_identity = _dataset_identity(name, source, rows=dataset if source is None else None)
    # The dataset's identity key: what a record id names the corpus by, so two corpora that share query and
    # document ids never share a record id (across stores, and in a merge).
    dataset_key = short(hash_payload(dataset_identity), 16)
    # Content only: the prompt and the tokenizer enter by their SHA-256 (the family's prompt_hash, and the
    # tokenizer's below), never by the name or path they were given, which is recorded beside it (sources).
    identity = {
        "stage": stage,
        "family": family.model_dump(mode="json"),
        "judge": client.config.identity(),
        "schedule": schedule.model_dump(mode="json", exclude={"prompt"}),
        # The penalty of the tournament's live Bradley-Terry fit, which chose its adaptive windows.
        **({"bt_l2": Priors().bt_l2} if stage == "tournament" else {}),
        "dataset": dataset_identity,
        "preprocessing": {
            **effective.model_dump(mode="json"),
            "tokenizer": {"sha256": tokenizer.sha256} if tokenizer is not None else None,
        },
    }
    sources = {
        "prompt": prompt.name,
        "schedule_prompt": schedule.prompt,
        "tokenizer": tokenizer.name if tokenizer is not None else None,
    }
    return _Plan(
        client, effective, loaded, name, queries, schedule, prompt, family, identity, sources, tokenizer, dataset_key
    )


def preflight(
    dataset: Any,
    candidates: Mapping[str, Sequence[str]] | None,
    judge_cfg: JudgeConfig | JudgeClient,
    *,
    stage: Literal["tournament", "rubric"],
    out: str | Path,
    docs: Mapping[str, Sequence[str]] | None = None,
    schedule: TournamentSchedule | RubricSchedule | None = None,
    preprocessing: Preprocessing | None = None,
    force: bool = False,
    windows: Mapping[str, Sequence[Sequence[str]]] | None = None,
) -> None:
    """Make every check :func:`judge` with the same arguments makes before it asks the judge; write nothing.

    What ``--estimate`` and ``--dry-run`` run, so they give the refusal the real command would give.

    Raises:
        IdentityError: ``out`` holds judgements of this stage under another identity (and ``force`` is false).
        ConfigError: as :func:`judge`: a prompt without its slots or criteria, a schedule of the other stage, a
            remote ``out``.
        DataError: the candidates or documents do not join the dataset.
    """
    plan = _plan(
        dataset, candidates, judge_cfg, stage=stage, out=out, docs=docs, schedule=schedule,
        preprocessing=preprocessing, windows=windows,
    )  # fmt: skip
    JudgementStore(out).check(stage, plan.identity, force=force)


async def ajudge(
    dataset: Any,
    candidates: Mapping[str, Sequence[str]] | None,
    judge_cfg: JudgeConfig | JudgeClient,
    *,
    stage: Literal["tournament", "rubric"],
    out: str | Path,
    docs: Mapping[str, Sequence[str]] | None = None,
    schedule: TournamentSchedule | RubricSchedule | None = None,
    preprocessing: Preprocessing | None = None,
    force: bool = False,
    windows: Mapping[str, Sequence[Sequence[str]]] | None = None,
) -> JudgementSet:
    """:func:`judge` inside a running event loop."""
    plan = _plan(
        dataset, candidates, judge_cfg, stage=stage, out=out, docs=docs, schedule=schedule,
        preprocessing=preprocessing, windows=windows,
    )  # fmt: skip
    client, queries, prompt = plan.client, plan.queries, plan.prompt
    store = JudgementStore(out)
    store.claim(stage, plan.identity, plan.family, force=force, sources=plan.sources)
    # What the endpoint says it serves: runtime information beside the identity, never part of it.
    store.note_engines(stage, await client.probe())
    store.keep_prompt(prompt.text)
    census = _store_census(store.root, plan.loaded)
    run = _Pass(
        stage=stage,
        client=client,
        prompt=prompt,
        family=plan.family,
        store=store,
        existing=store.records(stage),
        preprocessing=plan.preprocessing,
        census=census,
        media_census=MediaCensus(sink=store.root / PREPROCESSING_RECORD),
        tokenizer=plan.tokenizer,
        schedule_key=schedule_key(plan.schedule),
        dataset_key=plan.dataset_key,
    )
    logger.info(
        "judging %d queries of %s: %s with %s (%s), store %s",
        len(queries),
        plan.name,
        stage,
        client.model,
        prompt.name,
        out,
    )
    if windows is not None:
        mirror = isinstance(plan.schedule, TournamentSchedule) and plan.schedule.mirror
        jobs = [run_planned(query, windows[query.query_id], run, mirror=mirror) for query in queries]
    else:
        worker = run_tournament if stage == "tournament" else run_rubric
        jobs = [worker(query, plan.schedule, run) for query in queries]  # type: ignore[arg-type]
    outcomes = await asyncio.gather(*jobs, return_exceptions=True)
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    store.note_engines(stage, client.engines)  # now with the fingerprints the answers carried
    logger.info("judged %s: %d windows asked, %d reused from the store", stage, run.asked, run.reused)
    for failure in failures:
        raise failure
    in_scope = {(query.dataset, query.query_id) for query in queries}
    return store.read(stage).select(lambda judgement: (judgement.dataset, judgement.query_id) in in_scope)


__all__ = [
    "CHAT_TEMPLATE_TOKENS",
    "MAX_ATTEMPTS",
    "PREPROCESSING_RECORD",
    "WindowAnswer",
    "ajudge",
    "judge",
    "media_marker_tokens",
    "parse_window",
    "preflight",
    "prompt_overhead_tokens",
    "run_planned",
    "run_rubric",
    "run_tournament",
    "window_record",
    "window_tokens",
]
