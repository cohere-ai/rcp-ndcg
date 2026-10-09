"""The text census: where a truncation becomes observable.

One home for the cut record and the census that collects it. Every cut -- at load
(:class:`~rcp_ndcg.data.text_policy.TextPolicy`'s ``doc_policy``), in a judging window
(``window_budget``), or in a served request (:func:`~rcp_ndcg.data.text_budget.fit`'s
``text_budget``) -- is recorded here, never silently. The file I/O of a census sink (the append, the
read, the torn-tail repair) is :mod:`rcp_ndcg.storage.census`; this module holds the types.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rcp_ndcg.errors import DataError
from rcp_ndcg.storage.census import append_census_rows
from rcp_ndcg.support.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

CutCause = Literal["budget_cut", "query_share", "document_share"]
"""Why a role client changed what it sends (a ``text_budget`` row's ``cause``): see :class:`TextCutRecord`."""

CUT_CAUSES: tuple[CutCause, ...] = ("budget_cut", "query_share", "document_share")
"""Every :data:`CutCause`, in the order the client applies them (the budget binds last)."""


class TextCutRecord:
    """One observed cut: tokens (of the judge's tokenizer) and characters before and after.

    Frozen so a recorded event cannot be rewritten. The limits are in tokens; the characters are information.
    A ``text_budget`` record adds what the judge's mechanisms do not have: which request ``shape`` was cut,
    who computed the budget (``budget_source``: the declared tokenizer, or a hosted vendor's documented limit),
    and -- on a chunked document -- the ``aggregation`` its chunk scores pool by. A role client's cut also
    names why it changed what it sends (``cause``) and how large the uncut and the kept request are as the
    engine would read them (``original_request_tokens``, ``kept_request_tokens``: the frame, its specials,
    the content and the reserved media) -- ``original_tokens`` counts the content alone, so a request whose
    frame pushed it over the budget has a content count under it. They are ``None`` on the judge's rows and on
    a vendor's budget row (a recorded limit, not a cut), and are then left out of :meth:`as_row`. A role
    client's :class:`ProcessingRecord` of the row reads them.

    ``cause`` is one of :data:`CUT_CAUSES`: ``budget_cut`` (the uncut request exceeded its shape's budget),
    ``query_share`` (the rerank client settled the shared query to its declared ``query_max_tokens``, also
    inside a pair under the budget) or ``document_share`` (a document exceeded the declared
    ``document_max_tokens`` and was cut to it while the pair fitted the budget).
    """

    __slots__ = (
        "aggregation",
        "budget_source",
        "budget_tokens",
        "cause",
        "corpus",
        "doc_id",
        "kept_chars",
        "kept_tokens",
        "mechanism",
        "kept_request_tokens",
        "original_chars",
        "original_request_tokens",
        "original_tokens",
        "query_id",
        "shape",
    )

    cause: CutCause | None
    original_request_tokens: int | None
    kept_request_tokens: int | None

    def __init__(
        self,
        *,
        corpus: str,
        doc_id: str,
        original_chars: int,
        kept_chars: int,
        original_tokens: int,
        kept_tokens: int,
        mechanism: str,
        query_id: str | None = None,
        budget_source: str | None = None,
        aggregation: str | None = None,
        shape: str | None = None,
        budget_tokens: int | None = None,
        cause: CutCause | None = None,
        original_request_tokens: int | None = None,
        kept_request_tokens: int | None = None,
    ) -> None:
        self.corpus = corpus
        self.doc_id = doc_id
        self.original_chars = original_chars
        self.kept_chars = kept_chars
        self.original_tokens = original_tokens
        self.kept_tokens = kept_tokens
        self.mechanism = mechanism
        self.query_id = query_id
        self.budget_source = budget_source
        self.aggregation = aggregation
        self.shape = shape
        self.budget_tokens = budget_tokens
        self.cause = cause
        self.original_request_tokens = original_request_tokens
        self.kept_request_tokens = kept_request_tokens

    def as_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "corpus": self.corpus,
            "doc_id": self.doc_id,
            "original_tokens": self.original_tokens,
            "kept_tokens": self.kept_tokens,
            "tokens_lost": self.original_tokens - self.kept_tokens,
            "original_chars": self.original_chars,
            "kept_chars": self.kept_chars,
            "chars_lost": self.original_chars - self.kept_chars,
            "mechanism": self.mechanism,
            "query_id": self.query_id,
        }
        # The text_budget rows carry the budget's own facts; the judge's rows stay byte-for-byte as they were.
        if self.budget_source is not None:
            row["budget_source"] = self.budget_source
        if self.budget_tokens is not None:
            row["budget_tokens"] = self.budget_tokens
        if self.aggregation is not None:
            row["aggregation"] = self.aggregation
        if self.shape is not None:
            row["shape"] = self.shape
        if self.cause is not None:
            row["cause"] = self.cause
        if self.original_request_tokens is not None:
            row["original_request_tokens"] = self.original_request_tokens
        if self.kept_request_tokens is not None:
            row["kept_request_tokens"] = self.kept_request_tokens
        return row

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TextCutRecord) and self.as_row() == other.as_row()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TextCutRecord({self.as_row()!r})"


class TextTruncationCensus:
    """Where a truncation becomes observable.

    Records, for every cut: count, document id, tokens and characters before and after.
    Never a config field, so attaching it cannot move a dataset's identity: the
    run manifest records the *policy*, the census records the *cuts*.

    Three mechanisms, three referents, never merged:

    * ``doc_policy`` -- a document shortened once at load time, per corpus.
      Referent: documents.
    * ``window_budget`` -- a document presented to a judge shorter than it was
      loaded, per (query, window) presentation.  A document cut in five windows
      is five records.  Referent: document presentations.
    * ``text_budget`` -- a served role's request fitted into a model's input
      budget (mechanism of :func:`fit`): content cut, chunked or refused under
      a declared :class:`TextBudget`.  Referent: requests (a chunked input is
      one row per chunk, and every row names the aggregation its scores pool
      by; each row's ``original_tokens`` is the whole input's, so a chunked
      document's rows repeat it -- summing the column overcounts the input).
      A hosted vendor profile without a tokenizer records one row per
      (corpus, budget) per census (``doc_id`` ``<budget>``) naming its
      documented limit as the effective budget, and cuts nothing.

    When *sink* is set, every record is also appended to it as one JSON line.
    """

    DOC_POLICY = "doc_policy"
    WINDOW_BUDGET = "window_budget"
    TEXT_BUDGET = "text_budget"
    MECHANISMS = (DOC_POLICY, WINDOW_BUDGET, TEXT_BUDGET)

    def __init__(self, *, sink: str | Path | None = None) -> None:
        self._cuts: list[TextCutRecord] = []
        self._budget_rows: set[tuple[str, int]] = set()
        self.sink = sink

    def _append(self, row: dict[str, Any]) -> None:
        if self.sink is None:
            return
        try:
            # The one census append: the torn tail cut and the line written under the sink's writer lock, so a
            # peer's cut never truncates this writer's in-flight line.
            append_census_rows(self.sink, [row])
        except OSError:
            # A provenance guarantee that degrades silently is worth a warning a person can see: a resumed pass
            # reads this file to decide which cuts are already on record.
            logger.warning("text census sink %s is not writable; cuts stay in memory only", self.sink, exc_info=True)

    def record(
        self,
        *,
        corpus: str,
        doc_id: str,
        original_chars: int,
        kept_chars: int,
        original_tokens: int,
        kept_tokens: int,
        mechanism: str,
        query_id: str | None = None,
        budget_source: str | None = None,
        aggregation: str | None = None,
        shape: str | None = None,
        budget_tokens: int | None = None,
        cause: CutCause | None = None,
        original_request_tokens: int | None = None,
        kept_request_tokens: int | None = None,
    ) -> TextCutRecord:
        if mechanism not in self.MECHANISMS:
            raise DataError(
                f"mechanism must be one of {self.MECHANISMS}, got {mechanism!r}",
                hint="record under one of the census' three mechanisms (doc_policy, window_budget, text_budget)",
            )
        cut = TextCutRecord(
            corpus=corpus,
            doc_id=doc_id,
            original_chars=original_chars,
            kept_chars=kept_chars,
            original_tokens=original_tokens,
            kept_tokens=kept_tokens,
            mechanism=mechanism,
            query_id=query_id,
            budget_source=budget_source,
            aggregation=aggregation,
            shape=shape,
            budget_tokens=budget_tokens,
            cause=cause,
            original_request_tokens=original_request_tokens,
            kept_request_tokens=kept_request_tokens,
        )
        self._cuts.append(cut)
        self._append(cut.as_row())
        return cut

    def cuts(self, *, mechanism: str | None = None) -> list[TextCutRecord]:
        """The recorded cuts, in order; only those of one ``mechanism`` when given."""
        return [cut for cut in self._cuts if mechanism is None or cut.mechanism == mechanism]

    def __len__(self) -> int:
        return len(self._cuts)


# ---------------------------------------------------------------------------
# Chunking: the producer and the aggregation
# ---------------------------------------------------------------------------


__all__ = [
    "CUT_CAUSES",
    "CutCause",
    "TextCutRecord",
    "TextTruncationCensus",
]
