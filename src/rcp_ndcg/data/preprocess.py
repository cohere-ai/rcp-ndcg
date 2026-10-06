"""Preprocessing: the declared text policy, chunking, the text budget, and the record of every cut.

One home for what happens to text before a model reads it:

* :class:`TextPolicy` -- the per-document text policy (keep, truncate with a
  census record, chunk, or fail), applied by :func:`apply_text_policy`, the only
  place a judged document is shortened before its window;
* :class:`ChunkPolicy` -- how a long document is split into overlapping
  chunks (in tokens), the chunk producer (:func:`chunk_ranking_example`,
  chunk ids ``<doc_id>#<k>``) and the aggregation of chunk judgements back onto
  their document (maximum per criterion within one window, maximum score);
* :class:`TextBudget` and :func:`fit` -- the one mechanism every served role (an embedder, a
  reranker) uses to fit its requests into a model's input: the template declared as data
  (:mod:`rcp_ndcg.data.templates`), the fixed overhead measured once per (template, shape), only the
  content spans cut with :func:`token_prefix`, the template re-attached, chunking that gives every chunk
  the full template, and every cut recorded;
* :class:`TextTruncationCensus` -- every cut, at load (``doc_policy``) and in a
  judging window (``window_budget``), or in a served request (``text_budget``);
* :class:`Preprocessing` -- the effective policy of a judging pass, whose
  :attr:`~Preprocessing.key` is part of the judging identity and the family.

Nothing truncates silently: a cut is either declared policy (and counted) or a per-window budget cut (and
counted), or a declared :class:`TextBudget` cut (and counted).

Limits are in tokens of the judge's or the role's tokenizer (:mod:`rcp_ndcg.data.tokenizer`), and every cut is made
at a token boundary of the original text, located with the tokenizer's offset mapping: a truncated document is a
verbatim prefix, and a chunk a verbatim slice, of the document. There is no character fallback: a policy that cuts
refuses to run without a tokenizer.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rcp_ndcg_core._records import RankingExample
from rcp_ndcg_core.content import Content

from rcp_ndcg.data.resolution import ImagePolicy, VideoPolicy
from rcp_ndcg.data.templates import RequestShape, TemplateSpec
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.support.identity import FieldRole, identity_payload
from rcp_ndcg.support.logging import get_logger

if TYPE_CHECKING:
    from rcp_ndcg.data.tokenizer import TextTokenizer

logger = get_logger(__name__)

DEFAULT_MAX_TOKENS = 20000
"""The default cap of a ``truncate`` or ``fail`` text policy, in tokens of the judge's tokenizer."""

OnOverflow = Literal["keep", "truncate", "chunk", "fail"]
"""What a :class:`TextPolicy` does with a document longer than its ``max_tokens``.

``keep``
    Keep the document whole.  The judge-side per-window budget still cuts it per
    window when the judge has a tokenizer and a context; nothing is lost before that.
    The only policy that needs no tokenizer.
``truncate``
    Cap at ``max_tokens`` (default :data:`DEFAULT_MAX_TOKENS`), recording every cut --
    id, tokens and characters lost -- in the census.
``chunk``
    Keep the document whole, then split every document longer than the policy's
    :class:`ChunkPolicy` ``max_tokens`` into overlapping chunks where it meets its
    query (:func:`chunk_ranking_example`).  The judge sees every chunk;
    ``chunk_mapping`` carries each chunk back to its document and the judgements
    are max-pooled per document.  Nothing is cut.
``fail``
    Refuse any document over the cap -- for corpora where losing 60% of a document
    invalidates the judgement.
"""


class DocumentOverCapError(DataError):
    """A document exceeded the cap of a ``fail`` text policy."""


class ChunkPolicy(BaseModel):
    """How a long document is split for judging, in tokens of the judge's tokenizer.

    A document longer than ``max_tokens`` is split into consecutive chunks of at
    most ``max_tokens`` tokens, each repeating the last ``overlap_tokens``
    tokens of the previous one, so evidence straddling a boundary is seen
    whole by one chunk. A chunk is a verbatim slice of the document between
    token boundaries. Chunk ``k`` of document ``d`` has the id ``d#k`` (``k``
    from 0). The judge sees every chunk, and a document's judgements are pooled
    by ``aggregate``: within one window, a document passes a criterion when any
    of its chunks does, and its tournament score is its best chunk's.

    Attributes:
        max_tokens: The largest chunk, and the length above which a document is split (tokens).
        overlap_tokens: Tokens shared by consecutive chunks (``< max_tokens``).
        aggregate: How chunk judgements pool onto their document (``"max"``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "max_tokens": FieldRole.CONTENT,
        "overlap_tokens": FieldRole.CONTENT,
        "aggregate": FieldRole.CONTENT,
    }

    max_tokens: int = Field(ge=1)
    overlap_tokens: int = Field(ge=0)
    aggregate: Literal["max"] = "max"

    @model_validator(mode="after")
    def _chunks_advance(self) -> ChunkPolicy:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be smaller than max_tokens ({self.max_tokens}): "
                "consecutive chunks must advance"
            )
        return self

    @property
    def descriptor(self) -> str:
        """Human-readable one-liner, e.g. ``chunk:4000/250:max`` (tokens)."""
        return f"chunk:{self.max_tokens}/{self.overlap_tokens}:{self.aggregate}"


class TextPolicy(BaseModel):
    """What happens to a document's text before a judge reads it (see :data:`OnOverflow`).

    Frozen: a policy is config, and a mutable config object is how "which cap
    applied" stops being answerable after the fact.

    Attributes:
        on_overflow: ``keep``, ``truncate``, ``chunk`` or ``fail``.
        max_tokens: The cap, in tokens of the judge's tokenizer (``judge.tokenizer``). ``truncate`` and ``fail``
            default it to :data:`DEFAULT_MAX_TOKENS`; ``chunk`` takes it from ``chunk.max_tokens``; ``keep`` has none.
        chunk: ``chunk`` only: the chunk geometry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    on_overflow: OnOverflow = "keep"
    max_tokens: int | None = Field(default=None, ge=1)
    chunk: ChunkPolicy | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _fields_match_on_overflow(self) -> TextPolicy:
        if self.on_overflow == "chunk":
            if self.chunk is None:
                raise ValueError(
                    "text policy 'chunk' needs a chunk geometry: "
                    "{on_overflow: chunk, chunk: {max_tokens: ..., overlap_tokens: ...}}"
                )
            if self.max_tokens is not None and self.max_tokens != self.chunk.max_tokens:
                raise ValueError(
                    f"text policy 'chunk' splits at chunk.max_tokens ({self.chunk.max_tokens}); "
                    f"max_tokens ({self.max_tokens}) must be unset or equal"
                )
            object.__setattr__(self, "max_tokens", self.chunk.max_tokens)
            return self
        if self.chunk is not None:
            raise ValueError(f"a chunk geometry applies to text policy 'chunk' only, not {self.on_overflow!r}")
        if self.on_overflow == "keep":
            if self.max_tokens is not None:
                raise ValueError("text policy 'keep' takes no max_tokens: keeping the full text is the point")
        elif self.max_tokens is None:
            object.__setattr__(self, "max_tokens", DEFAULT_MAX_TOKENS)
        return self


DEFAULT_TEXT_POLICY = TextPolicy()
"""The policy in effect when none is declared: ``keep``. A document is shortened only under a declared policy, and
every such cut is recorded (:class:`TextTruncationCensus`)."""


#: The policies under which a document is returned whole.
_KEPT_WHOLE = frozenset({"keep", "chunk"})

_WARNED: set[tuple[str, str]] = set()


def needs_tokenizer(preprocessing: Preprocessing) -> bool:
    """Whether ``preprocessing`` measures text: a text policy other than ``keep``, or a chunk geometry."""
    return preprocessing.text.on_overflow != "keep" or preprocessing.chunk is not None


def require_tokenizer(tokenizer: TextTokenizer | None, what: str) -> TextTokenizer:
    """``tokenizer``, or the :class:`~rcp_ndcg.errors.ConfigError` of a token limit without one.

    Args:
        tokenizer: The judge's tokenizer, or ``None`` when the judge names none.
        what: What needs it, for the message (e.g. ``"text policy 'truncate'"``).
    """
    if tokenizer is None:
        raise ConfigError(
            f"{what} counts tokens with the judge's tokenizer, and the judge names none; "
            "text is never cut by characters",
            hint="set judge.tokenizer to a Hugging Face repo id (optionally @revision) or a tokenizer.json path, "
            "e.g. --set judge.tokenizer=Qwen/Qwen3.5-397B-A17B-FP8; or keep documents whole (on_overflow: keep)",
        )
    return tokenizer


def token_prefix(
    text: str,
    max_tokens: int,
    tokenizer: TextTokenizer,
    *,
    rendered: Callable[[str], str] | None = None,
    add_special_tokens: bool = False,
) -> str:
    """A prefix of ``text`` that ends at one of its first ``max_tokens`` token boundaries and counts at most
    ``max_tokens``: the longest such prefix the search finds.

    The cut is located with the tokenizer's offset mapping on the original text, so the result is a verbatim prefix
    of ``text`` (never tokens decoded back to text). A candidate is counted as the judge will read it --
    ``rendered(prefix)`` when given, e.g. the prompt's escaping -- because a cut word or an escaped character can
    tokenize to more tokens than it had in place: the cut starts at the ``max_tokens``-th boundary and moves back
    until the prefix fits (a galloping, then a binary search over token boundaries). A longer prefix that
    re-tokenizes into fewer tokens than it has boundaries is not sought, so the cut can stop a few characters short
    of the longest prefix that would fit. With ``add_special_tokens`` the count is the engine's (the tokenizer's
    post-processor tokens included), which is how a served request's budget is verified against what the engine
    actually reads.

    Args:
        text: The document text.
        max_tokens: The budget, in tokens.
        tokenizer: The judge's tokenizer.
        rendered: How the text appears to the judge, when that is not the text itself.
        add_special_tokens: Count as the engine does, the post-processor's tokens included.

    Returns:
        ``text`` itself when it fits; else its prefix (possibly empty).
    """

    def count(piece: str) -> int:
        return tokenizer.count(
            rendered(piece) if rendered is not None else piece, add_special_tokens=add_special_tokens
        )

    if count(text) <= max_tokens:
        return text
    offsets = tokenizer.offsets(text)

    def prefix(tokens: int) -> str:
        return text[: offsets[tokens - 1][1]] if tokens > 0 else ""

    def fits(tokens: int) -> bool:
        return count(prefix(tokens)) <= max_tokens

    # ``fitting`` fits (the empty prefix always does); ``over`` does not, or is past the budget.
    over = min(max_tokens, len(offsets))
    if fits(over):
        return prefix(over)
    fitting, step = over - 1, 1
    while fitting > 0 and not fits(fitting):
        over, fitting, step = fitting, max(fitting - step, 0), step * 2
    while over - fitting > 1:
        middle = (over + fitting) // 2
        fitting, over = (middle, over) if fits(middle) else (fitting, middle)
    return prefix(fitting)


def _warn_first_cut(mechanism: str, corpus: str, doc_id: str, original_tokens: int, kept_tokens: int) -> None:
    """One warning per (mechanism, corpus), process-wide.

    Lives here rather than on the census so a cut warns even when the caller
    passed no census (a direct :func:`apply_text_policy` user): a cut
    nobody can see is the defect this module exists to remove.
    """
    key = (mechanism, corpus)
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(
        "text truncation (%s) on corpus %s: first cut doc_id=%s %d -> %d tokens; "
        "further cuts are counted in the census, not logged per document",
        mechanism,
        corpus or "<unnamed>",
        doc_id,
        original_tokens,
        kept_tokens,
    )


def apply_text_policy(
    text: str,
    *,
    doc_id: str,
    policy: TextPolicy | None = None,
    tokenizer: TextTokenizer | None = None,
    corpus: str = "",
    census: TextTruncationCensus | None = None,
) -> str:
    """The one place a document's text is shortened before its window.

    Which cap applies depends on the declared policy and on nothing else.  ``None`` policy means the
    declared default, never "no policy": an undeclared corpus is a
    ``keep`` corpus, visibly, not an uncapped one by accident.  A
    ``chunk`` policy keeps the text whole here: its documents are split, not
    shortened, by :func:`chunk_ranking_example`. A cut is a verbatim prefix ending at a token boundary
    (:func:`token_prefix`).

    Args:
        text: The document text.
        doc_id: The document's id, for the census and the messages.
        policy: The declared text policy (``None``: :data:`DEFAULT_TEXT_POLICY`).
        tokenizer: The judge's tokenizer; required by ``truncate`` and ``fail``.
        corpus: The corpus name, for the census.
        census: Where the cut is recorded.

    Returns:
        The text the judge is shown.

    Raises:
        ConfigError: a ``truncate`` or ``fail`` policy without a tokenizer.
        DocumentOverCapError: a ``fail`` policy and a document over its cap.
    """
    effective = policy if policy is not None else DEFAULT_TEXT_POLICY
    if effective.on_overflow in _KEPT_WHOLE:
        return text
    tokenizer = require_tokenizer(tokenizer, f"text policy {effective.on_overflow!r}")
    cap = effective.max_tokens or 0
    original = tokenizer.count(text)
    if original <= cap:
        return text
    if effective.on_overflow == "fail":
        raise DocumentOverCapError(
            f"document {doc_id!r} in corpus {corpus!r} is {original:,} tokens, over the declared "
            f"text policy 'fail' cap of {cap:,}. Losing this much text "
            "invalidates the judgement; split the corpus, raise the cap explicitly, or declare "
            "'truncate' to accept the cut."
        )
    cut = token_prefix(text, cap, tokenizer)
    kept = tokenizer.count(cut)
    _warn_first_cut(TextTruncationCensus.DOC_POLICY, corpus, doc_id, original, kept)
    if census is not None:
        census.record(
            corpus=corpus,
            doc_id=doc_id,
            original_chars=len(text),
            kept_chars=len(cut),
            original_tokens=original,
            kept_tokens=kept,
            mechanism=TextTruncationCensus.DOC_POLICY,
        )
    return cut


class TextCutRecord:
    """One observed cut: tokens (of the judge's tokenizer) and characters before and after.

    Frozen so a recorded event cannot be rewritten. The limits are in tokens; the characters are information.
    A ``text_budget`` record adds what the judge's mechanisms do not have: which request ``shape`` was cut,
    who computed the budget (``budget_source``: the declared tokenizer, or a hosted vendor's documented limit),
    and -- on a chunked document -- the ``aggregation`` its chunk scores pool by.
    """

    __slots__ = (
        "aggregation",
        "budget_source",
        "budget_tokens",
        "corpus",
        "doc_id",
        "kept_chars",
        "kept_tokens",
        "mechanism",
        "original_chars",
        "original_tokens",
        "query_id",
        "shape",
    )

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

    Two mechanisms, two referents, never merged:

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
            with open(self.sink, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        except OSError:
            logger.debug("text census sink %s is not writable; cuts stay in memory only", self.sink, exc_info=True)

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
    ) -> TextCutRecord:
        if mechanism not in self.MECHANISMS:
            raise ValueError(f"mechanism must be one of {self.MECHANISMS}, got {mechanism!r}")
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

CHUNK_ID_SEPARATOR = "#"
"""Chunk ids are ``<doc_id>#<k>`` (``k`` from 0). The id is a label only:
``chunk_mapping`` is the authority for which document a chunk belongs to."""


def split_into_chunks(text: str, policy: ChunkPolicy, tokenizer: TextTokenizer) -> list[str]:
    """Split *text* into consecutive, overlapping chunks of at most ``policy.max_tokens`` tokens.

    The chunks are windows over the document's own tokens, located with the tokenizer's offset mapping, so each
    chunk is a verbatim slice of *text*. Chunk ``k + 1`` starts ``policy.overlap_tokens`` document tokens before
    chunk ``k`` ends, and with no overlap where chunk ``k`` ends, so the text between two tokens (whitespace, a
    character the tokenizer normalises away) belongs to a chunk too; the first chunk starts at the start of the text
    and the last ends at its end, so the chunks cover the text. A slice that re-tokenizes to more than ``max_tokens``
    on its own (a word cut at its edge) ends a token earlier until it fits. A chunk is at least one document token,
    so with a cap of a few tokens one token whose text re-tokenizes longer (a multi-byte character split across
    byte-level tokens) makes a chunk over the cap. One boundary caveat, declared here: a multi-byte character that
    a token boundary splits in two repeats -- its leading bytes end one chunk and the whole character opens the
    next -- so an ASCII-free corpus can see a character twice at a chunk edge (duplication only, never loss).

    Returns:
        The chunk texts in document order; ``[text]`` when *text* fits in one chunk.
    """
    offsets = tokenizer.offsets(text)
    n = len(offsets)
    if n <= policy.max_tokens:
        return [text]
    chunks: list[str] = []
    start = covered = 0
    while True:
        begin = min(offsets[start][0], covered)
        end = min(start + policy.max_tokens, n)
        while True:
            piece = text[begin : len(text) if end == n else offsets[end - 1][1]]
            excess = tokenizer.count(piece) - policy.max_tokens
            if excess <= 0 or end - start <= 1:
                break
            end = max(end - excess, start + 1)
        chunks.append(piece)
        if end == n:
            return chunks
        covered = offsets[end - 1][1]
        start = max(end - policy.overlap_tokens, start + 1)


def chunk_ranking_example(example: RankingExample, policy: ChunkPolicy, tokenizer: TextTokenizer) -> RankingExample:
    """One query's candidates with every long text document replaced by its chunks.

    Every text document longer than ``policy.max_tokens`` tokens is replaced, in place in
    the candidate order, by its chunks ``<doc_id>#<k>`` (:func:`split_into_chunks`); shorter documents keep
    their id and text. The result carries a ``chunk_mapping`` covering every
    candidate (chunk -> document, an unsplit document -> itself), and every
    document-aligned field (``scores``) is copied onto each of
    a document's chunks; ``qrels`` stay keyed by document.

    A document that carries media is never split (its text is not what a vision
    judge reads). An example with nothing over the cap is returned unchanged.

    Raises:
        DataError: the example has no document bodies, already carries a
            ``chunk_mapping``, or a chunk id collides with another candidate's id.
    """
    if example.chunk_mapping:
        raise DataError(f"query {example.id!r} already carries a chunk_mapping; chunking it again is refused")
    if not example.has_bodies:
        raise DataError(f"query {example.id!r} has no document bodies; hydrate it before chunking")
    contents = example.contents
    texts = example.docs or []
    splits = {
        index: pieces
        for index, text in enumerate(texts)
        if (contents is None or not contents[index].has_media)
        and len(pieces := split_into_chunks(text, policy, tokenizer)) > 1
    }
    if not splits:
        return example

    positions: list[int] = []
    ids: list[str] = []
    bodies: list[str] = []
    mapping: dict[str, str] = {}
    for index, doc_id in enumerate(example.doc_ids):
        pieces = splits.get(index, [texts[index]])
        chunk_ids = [f"{doc_id}{CHUNK_ID_SEPARATOR}{k}" for k in range(len(pieces))] if index in splits else [doc_id]
        for chunk_id, piece in zip(chunk_ids, pieces, strict=True):
            positions.append(index)
            ids.append(chunk_id)
            bodies.append(piece)
            mapping[chunk_id] = doc_id
    if len(set(ids)) != len(ids):
        clashes = sorted({chunk_id for chunk_id in ids if ids.count(chunk_id) > 1})
        raise DataError(f"query {example.id!r}: chunk ids collide with candidate ids: {clashes}")

    fields = example.model_dump(by_alias=False, exclude_none=True)
    fields.update(doc_ids=ids, docs=bodies, chunk_mapping=mapping)
    fields.pop("contents", None)
    if contents is not None:
        fields["contents"] = [
            Content.from_text(body) if positions[k] in splits else contents[positions[k]]
            for k, body in enumerate(bodies)
        ]
    if example.scores is not None:
        fields["scores"] = [example.scores[index] for index in positions]
        # The chunked row keeps the candidate order: scores repeat per chunk, so a
        # re-sort would reorder nothing, and the row is built in order already.
    return RankingExample.model_validate(fields)


def document_id_for_chunk(chunk_id: str, chunk_mapping: Mapping[str, str] | None) -> str:
    """The document a chunk belongs to (the id itself when there is no mapping).

    Raises:
        DataError: a non-empty mapping that does not cover *chunk_id*, or maps it to an empty id.
    """
    if not chunk_mapping:
        return chunk_id
    try:
        document_id = chunk_mapping[chunk_id]
    except KeyError as exc:
        raise DataError(f"chunk_mapping has no document ID for chunk {chunk_id!r}") from exc
    if not document_id:
        raise DataError(f"chunk_mapping maps chunk {chunk_id!r} to an empty document ID")
    return document_id


def document_ids_from_chunks(chunk_ids: Sequence[str], chunk_mapping: Mapping[str, str] | None) -> list[str]:
    """Unique document ids in first-chunk order."""
    return list(dict.fromkeys(document_id_for_chunk(chunk_id, chunk_mapping) for chunk_id in chunk_ids))


def max_pool_scores_by_document(
    chunk_scores: Mapping[str, float], chunk_mapping: Mapping[str, str] | None
) -> dict[str, float]:
    """A document's score is its best chunk's; documents in first-chunk order."""
    pooled: dict[str, float] = {}
    for chunk_id, score in chunk_scores.items():
        document_id = document_id_for_chunk(chunk_id, chunk_mapping)
        if document_id not in pooled or score > pooled[document_id]:
            pooled[document_id] = score
    return pooled


def max_pool_rubric_window_by_document(
    chunk_criteria: Mapping[str, Mapping[str, int]], chunk_mapping: Mapping[str, str] | None
) -> dict[str, dict[str, int]]:
    """Pool one rubric window's verdicts from chunks onto documents.

    A document passes criterion ``Ck`` in the window when any of its chunks in
    that window does. Pooling never crosses windows: each window stays one
    placement in the 2PL likelihood.

    Raises:
        DataError: a chunk without verdicts, or sibling chunks with different criteria.
    """
    pooled: dict[str, dict[str, int]] = {}
    for chunk_id, criteria in chunk_criteria.items():
        if not criteria:
            raise DataError(f"chunk {chunk_id!r} has no rubric criteria")
        document_id = document_id_for_chunk(chunk_id, chunk_mapping)
        current = pooled.get(document_id)
        if current is None:
            pooled[document_id] = dict(criteria)
            continue
        if current.keys() != criteria.keys():
            raise DataError(
                f"chunks for document {document_id!r} have inconsistent rubric criteria: "
                f"{sorted(current)} != {sorted(criteria)}"
            )
        for criterion, value in criteria.items():
            current[criterion] = max(current[criterion], value)
    return pooled


# ---------------------------------------------------------------------------
# The text budget: one fitting mechanism for every served role
# ---------------------------------------------------------------------------

BUDGET_DOC_ID = "<budget>"
"""The ``doc_id`` of the one per-corpus census row a hosted vendor profile records: its documented
limit as the effective budget (``budget_source: vendor``). No tokenizer means nothing is measured,
so there is no per-document row."""


class TextBudgetExceededError(DataError):
    """An input cannot be sent within the declared :class:`TextBudget`, and the declared policy refuses to
    shorten it -- ``on_overflow: fail``, or ``on_overflow: chunk`` applied to a query (queries are never
    chunked), or a pair whose query fills the budget and leaves the document nothing."""


ContentParts = str | tuple[str, str]
"""The cut content of one fitted output: the text itself for the ``query`` and ``document`` shapes, a
``(query, document)`` tuple for ``pair``."""


@dataclass(frozen=True)
class FitResult:
    """What :func:`fit` decided for one call's inputs, in input order (chunks flattened).

    Attributes:
        shape: The request shape the inputs were fitted as.
        texts: The rendered request strings, one per output, in ``ids`` order: the full template with the
            cut content spans re-attached. Empty only for an unrendered ``pair`` -- a hosted vendor profile
            (no tokenizer), or a pair whose budget declares no template (the engine renders it; the caller
            sends :attr:`contents`) -- a hosted profile's ``query`` and ``document`` shapes return their raw
            content strings, which is what those wires take. This is what a text or ``token_ids`` wire route
            sends; the strings are returned as text, not token ids, because every wire route accepts text,
            the engine's own tokenisation (with its ``add_special_tokens`` flag) stays authoritative, and the
            client-side tokenisation this mechanism needs to measure and cut is the same one either way.
        contents: The cut content per output: the span text (a str), or the ``(query, document)`` parts of
            a pair -- what a wire route the engine renders the template for receives.
        ids: The output id per rendered text: the input's id, or ``<id>#<k>`` for its chunks (as
            :func:`chunk_ranking_example` names them).
        chunk_mapping: For chunked inputs, chunk id -> input id (an unsplit input maps to itself), ready
            for :func:`max_pool_scores_by_document`. ``None`` when nothing was chunked.
        overhead: The measured fixed overhead of (template, shape), in tokens -- the empty render plus the
            shape's post-processor tokens. ``None`` when no tokenizer was declared (vendor mode: nothing is
            measured).
        budget_source: ``"tokenizer"`` when the budget was measured with the declared tokenizer (the
            content was cut to fit it); ``"vendor"`` when no tokenizer is declared and the budget is the
            hosted vendor's documented limit (content uncut).
        aggregation: How a chunked document's scores pool back onto it: ``"max"`` -- a document scores its
            best chunk, the same rule as :func:`max_pool_scores_by_document` -- recorded on every chunked
            census row. ``None`` when nothing was chunked.
        cuts: The recorded cuts (also recorded into the census passed to :func:`fit`, when one was).
    """

    shape: RequestShape
    texts: tuple[str, ...]
    contents: tuple[ContentParts, ...]
    ids: tuple[str, ...]
    chunk_mapping: dict[str, str] | None = None
    overhead: int | None = None
    budget_source: Literal["tokenizer", "vendor"] = "tokenizer"
    aggregation: Literal["max"] | None = None
    cuts: tuple[TextCutRecord, ...] = ()


class TextBudget(BaseModel):
    """The declared text budget of a served role: one mechanism that fits every request into a model's input.

    The budget counts the model's whole input sequence -- the template's fixed segments, their special
    tokens, the instruction and the content together -- in tokens of the declared tokenizer. The fixed
    overhead is measured by rendering the template once with every content span empty (per request shape,
    the instruction filled: it is fixed for the run), only the content spans are cut, with the offset-based
    :func:`token_prefix`, and the template is re-attached after the cut. The engine never cuts: no request
    asks for engine-side truncation, so the declared budget is the only budget.

    Frozen and part of the content identity: :meth:`identity` returns the payload -- the template's
    canonical JSON, the budgets, the overflow policy, the chunk geometry, and the tokenizer *file's*
    SHA-256 (its name is runtime, as the judge's is).

    Attributes:
        tokenizer: The tokenizer the budget counts in: a Hugging Face repository id with an optional
            ``@revision``, or a local path to a ``tokenizer.json``. ``None`` only for a hosted vendor
            profile without one: content is then sent uncut, the declared ``max_tokens`` is the vendor's
            documented limit, and nothing is measured or cut client-side. Runtime by name; the file's
            SHA-256 is content (:meth:`identity`).
        max_tokens: The budget: the largest total input sequence, in the declared tokenizer's tokens
            (``_tokens``). Required: a budget without a number is not a budget.
        query_max_tokens: The query's budget. On a ``pair`` budget it is the query's share: when a pair
            overflows, the query is cut to it first and the document gets what remains (an input under
            budget is sent byte-identical to the uncut render, so the share binds on overflow only). On the
            ``query`` shape (an embedding role's per-shape budget) it is that shape's WHOLE budget --
            ``max_tokens`` then caps the document shape only. ``None`` (the default) declares no split: on
            a pair, a query that does not fit the budget is then refused rather than cut undeclared (declare
            the split instead). A share above ``max_tokens`` is refused here; EQUALITY is legal (both shapes
            capped the same) -- a pair share at or over the budget is refused one layer up, by the rerank
            config, and ``fit``'s pair cut refuses a query whose settled render would leave the document
            nothing.
        template: The request template (:class:`~rcp_ndcg.data.templates.TemplateSpec`), whose fixed
            segments are measured once per (template, shape) and whose specials are resolved from the
            tokenizer. ``None`` fits raw text: the overhead is then the tokenizer post-processor's tokens
            under ``add_special_tokens=True`` -- the pooling routes' engine default -- so the appended
            anchor is reserved without a frame.
        on_overflow: What an input that does not fit does: ``cut`` (the default: the content is cut,
            every cut recorded in the census under ``text_budget``), ``chunk`` (the document span is split
            into :class:`ChunkPolicy` chunks, each carrying the full template), or ``fail`` (the input is
            refused with :class:`TextBudgetExceededError`). A query is never chunked.
        chunk: The chunk geometry, ``on_overflow: chunk`` only -- the judge's :class:`ChunkPolicy` reused,
            not copied: chunks of at most ``chunk.max_tokens`` content tokens, each carrying the full
            template, ids ``<id>#<k>``, scores pooled back by ``max``.
        aggregation: How a chunked document's scores pool back onto it: ``max``, its best chunk's -- the
            same rule as :func:`max_pool_scores_by_document`, which the caller applies to the returned
            chunk mapping. The only value for now; every census row of a chunked input names it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "tokenizer": FieldRole.RUNTIME,
        "max_tokens": FieldRole.CONTENT,
        "query_max_tokens": FieldRole.CONTENT,
        "template": FieldRole.CONTENT,
        "on_overflow": FieldRole.CONTENT,
        "chunk": FieldRole.CONTENT,
        "aggregation": FieldRole.CONTENT,
    }

    tokenizer: str | None = Field(default=None, min_length=1)
    max_tokens: int = Field(ge=1)
    query_max_tokens: int | None = Field(default=None, ge=1)
    template: TemplateSpec | None = None
    on_overflow: Literal["cut", "chunk", "fail"] = "cut"
    chunk: ChunkPolicy | None = Field(default=None, exclude_if=lambda value: value is None)
    aggregation: Literal["max"] = "max"

    @model_validator(mode="after")
    def _declared_overflow_has_its_geometry(self) -> TextBudget:
        """The same rule as ``TextPolicy``: a declared chunk policy needs its geometry, and nothing else
        carries one."""
        if self.on_overflow == "chunk" and self.chunk is None:
            raise ValueError(
                "on_overflow 'chunk' needs a chunk geometry: {on_overflow: chunk, chunk: {max_tokens: ..., "
                "overlap_tokens: ...}}"
            )
        if self.chunk is not None and self.on_overflow != "chunk":
            raise ValueError(f"a chunk geometry applies to on_overflow 'chunk' only, not {self.on_overflow!r}")
        return self

    @model_validator(mode="after")
    def _a_share_above_the_budget_is_meaningless(self) -> TextBudget:
        """A query share above the whole budget leaves the document nothing and the query over the served
        context. Equal is legal: on the embedding roles ``query_max_tokens`` is the query shape's WHOLE
        budget, and both shapes may be capped the same; on a pair budget an equal share is refused one layer
        up (the rerank config), and ``fit``'s pair cut refuses a query that would leave the document nothing
        at runtime."""
        if self.query_max_tokens is not None and self.query_max_tokens > self.max_tokens:
            raise ValueError(
                f"query_max_tokens ({self.query_max_tokens}) must not exceed max_tokens ({self.max_tokens}): "
                "the query budget would be over the model's whole input budget"
            )
        return self

    @model_validator(mode="after")
    def _a_budget_without_a_tokenizer_cuts_nothing(self) -> TextBudget:
        """A hosted vendor profile without a tokenizer sends content uncut: a template, chunk/fail and a
        query split would be silently inert, so they are refused instead of ignored."""
        if self.tokenizer is None:
            inert = [
                name
                for name, value in (
                    ("on_overflow", self.on_overflow),
                    ("query_max_tokens", self.query_max_tokens),
                    ("chunk", self.chunk),
                    ("template", self.template),
                )
                if value is not None and value != "cut"
            ]
            if inert:
                raise ConfigError(
                    f"this budget declares no tokenizer, so its content is sent uncut (a hosted vendor "
                    f"profile) and {inert} would be inert",
                    hint="declare tokenizer (the profile then cuts like a self-hosted one), or drop the "
                    "inert fields (on_overflow, query_max_tokens, chunk, template)",
                )
        return self

    def identity(self, tokenizer: TextTokenizer | None = None) -> dict[str, Any]:
        """The content identity payload of the budget, with the tokenizer file's SHA-256 when a tokenizer is
        loaded (the name is runtime, as the judge's is; the file's hash is what two tokenizers are told apart
        by). A hosted vendor profile (a budget that declares no tokenizer) carries no hash: nothing is
        measured against it."""
        payload = identity_payload(self)
        if tokenizer is not None:
            payload["tokenizer_sha256"] = tokenizer.sha256
        return payload


_VENDOR_WARNED: set[tuple[str, str]] = set()


def _fit_vendor(
    items: list[Any],
    shape: RequestShape,
    budget: TextBudget,
    names: Sequence[str],
    *,
    corpus: str,
    census: TextTruncationCensus | None,
) -> FitResult:
    """The hosted-vendor path: no tokenizer, so nothing is measured, framed or cut; the declared
    ``max_tokens`` (the vendor's documented limit) is recorded as the effective budget -- one row per
    (corpus, limit) per census, and one warning per (corpus) per process."""
    key = (TextTruncationCensus.TEXT_BUDGET, corpus)
    if key not in _VENDOR_WARNED:
        _VENDOR_WARNED.add(key)
        logger.warning(
            "text budget (%s) on corpus %s: no tokenizer declared for this hosted profile; content is sent "
            "uncut and the vendor's documented limit (%d tokens) is the effective budget (budget_source: "
            "vendor); nothing client-side is measured or cut",
            TextTruncationCensus.TEXT_BUDGET,
            corpus or "<unnamed>",
            budget.max_tokens,
        )
    if census is not None and (corpus, budget.max_tokens) not in census._budget_rows:
        census._budget_rows.add((corpus, budget.max_tokens))
        census.record(
            corpus=corpus,
            doc_id=BUDGET_DOC_ID,
            original_tokens=budget.max_tokens,
            kept_tokens=budget.max_tokens,
            original_chars=0,
            kept_chars=0,
            mechanism=TextTruncationCensus.TEXT_BUDGET,
            budget_source="vendor",
            shape=shape,
            budget_tokens=budget.max_tokens,
        )
    contents: list[ContentParts] = [item if shape != "pair" else (item[0], item[1]) for item in items]
    texts: tuple[str, ...] = () if shape == "pair" else tuple(part for part in contents if isinstance(part, str))
    return FitResult(
        shape=shape,
        # A hosted profile sends the content itself (or the pair's parts): nothing is rendered.
        texts=texts,
        contents=tuple(contents),
        ids=tuple(names),
        overhead=None,
        budget_source="vendor",
    )


def fit(
    inputs: Sequence[str] | Sequence[tuple[str, str]],
    shape: RequestShape,
    budget: TextBudget,
    tokenizer: TextTokenizer | None = None,
    *,
    ids: Sequence[str] | None = None,
    instruction: str | None = None,
    media_tokens: Sequence[int] | None = None,
    corpus: str = "",
    census: TextTruncationCensus | None = None,
) -> FitResult:
    """Fit every input into the model's input budget: the one call every served role's client makes.

    The algorithm, per request shape:

    1. **Measure the fixed overhead once** per (template, shape, tokenizer, instruction): the template
       rendered with every content span empty, counted as the engine reads it (the shape's
       ``add_special_tokens`` flag -- its post-processor tokens included). Without a template the
       overhead is the post-processor's tokens on the raw text (the routes' default), so an appended
       anchor is reserved even with no frame.
    2. **Budget the content spans**: ``remaining = max_tokens - overhead - media``. For a ``pair``, the
       query is cut to ``query_max_tokens`` first (or, with none declared, only when it alone fills the
       budget) and the document gets the rest.
    3. **Cut only the content spans**, with the offset-based :func:`token_prefix`, verified against the
       *assembled* render so a byte-level merge across a span join cannot push the request over the
       budget; the template is re-attached after the cut. Inputs under budget come back byte-identical
       to the uncut render.
    4. **Chunk** on ``on_overflow: chunk``: the document content is split into verbatim
       :class:`ChunkPolicy` slices and every chunk is rendered with the full template (engine-side
       chunking of a framed render loses the frame and the anchor -- chunk on the client, never on the
       engine). Output ids are ``<id>#<k>`` and ``FitResult.chunk_mapping`` carries each chunk back to
       its input, ready for :func:`max_pool_scores_by_document` (``aggregation: max``).
    5. **Record** every cut in the census under the ``text_budget`` mechanism, with the request shape,
       the budget source, and -- on chunked inputs -- the ``max`` aggregation.

    Media (a vision block, a video segment) are never cut: declare each input's media token count in
    ``media_tokens`` and it is reserved whole out of the budget before the content is cut. A declared
    media count that leaves no room for the frame is a :class:`ConfigError`.

    Args:
        inputs: The texts (``query`` and ``document`` shapes) or ``(query, document)`` pairs (``pair``
            shape), in order.
        shape: Which request shape the inputs are.
        budget: The declared :class:`TextBudget`.
        tokenizer: The loaded tokenizer, in whose tokens ``max_tokens`` is counted. ``None`` is the
            hosted-vendor path: content is sent uncut, the budget is the vendor's documented limit, and
            it is recorded with ``budget_source: vendor`` -- nothing is measured, so nothing is framed or
            cut (and a ``pair`` returns its parts, not rendered text).
        ids: Each input's id, for the census and the chunk ids; positional ``"<index>"`` when unset.
        instruction: The run-level instruction, where the template declares an ``instruction`` span; part
            of the fixed overhead (never cut).
        media_tokens: Each input's media token count, indivisible and reserved whole (the hook the media
            lane builds on: a vision block is counted, never cut).
        corpus: The corpus or role name, for the census and the vendor warning.
        census: Where the cuts are recorded.

    Returns:
        The :class:`FitResult`: the rendered strings (or, on ``text``/``token_ids`` routes without a
        frame, the contents), the cut contents per output, the output ids, the chunk mapping, the
        measured overhead, the budget source, and the cut records.

    Raises:
        ConfigError: the template or the media alone fill the budget, the declared chunk geometry cannot
            fit it, or a template special does not resolve in the tokenizer.
        TextBudgetExceededError: ``on_overflow: fail`` and an input over budget; a query under
            ``on_overflow: chunk``; a pair whose query leaves the document nothing.
    """
    items = list(inputs)
    if shape == "pair":
        for index, item in enumerate(items):
            if not (isinstance(item, (tuple, list)) and len(item) == 2 and all(isinstance(part, str) for part in item)):
                raise ValueError(f"inputs[{index}] must be a (query, document) pair of strings for the 'pair' shape")
    else:
        for index, item in enumerate(items):
            if not isinstance(item, str):
                raise ValueError(f"inputs[{index}] must be a string for the {shape!r} shape")
    names = [str(index) for index in range(len(items))] if ids is None else [str(name) for name in ids]
    if len(names) != len(items):
        raise ValueError(f"ids ({len(names)}) must name every input ({len(items)})")
    media = [0] * len(items) if media_tokens is None else list(media_tokens)
    if len(media) != len(items):
        raise ValueError(f"media_tokens ({len(media)}) must be declared for every input ({len(items)})")
    if any(not isinstance(count, int) or count < 0 for count in media):
        raise ValueError("media_tokens must be non-negative token counts")
    if budget.tokenizer is not None and (tokenizer is None or tokenizer.name != budget.tokenizer):
        raise ConfigError(
            f"fit was given {'no tokenizer' if tokenizer is None else f'the tokenizer {tokenizer.name!r}'} but "
            f"the budget declares {budget.tokenizer!r}: the budget's numbers are counted in the declared "
            "tokenizer's tokens",
            hint="load the budget's tokenizer and pass it; the hosted-vendor path (a tokenizer of none) is for "
            "budgets that declare none",
            cli_hint="set the same tokenizer the budget declares (judge-style: --set <role>.tokenizer=...), or "
            "drop the tokenizer field for a hosted profile",
        )
    # The shape's budget: on the embedding roles ``query_max_tokens`` caps the query shape whole (its whole
    # budget there; ``max_tokens`` caps the document shape), on a pair it stays the query's share of the
    # budget, and on the document shape (and without a declared share) ``max_tokens`` caps. Every cap and
    # every message below counts against the shape's own budget.
    if budget.query_max_tokens is not None and shape == "query":
        shape_budget = budget.query_max_tokens
    else:
        shape_budget = budget.max_tokens
    if tokenizer is None:
        if media_tokens is not None and any(media):
            raise ConfigError(
                "media_tokens need a tokenizer to reserve against, and this budget declares none (a hosted "
                "vendor profile sends content uncut): the media reservation cannot be honoured",
                hint="declare tokenizer on the budget, or drop media_tokens for this profile",
            )
        return _fit_vendor(items, shape, budget, names, corpus=corpus, census=census)

    template = budget.template
    instr = instruction or ""
    # The declared content normalisation: the template's per-shape strip/lowercase (the one call,
    # :meth:`TemplateSpec.normalize_text`), applied to the content spans before anything is measured (the
    # reference and the engine see the same text). Without a template there is no declaration, so nothing
    # is normalised.
    if template is not None and template.normalisers(shape):
        raw_items = list(items)  # the inputs as given, for the census rows' original side
        if shape == "pair":
            pairs = [(query, document) for query, document in items]
            items = [
                (template.normalize_text(shape, query), template.normalize_text(shape, document))
                for query, document in pairs
            ]
        else:
            assert all(isinstance(item, str) for item in items)  # validated at the top, for the type
            items = [template.normalize_text(shape, str(item)) for item in items]
    else:
        raw_items = items
    # The engine's behaviour for the route: declared on the template; a raw-text request gets the pooling
    # routes' default (the post-processor's tokens are appended), so its anchor is reserved either way.
    flag = template.adds_special_tokens(shape) if template is not None else True
    if template is not None:
        overhead = template.overhead(shape, tokenizer, instruction=instr)
    else:
        overhead = tokenizer.count("", add_special_tokens=True)

    def _budget_hint(verb: str, rest: str) -> str:
        """The raise hint that names the knob that binds: on a query shape budgeted by a declared
        ``query_max_tokens`` raising ``max_tokens`` moves nothing."""
        knob = "query_max_tokens" if shape == "query" and budget.query_max_tokens is not None else "max_tokens"
        return f"{verb} {knob}" + (f", {rest}" if rest else "")

    if overhead > shape_budget:
        raise ConfigError(
            f"the template's fixed overhead alone is {overhead} tokens, over the budget of {shape_budget}",
            hint=_budget_hint("raise", "or simplify the template"),
        )

    def assemble(query: str, document: str) -> str:
        """The full rendered request, the frame re-attached around whatever the spans now hold."""
        if template is None:
            return query + document if shape == "pair" else (query if shape == "query" else document)
        return template.render(shape, tokenizer, query=query, document=document, instruction=instr)

    def _cut_span(text: str, *, span: Literal["query", "document"], other: str = "", cap: int) -> str:
        """The longest prefix of a content span whose assembled render fits ``cap`` (the budget minus the
        media, which ride beside the rendered string and are never cut). The piece is rendered into its OWN
        span, with the other span held at ``other``, so the verified count is the render the engine reads --
        the two joins of an asymmetric frame tokenize differently, and measuring a piece in the wrong span
        would ship an over-budget render."""
        if span == "query":
            rendered = lambda piece: assemble(piece, other)  # noqa: E731  (shape 'query' ignores ``other``; pair holds the document there)
        else:
            rendered = lambda piece: assemble(other, piece)  # noqa: E731  (shape 'document' ignores ``other``; pair holds the query there)
        return token_prefix(text, cap, tokenizer, rendered=rendered, add_special_tokens=flag)

    texts: list[str] = []
    contents: list[ContentParts] = []
    entries: list[tuple[str, str]] = []
    cuts: list[TextCutRecord] = []
    chunked_any = False

    def _record(
        *,
        doc_id: str,
        original: ContentParts,
        kept: ContentParts,
        aggregation: str | None,
        raw: ContentParts | None = None,
    ) -> None:
        """One cut row (also appended to the census when the caller passed one).

        ``raw`` is the input as given, when a declared normalisation changed the spans before the cut: the
        row's original side is then the raw text (the input), never the normalised one (declared policy).
        """
        source = original if raw is None else raw
        original_text = source if isinstance(source, str) else source[0] + source[1]
        kept_text = kept if isinstance(kept, str) else kept[0] + kept[1]
        assert tokenizer is not None
        cut = TextCutRecord(
            corpus=corpus,
            doc_id=doc_id,
            original_chars=len(original_text),
            kept_chars=len(kept_text),
            original_tokens=tokenizer.count(original_text),
            kept_tokens=tokenizer.count(kept_text),
            mechanism=TextTruncationCensus.TEXT_BUDGET,
            budget_source="tokenizer",
            aggregation=aggregation,
            shape=shape,
            budget_tokens=shape_budget,
        )
        if census is not None:
            census.record(
                corpus=cut.corpus,
                doc_id=cut.doc_id,
                original_chars=cut.original_chars,
                kept_chars=cut.kept_chars,
                original_tokens=cut.original_tokens,
                kept_tokens=cut.kept_tokens,
                mechanism=cut.mechanism,
                budget_source=cut.budget_source,
                aggregation=cut.aggregation,
                shape=cut.shape,
                budget_tokens=cut.budget_tokens,
            )
        cuts.append(cut)

    def _chunks(content: str, room: int, query: str, cap: int) -> list[str]:
        """The document's chunks under the declared geometry, each guaranteed to render under the budget:
        the whole template is re-attached per chunk, so the frame and its anchors survive every one."""
        assert budget.chunk is not None  # validated with on_overflow
        if budget.chunk.max_tokens > room:
            raise ConfigError(
                f"chunk.max_tokens ({budget.chunk.max_tokens} tokens of content) cannot fit the {room} tokens "
                f"left under the budget of {shape_budget}: every chunk carries the full template",
                hint="lower chunk.max_tokens, raise max_tokens, or (for a pair) set query_max_tokens",
            )
        pieces = split_into_chunks(content, budget.chunk, tokenizer)
        for index, piece in enumerate(pieces):
            # A slice can re-tokenize longer on its own than it did in place (a byte-level join can inflate
            # where the frame meets it); trim it until the render fits. The trimmed tail is counted: the
            # chunk's own census row records the trimmed kept size against the whole input.
            if tokenizer.count(assemble(query, piece), add_special_tokens=flag) > cap:
                pieces[index] = _cut_span(piece, span="document", other=query, cap=cap)
        return pieces

    for index, item in enumerate(items):
        input_id = names[index]
        spent = media[index]
        # The item's total: the budget minus the media, which ride beside the rendered string and are never cut.
        cap = shape_budget - spent
        if overhead + spent > shape_budget:
            raise ConfigError(
                f"the fixed template overhead ({overhead} tokens) plus the declared media ({spent}) already "
                f"fill the budget of {shape_budget}; the media are never cut",
                hint=_budget_hint("raise", "or shrink the declared media (a media block is indivisible)"),
            )
        # The census rows compare the input AS GIVEN with what ships: normalisation is declared policy,
        # not a cut, so the row's original side stays the raw text even when the spans were normalised.
        raw = raw_items[index]
        if shape == "pair":
            assert isinstance(item, (tuple, list))
            assert isinstance(raw, (tuple, list))
            query, document = item
            original: ContentParts = (query, document)
        else:
            assert isinstance(item, str) and isinstance(raw, str)
            query, document = (item, "") if shape == "query" else ("", item)
            original = item
        if tokenizer.count(assemble(query, document), add_special_tokens=flag) <= cap:
            if not (template is None and shape == "pair"):
                texts.append(assemble(query, document))
            contents.append(original)
            entries.append((input_id, input_id))
            continue
        # Over budget: fail, cut, or chunk -- the declared policy and nothing else.
        if budget.on_overflow == "fail":
            media_note = f", the declared media {spent}" if spent else ""
            raise TextBudgetExceededError(
                f"input {input_id!r} is {tokenizer.count(query + document)} tokens of content, over the "
                f"declared text budget of {shape_budget} (the fixed template takes {overhead}{media_note})",
                hint="set on_overflow: 'cut' (or 'chunk' for documents) to shorten it, or " + _budget_hint("raise", ""),
            )
        if shape == "pair":
            # The query's span is settled first: to its declared share, else only when it fits the budget whole.
            if budget.query_max_tokens is None:
                if tokenizer.count(query) > cap - overhead:
                    raise TextBudgetExceededError(
                        f"the query of input {input_id!r} does not fit the pair budget of {shape_budget} "
                        "tokens, and no split is declared (query_max_tokens): cutting it undeclared would "
                        "silently eat the document's share",
                        hint="set query_max_tokens to the query's share, so the document keeps the rest",
                    )
                q_final = query
            else:
                share = budget.query_max_tokens
                q_final = query if tokenizer.count(query) <= share else token_prefix(query, share, tokenizer)
            # The query must leave room for the frame (and the post-processor's anchor) even with an empty document.
            q_final = _cut_span(q_final, span="query", other="", cap=cap)
            q_min = tokenizer.count(assemble(q_final, ""), add_special_tokens=flag)
            if q_min >= cap and tokenizer.count(document) > 0:
                raise TextBudgetExceededError(
                    f"the query of input {input_id!r} fills the pair budget of {shape_budget} tokens and "
                    "leaves the document nothing",
                    hint="lower query_max_tokens (or raise max_tokens), so the document keeps a share",
                )
            room = cap - q_min
            if budget.on_overflow == "cut":
                d_final = _cut_span(document, span="document", other=q_final, cap=cap)
                if (
                    tokenizer.count(assemble(q_final, d_final), add_special_tokens=flag) > cap
                ):  # pragma: no cover - guarded by construction
                    raise DataError(
                        f"the assembled render of input {input_id!r} exceeds the budget of {shape_budget} "
                        "tokens after both spans were verified: an internal invariant broke; report this",
                        hint="this is a bug in the text-budget mechanism: report it with the inputs",
                    )
                if not (template is None and shape == "pair"):
                    texts.append(assemble(q_final, d_final))
                contents.append((q_final, d_final))
                entries.append((input_id, input_id))
                _record(doc_id=input_id, original=original, kept=(q_final, d_final), aggregation=None, raw=raw)
            else:
                pieces = _chunks(document, room, q_final, cap)
                if len(pieces) == 1:
                    # One piece is the whole document: one request, its own id, no chunking (as
                    # chunk_ranking_example keeps an unsplit document).
                    if not (template is None and shape == "pair"):
                        texts.append(assemble(q_final, pieces[0]))
                    contents.append((q_final, pieces[0]))
                    entries.append((input_id, input_id))
                    _record(doc_id=input_id, original=original, kept=(q_final, pieces[0]), aggregation=None, raw=raw)
                    continue
                for k, piece in enumerate(pieces):
                    chunk_id = f"{input_id}{CHUNK_ID_SEPARATOR}{k}"
                    if not (template is None and shape == "pair"):
                        texts.append(assemble(q_final, piece))
                    contents.append((q_final, piece))
                    entries.append((chunk_id, input_id))
                    _record(
                        doc_id=chunk_id,
                        original=original,
                        kept=(q_final, piece),
                        aggregation=budget.aggregation,
                        raw=raw,
                    )
                chunked_any = True
        elif budget.on_overflow == "chunk":
            if shape == "query":
                raise TextBudgetExceededError(
                    f"input {input_id!r} does not fit the budget of {shape_budget} tokens, and a query is "
                    "never chunked: queries are cut or refused, never split",
                    hint=_budget_hint("raise", "or shorten the query"),
                )
            assert isinstance(item, str)  # a pair chunked above; this branch is single-text only
            pieces = _chunks(item, cap - overhead, "", cap)
            if len(pieces) == 1:
                if (
                    tokenizer.count(assemble("", pieces[0]), add_special_tokens=flag) > cap
                ):  # pragma: no cover - guarded by construction
                    raise DataError(
                        f"the assembled render of input {input_id!r} exceeds the budget of {shape_budget} "
                        "tokens after the span was verified: an internal invariant broke; report this",
                        hint="this is a bug in the text-budget mechanism: report it with the inputs",
                    )
                texts.append(assemble("", pieces[0]))
                contents.append(pieces[0])
                entries.append((input_id, input_id))
                _record(doc_id=input_id, original=item, kept=pieces[0], aggregation=None, raw=raw)
                continue
            for k, piece in enumerate(pieces):
                chunk_id = f"{input_id}{CHUNK_ID_SEPARATOR}{k}"
                texts.append(assemble("", piece))
                contents.append(piece)
                entries.append((chunk_id, input_id))
                _record(doc_id=chunk_id, original=item, kept=piece, aggregation=budget.aggregation, raw=raw)
            chunked_any = True
        else:  # cut
            assert isinstance(item, str)  # the pair's cut is handled above
            kept = _cut_span(item, span="query" if shape == "query" else "document", cap=cap)
            if not (template is None and shape == "pair"):
                rendered = assemble(kept, "") if shape == "query" else assemble("", kept)
                if (
                    tokenizer.count(rendered, add_special_tokens=flag) > cap
                ):  # pragma: no cover - guarded by construction
                    raise DataError(
                        f"the assembled render of input {input_id!r} exceeds the budget of {shape_budget} "
                        "tokens after the span was verified: an internal invariant broke; report this",
                        hint="this is a bug in the text-budget mechanism: report it with the inputs",
                    )
                texts.append(rendered)
            contents.append(kept)
            entries.append((input_id, input_id))
            _record(doc_id=input_id, original=item, kept=kept, aggregation=None, raw=raw)

    out = [entry[0] for entry in entries]
    if len(set(out)) != len(out):
        duplicates = sorted({name for name in out if out.count(name) > 1})
        raise DataError(
            f"two outputs share an id ({duplicates}): a chunk of one input collides with another input's id, "
            "and one score would be pooled over the other",
            hint="pass ids that do not collide with any input id plus its '<id>#<k>' chunks",
        )
    return FitResult(
        shape=shape,
        texts=tuple(texts),
        contents=tuple(contents),
        ids=tuple(out),
        chunk_mapping=dict(entries) if chunked_any else None,
        overhead=overhead,
        budget_source="tokenizer",
        aggregation=budget.aggregation if chunked_any else None,
        cuts=tuple(cuts),
    )


class Preprocessing(BaseModel):
    """The effective preprocessing of one judging pass: part of its identity and of its judgement family.

    Attributes:
        text: The per-document text policy.
        chunk: The chunk geometry in effect: the text policy's own (``on_overflow: chunk``) or one the judging
            pass applies to candidates longer than ``chunk.max_tokens``. ``None``: documents are judged whole.
        image: The pixel budget of page images and video frames, or ``None`` for text.
        video: Which frames of a video the judge sees and how they travel, or ``None`` (no video, or every frame
            as the corpus holds it).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: TextPolicy = DEFAULT_TEXT_POLICY
    chunk: ChunkPolicy | None = None
    image: ImagePolicy | None = None
    video: VideoPolicy | None = None

    @model_validator(mode="after")
    def _one_chunk_geometry(self) -> Preprocessing:
        if self.text.chunk is not None:
            if self.chunk is not None and self.chunk != self.text.chunk:
                raise ValueError(
                    f"the corpus is chunked at load ({self.text.chunk.descriptor}); a different judging chunk "
                    f"geometry ({self.chunk.descriptor}) would chunk the chunks"
                )
            object.__setattr__(self, "chunk", self.text.chunk)
        return self

    @property
    def key(self) -> str:
        """16-hex digest of the effective policy."""
        from rcp_ndcg.support.identity import hash_payload, short

        return short(hash_payload(self.model_dump(mode="json")), 16)


__all__ = [
    "BUDGET_DOC_ID",
    "CHUNK_ID_SEPARATOR",
    "ChunkPolicy",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEXT_POLICY",
    "DocumentOverCapError",
    "FitResult",
    "Preprocessing",
    "TextBudget",
    "TextBudgetExceededError",
    "TextCutRecord",
    "ContentParts",
    "OnOverflow",
    "TextTruncationCensus",
    "TextPolicy",
    "apply_text_policy",
    "chunk_ranking_example",
    "document_id_for_chunk",
    "document_ids_from_chunks",
    "fit",
    "max_pool_rubric_window_by_document",
    "max_pool_scores_by_document",
    "split_into_chunks",
    "token_prefix",
]
