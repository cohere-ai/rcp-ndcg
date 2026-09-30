"""Preprocessing: the declared text policy, chunking, and the record of every cut.

One home for what happens to a document's text before a judge reads it:

* :class:`TextPolicy` -- the per-document text policy (keep, truncate with a
  census record, chunk, or fail), applied by :func:`apply_text_policy`, the only
  place a document is shortened before its window;
* :class:`ChunkPolicy` -- how a long document is split into overlapping
  chunks (in tokens), the chunk producer (:func:`chunk_ranking_example`,
  chunk ids ``<doc_id>#<k>``) and the aggregation of chunk judgements back onto
  their document (maximum per criterion within one window, maximum score);
* :class:`TextTruncationCensus` -- every cut, at load (``doc_policy``) and in a
  judging window (``window_budget``);
* :class:`Preprocessing` -- the effective policy of a judging pass, whose
  :attr:`~Preprocessing.key` is part of the judging identity and the family.

Nothing truncates silently: a cut is either declared policy (and counted) or a
per-window budget cut (and counted).

Limits are in tokens of the judge's tokenizer (:mod:`rcp_ndcg.data.tokenizer`), and every cut is made at a token
boundary of the original text, located with the tokenizer's offset mapping: a truncated document is a verbatim
prefix, and a chunk a verbatim slice, of the document. There is no character fallback: a policy that cuts refuses
to run without a tokenizer.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rcp_ndcg_core._records import RankingExample
from rcp_ndcg_core.content import Content

from rcp_ndcg.data.resolution import ImagePolicy, VideoPolicy
from rcp_ndcg.errors import ConfigError, DataError
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
    text: str, max_tokens: int, tokenizer: TextTokenizer, *, rendered: Callable[[str], str] | None = None
) -> str:
    """A prefix of ``text`` that ends at one of its first ``max_tokens`` token boundaries and counts at most
    ``max_tokens``: the longest such prefix the search finds.

    The cut is located with the tokenizer's offset mapping on the original text, so the result is a verbatim prefix
    of ``text`` (never tokens decoded back to text). A candidate is counted as the judge will read it --
    ``rendered(prefix)`` when given, e.g. the prompt's escaping -- because a cut word or an escaped character can
    tokenize to more tokens than it had in place: the cut starts at the ``max_tokens``-th boundary and moves back
    until the prefix fits (a galloping, then a binary search over token boundaries). A longer prefix that
    re-tokenizes into fewer tokens than it has boundaries is not sought, so the cut can stop a few characters short
    of the longest prefix that would fit.

    Args:
        text: The document text.
        max_tokens: The budget, in tokens.
        tokenizer: The judge's tokenizer.
        rendered: How the text appears to the judge, when that is not the text itself.

    Returns:
        ``text`` itself when it fits; else its prefix (possibly empty).
    """

    def count(piece: str) -> int:
        return tokenizer.count(rendered(piece) if rendered is not None else piece)

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
    """

    __slots__ = (
        "corpus",
        "doc_id",
        "kept_chars",
        "kept_tokens",
        "mechanism",
        "original_chars",
        "original_tokens",
        "query_id",
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
    ) -> None:
        self.corpus = corpus
        self.doc_id = doc_id
        self.original_chars = original_chars
        self.kept_chars = kept_chars
        self.original_tokens = original_tokens
        self.kept_tokens = kept_tokens
        self.mechanism = mechanism
        self.query_id = query_id

    def as_row(self) -> dict[str, Any]:
        return {
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

    When *sink* is set, every record is also appended to it as one JSON line.
    """

    DOC_POLICY = "doc_policy"
    WINDOW_BUDGET = "window_budget"
    MECHANISMS = (DOC_POLICY, WINDOW_BUDGET)

    def __init__(self, *, sink: str | Path | None = None) -> None:
        self._cuts: list[TextCutRecord] = []
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
    byte-level tokens) makes a chunk over the cap.

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
    "CHUNK_ID_SEPARATOR",
    "ChunkPolicy",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEXT_POLICY",
    "DocumentOverCapError",
    "Preprocessing",
    "TextCutRecord",
    "OnOverflow",
    "TextTruncationCensus",
    "TextPolicy",
    "apply_text_policy",
    "chunk_ranking_example",
    "document_id_for_chunk",
    "document_ids_from_chunks",
    "max_pool_rubric_window_by_document",
    "max_pool_scores_by_document",
    "split_into_chunks",
    "token_prefix",
]
