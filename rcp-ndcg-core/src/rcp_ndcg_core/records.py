"""The pipeline records: the per-query row the retrieval and judging pipeline reads and writes.

The in-memory model of the data layer: :class:`Document`, :class:`Query` and :class:`RankingExample` (with their
bases :class:`Text` and :class:`Input`) are what a reader yields, a writer takes, and retrieval, judging, storage
and the role clients pass around. The public path is this module, ``rcp_ndcg_core.records``, re-exported by the
``rcp_ndcg_core`` and ``rcp_ndcg.data`` facades. The judgement and IRT records are in
:mod:`rcp_ndcg_core.schemas`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Self

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from rcp_ndcg_core._logging import get_logger
from rcp_ndcg_core.content import Content, MediaRef, Modality, Part, TextPart

# Identifiers always serialise as strings -- this keeps qrels / search-results
# parity with BEIR / MTEB conventions and avoids float / int collisions.
ID = str

# query_id -> {doc_id: relevance_label}
Qrels = dict[ID, float]
QrelsDict = dict[ID, dict[ID, float]]
# query_id -> {doc_id: score}
SearchResults = dict[ID, dict[ID, float]]

Metric = str
Results = dict[Metric, float]


logger = get_logger(__name__)

#: How a document's title reaches the model. ``join`` (the default): MTEB's dataloader rule, the title
#: joined to the body. ``separate``: the title as its own leading text part, for a model or recipe that
#: takes it that way (mteb keeps the title as its own field too).
DocumentTitle = Literal["join", "separate"]

TEXT_FORMATTING_VERSION = "rcp-text/1"
"""The text-formatting rule's version, part of the run identities: the join (:func:`mteb_document_text`,
:meth:`Document.model_content`), the two instruction frames (:meth:`Query.format_query`/``format_content``)
and the judge's window rendering. Bump it when the text a model reads changes shape (a different join, a
different placement), so a resume never reuses candidates, judgements or scores built from the old strings.
The rule itself is code, not a config field: this constant is how the identities see it."""


def mteb_document_text(title: str | None, body: str) -> str:
    """The text MTEB's retrieval dataloader reads for a document (``_create_dataloaders._corpus_to_dict``).

    ``(title + " " + body).strip()`` when the title is non-empty, else ``body.strip()``: the exact rule
    of mteb's own loader, so a document reads byte-for-byte the same whichever format held it. A ``None``
    title counts as no title (mteb's loader raises on it; our readers normalise a blank or NaN title to
    ``None``), and a title of ``""`` does too (mteb's ``len(row["title"]) > 0``).
    """
    if title:
        return f"{title} {body}".strip()
    return body.strip()


class Input(BaseModel):
    """Every record has an id; :class:`Query`, :class:`Document` and :class:`RankingExample` alias it to the
    field name of their input data (``query_id``, ``doc_id``).

    Unknown keys are refused and a numeric id reads as its string form, so a plain dict from a frame or a JSON
    line validates into the same record the readers yield (the strict rules the data layer's in-memory path has
    always applied).
    """

    model_config = ConfigDict(extra="forbid", coerce_numbers_to_str=True)

    id: ID


class Text(Input):
    """A piece of content with an id.

    ``content`` is the canonical body -- an ordered list of text / image / video
    parts.  ``text`` is the text-only view of it and remains a real field, so
    every text-only caller and every text-only JSONL line keeps working
    unchanged and costs nothing extra: a text document stores its string and
    never materialises a :class:`Content`.

    Exactly one of the two is authoritative per instance, and the validator
    keeps them consistent -- if ``content`` is supplied, ``text`` is derived
    from it, so ``text`` can never disagree with the parts it summarises.
    """

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    text: str = ""
    content: Content | None = None

    @model_validator(mode="after")
    def _derive_text_from_content(self) -> Self:
        if self.content is not None:
            # ``text`` is a view over the parts, so content wins on conflict.
            self.text = self.content.text
        return self

    @property
    def as_content(self) -> Content:
        """The body as parts, materialising a text-only part when needed."""
        if self.content is not None:
            return self.content
        return Content.from_text(self.text)

    @property
    def media(self) -> list[MediaRef]:
        return self.content.media if self.content is not None else []

    @property
    def has_media(self) -> bool:
        return self.content is not None and self.content.has_media

    @property
    def modality(self) -> Modality:
        return self.content.modality if self.content is not None else Modality.TEXT


class Query(Text):
    # Defaults to empty so an image query -- ``content`` set, no text -- is
    # representable, as image-to-image retrieval requires.
    text: str = Field(default="", alias="query")
    id: ID = Field(alias="query_id")
    instruction: str | None = None

    @property
    def query_id(self) -> ID:
        return self.id

    def format_query(self, *, task_instruction: str | None = None) -> str:
        """The query text a model reads, under the two generic defaults.

        The **per-query instruction** (:attr:`instruction`, the data's own) is appended exactly as mteb's
        dataloader appends it: ``query + " " + instruction``, the query text as given. The **task
        instruction** (the caller's, from ``Dataset.task_instruction``) is prefixed:
        ``Task: <instruction>\\nQuery: <text>``.

        A recipe that places the task instruction itself (a template ``instruction`` span) or whose model
        takes no instruction (``instruction: none``) does not call this: the two generic defaults are what
        a model without such a declaration reads. Each instruction appears once, never both appended and
        slotted.
        """
        text = self.text
        if self.instruction is not None:
            text = f"{text} {self.instruction}"
        task = (task_instruction or "").strip()
        if task:
            return f"Task: {task}\nQuery: {text}"
        return text

    def format_content(self, *, task_instruction: str | None = None) -> Content:
        """The query body as parts, under the same two generic defaults as :meth:`format_query`.

        The encoder-facing counterpart: an image query keeps its image parts, gains the task instruction
        as a leading text part and the per-query instruction as a trailing one (a text query's two
        instructions are the one string :meth:`format_query` builds).
        """
        task = (task_instruction or "").strip()
        content = self.as_content
        if not content.has_media:
            return Content.from_text(self.format_query(task_instruction=task_instruction))
        parts: list[Part] = []
        if task:
            parts.append(TextPart(text=f"Task: {task}"))
        parts.extend(content.parts)
        if self.instruction:
            parts.append(TextPart(text=self.instruction))
        return Content.from_parts(parts)


class Document(Text):
    id: ID = Field(alias="doc_id", validation_alias=AliasChoices("doc_id", "docno"))
    title: str | None = None

    @property
    def doc_id(self) -> ID:
        """Property for additionally accessing the document ID."""
        return self.id

    @property
    def body(self) -> str:
        """The document text without its title: :attr:`text` is the body, :attr:`title` is metadata.

        Read alongside :attr:`title`, never merged here: how a model's input combines a title with its body
        is a formatting decision (MTEB's join: ``(title + " " + body).strip()``), made where the model's text
        is formatted, so a document reads the same whichever format held it and a recipe can place the title
        differently.
        """
        return self.text

    def model_content(self, *, title: DocumentTitle = "join") -> Content:
        """The document as the content a model reads: MTEB's join, or the title as its own part.

        ``join`` (the default) is mteb's dataloader rule (:func:`mteb_document_text`), byte-identical:
        ``(title + " " + body).strip()``, or the body alone (stripped) when there is no title. A document
        with media keeps its parts: the joined text stands where its first text part stood (the text
        parts' own join is part of the joined text), and a title on a media-only document becomes a
        leading text part. The title is never joined at read time -- this is the one join, where a
        model's text is formatted.

        ``separate`` (a model's or recipe's declared choice): the title (stripped) as its own leading
        text part, the body untouched -- mteb keeps the title as its own field too, and a model with a
        title slot takes it that way instead of inside a joined string.
        """
        content = self.as_content
        if title == "separate":
            head = (self.title or "").strip()
            if not head:
                return content  # a blank title is no title, as in the join
            return Content.from_parts([TextPart(text=head), *content.parts])
        joined = mteb_document_text(self.title, content.text)
        if not content.has_media:
            return Content.from_text(joined)
        parts: list[Part] = []
        placed = False
        for part in content.parts:
            if not isinstance(part, TextPart):
                parts.append(part)
                continue
            if not placed:
                placed = True
                if joined:
                    parts.append(TextPart(text=joined))
        if not placed and joined:
            parts.insert(0, TextPart(text=joined))
        return Content.from_parts(parts)


class RankingExample(Query):
    # ``extra="allow"`` makes the line format lossless: dataset-specific fields we
    # have no schema for (e.g. the context-compression sets carry ``answer``,
    # ``evidence``, ``evidence_intersection``, ``query_types``) survive the
    # read -> rerank -> write round trip instead of being silently dropped, so a
    # rescored file still drops straight back into its original consumer.
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True, extra="allow")

    instruction: str | None = None
    id: ID = Field(alias="query_id")
    docs: list[str] | None = None
    # The multimodal counterpart of ``docs``, aligned with ``doc_ids``.  Present
    # only when at least one document carries media, so a text-only line
    # serialises as plain ``docs`` and a text-only corpus never pays to
    # materialise a Content per document.  Read through ``doc_contents``.
    contents: list[Content] | None = None
    doc_ids: list[ID]
    qrels: dict[ID, float] | None = None
    # Optional identity bridge for a chunked corpus. Every consumed chunk ID
    # maps to the original document ID used by qrels and evaluation. The map is
    # deliberately explicit: document identity is never inferred from an ID
    # delimiter or naming convention.
    chunk_mapping: dict[ID, ID] | None = None
    scores: list[float] | None = None

    @model_validator(mode="after")
    def validate_alignment_and_sort(self) -> Self:
        """Validate aligned document fields, then sort them by descending score."""
        if len(set(self.doc_ids)) != len(self.doc_ids):
            raise ValueError("doc_ids must not contain duplicates")
        if self.docs is not None and len(self.docs) != len(self.doc_ids):
            raise ValueError(
                f"docs must align with doc_ids: len(docs)={len(self.docs)}, len(doc_ids)={len(self.doc_ids)}"
            )
        if self.contents is not None and len(self.contents) != len(self.doc_ids):
            raise ValueError(
                f"contents must align with doc_ids: len(contents)={len(self.contents)}, "
                f"len(doc_ids)={len(self.doc_ids)}"
            )
        if self.scores is not None and len(self.scores) != len(self.doc_ids):
            raise ValueError(
                f"scores must align with doc_ids: len(scores)={len(self.scores)}, len(doc_ids)={len(self.doc_ids)}"
            )

        # ``docs`` is the text view of ``contents`` whenever the latter is set,
        # so the two can never drift apart.
        if self.contents is not None:
            self.docs = [content.text for content in self.contents]

        if self.scores is not None:
            from rcp_ndcg_core._util import descending_score_order

            self._reorder(descending_score_order(self.scores))
        return self

    def _reorder(self, order: Sequence[int]) -> None:
        """Apply one permutation to every document-aligned field.

        Only *declared* fields are permuted.  An undeclared extra that happens to
        be aligned with ``doc_ids`` -- ``scores`` in a ``rerank`` file, or the
        ``evidence`` list some context-compression sets carry -- cannot be
        permuted safely, because a list of the right length is not evidence of
        alignment.  Reordering a coincidence and leaving a real alignment behind
        are both silent corruption, so instead the suspects are named in a
        warning and left untouched; declaring it as a field makes it move
        correctly.
        """
        if list(order) != list(range(len(order))):
            self._warn_about_aligned_extras()
        self.doc_ids = [self.doc_ids[i] for i in order]
        if self.docs is not None:
            self.docs = [self.docs[i] for i in order]
        if self.contents is not None:
            self.contents = [self.contents[i] for i in order]
        if self.scores is not None:
            self.scores = [self.scores[i] for i in order]

    def _warn_about_aligned_extras(self) -> None:
        n_docs = len(self.doc_ids)
        suspects = sorted(
            name
            for name, value in (self.__pydantic_extra__ or {}).items()
            if isinstance(value, list) and len(value) == n_docs
        )
        if suspects:
            logger.warning(
                f"query {self.id!r}: reordering documents, but undeclared field(s) {suspects} are the same "
                f"length as doc_ids and will keep their original order. If they are aligned with the "
                f"documents, read them through a reader that declares them; "
                f"otherwise ignore this."
            )

    @property
    def has_bodies(self) -> bool:
        """Whether the document bodies are present at all.

        What judging checks before it renders a window. Distinct from
        :attr:`has_media`, and distinct from ``docs is not None`` -- either field
        can be the one carrying them.
        """
        return self.contents is not None or self.docs is not None

    @property
    def doc_contents(self) -> list[Content]:
        """Documents as parts, whatever they were stored as.

        This is what encoders and judges consume, so neither has to branch on
        whether a dataset happened to be text-only.
        """
        if self.contents is not None:
            return self.contents
        if self.docs is None:
            raise ValueError(
                "Cannot build doc_contents: neither `contents` nor `docs` is set. "
                "Load the example with its document bodies first (a dataset loaded with its corpus)."
            )
        return [Content.from_text(doc) for doc in self.docs]

    @property
    def doc_id2content(self) -> dict[ID, Content]:
        return dict(zip(self.doc_ids, self.doc_contents, strict=True))

    @property
    def query_id(self) -> ID:
        """Property for additionally accessing the query ID."""
        return self.id

    def serialize_jsonl(self) -> str:
        """The example as one JSONL line (``None`` fields left out)."""
        return self.model_dump_json(exclude_none=True)


__all__ = [
    "ID",
    "TEXT_FORMATTING_VERSION",
    "Document",
    "DocumentTitle",
    "Input",
    "Query",
    "RankingExample",
    "Text",
    "mteb_document_text",
]
