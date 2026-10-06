"""The one contract every dataset format implements.

Adding an ingestion or an export is one class and one entry in the reader or
writer table of :mod:`rcp_ndcg.data.io`.  Nothing else in the project learns about
the new format, because nothing else in the project reads formats -- it reads
:class:`SourceReader`.

**Two shapes, one contract.**  Retrieval datasets and rerank datasets are
genuinely different objects:

:attr:`DataShape.CORPUS`
    Queries, a document corpus, and qrels.  What you run first-stage retrieval
    over.  BEIR, MTEB/HF, a directory of page images, a PDF.
:attr:`DataShape.RANKING`
    One record per query carrying its candidate list.  What you rerank or judge.
    Our own pipeline JSONL, a rerank-ready JSONL.

A reader declares which shapes it can serve and implements the corresponding
method; :class:`SourceReader` derives the other from it.  That derivation is the
point: a caller asks for the shape it needs and never branches on the format, so
a corpus source can be judged and a ranking source can be re-retrieved over
without either reader knowing the other case exists.

**Why a base class rather than a bare Protocol.**  Independent text extractors
per format drift apart, so two consumers read different text from the same row.
A shared base plus the shared conformance suite in ``tests/data/test_io_contract.py``
means there is exactly one definition of what a reader must do, and a new reader is
wrong loudly rather than subtly.
"""

from __future__ import annotations

import abc
import inspect
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from enum import StrEnum
from typing import Any, ClassVar

from rcp_ndcg_core._records import ID, Document, Query, RankingExample

from rcp_ndcg.errors import DataError
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)


class DataShape(StrEnum):
    CORPUS = "corpus"
    RANKING = "ranking"


class SourceReader(abc.ABC):
    """Reads one dataset format into the canonical types.

    Subclasses set :attr:`name`, :attr:`shapes` and ``dataset_name`` (the dataset's name, usually the file or
    directory name unless the ``name`` option overrides it), then implement
    whichever of :meth:`documents` / :meth:`examples`
    matches their native shape.

    **Every constructor's first parameter is named ``uri``** -- the source
    locator, whatever the format calls it -- because :func:`get_reader` and
    ``rcp-ndcg data convert`` pass it by keyword. A reader that names it
    something else cannot be reached through the reader table at all, so the
    conformance suite checks the signature.

    Every iterator method may be called more than once and must yield the same
    records each time.  Streaming rather than returning lists is deliberate: a
    corpus is routinely 10^7 rows and must never have to fit in memory.
    """

    #: The URI scheme of :func:`rcp_ndcg.data.load_dataset` (``beir:``) and what ``rcp-ndcg data convert --format``
    #: takes.
    name: ClassVar[str]
    #: Which shapes this reader can serve natively or by derivation.
    shapes: ClassVar[frozenset[DataShape]] = frozenset({DataShape.CORPUS, DataShape.RANKING})
    #: The dataset's name (the :class:`~rcp_ndcg.data.Dataset` it loads is called this).
    dataset_name: str

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Refuse a reader that implements neither side of the derivation.

        Each shape's default is written in terms of the other, so a subclass that
        overrides neither makes the two ask each other for the answer forever.
        Failing at class-definition time turns a ``RecursionError`` five frames
        deep into a message naming the method that is missing.
        """
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        if cls.examples is SourceReader.examples and cls.documents is SourceReader.documents:
            raise TypeError(
                f"{cls.__name__} overrides neither `examples()` (the ranking shape) nor `documents()` "
                "(the corpus shape); one of them has to be native for the other to be derived from it."
            )

    # -- corpus shape ------------------------------------------------------
    def queries(self) -> Iterator[Query]:
        """The queries. Derived from :meth:`examples` when not native.

        A reader that serves *only* the corpus shape and does not override this
        has no queries to give -- a PDF and a directory of images are corpora with
        nothing to search them for -- so this yields nothing rather than trying to
        derive queries from the ranking shape it cannot produce.
        """
        if DataShape.RANKING not in self.shapes:
            return
        seen: set[ID] = set()
        for example in self.examples():
            if example.id in seen:
                continue
            seen.add(example.id)
            yield Query(
                query_id=example.id,
                query=example.text,
                content=example.content,
                instruction=example.instruction,
            )

    def documents(self) -> Iterator[Document]:
        """The document corpus. Derived from :meth:`examples` when not native.

        The derived corpus is the union of the candidate lists, which is a
        *pool*, not the full corpus.
        """
        seen: set[ID] = set()
        for example in self.examples():
            contents = example.contents
            for index, doc_id in enumerate(example.doc_ids):
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                content = contents[index] if contents is not None else None
                text = example.docs[index] if example.docs is not None else ""
                yield Document(doc_id=doc_id, text=text, content=content)

    def qrels(self) -> dict[ID, dict[ID, float]]:
        """``{query_id: {doc_id: grade}}``, grades as floats. Derived from :meth:`examples`.

        Materialised rather than streamed: qrels are small by construction and
        every consumer wants random access. Empty for a corpus-only source, for
        the same reason :meth:`queries` is.
        """
        if DataShape.RANKING not in self.shapes:
            return {}
        out: dict[ID, dict[ID, float]] = {}
        for example in self.examples():
            if example.qrels:
                judged = out.setdefault(example.id, {})
                for doc_id, value in example.qrels.items():
                    if doc_id in judged:
                        raise DataError(
                            f"{type(self).__name__}: query {example.id!r}, document {doc_id!r} is labelled twice",
                            details={"query_id": example.id, "doc_id": str(doc_id)},
                        )
                    judged[doc_id] = float(value)
        return out

    # -- ranking shape -----------------------------------------------------
    def examples(self) -> Iterator[RankingExample]:
        """One record per query with its candidate list.

        Derived from the corpus shape by treating the judged documents as the
        candidate list, which is what makes a BEIR corpus judgeable without an
        intervening retrieval run.
        """
        if DataShape.CORPUS not in self.shapes:
            raise NotImplementedError(
                f"{type(self).__name__} declares shapes={sorted(self.shapes)} but implements neither "
                "`examples()` nor the corpus methods it would be derived from."
            )
        qrels = self.qrels()
        if not qrels:
            # The derivation *is* "the judged documents are the candidate list", so
            # nothing judged means nothing to derive.  Checked before materialising
            # the corpus: a PDF source would otherwise render 40k pages to produce
            # zero examples.
            return
        corpus = {doc.id: doc for doc in self.documents()}
        for query in self.queries():
            judged = qrels.get(query.id, {})
            doc_ids = [doc_id for doc_id in judged if doc_id in corpus]
            if not doc_ids:
                continue
            documents = [corpus[doc_id] for doc_id in doc_ids]
            # ``contents`` only when some document carries media, so a text-only
            # dataset serialises as plain ``docs``.
            media_bearing = any(doc.has_media for doc in documents)
            yield RankingExample(
                query_id=query.id,
                query=query.text,
                content=query.content,
                instruction=query.instruction,
                doc_ids=doc_ids,
                docs=None if media_bearing else [doc.text for doc in documents],
                contents=[doc.as_content for doc in documents] if media_bearing else None,
                qrels={doc_id: judged[doc_id] for doc_id in doc_ids},
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class SinkWriter(abc.ABC):
    """Writes the canonical types out in one format.

    The mirror of :class:`SourceReader`: writing a dataset in another tool's
    format is a writer, not a bespoke script per consumer.
    """

    name: ClassVar[str]
    shapes: ClassVar[frozenset[DataShape]] = frozenset({DataShape.RANKING})

    def write_examples(self, examples: Iterable[RankingExample], uri: str) -> int:
        """Write ranking records to *uri*, returning how many were written. Not every format has the shape."""
        raise NotImplementedError(f"{type(self).__name__} cannot write the ranking shape")

    def write_corpus(
        self,
        documents: Iterable[Document],
        queries: Iterable[Query],
        qrels: dict[ID, dict[ID, float]],
        uri: str,
    ) -> int:
        """Write the corpus shape. Not every format has one."""
        raise NotImplementedError(f"{type(self).__name__} cannot write the corpus shape")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


def join_title(title: Any, text: Any) -> str:
    """A document's text as the BEIR convention joins it: the title, a blank line, the body; the body alone when
    there is no title. The one join every reader uses, so a document reads the same whichever format held it.

    A title that is not a string (a NaN from a float-typed column, a number) counts as no title: ``str(nan)``
    would join the literal text ``"nan"`` in front of the body.
    """
    head = title if isinstance(title, str) else ""
    title, body = head.strip(), str(text or "")
    return f"{title}\n\n{body}" if title else body


def unique_document_ids(pairs: Iterable[tuple[str, str]], root: str) -> dict[str, str]:
    """``{document id: uri}`` from the ``(id, uri)`` pairs of the files under ``root``, in order.

    Raises:
        DataError: Two files give one id.
    """
    out: dict[str, str] = {}
    for doc_id, uri in pairs:
        if doc_id in out:
            raise DataError(
                f"{out[doc_id]} and {uri} are both document {doc_id!r} of {root}",
                hint="rename or move one of them: a document id is the file's path without its suffix",
                details={"doc_id": doc_id, "files": [out[doc_id], uri]},
            )
        out[doc_id] = uri
    return out


def grade(value: Any, *, source: str) -> float:
    """A qrels label as a float grade.

    Grades stay floats end to end: a continuous label (a gain, a half grade) is kept exactly, never floored.

    Args:
        value: The label as read (int, float or numeric string).
        source: Where it came from, for the error message.

    Returns:
        The grade.

    Raises:
        DataError: The label is not a finite number.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = math.nan
    if not math.isfinite(out):
        raise DataError(f"{source}: qrels label {value!r} is not a finite number")
    return out


def required_id(row: Mapping[str, Any], keys: Sequence[str], *, source: str, what: str) -> str:
    """The first present, non-empty id among *keys*, as a string.

    Nothing is dropped silently: a row that names no id is refused where it is read, with its file and line
    number, instead of vanishing from a corpus or a set of judgements.

    Args:
        row: The raw row as read.
        keys: The id fields the format may spell the id with, in order.
        source: The file (and line) the row was read from, for the error.
        what: What the row holds (``"a corpus row"``, ``"a qrels row"``), for the error.

    Returns:
        The id.

    Raises:
        DataError: no key holds a non-empty id.
    """
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return str(value)
    raise DataError(f"{source}: {what} carries no id (looked for {', '.join(keys)}); an id-less row is not droppable")


def sidecar_qrels(rows: Iterable[tuple[int, Mapping[str, Any]]], *, source: str) -> dict[ID, dict[ID, float]]:
    """``{query_id: {doc_id: grade}}`` from ``{"query_id", "qrels": {doc_id: grade}}`` sidecar rows.

    The one reader of the sidecar format (the jsonl corpus layout's ``qrels.jsonl``, and the image, video
    and frame directories' ``qrels_uri``): one row shape, one error contract.

    Args:
        rows: ``(line number, row)`` pairs as read.
        source: The sidecar's path, for the error messages.

    Returns:
        The qrels table.

    Raises:
        DataError: a row is not ``{'query_id', 'qrels': {doc_id: grade}}``, a row names no query, or a
            ``(query, doc)`` pair is labelled twice (nothing is cut and nothing is last-wins).
    """
    out: dict[ID, dict[ID, float]] = {}
    for line_number, row in rows:
        where = f"{source}:{line_number}"
        if "query_id" not in row or not isinstance(row.get("qrels"), dict):
            raise DataError(f"{source}: a qrels row is {{'query_id', 'qrels': {{doc_id: grade}}}}, got {row}")
        query_id = required_id(row, ("query_id", "_id", "id"), source=f"{source}:{line_number}", what="a qrels row")
        judged = out.setdefault(query_id, {})
        for doc_id, label in row["qrels"].items():
            if doc_id in judged:
                raise DataError(
                    f"{source}:{line_number}: query {query_id!r}, document {doc_id!r} is labelled twice",
                    details={"query_id": query_id, "doc_id": str(doc_id)},
                )
            judged[str(doc_id)] = grade(label, source=where)
    return out


__all__ = [
    "DataShape",
    "SinkWriter",
    "SourceReader",
    "grade",
    "join_title",
    "required_id",
    "sidecar_qrels",
    "unique_document_ids",
]
