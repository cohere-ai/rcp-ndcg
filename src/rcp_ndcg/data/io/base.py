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
from collections.abc import Iterable, Iterator
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
                query=example.query or example.text,
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
                out.setdefault(example.id, {}).update({doc_id: float(g) for doc_id, g in example.qrels.items()})
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
    there is no title. The one join every reader uses, so a document reads the same whichever format held it."""
    title, body = str(title or "").strip(), str(text or "")
    return f"{title}\n\n{body}" if title else body


def iter_json_lines(uri: str) -> Iterator[dict[str, Any]]:
    """The JSON objects of a JSONL file, one per non-blank line, streamed."""
    import json

    from rcp_ndcg import storage

    with storage.open_path(uri, "r") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


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


__all__ = ["DataShape", "SinkWriter", "SourceReader", "grade"]
