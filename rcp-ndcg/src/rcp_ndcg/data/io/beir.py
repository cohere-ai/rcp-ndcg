"""BEIR / MTEB on-disk layout: ``corpus.jsonl``, ``queries.jsonl``, ``qrels/``.

Corpus-shaped, and the lingua franca of text retrieval evaluation.  Reading it
is unremarkable; writing it is what lets a RCP-nDCG dataset be evaluated by
any BEIR-compatible tool without a conversion script.  Grades are written as
integers (BEIR's own loader parses them with ``int()``); a fractional grade is
refused at write time, because no BEIR-compatible tool reads one.

The one wrinkle worth naming: BEIR qrels are TSV with a header line whose first
column is ``query-id``, and the sibling ``.tsv`` convention differs between the
original BEIR release and the MTEB mirrors.  Both are accepted here so a caller
never has to know which mirror they got.
"""

from __future__ import annotations

import csv
import io
import itertools
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from rcp_ndcg_core.records import ID, Document, Query

from rcp_ndcg import storage
from rcp_ndcg.data.io.base import (
    DataShape,
    DuplicateCounts,
    DuplicateFold,
    DuplicatesPolicy,
    Provenance,
    SinkWriter,
    SourceReader,
    grade,
    required_id,
)
from rcp_ndcg.errors import ConfigError, DataError, MissingInputError
from rcp_ndcg.storage.io import numbered_json_lines
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

CORPUS_FILENAMES = ("corpus.jsonl", "corpus.jsonl.gz")
QUERIES_FILENAMES = ("queries.jsonl", "queries.jsonl.gz")
QRELS_CANDIDATES = (
    "qrels/test.tsv",
    "qrels/dev.tsv",
    "qrels/train.tsv",
    "qrels.tsv",
    "qrels/test.tsv.gz",
    "qrels/dev.tsv.gz",
    "qrels/train.tsv.gz",
    "qrels.tsv.gz",
)


class BeirReader(SourceReader):
    """Reads a BEIR-format directory (plain or gzip-compressed: ``corpus.jsonl[.gz]``, ``queries.jsonl[.gz]``,
    ``qrels/<split>.tsv[.gz]``).

    The title is the document's own field and the body its text -- nothing joins at read time -- and the
    duplicates policy (decision 30) folds exact duplicates; a conflicting duplicate refuses, and
    ``duplicates='last'`` resolves it for the labels (the corpus and queries stream, so a conflict there
    refuses even under ``last``).

    Args:
        uri: The dataset directory.
        split: Which qrels split to read; ``None`` takes the first conventional one that exists. A requested
            split must exist as ``qrels/<split>.tsv[.gz]``: another split's labels are never served for it.
        duplicates: What a conflicting duplicate does (see :class:`~rcp_ndcg.data.io.base.DuplicatesPolicy`).
        name: Dataset name; defaults to the directory name.
    """

    name = "beir"
    shapes = frozenset({DataShape.CORPUS, DataShape.RANKING})

    def __init__(
        self,
        uri: str,
        *,
        split: str | None = None,
        duplicates: str | DuplicatesPolicy = DuplicatesPolicy.ERROR,
        name: str | None = None,
    ) -> None:
        self.uri = str(uri).rstrip("/")
        self.split = split
        self.duplicates_policy = DuplicatesPolicy(duplicates)
        self.dataset_name = name or self.uri.rsplit("/", 1)[-1]
        self._counts = DuplicateCounts(policy=self.duplicates_policy)
        self._counted: set[str] = set()

    def _note(self, what: str, fold: DuplicateFold) -> None:
        """Record one table's duplicates policy: merged into the reader's totals on its first pass (so a
        re-read never double-counts), and logged whenever it did anything."""
        counts = fold.counts()
        if what not in self._counted:
            self._counted.add(what)
            self._counts = DuplicateCounts(
                policy=self.duplicates_policy,
                folded=self._counts.folded + counts.folded,
                resolved=self._counts.resolved + counts.resolved,
            )
        if counts.folded or counts.resolved:
            logger.info(
                f"{self.uri}: {what} duplicates: {counts.folded} exact folded, "
                f"{counts.resolved} resolved by the {fold.policy.value} policy"
            )

    def documents(self) -> Iterator[Document]:
        path = self._first_existing(CORPUS_FILENAMES, "corpus")
        fold = DuplicateFold(self.duplicates_policy, source=self.uri, what="corpus row", replaceable=False)
        for line_number, row in numbered_json_lines(path):
            doc_id = required_id(
                row,
                ("_id", "id", "doc_id"),
                source=f"{path}:{line_number}",
                what="a corpus row",
            )
            title = row.get("title")
            text = row.get("text") or ""
            document = Document(
                doc_id=str(doc_id),
                title=title if isinstance(title, str) and title else None,
                text=str(text),
            )
            if fold.add(str(doc_id), (title if isinstance(title, str) else None, str(text))):
                yield document
        self._note("corpus", fold)

    def queries(self) -> Iterator[Query]:
        path = self._first_existing(QUERIES_FILENAMES, "queries")
        fold = DuplicateFold(self.duplicates_policy, source=self.uri, what="query row", replaceable=False)
        for line_number, row in numbered_json_lines(path):
            query_id = required_id(
                row,
                ("_id", "id", "query_id"),
                source=f"{path}:{line_number}",
                what="a query row",
            )
            text = row.get("text") or row.get("query") or ""
            instruction = row.get("instruction")
            query = Query(query_id=str(query_id), query=str(text), instruction=instruction)
            if fold.add(str(query_id), (str(text), instruction if isinstance(instruction, str) else None)):
                yield query
        self._note("queries", fold)

    def qrels(self) -> dict[ID, dict[ID, float]]:
        path = self._qrels_path()
        out: dict[ID, dict[ID, float]] = {}
        fold = DuplicateFold(self.duplicates_policy, source=path, what="qrels label")
        # A gzip-compressed qrels file reads through fsspec's own decompression, local or remote; the stream
        # is opened binary and wrapped, so the compressed and plain paths read the very same way (no seek).
        compression = {"compression": "gzip"} if path.lower().endswith(".gz") else {}
        with storage.open_path(path, "rb", **compression) as raw:
            lines = iter(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            first = next(lines, None)
            rows = lines if first is None or _looks_like_header(first.split("\t")) else itertools.chain([first], lines)
            for line_number, row in enumerate(csv.reader(rows, delimiter="\t"), start=1):
                if not row:
                    continue
                if len(row) < 3:
                    raise DataError(
                        f"{path}:{line_number}: a qrels row is (query-id, corpus-id, score[, ...]), got {row}"
                    )
                query_id, doc_id, score = row[0], row[1], row[2]
                label = grade(score, source=f"{path}:{line_number}")
                if fold.add(f"{query_id}/{doc_id}", (label,)):
                    out.setdefault(str(query_id), {})[str(doc_id)] = label
        self._note("qrels", fold)
        return out

    def _first_existing(self, names: tuple[str, ...], what: str) -> str:
        for filename in names:
            candidate = storage.join(self.uri, filename)
            if storage.exists(candidate):
                return candidate
        raise MissingInputError(
            f"no {what} file under {self.uri} (looked for {', '.join(names)})",
            hint="a BEIR directory holds corpus.jsonl, queries.jsonl and qrels/*.tsv; "
            "convert one with `rcp-ndcg data convert --from jsonl --to beir`",
        )

    def _qrels_path(self) -> str:
        """The labels file of the requested split, never another split's.

        A requested split is looked up as ``qrels/<split>.tsv`` and ``qrels/<split>.tsv.gz`` only: falling
        through to the conventional list used to return ``qrels/test.tsv`` for ``split='dev'`` (or for any
        split the directory lacks), so a user tuning on dev silently evaluated test. Without a requested
        split the first conventional file wins, as documented.
        """
        candidates = (f"qrels/{self.split}.tsv", f"qrels/{self.split}.tsv.gz") if self.split else QRELS_CANDIDATES
        for filename in candidates:
            candidate = storage.join(self.uri, filename)
            if storage.exists(candidate):
                return candidate
        existing = self._existing_qrels_splits()
        hint = "a BEIR directory holds its labels in qrels/<split>.tsv (test.tsv, dev.tsv or train.tsv)"
        if existing:
            hint = f"the qrels files under {self.uri} name the splits {existing}; read one of them"
        raise MissingInputError(
            f"no qrels under {self.uri}"
            + (f" for the split {self.split!r}" if self.split else "")
            + f" (looked for {', '.join(candidates)})",
            hint=hint,
        )

    def _existing_qrels_splits(self) -> list[str]:
        """The split names the directory's ``qrels/`` files state, for the missing-split hint (best effort:
        a listing that fails must not mask the missing-file error)."""
        try:
            names = storage.ls(storage.join(self.uri, "qrels"))
        except Exception:  # noqa: BLE001 - the hint is a nicety; the missing file is the error
            return []
        splits = [
            Path(Path(name).name.removesuffix(".gz")).stem
            for name in names
            if Path(name).name.endswith((".tsv", ".tsv.gz"))
        ]
        return sorted(set(splits))

    @property
    def provenance(self) -> Provenance:
        """The BEIR directory, the split the labels were read at, and the duplicates policy with counts.

        The counts are the tables read so far: a load reads the labels first, so a dataset's provenance records
        the label folds; the corpus and query folds are logged as they happen (they are read on demand).
        """
        try:
            split = _split_of_qrels_file(self._qrels_path())
        except MissingInputError:
            split = self.split or "test"
        return Provenance(source_uri=self.uri, subset="default", split=split, duplicates=self._counts)


class BeirWriter(SinkWriter):
    """Writes the BEIR layout, so any BEIR-compatible tool can read our datasets."""

    name = "beir"
    shapes = frozenset({DataShape.CORPUS})

    def write_corpus(
        self,
        documents: Iterable[Document],
        queries: Iterable[Query],
        qrels: dict[ID, dict[ID, float]],
        uri: str,
    ) -> int:
        storage.makedirs(uri)
        n_docs = 0
        with storage.open_path(storage.join(uri, "corpus.jsonl"), "w") as handle:
            for doc in documents:
                if doc.has_media:
                    raise ConfigError(
                        f"document {doc.id!r} carries media, which the BEIR format cannot express. "
                        "Export to `jsonl` instead, which keeps media as references."
                    )
                row = {"_id": doc.id, "title": doc.title or "", "text": doc.text}
                handle.write(json.dumps(row) + "\n")
                n_docs += 1

        with storage.open_path(storage.join(uri, "queries.jsonl"), "w") as handle:
            for query in queries:
                if query.has_media:
                    raise ConfigError(
                        f"query {query.id!r} carries media, which the BEIR format cannot express. "
                        "Export to `jsonl` instead, which keeps media as references."
                    )
                row: dict[str, str] = {"_id": query.id, "text": query.text}
                if query.instruction:
                    row["instruction"] = query.instruction  # the reader restores it
                handle.write(json.dumps(row) + "\n")

        storage.makedirs(storage.join(uri, "qrels"))
        with storage.open_path(storage.join(uri, "qrels", "test.tsv"), "w") as handle:
            handle.write("query-id\tcorpus-id\tscore\n")
            for query_id, judged in qrels.items():
                for doc_id, label in judged.items():
                    handle.write(f"{query_id}\t{doc_id}\t{_grade_cell(label, query_id, doc_id)}\n")

        logger.info(f"wrote BEIR layout to {uri}: {n_docs} docs, {len(qrels)} judged queries")
        return n_docs


def _grade_cell(label: float, query_id: ID, doc_id: ID) -> str:
    """One qrels score cell: an integral grade as an int, a fractional grade refused.

    BEIR's own loader parses ``int(row[2])``, so ``repr(1.0)`` made every export unreadable by BEIR itself.
    An integral grade is written as its int; a fractional grade is refused with a hint, because a ``0.7``
    cell is a non-standard file no BEIR-compatible tool reads (the reader keeps fractional labels; the
    ``jsonl`` and ``mteb`` writers carry the continuous signal).
    """
    value = float(label)
    if value.is_integer():
        return str(int(value))
    raise DataError(
        f"qrels: query {query_id!r}, document {doc_id!r}: the grade {label!r} is not a whole number, and "
        "BEIR's own loader parses qrels scores with int()",
        hint="BEIR qrels are integers: export as `jsonl` (which keeps float grades) or as `mteb` (whose "
        "gain/theta columns carry the continuous signal)",
        details={"query_id": str(query_id), "doc_id": str(doc_id), "grade": value},
    )


def _split_of_qrels_file(path: str) -> str:
    """The split a qrels file's name states: ``qrels/dev.tsv[.gz]`` is dev; a root-level ``qrels.tsv`` is the
    conventional test split (its own stem, "qrels", is not a split)."""
    name = Path(path.removesuffix(".gz")).name
    return "test" if name == "qrels.tsv" else Path(name).stem


def _looks_like_header(row: list[str]) -> bool:
    """BEIR qrels carry a header; some mirrors do not.

    Only a row whose first cell is one of the header's column names is a header: a headerless
    file's first *data* row must reach the label checks (its ``score`` cell being a non-integer
    string once meant the row was silently eaten as a "header").
    """
    return bool(row) and row[0].strip().lower() in _HEADER_COLUMNS


_HEADER_COLUMNS = frozenset({"query-id", "qid", "query_id", "query"})


__all__ = ["BeirReader", "BeirWriter"]
