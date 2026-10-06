"""BEIR / MTEB on-disk layout: ``corpus.jsonl``, ``queries.jsonl``, ``qrels/``.

Corpus-shaped, and the lingua franca of text retrieval evaluation.  Reading it
is unremarkable; writing it is what lets a RCP-nDCG dataset be evaluated by
any BEIR-compatible tool without a conversion script.

The one wrinkle worth naming: BEIR qrels are TSV with a header line whose first
column is ``query-id``, and the sibling ``.tsv`` convention differs between the
original BEIR release and the MTEB mirrors.  Both are accepted here so a caller
never has to know which mirror they got.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator

from rcp_ndcg_core._records import ID, Document, Query

from rcp_ndcg import storage
from rcp_ndcg.data.io.base import (
    DataShape,
    SinkWriter,
    SourceReader,
    grade,
    join_title,
    required_id,
)
from rcp_ndcg.errors import ConfigError, DataError, MissingInputError
from rcp_ndcg.storage.io import numbered_json_lines
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

CORPUS_FILENAMES = ("corpus.jsonl",)
QUERIES_FILENAMES = ("queries.jsonl",)
QRELS_CANDIDATES = ("qrels/test.tsv", "qrels/dev.tsv", "qrels/train.tsv", "qrels.tsv")


class BeirReader(SourceReader):
    """Reads a BEIR-format directory.

    Args:
        uri: The dataset directory.
        split: Which qrels split to read; ``None`` takes the first that exists.
        name: Dataset name; defaults to the directory name.
    """

    name = "beir"
    shapes = frozenset({DataShape.CORPUS, DataShape.RANKING})

    def __init__(self, uri: str, *, split: str | None = None, name: str | None = None) -> None:
        self.uri = str(uri).rstrip("/")
        self.split = split
        self.dataset_name = name or self.uri.rsplit("/", 1)[-1]

    def documents(self) -> Iterator[Document]:
        path = self._first_existing(CORPUS_FILENAMES, "corpus")
        for line_number, row in numbered_json_lines(path):
            doc_id = required_id(
                row,
                ("_id", "id", "doc_id"),
                source=f"{path}:{line_number}",
                what="a corpus row",
            )
            text = row.get("text") or ""
            yield Document(doc_id=str(doc_id), text=join_title(row.get("title"), text))

    def queries(self) -> Iterator[Query]:
        path = self._first_existing(QUERIES_FILENAMES, "queries")
        for line_number, row in numbered_json_lines(path):
            query_id = required_id(
                row,
                ("_id", "id", "query_id"),
                source=f"{path}:{line_number}",
                what="a query row",
            )
            yield Query(
                query_id=str(query_id),
                query=row.get("text") or row.get("query") or "",
                instruction=row.get("instruction"),
            )

    def qrels(self) -> dict[ID, dict[ID, float]]:
        path = self._qrels_path()
        out: dict[ID, dict[ID, float]] = {}
        with storage.open_path(path, "r") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None)
            if header is not None and not _looks_like_header(header):
                handle.seek(0)
                reader = csv.reader(handle, delimiter="\t")
            for line_number, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) < 3:
                    raise DataError(
                        f"{path}:{line_number}: a qrels row is (query-id, corpus-id, score[, ...]), got {row}"
                    )
                query_id, doc_id, score = row[0], row[1], row[2]
                judged = out.setdefault(str(query_id), {})
                if doc_id in judged:
                    raise DataError(
                        f"{path}:{line_number}: query {query_id!r}, document {doc_id!r} is labelled twice",
                        details={"query_id": str(query_id), "doc_id": str(doc_id)},
                    )
                judged[str(doc_id)] = grade(score, source=f"{path}:{line_number}")
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
        candidates = (f"qrels/{self.split}.tsv",) + QRELS_CANDIDATES if self.split else QRELS_CANDIDATES
        for filename in candidates:
            candidate = storage.join(self.uri, filename)
            if storage.exists(candidate):
                return candidate
        raise MissingInputError(
            f"no qrels under {self.uri} (looked for {', '.join(candidates)})",
            hint="a BEIR directory holds its labels in qrels/<split>.tsv (test.tsv, dev.tsv or train.tsv)",
        )


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
                handle.write(json.dumps({"_id": doc.id, "title": "", "text": doc.text}) + "\n")
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
                    # repr of a builtin float round-trips exactly; no silent 6-digit rounding.
                    handle.write(f"{query_id}\t{doc_id}\t{float(label)!r}\n")

        logger.info(f"wrote BEIR layout to {uri}: {n_docs} docs, {len(qrels)} judged queries")
        return n_docs


def _looks_like_header(row: list[str]) -> bool:
    """BEIR qrels carry a header; some mirrors do not.

    Only a row whose first cell is one of the header's column names is a header: a headerless
    file's first *data* row must reach the label checks (its ``score`` cell being a non-integer
    string once meant the row was silently eaten as a "header").
    """
    return bool(row) and row[0].strip().lower() in _HEADER_COLUMNS


_HEADER_COLUMNS = frozenset({"query-id", "qid", "query_id", "query"})


__all__ = ["BeirReader", "BeirWriter"]
