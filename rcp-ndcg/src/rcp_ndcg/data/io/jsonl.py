"""Line-delimited records -- our own native format, in two layouts.

* **ranking**: one file of :class:`RankingExample` lines, each a query with its candidate list; the format every
  pipeline stage reads and writes.
* **corpus**: a directory holding ``corpus.jsonl`` (:class:`Document` lines), ``queries.jsonl`` (:class:`Query`
  lines) and ``qrels.jsonl`` (``{"query_id", "qrels": {doc_id: grade}}`` lines), what ``rcp-ndcg data convert
  --to jsonl --shape corpus`` writes.

Media travels as :class:`~rcp_ndcg_core.content.MediaRef` inside the content parts, which is the whole reason the
wire format is references rather than bytes: a page-image dataset is still a JSONL file you can read with ``head``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from rcp_ndcg_core._records import ID, Document, Query, RankingExample

from rcp_ndcg import storage
from rcp_ndcg.data.io.base import DataShape, SinkWriter, SourceReader, sidecar_qrels
from rcp_ndcg.errors import ConfigError, MissingInputError
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)


CORPUS_FILES = ("corpus.jsonl", "queries.jsonl", "qrels.jsonl")
"""The corpus layout: the three files of a directory the jsonl writer's corpus shape produces."""


class JsonlReader(SourceReader):
    """Reads plain (uncompressed) JSONL in the ranking or the corpus layout (see the module docstring).

    Args:
        uri: A ranking JSONL file, a directory containing exactly one, or a corpus-layout directory (it holds
            ``corpus.jsonl``).
        name: The dataset name; defaults to the file's stem (everything before the ``.jsonl``/``.json``
            suffix, so ``nfcorpus.v2.jsonl`` is ``nfcorpus.v2``).

    Ranking rows are read losslessly (unknown keys survive as ``extra="allow"`` fields). Corpus rows are read
    strictly: a row whose keys are not the record's fields is refused rather than read past the unknown ones
    (a BEIR-shaped ``title`` would otherwise vanish into the text).
    """

    name = "jsonl"
    shapes = frozenset({DataShape.RANKING, DataShape.CORPUS})

    def __init__(self, uri: str, *, name: str | None = None) -> None:
        self.uri = str(uri).rstrip("/")
        self.dataset_name = name or (Path(self.uri).stem if Path(self.uri).suffix else Path(self.uri).name)

    @property
    def layout(self) -> str:
        """``corpus`` for a directory holding ``corpus.jsonl``, else ``ranking``."""
        return "corpus" if storage.exists(storage.join(self.uri, CORPUS_FILES[0])) else "ranking"

    def examples(self) -> Iterator[RankingExample]:
        from rcp_ndcg.storage.io import iter_jsonl

        if self.layout == "corpus":
            yield from super().examples()
            return
        yield from iter_jsonl(self._ranking_file(), example_class=RankingExample)

    def documents(self) -> Iterator[Document]:
        from rcp_ndcg.storage.io import iter_jsonl

        if self.layout == "corpus":
            yield from iter_jsonl(storage.join(self.uri, "corpus.jsonl"), example_class=Document, forbid_extra=True)
            return
        yield from super().documents()

    def queries(self) -> Iterator[Query]:
        from rcp_ndcg.storage.io import iter_jsonl

        if self.layout == "corpus":
            yield from iter_jsonl(storage.join(self.uri, "queries.jsonl"), example_class=Query, forbid_extra=True)
            return
        yield from super().queries()

    def qrels(self) -> dict[ID, dict[ID, float]]:
        from rcp_ndcg.storage.io import numbered_json_lines

        if self.layout != "corpus":
            return super().qrels()
        source = storage.join(self.uri, "qrels.jsonl")
        if not storage.exists(source):
            return {}
        return sidecar_qrels(numbered_json_lines(source), source=source)

    def _ranking_file(self) -> str:
        """The ranking JSONL file to read: ``uri`` itself, or the one JSONL file in the directory ``uri``."""
        if self.uri.endswith(".gz"):
            raise ConfigError(
                f"{self.uri} is compressed; the jsonl reader reads plain JSONL",
                hint=f"decompress it first: gunzip --keep {self.uri}",
            )
        if not self.uri.endswith((".jsonl", ".json")):
            candidates = [entry for entry in storage.ls(self.uri) if entry.endswith(".jsonl")]
            if len(candidates) == 1:
                return candidates[0]
            if not candidates:
                raise MissingInputError(f"no .jsonl files under {self.uri}")
            raise ConfigError(
                f"{self.uri} holds {len(candidates)} JSONL files; name one explicitly "
                f"(e.g. {candidates[0]}) rather than relying on a guess."
            )
        return self.uri


class JsonlWriter(SinkWriter):
    """Writes the ranking layout (one file of :class:`RankingExample` lines) or the corpus layout."""

    name = "jsonl"
    shapes = frozenset({DataShape.RANKING, DataShape.CORPUS})

    def write_examples(self, examples: Iterable[RankingExample], uri: str) -> int:
        count = 0
        with storage.open_path(uri, "w") as handle:
            for example in examples:
                handle.write(example.serialize_jsonl() + "\n")
                count += 1
        logger.info(f"wrote {count} ranking records to {uri}")
        return count

    def write_corpus(
        self,
        documents: Iterable[Document],
        queries: Iterable[Query],
        qrels: dict[ID, dict[ID, float]],
        uri: str,
    ) -> int:
        """Write the corpus shape as three sibling JSONL files (:data:`CORPUS_FILES`), which this reader loads.

        Separate files rather than one interleaved stream because the corpus is
        the large object and the queries are the small one, and every consumer
        reads them at different times and at different rates.
        """
        storage.makedirs(uri)
        counts: dict[str, int] = {}
        for filename, records in (("corpus.jsonl", documents), ("queries.jsonl", queries)):
            path = storage.join(uri, filename)
            written = 0
            with storage.open_path(path, "w") as handle:
                for record in records:
                    handle.write(record.model_dump_json(exclude_none=True) + "\n")
                    written += 1
            counts[filename] = written

        import json

        with storage.open_path(storage.join(uri, "qrels.jsonl"), "w") as handle:
            for query_id, judged in qrels.items():
                handle.write(json.dumps({"query_id": query_id, "qrels": judged}) + "\n")

        logger.info(f"wrote corpus to {uri}: {counts['corpus.jsonl']} docs, {counts['queries.jsonl']} queries")
        return counts["corpus.jsonl"]


__all__ = ["CORPUS_FILES", "JsonlReader", "JsonlWriter"]
