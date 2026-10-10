"""The Hub reader behind ``hf://<owner>/<repo>[/<subset>][@revision]``: mteb's layout rules, no
``datasets`` and no ``mteb`` dependency.

The dataset card (its YAML ``configs`` header) declares where every table lives; the reader resolves it the
way mteb's own :class:`~mteb.abstasks.retrieval_dataset_loaders.RetrievalDatasetLoader` does, and reads the
files it names directly through ``huggingface_hub`` (download once, at one commit) and pyarrow:

* **Subsets** are the config names' prefixes: ``{s}-corpus`` / ``{s}-queries`` / ``{s}-qrels`` name subset
  ``s``; the unprefixed names (``corpus``, ``queries``, ``default``, ``qrels``, ``top_ranked``,
  ``instruction``) belong to the subset ``default``. Every other config (``qrel_diff``, ``documents``, our
  ``provenance``, ...) the loader ignores, exactly like mteb.
* **Config resolution** per subset ``s`` (the prefix is dropped for ``default``): the corpus config is
  ``{s-}corpus`` (required); the queries config ``{s-}queries``, except that a config named exactly ``query``
  always wins, whatever the subset (mteb's Any2Any layout); the qrels config ``{s-}qrels``, with the
  ``default``-then-``qrels`` fallback in the unprefixed case; ``top_ranked``, ``instruction`` and our
  ``-excluded`` are optional, under their exact names.
* **Split resolution** per config: the requested split when the config declares it, else the config's only
  split, else an error naming the splits (mteb's rule; this is why the BEIR ``corpus``/``queries`` splits
  work).
* **Columns** are normalised: ``_id`` becomes ``id`` and every id is a string; a corpus row's ``title`` is
  the document's :attr:`~rcp_ndcg_core.records.Document.title` and ``text`` its body -- nothing joins at
  read time; ``instruction`` merges by id, the config's rows winning over the queries' own column (as in
  mteb); ``top_ranked`` becomes the candidates; the media columns ``image`` and ``video`` become content
  parts (audio is deferred). Queries are cut to those with qrels, as mteb cuts them.
* **rcp-ndcg's extras** are read where present: the qrels table's ``gain``/``theta`` columns (the released
  calibrated gains over the judged pool) and the ``-excluded`` config (ids removed from rankings and ideals).
  A repository that also carries its tables in the plain ``{subset}/`` path layout (the released rcp-ndcg
  layout before its cards declared every table) is served through those conventional paths when the card
  does not declare the table.

Files are read at the commit the revision resolves to, online or from the local cache (see
:func:`rcp_ndcg.data.revisions.resolve_revision`); the absence and offline semantics are the ones the
released datasets have always been read with (:meth:`HubReader._local`).
"""

from __future__ import annotations

import csv
import fnmatch
import gzip
import json
import math
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rcp_ndcg_core.content import Content, ImagePart, MediaRef, TextPart, VideoPart
from rcp_ndcg_core.records import ID, Document, Query

from rcp_ndcg import storage
from rcp_ndcg.data.io.base import (
    DataShape,
    DuplicateCounts,
    DuplicateFold,
    DuplicatesPolicy,
    Provenance,
    SourceReader,
    grade,
    lift_task_instruction,
    required_id,
)
from rcp_ndcg.data.media import (
    IMAGE_MIME_BY_SUFFIX,
    VIDEO_MIME_BY_SUFFIX,
    image_dimensions,
    media_extension,
    store_media,
)
from rcp_ndcg.data.revisions import _hub_cache_dir, _hub_offline, is_commit, resolve_revision
from rcp_ndcg.errors import (
    ConfigError,
    DataError,
    MissingInputError,
    ProviderError,
    RcpNdcgError,
    RcpNdcgWarning,
    classify,
)
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

DocumentParts = Literal["auto", "text", "image", "both"]
DOCUMENT_PARTS: tuple[DocumentParts, ...] = ("auto", "text", "image", "both")
"""Which of a corpus row's columns to read: everything present (``auto``), the text columns only, the media
columns only, or both, which refuses a corpus with either absent (ViDoRe v3 ships OCR text and page images in
one row; the two are different experiments over the same corpus)."""

MEDIA_COLUMNS = ("image", "video")
"""The media columns of a corpus or query row; the names mteb's loader reserves (audio is deferred)."""

_PART_SUFFIXES = ("-corpus", "-queries", "-qrels", "-top_ranked", "-instruction", "-excluded")
_UNPREFIXED_PARTS = frozenset(
    {"corpus", "queries", "query", "default", "qrels", "top_ranked", "instruction", "excluded"}
)


@dataclass(frozen=True)
class _Config:
    """One config of a dataset card: its name, the ``(split, pattern)`` entries of its ``data_files``, and
    the feature (column) names the card's ``dataset_info`` declares for it."""

    name: str
    entries: tuple[tuple[str | None, str], ...]
    features: frozenset[str] = frozenset()


class HubReader(SourceReader):
    """Reads a HuggingFace dataset repository at one commit, through the layout its card declares.

    Args:
        uri: ``<owner>/<repo>[/<subset>][@revision]`` (the locator after the ``hf://`` scheme). A subset in
            the URI overrides the ``subset`` option, a revision in the URI the ``revision`` option.
        subset: The subset (mteb's ``hf_subset``); ``None`` names the repository's default subset.
        revision: The Hub revision (commit, tag or branch); ``None`` is ``main``, resolved to a commit once.
        split: The split to read; ``None`` is ``"test"``, resolved per config (the requested one when the
            config declares it, else the config's only split).
        document_parts: Which of a corpus row's columns to read (see :data:`DOCUMENT_PARTS`).
        duplicates: What a conflicting duplicate does (see :class:`~rcp_ndcg.data.io.base.DuplicatesPolicy`).
        media_out_uri: Where decoded media is written; ``None`` writes into the package's content-addressed
            media cache, so reading the reference back needs no second copy.
        name: The dataset name; defaults to the subset's name, or the repository's last segment when no
            subset is named.

    Raises:
        ConfigError: The URI is not ``<owner>/<repo>[/<subset>][@revision]``, or names two subsets or two
            revisions, or an unknown ``document_parts`` or ``duplicates`` value.
    """

    name = "hf"
    shapes = frozenset({DataShape.CORPUS, DataShape.RANKING})

    def __init__(
        self,
        uri: str,
        *,
        subset: str | None = None,
        revision: str | None = None,
        split: str | None = None,
        name: str | None = None,
        document_parts: DocumentParts = "auto",
        duplicates: str | DuplicatesPolicy = DuplicatesPolicy.ERROR,
        media_out_uri: str | None = None,
    ) -> None:
        if document_parts not in DOCUMENT_PARTS:
            raise ConfigError(f"document_parts must be one of {DOCUMENT_PARTS}, got {document_parts!r}")
        self.document_parts: DocumentParts = document_parts
        try:
            self.duplicates_policy = DuplicatesPolicy(duplicates)
        except ValueError as exc:
            raise ConfigError(
                f"duplicates must be one of {[policy.value for policy in DuplicatesPolicy]}, got {duplicates!r}",
            ) from exc
        self.media_out_uri = media_out_uri

        path, _, uri_revision = uri.partition("@")
        if uri_revision and revision and uri_revision != revision:
            raise ConfigError(f"two revisions for one dataset: {uri_revision!r} in the URI and {revision!r}")
        revision = revision or uri_revision or None
        parts = path.strip("/").split("/")
        if len(parts) not in (2, 3):
            raise ConfigError(f"a Hub dataset is 'hf://<owner>/<repo>[/<subset>]', got 'hf://{uri}'")
        self.repo = "/".join(parts[:2])
        if len(parts) == 3:
            if subset is not None and subset != parts[2]:
                raise ConfigError(f"two subsets for one dataset: {parts[2]!r} in the URI and {subset!r}")
            subset = parts[2]
        self.subset = subset or "default"
        self.dataset_name = name or subset or self.repo.rsplit("/", 1)[-1]
        self.split = split or "test"
        # Every file is read at one commit, the one the identities record, even if the branch moves meanwhile.
        self.commit: str | None = resolve_revision(self.repo, revision).commit or revision

        self._listing: list[str] | None = None
        self._configs: dict[str, _Config] | None = None
        self._labels: _Labels | None = None
        self._instruction_rows: dict[str, str] = {}
        self._instruction_rows_complete = True
        self._instruction_rows_read = False
        self._counts = DuplicateCounts(policy=self.duplicates_policy)
        self._counted: set[str] = set()

    # -- locating the tables -------------------------------------------------

    @property
    def source(self) -> str:
        """Where this reader's errors locate themselves."""
        return f"hf://{self.repo}/{self.subset}"

    def _files(self) -> list[str]:
        """Every file path of the repository at the commit (the listing, read once per reader)."""
        if self._listing is None:
            self._listing = _hub_listing(self.repo, self.commit)
        return self._listing

    def _configs_of_card(self) -> dict[str, _Config]:
        """The card's configs, parsed once; ``{}`` for a repository without a card (or without configs)."""
        if self._configs is None:
            self._configs = _card_configs(self.repo, self.commit)
        return self._configs

    def _config_for(self, part: str) -> _Config | None:
        """The card config holding *part* of this reader's subset (mteb's resolution, see the module docstring)."""
        configs = self._configs_of_card()
        if part == "queries" and "query" in configs:  # the `query` override ignores the subset, as in mteb
            return configs["query"]
        prefix = "" if self.subset == "default" else f"{self.subset}-"
        if part == "qrels":
            names = (f"{prefix}qrels",) if prefix else ("default", "qrels")
        else:
            names = (f"{prefix}{part}",)
        return next((configs[found] for found in names if found in configs), None)

    def _patterns(self, part: str) -> tuple[str, ...]:
        """The file patterns of one part: its card config's, for the resolved split; else the conventional
        path. Whether the repository has the table is decided by the download, never by a listing.

        A card config the card lists only under ``dataset_info`` (its features, no ``data_files``) names no
        file: it does not shadow the conventional ``{subset}/{part}.parquet`` path, it falls back to it.
        """
        config = self._config_for(part)
        if config is not None and config.entries:
            patterns = _patterns_for_split(self.source, config, self.split)
            if not patterns:
                raise MissingInputError(
                    f"{self.source}: the {config.name!r} config names no file for the split {self.split!r}",
                    hint="check the split (the config declares the splits its data files are in)",
                    details={"repo": self.repo, "subset": self.subset, "config": config.name, "split": self.split},
                )
            return patterns
        # An exact conventional path is decided by the download (the cache alone offline), never by a listing.
        return (f"{self.subset}/{part}.parquet",)

    def _paths(self, part: str, *, optional: bool) -> list[tuple[str, Path]]:
        """The local copies of one part's files (with their repository-relative paths), in a stable order.

        An exact path (no glob) is downloaded directly -- offline, the cache alone decides, never a listing
        request; only a card pattern with wildcards consults the file listing, whose offline snapshot stands
        in with the ``SNAPSHOT_LISTING`` warning (a partial cache reads as missing data).
        """
        patterns = self._patterns(part)
        if not any(char in pattern for pattern in patterns for char in "*?["):
            found: list[tuple[str, Path]] = []
            for path in patterns:
                local = self._local(path, optional=optional)
                if local is not None:
                    found.append((path, local))
            return found
        matched = sorted(file for file in self._files() if any(fnmatch.fnmatch(file, pattern) for pattern in patterns))
        if not matched:
            raise MissingInputError(
                f"{self.source}: no {part} file matches {', '.join(patterns)}",
                hint="the card's patterns name files the repository does not have at this revision",
                details={"repo": self.repo, "subset": self.subset, "patterns": list(patterns)},
            )
        locals_: list[tuple[str, Path]] = []
        for path in matched:
            local = self._local(path, optional=optional)
            if local is not None:
                locals_.append((path, local))
        return locals_

    def _local(self, path: str, *, optional: bool) -> Path | None:
        """One file of the repository at the commit, downloaded once; ``None`` when the repository has none.

        An offline (or unreachable-Hub) cache miss raises the typed failure :func:`classify` picks from the
        download's cause, with the file's location in ``details``. An optional file is ``None`` when the
        repository has none: online the Hub answers 404; offline only the cache's own ``.no_exist`` record
        (written by an online download) counts as that, and a file the cache knows nothing about raises like a
        required file, never a silent absence.
        """

        def missing() -> MissingInputError:
            """The repository has no such file, as the Hub's 404 or the cache's mark says."""
            return MissingInputError(
                f"hf://{self.repo}: {path} does not exist" + (f" at {self.commit}" if self.commit else ""),
                hint="check the subset and the revision: the repository has no such file at it",
                details={"repo": self.repo, "path": path, "revision": self.commit},
            )

        try:
            local = _hub_file(self.repo, path, self.commit)
        except _local_entry_not_found() as exc:
            if _hub_absent(self.repo, path, self.commit):
                if optional:
                    return None
                raise missing() from exc
            raise _hub_miss(exc, self.repo, path, self.commit) from exc
        if local is None:
            if optional:
                return None
            raise missing()
        return local

    # -- the rows of one table ---------------------------------------------
    def _rows(self, part: str, *, optional: bool = False) -> Iterator[Mapping[str, Any]]:
        """The rows of one part, across its files, in a stable order (parquet, jsonl, jsonl.gz, tsv, tsv.gz)."""
        for path, local in self._paths(part, optional=optional):
            yield from _iter_table_rows(local, source=f"hf://{self.repo}/{path}")

    # -- the corpus shape --------------------------------------------------
    def documents(self) -> Iterator[Document]:
        """The corpus: ``title`` as its own field, the body in ``text``, the media columns as content parts.

        ``document_parts`` declares what a multi-column corpus is read as (see :data:`DOCUMENT_PARTS`) -- the
        check runs on the first row, where a corpus that lacks what was asked for refuses loudly instead of
        reporting a visual number for an OCR run. Exact duplicate rows fold into the first occurrence
        (decision 30), counted in a log note; a conflicting duplicate refuses, naming the rows -- a corpus
        streams, so even ``duplicates="last"`` cannot replace a row it has already yielded (the labels, the
        pools and the exclusions take the last row).
        """
        fold = self._fold("corpus row", replaceable=False)
        checked = False
        for row in self._rows("corpus"):
            if not checked:
                checked = True
                self._check_document_parts(row)
            document = self._document(row)
            if fold.add(document.id, self._fingerprint(row)):
                yield document
        if checked:
            self._note_row_duplicates("corpus", fold)

    def _check_document_parts(self, row: Mapping[str, Any]) -> None:
        """Refuse a corpus that lacks what ``document_parts`` asks for (nothing is silently substituted)."""
        if self.document_parts in ("text", "both") and "text" not in row:
            raise DataError(
                f"{self.source}: document_parts={self.document_parts!r} needs a text column, and the corpus "
                "rows have none"
            )
        if self.document_parts in ("image", "both") and not any(column in row for column in MEDIA_COLUMNS):
            raise DataError(
                f"{self.source}: document_parts={self.document_parts!r} needs a media column "
                f"({', '.join(MEDIA_COLUMNS)}), and the corpus rows have none"
            )

    def _document(self, row: Mapping[str, Any]) -> Document:
        """A corpus row (``id``/``_id``, ``title``, ``text``, media columns) as a document."""
        doc_id = required_id(row, ("_id", "id"), source=self.source, what="a corpus row")
        title = _text_or_none(row.get("title"))
        media = self._media_of(row)
        body = "" if self.document_parts == "image" else _text_of(row, "text", what="a corpus row", source=self.source)
        if not media:
            return Document(doc_id=doc_id, title=title, text=body)
        parts: list[TextPart | ImagePart | VideoPart] = [TextPart(text=body)] if body else []
        parts.extend(media)
        return Document(doc_id=doc_id, title=title, content=Content.from_parts(parts))

    def queries(self) -> Iterator[Query]:
        """The queries, cut to those with qrels (mteb's rule), their ``instruction`` merged by id.

        The instruction config's rows win over the queries' own ``instruction`` column (as in mteb); a query
        with no instruction row is refused when a config exists, as mteb refuses it. A uniform instruction
        (every query of the subset carrying the same one) is lifted to :attr:`task_instruction` instead --
        BRIGHT's per-domain instructions are task instructions stored per query -- and the queries then
        carry none.
        """
        instructions = self._instructions()
        labels = self.qrels()
        fold = self._fold("query row", replaceable=False)
        for row in self._rows("queries"):
            query_id = required_id(row, ("_id", "id"), source=self.source, what="a query row")
            if query_id not in labels:
                continue  # mteb cuts the queries to those with qrels
            query = self._query(query_id, row, instructions(query_id, row))
            if fold.add(query_id, self._fingerprint(row)):
                yield query
        self._note_row_duplicates("query", fold)

    def _query(self, query_id: str, row: Mapping[str, Any], instruction: str | None) -> Query:
        text = _text_of(row, "text", what="a query row", source=self.source)
        media = self._media_of(row, corpus=False)
        if not media:
            return Query(query_id=query_id, query=text, instruction=instruction)
        parts: list[TextPart | ImagePart | VideoPart] = [TextPart(text=text)] if text else []
        parts.extend(media)
        return Query(query_id=query_id, content=Content.from_parts(parts), instruction=instruction)

    def _instructions(self) -> Callable[[str, Mapping[str, Any]], str | None]:
        """The instruction lookup: the instruction config's rows when the repository has one -- the only
        source then, as in mteb, which merges by id and refuses a query the config does not instruct.

        A uniform table (every query of the subset carrying the same instruction) is lifted to
        :attr:`task_instruction` (:func:`lift_task_instruction`); the lookup then answers ``None`` for every
        query, so none carries it twice. The refusals stay where they were: a query the config does not
        instruct, and a column that instructs only some queries, are both refused when the queries are read
        (the load stays lazy).
        """
        table, complete = self._instruction_table()
        lifted = self.task_instruction is not None
        if self._config_for("instruction") is None:

            def column_lookup(_query_id: str, row: Mapping[str, Any]) -> str | None:
                if not complete:
                    raise DataError(
                        f"{self.source}: {len(table)} of this subset's queries carry an instruction and the rest "
                        "do not (a mixed subset): an instruction is either the whole task's or every query's own",
                        details={"repo": self.repo, "subset": self.subset},
                    )
                return None if lifted else _text_or_none(row.get("instruction"))

            return column_lookup

        def lookup(query_id: str, row: Mapping[str, Any]) -> str | None:
            if query_id not in table:
                raise DataError(
                    f"{self.source}: query {query_id!r} has no row in the instruction config (as in mteb, a "
                    "repository with an instruction config instructs every query)",
                    details={"repo": self.repo, "subset": self.subset, "query_id": query_id},
                )
            return None if lifted else table[query_id]

        return lookup

    def _instruction_table(self) -> tuple[dict[str, str], bool]:
        """``(table, complete)``: every query's instruction the repository carries, read once, and whether
        every query of the subset carries one.

        The instruction config's rows when the repository has one (the source mteb merges by id), else the
        queries' own ``instruction`` column -- read only when the card declares that column for the subset's
        queries config, so a repository without instructions never pays for a table read (the load stays
        lazy). ``({}, True)`` when the repository carries no instruction at all; ``complete`` is ``True``
        for the config form, whose missing rows the lookup refuses lazily, and for the column form it is
        decided over the queries mteb keeps (those with qrels).
        """
        if self._instruction_rows_read:
            return self._instruction_rows, self._instruction_rows_complete
        self._instruction_rows_read = True
        rows: dict[str, str] = {}
        complete = True
        if self._config_for("instruction") is not None:
            for row in self._rows("instruction"):
                query_id = required_id(row, ("query-id", "query_id"), source=self.source, what="an instruction row")
                text = _text_or_none(row.get("instruction"))
                if text is None:
                    raise DataError(
                        f"{self.source}: the instruction row of query {query_id!r} carries no instruction text",
                        details={"repo": self.repo, "subset": self.subset, "query_id": query_id},
                    )
                rows[query_id] = text
        elif self._declares_instruction_column():
            labelled = self.qrels()
            for row in self._rows("queries"):
                query_id = required_id(row, ("_id", "id"), source=self.source, what="a query row")
                if query_id not in labelled:
                    continue  # mteb cuts the queries to those with qrels; an unlabelled row is never read
                text = _text_or_none(row.get("instruction"))
                if text is not None:
                    rows[query_id] = text
            # An empty table is no instruction source at all (nothing to be incomplete about); a non-empty one
            # that does not cover every KEPT query is a mixed subset. Completeness is decided over the queries
            # mteb keeps (those with qrels): a row the qrels cut drops is never read, so its missing instruction
            # cannot make a coherent subset look mixed.
            complete = not rows or all(query_id in rows for query_id in labelled)
        self._instruction_rows = rows
        self._instruction_rows_complete = complete
        return rows, complete

    def _declares_instruction_column(self) -> bool:
        """Whether this subset's queries table carries an ``instruction`` column.

        The card's ``dataset_info`` features answer it without reading the table; a card that declares none
        (the released rcp-ndcg repositories' cards) is answered from the first queries file's own schema or
        header -- one small file, never the rows. A repository without a queries table answers ``False``.
        """
        config = self._config_for("queries")
        if config is not None and config.features:
            return "instruction" in config.features
        return "instruction" in self._columns("queries")

    def _columns(self, part: str) -> frozenset[str]:
        """The column names of a table's first file, from its schema or header (parquet's schema, jsonl's
        first line, tsv's header) -- what a card that declares no features leaves the reader to discover,
        without reading the table's rows.

        Empty when the repository has no such table *or* the file cannot be served right now (an offline
        cache miss, an uncached card pattern): the decision -- whether there is an instruction column to
        lift -- is then simply "none", and a read that truly needs the table raises its own typed error
        where it is read (the load stays lazy, and an offline scoring pass that never touches the queries
        is not failed by a lookup for a column it does not have).
        """
        try:
            paths = self._paths(part, optional=True)
        except MissingInputError:
            return frozenset()
        if not paths:
            return frozenset()
        local = paths[0][1]
        name = local.name.lower()
        if name.endswith(".parquet"):
            import pyarrow.parquet as pq

            return frozenset(pq.ParquetFile(local).schema_arrow.names)
        opener = gzip.open if name.endswith(".gz") else open
        if name.endswith((".jsonl", ".json", ".jsonl.gz")):
            with opener(local, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        return frozenset(row) if isinstance(row, dict) else frozenset()
            return frozenset()
        if name.endswith((".tsv", ".tsv.gz")):
            with opener(local, "rt", encoding="utf-8", newline="") as handle:  # type: ignore[operator]
                return frozenset(next(csv.reader(handle, delimiter="\t"), []))
        return frozenset()

    # -- the labels --------------------------------------------------------
    def qrels(self) -> dict[ID, dict[ID, float]]:
        """``{query_id: {doc_id: grade}}``, the float grades of the labels table."""
        return self._labels_table().qrels

    def gains(self) -> dict[ID, dict[ID, float]] | None:
        """The released calibrated gains (the qrels table's ``gain`` column), when the source has them."""
        return self._labels_table().gains or None

    def thetas(self) -> dict[ID, dict[ID, float]] | None:
        """The abilities behind :meth:`gains` (the ``theta`` column), in logits."""
        return self._labels_table().thetas or None

    def _labels_table(self) -> _Labels:
        """The labels table, parsed once: grades, the gain/theta extras where present, duplicates folded.

        A ``(query, document)`` pair labelled twice with the same grade folds (decision 30); a conflict
        refuses, naming the rows, unless the reader's policy is ``last`` -- mteb's own behaviour, which takes
        the last value. The counts land in :attr:`provenance`.
        """
        if self._labels is None:
            qrels: dict[str, dict[str, float]] = {}
            gains: dict[str, dict[str, float]] = {}
            thetas: dict[str, dict[str, float]] = {}
            fold = DuplicateFold(self.duplicates_policy, source=self.source, what="qrels label")
            has_gains: bool | None = None
            for row in self._rows("qrels"):
                query_id = required_id(row, ("query-id", "query_id"), source=self.source, what="a qrels row")
                doc_id = required_id(row, ("corpus-id", "corpus_id"), source=self.source, what="a qrels row")
                label = grade(row.get("score"), source=f"{self.source}: qrels of {query_id!r}/{doc_id!r}")
                if fold.add(f"{query_id}/{doc_id}", (label,)):
                    qrels.setdefault(query_id, {})[doc_id] = label
                    if has_gains is None:
                        has_gains = "gain" in row and "theta" in row
                    if has_gains and row.get("theta") == row.get("theta"):  # null outside the judged pool
                        gain, theta = row["gain"], row["theta"]
                        if not _is_finite_number(gain) or not 0.0 <= float(gain) <= 1.0:
                            raise DataError(
                                f"{self.source}: gain {gain!r} of query {query_id!r}, document {doc_id!r} "
                                "is not a gain in [0, 1]"
                            )
                        if not _is_finite_number(theta):
                            raise DataError(
                                f"{self.source}: theta {theta!r} of query {query_id!r}, document {doc_id!r} "
                                "is not finite"
                            )
                        gains.setdefault(query_id, {})[doc_id] = float(gain)
                        thetas.setdefault(query_id, {})[doc_id] = float(theta)
            self._note_row_duplicates("qrels", fold)
            self._labels = _Labels(qrels, gains, thetas)
        return self._labels

    # -- the pools and exclusions ------------------------------------------
    def candidates(self) -> dict[ID, list[ID]] | None:
        """Each query's judged pool in pool order (``top_ranked``), or ``None`` when the source has none."""
        candidates: dict[str, list[str]] = {}
        fold = self._fold("top_ranked row")
        for row in self._rows("top_ranked", optional=True):
            query_id = required_id(row, ("query-id", "query_id"), source=self.source, what="a top_ranked row")
            docs = _list_column(self, row, ("corpus-ids", "corpus_ids"), query_id, "top_ranked")
            if fold.add(query_id, (tuple(str(doc) for doc in docs),)):
                candidates[query_id] = [str(doc) for doc in docs]
        self._note_row_duplicates("top_ranked", fold)
        return candidates or None

    def excluded(self) -> dict[ID, list[ID]]:
        """The ids removed from rankings and ideals: our ``-excluded`` config, where the repository has one."""
        excluded: dict[str, list[str]] = {}
        fold = self._fold("excluded row")
        for row in self._rows("excluded", optional=True):
            query_id = required_id(row, ("query-id", "query_id"), source=self.source, what="an excluded row")
            docs = _list_column(self, row, ("excluded-corpus-ids", "excluded_corpus_ids"), query_id, "excluded")
            if fold.add(query_id, (tuple(str(doc) for doc in docs),)):
                excluded[query_id] = [str(doc) for doc in docs]
        self._note_row_duplicates("excluded", fold)
        return excluded

    # -- the split of the labels, for the provenance -----------------------
    def _qrels_split(self) -> str:
        """The split the labels were read at (the requested one when the qrels config declares it, else the
        config's only split; the requested split when the labels come from a conventional path)."""
        config = self._config_for("qrels")
        return _split_of(config, self.split) if config is not None else self.split

    # -- provenance --------------------------------------------------------
    @property
    def provenance(self) -> Provenance:
        """The source URI, the resolved commit, the subset, the split and the duplicates policy with counts.

        The counts are the tables read so far -- the labels, pools and exclusions every load reads, and a
        corpus's or queries' folds when they have been read (a corpus is read on demand, so a load's provenance
        records the labels, pools and exclusions; the corpus and query folds are logged as they happen).
        """
        return Provenance(
            source_uri=f"hf://{self.repo}/{self.subset}",
            revision=self.commit,
            subset=self.subset,
            split=self._qrels_split(),
            duplicates=self._counts,
        )

    @property
    def task(self) -> str | None:
        """The mteb task name, when the data realises one; the Hub reader knows none."""
        return None

    @property
    def task_instruction(self) -> str | dict[Literal["query", "document"], str] | None:
        """One instruction for the whole task, lifted from a uniform per-query instruction.

        BRIGHT's per-domain instructions are task instructions the data stores per query: when every query
        of the subset carries the same one, the reader lifts it here (and the queries carry none).
        Instructions that differ per query are mteb's InstructionRetrieval data and stay per-query.
        """
        table, complete = self._instruction_table()
        if not complete:
            return None  # a mixed column is no task instruction (the query read refuses it)
        return lift_task_instruction(table, source=self.source)

    # -- helpers -----------------------------------------------------------
    def _fold(self, what: str, *, replaceable: bool = True) -> DuplicateFold:
        """A fresh fold pass for one table. ``replaceable=False`` is for a *streamed* table (the corpus and
        the queries): the earlier row is already yielded, so a conflict there refuses even under
        ``duplicates="last"``. The labels, the pools and the exclusions are materialised and take the last
        row."""
        return DuplicateFold(self.duplicates_policy, source=self.source, what=what, replaceable=replaceable)

    def _note_row_duplicates(self, what: str, fold: DuplicateFold) -> None:
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
                f"{self.source}: {what} duplicates: {counts.folded} exact folded, "
                f"{counts.resolved} resolved by the {fold.policy.value} policy"
            )

    def _media_of(self, row: Mapping[str, Any], *, corpus: bool = True) -> list[ImagePart | VideoPart]:
        """A row's media columns as content parts (``image``, ``video``; audio is deferred), persisted once.

        ``document_parts`` is a corpus setting (what a multi-column CORPUS is read as); a query always reads
        its own media -- an Any2Any query's image is the query, not a corpus-column choice.
        """
        if corpus and self.document_parts == "text":
            return []
        parts: list[ImagePart | VideoPart] = []
        for column, part_type in (("image", ImagePart), ("video", VideoPart)):
            cell = row.get(column)
            if cell is None:
                continue
            for ref in self._persist(cell, column):
                parts.append(part_type(ref=ref))
        return parts

    def _persist(self, cell: Any, column: str) -> list[MediaRef]:
        """One media cell (or list of them) written out once, content-addressed."""
        cells = cell if isinstance(cell, list) else [cell]
        refs: list[MediaRef] = []
        for value in cells:
            if value is None:
                continue
            payload, extension, mime = _encode_media(value, column=column)
            width, height = getattr(value, "width", None), getattr(value, "height", None)
            if width is None and height is None and column == "image":
                width, height = image_dimensions(payload)  # the bytes are in hand: record what they state
            refs.append(store_media(payload, extension, root=self.media_out_uri, width=width, height=height, mime=mime))
        return refs

    def _fingerprint(self, row: Mapping[str, Any]) -> tuple[Any, ...]:
        """What makes a row the row it is, for the exact-duplicate fold: the ids out, the content in."""
        media = tuple(
            json.dumps(row[column], sort_keys=True, default=repr)
            for column in MEDIA_COLUMNS
            if row.get(column) is not None
        )
        return (
            _text_or_none(row.get("title")),
            _text_or_none(row.get("text")),
            _text_or_none(row.get("instruction")),
            media,
        )


@dataclass(frozen=True)
class _Labels:
    """The parsed labels table: the grades and the gain/theta extras."""

    qrels: dict[str, dict[str, float]]
    gains: dict[str, dict[str, float]]
    thetas: dict[str, dict[str, float]]


def _list_column(reader: HubReader, row: Mapping[str, Any], names: Sequence[str], query_id: str, what: str) -> Any:
    """The first present list column among *names* of a row (a numpy array from parquet is fine); refused
    loudly when none holds a list."""
    docs = None
    for name in names:
        value = row.get(name)
        if value is not None:
            docs = value
            break
    if docs is None or isinstance(docs, str | bytes) or not hasattr(docs, "__iter__"):
        raise DataError(
            f"{reader.source}: the {what} row of query {query_id!r} carries no '{names[0]}' list; "
            f"columns: {sorted(row)}",
            details={"repo": reader.repo, "subset": reader.subset, "query_id": query_id},
        )
    return docs


def _patterns_for_split(source: str, config: _Config, split: str) -> tuple[str, ...]:
    """The config's file patterns for *split* (mteb's rule: the requested split when declared, else the
    config's only split, else an error). Entries without a split belong to every split."""
    declared = sorted({name for name, _ in config.entries if name is not None})
    if not declared:
        return tuple(pattern for _, pattern in config.entries)
    if split in declared:
        return tuple(pattern for name, pattern in config.entries if name == split or name is None)
    if len(declared) == 1:
        return tuple(pattern for _, pattern in config.entries)
    raise ConfigError(
        f"{source}: the config {config.name!r} declares the splits {declared}, not {split!r}",
        hint="name the split with --split <split> (the load_dataset split option)",
        details={"config": config.name, "splits": declared, "requested": split},
    )


def _split_of(config: _Config, split: str) -> str:
    """The split *config* is read at, without building the patterns (for the provenance)."""
    declared = sorted({name for name, _ in config.entries if name is not None})
    if split in declared:
        return split
    if len(declared) == 1:
        return declared[0]
    return split


def _text_of(row: Mapping[str, Any], column: str, *, what: str, source: str) -> str:
    """A row's text column: a string; anything else is refused where it is read (nothing is defaulted)."""
    value = row.get(column)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    raise DataError(
        f"{source}: the {what}'s {column!r} is a {type(value).__name__}, not text",
        hint="the Hub layout's text is a string (mteb's conversation form is not loaded yet: convert the "
        "repository, or read it through mteb:<Task>)",
        details={"source": source, "column": column},
    )


def _text_or_none(value: Any) -> str | None:
    """A title-ish column as a string or ``None``: a blank (an empty string, a NaN from a float-typed
    column) counts as no title, so ``str(nan)`` never joins the literal text ``"nan"`` in front of a body."""
    return value if isinstance(value, str) and value.strip() else None


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _encode_media(cell: Any, *, column: str) -> tuple[bytes, str, str | None]:
    """Bytes, extension and MIME type for a Hub media cell, preferring the undecoded original.

    A cell is a ``{"bytes", "path"}`` dict (the ``datasets`` struct), raw ``bytes`` (what a parquet media
    column may hold: mteb's Any2Any repositories store the binary directly), or a decoded object (what
    ``datasets`` hands back for an ``Image`` column: a PIL image). Using the bytes as given avoids a
    decode/re-encode round trip, which would change the hash of a byte-identical page or clip; a raw cell's
    format comes from its magic numbers. The MIME is the video table's for a container suffix and ``None``
    for an image one (the extension states it).
    """
    default_extension = ".png" if column == "image" else ".mp4"
    if isinstance(cell, dict):
        payload = cell.get("bytes")
        if payload:
            return _with_mime(payload, _extension_of(str(cell.get("path") or ""), default_extension))
        if cell.get("path"):
            path = str(cell["path"])
            return _with_mime(storage.read_bytes(path), _extension_of(path, default_extension))
        raise DataError(f"the {column} cell has neither bytes nor a path: {sorted(cell)}")
    if isinstance(cell, bytes | bytearray | memoryview):
        payload = bytes(cell)
        extension = media_extension(payload)
        if extension is None:
            raise DataError(
                f"the {column} cell is raw bytes whose format no magic number names ({len(payload)} bytes)",
                hint="the cell is neither a known image nor a known video container; convert it at the source",
            )
        return _with_mime(payload, extension)

    import io

    if hasattr(cell, "save"):
        buffer = io.BytesIO()
        cell.save(buffer, format="PNG")
        return _with_mime(buffer.getvalue(), ".png")
    # A decoded video (datasets' Video feature hands back a decoder object) has no encoder this package
    # could use without ffmpeg: refuse it by name, never crash with a bare AttributeError.
    raise DataError(
        f"the {column} cell is a decoded {type(cell).__name__}, which this reader cannot encode",
        hint="read the container bytes instead (a repository whose media column holds the raw file reads "
        "as a container); a decoded video needs a frame policy, which belongs to the judge's media "
        "preparation, not to the dataset reader",
    )


def _with_mime(payload: bytes, extension: str) -> tuple[bytes, str, str | None]:
    """The payload with the MIME type its suffix states: the video table's for a container, ``None`` for an
    image (``store_media`` derives that one from the extension)."""
    mime = VIDEO_MIME_BY_SUFFIX.get(extension) if extension not in IMAGE_MIME_BY_SUFFIX else None
    return payload, extension, mime


def _extension_of(path: str, default: str) -> str:
    """The file suffix of a media cell's path, with its dot; the default when the path names none."""
    suffix = Path(path).suffix.lower()
    return suffix if suffix else default


def _card_configs(repo: str, revision: str | None) -> dict[str, _Config]:
    """``{config_name: _Config}`` from the dataset card's YAML header; ``{}`` without a card.

    A card the cache cannot serve offline (and has not marked absent) is not a silent layout switch: one
    typed warning says the reader is falling back to the plain ``{subset}/`` path layout, which only the
    released rcp-ndcg repositories are laid out in. Without a resolved commit there is nothing the cache can
    serve for any revision, so the reader goes to the path layout at once (the table reads fail with the
    revision fix when the commit is the problem, which is the failure that matters).

    The card's ``dataset_info`` names each config's features (columns); they are recorded beside the
    ``data_files`` entries, so the reader can tell whether a table has a column without reading the table
    (the instruction column decides the task-instruction lift, and a table read for it would make every
    load eager).
    """
    if revision is None:
        logger.debug(f"hf://{repo}: no resolved commit; the card is not consulted (the path layout serves)")
        return {}
    try:
        card = _hub_file(repo, "README.md", revision)
    except _local_entry_not_found() as exc:
        if _hub_absent(repo, "README.md", revision):
            return {}
        warnings.warn(
            RcpNdcgWarning(
                "CARD_UNCACHED",
                f"Serving hf://{repo} through the plain {{subset}}/ path layout (its dataset card is not in "
                "the local cache and the Hub is unreachable); the card-declared configs, and any table they "
                "alone name, are unavailable until one online run caches the card.",
            ),
            stacklevel=2,
        )
        logger.debug(f"the card of hf://{repo} is uncached offline ({type(exc).__name__}); using the path layout")
        return {}
    if card is None:
        return {}
    import yaml

    text = card.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    header = yaml.safe_load(text.split("---", 2)[1]) or {}
    features: dict[str, frozenset[str]] = {}
    info = header.get("dataset_info") or []
    for entry in [info] if isinstance(info, dict) else info:
        if not isinstance(entry, dict) or not entry.get("config_name"):
            continue
        names = frozenset(
            str(feature.get("name"))
            for feature in entry.get("features") or []
            if isinstance(feature, dict) and feature.get("name")
        )
        features[str(entry["config_name"])] = names
    configs: dict[str, _Config] = {}
    for entry in header.get("configs") or []:
        name = entry.get("config_name")
        if not name:
            continue
        files = tuple((item.get("split"), item["path"]) for item in entry.get("data_files", []) if "path" in item)
        configs[name] = _Config(name=name, entries=files, features=features.get(name, frozenset()))
    for name, names in features.items():
        # A config the card lists only under dataset_info (no data_files): its features are still its own.
        configs.setdefault(name, _Config(name=name, entries=(), features=names))
    return configs


def hub_subsets(repo: str, revision: str | None) -> tuple[str, ...]:
    """The subsets of a Hub dataset repository: the config names' ``{s}-corpus|queries|qrels|...`` prefixes,
    else ``default``; the configs the loader ignores (``qrel_diff``, ``documents``, ...) name no subset."""
    subsets: set[str] = set()
    for name in _card_configs(repo, revision):
        for suffix in _PART_SUFFIXES:
            if name.endswith(suffix):
                subsets.add(name[: -len(suffix)])
                break
        else:
            if name in _UNPREFIXED_PARTS:
                subsets.add("default")
    return tuple(sorted(subsets))


# ---------------------------------------------------------------------------
# Reading the Hub through the local cache (the released datasets' plumbing)
# ---------------------------------------------------------------------------


def _hub_file(repo: str, path: str, revision: str | None) -> Path | None:
    """The local copy of one file of a public dataset repository (downloaded once), or ``None`` if it is absent."""
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    except ImportError as exc:
        raise ImportError("downloading the released data needs huggingface_hub: pip install 'rcp-ndcg[hf]'") from exc
    try:
        return Path(hf_hub_download(repo, path, repo_type="dataset", revision=revision))
    except LocalEntryNotFoundError:
        raise  # offline and not cached: the file may well exist
    except EntryNotFoundError:
        return None


def _local_entry_not_found() -> type[Exception]:
    """The ``LocalEntryNotFoundError`` of the installed huggingface_hub, with the curated error when it is absent."""
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError as exc:
        raise ImportError("downloading the released data needs huggingface_hub: pip install 'rcp-ndcg[hf]'") from exc
    return LocalEntryNotFoundError


def _hub_absent(repo: str, path: str, revision: str | None) -> bool:
    """Whether the local cache records the repository as having no ``path`` at ``revision``.

    An online download writes the ``.no_exist`` marker when the Hub answers 404; offline it is the only way to
    tell "absent upstream" from "not cached" (``huggingface_hub.try_to_load_from_cache``).

    The marker's sentinel is a private name (``_CACHED_NO_EXIST``); a huggingface_hub without it cannot tell the
    two apart, so the file is treated as "not cached" -- the caller then raises with the offline hint instead of
    reporting a silent absence (an optional table never reads as ``None``, a required one never as upstream-404).
    One debug line records the degradation.
    """
    try:
        from huggingface_hub import _CACHED_NO_EXIST, try_to_load_from_cache
    except ImportError:
        logger.debug(
            f"huggingface_hub has no _CACHED_NO_EXIST; treating hf://{repo}/{path} at {revision} as not cached "
            "rather than absent"
        )
        return False
    return try_to_load_from_cache(repo, path, repo_type="dataset", revision=revision) is _CACHED_NO_EXIST


def _hub_miss(exc: BaseException, repo: str, path: str, revision: str | None) -> RcpNdcgError:
    """The typed failure of a hub read the cache cannot serve: :func:`classify` picks it from the cause's chain.

    Offline that is a non-retryable :class:`MissingInputError`; its hint is the revision fix when nothing is
    resolved (the cache was filled by a commit-pinned download, so only a recorded ref resolves a branch), and
    the cache-miss one when the revision is a commit the file is simply not cached at -- it never tells a caller
    to pass a revision they already passed, and never overrides what a non-offline cause (a repository that does
    not exist, say) asked the caller to check. A Hub that cannot be reached is a retryable
    :class:`ProviderError` naming ``HF_ENDPOINT``. The details name what was looked for, whatever the cause.
    """
    typed = classify(exc)
    offline = _hub_offline() or _named_offline(exc)
    if isinstance(typed, MissingInputError) and offline:
        if revision is not None and is_commit(revision):
            typed.hint = (
                "the file is not in the local Hub cache and the Hub is unreachable (HF_HUB_OFFLINE); run once "
                "online to download it"
            )
        else:
            typed.hint = (
                "the cache has no ref to resolve and the Hub is unreachable offline (HF_HUB_OFFLINE): pass "
                "--revision <full sha> (the cache was filled by a commit-pinned download), or run once online"
            )
    elif isinstance(typed, ProviderError):
        typed.hint = "the Hugging Face Hub could not be reached; check connectivity and HF_ENDPOINT, then retry"
    typed.details.update({"repo": repo, "path": path, "revision": revision})
    return typed


def _named_offline(exc: BaseException) -> bool:
    """Whether *exc* is the library's offline refusal, or its cache miss chained from one."""
    from huggingface_hub.errors import OfflineModeIsEnabled

    if isinstance(exc, OfflineModeIsEnabled):
        return True
    if isinstance(exc, _local_entry_not_found()):
        return isinstance(exc.__cause__ or exc.__context__, OfflineModeIsEnabled)
    return False


def _hub_listing(repo: str, revision: str | None) -> list[str]:
    """Every file path of a public dataset repository at one commit.

    Online the Hub answers. Offline, or with the Hub unreachable or down, the local snapshot for the commit
    stands in -- it holds the files the download left -- with one warning that it does; with no snapshot the
    failure names the real cause (see :func:`_hub_miss`), so a run materializes its corpus from a cache an
    online run filled. An answer that is not the Hub's JSON names the endpoint instead.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled

    unreachable: BaseException | None = None
    try:
        if not _hub_offline():
            return list(HfApi().list_repo_files(repo, repo_type="dataset", revision=revision))
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"{type(exc).__name__}: {exc}",
            hint="the endpoint did not answer with the Hub's JSON: check HF_ENDPOINT (a mirror or captive portal "
            "may be in the way)",
            details={"repo": repo, "path": "(file listing)", "revision": revision},
        ) from exc
    except _hub_unreachable_errors() as exc:
        status = getattr(getattr(exc, "response", None), "status_code", 0)
        if isinstance(exc, HfHubHTTPError) and status < 500 and status != 429:
            raise  # the Hub answered: the repository, the revision or the credentials are the problem
        unreachable = exc
    listing = _snapshot_listing(repo, revision)
    if listing is not None:
        warnings.warn(
            RcpNdcgWarning(
                "SNAPSHOT_LISTING",
                f"Serving the file listing of hf://{repo} from the local snapshot at {revision} (the Hub is "
                "unreachable); it holds only the files a download left, and a partial cache reads as missing data.",
            ),
            stacklevel=2,
        )
        return listing
    if unreachable is not None:
        raise _hub_miss(unreachable, repo, "(file listing)", revision) from unreachable
    offline = OfflineModeIsEnabled(f"cannot list the files of hf://{repo} offline (HF_HUB_OFFLINE)")
    raise _hub_miss(offline, repo, "(file listing)", revision) from offline


def _hub_unreachable_errors() -> tuple[type[BaseException], ...]:
    """The exception types of a Hub that did not answer, across the huggingface_hub generations.

    huggingface-hub >= 1.x speaks httpx, the ``>=0.34`` floor speaks requests; their transport errors, the
    offline refusal and a Hub answering 5xx or 429 all mean "not answered usable".
    """
    import httpx
    from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled

    errors: list[type[BaseException]] = [httpx.TransportError, OfflineModeIsEnabled, HfHubHTTPError]
    try:
        from requests.exceptions import ConnectionError as RequestsConnectionError
        from requests.exceptions import Timeout as RequestsTimeout
    except ImportError:  # pragma: no cover - the 1.x line does not need requests
        return tuple(errors)
    return tuple(errors + [RequestsConnectionError, RequestsTimeout])


def _snapshot_listing(repo: str, revision: str | None) -> list[str] | None:
    """The file paths of the local snapshot for *revision*, or ``None`` when the cache holds no snapshot of it."""
    if revision is None or not is_commit(revision):
        return None  # the snapshot tree is per commit; without one there is nothing this cache can list
    snapshot = _hub_cache_dir() / f"datasets--{repo.replace('/', '--')}" / "snapshots" / revision
    if not snapshot.is_dir():
        return None
    return sorted(str(path.relative_to(snapshot)) for path in snapshot.rglob("*") if path.is_file())


def _iter_table_rows(local: Path, source: str) -> Iterator[Mapping[str, Any]]:
    """The rows of one local data file, whatever its format (parquet, jsonl, jsonl.gz, tsv, tsv.gz)."""
    name = local.name.lower()
    compressed = name.endswith(".gz")
    if name.endswith(".parquet"):
        import pandas as pd
        import pyarrow  # noqa: F401  (pandas' parquet engine)

        yield from pd.read_parquet(local).to_dict("records")
    elif name.endswith((".jsonl", ".json", ".jsonl.gz")):
        opener = gzip.open if compressed else open
        with opener(local, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataError(f"{source}:{line_number}: not a JSON line: {exc}") from exc
                if not isinstance(row, dict):
                    raise DataError(f"{source}:{line_number}: a row is a JSON object, got {type(row).__name__}")
                yield row
    elif name.endswith((".tsv", ".tsv.gz")):
        opener = gzip.open if compressed else open
        with opener(local, "rt", encoding="utf-8", newline="") as handle:  # type: ignore[operator]
            for row in csv.DictReader(handle, delimiter="\t"):
                yield dict(row)
    else:
        raise DataError(
            f"{source}: the Hub reader reads parquet, jsonl, jsonl.gz, tsv and tsv.gz, not {local.name}",
            hint="convert the repository to one of those formats (mteb's writers write parquet)",
        )


__all__ = [
    "DOCUMENT_PARTS",
    "DocumentParts",
    "HubReader",
    "hub_subsets",
]
