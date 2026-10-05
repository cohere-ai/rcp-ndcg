"""The :class:`Dataset` and :func:`load_dataset`: one object for a retrieval task, loaded from one URI.

URIs (one resolver; the scheme picks the reader):

* ``hf://<owner>/<repo>/<subset>[@revision]`` -- the public RCP-nDCG layout on the HuggingFace Hub:
  ``{subset}/qrels.parquet`` (``query-id, corpus-id, score, gain, theta``), ``{subset}/top_ranked.parquet`` (the
  judged pool), ``{subset}/excluded.parquet`` (ids removed per query), ``{subset}/queries.parquet``, and the corpus
  the dataset card declares for ``{subset}-corpus``. Without a subset, a public suite's repository loads all of its
  subsets.
* ``suite:<name>`` -- every subset of a public suite (:data:`SUITES`: ``nanobeir``, ``bright``, ``vidore``,
  ``trecdl``), each scored with the suite's protocol.
* ``beir:<dir>``, ``jsonl:<path>``, ``images:<dir>``, ``videos:<dir>``, ``frames:<dir>`` -- the readers of
  :mod:`rcp_ndcg.data.io`. A PDF has no queries or qrels to score against: convert it (``rcp-ndcg data convert
  --format pdf``) and add them.

Qrels are float grades everywhere. Queries and documents are read on first access, so scoring a run needs only
the qrels, pools and exclusions.
"""

from __future__ import annotations

import fnmatch
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
from rcp_ndcg_core._records import Document, Query
from rcp_ndcg_core.content import Content, ImagePart, MediaRef, TextPart

from rcp_ndcg.data.io import READERS, JsonlReader, get_reader, grade
from rcp_ndcg.data.io.base import join_title
from rcp_ndcg.data.revisions import hub_cache_dir, hub_offline, is_commit, resolve_revision
from rcp_ndcg.errors import (
    ConfigError,
    DataError,
    MissingInputError,
    ProviderError,
    RcpNdcgError,
    classify,
)
from rcp_ndcg.support.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)


class Suite(NamedTuple):
    """A public benchmark: its HuggingFace repository and subsets; its name is also its protocol preset."""

    repo: str
    subsets: tuple[str, ...]


VIDORE_NATIVE_LANGUAGE: dict[str, str] = {
    "computer_science": "english",
    "energy": "french",
    "finance_en": "english",
    "finance_fr": "french",
    "hr": "english",
    "industrial": "english",
    "pharmaceuticals": "english",
    "physics": "french",
}
"""The language each ViDoRe v3 domain was written in; the paper scores each question in it."""

SUITES: dict[str, Suite] = {
    "nanobeir": Suite(
        "fabianschmidt-cohere/rcp-ndcg-nanobeir",
        tuple(
            f"Nano{name}Retrieval"
            for name in (
                "ArguAna",
                "ClimateFever",
                "DBPedia",
                "FEVER",
                "FiQA2018",
                "HotpotQA",
                "MSMARCO",
                "NFCorpus",
                "NQ",
                "Quora",
                "SCIDOCS",
                "SciFact",
                "Touche2020",
            )
        ),
    ),
    "bright": Suite(
        "fabianschmidt-cohere/rcp-ndcg-bright",
        (
            "aops",
            "biology",
            "earth_science",
            "economics",
            "leetcode",
            "pony",
            "psychology",
            "robotics",
            "stackoverflow",
            "sustainable_living",
            "theoremqa_questions",
            "theoremqa_theorems",
        ),
    ),
    "vidore": Suite(
        "fabianschmidt-cohere/rcp-ndcg-vidore-v3",
        tuple(f"{domain}__{language}" for domain, language in VIDORE_NATIVE_LANGUAGE.items()),
    ),
    "trecdl": Suite("fabianschmidt-cohere/rcp-ndcg-trecdl", ("trec_dl_2019", "trec_dl_2020")),
}
"""The public suites; each name is also the :data:`~rcp_ndcg_core.protocol.PROTOCOLS` preset they are scored with."""


_ROW = ConfigDict(frozen=True, extra="forbid", coerce_numbers_to_str=True)
"""The record models: immutable, unknown keys refused, numeric ids read as strings."""


class QueryRow(BaseModel):
    """One row of a dataset's query table (and one record of :meth:`Dataset.from_records`).

    Attributes:
        query_id: The query id.
        text: The query text (the text view of ``content``; empty for a pure image query).
        instruction: A task instruction the query is asked under (BRIGHT), or ``None``.
        content: The query as parts when it carries media; ``None`` for text.
    """

    model_config = _ROW

    query_id: str
    text: str = ""
    instruction: str | None = None
    content: Content | None = None

    @property
    def as_content(self) -> Content:
        """The query as parts (a text part for a text query)."""
        return self.content if self.content is not None else Content.from_text(self.text)

    def format_query(self) -> str:
        """The text a text model reads: ``Task: <instruction>\nQuery: <text>``, or the text alone."""
        return self._query().format_query()

    def format_content(self) -> Content:
        """The parts an encoder reads: the query with the instruction prefixed as text."""
        return self._query().format_content()

    def _query(self) -> Query:
        return Query(query_id=self.query_id, query=self.text, instruction=self.instruction, content=self.content)


class DocumentRow(BaseModel):
    """One row of a dataset's corpus table (and one record of :meth:`Dataset.from_records`).

    Attributes:
        doc_id: The document id.
        text: The document text (title and body; the text view of ``content``).
        content: The document as parts when it carries media; ``None`` for text.
    """

    model_config = _ROW

    doc_id: str
    text: str = ""
    content: Content | None = None

    @property
    def as_content(self) -> Content:
        """The document as parts (a text part for a text document)."""
        return self.content if self.content is not None else Content.from_text(self.text)


class QrelRow(BaseModel):
    """One relevance label (a record of :meth:`Dataset.from_records`), optionally with a released gain.

    Attributes:
        query_id: The query id.
        doc_id: The document id.
        grade: The human grade, a float (0 for a pooled document without a positive label).
        gain: The calibrated gain in ``[0, 1]``, when the data carries one.
        theta: The calibrated ability behind ``gain``, in logits.
    """

    model_config = _ROW

    query_id: str
    doc_id: str
    grade: float
    gain: float | None = Field(default=None, ge=0.0, le=1.0)
    theta: float | None = None

    @field_validator("grade", "theta")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("not finite")
        return value


def _query_row(record: Query) -> QueryRow:
    return QueryRow(query_id=str(record.id), text=record.text, instruction=record.instruction, content=record.content)


def _document_row(record: Document) -> DocumentRow:
    return DocumentRow(doc_id=str(record.id), text=record.text, content=record.content)


class Dataset(BaseModel):
    """A retrieval task: qrels, released gains, candidate pools and exclusions; queries and corpus on demand.

    Attributes:
        name: The dataset (subset) name.
        uri: The URI it was loaded from.
        revision: The source revision: for Hub data, the commit it was read at (the revision as given when it
            could not be resolved, offline and not in the local cache).
        protocol: The :data:`~rcp_ndcg_core.protocol.PROTOCOLS` preset the data is scored with (set for the public
            suites).
        qrels: ``{query_id: {doc_id: grade}}``, float grades (human labels; 0 for pooled documents without one).
        gains: ``{query_id: {doc_id: gain}}``, the released calibrated gains in ``[0, 1]`` over each query's judged
            pool; ``None`` when the source has none.
        thetas: ``{query_id: {doc_id: theta}}``, the calibrated abilities behind ``gains``, in logits.
        candidates: ``{query_id: [doc_id, ...]}``, each query's judged pool in pool order (HF ``top_ranked``).
        excluded: ``{query_id: [doc_id, ...]}``, ids removed from rankings and ideals (HF ``excluded``).
        subsets: A suite's datasets, one per subset; the fields above are then empty.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    uri: str | None = None
    revision: str | None = None
    protocol: str | None = None
    qrels: dict[str, dict[str, float]] = {}
    gains: dict[str, dict[str, float]] | None = None
    thetas: dict[str, dict[str, float]] | None = None
    candidates: dict[str, list[str]] | None = None
    excluded: dict[str, list[str]] = {}
    subsets: tuple[Dataset, ...] = ()

    _load_queries: Callable[[], Iterable[Query]] | None = PrivateAttr(default=None)
    _load_corpus: Callable[[], Iterable[Document]] | None = PrivateAttr(default=None)
    _cache: dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def queries(self) -> dict[str, QueryRow]:
        """The query table, ``{query_id: QueryRow}``, read on first access."""
        return self._lazy("queries", self._load_queries, _query_row)

    @property
    def corpus(self) -> dict[str, DocumentRow]:
        """The corpus table, ``{doc_id: DocumentRow}``, read on first access."""
        return self._lazy("corpus", self._load_corpus, _document_row)

    @property
    def parts(self) -> tuple[Dataset, ...]:
        """The datasets to score: the subsets of a suite, or this dataset alone."""
        return self.subsets or (self,)

    def __repr__(self) -> str:
        if self.subsets:
            return f"Dataset({self.name!r}, protocol={self.protocol}, {len(self.subsets)} subsets)"
        judged = sum(len(docs) for docs in (self.gains or {}).values())
        pools = f", {len(self.candidates)} pools" if self.candidates is not None else ""
        gains = f", {judged} gains" if self.gains is not None else ""
        return f"Dataset({self.name!r}, protocol={self.protocol}, {len(self.qrels)} labelled queries{pools}{gains})"

    __str__ = __repr__

    @classmethod
    def from_records(
        cls,
        *,
        name: str,
        queries: Iterable[QueryRow | Mapping[str, Any]] = (),
        corpus: Iterable[DocumentRow | Mapping[str, Any]] = (),
        qrels: Iterable[QrelRow | Mapping[str, Any]] = (),
        candidates: Mapping[str, Sequence[str]] | None = None,
        excluded: Mapping[str, Sequence[str]] | None = None,
        protocol: str | None = None,
    ) -> Dataset:
        """A dataset held in memory, from plain records, validated strictly.

        Each record is a dict or the row model itself; unknown keys, missing fields and wrong values are refused.
        A pandas frame becomes records with ``frame.to_dict("records")``.

        Args:
            name: The dataset name (what evaluation reports and judgement stores call it).
            queries: :class:`QueryRow` records: ``query_id``, ``text``, optional ``instruction`` and ``content``.
            corpus: :class:`DocumentRow` records: ``doc_id``, ``text``, optional ``content``.
            qrels: :class:`QrelRow` records: ``query_id``, ``doc_id``, ``grade`` (a float), optional ``gain`` in
                ``[0, 1]`` and ``theta`` in logits (the released calibrated values).
            candidates: ``{query_id: [doc_id, ...]}``, each query's pool in pool order.
            excluded: ``{query_id: [doc_id, ...]}``, ids removed from rankings and ideals.
            protocol: A :data:`~rcp_ndcg_core.protocol.PROTOCOLS` name the data is scored with by default.

        Returns:
            The :class:`Dataset` (``uri`` is ``None``); ``queries`` and ``corpus`` are the given records.

        Raises:
            DataError: A malformed record, a duplicate id or label, a pool listing a document twice, or a label,
                pool or exclusion that names a query or document the given queries or corpus lack.
        """
        from rcp_ndcg.data._rows import validate_rows

        query_rows = _unique(validate_rows(QueryRow, queries, what="queries"), "query_id", "queries")
        document_rows = _unique(validate_rows(DocumentRow, corpus, what="corpus"), "doc_id", "corpus")
        labels: dict[str, dict[str, float]] = {}
        gains: dict[str, dict[str, float]] = {}
        thetas: dict[str, dict[str, float]] = {}
        for row in validate_rows(QrelRow, qrels, what="qrels"):
            if row.doc_id in labels.get(row.query_id, {}):
                raise DataError(
                    f"qrels: query {row.query_id!r}, document {row.doc_id!r} is labelled twice",
                    details={"query_id": row.query_id, "doc_id": row.doc_id},
                )
            labels.setdefault(row.query_id, {})[row.doc_id] = row.grade
            if row.gain is not None:
                gains.setdefault(row.query_id, {})[row.doc_id] = row.gain
            if row.theta is not None:
                thetas.setdefault(row.query_id, {})[row.doc_id] = row.theta
        pools = _id_lists(candidates, "candidates", duplicates=True)
        dropped = _id_lists(excluded, "excluded", duplicates=False) or {}
        for what, table in (("qrels", labels), ("candidates", pools or {}), ("excluded", dropped)):
            _check_ids(what, table, query_rows, document_rows)
        dataset = cls(
            name=name,
            protocol=protocol,
            qrels=labels,
            gains=gains or None,
            thetas=thetas or None,
            candidates=pools,
            excluded=dropped,
        )
        dataset._cache.update(queries=query_rows, corpus=document_rows)
        return dataset

    def _lazy(self, key: str, loader: Callable[[], Iterable[Any]] | None, row: Callable[[Any], Any]) -> dict[str, Any]:
        if self.subsets:
            raise DataError(f"{self.name!r} is a suite; read {key} from one of its subsets (Dataset.subsets)")
        if key not in self._cache:
            rows = [] if loader is None else [row(record) for record in loader()]
            self._cache[key] = {(r.query_id if key == "queries" else r.doc_id): r for r in rows}
        return self._cache[key]


def _unique[Keyed: (QueryRow, DocumentRow)](rows: list[Keyed], key: str, what: str) -> dict[str, Keyed]:
    table: dict[str, Keyed] = {}
    for row in rows:
        value = getattr(row, key)
        if value in table:
            raise DataError(f"{what}: {key} {value!r} appears twice", details={"records": what, key: value})
        table[value] = row
    return table


def _id_lists(table: Mapping[str, Sequence[str]] | None, what: str, *, duplicates: bool) -> dict[str, list[str]] | None:
    """``{query_id: [doc_id, ...]}`` with string ids; a pool (``duplicates``) must not list a document twice."""
    if table is None:
        return None
    if not isinstance(table, Mapping):
        raise DataError(f"{what} is a {type(table).__name__}; pass {{query_id: [doc_id, ...]}}")
    out: dict[str, list[str]] = {}
    for query_id, ids in table.items():
        if isinstance(ids, str) or not isinstance(ids, Sequence):
            raise DataError(f"{what}[{query_id!r}] is not a list of document ids", details={"query_id": str(query_id)})
        docs = [str(doc_id) for doc_id in ids]
        if duplicates and len(set(docs)) != len(docs):
            twice = sorted({d for d in docs if docs.count(d) > 1})
            raise DataError(f"{what}[{query_id!r}] lists {twice[:3]} twice", details={"query_id": str(query_id)})
        out[str(query_id)] = docs
    return out


def _check_ids(
    what: str,
    table: Mapping[str, Iterable[str]],
    queries: Mapping[str, QueryRow],
    corpus: Mapping[str, DocumentRow],
) -> None:
    """Refuse a query or document id that the given queries or corpus lack (each checked when it was given)."""
    if queries:
        unknown = sorted(q for q in table if q not in queries)
        if unknown:
            raise DataError(
                f"{what} name queries that are not in queries: {unknown[:5]}",
                details={"records": what, "unknown_queries": unknown[:20]},
            )
    if corpus:
        missing = sorted({d for docs in table.values() for d in docs if d not in corpus})
        if missing:
            raise DataError(
                f"{what} name documents that are not in the corpus: {missing[:5]}",
                details={"records": what, "unknown_documents": missing[:20]},
            )


def load_dataset(uri: str, *, subset: str | None = None, revision: str | None = None, **options: Any) -> Dataset:
    """Load a dataset from a URI (see the module docstring for the schemes).

    Args:
        uri: The dataset URI, e.g. ``hf://fabianschmidt-cohere/rcp-ndcg-nanobeir/NanoFiQA2018Retrieval``,
            ``suite:bright`` or ``beir:/data/nfcorpus``.
        subset: The subset of a ``hf://`` repository or a ``suite:`` (instead of naming it in the URI). A bare
            ViDoRe v3 domain (``"energy"``) selects its native-language subset.
        revision: The Hub revision (commit, tag or branch) for ``hf://`` and ``suite:``; ``None`` is ``main``. It
            is resolved to a commit once, and every file is read at that commit (:attr:`Dataset.revision`).
        options: Reader options for the reader schemes, e.g. ``qrels_uri`` and ``queries_uri`` for ``images:``.

    Returns:
        The :class:`Dataset`.

    Raises:
        ConfigError: The URI has no known scheme, or names a subset twice.
        MissingInputError: A required file does not exist.
        DataError: A file is malformed.
    """
    scheme, sep, rest = uri.partition(":")
    if not sep or not rest:
        raise ConfigError(f"dataset URI {uri!r} has no scheme", hint=f"use one of {_scheme_list()}")
    if scheme == "hf":
        if not rest.startswith("//"):
            raise ConfigError(f"a Hub dataset URI starts with 'hf://', got {uri!r}")
        return _load_hub(rest[2:], subset=subset, revision=revision)
    if scheme == "suite":
        if rest not in SUITES:
            raise ConfigError(f"unknown suite {rest!r}", hint=f"public suites: {sorted(SUITES)}")
        return _load_hub(SUITES[rest].repo, subset=subset, revision=revision)
    if scheme in READERS and scheme not in _NOT_DATASETS:
        if subset is not None or revision is not None:
            raise ConfigError(f"subset and revision apply to hf:// and suite: URIs, not {scheme}:")
        return _load_reader(scheme, rest, uri, options)
    raise ConfigError(f"unknown dataset URI scheme {scheme!r} in {uri!r}", hint=f"use one of {_scheme_list()}")


_NOT_DATASETS = frozenset({"hf", "pdf"})
"""Readers that are no dataset URI scheme: ``hf`` is the Hub layout's ``hf://``, and a PDF has no queries."""


def _scheme_list() -> str:
    return ", ".join(["hf://", "suite:"] + [f"{name}:" for name in sorted(READERS) if name not in _NOT_DATASETS])


# ---------------------------------------------------------------------------
# Reader schemes
# ---------------------------------------------------------------------------


def _load_reader(scheme: str, location: str, uri: str, options: dict[str, Any]) -> Dataset:
    reader = get_reader(scheme, uri=location, **options)
    candidates: dict[str, list[str]] | None = None
    if isinstance(reader, JsonlReader) and reader.layout == "ranking":  # each record carries its query's candidate list
        candidates = {example.id: list(example.doc_ids) for example in reader.examples()}
    dataset = Dataset(name=reader.dataset_name, uri=uri, qrels=reader.qrels(), candidates=candidates)
    dataset._load_queries = reader.queries
    dataset._load_corpus = reader.documents
    return dataset


# ---------------------------------------------------------------------------
# The public HF layout
# ---------------------------------------------------------------------------


def _load_hub(path: str, *, subset: str | None, revision: str | None) -> Dataset:
    path, _, uri_revision = path.partition("@")
    if uri_revision and revision and uri_revision != revision:
        raise ConfigError(f"two revisions for one dataset: {uri_revision!r} in the URI and {revision!r}")
    revision = revision or uri_revision or None
    parts = path.strip("/").split("/")
    if len(parts) not in (2, 3):
        raise ConfigError(f"a Hub dataset is 'hf://<owner>/<repo>[/<subset>]', got 'hf://{path}'")
    repo = "/".join(parts[:2])
    if len(parts) == 3:
        if subset is not None and subset != parts[2]:
            raise ConfigError(f"two subsets for one dataset: {parts[2]!r} in the URI and {subset!r}")
        subset = parts[2]
    suite = next((name for name, s in SUITES.items() if s.repo == repo), None)
    # Every file is read at one commit, the one the identities record, even if the branch moves meanwhile.
    revision = resolve_revision(repo, revision).commit or revision
    if subset is not None:
        return _load_hub_subset(repo, _subset_dir(suite, subset), revision=revision, protocol=suite)
    if suite is None:
        raise ConfigError(f"name the subset of {repo!r}: 'hf://{repo}/<subset>' or subset=...")
    subsets = tuple(_load_hub_subset(repo, name, revision=revision, protocol=suite) for name in SUITES[suite].subsets)
    return Dataset(name=suite, uri=f"hf://{repo}", revision=revision, protocol=suite, subsets=subsets)


def _subset_dir(suite: str | None, subset: str) -> str:
    """The subset directory; a bare ViDoRe v3 domain resolves to its native-language subset."""
    if suite == "vidore" and "__" not in subset:
        if subset not in VIDORE_NATIVE_LANGUAGE:
            raise ConfigError(f"unknown ViDoRe v3 domain {subset!r}; expected one of {sorted(VIDORE_NATIVE_LANGUAGE)}")
        return f"{subset}__{VIDORE_NATIVE_LANGUAGE[subset]}"
    return subset


def _load_hub_subset(repo: str, subset: str, *, revision: str | None, protocol: str | None) -> Dataset:
    where = f"hf://{repo}/{subset}"
    qrels_frame = _read_hub_table(repo, f"{subset}/qrels.parquet", revision)
    pools_frame = _read_hub_table(repo, f"{subset}/top_ranked.parquet", revision, optional=True)
    excluded_frame = _read_hub_table(repo, f"{subset}/excluded.parquet", revision, optional=True)
    assert qrels_frame is not None

    for column in ("query-id", "corpus-id", "score"):
        if column not in qrels_frame.columns:
            raise DataError(f"{where}: qrels.parquet has no {column!r} column; columns: {list(qrels_frame.columns)}")
    has_gains = "gain" in qrels_frame.columns and "theta" in qrels_frame.columns
    qrels: dict[str, dict[str, float]] = {}
    gains: dict[str, dict[str, float]] = {}
    thetas: dict[str, dict[str, float]] = {}
    source = f"{where}/qrels.parquet"
    columns = [qrels_frame[c] for c in ("query-id", "corpus-id", "score")]
    extra = [qrels_frame["gain"], qrels_frame["theta"]] if has_gains else [[None] * len(qrels_frame)] * 2
    for query_id, doc_id, label, gain, theta in zip(*columns, *extra, strict=True):
        qrels.setdefault(str(query_id), {})[str(doc_id)] = grade(label, source=source)
        if has_gains and theta == theta:  # theta is null (NaN) outside the judged pool
            gains.setdefault(str(query_id), {})[str(doc_id)] = float(gain)
            thetas.setdefault(str(query_id), {})[str(doc_id)] = float(theta)

    candidates = None
    if pools_frame is not None:
        candidates = {
            str(q): [str(d) for d in docs]
            for q, docs in zip(pools_frame["query-id"], pools_frame["corpus-ids"], strict=True)
        }
    excluded: dict[str, list[str]] = {}
    if excluded_frame is not None:
        for query_id, docs in zip(excluded_frame["query-id"], excluded_frame["excluded-corpus-ids"], strict=True):
            excluded[str(query_id)] = [str(d) for d in docs]

    dataset = Dataset(
        name=subset,
        uri=where,
        revision=revision,
        protocol=protocol,
        qrels=qrels,
        gains=gains if has_gains else None,
        thetas=thetas if has_gains else None,
        candidates=candidates,
        excluded=excluded,
    )
    dataset._load_queries = lambda: _hub_queries(repo, subset, revision)
    dataset._load_corpus = lambda: _hub_corpus(repo, subset, revision)
    return dataset


def _hub_queries(repo: str, subset: str, revision: str | None) -> Iterable[Query]:
    frame = _read_hub_table(repo, f"{subset}/queries.parquet", revision)
    assert frame is not None
    for row in frame.to_dict("records"):
        yield Query(query_id=str(row["id"]), query=str(row.get("text") or ""), instruction=row.get("instruction"))


def _hub_corpus(repo: str, subset: str, revision: str | None) -> Iterable[Document]:
    patterns = _card_paths(repo, revision).get(f"{subset}-corpus") or [f"{subset}/corpus.parquet"]
    files = [f for f in _hub_listing(repo, revision) if any(fnmatch.fnmatch(f, p) for p in patterns)]
    if not files:
        raise MissingInputError(f"hf://{repo}: no corpus file for {subset!r} (looked for {patterns})")
    for path in sorted(files):
        frame = _read_hub_table(repo, path, revision)
        assert frame is not None
        for row in frame.to_dict("records"):
            yield _hub_document(row)


def _hub_document(row: dict[str, Any]) -> Document:
    """A corpus row (``id``, ``title``, ``text``, optional ``image``) as a document; images go to the media cache."""
    body = join_title(row.get("title"), row.get("text"))
    image = row.get("image")
    if image is None:
        return Document(doc_id=str(row["id"]), text=body)
    parts: list[TextPart | ImagePart] = [TextPart(text=body)] if body else []
    parts.append(ImagePart(ref=_cache_image(image)))
    return Document(doc_id=str(row["id"]), content=Content.from_parts(parts))


def _cache_image(cell: Any) -> MediaRef:
    """Write an HF image cell (``{"bytes", "path"}``) into the media cache once, where the resolver reads it."""
    from rcp_ndcg.data.io.hf import _encode_hf_image
    from rcp_ndcg.data.media import store_media

    return store_media(*_encode_hf_image(cell))


def _read_hub_table(repo: str, path: str, revision: str | None, *, optional: bool = False) -> pd.DataFrame | None:
    """One table of a public repository, read at one commit; ``None`` for an optional table the repository lacks.

    An offline (or unreachable-Hub) cache miss raises the typed failure :func:`classify` picks from the download's
    cause, with the table's location in ``details``. An optional table is ``None`` when the repository has none:
    online the Hub answers 404; offline only the cache's own ``.no_exist`` record (written by an online download)
    counts as that, and a file the cache knows nothing about raises like a required table, never a silent absence.
    """

    def missing() -> MissingInputError:
        """The repository has no such table, as the Hub's 404 or the cache's mark says."""
        return MissingInputError(
            f"hf://{repo}: {path} does not exist" + (f" at {revision}" if revision else ""),
            hint="check the subset and the revision: the repository has no such table at it",
            details={"repo": repo, "path": path, "revision": revision},
        )

    try:
        local = _hub_file(repo, path, revision)
    except _local_entry_not_found() as exc:
        if _hub_absent(repo, path, revision):
            if optional:
                return None
            raise missing() from exc
        raise _hub_miss(exc, repo, path, revision) from exc
    if local is None:
        if optional:
            return None
        raise missing()
    try:
        import pandas as pd
        import pyarrow  # noqa: F401  (pandas' parquet engine)
    except ImportError as exc:
        raise ImportError("reading the released data needs pyarrow: pip install 'rcp-ndcg[data]'") from exc
    return pd.read_parquet(local)


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
    two apart, so the file is treated as "not cached" — the caller then raises with the offline hint instead of
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
    the cache-miss one when the revision is a commit the file is simply not cached at — it never tells a caller
    to pass a revision they already passed, and never overrides what a non-offline cause (a repository that does
    not exist, say) asked the caller to check. A Hub that cannot be reached is a retryable
    :class:`ProviderError` naming ``HF_ENDPOINT``. The details name what was looked for, whatever the cause.
    """
    typed = classify(exc)
    offline = hub_offline() or _named_offline(exc)
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


def _card_paths(repo: str, revision: str | None) -> dict[str, list[str]]:
    """``{config_name: [path patterns]}`` from the dataset card's YAML header."""
    import yaml

    try:
        card = _hub_file(repo, "README.md", revision)
    except _local_entry_not_found() as exc:
        if _hub_absent(repo, "README.md", revision):
            return {}
        raise _hub_miss(exc, repo, "README.md", revision) from exc
    if card is None:
        return {}
    text = card.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    header = yaml.safe_load(text.split("---", 2)[1]) or {}
    return {
        config["config_name"]: [entry["path"] for entry in config.get("data_files", []) if "path" in entry]
        for config in header.get("configs", [])
        if "config_name" in config
    }


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


def _hub_listing(repo: str, revision: str | None) -> list[str]:
    """Every file path of a public dataset repository at one commit.

    Online the Hub answers. Offline, or with the Hub unreachable or down, the local snapshot for the commit
    stands in — it holds the files the download left — with one warning that it does; with no snapshot the
    failure names the real cause (see :func:`_hub_miss`), so a run materializes its corpus from a cache an
    online run filled. An answer that is not the Hub's JSON names the endpoint instead.
    """
    import json

    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError, OfflineModeIsEnabled

    unreachable: BaseException | None = None
    try:
        if not hub_offline():
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
        logger.warning(
            f"Serving the file listing of hf://{repo} from the local snapshot at {revision} (the Hub is "
            "unreachable); it holds only the files a download left, and a partial cache reads as missing data."
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
    snapshot = hub_cache_dir() / f"datasets--{repo.replace('/', '--')}" / "snapshots" / revision
    if not snapshot.is_dir():
        return None
    return sorted(str(path.relative_to(snapshot)) for path in snapshot.rglob("*") if path.is_file())


__all__ = ["SUITES", "VIDORE_NATIVE_LANGUAGE", "Dataset", "DocumentRow", "QrelRow", "QueryRow", "Suite", "load_dataset"]
