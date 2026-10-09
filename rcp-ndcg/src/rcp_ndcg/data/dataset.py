"""The :class:`Dataset` and :func:`load_dataset`: one object for a retrieval task, loaded from one URI.

URIs (one resolver; the scheme picks the reader):

* ``hf://<owner>/<repo>[/<subset>][@revision]`` -- a HuggingFace dataset repository, read at one commit through
  the layout its dataset card declares (mteb's own rules: ``{s-}corpus`` / ``{s-}queries`` / ``{s-}qrels``,
  ``{s-}top_ranked``, an ``{s-}instruction`` config, and rcp-ndcg's ``{s-}excluded`` and ``gain``/``theta``
  qrels columns). Without a subset, a public suite's repository loads all of its subsets, and a repository with
  exactly one subset loads it. See :mod:`rcp_ndcg.data.io.hub`.
* ``suite:<name>`` -- every subset of a public suite (:data:`SUITES`: ``nanobeir``, ``bright``, ``vidore``,
  ``trecdl``), each scored with the suite's protocol.
* ``beir:<dir>``, ``jsonl:<path>``, ``mteb:<Task>[/<subset>][@split]``, ``images:<dir>``, ``videos:<dir>``,
  ``frames:<dir>`` -- the readers of :mod:`rcp_ndcg.data.io` (one entry point per format; a third-party format
  is one class in its own package). A PDF has no queries or qrels to score against: convert it
  (``rcp-ndcg data convert --format pdf``) and add them.

Qrels are float grades everywhere. Queries and documents are read on first access, so scoring a run needs only
the qrels, pools and exclusions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, field_validator
from rcp_ndcg_core._records import Document, Query
from rcp_ndcg_core.content import Content

from rcp_ndcg.data.io import READERS, Provenance, get_reader
from rcp_ndcg.data.io.base import SourceReader
from rcp_ndcg.data.io.hub import HubReader, hub_subsets
from rcp_ndcg.data.revisions import resolve_revision
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.support.logging import get_logger

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
        text: The query text as given; the parts of ``content`` are authoritative when it is set, and
            :attr:`as_content` reads them (a text query's ``text`` is its one part's text).
        instruction: A per-query instruction (mteb's InstructionRetrieval data), or ``None``. A field of its
            own, never merged into ``text`` at load: how a model's input combines them is a formatting
            decision made where the text is formatted.
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
        title: The document title, when the source has one; ``None`` otherwise. A field of its own
            (mteb keeps it as one too): nothing joins a title with the body at read time -- how a model's
            input combines them is a formatting decision made where the text is formatted.
        text: The document body as given; the parts of ``content`` are authoritative when it is set, and
            :attr:`as_content` reads them.
        content: The document as parts when it carries media; ``None`` for text.
    """

    model_config = _ROW

    doc_id: str
    title: str | None = None
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
    return DocumentRow(doc_id=str(record.id), title=record.title, text=record.text, content=record.content)


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
        subset: The source subset this dataset was read for (mteb's ``hf_subset``; ``"default"`` when the
            source has no subsets).
        split: The source split the labels were read at (mteb's ``eval split``; ``"test"`` by convention).
        task: The mteb task this dataset realises, when it was loaded through one (``mteb:<Task>``); exports
            are keyed by :attr:`export_key`, ``(task, subset, split)``.
        task_instruction: One instruction for the whole task (mteb's ``TaskMetadata.prompt``): what the model
            is asked to do, as a string or per side ``{"query": ..., "document": ...}``. Model-owned: a
            recipe places it (the generic default prefixes it); never merged into a text at load.
        provenance: Where the data came from and how it was read (the reader's
            :attr:`~rcp_ndcg.data.io.base.SourceReader.provenance`): source URI, resolved commit, subset,
            split and the duplicates policy with its counts; ``None`` for in-memory data.
        subsets: A suite's datasets, one per subset; the fields above are then empty.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    uri: str | None = None
    revision: str | None = None
    protocol: str | None = None
    subset: str = "default"
    split: str = "test"
    task: str | None = None
    task_instruction: str | dict[Literal["query", "document"], str] | None = None
    provenance: Provenance | None = None
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

    @property
    def export_key(self) -> tuple[str, str, str]:
        """The ``(task, subset, split)`` key exports are keyed by (decision 29); the task falls back to the
        dataset's name when the source named no mteb task."""
        return (self.task or self.name, self.subset, self.split)

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
        subset: str = "default",
        split: str = "test",
        task: str | None = None,
        task_instruction: str | dict[Literal["query", "document"], str] | None = None,
    ) -> Dataset:
        """A dataset held in memory, from plain records, validated strictly.

        Each record is a dict or the row model itself; unknown keys, missing fields and wrong values are refused.
        A pandas frame becomes records with ``frame.to_dict("records")``.

        Args:
            name: The dataset name (what evaluation reports and judgement stores call it).
            queries: :class:`QueryRow` records: ``query_id``, ``text``, optional ``instruction`` and ``content``.
            corpus: :class:`DocumentRow` records: ``doc_id``, optional ``title``, ``text``, optional ``content``.
            qrels: :class:`QrelRow` records: ``query_id``, ``doc_id``, ``grade`` (a float), optional ``gain`` in
                ``[0, 1]`` and ``theta`` in logits (the released calibrated values).
            candidates: ``{query_id: [doc_id, ...]}``, each query's pool in pool order.
            excluded: ``{query_id: [doc_id, ...]}``, ids removed from rankings and ideals.
            protocol: A :data:`~rcp_ndcg_core.protocol.PROTOCOLS` name the data is scored with by default.
            subset, split, task, task_instruction: The provenance fields of :class:`Dataset`.

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
        try:
            dataset = cls(
                name=name,
                protocol=protocol,
                subset=subset,
                split=split,
                task=task,
                task_instruction=task_instruction,
                qrels=labels,
                gains=gains or None,
                thetas=thetas or None,
                candidates=pools,
                excluded=dropped,
            )
        except ValidationError as exc:
            # The most specific problem (a union reports one error per branch): the deepest location.
            problem = max(exc.errors(include_url=False), key=lambda error: len(error["loc"]))
            field = str(problem["loc"][0]) if problem["loc"] else "<dataset>"
            raise DataError(
                f"dataset: {field}: {problem['msg'].removeprefix('Value error, ')}",
                details={"field": field, "input": _jsonable_input(problem.get("input"))},
            ) from None
        dataset._cache.update(queries=query_rows, corpus=document_rows)
        return dataset

    def _lazy(self, key: str, loader: Callable[[], Iterable[Any]] | None, row: Callable[[Any], Any]) -> dict[str, Any]:
        if self.subsets:
            raise DataError(f"{self.name!r} is a suite; read {key} from one of its subsets (Dataset.subsets)")
        if key not in self._cache:
            rows = [] if loader is None else [row(record) for record in loader()]
            self._cache[key] = {}
            for record in rows:
                record_id = record.query_id if key == "queries" else record.doc_id
                if record_id in self._cache[key]:
                    raise DataError(
                        f"{key}: {key[:-1] if key.endswith('s') else key}_id {record_id!r} appears twice",
                        details={"records": key, f"{key[:-1] if key.endswith('s') else key}_id": record_id},
                    )
                self._cache[key][record_id] = record
        return self._cache[key]


def _jsonable_input(value: Any) -> Any:
    """A failing input as something the error's ``details`` can carry (the row validator's own rule)."""
    from rcp_ndcg.data._rows import _jsonable

    return _jsonable(value)


def _unique[Keyed: QueryRow | DocumentRow](rows: list[Keyed], key: str, what: str) -> dict[str, Keyed]:
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


def load_dataset(
    uri: str,
    *,
    subset: str | None = None,
    revision: str | None = None,
    split: str | None = None,
    **options: Any,
) -> Dataset:
    """Load a dataset from a URI (see the module docstring for the schemes).

    Args:
        uri: The dataset URI, e.g. ``hf://fabianschmidt-cohere/rcp-ndcg-nanobeir/NanoFiQA2018Retrieval``,
            ``suite:bright`` or ``beir:/data/nfcorpus``.
        subset: The subset of a ``hf://`` repository or a ``suite:`` (instead of naming it in the URI). A bare
            ViDoRe v3 domain (``"energy"``) selects its native-language subset.
        revision: The Hub revision (commit, tag or branch) for ``hf://`` and ``suite:``; ``None`` is ``main``. It
            is resolved to a commit once, and every file is read at that commit (:attr:`Dataset.revision`).
        split: The split to read (a ``beir:`` qrels split, an ``hf://`` or ``mteb:`` split); ``None`` is
            ``"test"``, or whatever the source's own convention resolves to (a BEIR directory's first qrels
            file; a Hub config's only split).
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
        return _load_hub(rest[2:], subset=subset, revision=revision, split=split, options=options)
    if scheme == "suite":
        if rest not in SUITES:
            raise ConfigError(f"unknown suite {rest!r}", hint=f"public suites: {sorted(SUITES)}")
        return _load_hub(SUITES[rest].repo, subset=subset, revision=revision, split=split, options=options)
    if scheme in READERS and scheme not in _NOT_DATASETS:
        if subset is not None or revision is not None:
            raise ConfigError(f"subset and revision apply to hf:// and suite: URIs, not {scheme}:")
        if split is not None:
            if "split" in options:
                raise ConfigError(f"two splits for one dataset: {options['split']!r} and {split!r}")
            options = {**options, "split": split}
        return _from_reader(get_reader(scheme, uri=rest, **options), uri=uri)
    raise ConfigError(f"unknown dataset URI scheme {scheme!r} in {uri!r}", hint=f"use one of {_scheme_list()}")


_NOT_DATASETS = frozenset({"hf", "pdf"})
"""Readers that are no dataset URI scheme: ``hf`` is the Hub layout's ``hf://``, and a PDF has no queries."""


def _scheme_list() -> str:
    return ", ".join(["hf://", "suite:"] + [f"{name}:" for name in sorted(READERS) if name not in _NOT_DATASETS])


# ---------------------------------------------------------------------------
# Reader schemes
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# One dataset path
# ---------------------------------------------------------------------------


def _from_reader(reader: SourceReader, *, uri: str, protocol: str | None = None) -> Dataset:
    """One :class:`Dataset`, from one reader: the one path every source takes.

    The labels, pools, exclusions and gains come from the reader's own methods (a ranking source's pool is its
    candidate lists, a Hub repository's ``top_ranked`` its pools); the provenance fields (subset, split, task,
    the duplicates policy with its counts) come from the reader's provenance; queries and corpus are read on
    demand, through the reader.
    """
    qrels = reader.qrels()
    gains = reader.gains()
    thetas = reader.thetas()
    candidates = reader.candidates()
    excluded = reader.excluded()
    provenance = reader.provenance  # after the tables: the duplicates counts are in
    dataset = Dataset(
        name=reader.dataset_name,
        uri=uri,
        revision=provenance.revision,
        protocol=protocol,
        qrels=qrels,
        gains=gains,
        thetas=thetas,
        candidates=candidates,
        excluded=excluded,
        subset=provenance.subset,
        split=provenance.split,
        task=reader.task,
        task_instruction=reader.task_instruction,
        provenance=provenance,
    )
    dataset._load_queries = reader.queries
    dataset._load_corpus = reader.documents
    return dataset


# ---------------------------------------------------------------------------
# The public HF layout, through the Hub reader of rc_ndcg.data.io.hub
# ---------------------------------------------------------------------------


def _load_hub(
    path: str,
    *,
    subset: str | None,
    revision: str | None,
    split: str | None,
    options: dict[str, Any],
) -> Dataset:
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
    options = {**options}
    if split is not None:
        options["split"] = split
    if subset is not None:
        reader = get_reader("hf", uri=f"{repo}/{_subset_dir(suite, subset)}", revision=revision, **options)
        assert isinstance(reader, HubReader)
        return _from_reader(reader, uri=f"hf://{repo}/{reader.subset}", protocol=suite)
    if suite is None:
        commit = resolve_revision(repo, revision).commit or revision
        derived = hub_subsets(repo, commit)
        if len(derived) == 1:
            reader = get_reader(
                "hf", uri=f"{repo}/{derived[0]}", revision=revision, name=repo.rsplit("/", 1)[-1], **options
            )
            assert isinstance(reader, HubReader)
            return _from_reader(reader, uri=f"hf://{repo}")
        raise ConfigError(
            f"name the subset of {repo!r}: 'hf://{repo}/<subset>' or subset=...",
            hint=(
                f"the repository's subsets: {list(derived)}"
                if derived
                else "the repository declares no subset configs; a raw repository is loaded by its task's "
                "custom loader (mteb:<Task>, the [mteb] extra)"
            ),
        )
    # Every file is read at one commit, the one the identities record, even if the branch moves meanwhile.
    commit = resolve_revision(repo, revision).commit or revision
    subsets = tuple(
        _from_reader(
            get_reader("hf", uri=f"{repo}/{name}", revision=revision, **options),
            uri=f"hf://{repo}/{name}",
            protocol=suite,
        )
        for name in SUITES[suite].subsets
    )
    return Dataset(name=suite, uri=f"hf://{repo}", revision=commit, protocol=suite, subsets=subsets)


def _subset_dir(suite: str | None, subset: str) -> str:
    """The subset directory; a bare ViDoRe v3 domain resolves to its native-language subset."""
    if suite == "vidore" and "__" not in subset:
        if subset not in VIDORE_NATIVE_LANGUAGE:
            raise ConfigError(f"unknown ViDoRe v3 domain {subset!r}; expected one of {sorted(VIDORE_NATIVE_LANGUAGE)}")
        return f"{subset}__{VIDORE_NATIVE_LANGUAGE[subset]}"
    return subset


__all__ = ["SUITES", "VIDORE_NATIVE_LANGUAGE", "Dataset", "DocumentRow", "QrelRow", "QueryRow", "Suite", "load_dataset"]
