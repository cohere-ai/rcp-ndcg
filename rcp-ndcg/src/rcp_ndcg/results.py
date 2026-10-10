"""The results-export seam: one versioned record per evaluated row, and the sink registry.

A run, a report or a system's rankings is exported as ``rcp-ndcg.result-record.v1`` records: one record per
``(system, dataset, metric, cutoff)`` row, carrying the run identity, the dataset revision, the scoring
protocol (qrel gain, tie rule, the pool the ideal comes from) and the artifacts the row was computed from.
The record is a public contract: a field's meaning or type changes only with a new schema id, and fields are
added to ``v1`` additively.

A **sink** consumes records; it never reads the run. ``jsonl`` writes one record per line, ``parquet`` one row
per metric row, ``null`` discards (a dry run, and the conformance suite). Third-party sinks are one class in
their own package, published under the ``rcp_ndcg.results`` entry-point group::

    [project.entry-points."rcp_ndcg.results"]
    mine = "my_package.results:MySink"

``MySink`` implements :class:`ResultsSink` (its first constructor parameter is named ``uri``) and passes
:func:`rcp_ndcg.testing.results_conformance`. The record itself carries no consumer's concepts: no view, no
pivot, no path layout. ``rcp-ndcg results export`` builds records from a run directory or a report file and
hands them to the named sink; :func:`records_from_report` and :func:`records_from_run` are the same builders
in Python.

The protocol travels in full: ``dataset.protocol`` is the preset name and ``dataset.protocol_spec`` the
:class:`~rcp_ndcg_core.protocol.Protocol` (qrel gain, tie rule, pool restriction, rounding). The ideal DCG
follows from the metric and that protocol: qrel-nDCG sorts the query's positive qrels, RCP-nDCG and Count-nDCG
sort all of its gains, and ``restrict_to_candidates`` decides which scored documents enter the ranking.
Two records that differ only in the protocol are different records -- :func:`record_identity` digests the
subject, the dataset and the metrics, so they never compare equal.
"""

from __future__ import annotations

import abc
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import ExitStack
from datetime import UTC, datetime
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from rcp_ndcg_core.protocol import Protocol

from rcp_ndcg import storage
from rcp_ndcg.errors import ConfigError, DataError, MissingInputError
from rcp_ndcg.runs.pipeline import CANDIDATES, JUDGE, REFERENCE_SYSTEMS
from rcp_ndcg.support.entrypoints import load_entry_point, provider_of, registered_names
from rcp_ndcg.support.identity import hash_payload
from rcp_ndcg.support.logging import get_logger

if TYPE_CHECKING:
    from rcp_ndcg.eval import EvalReport

logger = get_logger(__name__)

RESULT_SCHEMA = "rcp-ndcg.result-record.v1"
"""The schema id every record carries in its ``schema`` field."""

RESULTS_GROUP = "rcp_ndcg.results"
"""The entry-point group of the result sinks."""

_ArtifactRole = Literal["rankings", "report", "manifest", "calibration", "judgements", "log"]


class ResultSubject(BaseModel):
    """What a row is about: a system of a run, a run, or a calibration.

    Attributes:
        kind: ``"system"`` (one ranker of a run), ``"run"`` (the run as a whole) or ``"calibration"``.
        name: The subject's name: the system name, the run id, or the calibration's name.
        system: The ``Rankings`` system value, when ``kind == "system"``.
        run_id: The run the row belongs to, when it belongs to one.
        identity: The rcp-ndcg digest of the subject's content (a rankings file's SHA-256, the run's resolved
            config, a family key digest), or ``None`` when the export does not know it.
        labels: Free-form provenance labels the exporter owns (e.g. ``{"model": "org/model"}``); never a host
            or a bucket.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["system", "run", "calibration"]
    name: str
    system: str | None = None
    run_id: str | None = None
    identity: str | None = None
    labels: dict[str, str] = {}


class ResultDataset(BaseModel):
    """The dataset a row was scored on, and the convention it was scored under.

    Attributes:
        name: The dataset (subset) name.
        subset: The source subset (mteb's ``hf_subset``); ``"default"`` when the source has none.
        split: The source split the labels were read at; ``"test"`` by convention.
        task: The mteb task the dataset realises, when it was loaded through one.
        revision: The source revision: the Hub commit the data was read at, or ``None``.
        protocol: The protocol preset name (``"mteb"``, ``"nanobeir"``, ``"plain"``, ...): the comparability
            label a reader groups rows by.
        protocol_spec: The full scoring convention -- qrel gain (linear/exponential), tie rule, pool
            restriction, rounding. The ideal DCG follows from the metric and this protocol (qrel-nDCG sorts
            the query's positive qrels; RCP-nDCG and Count-nDCG sort all of its gains). Two rows with
            different protocols are never comparable.
        gains_source: Where the RCP gains came from: ``"gains"`` (passed in), ``"calibration"``,
            ``"dataset"`` (the released gains) or ``"none"``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    subset: str = "default"
    split: str = "test"
    task: str | None = None
    revision: str | None = None
    protocol: str | None = None
    protocol_spec: Protocol | None = None
    gains_source: Literal["gains", "calibration", "dataset", "none"] | None = None


class ResultMetric(BaseModel):
    """One metric value of one row: the headline number and its interval.

    Attributes:
        metric: ``"rcp_ndcg"``, ``"qrel_ndcg"``, ``"count_ndcg"`` or a metric an exporter owns.
        k: The cutoff.
        value: The value, or ``None`` when the metric is undefined for the row.
        num_queries: The queries with a defined value.
        num_datasets: The datasets with one (a summary row); 0 for a per-dataset row.
        ci_low: The bootstrap interval's lower bound, or ``None``.
        ci_high: The bootstrap interval's upper bound, or ``None``.
        dataset: The dataset this row is the mean of, or ``None`` for the summary row.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: str
    k: int = Field(gt=0)
    value: float | None = Field(allow_inf_nan=False)
    num_queries: int = Field(default=0, ge=0)
    num_datasets: int = Field(default=0, ge=0)
    ci_low: float | None = Field(default=None, allow_inf_nan=False)
    ci_high: float | None = Field(default=None, allow_inf_nan=False)
    dataset: str | None = None


class ResultArtifact(BaseModel):
    """One artifact a row was computed from.

    Attributes:
        role: What the artifact is: rankings, report, manifest, calibration, judgements or log.
        uri: The artifact's location, relative to the run or a URI the exporter owns.
        sha256: The file's SHA-256, or ``None`` (a directory, or a sink that records no hash).
        schema_name: The artifact's schema id (e.g. ``"rcp-ndcg.eval-report.v1"``), or ``None``.
        media_type: The media type, or ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: _ArtifactRole
    uri: str
    sha256: str | None = None
    schema_name: str | None = None
    media_type: str | None = None


class ResultRecord(BaseModel):
    """One exported row: a system's value for a dataset, metric and cutoff (``rcp-ndcg.result-record.v1``).

    Attributes:
        schema_name: ``"rcp-ndcg.result-record.v1"`` (serialised as ``schema``).
        record_id: :func:`record_identity` of the subject, the dataset and the metrics: two records with the
            same id say the same thing, and a protocol difference is part of the id.
        created_at: When the record was built (timezone-aware).
        subject: What the row is about.
        dataset: The dataset and the scoring convention, or ``None`` when the row names no dataset.
        metrics: The metric rows (one record usually holds exactly one).
        artifacts: The files the row was computed from.
        provenance: Free-form provenance the exporter owns (code version, config digest, environment).
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True, serialize_by_alias=True, extra="forbid")

    schema_name: Literal["rcp-ndcg.result-record.v1"] = Field(default=RESULT_SCHEMA, alias="schema")
    record_id: str
    created_at: datetime
    subject: ResultSubject
    dataset: ResultDataset | None = None
    metrics: list[ResultMetric] = []
    artifacts: list[ResultArtifact] = []
    provenance: dict[str, Any] = {}

    @field_validator("created_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware (a naive datetime names no instant)")
        return value


def record_identity(subject: ResultSubject, dataset: ResultDataset | None, metrics: Sequence[ResultMetric]) -> str:
    """The deterministic id of a record: a digest of the subject, the dataset and the metrics.

    The dataset's protocol (its preset name and its full :class:`~rcp_ndcg_core.protocol.Protocol`) is part of
    the payload, so two rows scored under different conventions never share an id even when every number
    matches.
    """
    return hash_payload(
        {
            "subject": subject.model_dump(),
            "dataset": dataset.model_dump() if dataset is not None else None,
            "metrics": [metric.model_dump() for metric in metrics],
        }
    )


# ---------------------------------------------------------------------------
# The sink contract and the built-in sinks
# ---------------------------------------------------------------------------


class ResultsSink(abc.ABC):
    """Consumes :class:`ResultRecord`s; the public half of the results-export seam.

    A sink is one class in a package that publishes it under the ``rcp_ndcg.results`` entry-point group. Its
    constructor's first parameter after ``self`` is named ``uri`` (what ``--out`` passes), and its ``emit``
    refuses a record whose schema it does not know. :meth:`flush` is called once after the last record; a sink
    that buffers writes there.
    """

    name: ClassVar[str] = ""

    def __init__(self, uri: str | None = None, options: Mapping[str, str] | None = None) -> None:
        self.uri = uri
        self.options = dict(options or {})

    @abc.abstractmethod
    def emit(self, record: ResultRecord) -> None:
        """Write one record."""

    def flush(self) -> None:
        """Flush what the sink buffers (the default: nothing)."""
        return None

    def _known(self, record: object) -> ResultRecord:
        """``record`` as a :class:`ResultRecord` of this schema.

        Raises:
            ConfigError: ``record`` is not a record, or its schema is not :data:`RESULT_SCHEMA` (a v2 record
                needs a sink that knows v2; silently writing it as v1 would corrupt the file).
        """
        if not isinstance(record, ResultRecord):
            raise ConfigError(
                f"the {type(self).name or type(self).__name__} sink takes ResultRecord records, "
                f"got {type(record).__name__}",
                hint=f"emit a record built by rcp_ndcg.results (the {RESULT_SCHEMA} schema)",
            )
        if record.schema_name != RESULT_SCHEMA:
            raise ConfigError(
                f"the {type(self).name or type(self).__name__} sink does not know schema {record.schema_name!r}; "
                f"it writes {RESULT_SCHEMA!r}",
                hint=f"install the sink of {record.schema_name!r}, or export with a sink of this version",
            )
        return record


class JsonlResultSink(ResultsSink):
    """Write one record per line as JSON (``--out records.jsonl``).

    The file is truncated when the first record is emitted and closed by :meth:`flush`, so a re-export replaces
    the file rather than doubling it. ``options={"append": "1"}`` appends to an existing file instead (for an
    export that continues an earlier one).
    """

    name = "jsonl"

    def __init__(self, uri: str | None = None, options: Mapping[str, str] | None = None) -> None:
        super().__init__(uri, options)
        if uri is None:
            raise ConfigError(
                "the jsonl sink needs a uri",
                hint="pass uri=<file.jsonl> (a local path or a storage URI)",
                cli_hint="pass --out <file.jsonl>",
            )
        self._append = str(self.options.get("append", "")).strip().lower() in {"1", "true", "yes"}
        self._stack: ExitStack | None = None
        self._handle: Any = None

    def emit(self, record: ResultRecord) -> None:
        self._known(record)
        if self._handle is None:
            self._stack = ExitStack()
            mode = "a" if self._append else "w"
            self._handle = self._stack.enter_context(storage.open_path(self.uri or "", mode))
        self._handle.write(record.model_dump_json(by_alias=True) + "\n")

    def flush(self) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None
            self._handle = None


class ParquetResultSink(ResultsSink):
    """Write one row per metric row as Parquet (``--out records.parquet``).

    A record with several metrics becomes several rows sharing its record, subject, dataset and artifact
    columns; a record with no metric rows becomes one row with null metric columns (a record is never dropped).
    The table is buffered and written by :meth:`flush`.
    """

    name = "parquet"

    def __init__(self, uri: str | None = None, options: Mapping[str, str] | None = None) -> None:
        super().__init__(uri, options)
        if uri is None:
            raise ConfigError(
                "the parquet sink needs a uri",
                hint="pass uri=<file.parquet> (a local path or a storage URI)",
                cli_hint="pass --out <file.parquet>",
            )
        self._records: list[ResultRecord] = []

    def emit(self, record: ResultRecord) -> None:
        self._records.append(self._known(record))

    def flush(self) -> None:
        if not self._records:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(_parquet_rows(self._records), schema=_parquet_schema())
        with storage.open_path(self.uri or "", "wb") as handle:
            pq.write_table(table, handle)
        self._records.clear()


class NullResultSink(ResultsSink):
    """Accept records and discard them (a dry run, and the conformance suite)."""

    name = "null"

    def __init__(self, uri: str | None = None, options: Mapping[str, str] | None = None) -> None:
        super().__init__(uri, options)
        self.emitted = 0

    def emit(self, record: ResultRecord) -> None:
        self._known(record)
        self.emitted += 1


_PARQUET_COLUMNS: tuple[tuple[str, str], ...] = (
    ("record_id", "string"),
    ("created_at", "string"),
    ("subject_kind", "string"),
    ("subject_name", "string"),
    ("subject_system", "string"),
    ("subject_run_id", "string"),
    ("subject_identity", "string"),
    ("subject_labels", "string"),
    ("dataset_name", "string"),
    ("dataset_subset", "string"),
    ("dataset_split", "string"),
    ("dataset_task", "string"),
    ("dataset_revision", "string"),
    ("dataset_protocol", "string"),
    ("dataset_protocol_spec", "string"),
    ("dataset_gains_source", "string"),
    ("metric", "string"),
    ("k", "int64"),
    ("value", "float64"),
    ("num_queries", "int64"),
    ("num_datasets", "int64"),
    ("ci_low", "float64"),
    ("ci_high", "float64"),
    ("metric_dataset", "string"),
    ("artifacts", "string"),
    ("provenance", "string"),
)
"""The parquet table's columns, in order: flat record, subject and dataset fields plus one metric row."""


def _parquet_schema():
    import pyarrow as pa

    return pa.schema([(name, getattr(pa, kind)()) for name, kind in _PARQUET_COLUMNS])


def _parquet_rows(records: Iterable[ResultRecord]) -> list[dict[str, Any]]:
    """One row per metric row, the record's other columns repeated (a metric-less record keeps one null row).

    The nested fields are read from ``model_dump(mode="json")``, so a provenance value pydantic can serialise
    (a ``datetime``, an enum) is serialised here the same way the JSONL sink serialises it.
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        subject, dataset = record.subject, record.dataset
        payload = record.model_dump(mode="json")
        common: dict[str, Any] = {
            "record_id": record.record_id,
            "created_at": record.created_at.isoformat(),
            "subject_kind": subject.kind,
            "subject_name": subject.name,
            "subject_system": subject.system,
            "subject_run_id": subject.run_id,
            "subject_identity": subject.identity,
            "subject_labels": json.dumps(payload["subject"]["labels"], sort_keys=True),
            "dataset_name": dataset.name if dataset is not None else None,
            "dataset_subset": dataset.subset if dataset is not None else None,
            "dataset_split": dataset.split if dataset is not None else None,
            "dataset_task": dataset.task if dataset is not None else None,
            "dataset_revision": dataset.revision if dataset is not None else None,
            "dataset_protocol": dataset.protocol if dataset is not None else None,
            "dataset_protocol_spec": (
                json.dumps(payload["dataset"]["protocol_spec"], sort_keys=True)
                if dataset is not None and dataset.protocol_spec is not None
                else None
            ),
            "dataset_gains_source": dataset.gains_source if dataset is not None else None,
            "artifacts": json.dumps(payload["artifacts"], sort_keys=True),
            "provenance": json.dumps(payload["provenance"], sort_keys=True),
        }
        for metric in record.metrics or [None]:
            rows.append(
                {
                    **common,
                    "metric": metric.metric if metric is not None else None,
                    "k": metric.k if metric is not None else None,
                    "value": metric.value if metric is not None else None,
                    "num_queries": metric.num_queries if metric is not None else None,
                    "num_datasets": metric.num_datasets if metric is not None else None,
                    "ci_low": metric.ci_low if metric is not None else None,
                    "ci_high": metric.ci_high if metric is not None else None,
                    "metric_dataset": metric.dataset if metric is not None else None,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def result_sink_class(name: str, /) -> type[ResultsSink]:
    """The sink class registered under ``name`` in the ``rcp_ndcg.results`` entry-point group.

    Raises:
        ConfigError: No sink has that name (the installed ones are named), or the entry point does not load
            to a :class:`ResultsSink`.
    """
    entries = tuple(entry_points(group=RESULTS_GROUP))
    available = registered_names(entries, RESULTS_GROUP)
    if name not in available:
        raise ConfigError(
            f"unknown result sink {name!r}. Available: {list(available)}.",
            hint=f"a sink is a {RESULTS_GROUP!r} entry point; install the package that provides it",
        )
    return load_entry_point(provider_of(entries, RESULTS_GROUP, name, kind="result sink"), ResultsSink, kind="sink")


def registered_result_sinks() -> tuple[str, ...]:
    """The registered sink names, sorted (built-ins and plugins)."""
    return registered_names(entry_points(group=RESULTS_GROUP), RESULTS_GROUP)


# ---------------------------------------------------------------------------
# Building records from rcp-ndcg's own outputs
# ---------------------------------------------------------------------------


def records_from_report(
    report: EvalReport,
    *,
    subject: ResultSubject | Callable[[str], ResultSubject] | None = None,
    dataset: ResultDataset | None = None,
    systems: Sequence[str] | None = None,
    artifacts: Sequence[ResultArtifact] = (),
    provenance: Mapping[str, Any] | None = None,
    created_at: datetime | None = None,
    per_dataset: bool = True,
    summary: bool = True,
) -> list[ResultRecord]:
    """One record per ``(system, dataset, metric, cutoff)`` row of ``report``.

    Args:
        report: The scored report (:func:`rcp_ndcg.eval.evaluate`), from Python or read from
            ``metrics/report.json``.
        subject: What the records are about. ``None`` (the default) derives a
            ``ResultSubject(kind="system", ...)`` per row; a callable is called with the row's system; a
            ``ResultSubject`` is used for every record and is refused for a report of several systems (one
            subject cannot describe several).
        dataset: The scored dataset. Given, every record carries it, with ``protocol``, ``protocol_spec`` and
            ``gains_source`` taken from the report (the report's convention is what was scored). Omitted, a
            per-dataset row derives ``ResultDataset(name=<row dataset>)`` and the summary rows need the
            report's per-dataset rows to name exactly one dataset, else a :class:`ConfigError` (pass
            ``dataset=``).
        systems: Score only these systems (the report's order; default every system). An unknown name is
            refused, listing the systems the report has.
        artifacts: The files the records were computed from.
        provenance: Free-form provenance (code version, config digest, environment).
        created_at: The timestamp every record carries (default: now, once for the whole export).
        per_dataset: Emit the per-dataset mean rows.
        summary: Emit the summary rows (the aggregate with its interval).

    Returns:
        The records, in the report's row order (per system: the per-dataset rows, then the summary rows).

    Raises:
        ConfigError: A subject for several systems, an unknown system, a summary over several datasets without
            ``dataset=``, or a value that is not an :class:`~rcp_ndcg.eval.EvalReport`.
    """
    from rcp_ndcg.eval import EvalReport

    if not isinstance(report, EvalReport):
        raise ConfigError(f"records_from_report needs an EvalReport, got {type(report).__name__}")
    moment = created_at or datetime.now(UTC)
    available = report.systems
    if systems is None:
        selected = list(available)
    else:
        unknown = sorted(set(systems) - set(available))
        if unknown:
            raise ConfigError(
                f"systems {unknown} are not in the report; systems: {available}",
                hint="name one of the report's systems, or drop systems=",
                cli_hint="name one of the report's systems with --system, or drop --system",
                details={"unknown": unknown, "systems": available},
            )
        if not systems:
            raise ConfigError("systems names no system; pass the systems to export, or None for all")
        selected = [system for system in available if system in set(systems)]
    if callable(subject):
        subject_for: Callable[[str], ResultSubject] = subject
    elif subject is not None:
        if len(selected) > 1:
            raise ConfigError(
                f"subject names one subject for a report of {len(selected)} systems: {selected}",
                hint="pass subject=None to derive one per system, or a callable(system) -> ResultSubject",
                cli_hint="the command line names no subject: export a run with --run, or a report with --report",
            )
        subject_for = lambda system: subject  # noqa: E731 - the one-subject branch
    else:
        subject_for = lambda system: ResultSubject(kind="system", name=system, system=system)  # noqa: E731
    summary_dataset = dataset
    if summary and report.summary and summary_dataset is None:
        names = list(dict.fromkeys(row.dataset for row in report.per_dataset))
        if len(names) != 1:
            raise ConfigError(
                f"the report spans {len(names)} datasets {names}; the summary rows need dataset=",
                hint="pass dataset=ResultDataset(name=<the scored dataset>)",
                cli_hint="export a run with --run (it records the dataset name), or write the report with "
                "`rcp-ndcg eval score --out`",
                details={"datasets": names},
            )
        summary_dataset = ResultDataset(name=names[0])
    rows = [*report.per_dataset] if per_dataset else []
    totals = [*report.summary] if summary else []
    records: list[ResultRecord] = []
    for system in selected:
        for row in rows:
            if row.system != system:
                continue
            records.append(
                _record(
                    subject_for(system),
                    _scored(dataset if dataset is not None else ResultDataset(name=row.dataset), report),
                    ResultMetric(
                        metric=row.metric,
                        k=row.k,
                        value=row.value,
                        num_queries=row.num_queries,
                        dataset=row.dataset,
                    ),
                    artifacts=artifacts,
                    provenance=provenance,
                    created_at=moment,
                )
            )
        for row in totals:
            if row.system != system:
                continue
            assert summary_dataset is not None  # set above whenever summary rows are emitted
            records.append(
                _record(
                    subject_for(system),
                    _scored(summary_dataset, report),
                    ResultMetric(
                        metric=row.metric,
                        k=row.k,
                        value=row.value,
                        num_queries=row.num_queries,
                        num_datasets=row.num_datasets,
                        ci_low=row.ci_low,
                        ci_high=row.ci_high,
                    ),
                    artifacts=artifacts,
                    provenance=provenance,
                    created_at=moment,
                )
            )
    return records


def _scored(dataset: ResultDataset, report: EvalReport) -> ResultDataset:
    """``dataset`` with the report's protocol: the report's convention is what was scored, whatever the caller
    declared on the dataset."""
    return dataset.model_copy(
        update={
            "protocol": report.protocol.name,
            "protocol_spec": report.protocol,
            "gains_source": report.gains_source,
        }
    )


def _record(
    subject: ResultSubject,
    dataset: ResultDataset | None,
    metric: ResultMetric,
    *,
    artifacts: Sequence[ResultArtifact],
    provenance: Mapping[str, Any] | None,
    created_at: datetime,
) -> ResultRecord:
    return ResultRecord(
        record_id=record_identity(subject, dataset, [metric]),
        created_at=created_at,
        subject=subject,
        dataset=dataset,
        metrics=[metric],
        artifacts=list(artifacts),
        provenance=dict(provenance or {}),
    )


def records_from_run(
    run_dir: str | Path,
    *,
    report: str | Path | EvalReport | None = None,
    systems: Sequence[str] | None = None,
    include_reference: bool = False,
    created_at: datetime | None = None,
) -> list[ResultRecord]:
    """One record per ``(system, dataset, metric, cutoff)`` row of a run's evaluation report.

    The records carry the run's id and resolved config digest, the dataset's name and resolved revision, the
    run's artifacts with their hashes, and a per-system subject identity (the rankings file's digest, or the
    judge's family-key digest). The reference systems (``candidates``, ``judge``) are left out unless
    ``include_reference`` or named in ``systems``.

    Args:
        run_dir: The run directory (``runs/<run_id>``).
        report: The report to export: an :class:`~rcp_ndcg.eval.EvalReport`, a report file, or ``None`` to read
            the run's ``metrics/report.json`` (and recompute it from the run's artifacts when the file is
            absent).
        systems: Export only these systems (the report's order); default every system, minus the reference
            systems.
        include_reference: Also export ``candidates`` and ``judge``.
        created_at: The timestamp every record carries (default: now, once for the whole export).

    Returns:
        The records.

    Raises:
        MissingInputError: No run manifest at ``run_dir``, no report and no finished calibration to recompute
            one from.
        ConfigError: An unknown system name (the report's systems are listed).
        DataError: The run records no dataset name.
    """
    from rcp_ndcg.runs.layout import RunLayout
    from rcp_ndcg.runs.manifest import RunManifest

    layout = RunLayout.at(run_dir)
    if not Path(layout.manifest).exists():
        raise MissingInputError(
            f"no run manifest at {layout.root}", hint="pass a run directory (runs/<run_id>), not the runs root"
        )
    manifest = RunManifest.load(layout)
    value = _run_report(layout, manifest, report)
    available = value.systems
    if systems is None:
        selected = [system for system in available if include_reference or system not in REFERENCE_SYSTEMS]
    else:
        unknown = sorted(set(systems) - set(available))
        if unknown:
            raise ConfigError(
                f"systems {unknown} are not in the run's report; systems: {available}",
                hint="name one of the report's systems, or drop systems=",
                cli_hint="name one of the report's systems with --system, or drop --system",
                details={"unknown": unknown, "systems": available},
            )
        if not systems:
            raise ConfigError("systems names no system; pass the systems to export, or None for all")
        selected = [system for system in available if system in set(systems)]
    if not selected:
        if systems is None and not include_reference:
            raise ConfigError(
                f"the run's report scores only its reference systems {list(REFERENCE_SYSTEMS)}",
                hint="pass include_reference=True to export them, or systems=[...] to name the systems you want",
                cli_hint="pass --include-reference to export them, or --system NAME",
            )
        return []
    dataset = _run_dataset(manifest)
    artifacts = _run_artifacts(layout)
    provenance = {
        "code": manifest.code.model_dump(mode="json"),
        "run_config_sha256": hash_payload(manifest.config),
    }
    return records_from_report(
        value,
        subject=lambda system: ResultSubject(
            kind="system",
            name=system,
            system=system,
            run_id=manifest.run_id,
            identity=_system_identity(layout, manifest, system),
            labels=_system_labels(manifest, system),
        ),
        dataset=dataset,
        systems=selected,
        artifacts=artifacts,
        provenance=provenance,
        created_at=created_at,
    )


def _run_report(layout: Any, manifest: Any, report: str | Path | EvalReport | None) -> EvalReport:
    """The run's report: the argument, the run's ``metrics/report.json``, or one recomputed from its artifacts."""
    from rcp_ndcg.eval import EvalReport

    if isinstance(report, EvalReport):
        return report
    if report is not None:
        path = Path(report)
        if not path.is_file():
            raise MissingInputError(f"no report at {path}", hint="pass a report written by `eval score --out`")
        return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
    if Path(layout.metrics).exists():
        return EvalReport.model_validate_json(Path(layout.metrics).read_text(encoding="utf-8"))
    from rcp_ndcg.runs.inspect import evaluation_report

    return evaluation_report(layout.root)


def _run_dataset(manifest: Any) -> ResultDataset:
    """The run's dataset: its name, the provenance the manifest recorded, the resolved revision.

    The manifest's :class:`~rcp_ndcg.runs.manifest.DatasetRef` holds the subset, split and task the data was
    read at (``None`` on a manifest written before those fields existed); the config's ``subset`` is the
    fallback, and ``split``/``task`` keep the record's convention (``test``/``None``) when nothing recorded
    them.
    """
    name = manifest.dataset.name if manifest.dataset is not None else None
    source = manifest.config.get("dataset")
    if name is None and isinstance(source, str):
        name = source
    if name is None and isinstance(source, dict):
        name = source.get("uri")
    if not name:
        raise DataError(
            "the run records no dataset name; the manifest and the config name none",
            hint="re-run with a dataset (the config's `dataset:`), or export a report with --report",
        )
    recorded = manifest.dataset
    configured = source.get("subset") if isinstance(source, dict) else None
    subset = (recorded.subset if recorded is not None else None) or configured or "default"
    revision = None
    if recorded is not None and recorded.revisions:
        revision = next((entry.get("commit") for entry in recorded.revisions.values() if entry.get("commit")), None)
    return ResultDataset(
        name=str(name),
        subset=subset,
        split=(recorded.split if recorded is not None else None) or "test",
        task=recorded.task if recorded is not None else None,
        revision=revision,
    )


def _run_artifacts(layout: Any) -> list[ResultArtifact]:
    """The run's artifacts, run-relative, with the SHA-256 of every file (a directory is named without a hash)."""
    from rcp_ndcg.runs.manifest import MANIFEST_SCHEMA
    from rcp_ndcg.storage.artifacts import artifact_ref

    found: list[ResultArtifact] = []
    entries: tuple[tuple[_ArtifactRole, str, str | None], ...] = (
        ("manifest", layout.manifest, MANIFEST_SCHEMA),
        ("report", layout.metrics, "rcp-ndcg.eval-report.v1"),
        ("report", layout.comparison, "rcp-ndcg.comparison.v1"),
        ("rankings", layout.candidates, None),
        ("calibration", layout.calibration, None),
        ("judgements", layout.judgements, None),
        ("log", layout.log, None),
    )
    for role, path, schema_name in entries:
        uri = layout.relative(path)
        if Path(path).is_file():
            found.append(ResultArtifact(role=role, uri=uri, sha256=artifact_ref(path).sha256, schema_name=schema_name))
        elif Path(path).is_dir():
            found.append(ResultArtifact(role=role, uri=uri, schema_name=schema_name))
    return found


def _system_identity(layout: Any, manifest: Any, system: str) -> str | None:
    """The subject's content identity: the system's rankings digest, or the judge's family-key digest."""
    from rcp_ndcg.storage.artifacts import artifact_ref

    if system == CANDIDATES and Path(layout.candidates).is_file():
        return artifact_ref(layout.candidates).sha256
    if system == JUDGE:
        return hash_payload(sorted(manifest.families)) if manifest.families else None
    evaluation = manifest.config.get("evaluation")
    location = evaluation.get("systems", {}).get(system) if isinstance(evaluation, dict) else None
    if isinstance(location, str):
        path = location.split("#", 1)[0]
        if storage.exists(path):
            return artifact_ref(path).sha256
    return None


def _system_labels(manifest: Any, system: str) -> dict[str, str]:
    """The subject's provenance labels: the model or recipe the config names, when it names one."""
    config = manifest.config
    labels: dict[str, str] = {}
    if system == JUDGE:
        judge = config.get("judge")
        if isinstance(judge, str):
            labels["model"] = judge
        elif isinstance(judge, dict):
            _model_label(labels, judge)
        for key in sorted(manifest.families):
            labels.setdefault("judge_family", key)
    elif system == CANDIDATES:
        candidates = config.get("candidates")
        if isinstance(candidates, dict):
            for key in ("rerank", "retrieval"):
                block = candidates.get(key)
                if isinstance(block, dict) and _model_label(labels, block):
                    break
                encoder = block.get("encoder") if isinstance(block, dict) else None
                if isinstance(encoder, dict) and _model_label(labels, encoder):
                    break
    return labels


def _model_label(labels: dict[str, str], block: Mapping[str, Any]) -> bool:
    """Add ``model``/``recipe`` from ``block`` when it names one; return whether it did."""
    found = False
    for field in ("model", "recipe"):
        value = block.get(field)
        if isinstance(value, str) and value:
            labels[field] = value
            found = True
    return found


__all__ = [
    "RESULT_SCHEMA",
    "RESULTS_GROUP",
    "JsonlResultSink",
    "NullResultSink",
    "ParquetResultSink",
    "ResultArtifact",
    "ResultDataset",
    "ResultMetric",
    "ResultRecord",
    "ResultSubject",
    "ResultsSink",
    "record_identity",
    "records_from_report",
    "records_from_run",
    "registered_result_sinks",
    "result_sink_class",
]
