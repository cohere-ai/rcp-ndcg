"""``evaluate``: RCP-nDCG, qrel-nDCG and Count-nDCG of rankings under a scoring protocol, as an :class:`EvalReport`.

Every number comes from :func:`rcp_ndcg_core.protocol.score_query` (and through it :func:`rcp_ndcg_core.ndcg`);
this module only chooses the gains, walks the systems, datasets and queries, and aggregates.

**Gains** for RCP-nDCG are resolved in one order: the ``gains`` argument (a ``{query_id: {doc_id: gain}}`` mapping,
or any object with a ``gains()`` method, such as a calibration), else the released gains of the dataset (the
``gain`` column of the HF qrels), else a :class:`~rcp_ndcg.errors.DataError`. Integer qrels are never used as
RCP gains.

**Rankings of a suite** are read per subset: the rows whose ``dataset`` names the subset, else the rows that name
no dataset (:meth:`~rcp_ndcg.data.Rankings.resolve_dataset`). Rows that name no dataset are refused when subsets
that read them share query ids, since a query id alone does not say which subset it belongs to.

**Queries.** Each metric is scored on the queries that have its labels: RCP-nDCG on the queries with gains,
Count-nDCG on those with count gains, qrel-nDCG on those with qrels. A labelled query that a system did not rank
scores 0 and is reported in :attr:`EvalReport.warnings`. qrel-nDCG is undefined (``None``) for a query without a
positive grade and drops out of the means.

**Aggregation** follows the paper: the mean over queries per dataset, then the unweighted mean over datasets. The
summary interval is a percentile bootstrap that resamples queries within each dataset (query-clustered,
stratified by dataset) with a fixed seed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from rcp_ndcg_core.protocol import PROTOCOLS, MetricName, Protocol, aggregate, resolve_protocol, score_query

from rcp_ndcg.data.dataset import Dataset, load_dataset
from rcp_ndcg.data.rankings import Rankings
from rcp_ndcg.errors import ConfigError, DataError, WarningCode

if TYPE_CHECKING:
    import pandas as pd

METRICS: tuple[MetricName, ...] = ("rcp_ndcg", "qrel_ndcg", "count_ndcg")

GainsSource = Literal["gains", "calibration", "dataset", "none"]


class ReportWarning(BaseModel):
    """A typed warning: a ``code`` from a closed list and a message.

    The report's own codes, plus the warnings of the calibration whose gains it scores
    (:data:`rcp_ndcg.errors.WarningCode`, e.g. ``INVALID_WINDOWS``).
    """

    model_config = ConfigDict(frozen=True)

    code: Literal["UNRANKED_QUERIES", "NO_POSITIVE_QRELS"] | WarningCode
    message: str


class QueryValue(BaseModel):
    """One metric value of one system on one query (``value`` is ``None`` when the metric is undefined)."""

    model_config = ConfigDict(frozen=True)

    system: str
    dataset: str
    query_id: str
    metric: MetricName
    k: int
    value: float | None


class DatasetValue(BaseModel):
    """The mean of one metric over one dataset's queries (``num_queries``: the queries with a defined value)."""

    model_config = ConfigDict(frozen=True)

    system: str
    dataset: str
    metric: MetricName
    k: int
    value: float | None
    num_queries: int


class SummaryValue(BaseModel):
    """One system's headline number for a metric: the protocol's aggregate, with a bootstrap interval.

    ``num_queries`` counts the queries with a defined value, and ``num_datasets`` the datasets with one.
    """

    model_config = ConfigDict(frozen=True)

    system: str
    metric: MetricName
    k: int
    value: float | None
    ci_low: float | None
    ci_high: float | None
    num_queries: int
    num_datasets: int


class ReportInputs(BaseModel):
    """Where a report's inputs came from, as the command line that scored them names them.

    ``rcp-ndcg eval score --out`` records them, and ``rcp-ndcg eval explain --report`` reads them again; a report
    from :func:`evaluate` in Python has none (it keeps the objects it scored instead).

    Attributes:
        rankings: The rankings file.
        suite: The public suite scored against, or ``None``.
        dataset: The dataset URI scored against, or ``None``.
        subset: The subset of a ``hf://`` or ``suite:`` dataset.
        revision: The Hub commit the data was read at (the revision as given when it could not be resolved).
        calibration: The calibration (or run) directory whose gains were scored, or ``None``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rankings: str
    suite: str | None = None
    dataset: str | None = None
    subset: str | None = None
    revision: str | None = None
    calibration: str | None = None


class EvalReport(BaseModel):
    """The result of :func:`evaluate`.

    Attributes:
        schema_name: ``"rcp-ndcg.eval-report.v1"`` (serialised as ``schema``).
        protocol: The scoring protocol every value was computed under.
        gains_source: Where the RCP gains came from: ``"gains"`` (passed in), ``"calibration"`` (its ``gains()``),
            ``"dataset"`` (the released gains), or ``"none"`` (RCP-nDCG not computed).
        metrics: The metrics computed.
        k: The cutoffs.
        per_query: Every value, one row per (system, dataset, query, metric, k).
        per_dataset: The mean per (system, dataset, metric, k).
        summary: The aggregate per (system, metric, k), with its bootstrap interval.
        warnings: What the numbers do not show on their own.
        inputs: The input files, when the command line recorded them (:class:`ReportInputs`); ``None`` from Python.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    schema_name: Literal["rcp-ndcg.eval-report.v1"] = Field(default="rcp-ndcg.eval-report.v1", alias="schema")
    protocol: Protocol
    gains_source: GainsSource
    metrics: list[MetricName]
    k: list[int]
    per_query: list[QueryValue]
    per_dataset: list[DatasetValue]
    summary: list[SummaryValue]
    warnings: list[ReportWarning] = []
    inputs: ReportInputs | None = None

    _inputs: dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def systems(self) -> list[str]:
        """The systems, in the order of the rankings."""
        return list(dict.fromkeys(row.system for row in self.summary))

    def value(self, system: str, metric: MetricName = "rcp_ndcg", k: int | None = None) -> float | None:
        """The summary value of one system (``k`` may be omitted when the report has one cutoff)."""
        k = _one_k(self, k)
        for row in self.summary:
            if (row.system, row.metric, row.k) == (system, metric, k):
                return row.value
        raise DataError(f"the report has no {metric}@{k} for system {system!r}")

    def to_json(self, **kwargs: Any) -> str:
        """The report as JSON (``kwargs`` go to :meth:`pydantic.BaseModel.model_dump_json`)."""
        return self.model_dump_json(by_alias=True, **kwargs)

    def to_pandas(self, table: Literal["per_query", "per_dataset", "summary"] = "per_query") -> pd.DataFrame:
        """One of the three tables as a data frame (long format: one row per value)."""
        import pandas as pd

        rows = getattr(self, table)
        columns = list(type(rows[0]).model_fields) if rows else list(_TABLE_ROWS[table].model_fields)
        return pd.DataFrame([row.model_dump() for row in rows], columns=columns)

    def leaderboard(self, metric: MetricName = "rcp_ndcg", k: int | None = None) -> pd.DataFrame:
        """The wide table of one metric: a row per system, a column per dataset, and the summary as ``mean``.

        ``mean`` is the protocol's aggregate (the unweighted mean over datasets), so it equals :meth:`value`.
        Rows are sorted by ``mean``, best first.

        Args:
            metric: The metric.
            k: The cutoff (may be omitted when the report has one).
        """
        import pandas as pd

        k = _one_k(self, k)
        if metric not in self.metrics:
            raise DataError(f"the report has no {metric}; it has {self.metrics}")
        cells = {(r.system, r.dataset): r.value for r in self.per_dataset if (r.metric, r.k) == (metric, k)}
        datasets = list(dict.fromkeys(dataset for _, dataset in cells))
        means = {r.system: r.value for r in self.summary if (r.metric, r.k) == (metric, k)}
        frame = pd.DataFrame(
            [[cells.get((system, d)) for d in datasets] + [means.get(system)] for system in self.systems],
            index=pd.Index(self.systems, name="system"),
            columns=[*datasets, "mean"],
            dtype=float,
        )
        return frame.sort_values("mean", ascending=False, kind="mergesort")

    def __repr__(self) -> str:
        datasets = len({row.dataset for row in self.per_dataset})
        return (
            f"EvalReport(protocol={self.protocol.name}, {len(self.systems)} systems, {datasets} datasets, "
            f"metrics={self.metrics}, k={self.k}, gains from {self.gains_source})"
        )

    __str__ = __repr__


_TABLE_ROWS: dict[str, type[BaseModel]] = {
    "per_query": QueryValue,
    "per_dataset": DatasetValue,
    "summary": SummaryValue,
}


def evaluate(
    rankings: Rankings,
    *,
    suite: str | None = None,
    dataset: Dataset | None = None,
    gains: Mapping[str, Mapping[str, float]] | Any | None = None,
    protocol: str | Protocol | None = None,
    k: int | Sequence[int] = 10,
    metrics: Sequence[MetricName] = ("rcp_ndcg", "qrel_ndcg"),
    count_gains: Mapping[str, Mapping[str, float]] | None = None,
    bootstrap: int = 1000,
    seed: int = 0,
) -> EvalReport:
    """Score rankings with RCP-nDCG, qrel-nDCG and Count-nDCG under a scoring protocol.

    Args:
        rankings: The systems' scores (:class:`~rcp_ndcg.data.Rankings`).
        suite: A public suite (``nanobeir``, ``bright``, ``vidore``, ``trecdl``): its data and its protocol.
        dataset: The dataset instead of a suite: qrels, pools, exclusions, and possibly released gains.
        gains: RCP gains in ``[0, 1]``, ``{query_id: {doc_id: gain}}``, or any object with a ``gains()`` method
            returning that (a calibration). For a suite, keys may also be ``"<subset>/<query_id>"``, and must be
            when subsets share query ids. A calibration gives each subset of a suite the gains of its dataset of
            that name (``gains(dataset=<subset>)``).
        protocol: A :data:`~rcp_ndcg_core.protocol.PROTOCOLS` name or a :class:`~rcp_ndcg_core.protocol.Protocol`;
            defaults to the dataset's (a suite's) protocol, else ``"plain"``.
        k: One cutoff or several.
        metrics: Which of ``"rcp_ndcg"``, ``"qrel_ndcg"``, ``"count_ndcg"`` to compute.
        count_gains: The Count-nDCG gains (:func:`rcp_ndcg_core.count_gain` per document), required for
            ``"count_ndcg"``.
        bootstrap: Bootstrap resamples for the summary interval (0: no interval).
        seed: The bootstrap seed.

    Returns:
        The :class:`EvalReport`.

    Raises:
        ConfigError: Conflicting or unknown arguments.
        DataError: No gains for RCP-nDCG, gains outside ``[0, 1]``, a protocol that needs pools the dataset
            lacks, or rankings (or gains) keyed by bare query ids over subsets that share query ids.
    """
    ks = sorted({k} if isinstance(k, int) else set(k))
    if not ks or ks[0] <= 0:
        raise ConfigError(f"k must be positive cutoffs, got {k!r}")
    unknown = sorted(set(metrics) - set(METRICS))
    if unknown or not metrics:
        raise ConfigError(f"unknown metrics {unknown or list(metrics)}; expected some of {list(METRICS)}")
    if suite is not None and dataset is not None:
        raise ConfigError("pass either suite= or dataset=, not both")
    if suite is not None:
        dataset = load_dataset(f"suite:{suite}")
    if dataset is None:
        raise ConfigError("evaluate needs the data to score against: suite= or dataset=")
    rules = _protocol(protocol if protocol is not None else dataset.protocol or "plain")
    _refuse_undivided(rankings, dataset)

    rcp_gains, source = _resolve_gains(gains, dataset) if "rcp_ndcg" in metrics else (None, "none")
    if "count_ndcg" in metrics and count_gains is None:
        raise DataError("count_ndcg needs count_gains= (the share of passed rubric criteria per document)")
    labels: dict[MetricName, dict[str, Mapping[str, Mapping[str, float]]]] = {}
    for part in dataset.parts:
        if rcp_gains is not None:
            labels.setdefault("rcp_ndcg", {})[part.name] = _gains_for(rcp_gains[part.name], part, dataset)
        if count_gains is not None and "count_ndcg" in metrics:
            labels.setdefault("count_ndcg", {})[part.name] = _gains_for(count_gains, part, dataset)
        if "qrel_ndcg" in metrics:
            labels.setdefault("qrel_ndcg", {})[part.name] = part.qrels

    computed: list[MetricName] = [m for m in metrics if m in labels]
    per_query: list[QueryValue] = []
    unranked: dict[str, set[tuple[str, str]]] = {}
    for system in rankings.systems:
        for part in dataset.parts:
            scores = _system_queries(rankings, system, part.name)
            for metric in computed:
                for query_id, query_labels in labels[metric][part.name].items():
                    ranked = scores.get(query_id)
                    if ranked is None:
                        unranked.setdefault(system, set()).add((part.name, query_id))
                    for cutoff in ks:
                        value = _score(rules, ranked or {}, query_labels, part, query_id, metric, cutoff)
                        per_query.append(
                            QueryValue(
                                system=system,
                                dataset=part.name,
                                query_id=query_id,
                                metric=metric,
                                k=cutoff,
                                value=None if math.isnan(value) else value,
                            )
                        )

    per_dataset, summary = _aggregate(per_query, bootstrap=bootstrap, seed=seed)
    calibration_warnings = (getattr(gains, "warnings", None) or []) if source == "calibration" else []
    warnings = [
        ReportWarning(code=warning["code"], message=f"calibration: {warning['message']}")
        for warning in calibration_warnings
    ]
    warnings += [
        ReportWarning(
            code="UNRANKED_QUERIES", message=f"{system}: {len(missing)} labelled queries have no ranking; scored 0"
        )
        for system, missing in unranked.items()
    ]
    undefined = sum(1 for row in per_query if row.value is None)
    if undefined:
        warnings.append(
            ReportWarning(
                code="NO_POSITIVE_QRELS",
                message=f"{undefined} qrel-nDCG values are undefined (no positive grade) and left out of the means",
            )
        )
    report = EvalReport(
        protocol=rules,
        gains_source=source,
        metrics=computed,
        k=ks,
        per_query=per_query,
        per_dataset=per_dataset,
        summary=summary,
        warnings=warnings,
    )
    report._inputs.update(rankings=rankings, dataset=dataset, labels=labels)
    return report


def _protocol(protocol: str | Protocol) -> Protocol:
    """The core's resolution of a protocol name (:data:`~rcp_ndcg_core.protocol.PROTOCOLS`), as a typed error."""
    try:
        return resolve_protocol(protocol)
    except ValueError as exc:
        raise ConfigError(str(exc), hint=f"presets: {sorted(PROTOCOLS)}") from exc


def _resolve_gains(gains: Any, dataset: Dataset) -> tuple[dict[str, Mapping[str, Mapping[str, float]]], GainsSource]:
    """``{part name: gains}``: the gains that :func:`_gains_for` reads for each part, and where they came from.

    A calibration that names its datasets (``datasets``) gives each part of a suite, and a dataset when it holds
    several, the gains of its dataset of that name (``gains(dataset=<part>)``); its undivided ``gains()`` keys
    ``<dataset>||<query_id>``. The gains of one dataset scored alone need no name.

    Raises:
        DataError: No gains at all, gains that match no labelled query of the dataset, a calibration that holds
            none of the suite's parts, or bare query-id keys for a suite whose parts share query ids.
    """
    parts = dataset.parts
    resolved: dict[str, Mapping[str, Mapping[str, float]]]
    if gains is not None:
        if callable(getattr(gains, "gains", None)):
            held = list(getattr(gains, "datasets", None) or [])
            if len(held) > 1 or (held and dataset.subsets):
                names = [part.name for part in parts]
                if not set(held) & set(names):
                    raise DataError(
                        f"the calibration holds the datasets {held}, and {dataset.name!r} has none of them: {names}",
                        hint="pass the calibration of these datasets, or score the dataset it holds",
                        details={"calibration": held, "datasets": names},
                    )
                resolved = {part.name: gains.gains(dataset=part.name) if part.name in held else {} for part in parts}
            else:
                undivided: Mapping[str, Mapping[str, float]] = gains.gains()
                _refuse_bare_keys(undivided, dataset)
                resolved = dict.fromkeys((part.name for part in parts), undivided)
            source: GainsSource = "calibration"
        elif isinstance(gains, Mapping):
            resolved, source = dict.fromkeys((part.name for part in parts), gains), "gains"
            _refuse_bare_keys(gains, dataset)
        else:
            raise ConfigError(f"gains must be a mapping or have a gains() method, got {type(gains).__name__}")
        if not any(_gains_for(resolved[part.name], part, dataset) for part in parts):
            raise DataError(
                f"the gains match no labelled query of {dataset.name!r}",
                hint="key the gains by the dataset's query ids (or '<subset>/<query_id>' for a suite), or pass the "
                "calibration of this dataset",
            )
        return resolved, source
    released = {part.name: part.gains for part in parts if part.gains is not None}
    if len(released) == len(parts):
        keyed: dict[str, Mapping[str, Mapping[str, float]]] = {
            name: {f"{name}/{q}": docs for q, docs in queries.items()} for name, queries in released.items()
        }
        return keyed, "dataset"
    raise DataError(
        "RCP-nDCG needs calibrated gains: a calibration (gains= in Python, --calibration DIR on the command line) "
        "or a dataset with released gains; integer qrels are not RCP gains",
        hint="pass gains=, or score the qrels alone with metrics=['qrel_ndcg']",
        cli_hint="pass --calibration DIR, or score the qrels alone with --metrics qrel_ndcg",
    )


def _gains_for(
    gains: Mapping[str, Mapping[str, float]], part: Dataset, dataset: Dataset
) -> dict[str, Mapping[str, float]]:
    """The gains of one dataset part: keys ``"<part>/<query_id>"``, or bare query ids for a single dataset."""
    prefix = f"{part.name}/"
    out = {key[len(prefix) :]: docs for key, docs in gains.items() if key.startswith(prefix)}
    if not out:  # bare query ids: all of them for one dataset, the part's own queries for a suite
        out = {q: docs for q, docs in gains.items() if not dataset.subsets or q in part.qrels}
    for query_id, docs in out.items():
        bad = [d for d, g in docs.items() if not (0.0 <= g <= 1.0)]
        if bad:
            raise DataError(
                f"gains of query {query_id!r} fall outside [0, 1] ({bad[:3]}); RCP and Count gains are probabilities",
                hint="grades belong in the dataset's qrels and are scored as qrel_ndcg",
            )
    return out


def _system_queries(rankings: Rankings, system: str, dataset: str) -> dict[str, dict[str, float]]:
    """A system's scores for one dataset (:meth:`Rankings.resolve_dataset`); empty when no row ranks it."""
    if rankings.resolve_dataset(dataset) is None:
        return {}
    return rankings.queries(system=system, dataset=dataset)


def _shared_query_ids(parts: Iterable[Dataset]) -> dict[str, list[str]]:
    """``{query_id: [part, ...]}`` of the query ids that more than one of ``parts`` labels or pools."""
    owners: dict[str, list[str]] = {}
    for part in parts:
        for query_id in set(part.qrels) | set(part.gains or {}) | set(part.candidates or {}):
            owners.setdefault(query_id, []).append(part.name)
    return {query_id: names for query_id, names in owners.items() if len(names) > 1}


def _collision_error(what: str, shared: Mapping[str, list[str]], fix: str, cli_fix: str) -> DataError:
    subsets = sorted({name for names in shared.values() for name in names})
    examples = [f"{names[0]}/{q} vs {names[1]}/{q}" for q, names in sorted(shared.items())[:3]]
    return DataError(
        f"{what}, and the subsets {subsets} share query ids ({len(shared)}, e.g. {', '.join(examples)}), so a "
        "query id does not say which subset it belongs to",
        hint=fix,
        cli_hint=cli_fix,
        details={"subsets": subsets, "shared_query_ids": sorted(shared)[:20], "num_shared": len(shared)},
    )


def _refuse_undivided(rankings: Rankings, dataset: Dataset) -> None:
    """Refuse rows that name no dataset when the subsets that would read them share query ids.

    Raises:
        DataError: Naming the subsets that share ids, and the fix.
    """
    if not dataset.subsets:
        return
    readers = [part for part in dataset.parts if rankings.resolve_dataset(part.name) == ""]
    shared = _shared_query_ids(readers)
    if shared:
        raise _collision_error(
            "the rankings name no dataset (no `dataset` column)",
            shared,
            "add a `dataset` column naming each row's subset (parquet, CSV, JSONL; "
            "load_rankings(path, dataset=<subset>) for a TREC run of one subset), or score one subset "
            "(load_dataset(..., subset=<subset>))",
            "add a `dataset` column naming each row's subset (parquet, CSV, JSONL), or score one subset with "
            "--subset (a TREC run holds one subset per file)",
        )


def _refuse_bare_keys(gains: Mapping[str, Any], dataset: Dataset) -> None:
    """Refuse gains keyed by bare query ids that more than one part of a suite has.

    Raises:
        DataError: Naming the subsets that share ids, and the fix.
    """
    if not dataset.subsets:
        return
    prefixes = tuple(f"{part.name}/" for part in dataset.parts)
    bare = {key for key in gains if not key.startswith(prefixes)}
    shared = {q: names for q, names in _shared_query_ids(dataset.parts).items() if q in bare}
    if shared:
        raise _collision_error(
            "the gains are keyed by bare query ids",
            shared,
            "key them '<subset>/<query_id>', or pass the calibration of the subsets",
            "pass the calibration of the subsets, or score one subset with --subset",
        )


def _score(
    rules: Protocol,
    scores: Mapping[str, float],
    labels: Mapping[str, float],
    part: Dataset,
    query_id: str,
    metric: MetricName,
    k: int,
) -> float:
    candidates = (part.candidates or {}).get(query_id)
    if candidates is None and (rules.restrict_to_candidates or rules.ties == "input_order"):
        raise DataError(
            f"protocol {rules.name!r} scores each query's judged pool, and {part.name!r} has none for {query_id!r}",
            hint="load the dataset with its pools (HF top_ranked), or score it with protocol='plain'",
            cli_hint="load the dataset with its pools (HF top_ranked), or score it with --protocol plain",
        )
    return score_query(
        scores,
        labels,
        protocol=rules,
        k=k,
        metric=metric,
        candidates=candidates,
        excluded=part.excluded.get(query_id, ()),
        query_id=query_id,
    )


def _aggregate(
    per_query: Iterable[QueryValue], *, bootstrap: int, seed: int
) -> tuple[list[DatasetValue], list[SummaryValue]]:
    groups: dict[tuple[str, MetricName, int], dict[str, list[float]]] = {}
    for row in per_query:
        values = groups.setdefault((row.system, row.metric, row.k), {}).setdefault(row.dataset, [])
        if row.value is not None:
            values.append(row.value)
    per_dataset: list[DatasetValue] = []
    summary: list[SummaryValue] = []
    for (system, metric, k), datasets in groups.items():
        means = {}
        for name, values in datasets.items():
            means[name] = math.fsum(values) / len(values) if values else None
            per_dataset.append(
                DatasetValue(
                    system=system,
                    dataset=name,
                    metric=metric,
                    k=k,
                    value=means[name],
                    num_queries=len(values),
                )
            )
        defined = [m for m in means.values() if m is not None]
        low, high = bootstrap_interval(datasets, resamples=bootstrap, seed=seed)
        value = _aggregate_values(datasets)
        summary.append(
            SummaryValue(
                system=system,
                metric=metric,
                k=k,
                value=None if math.isnan(value) else value,
                ci_low=low,
                ci_high=high,
                num_queries=sum(len(v) for v in datasets.values()),
                num_datasets=len(defined),
            )
        )
    return per_dataset, summary


def _aggregate_values(datasets: Mapping[str, Sequence[float] | np.ndarray]) -> float:
    """:func:`~rcp_ndcg_core.protocol.aggregate` of ``{dataset: per-query values}``: it reads the values alone, so
    their positions stand in for the query ids."""
    return aggregate({name: {str(i): value for i, value in enumerate(values)} for name, values in datasets.items()})


def bootstrap_interval(
    datasets: Mapping[str, Sequence[float] | np.ndarray], *, resamples: int, seed: int, alpha: float = 0.05
) -> tuple[float | None, float | None]:
    """Percentile interval of the dataset-mean-then-mean aggregate, resampling queries within each dataset.

    Args:
        datasets: ``{dataset: per-query values}``.
        resamples: Bootstrap resamples (0: no interval).
        seed: The random seed.
        alpha: One minus the coverage.

    Returns:
        ``(low, high)``, or ``(None, None)`` without resamples or values.
    """
    draws = _stratified_draws({d: np.asarray(v, float) for d, v in datasets.items() if len(v)}, resamples, seed)
    if draws is None:
        return None, None
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def _stratified_draws(datasets: Mapping[str, np.ndarray], resamples: int, seed: int) -> np.ndarray | None:
    """``resamples`` bootstrap draws of the aggregate; one query resample per dataset and draw."""
    if resamples <= 0 or not datasets:
        return None
    rng = np.random.default_rng(seed)
    means = [
        values[rng.integers(0, len(values), (resamples, len(values)))].mean(axis=1) for values in datasets.values()
    ]
    return np.mean(means, axis=0)


def _one_k(report: EvalReport, k: int | None) -> int:
    if k is not None:
        return k
    if len(report.k) != 1:
        raise DataError(f"the report has cutoffs {report.k}; pass k=")
    return report.k[0]


__all__ = [
    "METRICS",
    "DatasetValue",
    "EvalReport",
    "QueryValue",
    "ReportInputs",
    "ReportWarning",
    "SummaryValue",
    "bootstrap_interval",
    "evaluate",
]
