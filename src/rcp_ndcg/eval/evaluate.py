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
positive grade and drops out of the means. A system whose rankings match nothing of the dataset — no row names
any of its subsets, or not one ranked id is in its pools or labels — is refused with a
:class:`~rcp_ndcg.errors.DataError`: every score would be 0, which reads as a weak system where the input is
broken. ``systems=`` (``--system`` on the command line) scores only the named systems, so one broken system of
a multi-system file no longer stops the others; an unknown name is refused, listing the systems the file holds.

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
from rcp_ndcg.data.rankings import Rankings, no_rankings_error
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

        Raises:
            DataError: The report has no such metric, or no ``(metric, k)`` row (the error names the cutoffs
                it has), so a wrong k never returns an empty table.
        """
        import pandas as pd

        k = _one_k(self, k)
        if metric not in self.metrics:
            raise DataError(f"the report has no {metric}; it has {self.metrics}")
        _refuse_missing_cutoff(self, metric, k)
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
    systems: Sequence[str] | None = None,
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
            ``"count_ndcg"``. Keys are the gains': ``{query_id: {doc_id: gain}}``, or ``"<subset>/<query_id>"``
            for a suite -- and must be when subsets share query ids.
        systems: The systems to score, in the rankings' order in the report (``None``: every system the rankings
            hold). Use it to score the healthy systems of a file one of whose systems matches nothing of the
            dataset (``--system`` on the command line, repeatable).
        bootstrap: Bootstrap resamples for the summary interval (0: no interval).
        seed: The bootstrap seed.

    Returns:
        The :class:`EvalReport`.

    Raises:
        ConfigError: Conflicting or unknown arguments, or a ``systems`` name the rankings do not hold (the error
            lists the systems they do).
        DataError: No gains for RCP-nDCG, gains outside ``[0, 1]``, a protocol that needs pools the dataset
            lacks, rankings or gains (RCP or count) keyed by bare query ids over subsets that share query ids,
            gains that mix the ``"<subset>/<query_id>"`` and bare styles for one subset, gains or count gains
            that match no labelled query, or a system's rankings that match
            nothing of the scored dataset (no row names any of its subsets, or not one ranked document id is in
            its pools or labels): every score would be 0.
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
    selected = _selected_systems(rankings, systems)

    rcp_gains, source = _resolve_gains(gains, dataset) if "rcp_ndcg" in metrics else (None, "none")
    if "count_ndcg" in metrics:
        # The gains are validated where they are scored: a qrel-only run that carries count gains for a later
        # run reads them as it left them, exactly as gains= on a qrel-only run reads the RCP ones.
        if count_gains is None:
            raise DataError("count_ndcg needs count_gains= (the share of passed rubric criteria per document)")
        _refuse_bare_keys(count_gains, dataset)
        _refuse_unknown_prefixes(count_gains, dataset)
        if not any(_gains_for(count_gains, part, dataset) for part in dataset.parts):
            raise DataError(
                f"the count gains match no labelled query of {dataset.name!r}",
                hint="key them by the dataset's query ids (or '<subset>/<query_id>' for a suite)",
            )
    labels: dict[MetricName, dict[str, Mapping[str, Mapping[str, float]]]] = {}
    for part in dataset.parts:
        if rcp_gains is not None:
            labels.setdefault("rcp_ndcg", {})[part.name] = _gains_for(rcp_gains[part.name], part, dataset)
        if count_gains is not None and "count_ndcg" in metrics:
            labels.setdefault("count_ndcg", {})[part.name] = _gains_for(count_gains, part, dataset)
        if "qrel_ndcg" in metrics:
            labels.setdefault("qrel_ndcg", {})[part.name] = part.qrels

    computed: list[MetricName] = [m for m in metrics if m in labels]
    dataset_pools, dataset_labels = _dataset_targets(dataset)
    per_query: list[QueryValue] = []
    unranked: dict[str, set[tuple[str, str]]] = {}
    for system in selected:
        scores = {part.name: _system_queries(rankings, system, part.name) for part in dataset.parts}
        _refuse_unmatched(rankings, dataset, system, scores, dataset_pools, dataset_labels)
        for part in dataset.parts:
            for metric in computed:
                for query_id, query_labels in labels[metric][part.name].items():
                    ranked = scores[part.name].get(query_id)
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
            code="UNRANKED_QUERIES",
            message=f"{system}: {len(missing)} labelled queries have no ranking; scored 0"
            + (f" (subsets: {', '.join(sorted({name for name, _ in missing}))})" if len(dataset.parts) > 1 else ""),
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


def _selected_systems(rankings: Rankings, systems: Sequence[str] | None) -> list[str]:
    """The systems to score: the named ones in the rankings' order, or every system of the rankings.

    Raises:
        ConfigError: A name the rankings do not hold (the error lists the systems they do), or an empty sequence.
    """
    if systems is None:
        return rankings.systems
    held = rankings.systems
    unknown = sorted(set(systems) - set(held))
    if unknown:
        raise ConfigError(
            f"systems {unknown} are not in the rankings; systems: {held}",
            hint="score one of the systems the rankings hold (--system, repeatable)",
            details={"unknown": unknown, "systems": held},
        )
    if not systems:
        raise ConfigError("systems names no system; pass the systems to score, or None (the default) for all")
    return [name for name in held if name in set(systems)]


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
            _refuse_unknown_prefixes(gains, dataset)
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
    """The gains of one dataset part: keys ``"<part>/<query_id>"``, or bare query ids for a single dataset.

    A part reads either the gains keyed with its own prefix, or the bare-keyed ones -- never a mix of the two
    styles for one part, which would quietly score the prefixed queries with their gains and the bare-keyed
    ones without. A bare key is this part's when it labels the query (qrels or released gains -- the module's
    labelled-query definition, the one :func:`rcp_ndcg.eval.explain` reads too); a dataset scored alone has no
    subset to disambiguate, so every bare key is its, whatever the qrels say.

    Raises:
        DataError: The gains mix ``"<part>/<query_id>"`` keys with bare query ids of this part, or a gain
            falls outside ``[0, 1]``.
    """
    prefix = f"{part.name}/"
    prefixed = {key[len(prefix) :]: docs for key, docs in gains.items() if key.startswith(prefix)}
    if dataset.subsets:
        bare = {q: docs for q, docs in gains.items() if q in set(part.qrels) | set(part.gains or {})}
    else:  # one dataset, no ambiguity: every bare key is its, unlabelled queries included
        bare = {q: docs for q, docs in gains.items() if not q.startswith(prefix)}
    if prefixed and bare:
        first_prefixed = sorted(key for key in gains if key.startswith(prefix))[0]
        raise DataError(
            f"the gains of {part.name!r} mix '<subset>/<query_id>' keys with bare query ids (e.g. "
            f"{first_prefixed!r} and {sorted(bare)[0]!r}): the prefixed ones would win and the "
            "bare-keyed queries would silently lose their gains",
            hint="key every gain '<subset>/<query_id>' (a suite's parts may share query ids), or key them all "
            "by bare query id",
        )
    out = prefixed or bare
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


def _refuse_unmatched(
    rankings: Rankings,
    dataset: Dataset,
    system: str,
    scores: Mapping[str, dict[str, dict[str, float]]],
    pools: set[str],
    label_ids: set[str],
) -> None:
    """Refuse a system whose rankings match nothing of the scored dataset: every score would be 0.

    Two broken inputs otherwise score every query 0 with ``ok``; both are refused instead:

    * no row of the system names any subset of the scored dataset (the ``dataset`` column holds another name);
    * the system ranks rows, but not one ranked document id is in the dataset's pools or labels (qrels, gains).

    A system that matches some subsets, or some documents, keeps scoring as before: the missing subsets score 0
    with ``UNRANKED_QUERIES`` (naming them), and out-of-pool documents score 0 silently.

    Args:
        rankings: The systems' scores, for the datasets the rows name.
        dataset: The scored dataset.
        system: The system being scored; the error names it.
        scores: The system's per-subset scores, as :func:`_system_queries` returned them (read once, reused for
            the scoring, so the check adds no pass over the rankings).
        pools: The dataset's pool ids (``top_ranked``), from :func:`_dataset_targets` (loop-invariant).
        label_ids: The dataset's label ids (qrels, released gains), from :func:`_dataset_targets`.

    Raises:
        DataError: Naming the system and the datasets its rows do name, or one ranked id next to one dataset id.
    """
    if not any(scores.values()):
        error = no_rankings_error(dataset.name, rankings.datasets, system=system, hint=_dataset_column_hint(dataset))
        raise _way_out_for_the_others(error, rankings)
    ranked = {doc_id for part_scores in scores.values() for docs in part_scores.values() for doc_id in docs}
    targets = pools | label_ids
    if targets and not ranked & targets:
        raise _way_out_for_the_others(_no_overlap_error(system, dataset, ranked, pools, label_ids), rankings)


def _way_out_for_the_others(error: DataError, rankings: Rankings) -> DataError:
    """A system's refusal, with the way out for the other systems of a multi-system file appended.

    One broken system would fail the scoring of every system in the file; the hint names the way out: drop the
    broken system's rows, or score the others (``systems=`` in Python, ``--system`` on the command line). A
    single-system file keeps its own hint: neither way out has anything left to score.
    """
    if len(rankings.systems) <= 1:
        return error
    python = "drop this system's rows, or score the others with systems=[...]"
    cli = "drop this system's rows, or score the others with --system (repeatable)"
    base = error.hint
    error.hint = f"{base}; {python}" if base else python
    error.cli_hint = f"{base}; {cli}" if base else cli
    return error


def _dataset_targets(dataset: Dataset) -> tuple[set[str], set[str]]:
    """``(pool ids, label ids)`` of the whole scored dataset: the pools (``top_ranked``) and the doc ids of its
    qrels and released gains, across every subset. What a ranked id must hit for the system to be scoring this
    dataset at all."""
    pools = {doc_id for part in dataset.parts for pool in (part.candidates or {}).values() for doc_id in pool}
    label_ids = {
        doc_id
        for part in dataset.parts
        for docs in [*part.qrels.values(), *(part.gains or {}).values()]
        for doc_id in docs
    }
    return pools, label_ids


def _dataset_column_hint(dataset: Dataset) -> str:
    """The fix for rows whose ``dataset`` column names no subset of the scored dataset: the exact names."""
    names = [part.name for part in dataset.parts]
    if len(names) > 1:
        return f"the `dataset` column must hold the exact subset name, one of {names}"
    (name,) = names
    stem = name.partition("__")[0]
    if stem and stem != name:
        return f"the `dataset` column must hold the exact subset name (e.g. {name!r}, not {stem!r})"
    return f"the `dataset` column must hold the exact dataset name (e.g. {name!r})"


def _no_overlap_error(system: str, dataset: Dataset, ranked: set[str], pools: set[str], labels: set[str]) -> DataError:
    """The error for a system whose every ranked document is outside the dataset's pools and labels."""
    target, kind = (pools, "pool") if pools else (labels, "label")
    ranked_id, dataset_id = min(ranked), min(target)
    return DataError(
        f"system {system!r}: no ranked document is in the pools or labels of {dataset.name!r} "
        f"(e.g. ranked {ranked_id!r} vs {kind} {dataset_id!r})",
        hint=f"rank the dataset's own document ids (e.g. {dataset_id!r}); ids outside its pools and qrels score 0",
        details={
            "system": system,
            "dataset": dataset.name,
            "ranked_doc_id": ranked_id,
            "dataset_doc_id": dataset_id,
            "num_ranked_docs": len(ranked),
            "num_dataset_docs": len(pools | labels),
        },
    )


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


def _refuse_unknown_prefixes(gains: Mapping[str, Any], dataset: Dataset) -> None:
    """Refuse gains keyed ``'<something>/<query_id>'`` where no subset of the suite has that name.

    The mirror of :func:`_refuse_bare_keys`: a typo'd subset name would silently drop that subset's gains
    from its part (one dataset's rows in the aggregate, no warning, and the gains' bounds unchecked). A key
    that is a labelled query of some part is a bare id and reads as one, whatever slashes it carries.

    Raises:
        DataError: Naming the stray keys and the subsets the suite has.
    """
    if not dataset.subsets:
        return
    prefixes = tuple(f"{part.name}/" for part in dataset.parts)
    labelled = {q for part in dataset.parts for q in set(part.qrels) | set(part.gains or {})}
    strays = sorted(key for key in gains if "/" in key and not key.startswith(prefixes) and key not in labelled)
    if strays:
        raise DataError(
            f"the gains are keyed '<{strays[0].split('/', 1)[0]}>/<query_id>' but no subset has that name "
            f"(e.g. {strays[0]!r}); the subsets are {sorted(prefixes)}",
            hint="key them '<subset>/<query_id>' with the exact subset names, or by bare query id",
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


def _has_cutoff(report: EvalReport, metric: MetricName, k: int) -> bool:
    """Whether the report computed ``(metric, k)`` for at least one system."""
    return any((row.metric, row.k) == (metric, k) for row in report.summary)


def _refuse_missing_cutoff(report: EvalReport, metric: MetricName, k: int) -> None:
    """Refuse a ``(metric, k)`` the report never computed, before a reader gets an empty table from it.

    ``value()`` refuses a missing summary row by name; this is the same refusal for the readers that would
    otherwise silently return nothing (:meth:`EvalReport.leaderboard`) or fail later with a message about
    shared queries (:func:`rcp_ndcg.eval.compare`, :func:`rcp_ndcg.eval.sensitivity`).

    Raises:
        DataError: Naming the metric, the cutoff and the cutoffs the report has.
    """
    if not _has_cutoff(report, metric, k):
        raise DataError(f"the report has no {metric}@{k} for any system; it has cutoffs {report.k}")


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
