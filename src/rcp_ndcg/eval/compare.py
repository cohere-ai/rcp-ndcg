"""``compare`` and ``sensitivity``: is system B better than system A, and how often can a metric tell?

One sign convention everywhere: a delta is **B minus A**, where A is the baseline (or the earlier system of a
pair). A pair is compared on the queries both systems have a value for:

* the delta is the difference of the protocol aggregates (the mean per dataset, then over datasets);
* the paired t-test runs on the per-query differences, pooled over datasets (the paper's test);
* the interval is a percentile bootstrap of the delta that resamples queries within each dataset, the same draw
  for both systems (query-clustered, fixed seed).

When the report holds both RCP-nDCG and qrel-nDCG, each pair also lists its sign flips: the queries where the two
metrics prefer different systems.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from rcp_ndcg_core.protocol import MetricName

from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.eval.evaluate import EvalReport, _aggregate_values, _one_k, _refuse_missing_cutoff, bootstrap_interval

if TYPE_CHECKING:
    import pandas as pd


class SignFlip(BaseModel):
    """A query where RCP-nDCG and qrel-nDCG prefer different systems of a pair (deltas are B minus A)."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    query_id: str
    delta_rcp_ndcg: float
    delta_qrel_ndcg: float


class PairComparison(BaseModel):
    """System B against system A on one metric.

    Attributes:
        system_a: The baseline (A).
        system_b: The compared system (B).
        value_a: A's aggregate over the paired queries.
        value_b: B's aggregate over the paired queries.
        delta: ``value_b - value_a``.
        ci_low: Lower bootstrap bound of ``delta``.
        ci_high: Upper bootstrap bound of ``delta``.
        t: The paired t statistic (``None`` when it is undefined, e.g. the differences have no variance).
        p_value: Its two-sided p-value (``None`` when undefined; such a pair is not significant).
        significant: ``p_value < alpha``.
        num_queries: The number of paired queries.
        sign_flips: Queries where RCP-nDCG and qrel-nDCG disagree about the sign of the delta.
    """

    model_config = ConfigDict(frozen=True)

    system_a: str
    system_b: str
    value_a: float
    value_b: float
    delta: float
    ci_low: float | None
    ci_high: float | None
    t: float | None
    p_value: float | None
    significant: bool
    num_queries: int
    sign_flips: list[SignFlip] = []


class Comparison(BaseModel):
    """The result of :func:`compare`: every compared pair of one metric at one cutoff."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    schema_name: Literal["rcp-ndcg.comparison.v1"] = Field(default="rcp-ndcg.comparison.v1", alias="schema")
    metric: MetricName
    k: int
    alpha: float
    baseline: str | None
    bootstrap: int
    seed: int
    pairs: list[PairComparison]

    def to_json(self, **kwargs: Any) -> str:
        """The comparison as JSON."""
        return self.model_dump_json(by_alias=True, **kwargs)

    def to_pandas(self) -> pd.DataFrame:
        """One row per pair; the sign flips as their count, ``num_sign_flips``."""
        import pandas as pd

        fields = [f for f in PairComparison.model_fields if f != "sign_flips"]
        rows = [{**pair.model_dump(include=set(fields)), "num_sign_flips": len(pair.sign_flips)} for pair in self.pairs]
        return pd.DataFrame(rows, columns=[*fields, "num_sign_flips"])

    def __repr__(self) -> str:
        significant = sum(pair.significant for pair in self.pairs)
        return (
            f"Comparison({self.metric}@{self.k}, {len(self.pairs)} pairs, {significant} significant at "
            f"alpha={self.alpha}, baseline={self.baseline})"
        )

    __str__ = __repr__


def compare(
    report: EvalReport,
    *,
    baseline: str | None = None,
    metric: MetricName = "rcp_ndcg",
    k: int | None = None,
    alpha: float = 0.05,
    bootstrap: int = 10_000,
    seed: int = 0,
    systems: Sequence[str] | None = None,
) -> Comparison:
    """Compare the systems of a report pairwise: delta (B minus A), the paired t-test (the paper's), bootstrap interval.

    Args:
        report: An :class:`~rcp_ndcg.eval.EvalReport` with at least two systems.
        baseline: Compare every other system against this one (A); ``None`` compares all pairs, A being the
            earlier system of the report.
        metric: The metric to compare.
        k: The cutoff (may be omitted when the report has one).
        alpha: The significance level and one minus the interval's coverage.
        bootstrap: Bootstrap resamples for the interval (0: none).
        seed: The bootstrap seed.
        systems: Compare only these systems of the report, in the report's order (default: all). A run's report
            holds the reference systems ``candidates`` (the pool order) and ``judge`` (the judge's own abilities),
            which the command line leaves out unless asked for.

    Returns:
        The :class:`Comparison`.

    Raises:
        ConfigError: The baseline or a named system is not in the report (or not among ``systems``).
        DataError: Fewer than two systems to compare, or a ``(metric, k)`` the report never computed (the
            error names the cutoffs it has).
    """
    k = _one_k(report, k)
    values = _values(report, metric, k)
    compared = report.systems
    if systems is not None:
        unknown = sorted(set(systems) - set(compared))
        if unknown:
            raise ConfigError(f"systems {unknown} are not in the report; systems: {compared}")
        compared = [s for s in compared if s in set(systems)]
    if baseline is not None and baseline not in compared:
        raise ConfigError(f"baseline {baseline!r} is not a compared system; systems: {compared}")
    if len(compared) < 2:
        raise DataError(f"a comparison needs at least two systems; compared: {compared}")
    pairs = (
        [(baseline, s) for s in compared if s != baseline]
        if baseline is not None
        else list(itertools.combinations(compared, 2))
    )
    other = "qrel_ndcg" if metric == "rcp_ndcg" else "rcp_ndcg" if metric == "qrel_ndcg" else None
    other_values = _values(report, other, k) if other in report.metrics else None
    return Comparison(
        metric=metric,
        k=k,
        alpha=alpha,
        baseline=baseline,
        bootstrap=bootstrap,
        seed=seed,
        pairs=[
            _compare_pair(values, a, b, alpha=alpha, bootstrap=bootstrap, seed=seed, flips=other_values, metric=metric)
            for a, b in pairs
        ],
    )


def sensitivity(
    report: EvalReport, *, metric: MetricName = "rcp_ndcg", k: int | None = None, alpha: float = 0.05
) -> float:
    """How often a metric separates systems, the paper's sensitivity (appendix B).

    Per dataset, the share of all system pairs whose paired t-test over their shared queries has ``p < alpha`` (a
    pair with fewer than two shared queries, or without variance in its differences, is not separated); then the
    unweighted mean of these shares over the datasets. No multiplicity correction.

    Args:
        report: An :class:`~rcp_ndcg.eval.EvalReport` with at least two systems.
        metric: The metric.
        k: The cutoff (may be omitted when the report has one).
        alpha: The significance level.

    Returns:
        The mean over datasets of the separated share of (dataset, system pair) comparisons, in ``[0, 1]``.

    Raises:
        DataError: fewer than two systems, no dataset where a pair shares two queries, or a ``(metric, k)``
            the report never computed (the error names the cutoffs it has).
    """
    k = _one_k(report, k)
    values = _values(report, metric, k)
    pairs = list(itertools.combinations(report.systems, 2))
    if not pairs:
        raise DataError("sensitivity needs at least two systems")
    datasets = sorted({d for per_system in values.values() for d in per_system})
    shares, tested = [], False
    for dataset in datasets:
        separated = 0
        for a, b in pairs:
            x, y = _paired(values.get(a, {}).get(dataset, {}), values.get(b, {}).get(dataset, {}))
            tested |= len(x) >= 2
            _, p = _paired_t(y - x)
            separated += p is not None and p < alpha
        shares.append(separated / len(pairs))
    if not tested:
        raise DataError("no dataset has two or more queries scored by both systems of a pair")
    return math.fsum(shares) / len(shares)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

Values = dict[str, dict[str, dict[str, float]]]
"""``{system: {dataset: {query_id: value}}}`` (defined values only)."""


def _values(report: EvalReport, metric: str, k: int) -> Values:
    if metric not in report.metrics:
        raise DataError(f"the report has no {metric}; it has {report.metrics}")
    _refuse_missing_cutoff(report, metric, k)  # without it, a wrong k read as 'share no scored query' below
    out: Values = {}
    for row in report.per_query:
        if row.metric == metric and row.k == k and row.value is not None:
            out.setdefault(row.system, {}).setdefault(row.dataset, {})[row.query_id] = row.value
    return out


def _paired(a: Mapping[str, float], b: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    common = [q for q in a if q in b]
    return np.array([a[q] for q in common]), np.array([b[q] for q in common])


def _paired_t(differences: np.ndarray) -> tuple[float | None, float | None]:
    """The paired t statistic and two-sided p-value of B - A differences (scipy's); ``None`` where undefined."""
    import warnings

    from scipy.stats import ttest_1samp

    if len(differences) < 2:
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # no variance: scipy returns NaN, reported as None
        # A TtestResult is a (statistic, pvalue) tuple; scipy's stubs leave its fields untyped.
        statistic, pvalue = cast("tuple[float, float]", ttest_1samp(differences, 0.0))
    t, p = float(statistic), float(pvalue)
    return (None, None) if math.isnan(p) else (t, p)


def _compare_pair(
    values: Values,
    a: str,
    b: str,
    *,
    alpha: float,
    bootstrap: int,
    seed: int,
    flips: Values | None,
    metric: str,
) -> PairComparison:
    per_dataset = {
        dataset: _paired(values.get(a, {}).get(dataset, {}), values.get(b, {}).get(dataset, {}))
        for dataset in sorted(set(values.get(a, {})) | set(values.get(b, {})))
    }
    per_dataset = {d: xy for d, xy in per_dataset.items() if len(xy[0])}
    if not per_dataset:
        raise DataError(f"systems {a!r} and {b!r} share no scored query")
    value_a = _aggregate_values({d: x for d, (x, _) in per_dataset.items()})
    value_b = _aggregate_values({d: y for d, (_, y) in per_dataset.items()})
    t, p = _paired_t(np.concatenate([y - x for x, y in per_dataset.values()]))
    differences = {d: y - x for d, (x, y) in per_dataset.items()}
    low, high = bootstrap_interval(differences, resamples=bootstrap, seed=seed, alpha=alpha)
    return PairComparison(
        system_a=a,
        system_b=b,
        value_a=value_a,
        value_b=value_b,
        delta=value_b - value_a,
        ci_low=low,
        ci_high=high,
        t=t,
        p_value=p,
        significant=p is not None and p < alpha,
        num_queries=sum(len(x) for x, _ in per_dataset.values()),
        sign_flips=_sign_flips(values, flips, a, b, metric) if flips is not None else [],
    )


def _sign_flips(values: Values, other: Values, a: str, b: str, metric: str) -> list[SignFlip]:
    flips = []
    for dataset, queries in values.get(a, {}).items():
        for query_id, value_a in queries.items():
            try:
                delta = values[b][dataset][query_id] - value_a
                delta_other = other[b][dataset][query_id] - other[a][dataset][query_id]
            except KeyError:
                continue
            if delta * delta_other < 0:
                rcp, qrel = (delta, delta_other) if metric == "rcp_ndcg" else (delta_other, delta)
                flips.append(SignFlip(dataset=dataset, query_id=query_id, delta_rcp_ndcg=rcp, delta_qrel_ndcg=qrel))
    return sorted(flips, key=lambda f: -abs(f.delta_rcp_ndcg - f.delta_qrel_ndcg))


__all__ = ["Comparison", "PairComparison", "SignFlip", "compare", "sensitivity"]
