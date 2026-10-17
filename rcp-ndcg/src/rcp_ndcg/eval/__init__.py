"""Evaluation: score rankings, compare systems, explain a query.

* :func:`evaluate` -- RCP-nDCG, qrel-nDCG and Count-nDCG of :class:`~rcp_ndcg.data.Rankings` under a scoring
  protocol, as an :class:`EvalReport` (per query, per dataset, summary with bootstrap intervals).
* :func:`compare` -- system pairs: delta (B minus A), paired t-test, query-clustered bootstrap interval, and the
  queries where RCP-nDCG and qrel-nDCG disagree, as a :class:`Comparison`.
* :func:`sensitivity` -- the share of system pairs a metric separates.
* :func:`explain` -- one query side by side: each system's top k with gains, thetas and per-criterion
  probabilities, and the selection/ordering split of the gaps, as a :class:`QueryExplanation`.

All metric arithmetic is :mod:`rcp_ndcg_core`'s. The MTEB tasks of the public suites are in
:mod:`rcp_ndcg.eval.mteb` (extra ``mteb``).
"""

from rcp_ndcg.eval.compare import Comparison, PairComparison, SignFlip, compare, sensitivity
from rcp_ndcg.eval.evaluate import (
    METRICS,
    DatasetValue,
    EvalReport,
    QueryValue,
    ReportInputs,
    ReportWarning,
    SummaryValue,
    evaluate,
)
from rcp_ndcg.eval.explain import (
    CriterionContribution,
    CriterionParams,
    QueryExplanation,
    RankedDocument,
    ScoreDelta,
    SystemExplanation,
    explain,
    per_criterion,
)

__all__ = [
    "METRICS",
    "Comparison",
    "CriterionContribution",
    "CriterionParams",
    "DatasetValue",
    "EvalReport",
    "PairComparison",
    "QueryExplanation",
    "QueryValue",
    "RankedDocument",
    "ReportInputs",
    "ReportWarning",
    "ScoreDelta",
    "SignFlip",
    "SummaryValue",
    "SystemExplanation",
    "compare",
    "evaluate",
    "explain",
    "per_criterion",
    "sensitivity",
]
