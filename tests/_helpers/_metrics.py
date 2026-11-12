"""The metric plugins the registry tests install through the ``rcp.metrics`` entry-point group.

They are the seam's test doubles: one metric that takes a cutoff (MAP), one that ignores it (MRR), one whose
value is undefined (NaN), one that breaks the return contract (a string), and one that takes a shipped
metric's name (the duplicate-name policy refuses it). Their names are plainly not metrics -- ``toy_metric``,
``toy_metric_no_cutoff`` -- because a name a shipped metric holds is refused, so a double cannot borrow one.
``install_metrics`` points the registry's ``entry_points`` at the entries a test passes, the way
``packages/rcp/tests/results/_sinks.py`` does for sinks.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from importlib.metadata import EntryPoint

import pytest
from rcp_core.metric import rank_by_score
from rcp_core.protocol import Protocol, candidate_docs

from rcp.eval.metrics import Metric


def _ranking(
    query: str,
    ranking: Mapping[str, float],
    *,
    protocol: Protocol,
    candidates: Sequence[str] | None,
    excluded: Collection[str],
) -> list[str]:
    """The protocol's entering documents, best first.

    ``candidate_docs`` is the one rule for *which* documents enter (pool restriction, exclusions); a
    rank-based metric then orders them by score, with ``doc_id_desc`` as its deterministic tie-break
    (``group_mean`` is nDCG's expectation rule, not an order).
    """
    entering = candidate_docs(protocol, ranking, candidates=candidates, excluded=excluded, query_id=query)
    return rank_by_score({doc_id: ranking[doc_id] for doc_id in entering}, ties="doc_id_desc")


class MeanAveragePrecision(Metric):
    """MAP@k over the protocol's ranking, binary relevance from the qrel grades."""

    name = "toy_metric"
    k_required = True

    def score(
        self,
        query: str,
        ranking: Mapping[str, float],
        labels: Mapping[str, float],
        *,
        protocol: Protocol,
        k: int,
        candidates: Sequence[str] | None = None,
        excluded: Collection[str] = (),
    ) -> float | None:
        ranked = _ranking(query, ranking, protocol=protocol, candidates=candidates, excluded=excluded)
        relevant = {doc_id for doc_id, grade in labels.items() if grade > 0}
        if not relevant:
            return None
        hits, total = 0, 0.0
        for rank, doc_id in enumerate(ranked[:k], start=1):
            if doc_id in relevant:
                hits += 1
                total += hits / rank
        return total / min(len(relevant), k)


class MeanReciprocalRank(Metric):
    """MRR: the reciprocal rank of the first relevant document; it takes no cutoff."""

    name = "toy_metric_no_cutoff"
    k_required = False

    def score(
        self,
        query: str,
        ranking: Mapping[str, float],
        labels: Mapping[str, float],
        *,
        protocol: Protocol,
        k: int,
        candidates: Sequence[str] | None = None,
        excluded: Collection[str] = (),
    ) -> float | None:
        del k  # the metric is cutoff-free; the registry scores it once, at k=0
        ranked = _ranking(query, ranking, protocol=protocol, candidates=candidates, excluded=excluded)
        for rank, doc_id in enumerate(ranked, start=1):
            if labels.get(doc_id, 0.0) > 0:
                return 1.0 / rank
        return 0.0


class NanValue(MeanAveragePrecision):
    """A plugin whose value is undefined: the contract reads ``nan`` as ``None``, not as a value to average."""

    name = "nan_value"

    def score(
        self,
        query: str,
        ranking: Mapping[str, float],
        labels: Mapping[str, float],
        *,
        protocol: Protocol,
        k: int,
        candidates: Sequence[str] | None = None,
        excluded: Collection[str] = (),
    ) -> float | None:
        del query, ranking, labels, protocol, k, candidates, excluded
        return math.nan


class NotANumber(MeanAveragePrecision):
    """A plugin that breaks the return contract: the seam names it instead of crashing on ``math.isnan``."""

    name = "not_a_number"

    def score(
        self,
        query: str,
        ranking: Mapping[str, float],
        labels: Mapping[str, float],
        *,
        protocol: Protocol,
        k: int,
        candidates: Sequence[str] | None = None,
        excluded: Collection[str] = (),
    ) -> float | None:
        del query, ranking, labels, protocol, k, candidates, excluded
        return "high"  # type: ignore[return-value]  # the point: a plugin that breaks the contract


class CollidingRcp(Metric):
    """A plugin that publishes a shipped metric's name: the one duplicate-name policy must refuse it."""

    name = "rcp_ndcg"
    k_required = True

    def score(
        self,
        query: str,
        ranking: Mapping[str, float],
        labels: Mapping[str, float],
        *,
        protocol: Protocol,
        k: int,
        candidates: Sequence[str] | None = None,
        excluded: Collection[str] = (),
    ) -> float | None:
        del query, ranking, labels, protocol, k, candidates, excluded
        return 0.0


def metric_entry(name: str, value: str) -> EntryPoint:
    """One entry point of the metric group (the group is the registry module's constant)."""
    from rcp.eval.metrics import METRICS_GROUP

    return EntryPoint(name=name, value=value, group=METRICS_GROUP)


def install_metrics(monkeypatch: pytest.MonkeyPatch, *entries: EntryPoint) -> list[EntryPoint]:
    """Point ``rcp.eval.metrics.entry_points`` at *entries* (the plugins under test)."""
    import rcp.eval.metrics as module

    installed = list(entries)
    monkeypatch.setattr(
        module,
        "entry_points",
        lambda group: installed if group == module.METRICS_GROUP else [],
    )
    return installed
