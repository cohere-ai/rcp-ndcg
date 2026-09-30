"""The paper's scoring protocol: which candidates a query ranks, and how ties and ideals are set.

A reranker's output is scored per query from plain mappings:

* ``scores`` -- ``{doc_id: score}``, the system's scores;
* ``gains`` -- ``{doc_id: gain}``, the calibrated RCP gains ``g(theta)`` of the query's
  whole judged pool (unjudged candidates count 0);
* ``qrels`` -- ``{doc_id: grade}``, the human labels;
* ``candidates`` -- the judged pool in pool order (HF ``top_ranked``);
* ``excluded`` -- document ids the benchmark removes for the query (HF ``excluded``:
  NanoArguAna's own argument of the query, BRIGHT's ``excluded_ids``).

Every metric uses linear gains through :func:`rcp_ndcg_core.metric.ndcg`.  The ideal DCG of
RCP-nDCG and Count-nDCG sorts all gains of the query; qrel-nDCG's ideal DCG sorts all
positive qrels, including positives outside the scored candidates.  Excluded documents
leave the ranking and every ideal.  A :class:`Protocol` adds the suite rules,
:data:`PROTOCOLS` holds the presets, :func:`score_query` scores one query and
:func:`aggregate` averages the per-query values like the paper:

=========  ===========  ============================  ======================
preset     ties         candidates                    qrel-nDCG rounding
=========  ===========  ============================  ======================
nanobeir   doc_id_desc  judged pool only              5 decimals (BEIR)
bright     doc_id_desc  every scored document         5 decimals (BEIR)
vidore     group_mean   judged pool only              none
trecdl     input_order  judged pool only, pool order  none
mteb       group_mean   judged pool only              none
plain      group_mean   every scored document         none
=========  ===========  ============================  ======================

BRIGHT is not restricted to the judged pool because its TheoremQA reranker runs hold
candidates outside it, which the paper scores with gain 0.  Loading the released
HuggingFace data is ``rcp_ndcg.data.load_dataset``; this module never reads files.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rcp_ndcg_core.gain import qrel_gain
from rcp_ndcg_core.metric import TieRule, _validate_k, _validate_scores, ndcg


class Protocol(BaseModel):
    """How one suite's runs are scored.

    Attributes:
        name: Preset name.
        ties: The tie rule for equal scores.
        qrel_gain: ``"linear"`` (the grade; the paper) or ``"exponential"`` (``2**grade - 1``).
        restrict_to_candidates: Rank only the query's judged pool; other scored
            documents are dropped.
        drop_identical_ids: Also exclude the document whose id equals the query id
            (BEIR's ``ignore_identical_ids``), for data without an excluded list.
        round_digits: Round qrel-nDCG to this many decimals (BEIR uses 5); RCP-nDCG
            is never rounded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    ties: TieRule
    qrel_gain: Literal["linear", "exponential"] = "linear"
    restrict_to_candidates: bool = False
    drop_identical_ids: bool = False
    round_digits: int | None = Field(default=None, ge=0)


PROTOCOLS: dict[str, Protocol] = {
    "nanobeir": Protocol(name="nanobeir", ties="doc_id_desc", restrict_to_candidates=True, round_digits=5),
    "bright": Protocol(name="bright", ties="doc_id_desc", round_digits=5),
    "vidore": Protocol(name="vidore", ties="group_mean", restrict_to_candidates=True),
    "trecdl": Protocol(name="trecdl", ties="input_order", restrict_to_candidates=True),
    "mteb": Protocol(name="mteb", ties="group_mean", restrict_to_candidates=True),
    "plain": Protocol(name="plain", ties="group_mean"),
}
"""Named presets: the paper's four suites, mteb's ``ndcg_float_at_k`` rule, and ``plain``."""

MetricName = Literal["rcp_ndcg", "qrel_ndcg", "count_ndcg"]
"""The three nDCGs; they differ only in their gains (paper, appendix B)."""


def resolve_protocol(protocol: str | Protocol) -> Protocol:
    """The :class:`Protocol` a preset name (a key of :data:`PROTOCOLS`) stands for; a ``Protocol`` passes through.

    Raises:
        ValueError: an unknown preset name (the message lists the presets).
    """
    if isinstance(protocol, Protocol):
        return protocol
    try:
        return PROTOCOLS[protocol]
    except KeyError:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {sorted(PROTOCOLS)}") from None


def _excluded(protocol: Protocol, excluded: Iterable[str], query_id: str | None) -> set[str]:
    drop = set(excluded)
    if protocol.drop_identical_ids:
        if query_id is None:
            raise ValueError("drop_identical_ids needs the query id (query_id=...)")
        drop.add(query_id)
    return drop


def candidate_docs(
    protocol: str | Protocol,
    scored: Iterable[str],
    *,
    candidates: Sequence[str] | None = None,
    excluded: Collection[str] = (),
    query_id: str | None = None,
) -> list[str]:
    """The documents of one query that enter the ranking.

    Args:
        protocol: A :class:`Protocol` or a :data:`PROTOCOLS` name.
        scored: The ids the system scored.
        candidates: The judged pool in pool order; required when the protocol
            restricts to it or breaks ties by input order.
        excluded: Ids the benchmark removes for this query.
        query_id: The query's id (for ``drop_identical_ids``).

    Returns:
        The entering ids: in pool order when ``candidates`` is given, else in the
        order of ``scored``.
    """
    rules = resolve_protocol(protocol)
    drop = _excluded(rules, excluded, query_id)
    scored_ids = [doc_id for doc_id in scored if doc_id not in drop]
    if candidates is None:
        if rules.restrict_to_candidates or rules.ties == "input_order":
            raise ValueError(f"protocol {rules.name!r} needs the query's judged pool (candidates=...)")
        return scored_ids
    entering = set(scored_ids)
    in_pool = [doc_id for doc_id in candidates if doc_id in entering]
    if rules.restrict_to_candidates:
        return in_pool
    pooled = set(in_pool)
    return in_pool + [doc_id for doc_id in scored_ids if doc_id not in pooled]


def score_query(
    scores: Mapping[str, float],
    gains: Mapping[str, float],
    *,
    protocol: str | Protocol,
    k: int = 10,
    metric: MetricName = "rcp_ndcg",
    candidates: Sequence[str] | None = None,
    excluded: Collection[str] = (),
    query_id: str | None = None,
) -> float:
    """One query's nDCG@k under a protocol.

    Args:
        scores: ``{doc_id: score}`` of the system; higher is better; finite.
        gains: For ``"rcp_ndcg"`` and ``"count_ndcg"``, ``{doc_id: gain}`` over the query's judged pool (gains in
            ``[0, 1]``); the ideal DCG sorts all of them.  For ``"qrel_ndcg"``, ``{doc_id: grade}``: grades ``> 0``
            are positives, mapped by the protocol's ``qrel_gain``; the ideal DCG sorts all positives.
        protocol: A :class:`Protocol` or a :data:`PROTOCOLS` name.
        k: Cutoff.
        metric: Which nDCG ``gains`` belong to.  Only ``"qrel_ndcg"`` is rounded (``round_digits``).
        candidates: The judged pool in pool order (see :func:`candidate_docs`).
        excluded: Ids removed from the ranking and from the ideal.
        query_id: The query's id (for ``drop_identical_ids``).

    Returns:
        nDCG@k in ``[0, 1]``.  qrel-nDCG is NaN when the query has no positive grade, so it drops out of the means
        of :func:`aggregate`.
    """
    rules = resolve_protocol(protocol)
    _validate_k(k)
    _validate_scores(scores)
    drop = _excluded(rules, excluded, query_id)
    entering = candidate_docs(rules, scores, candidates=candidates, excluded=drop, query_id=query_id)
    ranked = {doc_id: scores[doc_id] for doc_id in entering}

    if metric == "qrel_ndcg":
        positives = {doc_id: grade for doc_id, grade in gains.items() if grade > 0 and doc_id not in drop}
        if not positives:
            return math.nan
        value = ndcg(ranked, {d: qrel_gain(g, rules.qrel_gain) for d, g in positives.items()}, k=k, ties=rules.ties)
        return value if rules.round_digits is None else round(value, rules.round_digits)
    if metric not in ("rcp_ndcg", "count_ndcg"):
        raise ValueError(f"unknown metric {metric!r}; expected 'rcp_ndcg', 'qrel_ndcg' or 'count_ndcg'")
    return ndcg(ranked, {doc_id: g for doc_id, g in gains.items() if doc_id not in drop}, k=k, ties=rules.ties)


def _nanmean(values: Iterable[float]) -> float:
    finite = [value for value in values if not math.isnan(value)]
    return math.fsum(finite) / len(finite) if finite else math.nan


def aggregate(per_dataset: Mapping[str, Mapping[str, float]]) -> float:
    """The paper's leaderboard number: the mean over queries per dataset, then the unweighted mean over datasets.

    Args:
        per_dataset: ``{dataset: {query_id: value}}`` of one metric.

    Returns:
        The mean.  NaN values (queries without a positive qrel) are skipped; a dataset whose values are all NaN is
        skipped too; NaN when nothing is left.
    """
    return _nanmean(_nanmean(values.values()) for values in per_dataset.values() if values)


__all__ = [
    "PROTOCOLS",
    "MetricName",
    "Protocol",
    "aggregate",
    "candidate_docs",
    "resolve_protocol",
    "score_query",
]
