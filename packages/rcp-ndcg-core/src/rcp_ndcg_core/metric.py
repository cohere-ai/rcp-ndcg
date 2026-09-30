"""nDCG with float gains and explicit tie rules (pure Python, no torch).

* :func:`ndcg` -- the one nDCG@k of this project. RCP-, qrel- and Count-nDCG differ only in the gains
  (:mod:`rcp_ndcg_core.gain`).
* :func:`dcg`, :func:`ideal_dcg`, :func:`discount` -- its parts, for callers that need a DCG or the position
  discount themselves.
* :data:`TieRule` -- how equal scores are ordered; :func:`rank_by_score` applies the two rules that give an order.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, get_args

TieRule = Literal["doc_id_desc", "group_mean", "input_order"]
"""How documents with equal scores are ordered when a score mapping is scored.

* ``"group_mean"`` -- every document of an equal-score class is credited the class's mean gain: the expected nDCG
  over all orders of the tie. It depends on the scores alone (mteb's ``ndcg_float_at_k``; the paper's ViDoRe v3
  table).
* ``"doc_id_desc"`` -- ties broken by document id, descending (trec_eval / pytrec_eval; the paper's NanoBEIR and
  BRIGHT tables).
* ``"input_order"`` -- ties keep the order in which the scores are given (the paper's TREC-DL tables pass them in
  judge-pool order). A relevance-ordered input leaks relevance into tied scores, so use it only with an order you
  constructed.
"""

_TIE_RULES: tuple[str, ...] = get_args(TieRule)


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be greater than 0, got {k}")


def _validate_ranking(ranking: Sequence[str]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for doc_id in ranking:
        if doc_id in seen and doc_id not in duplicates:
            duplicates.append(doc_id)
        seen.add(doc_id)
    if duplicates:
        raise ValueError(f"ranking contains duplicate document IDs: {duplicates}")


def _validate_scores(scores: Mapping[str, float]) -> None:
    bad = [doc_id for doc_id, score in scores.items() if not math.isfinite(score)]
    if bad:
        raise ValueError(f"scores must be finite; got non-finite scores for {bad[:5]}")


def discount(rank: int) -> float:
    """The nDCG position discount ``1 / log2(rank + 1)``; ``rank`` counts from 1."""
    return 1.0 / math.log2(rank + 1)


def dcg(gains_in_rank_order: Sequence[float], k: int) -> float:
    """``DCG@k = sum_{r=1..k} gain(r) / log2(r + 1)`` of gains already in rank order (best first)."""
    _validate_k(k)
    return sum(gain * discount(rank) for rank, gain in enumerate(gains_in_rank_order[:k], start=1))


def ideal_dcg(gains: Iterable[float], k: int) -> float:
    """The DCG@k of ``gains`` sorted best first: the normaliser of :func:`ndcg`."""
    return dcg(sorted(gains, reverse=True)[:k], k)


def rank_by_score(
    scores: Mapping[str, float],
    *,
    ties: Literal["doc_id_desc", "input_order"] = "doc_id_desc",
) -> list[str]:
    """Order document ids by decreasing score with a deterministic tie-break.

    Args:
        scores: ``{doc_id: score}``; higher is better. Scores must be finite.
        ties: ``"doc_id_desc"`` (equal scores ordered by document id, descending) or ``"input_order"`` (equal scores
            keep their order in ``scores``). ``"group_mean"`` has no order; it is a rule of :func:`ndcg`.

    Returns:
        The document ids, best first.
    """
    _validate_scores(scores)
    if ties == "doc_id_desc":
        return sorted(scores, key=lambda doc_id: (scores[doc_id], doc_id), reverse=True)
    if ties == "input_order":
        return sorted(scores, key=lambda doc_id: -scores[doc_id])  # sorted() is stable
    raise ValueError(f"rank_by_score orders by 'doc_id_desc' or 'input_order', got ties={ties!r}")


def tie_groups(scores: Mapping[str, float]) -> list[list[str]]:
    """The documents grouped by equal score, the best score first; within a group, in the order of ``scores``.

    The groups are the equal-score classes that the ``"group_mean"`` tie rule of :func:`ndcg` credits with their
    mean gain: a custom metric can apply the same rule to the same classes.

    Args:
        scores: ``{doc_id: score}``, higher is better, finite.

    Returns:
        ``[[doc_id, ...], ...]``: one list per distinct score, from the highest score to the lowest.

    Raises:
        ValueError: a score is not a finite number.
    """
    _validate_scores(scores)
    groups: list[list[str]] = []
    for doc_id in sorted(scores, key=lambda doc_id: -scores[doc_id]):  # sorted() is stable
        if groups and scores[groups[-1][0]] == scores[doc_id]:
            groups[-1].append(doc_id)
        else:
            groups.append([doc_id])
    return groups


def _group_mean_gains(scores: Mapping[str, float], gains: Mapping[str, float]) -> list[float]:
    """Gains in rank order, each document credited the mean gain of its equal-score class."""
    ranked: list[float] = []
    for group in tie_groups(scores):
        mean = sum(gains.get(doc_id, 0.0) for doc_id in group) / len(group)
        ranked.extend([mean] * len(group))
    return ranked


def ndcg(
    scores: Mapping[str, float] | Sequence[str],
    gains: Mapping[str, float],
    *,
    k: int = 10,
    ties: TieRule = "group_mean",
    ideal: Iterable[float] | None = None,
) -> float:
    """nDCG@k with linear gains.

    RCP-nDCG, qrel-nDCG and Count-nDCG differ only in ``gains``: the calibrated gain ``g(theta)``, the human grade,
    or the share of passed criteria.

    Args:
        scores: The system's output for one query: a mapping ``{doc_id: score}`` (higher is better, finite), ranked
            under ``ties``; or a ranking ``[doc_id, ...]`` (best first, no duplicates), taken as ordered.
        gains: ``{doc_id: gain}``, gains in ``[0, 1]`` (RCP, Count) or grades (qrels). Ranked documents missing from
            ``gains`` count 0.
        k: Cutoff.
        ties: The :data:`TieRule` for a score mapping (ignored for a ranking).
        ideal: The gains the ideal DCG is built from; defaults to ``gains.values()``. Pass the query's whole labelled
            set (its calibrated pool, or all positive qrels), not only the ranked candidates.

    Returns:
        nDCG@k in ``[0, 1]``; 0.0 when nothing is ranked or the ideal DCG is 0.
    """
    _validate_k(k)
    if isinstance(scores, Mapping):
        if ties not in _TIE_RULES:
            raise ValueError(f"Unknown tie rule {ties!r}; expected one of {_TIE_RULES}")
        _validate_scores(scores)
        if ties == "group_mean":
            ranked_gains = _group_mean_gains(scores, gains)[:k]
        else:
            ranked_gains = [gains.get(doc_id, 0.0) for doc_id in rank_by_score(scores, ties=ties)[:k]]
    else:
        _validate_ranking(scores)
        ranked_gains = [gains.get(doc_id, 0.0) for doc_id in scores[:k]]
    normaliser = ideal_dcg(gains.values() if ideal is None else ideal, k)
    if not ranked_gains or normaliser == 0.0:
        return 0.0
    return dcg(ranked_gains, k) / normaliser


__all__ = ["TieRule", "dcg", "discount", "ideal_dcg", "ndcg", "rank_by_score", "tie_groups"]
