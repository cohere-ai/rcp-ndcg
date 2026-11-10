"""nDCG with float gains and explicit tie rules (pure Python, no torch).

* :func:`ndcg` -- the one nDCG@k of this project. RCP-, qrel- and Count-nDCG differ only in the gains
  (:mod:`rcp_core.gain`).
* :func:`dcg`, :func:`ideal_dcg`, :func:`discount` -- its parts, for callers that need a DCG or the position
  discount themselves.
* :data:`TieRule` -- how equal scores are ordered; :func:`rank_by_score` applies the two rules that give an order.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, get_args

from rcp_core._util import descending_score_order

TieRule = Literal["doc_id_desc", "group_mean", "input_order"]
"""How documents with equal scores are ordered when a score mapping is scored.

* ``"group_mean"`` -- every document of an equal-score class is credited the class's mean gain: the expected nDCG
  over all orders of the tie. It depends on the scores alone (mteb's ``ndcg_float_at_k``; the default, and the rule
  of every :data:`~rcp_core.protocol.PROTOCOLS` preset and of the paper's tables).
* ``"doc_id_desc"`` -- ties broken by document id, descending (trec_eval / pytrec_eval; arXiv v1's NanoBEIR and
  BRIGHT tables).
* ``"input_order"`` -- ties keep the order in which the scores are given (arXiv v1's TREC-DL tables pass them in
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


def _validate_gains(values: Iterable[float], *, non_negative: bool = False) -> None:
    """Refuse a gain that is not a finite number, and -- with ``non_negative`` -- one below 0.

    The default is the finite-only contract of a caller that validates raw labels: :func:`protocol.score_query`
    checks the qrels grades themselves, and a negative label is not a positive (it leaves the ranking and the
    ideal exactly as grade 0 does), so refusing it here would abort an evaluation on a dataset that carries
    one.

    :func:`ndcg`, :func:`dcg` and :func:`ideal_dcg` pass ``non_negative=True``: a gain below 0 leaves their
    documented ``[0, 1]``, because the ideal DCG is built from the same values and can be smaller in
    magnitude than the DCG it normalises (``ndcg(["a", "b"], {"a": -1.0, "b": 1.0}, k=2)`` was -1.0).

    A non-finite gain is refused either way: it flows through the sums as a NaN. A non-numeric value is
    refused either way too: the contract is the error, never a silent 0.
    """
    materialised = list(values)
    non_numeric = [value for value in materialised if not isinstance(value, numbers.Real)]
    if non_numeric:
        raise ValueError(
            f"gains must be finite numbers; got {len(non_numeric)} non-numeric gain(s), first {non_numeric[:5]}"
        )
    bad = [value for value in materialised if not math.isfinite(value)]
    if bad:
        raise ValueError(f"gains must be finite; got {len(bad)} non-finite gain(s), first {bad[:5]}")
    negative = [value for value in materialised if value < 0.0] if non_negative else []
    if negative:
        raise ValueError(
            f"gains must be non-negative (the gain is in [0, 1], and a negative one makes the ideal DCG "
            f"smaller in magnitude than the DCG it normalises); got {len(negative)} negative gain(s), "
            f"first {negative[:5]}"
        )


def discount(rank: int) -> float:
    """The nDCG position discount ``1 / log2(rank + 1)``; ``rank`` counts from 1.

    Raises:
        ValueError: ``rank`` is not finite, or is below 1.
    """
    if not math.isfinite(rank):
        raise ValueError(f"rank must be finite, got {rank!r}")
    if rank < 1:
        raise ValueError(f"rank counts from 1, got {rank}")
    return 1.0 / math.log2(rank + 1)


def dcg(gains_in_rank_order: Sequence[float], k: int) -> float:
    """``DCG@k = sum_{r=1..k} gain(r) / log2(r + 1)`` of gains already in rank order (best first).

    Raises:
        ValueError: ``k`` is not positive, or a gain entering the sum is not a finite number of at least 0.
    """
    _validate_k(k)
    entering = list(gains_in_rank_order[:k])
    _validate_gains(entering, non_negative=True)
    return sum(gain / math.log2(rank + 1) for rank, gain in enumerate(entering, start=1))


def ideal_dcg(gains: Iterable[float], k: int) -> float:
    """The DCG@k of ``gains`` sorted best first: the normaliser of :func:`ndcg`.

    Raises:
        ValueError: ``k`` is not positive, or a gain is not a finite number of at least 0 (checked before the
            sort: a NaN has no sort position).
    """
    values = list(gains)
    _validate_gains(values, non_negative=True)
    return dcg(sorted(values, reverse=True)[:k], k)


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
        # The rule has one home: ``descending_score_order``, which ``RankingExample``'s auto-sort calls too.
        # Two implementations of "descending score, ties keep input order" can disagree on a tie.
        doc_ids = list(scores)
        return [doc_ids[index] for index in descending_score_order(list(scores.values()))]
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
    doc_ids = list(scores)
    groups: list[list[str]] = []
    for index in descending_score_order(list(scores.values())):  # the input_order rule, one home
        doc_id = doc_ids[index]
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
        gains: ``{doc_id: gain}``, gains in ``[0, 1]`` (RCP, Count) or non-negative grades (qrels). Ranked documents
            missing from ``gains`` count 0.
        k: Cutoff.
        ties: The :data:`TieRule` for a score mapping (ignored for a ranking).
        ideal: The gains the ideal DCG is built from; defaults to ``gains.values()``. Pass the query's whole labelled
            set (its calibrated pool, or all positive qrels), not only the ranked candidates.

    Returns:
        nDCG@k in ``[0, 1]``; 0.0 when nothing is ranked or the ideal DCG is 0.

    Raises:
        ValueError: a score, a gain or an ``ideal`` gain is not a finite number, or a gain is negative (a
            non-finite gain would flow through the sums and return a NaN with no error; a negative one returns a
            value outside ``[0, 1]``).
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
    _validate_gains(gains.values(), non_negative=True)
    ideal_values = list(ideal) if ideal is not None else list(gains.values())
    _validate_gains(ideal_values, non_negative=True)
    normaliser = ideal_dcg(ideal_values, k)
    if not ranked_gains or normaliser == 0.0:
        return 0.0
    return dcg(ranked_gains, k) / normaliser


__all__ = ["TieRule", "dcg", "discount", "ideal_dcg", "ndcg", "rank_by_score", "tie_groups"]
