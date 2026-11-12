"""nDCG with float gains and explicit tie rules (pure Python, no torch).

* :func:`ndcg` -- the one nDCG@k of this project. RCP-, qrel- and Count-nDCG differ only in the gains
  (:mod:`rcp_core.gain`).
* :func:`dcg`, :func:`ideal_dcg`, :func:`discount` -- its parts, for callers that need a DCG or the position
  discount themselves.
* :data:`TieRule` -- how equal scores are ordered; :func:`rank_by_score` applies the two rules that give an order.
* the corpus metrics -- one value per ``(qrels, run)`` corpus. :func:`ndcg_float` and :func:`per_query_ndcg` score
  continuous qrels with the ``"group_mean"`` credit; the binary-relevance metrics read a grade above 0 as relevant
  and form a ranking (:func:`map`, :func:`recall`, :func:`precision`, :func:`success`, :func:`mrr`,
  :func:`recall_cap`, :func:`hole`); :func:`paired_accuracy` and :func:`coverage` form no ranking at all.
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


# ---------------------------------------------------------------------------------------------------------------
# Corpus metrics: the qrels and the run of a whole corpus, one value per metric and cutoff.
#
# Every one of them takes ``qrels = {query_id: {doc_id: grade}}`` and ``results = {query_id: {doc_id: score}}`` and
# names its own mean denominator in its docstring. Every one of them refuses, with a message, rather than returning a
# number for an undefined input: a non-finite or negative grade, a non-finite score, a duplicate document id (a
# mapping cannot carry one) and an empty denominator are errors, never 0.0.
# ---------------------------------------------------------------------------------------------------------------

_PerQuery = Mapping[str, Mapping[str, float]]
"""``{query_id: {doc_id: value}}``: a run's scores, or the qrels' grades, for a whole corpus."""


def _require_finite_number(value: float, *, what: str) -> None:
    """Refuse a value that is not a real number, or not a finite one, naming what it was meant to be."""
    if not isinstance(value, numbers.Real):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{what} must be finite, got {value!r}")


def _validate_qrels(qrels: _PerQuery) -> None:
    """Refuse a grade that is not a finite number of at least 0, naming the query and the document.

    The grade is a gain (:func:`ndcg_float`) or a relevance flag (the binary metrics), and every metric refuses
    both faults: a NaN grade flows through the sums and returns a NaN, and a negative one is not a relevance a
    binary metric can read (`protocol.score_query` calls it not positive) while as a gain it leaves ``[0, 1]``.
    """
    for query_id, grades in qrels.items():
        for doc_id, grade in grades.items():
            _require_finite_number(grade, what=f"the qrel grade for query {query_id!r}, document {doc_id!r}")
            if grade < 0:
                raise ValueError(
                    f"the qrel grade must be at least 0, got {grade!r} for query {query_id!r}, document {doc_id!r}"
                )


def _validate_run(results: _PerQuery) -> None:
    """Refuse a score that is not a finite number, naming the query and the document.

    A NaN score compares False with everything, so a sort keeps the input order instead -- for a judged pool,
    that is the relevance order the tie rules exist to keep out of the metric.
    """
    for query_id, scores in results.items():
        for doc_id, score in scores.items():
            _require_finite_number(score, what=f"the score for query {query_id!r}, document {doc_id!r}")


def _positive_docs(grades: Mapping[str, float]) -> list[str]:
    """One query's relevant documents -- those with a grade above 0 -- in the qrels' order."""
    return [doc_id for doc_id, grade in grades.items() if grade > 0]


def _evaluated_queries(qrels: _PerQuery, results: _PerQuery, *, metric: str) -> list[str]:
    """The queries the ``evaluated`` denominator averages over, in the qrels' order; refuse when there is none.

    The set is ``qrels`` intersected with ``results`` -- one set and one order, so two metrics that share it sum
    the same values in the same order. A corpus that scores nothing has no mean: 0.0 would report an undefined
    value as the measurement "nothing ranked correctly".
    """
    evaluated = [query_id for query_id in qrels if query_id in results]
    if not evaluated:
        raise ValueError(
            f"{metric} has no query to average over: it is a mean over the queries in both the qrels and the run, "
            f"and the qrels name {len(qrels)} queries while the run names {len(results)}. 0.0 would report an "
            f"undefined mean as a measurement"
        )
    return evaluated


def _scored_queries(qrels: _PerQuery, results: _PerQuery, k: int, *, metric: str) -> list[str]:
    """The four refusals every corpus metric shares, then the ``evaluated`` queries.

    One home for the order of the checks, so the metrics that average over ``evaluated`` cannot disagree about
    which input is refused first.
    """
    _validate_k(k)
    _validate_qrels(qrels)
    _validate_run(results)
    return _evaluated_queries(qrels, results, metric=metric)


def _require_matching_query_ids(qrels: _PerQuery, results: _PerQuery, *, metric: str) -> None:
    """Refuse a run whose query ids are not the qrels', naming the ids on each side.

    ``mrr``, ``recall_cap`` and ``hole`` divide by the number of qrels queries, so a query on one side only makes
    the value leave ``[0, 1]`` (``hole@1`` returned 2.0 for two runs and one qrel).
    """
    only_qrels = sorted(set(qrels) - set(results))
    only_run = sorted(set(results) - set(qrels))
    if only_qrels or only_run:
        raise ValueError(
            f"{metric} divides by the {len(qrels)} qrels queries, so the run must name exactly those queries; "
            f"{len(only_qrels)} query id(s) are in the qrels only ({only_qrels[:5]}) and {len(only_run)} are in "
            f"the run only ({only_run[:5]})"
        )


def _qrels_queries(qrels: _PerQuery, results: _PerQuery, k: int, *, metric: str) -> list[str]:
    """The shared refusals, the key-set check, then the ``all-qrels`` queries; refuse an empty qrels.

    The ``all-qrels`` denominator divides by the number of qrels queries, so the run must name exactly those
    queries: a query on one side only would let the value leave ``[0, 1]``. An empty qrels leaves no denominator
    at all, and 0.0 would report that as a measurement.
    """
    _validate_k(k)
    _validate_qrels(qrels)
    _validate_run(results)
    _require_matching_query_ids(qrels, results, metric=metric)
    if not qrels:
        raise ValueError(
            f"{metric} divides by the number of qrels queries, and the qrels name none: there is no denominator, "
            f"and 0.0 would report an undefined mean as a measurement"
        )
    return list(qrels)


def _ndcg_float_query(grades: Mapping[str, float], scores: Mapping[str, float], k: int) -> float:
    """One query's continuous-qrel nDCG@k: :func:`ndcg` with the ``"group_mean"`` credit and the identity gain.

    The one body :func:`ndcg_float` and :func:`per_query_ndcg` share, so the two cannot disagree about a query.
    The ideal is the query's whole float-qrel set, never only the ranked candidates.
    """
    return ndcg(scores, grades, k=k, ties="group_mean", ideal=grades.values())


def ndcg_float(qrels: _PerQuery, results: _PerQuery, *, k: int = 10) -> float:
    """nDCG@k over continuous qrels, averaged over the evaluated queries.

    Equal scores are credited the mean gain of their equal-score class, and the gain is the float qrel score
    itself, which must be non-negative. The credit is taken over the whole class and the cutoff is applied after
    it, so a class that straddles ``k`` credits every member with the same mean and only the members inside the
    cutoff enter the sum.

    Args:
        qrels: ``{query_id: {doc_id: float qrel}}``. Each value is the gain, so it is finite and at least 0 (apply
            a sigmoid at dataset creation time, not here). A query's whole set of values is its ideal DCG.
        results: ``{query_id: {doc_id: score}}``, higher is better, every score finite. A ranked document absent
            from the qrels counts 0.
        k: Cutoff, at least 1.

    Returns:
        The mean of the per-query nDCG@k over the ``evaluated`` queries -- those in both the qrels and the run --
        in ``[0, 1]``. A query with no positive qrel scores 0.0 and stays in the mean.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, a score is not finite, or no
            query is evaluated (an empty qrels, or a run that shares no query id with it).
    """
    evaluated = _scored_queries(qrels, results, k, metric="ndcg_float")
    return sum(_ndcg_float_query(qrels[query_id], results[query_id], k) for query_id in evaluated) / len(evaluated)


def per_query_ndcg(
    qrels: Mapping[str, _PerQuery],
    results: Mapping[str, _PerQuery],
    *,
    k: int = 10,
) -> dict[str, list[float]]:
    """Every dataset's per-query nDCG@k, as ``{dataset: [value, ...]}``; this function never takes a mean.

    This function returns the per-query values behind :func:`ndcg_float`, so it uses the same class-mean credit
    and the same identity gain on float qrels: one body, so the two cannot disagree about one query. The list
    holds one value per scored query, in the qrels' order.

    Args:
        qrels: ``{dataset: {query_id: {doc_id: float qrel}}}``, the values finite and at least 0.
        results: ``{dataset: {query_id: {doc_id: score}}}``, the scores finite. A dataset the run does not carry
            scores nothing, and a dataset that only the run carries contributes nothing.
        k: Cutoff, at least 1.

    Returns:
        One entry per dataset named by ``qrels``. The list holds the value of every query present in both sides
        and is empty when none is; a query in the qrels with no ranking is not scored and does not appear. The
        caller takes the mean (``rcp_core.protocol.aggregate``) and must refuse an empty result rather than
        average it.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, or a score is not finite.
    """
    _validate_k(k)
    for dataset_qrels in qrels.values():
        _validate_qrels(dataset_qrels)
    for dataset_results in results.values():
        _validate_run(dataset_results)
    values: dict[str, list[float]] = {}
    for dataset, dataset_qrels in qrels.items():
        dataset_results = results.get(dataset, {})
        values[dataset] = [
            _ndcg_float_query(grades, dataset_results[query_id], k)
            for query_id, grades in dataset_qrels.items()
            if query_id in dataset_results
        ]
    return values


def map(qrels: _PerQuery, results: _PerQuery, *, k: int = 10) -> float:
    """MAP@k averaged over the evaluated queries: the mean of the precisions at every relevant rank.

    Equal scores are ordered by document id, descending, and a document is relevant when its qrel grade is above
    0. The per-query denominator is ``|R_q|``, the number of positives in the query's whole qrels, never capped at
    ``k``: MAP@k is therefore below 1.0 when the query has more positives than ``k``.

    Args:
        qrels: ``{query_id: {doc_id: grade}}``; every grade is finite and at least 0, and only the sign is read.
        results: ``{query_id: {doc_id: score}}``, higher is better, every score finite.
        k: Cutoff, at least 1.

    Returns:
        The mean of ``(1 / |R_q|) * sum_{r=1..k} hits(r) / r`` over the ``evaluated`` queries, in ``[0, 1]``. A
        query with no relevant document scores 0.0 and stays in the mean.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, a score is not finite, or no
            query is evaluated (an empty qrels, or a run that shares no query id with it).
    """
    evaluated = _scored_queries(qrels, results, k, metric="map")
    total = 0.0
    for query_id in evaluated:
        grades = qrels[query_id]
        positives = len(_positive_docs(grades))
        if not positives:  # no relevant document: the query scores 0.0 and stays in the mean
            continue
        hits = 0
        precision_at_hits = 0.0
        for rank, doc_id in enumerate(rank_by_score(results[query_id], ties="doc_id_desc")[:k], start=1):
            if grades.get(doc_id, 0.0) > 0:
                hits += 1
                precision_at_hits += hits / rank
        total += precision_at_hits / positives
    return total / len(evaluated)


def recall(qrels: _PerQuery, results: _PerQuery, *, k: int = 10) -> float:
    """Recall@k averaged over the evaluated queries: the share of the query's positives inside the top k.

    Equal scores are ordered by document id, descending, and a document is relevant when its qrel grade is above
    0. The denominator is ``|R_q|``, the number of positives in the query's whole qrels, so a run shorter than
    the positives cannot reach 1.0.

    Args:
        qrels: ``{query_id: {doc_id: grade}}``; every grade is finite and at least 0, and only the sign is read.
        results: ``{query_id: {doc_id: score}}``, higher is better, every score finite.
        k: Cutoff, at least 1.

    Returns:
        The mean of ``|{d in top_k : rel(q, d)}| / |R_q|`` over the ``evaluated`` queries, in ``[0, 1]``. A query
        with no relevant document scores 0.0 and stays in the mean.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, a score is not finite, or no
            query is evaluated (an empty qrels, or a run that shares no query id with it).
    """
    evaluated = _scored_queries(qrels, results, k, metric="recall")
    total = 0.0
    for query_id in evaluated:
        grades = qrels[query_id]
        positives = len(_positive_docs(grades))
        if not positives:  # no relevant document: the query scores 0.0 and stays in the mean
            continue
        ranking = rank_by_score(results[query_id], ties="doc_id_desc")[:k]
        total += sum(1 for doc_id in ranking if grades.get(doc_id, 0.0) > 0) / positives
    return total / len(evaluated)


def precision(qrels: _PerQuery, results: _PerQuery, *, k: int = 10) -> float:
    """P@k averaged over the evaluated queries.

    Equal scores are ordered by document id, descending, and the denominator is the requested ``k``, so a ranking
    shorter than ``k`` cannot reach 1.0. That is the ``pytrec_eval`` ``P_cut`` definition and the number the
    legacy runs report.

    Args:
        qrels: ``{query_id: {doc_id: grade}}``; every grade is finite and at least 0, and only the sign is read.
        results: ``{query_id: {doc_id: score}}``, higher is better, every score finite.
        k: Cutoff, at least 1. It is the denominator, not the ranking's length.

    Returns:
        The mean of ``|{d in top_k : rel(q, d)}| / k`` over the ``evaluated`` queries, in ``[0, 1]``. A query with
        no relevant document scores 0.0 and stays in the mean.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, a score is not finite, or no
            query is evaluated (an empty qrels, or a run that shares no query id with it).
    """
    evaluated = _scored_queries(qrels, results, k, metric="precision")
    total = 0.0
    for query_id in evaluated:
        grades = qrels[query_id]
        ranking = rank_by_score(results[query_id], ties="doc_id_desc")[:k]
        total += sum(1 for doc_id in ranking if grades.get(doc_id, 0.0) > 0) / k
    return total / len(evaluated)


def success(qrels: _PerQuery, results: _PerQuery, *, k: int = 10) -> float:
    """success@k averaged over the evaluated queries: the share of queries with a hit anywhere in their top k.

    Equal scores are ordered by document id, descending, and a query scores 1.0 when any document in its top k has
    a qrel grade above 0. The per-query value is binary, so no partial credit can hide a wrong tie order: the rule
    matters only when a tie straddles the cutoff.

    Args:
        qrels: ``{query_id: {doc_id: grade}}``; every grade is finite and at least 0, and only the sign is read.
        results: ``{query_id: {doc_id: score}}``, higher is better, every score finite.
        k: Cutoff, at least 1.

    Returns:
        The mean of the per-query 0.0 or 1.0 over the ``evaluated`` queries, in ``[0, 1]``. A query with no
        relevant document scores 0.0 and stays in the mean.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, a score is not finite, or no
            query is evaluated (an empty qrels, or a run that shares no query id with it).
    """
    evaluated = _scored_queries(qrels, results, k, metric="success")
    total = 0.0
    for query_id in evaluated:
        grades = qrels[query_id]
        ranking = rank_by_score(results[query_id], ties="doc_id_desc")[:k]
        total += 1.0 if any(grades.get(doc_id, 0.0) > 0 for doc_id in ranking) else 0.0
    return total / len(evaluated)


def mrr(qrels: _PerQuery, results: _PerQuery, *, k: int = 10) -> float:
    """MRR@k over the ``all-qrels`` denominator: the mean of ``1 / rank`` of the first relevant document.

    Equal scores keep the run's insertion order, and a document is relevant when its qrel grade is above 0. This is
    the first metric whose value the tie rule moves: a tie at the top moves the first hit from rank 1 to rank 2.

    The mean divides by the number of qrels queries, so a qrels query with no run contributes 0.0 -- and a run
    query the qrels do not name is refused rather than counted, which is how the value stayed in ``[0, 1]``.

    Args:
        qrels: ``{query_id: {doc_id: grade}}``; every grade is finite and at least 0, and only the sign is read.
        results: ``{query_id: {doc_id: score}}``, higher is better, every score finite, and its query ids exactly
            the qrels'.
        k: Cutoff, at least 1. A hit is found or it is not, so a ``k`` deeper than the run changes nothing.

    Returns:
        The mean over the qrels' queries of ``1 / rank`` of the first relevant document inside the top k, or 0.0
        when there is none, in ``[0, 1]``.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, a score is not finite, the qrels
            are empty, or the run's query ids differ from the qrels' (both sides are named).
    """
    queries = _qrels_queries(qrels, results, k, metric="mrr")
    total = 0.0
    for query_id in queries:
        grades = qrels[query_id]
        for rank, doc_id in enumerate(rank_by_score(results[query_id], ties="input_order")[:k], start=1):
            if grades.get(doc_id, 0.0) > 0:
                total += 1.0 / rank
                break
    return total / len(queries)


def recall_cap(qrels: _PerQuery, results: _PerQuery, *, k: int = 10) -> float:
    """R_cap@k over the ``all-qrels`` denominator: the depth-capped recall of BEIR's ``custom_metrics``.

    Equal scores keep the run's insertion order, and the denominator is the smaller of the positive count and
    ``k``. A run of ``k`` documents cannot retrieve more than ``k`` positives, so the cap keeps the value at 1.0
    when the query has more positives than ``k`` -- which is the one difference from :func:`recall`.

    The mean divides by the number of qrels queries, so a qrels query with no run contributes 0.0 -- and a run
    query the qrels do not name is refused rather than counted.

    Args:
        qrels: ``{query_id: {doc_id: grade}}``; every grade is finite and at least 0, and only the sign is read.
        results: ``{query_id: {doc_id: score}}``, higher is better, every score finite, and its query ids exactly
            the qrels'.
        k: Cutoff, at least 1. It is also the cap on the denominator.

    Returns:
        The mean over the qrels' queries of ``|{d in top_k : rel(q, d)}| / min(|R_q|, k)``, in ``[0, 1]``. A query
        with no relevant document has no denominator, scores 0.0 and stays in the mean.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, a score is not finite, the qrels
            are empty, or the run's query ids differ from the qrels' (both sides are named).
    """
    queries = _qrels_queries(qrels, results, k, metric="recall_cap")
    total = 0.0
    for query_id in queries:
        grades = qrels[query_id]
        ranking = rank_by_score(results[query_id], ties="input_order")[:k]
        retrieved = sum(1 for doc_id in ranking if grades.get(doc_id, 0.0) > 0)
        denominator = min(len(_positive_docs(grades)), k)
        if denominator:  # no relevant document: the query scores 0.0 and stays in the mean
            total += retrieved / denominator
    return total / len(queries)


def hole(qrels: _PerQuery, results: _PerQuery, *, k: int = 10) -> float:
    """hole@k over the ``all-qrels`` denominator: the share of the top k that the qrels never annotated.

    Equal scores keep the run's insertion order, and a document is annotated when any query of the corpus names it
    in the qrels. The annotated set is therefore corpus-wide: a document judged for one query is judged for this
    metric. This is BEIR's coverage gap, read from the annotated set rather than from a grade.

    The mean divides by the number of qrels queries, so a qrels query with no run contributes 0.0 -- and a run
    query the qrels do not name is refused rather than counted, which is the case that returned 2.0 for a share.

    Args:
        qrels: ``{query_id: {doc_id: grade}}``; every grade is finite and at least 0. Only the document ids are
            read, never the grades.
        results: ``{query_id: {doc_id: score}}``, higher is better, every score finite, and its query ids exactly
            the qrels'.
        k: Cutoff, at least 1. It is the denominator, so an empty ranking scores 0.0.

    Returns:
        The mean over the qrels' queries of ``|{d in top_k : d is not annotated}| / k``, in ``[0, 1]``.

    Raises:
        ValueError: ``k`` is not positive, a grade is not finite or is negative, a score is not finite, the qrels
            are empty, or the run's query ids differ from the qrels' (both sides are named).
    """
    queries = _qrels_queries(qrels, results, k, metric="hole")
    annotated = {doc_id for grades in qrels.values() for doc_id in grades}
    total = 0.0
    for query_id in queries:
        ranking = rank_by_score(results[query_id], ties="input_order")[:k]
        total += sum(1 for doc_id in ranking if doc_id not in annotated) / k
    return total / len(queries)


__all__ = [
    "TieRule",
    "dcg",
    "discount",
    "hole",
    "ideal_dcg",
    "map",
    "mrr",
    "ndcg",
    "ndcg_float",
    "per_query_ndcg",
    "precision",
    "rank_by_score",
    "recall",
    "recall_cap",
    "success",
    "tie_groups",
]
