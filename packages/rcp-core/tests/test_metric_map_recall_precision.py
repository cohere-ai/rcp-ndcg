"""The BEIR-order binary metrics: ``rcp_core.metric.map``, ``recall`` and ``precision``.

The vectors are the ones frozen in the metric specification (sections 3 and 8.1 to 8.3). Each corpus carries two
ties -- q1's class {b, c} at 0.5 and q2's class {a, b} at 0.8 -- and the tie order decides the value, so a metric
that ordered a tie by the run's insertion order would fail every vector here.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from rcp_core import map, precision, recall

# Query q1: the tie is the class {b, c} at 0.5, and it straddles the k=2 cutoff. R_q1 = {a, b, d}.
Q1_SCORES = {"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.4, "e": 0.2}
Q1_GRADES = {"a": 2, "b": 1, "c": 0, "d": 1, "e": 0}

# Query q2: the tie is the class {a, b} at 0.8, and a is not in the qrels. R_q2 = {b, c}.
Q2_SCORES = {"a": 0.8, "b": 0.8, "c": 0.6, "d": 0.4, "e": 0.2}
Q2_GRADES = {"b": 1, "c": 1, "d": 0}

QRELS = {"q1": Q1_GRADES, "q2": Q2_GRADES}
RESULTS = {"q1": Q1_SCORES, "q2": Q2_SCORES}


def test_map_of_the_corpus_with_a_tie_at_the_cutoff() -> None:
    """q1 keeps only a at k=2 (1.0 / 3), q2 only b (1.0 / 2), so the mean is 0.41666…"""
    assert map(QRELS, RESULTS, k=2) == pytest.approx(0.41666666666666663, rel=1e-12)


def test_map_counts_every_positive_not_only_the_ones_inside_the_cutoff() -> None:
    """At k=5 both queries retrieve all their positives, and the mean is 0.8194444444444444."""
    assert map(QRELS, RESULTS, k=5) == pytest.approx(0.8194444444444444, rel=1e-12)


def test_recall_of_the_corpus_with_a_tie_at_the_cutoff() -> None:
    """q1 retrieves 1 of 3, q2 1 of 2: 0.41666…  At k=5 both retrieve every positive, so the value is 1.0."""
    assert recall(QRELS, RESULTS, k=2) == pytest.approx(0.41666666666666663, rel=1e-12)
    assert recall(QRELS, RESULTS, k=5) == pytest.approx(1.0, rel=1e-12)


def test_precision_of_the_corpus_with_a_tie_at_the_cutoff() -> None:
    """One hit in each top 2, so P@2 = 0.5; at k=5 the two queries give 3/5 and 2/5, so P@5 = 0.5 as well."""
    assert precision(QRELS, RESULTS, k=2) == pytest.approx(0.5, rel=1e-12)
    assert precision(QRELS, RESULTS, k=5) == pytest.approx(0.5, rel=1e-12)


def test_the_tie_rule_is_the_document_id_descending_one() -> None:
    """The neighbouring rule gives a different number, so these vectors pin the rule, not just the arithmetic.

    Under the run's insertion order, q1 keeps a and b (MAP@2 = 0.6666…, Recall@2 = 2/3, P@2 = 1.0) and the corpus
    values become 0.5833333333333333, 0.5833333333333333 and 0.75.
    """
    assert map(QRELS, RESULTS, k=2) != pytest.approx(0.5833333333333333, rel=1e-6)
    assert recall(QRELS, RESULTS, k=2) != pytest.approx(0.5833333333333333, rel=1e-6)
    assert precision(QRELS, RESULTS, k=2) != pytest.approx(0.75, rel=1e-6)


def test_precision_divides_by_the_requested_cutoff() -> None:
    """A two-document run at k=5 can never reach 1.0: one hit is 1/5, not 1/2."""
    assert precision({"q": {"a": 1}}, {"q": {"a": 0.9, "b": 0.1}}, k=5) == pytest.approx(0.2, rel=1e-12)
    assert recall({"q": {"a": 1}}, {"q": {"a": 0.9, "b": 0.1}}, k=5) == pytest.approx(1.0, rel=1e-12)


def test_a_query_with_no_relevant_document_scores_zero_and_stays_in_the_mean() -> None:
    qrels = {"q1": Q1_GRADES, "q2": {"b": 0}}
    assert map(qrels, RESULTS, k=2) == pytest.approx(0.3333333333333333 / 2, rel=1e-12)
    assert recall(qrels, RESULTS, k=2) == pytest.approx(0.3333333333333333 / 2, rel=1e-12)
    assert precision(qrels, RESULTS, k=2) == pytest.approx(0.5 / 2, rel=1e-12)


@pytest.mark.parametrize("metric", [map, recall, precision])
def test_the_binary_metrics_refuse_an_undefined_mean(metric: Callable[..., float]) -> None:
    """0.0 would report "nothing ranked correctly" for a corpus that scored nothing at all."""
    with pytest.raises(ValueError, match="no query to average over"):
        metric({}, RESULTS, k=2)
    with pytest.raises(ValueError, match="no query to average over"):
        metric(QRELS, {"other": Q1_SCORES}, k=2)
    for k in (0, -1):
        with pytest.raises(ValueError, match="k must be greater than 0"):
            metric(QRELS, RESULTS, k=k)


@pytest.mark.parametrize("metric", [map, recall, precision])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_the_binary_metrics_refuse_a_non_finite_grade_or_score(metric: Callable[..., float], bad: float) -> None:
    """A NaN grade used to read as "not relevant" and an inf one as relevant; both are refused now."""
    with pytest.raises(ValueError, match="must be finite"):
        metric({"q1": {"a": bad}}, {"q1": {"a": 1.0}}, k=1)
    with pytest.raises(ValueError, match="must be finite"):
        metric({"q1": {"a": 1.0}}, {"q1": {"a": bad}}, k=1)


@pytest.mark.parametrize("metric", [map, recall, precision])
def test_the_binary_metrics_refuse_a_negative_grade(metric: Callable[..., float]) -> None:
    """Every metric refuses a negative grade, so the binary metrics cannot disagree with the gain ones."""
    with pytest.raises(ValueError, match="at least 0"):
        metric({"q1": {"a": -1.0}}, {"q1": {"a": 1.0}}, k=1)
