"""BEIR's ``custom_metrics`` pair: ``rcp_core.metric.recall_cap`` and ``rcp_core.metric.hole``.

The vectors are the ones frozen in the metric specification (sections 3 and 8.6 to 8.7). Both keep the run's
insertion order, so a tie straddling the cutoff moves the value: q2's class {a, b} at 0.8 is what makes
``recall_cap@1`` 0.5 rather than 1.0, and q3's is what makes ``hole@1`` 1.0 rather than 0.0.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from rcp_core import hole, recall_cap

# Query q1: the tie is the class {b, c} at 0.5. R_q1 = {a, b, d}.
Q1_SCORES = {"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.4, "e": 0.2}
Q1_GRADES = {"a": 2, "b": 1, "c": 0, "d": 1, "e": 0}

# Query q2: the tie is the class {a, b} at 0.8, and a is not in the qrels. R_q2 = {b, c}.
Q2_SCORES = {"a": 0.8, "b": 0.8, "c": 0.6, "d": 0.4, "e": 0.2}
Q2_GRADES = {"b": 1, "c": 1, "d": 0}

QRELS = {"q1": Q1_GRADES, "q2": Q2_GRADES}
RESULTS = {"q1": Q1_SCORES, "q2": Q2_SCORES}

# Query q3: the tie is the class {a, b} at 0.7, and a and d are annotated by no qrel of the corpus.
Q3_SCORES = {"a": 0.7, "b": 0.7, "c": 0.5, "d": 0.3}
Q3_GRADES = {"b": 1, "c": 1}


def test_recall_cap_at_one_and_two_with_a_tie_at_the_top() -> None:
    """q1 keeps a (1 of min(3, 1)) and q2 keeps a (0 of min(2, 1)), so 0.5; at k=2 they give 1.0 and 0.5."""
    assert recall_cap(QRELS, RESULTS, k=1) == pytest.approx(0.5, rel=1e-12)
    assert recall_cap(QRELS, RESULTS, k=2) == pytest.approx(0.75, rel=1e-12)


def test_recall_cap_caps_the_denominator_at_the_cutoff() -> None:
    """q1 has 3 positives and q2 has 2, so at k=5 both denominators are their positive counts: 1.0 each."""
    assert recall_cap(QRELS, RESULTS, k=5) == pytest.approx(1.0, rel=1e-12)


def test_recall_cap_at_k_one_is_where_the_tie_rule_shows() -> None:
    """Under the document-id order q2's top 1 is b, which is relevant, so the value would be 1.0."""
    assert recall_cap(QRELS, RESULTS, k=1) != pytest.approx(1.0, rel=1e-6)


def test_recall_cap_of_a_query_with_no_relevant_document_scores_zero() -> None:
    """min(0, k) is no denominator, so the query scores 0.0 and stays in the mean."""
    assert recall_cap({"q1": Q1_GRADES, "q2": {"b": 0}}, RESULTS, k=2) == pytest.approx(0.5, rel=1e-12)


def test_hole_of_the_unannotated_document_inside_the_cutoff() -> None:
    """The annotated set of q3's corpus is {b, c}, so the insertion order a, b, c, d gives 1/1, 1/2 and 1/3."""
    qrels = {"q3": Q3_GRADES}
    results = {"q3": Q3_SCORES}
    assert hole(qrels, results, k=1) == pytest.approx(1.0, rel=1e-12)
    assert hole(qrels, results, k=2) == pytest.approx(0.5, rel=1e-12)
    assert hole(qrels, results, k=3) == pytest.approx(0.3333333333333333, rel=1e-12)


def test_hole_at_k_one_is_where_the_tie_rule_shows() -> None:
    """Under the document-id order q3's top 1 is b, which is annotated, so the value would be 0.0."""
    assert hole({"q3": Q3_GRADES}, {"q3": Q3_SCORES}, k=1) != pytest.approx(0.0, rel=1e-6)


def test_hole_is_zero_when_every_returned_document_is_annotated() -> None:
    """The q1/q2 corpus annotates every document of both runs, so the coverage gap is 0.0."""
    assert hole(QRELS, RESULTS, k=2) == pytest.approx(0.0, rel=1e-12)


def test_hole_annotates_corpus_wide() -> None:
    """A document judged for one query is judged for the metric: q2's qrels annotate d, so q1 ranking it first
    is not a hole."""
    assert hole({"q1": {"a": 1}, "q2": {"d": 0}}, {"q1": {"d": 0.9, "a": 0.1}, "q2": {"d": 0.5}}, k=1) == pytest.approx(
        0.0, rel=1e-12
    )


def test_hole_divides_by_the_requested_cutoff() -> None:
    """A four-document run at k=5 has two holes in five positions, not two in four."""
    assert hole({"q3": Q3_GRADES}, {"q3": Q3_SCORES}, k=5) == pytest.approx(0.4, rel=1e-12)


def test_hole_refuses_the_key_set_that_returned_two() -> None:
    """Two runs against one qrel returned hole@1 = 2.0 for a share; both sides are named instead."""
    with pytest.raises(ValueError, match="in the run only"):
        hole({"q3": Q3_GRADES}, {"q3": Q3_SCORES, "q4": Q3_SCORES}, k=1)
    with pytest.raises(ValueError, match="in the qrels only"):
        hole({"q3": Q3_GRADES, "q4": Q3_GRADES}, {"q3": Q3_SCORES}, k=1)
    with pytest.raises(ValueError, match="in the run only"):
        recall_cap({"q3": Q3_GRADES}, {"q3": Q3_SCORES, "q4": Q3_SCORES}, k=1)


def test_the_pair_refuses_an_empty_qrels() -> None:
    for metric in (recall_cap, hole):
        with pytest.raises(ValueError, match="no denominator"):
            metric({}, {}, k=1)


@pytest.mark.parametrize("metric", [recall_cap, hole])
def test_the_pair_refuses_a_non_positive_cutoff(metric: Callable[..., float]) -> None:
    for k in (0, -1):
        with pytest.raises(ValueError, match="k must be greater than 0"):
            metric(QRELS, RESULTS, k=k)


@pytest.mark.parametrize("metric", [recall_cap, hole])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_the_pair_refuses_a_non_finite_grade_or_score(metric: Callable[..., float], bad: float) -> None:
    """hole reads no grade, but the grade is still validated: a metric refuses a corpus it cannot read whole."""
    with pytest.raises(ValueError, match="must be finite"):
        metric({"q1": {"a": bad}}, {"q1": {"a": 1.0}}, k=1)
    with pytest.raises(ValueError, match="must be finite"):
        metric({"q1": {"a": 1.0}}, {"q1": {"a": bad}}, k=1)


@pytest.mark.parametrize("metric", [recall_cap, hole])
def test_the_pair_refuses_a_negative_grade(metric: Callable[..., float]) -> None:
    with pytest.raises(ValueError, match="at least 0"):
        metric({"q1": {"a": -1.0}}, {"q1": {"a": 1.0}}, k=1)
