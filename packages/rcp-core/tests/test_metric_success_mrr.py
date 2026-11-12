"""The pair whose tie rule is the run's order, or the document id's: ``success`` and ``mrr``.

The vectors are the ones frozen in the metric specification (sections 3 and 8.4 to 8.5). ``success`` orders a tie by
document id, descending; ``mrr`` keeps the run's insertion order, and it is the first metric where the rule changes a
number -- q2's tie {a, b} at 0.8 puts the first hit at rank 2 instead of rank 1.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
from rcp_core import mrr, success

# Query q1: the tie is the class {b, c} at 0.5. R_q1 = {a, b, d}.
Q1_SCORES = {"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.4, "e": 0.2}
Q1_GRADES = {"a": 2, "b": 1, "c": 0, "d": 1, "e": 0}

# Query q2: the tie is the class {a, b} at 0.8, and a is not in the qrels. R_q2 = {b, c}.
Q2_SCORES = {"a": 0.8, "b": 0.8, "c": 0.6, "d": 0.4, "e": 0.2}
Q2_GRADES = {"b": 1, "c": 1, "d": 0}

QRELS = {"q1": Q1_GRADES, "q2": Q2_GRADES}
RESULTS = {"q1": Q1_SCORES, "q2": Q2_SCORES}


@pytest.mark.parametrize("k", [1, 2, 5])
def test_success_of_the_corpus_with_a_tie_at_the_cutoff(k: int) -> None:
    """Both queries have a relevant document first under the document-id order, so every cutoff gives 1.0."""
    assert success(QRELS, RESULTS, k=k) == pytest.approx(1.0, rel=1e-12)


def test_success_at_k_one_is_where_the_tie_rule_shows() -> None:
    """Under the run's insertion order q2's top 1 is a, which no qrel names, so the value would be 0.5."""
    assert success(QRELS, RESULTS, k=1) != pytest.approx(0.5, rel=1e-6)


def test_mrr_at_k_one_and_two_with_a_tie_at_the_top() -> None:
    """q1: a is relevant at rank 1 (1.0). q2: a is not, b is at rank 2 (0.5). So 0.5 at k=1 and 0.75 at k=2."""
    assert mrr(QRELS, RESULTS, k=1) == pytest.approx(0.5, rel=1e-12)
    assert mrr(QRELS, RESULTS, k=2) == pytest.approx(0.75, rel=1e-12)


def test_mrr_of_a_cutoff_deeper_than_the_run_is_the_same_number() -> None:
    assert mrr(QRELS, RESULTS, k=5) == pytest.approx(0.75, rel=1e-12)


def test_mrr_keeps_the_run_s_insertion_order() -> None:
    """The document-id order would put q2's b first and give mrr@1 = 1.0: one tie rule, two numbers."""
    assert mrr(QRELS, RESULTS, k=1) != pytest.approx(1.0, rel=1e-6)


def test_mrr_divides_by_the_number_of_qrels_queries() -> None:
    """A qrels query with an empty ranking contributes 0.0 and stays in the denominator."""
    assert mrr(QRELS, {"q1": Q1_SCORES, "q2": {}}, k=2) == pytest.approx(0.5, rel=1e-12)


def test_a_query_with_no_relevant_document_scores_zero_and_stays_in_the_mean() -> None:
    qrels = {"q1": Q1_GRADES, "q2": {"b": 0}}
    assert mrr(qrels, RESULTS, k=2) == pytest.approx(0.5, rel=1e-12)
    assert success(qrels, RESULTS, k=2) == pytest.approx(0.5, rel=1e-12)


def test_mrr_refuses_a_run_that_names_a_query_the_qrels_do_not() -> None:
    """The legacy mean summed over the run and divided by len(qrels), so the value left [0, 1]."""
    with pytest.raises(ValueError, match="in the run only"):
        mrr(QRELS, {**RESULTS, "q3": Q2_SCORES}, k=2)
    with pytest.raises(ValueError, match="in the qrels only"):
        mrr(QRELS, {"q1": Q1_SCORES}, k=2)


def test_mrr_refuses_an_empty_qrels() -> None:
    with pytest.raises(ValueError, match="no denominator"):
        mrr({}, {}, k=2)


def test_success_refuses_an_undefined_mean() -> None:
    """success averages over the evaluated queries; a corpus that scores none has no mean."""
    with pytest.raises(ValueError, match="no query to average over"):
        success({}, RESULTS, k=2)


@pytest.mark.parametrize("metric", [success, mrr])
def test_the_pair_refuses_a_non_positive_cutoff(metric: Callable[..., float]) -> None:
    for k in (0, -1):
        with pytest.raises(ValueError, match="k must be greater than 0"):
            metric(QRELS, RESULTS, k=k)


@pytest.mark.parametrize("metric", [success, mrr])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_the_pair_refuses_a_non_finite_grade_or_score(metric: Callable[..., float], bad: float) -> None:
    """A NaN score used to return mrr = 1.0: it compares False with everything, so the sort kept the input order."""
    with pytest.raises(ValueError, match="must be finite"):
        metric({"q1": {"a": bad}}, {"q1": {"a": 1.0}}, k=1)
    with pytest.raises(ValueError, match="must be finite"):
        metric({"q1": {"a": 1.0}}, {"q1": {"a": bad}}, k=1)


@pytest.mark.parametrize("metric", [success, mrr])
def test_the_pair_refuses_a_negative_grade(metric: Callable[..., float]) -> None:
    with pytest.raises(ValueError, match="at least 0"):
        metric({"q1": {"a": -1.0}}, {"q1": {"a": 1.0}}, k=1)
