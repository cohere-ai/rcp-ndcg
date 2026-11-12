"""The two metrics that form no ranking: ``rcp_core.metric.paired_accuracy`` and ``coverage``.

The vectors are the ones frozen in the metric specification (sections 9.5 and 10.5). Neither takes a ``k``, so none
of the three ranking tie rules applies; each has one boundary instead -- a strict score comparison, and ``>=``
against the coverage threshold -- and both boundaries decide the value of their vector.
"""

from __future__ import annotations

import math

import pytest
from rcp_core import coverage, paired_accuracy

# Group g1 is correct in both directions; group g2 ties, and a strict comparison is not a win.
PAIR_QRELS = {"q1": {"d1": 1}, "q2": {"d2": 1}, "q3": {"d3": 1}, "q4": {"d4": 1}}
PAIR_RESULTS = {
    "q1": {"d1": 0.9, "d2": 0.1},
    "q2": {"d2": 0.8, "d1": 0.2},
    "q3": {"d3": 0.5, "d4": 0.5},
    "q4": {"d4": 0.9, "d3": 0.1},
}
PAIR_GROUPS = {"q1": "g1", "q2": "g1", "q3": "g2", "q4": "g2"}


def test_paired_accuracy_with_a_tied_pair() -> None:
    """g1 is correct (0.9 > 0.1 and 0.8 > 0.2); g2 is not, because 0.5 > 0.5 is false. So 1 / 2."""
    assert paired_accuracy(PAIR_QRELS, PAIR_RESULTS, PAIR_GROUPS) == pytest.approx(0.5, rel=1e-12)


def test_the_strict_comparison_decides_the_vector() -> None:
    """With 0.6 instead of 0.5 for q3's own document, both directions are strict wins and the value is 1.0."""
    results = {**PAIR_RESULTS, "q3": {"d3": 0.6, "d4": 0.5}}
    assert paired_accuracy(PAIR_QRELS, results, PAIR_GROUPS) == pytest.approx(1.0, rel=1e-12)


def test_a_document_missing_from_the_run_loses_every_comparison() -> None:
    """A missing score is -inf, so q1's own document is below the other query's and the pair is not correct."""
    results = {**PAIR_RESULTS, "q1": {"d2": 0.1}}
    assert paired_accuracy(PAIR_QRELS, results, PAIR_GROUPS) == pytest.approx(0.0, rel=1e-12)


def test_a_group_that_is_not_exactly_two_queries_is_skipped() -> None:
    """A third query in g1 makes it no pair at all, so g1 leaves the mean and only g2 (not correct) is left."""
    assert paired_accuracy(PAIR_QRELS, PAIR_RESULTS, {**PAIR_GROUPS, "q5": "g1"}) == pytest.approx(0.0, rel=1e-12)
    assert paired_accuracy(PAIR_QRELS, PAIR_RESULTS, {**PAIR_GROUPS, "q5": "g3"}) == pytest.approx(0.5, rel=1e-12)


def test_a_pair_with_no_positive_on_one_side_is_skipped() -> None:
    qrels = {**PAIR_QRELS, "q3": {"d3": 0}}
    assert paired_accuracy(qrels, PAIR_RESULTS, PAIR_GROUPS) == pytest.approx(1.0, rel=1e-12)


def test_paired_accuracy_refuses_an_undefined_denominator() -> None:
    """Every one of these returned 0.0, which reports an undefined value as a measurement."""
    with pytest.raises(ValueError, match="pair_groups"):
        paired_accuracy(PAIR_QRELS, PAIR_RESULTS, {})
    with pytest.raises(ValueError, match="no pair to divide by"):
        paired_accuracy(PAIR_QRELS, PAIR_RESULTS, {"q1": "g", "q2": "g", "q3": "g"})
    with pytest.raises(ValueError, match="no pair to divide by"):
        paired_accuracy(PAIR_QRELS, {"q1": PAIR_RESULTS["q1"]}, PAIR_GROUPS)
    with pytest.raises(ValueError, match="no pair to divide by"):
        paired_accuracy({"q1": {"d1": 0}, "q2": {"d2": 0}}, {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}, PAIR_GROUPS)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_paired_accuracy_refuses_a_non_finite_score_or_grade(bad: float) -> None:
    """A NaN comparison is false, so it would count as a loss with no error."""
    with pytest.raises(ValueError, match="must be finite"):
        paired_accuracy({**PAIR_QRELS, "q1": {"d1": bad}}, PAIR_RESULTS, PAIR_GROUPS)
    with pytest.raises(ValueError, match="must be finite"):
        paired_accuracy(PAIR_QRELS, {**PAIR_RESULTS, "q1": {"d1": bad}}, PAIR_GROUPS)


def test_paired_accuracy_refuses_a_negative_grade() -> None:
    with pytest.raises(ValueError, match="at least 0"):
        paired_accuracy({**PAIR_QRELS, "q1": {"d1": -1}}, PAIR_RESULTS, PAIR_GROUPS)


def test_coverage_of_the_threshold_boundary() -> None:
    """i2 and i4 sit exactly on 0.90 and are covered, so 3 of 4: a '>' comparison would give 0.25."""
    assert coverage({"i1": 0.95, "i2": 0.90, "i3": 0.85, "i4": 0.90}, threshold=0.90) == pytest.approx(0.75, rel=1e-12)


def test_coverage_defaults_to_the_ninety_percent_threshold() -> None:
    assert coverage({"i1": 0.95, "i2": 0.90, "i3": 0.85, "i4": 0.90}) == pytest.approx(0.75, rel=1e-12)


def test_coverage_reports_zero_and_one_as_real_values() -> None:
    assert coverage({"i1": 0.1, "i2": 0.2}) == pytest.approx(0.0, rel=1e-12)
    assert coverage({"i1": 1.0, "i2": 0.9}) == pytest.approx(1.0, rel=1e-12)


def test_coverage_refuses_an_empty_answered_set() -> None:
    """0.0 is the different fact that every answered item fell below the threshold."""
    with pytest.raises(ValueError, match="at least one answered item"):
        coverage({})


@pytest.mark.parametrize("threshold", [0.0, -0.5, 1.5, math.nan])
def test_coverage_refuses_a_threshold_outside_the_half_open_unit_interval(threshold: float) -> None:
    """A 0.0 threshold would cover every item, which is not a confidence."""
    with pytest.raises(ValueError, match=r"threshold must be in \(0, 1\]"):
        coverage({"i1": 0.5}, threshold=threshold)


@pytest.mark.parametrize("probability", [math.nan, math.inf, -math.inf, 1.5, -0.1])
def test_coverage_refuses_a_probability_that_is_not_a_finite_fraction(probability: float) -> None:
    with pytest.raises(ValueError, match="must be finite|fraction in"):
        coverage({"i1": probability})


def test_coverage_refuses_an_empty_item_id() -> None:
    """The mapping's key is the item, so an empty key is a probability no item can own."""
    with pytest.raises(ValueError, match="non-empty item id"):
        coverage({"": 0.95})
