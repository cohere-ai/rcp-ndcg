"""The Spearman gate's one home: tie-corrected, hand-computed values, no SciPy."""

from __future__ import annotations

import math

import pytest

from rcp_ndcg_test.errors import ConformanceError
from rcp_ndcg_test.ranks import average_ranks, spearman


def test_identical_rows_correlate_one_and_reversed_minus_one() -> None:
    assert spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    # a monotone transform of the scores does not move the rank correlation
    assert spearman([1, 10, 100, 1000], [5, 6, 7, 8]) == 1.0


def test_a_known_table_value() -> None:
    """x=[1,2,3,4], y=[2,1,4,3]: dx=[-1.5,-0.5,0.5,1.5], dy=[-0.5,-1.5,1.5,0.5]; the products sum to
    3.0 and both variances are 5, so rho = 3/5 = 0.6 (hand)."""
    assert spearman([1, 2, 3, 4], [2, 1, 4, 3]) == pytest.approx(0.6)


def test_ties_are_averaged_and_the_closed_form_is_not_used() -> None:
    """x=[1,2,2,4], y=[1,2,3,4]: the tied pair shares the average rank 1.5; dx = [-1.5,0,0,1.5] and
    dy = [-1.5,-0.5,0.5,1.5], so cov = 4.5, var_x = 4.5, var_y = 5 and rho = 4.5/sqrt(22.5) = 3/sqrt(10)
    -- the no-tie shortcut 1 - 6*sum(d^2)/(n(n^2-1)) would wrongly give 0.95."""
    assert average_ranks([1, 2, 2, 4]).tolist() == [0.0, 1.5, 1.5, 3.0]
    expected = 3.0 / math.sqrt(10.0)
    assert spearman([1, 2, 2, 4], [1, 2, 3, 4]) == pytest.approx(expected, rel=1e-12)
    assert expected != pytest.approx(0.95)


def test_ties_on_both_sides_cancel_to_zero() -> None:
    """x=[1,1,2,2], y=[1,2,1,2]: the rank rows are [0.5,0.5,2.5,2.5] and [0.5,2.5,0.5,2.5]; every
    product term cancels, so rho = 0."""
    assert average_ranks([1, 1, 2, 2]).tolist() == [0.5, 0.5, 2.5, 2.5]
    assert spearman([1, 1, 2, 2], [1, 2, 1, 2]) == pytest.approx(0.0)


def test_constant_rows_carry_no_ranking_information() -> None:
    """A constant row correlates at nan (a comparison against nan fails, never silently passes)."""
    assert math.isnan(spearman([2, 2, 2], [1, 2, 3]))
    assert math.isnan(spearman([1, 2, 3], [7, 7, 7]))


def test_rows_of_one_element_and_of_mismatched_lengths() -> None:
    assert spearman([1], [9]) == 1.0
    with pytest.raises(ConformanceError, match="one length"):
        spearman([1, 2], [1])


def test_the_rank_convention_is_zero_based_with_averaged_ties() -> None:
    assert average_ranks([30, 10, 20]).tolist() == [2.0, 0.0, 1.0]
    assert average_ranks([5, 5, 5, 1]).tolist() == [2.0, 2.0, 2.0, 0.0]  # three tied positions share (1+2+3)/3
