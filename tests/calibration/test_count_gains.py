"""Count-nDCG's product path: ``count_gains()`` derives the rubric-only gains from the rubric windows.

The paper's Count-nDCG gain is the share of passed criteria, ``sum_c S_c / (C * n)``, where the unit of evidence
is the window and sibling chunks pool inside it. The helper is the one derivation: ``evaluate(count_gains=...)``
and ``rcp-ndcg eval score --metrics count_ndcg --judgements STORE`` read it.
"""

from __future__ import annotations

import pytest
from rcp_ndcg_core.schemas import JudgementFamily, criterion_labels

from rcp_ndcg.calibration import count_gains
from rcp_ndcg.errors import DataError

from .conftest import rubric_set


def _windows() -> dict:
    return {
        ("ds", "q"): [
            [("a", None, [1, 1, 0, 1, 0]), ("b", None, [0, 0, 0, 0, 0])],
            [("a", None, [1, 0, 0, 1, 0])],
            [("a", None, [0, 0, 0, 1, 0]), ("b", None, [1, 1, 1, 1, 1])],
        ]
    }


def test_the_gain_is_the_share_of_passed_criteria_over_the_windows() -> None:
    gains = count_gains(rubric_set(_windows()))
    # a passed C1 in 2 of its 3 windows, C2 in 1, C4 in 3: (2 + 1 + 3) / (5 * 3); b passed everything twice.
    assert gains == {"q": {"a": pytest.approx(0.4), "b": pytest.approx(0.5)}}
    assert all(0.0 <= gain <= 1.0 for docs in gains.values() for gain in docs.values())


def test_sibling_chunks_are_one_placement_and_pool_by_criterion() -> None:
    rows = rubric_set(
        {
            ("ds", "q"): [
                [("a", "a#0", [1, 0, 0, 0, 0]), ("a", "a#1", [0, 0, 0, 0, 1])],
                [("a", None, [0, 0, 0, 0, 0])],
            ]
        }
    )
    # One window with two chunks is one placement: a passed C1 (chunk 0) and C5 (chunk 1) in 2 of 2 windows.
    assert count_gains(rows) == {"q": {"a": pytest.approx(2 / 10)}}


def test_keys_follow_the_calibration_rule() -> None:
    one = rubric_set(_windows())
    two = rubric_set({**{("ds", "q"): _windows()[("ds", "q")]}, ("other", "q"): _windows()[("ds", "q")]})
    assert set(count_gains(one)) == {"q"}
    assert set(count_gains(one, dataset="ds")) == {"q"}
    assert set(count_gains(two)) == {"ds||q", "other||q"}
    assert set(count_gains(two, dataset="other")) == {"q"}
    with pytest.raises(DataError, match="no rubric verdicts for dataset 'nope'"):
        count_gains(two, dataset="nope")


def test_a_document_judged_under_two_rubric_sizes_is_refused() -> None:
    five = rubric_set(_windows())
    (family,) = five.families.values()
    three = JudgementFamily(
        stage="rubric",
        judge_model="hand",
        prompt_hash="f" * 64,
        criteria=criterion_labels(3),
        parse_version=1,
    )
    short = rubric_set({("ds", "q"): [[("a", None, [1, 1, 1])]]}, family=three)
    with pytest.raises(DataError, match="rubrics of different criteria"):
        count_gains([five, short])
