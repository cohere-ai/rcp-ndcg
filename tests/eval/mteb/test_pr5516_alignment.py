"""The float nDCG equals mteb PR #5516's own ``ndcg_float_scores`` (the vendored oracle in ``_pr5516``).

The matrix below covers the PR's documented cases -- distinct scores, exact ties, a tie block crossing the
cutoff, unjudged documents, all-zero gains, a query whose gains are all null, several k values -- so the
repository's metric is checked against the PR's code rather than against itself.
"""

from __future__ import annotations

import math

import pytest

from rcp_ndcg.errors import DataError
from rcp_ndcg.eval.mteb import ndcg_float_scores

from ._pr5516 import ndcg_float_scores as pr_ndcg_float_scores

K_VALUES = (1, 2, 3, 10)

#: (gains, results): every case the PR's tests cover, plus a mixed one with ties across queries.
MATRIX: list[tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]] = [
    (  # distinct scores, the ideal uses the query's best gains
        {"q1": {"d1": 1.0, "d2": 0.5, "d3": 0.0}},
        {"q1": {"d2": 0.9, "d1": 0.7, "d3": 0.1}},
    ),
    (  # every score tied: each position is credited the group-mean gain
        {"q1": {"d1": 1.0, "d2": 0.0, "d3": 0.0}},
        {"q1": {"d1": 0.5, "d2": 0.5, "d3": 0.5}},
    ),
    (  # a tie block crossing the cutoff
        {"q1": {"d1": 0.0, "d2": 1.0, "d3": 0.0}},
        {"q1": {"d1": 1.0, "d2": 0.5, "d3": 0.5}},
    ),
    (  # an unjudged document scores gain 0
        {"q1": {"d1": 1.0}},
        {"q1": {"d1": 0.9, "d_unjudged": 0.8}},
    ),
    (  # all-zero gains score 0 without dividing by zero
        {"q1": {"d1": 0.0, "d2": 0.0}},
        {"q1": {"d1": 0.9, "d2": 0.1}},
    ),
    (  # a query with no gains at all scores 0 and stays in the mean
        {"q1": {"d1": 1.0, "d2": 0.0}},
        {"q1": {"d1": 0.9, "d2": 0.1}, "q2": {"d1": 0.9, "d2": 0.1}},
    ),
    (  # an empty ranking scores 0
        {"q1": {"d1": 1.0}},
        {"q1": {}},
    ),
    (  # a tie group whose mean gain is fractional, across several k values
        {"q1": {"a": 0.3, "b": 0.7, "c": 0.1, "d": 0.9}, "q2": {"a": 0.5}},
        {"q1": {"a": 0.4, "b": 0.4, "c": 0.2, "d": 0.9}, "q2": {"a": 0.1}},
    ),
    (  # the verifier's adversarial case: the nAUC keys expose a 1-ulp per-query difference
        {"q0": {"d0": 0.25, "d1": 0.25, "d3": 1.0, "d4": 1.0}},
        {"q0": {"d0": 0.5, "d1": 1.0, "d2": 0.0, "d3": 0.0, "d4": 0.5, "d5": 0.0}},
    ),
]


def _same_metric(ours: dict[str, float], theirs: dict[str, float]) -> None:
    """Every key agrees; a NaN nAUC counts as equal to a NaN nAUC (a constant per-query score has no AUC)."""
    assert set(ours) == set(theirs)
    for key, value in ours.items():
        other = theirs[key]
        if math.isnan(value) and math.isnan(other):
            continue
        assert value == other, f"{key}: ours={value!r}, PR={other!r}"


@pytest.mark.parametrize(("gains", "results"), MATRIX)
def test_the_float_metric_equals_the_pr_s_own_function(
    gains: dict[str, dict[str, float]], results: dict[str, dict[str, float]]
) -> None:
    ours = ndcg_float_scores(gains, results, k_values=K_VALUES)
    theirs = pr_ndcg_float_scores(gains, results, K_VALUES)

    _same_metric(ours, theirs)


def test_the_pr_s_function_is_reachable_without_mteb(monkeypatch: pytest.MonkeyPatch) -> None:
    """The oracle is pure Python (its nAUCs are empty without mteb): the comparison above must not need the
    ``[mteb]`` extra to run, which is what lets the gate's plain environment exercise it."""
    import sys

    monkeypatch.setitem(sys.modules, "mteb._evaluators.retrieval_metrics", None)
    scores = pr_ndcg_float_scores({"q1": {"d1": 1.0}}, {"q1": {"d1": 1.0}}, [10])
    assert scores["ndcg_float_at_10"] == 1.0


def test_the_full_dict_including_the_naucs_matches_when_mteb_is_installed() -> None:
    """With mteb present both functions add the abstention nAUCs; the per-query values must be bit-identical
    (core's DCG divides by ``log2(rank+1)``, the PR's operation order), or the nAUC keys can differ (the
    verifier's adversarial case, where a 1-ulp per-query difference turned a nAUC into 1.0 from NaN)."""
    pytest.importorskip("mteb")
    gains = {"q0": {"d0": 0.25, "d1": 0.25, "d3": 1.0, "d4": 1.0}}
    results = {"q0": {"d0": 0.5, "d1": 1.0, "d2": 0.0, "d3": 0.0, "d4": 0.5, "d5": 0.0}}
    ours = ndcg_float_scores(gains, results, k_values=(1, 2, 3, 5, 10))
    theirs = pr_ndcg_float_scores(gains, results, [1, 2, 3, 5, 10])
    assert "nauc_ndcg_float_at_10_max" in ours
    _same_metric(ours, theirs)


def test_invalid_gains_and_nan_scores_are_refused_on_both_sides() -> None:
    for bad_gain in (-0.1, float("nan"), float("inf")):
        with pytest.raises(DataError, match="gain"):
            ndcg_float_scores({"q1": {"d1": bad_gain}}, {"q1": {"d1": 1.0}}, k_values=(10,))
        with pytest.raises(ValueError, match="finite and non-negative"):
            pr_ndcg_float_scores({"q1": {"d1": bad_gain}}, {"q1": {"d1": 1.0}}, [10])
    with pytest.raises(DataError, match="finite"):
        ndcg_float_scores({"q1": {"d1": 1.0}}, {"q1": {"d1": float("nan")}}, k_values=(10,))
    with pytest.raises(ValueError, match="NaN model score"):
        pr_ndcg_float_scores({"q1": {"d1": 1.0}}, {"q1": {"d1": float("nan")}}, [10])


def test_the_pr_s_own_tests_reproduce_against_our_function() -> None:
    """The five numbers the PR's ``tests/test_evaluators/test_evaluation_metrics.py`` asserts, against ours."""
    import math

    assert ndcg_float_scores(
        {"q1": {"d1": 1.0, "d2": 0.5, "d3": 0.0}}, {"q1": {"d2": 0.9, "d1": 0.7, "d3": 0.1}}, k_values=(2,)
    )["ndcg_float_at_2"] == pytest.approx((0.5 + 1.0 / math.log2(3)) / (1.0 + 0.5 / math.log2(3)), abs=1e-5)
    assert ndcg_float_scores(
        {"q1": {"d1": 1.0, "d2": 0.0, "d3": 0.0}}, {"q1": {"d1": 0.5, "d2": 0.5, "d3": 0.5}}, k_values=(10,)
    )["ndcg_float_at_10"] == pytest.approx((1 / 3) * (1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)), abs=1e-5)
    assert ndcg_float_scores(
        {"q1": {"d1": 0.0, "d2": 1.0, "d3": 0.0}}, {"q1": {"d1": 1.0, "d2": 0.5, "d3": 0.5}}, k_values=(2,)
    )["ndcg_float_at_2"] == pytest.approx(0.5 / math.log2(3), abs=1e-5)
    assert ndcg_float_scores({"q1": {"d1": 1.0}}, {"q1": {"d1": 0.9, "d_unjudged": 0.8}}, k_values=(2,))[
        "ndcg_float_at_2"
    ] == pytest.approx(1.0, abs=1e-5)
    assert (
        ndcg_float_scores({"q1": {"d1": 0.0, "d2": 0.0}}, {"q1": {"d1": 0.9, "d2": 0.1}}, k_values=(10,))[
            "ndcg_float_at_10"
        ]
        == 0.0
    )
