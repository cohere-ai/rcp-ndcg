"""The metric and the gains: ``rcp_ndcg_core.metric`` and ``rcp_ndcg_core.gain``."""

from __future__ import annotations

import math

import numpy as np
import pytest
from rcp_ndcg_core import count_gain, dcg, gain, ndcg, pass_probabilities, qrel_gain
from rcp_ndcg_core.metric import discount, ideal_dcg, rank_by_score

LOG2_3 = math.log2(3)
ITEMS = {"gamma": [1.2, 1.0, 0.8], "beta": [-2.0, 0.0, 3.0]}


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# DCG and nDCG
# ---------------------------------------------------------------------------


def test_dcg_discounts_by_log2_of_the_rank_plus_one() -> None:
    assert discount(1) == 1.0
    assert dcg([3.0, 2.0, 1.0], k=3) == pytest.approx(3.0 + 2.0 / LOG2_3 + 0.5)
    assert dcg([3.0, 2.0, 1.0], k=2) == pytest.approx(3.0 + 2.0 / LOG2_3)
    assert dcg([], k=5) == 0.0
    assert ideal_dcg([1.0, 3.0, 2.0], k=3) == dcg([3.0, 2.0, 1.0], k=3)


def test_ndcg_of_a_ranking_against_float_gains() -> None:
    gains = {"d1": 0.9, "d2": 0.1, "d3": 0.5}
    ideal = 0.9 + 0.5 / LOG2_3 + 0.1 / 2
    assert ndcg(["d1", "d3", "d2"], gains, k=3) == pytest.approx(1.0)
    assert ndcg(["d2", "d3", "d1"], gains, k=3) == pytest.approx((0.1 + 0.5 / LOG2_3 + 0.9 / 2) / ideal)
    assert ndcg(["unknown", "d1"], {"d1": 1.0}, k=2) == pytest.approx(1 / LOG2_3), "a missing id counts 0"
    assert ndcg([], gains, k=3) == 0.0
    assert ndcg(["d1"], {}, k=3) == 0.0
    assert ndcg(["a", "b"], {"a": 0.0, "b": 0.0}, k=2) == 0.0


def test_the_ideal_can_come_from_a_larger_labelled_set() -> None:
    assert ndcg(["a"], {"a": 1.0}, k=2, ideal=[1.0, 1.0]) == pytest.approx(1 / (1 + 1 / LOG2_3))


def test_ndcg_refuses_bad_input() -> None:
    for k in (0, -1):
        with pytest.raises(ValueError, match="k must be greater than 0"):
            ndcg([], {}, k=k)
    with pytest.raises(ValueError, match="duplicate document IDs"):
        ndcg(["d1", "d1"], {"d1": 1.0}, k=2)
    with pytest.raises(ValueError, match="finite"):
        ndcg({"a": math.nan}, {"a": 1.0})
    with pytest.raises(ValueError, match="Unknown tie rule"):
        ndcg({"a": 1.0}, {"a": 1.0}, ties="stable")  # type: ignore[arg-type]


class TestTieRules:
    """``ndcg`` of a score mapping under each tie rule."""

    SCORES = {"a": 1.0, "b": 1.0, "c": 0.0}
    GAINS = {"a": 1.0, "b": 0.0, "c": 0.0}

    def test_rules_agree_with_the_ranking_form_when_nothing_ties(self) -> None:
        scores = {"a": 3.0, "b": 2.0, "c": 1.0}
        gains = {"a": 0.2, "b": 0.9, "c": 0.4}
        for ties in ("doc_id_desc", "group_mean", "input_order"):
            assert ndcg(scores, gains, k=3, ties=ties) == pytest.approx(ndcg(["a", "b", "c"], gains, k=3))

    def test_group_mean_credits_the_tie_class_mean_whatever_the_input_order(self) -> None:
        expected = 0.5 + 0.5 / LOG2_3
        assert ndcg(self.SCORES, self.GAINS, k=2) == pytest.approx(expected)
        assert ndcg({"b": 1.0, "a": 1.0, "c": 0.0}, self.GAINS, k=2) == pytest.approx(expected)

    def test_doc_id_desc_breaks_ties_by_descending_id(self) -> None:
        assert rank_by_score(self.SCORES, ties="doc_id_desc") == ["b", "a", "c"]
        assert ndcg(self.SCORES, self.GAINS, k=2, ties="doc_id_desc") == pytest.approx(1 / LOG2_3)

    def test_input_order_keeps_the_order_the_scores_come_in(self) -> None:
        assert rank_by_score({"c": 0.0, "b": 1.0, "a": 1.0}, ties="input_order") == ["b", "a", "c"]
        assert ndcg(self.SCORES, self.GAINS, k=2, ties="input_order") == 1.0

    def test_a_fully_tied_model_scores_the_random_ranking_expectation(self) -> None:
        """Never the pool order, which is 1.0 when the pool is relevance-ordered."""
        gains = {"a": 1.0, "b": 0.8, "c": 0.6, "d": 0.4, "e": 0.2}
        degenerate = ndcg({d: 1.0 for d in gains}, gains, k=5)
        assert degenerate < 1.0
        assert degenerate == ndcg({d: 0.0 for d in gains}, gains, k=5)


# ---------------------------------------------------------------------------
# Gains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [0.0, -5.0, 5.0])
def test_the_gain_is_the_discrimination_weighted_mean_of_the_pass_probabilities(theta: float) -> None:
    probs = [_logistic(1.2 * (theta + 2.0)), _logistic(1.0 * theta), _logistic(0.8 * (theta - 3.0))]
    assert pass_probabilities(theta, ITEMS) == pytest.approx(probs, abs=1e-15)
    assert gain(theta, ITEMS) == pytest.approx((1.2 * probs[0] + probs[1] + 0.8 * probs[2]) / 3.0, abs=1e-15)


def test_gain_takes_item_parameter_objects_and_arrays_of_abilities() -> None:
    class Items:
        gamma = ITEMS["gamma"]
        beta = ITEMS["beta"]

    thetas = np.array([[-3.0, 0.5], [2.0, 40.0]])
    assert gain(0.5, Items()) == gain(0.5, ITEMS)
    assert gain(thetas, ITEMS) == pytest.approx(np.array([[gain(t, ITEMS) for t in row] for row in thetas.tolist()]))
    assert pass_probabilities(thetas, ITEMS).shape == (2, 2, 3)


def test_pass_probabilities_are_stable_far_from_the_difficulties() -> None:
    items = {"gamma": [1.0], "beta": [0.0]}
    assert pass_probabilities(0.0, items)[0] == 0.5
    assert pass_probabilities(-800.0, items)[0] == 0.0
    assert pass_probabilities(800.0, items)[0] == 1.0
    assert pass_probabilities(np.array([-800.0, 800.0]), items).ravel().tolist() == [0.0, 1.0]


def test_gain_increases_with_theta() -> None:
    thetas = np.linspace(-6, 6, 25)
    assert np.all(np.diff(gain(thetas, ITEMS)) > 0)


@pytest.mark.parametrize(
    "items",
    [
        {"gamma": [1.0, 2.0], "beta": [0.0]},
        {"gamma": [float("nan")], "beta": [0.0]},
        {"gamma": [1.0], "beta": [float("inf")]},
        {"gamma": [], "beta": []},
        {"gamma": "1", "beta": "0"},
        {"beta": [0.0]},
        object(),
    ],
)
def test_gain_rejects_malformed_item_parameters(items: object) -> None:
    with pytest.raises(ValueError, match="item parameter"):
        gain(0.0, items)


def test_the_weighted_gain_needs_a_positive_gamma_sum() -> None:
    items = {"gamma": [-1.0, 1.0], "beta": [0.0, 1.0]}
    with pytest.raises(ValueError, match=r"sum\(gamma\)"):
        gain(0.0, items)
    assert len(pass_probabilities(0.0, items)) == 2, "the pass probabilities need no weights"


def test_count_gain_is_the_share_of_passed_criteria() -> None:
    assert count_gain([2, 2, 1, 0, 0], placements=2) == pytest.approx(5 / 10)
    assert count_gain([1, 0, 0, 0, 0], placements=1) == pytest.approx(1 / 5)
    with pytest.raises(ValueError, match="placements"):
        count_gain([1, 0], placements=0)
    with pytest.raises(ValueError, match="criterion"):
        count_gain([], placements=1)


def test_qrel_gain_is_linear_unless_exponential_is_asked_for() -> None:
    assert [qrel_gain(g) for g in (2, 1, 0, 0.5)] == [2.0, 1.0, 0.0, 0.5]
    assert [qrel_gain(g, "exponential") for g in (2, 1, 0)] == [3.0, 1.0, 0.0]
    with pytest.raises(ValueError, match="scheme"):
        qrel_gain(1, "log")  # type: ignore[arg-type]


def test_tie_groups_are_the_equal_score_classes_the_group_mean_rule_credits() -> None:
    from rcp_ndcg_core.metric import tie_groups

    scores = {"a": 0.5, "b": 0.9, "c": 0.5, "d": 0.1, "e": 0.9}
    gains = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0, "e": 1.0}
    groups = tie_groups(scores)
    assert groups == [["b", "e"], ["a", "c"], ["d"]]
    # A custom metric can apply group_mean to the same classes: here, nDCG itself.
    ranked = [sum(gains[d] for d in group) / len(group) for group in groups for _ in group]
    assert ndcg(scores, gains, k=5) == pytest.approx(dcg(ranked, 5) / ideal_dcg(gains.values(), 5))
    with pytest.raises(ValueError):
        tie_groups({"a": float("nan")})


def test_item_arrays_reads_item_parameters_of_any_shape_and_refuses_bad_ones() -> None:
    from rcp_ndcg_core.gain import item_arrays
    from rcp_ndcg_core.schemas import ItemParams

    assert item_arrays({"gamma": [1.0, 2.0], "beta": [0.0, 1.0]}) == ([1.0, 2.0], [0.0, 1.0])
    assert item_arrays(ItemParams(gamma=(1.0,), beta=(0.5,))) == ((1.0,), (0.5,))
    with pytest.raises(ValueError, match="as many betas"):
        item_arrays({"gamma": [1.0], "beta": []})
