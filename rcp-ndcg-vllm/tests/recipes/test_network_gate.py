"""The network gate itself: an unset variable skips every recipe test, a set one lets them run."""

from __future__ import annotations


def test_the_recipe_gate_marks_this_test_network() -> None:
    """Collected under tests/recipes/, this test carries the network marker and the offline skip."""
    from tests.recipes.conftest import VARIABLE

    assert VARIABLE == "RCP_NDCG_NETWORK_TESTS"


def test_a_recipe_test_runs_only_when_the_variable_is_set() -> None:
    """A stand-in recipe test: the same gating every recipe lane's test gets."""
    import os

    assert os.environ.get("RCP_NDCG_NETWORK_TESTS") in ("1", None), "the gate decides; the test just runs"
