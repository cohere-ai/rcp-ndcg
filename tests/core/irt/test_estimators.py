"""Bradley-Terry through the public entry point: order, shrinkage, centring, soft labels."""

from __future__ import annotations

import pytest
from rcp_ndcg_core.irt import fit_bradley_terry

CHAIN = [("a", "b", 1.0, 1.0), ("b", "c", 1.0, 1.0), ("a", "c", 1.0, 1.0)]


def test_abilities_follow_the_comparisons_and_are_centred() -> None:
    scores = fit_bradley_terry(CHAIN * 5, l2=1e-4)
    assert scores["a"] > scores["b"] > scores["c"]
    assert sum(scores.values()) == pytest.approx(0.0, abs=1e-5)


def test_a_larger_penalty_compresses_the_range() -> None:
    low, high = fit_bradley_terry(CHAIN, l2=1e-6), fit_bradley_terry(CHAIN, l2=1.0)
    assert max(high.values()) - min(high.values()) < max(low.values()) - min(low.values())


def test_a_soft_label_of_one_half_is_a_tie() -> None:
    scores = fit_bradley_terry([("a", "b", 1.0, 0.5)] * 4, l2=1e-4)
    assert scores["a"] == pytest.approx(scores["b"], abs=1e-5)


def test_a_document_without_comparisons_sits_at_the_mean() -> None:
    scores = fit_bradley_terry(CHAIN, l2=1e-4, doc_ids=["a", "b", "c", "unseen"])
    assert scores["unseen"] == pytest.approx(sum(scores.values()) / 4, abs=1e-4)
    assert fit_bradley_terry([], l2=1e-4, doc_ids=["a", "b"]) == {"a": 0.0, "b": 0.0}
