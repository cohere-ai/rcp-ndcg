"""Bradley-Terry through the public entry point: order, shrinkage, centring, soft labels."""

from __future__ import annotations

import numpy as np
import pytest
from rcp_ndcg_core.irt import fit_bradley_terry
from rcp_ndcg_core.irt._bradley_terry import BradleyTerryEstimator
from rcp_ndcg_core.irt._rasch import RaschEstimator

torch = pytest.importorskip("torch")

CHAIN = [("a", "b", 1.0, 1.0), ("b", "c", 1.0, 1.0), ("a", "c", 1.0, 1.0)]


def test_abilities_follow_the_comparisons_and_are_centred() -> None:
    scores = fit_bradley_terry(CHAIN * 5, l2=1e-4)
    assert scores["a"] > scores["b"] > scores["c"]
    assert sum(scores.values()) == pytest.approx(0.0, abs=1e-5)


def test_the_bradley_terry_fit_does_not_depend_on_the_thread_count() -> None:
    """Above torch's 32768-element reduction grain the float32 sum order follows the core count.

    The calibration refits Bradley-Terry for every query and calibrated abilities are
    ``tau * theta_bt + alpha``, so a thread-dependent fit moves calibrated numbers across machines.
    """
    rng = np.random.default_rng(0)
    docs = [f"d{i:03d}" for i in range(100)]
    pairs = (rng.choice(len(docs), size=2, replace=False) for _ in range(40000))
    comparisons = [
        (docs[a], docs[b], float(rng.choice([0.5, 1.0])), float(rng.random())) for a, b in pairs
    ]
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        one = fit_bradley_terry(comparisons, l2=1e-4, doc_ids=docs)
        torch.set_num_threads(4)
        four = fit_bradley_terry(comparisons, l2=1e-4, doc_ids=docs)
        assert torch.get_num_threads() == 4, "the fit restores the caller's thread count"
    finally:
        torch.set_num_threads(before)
    assert [one[d].hex() for d in docs] == [four[d].hex() for d in docs]


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


def test_a_comparison_outside_doc_ids_is_refused_not_dropped() -> None:
    """A comparison the estimator cannot attach is evidence it would silently lose: an
    id-format mismatch (chunk id vs doc id) must fail loudly, not fit a weaker model."""
    with pytest.raises(ValueError, match="unknown document"):
        fit_bradley_terry([("a", "b", 1.0, 1.0), ("a", "z", 1.0, 1.0)], l2=1e-4, doc_ids=["a", "b"])
    with pytest.raises(ValueError, match="unknown document"):
        fit_bradley_terry([("z", "a", 1.0, 1.0)], l2=1e-4, doc_ids=["a", "b"])


def test_a_self_comparison_is_refused_not_dropped() -> None:
    with pytest.raises(ValueError, match="itself"):
        fit_bradley_terry([("a", "b", 1.0, 1.0), ("a", "a", 1.0, 1.0)], l2=1e-4, doc_ids=["a", "b"])


def test_a_comparison_weight_and_soft_label_are_validated() -> None:
    for weight in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="weight"):
            fit_bradley_terry([("a", "b", weight, 1.0)], l2=1e-4)
    for soft_label in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="soft_label"):
            fit_bradley_terry([("a", "b", 1.0, soft_label)], l2=1e-4)


def test_an_l2_penalty_that_is_not_finite_and_non_negative_is_refused() -> None:
    """A negative ridge made the fit return all-NaN abilities; nan/inf raised a bare torch
    ``RuntimeError: value cannot be converted to type float without overflow``."""
    for l2 in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="l2_reg must be finite and >= 0"):
            fit_bradley_terry(CHAIN, l2=l2)
    with pytest.raises(ValueError, match="l2_reg must be finite and >= 0"):
        BradleyTerryEstimator(doc_ids=["a", "b"], l2_reg=float("nan"))


def test_the_bare_estimator_default_is_the_documented_bt_ridge() -> None:
    """The paper's Table 2 fixes the Stage A Bradley-Terry ridge at 1e-4 (``docs/concepts/calibration.md``
    and ``docs/concepts/tournament.md`` say so; ``Priors.bt_l2`` is it), but the base class defaulted to
    1e-5: a direct user of the public ``BradleyTerryEstimator`` fitted a 10x weaker ridge. Every product
    call site passes an explicit value (``fit_bradley_terry`` requires one), so no shipped number moves."""
    from rcp_ndcg_core.irt import Priors

    assert BradleyTerryEstimator(doc_ids=["a"]).l2_reg == Priors().bt_l2 == 1e-4
    assert RaschEstimator(doc_ids=["a"], num_criteria=2).l2_reg == 1e-4


def test_duplicate_doc_ids_are_refused_not_silently_collapsed() -> None:
    """``doc_to_idx`` collapses a duplicate to its last index while ``n_docs`` counts every entry,
    leaving an orphan ``theta_raw`` slot pinned at 0 that still enters the mean-centring and the
    ridge. On ``CHAIN*5`` with l2=1e-4, ``doc_ids=["a","b","c","a"]`` gives
    a=6.2751122, c=-6.2751279 against a=6.2750177, c=-6.2750173 for the unique ids -- a silent
    ~1.1e-4 shift. The public records already refuse duplicate ``doc_ids``."""
    with pytest.raises(ValueError, match="duplicate"):
        BradleyTerryEstimator(doc_ids=["a", "b", "a"])
    with pytest.raises(ValueError, match="duplicate"):
        RaschEstimator(doc_ids=["a", "a"], num_criteria=2)
    with pytest.raises(ValueError, match="duplicate"):
        fit_bradley_terry(CHAIN, l2=1e-4, doc_ids=["a", "b", "c", "a"])


def test_rasch_criteria_of_an_unknown_document_are_refused_not_dropped() -> None:
    """One policy for both estimators: an observation that cannot be attached is evidence the model would
    silently lose (a chunk id where a document id belongs fits a weaker stratification)."""
    rasch = RaschEstimator(doc_ids=["a", "b"], num_criteria=2)
    rasch.add_criteria("a", {"C1": 1, "C2": 0})

    with pytest.raises(ValueError, match="unknown document"):
        rasch.add_criteria("a#0", {"C1": 1, "C2": 0})

    assert rasch.observations == [(0, [1, 0])], "the known observation was kept"
    assert rasch.get_scores() is not None
