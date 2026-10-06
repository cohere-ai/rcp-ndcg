"""A document's standard error uses its own information, not the fit's size.

A document's standard error is ``1 / sqrt(its own accumulated Fisher
information + l2_reg)``. It must not change when unrelated documents join the
fit: dividing by the global observation count instead made the same document
with the same evidence report an SE that grew with the fit's size.

Two guards:

* **Referent match** (tight): the estimator's SE equals
  ``1/sqrt(info_own + l2_reg)`` evaluated at the fitted parameters; it fails
  under a global denominator.
* **Fit-size invariance** (bounded): the SE does not inflate as unrelated
  documents join. The residual movement is co-fitting (the mean-reduced loss
  and the shared, co-fitted beta); each bound is the measured residual with
  headroom, an order of magnitude below what a global denominator produces.
"""

from __future__ import annotations

import math

import pytest
import torch
from rcp_ndcg_core.irt._bradley_terry import BradleyTerryEstimator

torch.manual_seed(0)


def _se(est, doc_id: str = "d0") -> float:
    idx = est.doc_to_idx[doc_id]
    assert est.theta_se is not None, "theta_se must be computed by fit_lbfgs"
    return float(est.theta_se[idx].item())


# ---------------------------------------------------------------------------
# Bradley-Terry
# ---------------------------------------------------------------------------


def _bt_fit(n_padding_pairs: int, l2_reg: float) -> BradleyTerryEstimator:
    """d0 vs d1 balanced (2 wins each way); padding docs form independent
    balanced pairs. Every pair is symmetric, so d0's optimum is the same
    interior point (theta = 0) at every fit size."""
    docs = ["d0", "d1"]
    for i in range(n_padding_pairs):
        docs += [f"a{i}", f"b{i}"]
    est = BradleyTerryEstimator(doc_ids=docs, l2_reg=l2_reg)
    for _ in range(2):
        est.add_comparison("d0", "d1")
        est.add_comparison("d1", "d0")
    for i in range(n_padding_pairs):
        for _ in range(2):
            est.add_comparison(f"a{i}", f"b{i}")
            est.add_comparison(f"b{i}", f"a{i}")
    est.fit_lbfgs()
    return est


@pytest.mark.parametrize("l2_reg", [1e-4, 1.0])
def test_bt_se_matches_per_document_referent(l2_reg: float) -> None:
    for n_pairs in (0, 5, 25):
        est = _bt_fit(n_pairs, l2_reg)
        idx = est.doc_to_idx["d0"]
        winners, losers, weights, _ = est._prepare_observation_tensors()
        p = torch.sigmoid(est.theta[winners].detach() - est.theta[losers].detach())
        contrib = weights.detach() * p * (1 - p)
        info_own = float(contrib[winners == idx].sum() + contrib[losers == idx].sum())
        want = 1.0 / math.sqrt(info_own + est.l2_reg)
        got = _se(est)
        # theta_se is float32, so the referent comparison stops at its precision.
        assert abs(got - want) < 1e-5, (n_pairs, got, want)


def test_bt_se_matches_the_hand_computed_value_at_a_visible_ridge() -> None:
    """The ridge is pinned against a hand-computed value, not only the estimator's own formula.

    d0's evidence is 4 balanced comparisons, so its optimum is theta = 0 and its own
    information is 4 * 1 * sigmoid(0) * (1 - sigmoid(0)) = 1. With l2_reg = 1.0 the SE is
    1 / sqrt(2) -- a dropped ridge (SE 1.0) is 0.29 away, far outside the tolerance.
    """
    est = _bt_fit(n_padding_pairs=0, l2_reg=1.0)
    got = _se(est)
    assert abs(got - 1.0 / math.sqrt(2.0)) < 1e-3, got


def test_bt_se_invariant_to_unrelated_comparisons() -> None:
    """d0's evidence is 4 balanced comparisons throughout. A global
    weight denominator grew 4 -> 104 and would inflate the SE ~3.1x at n=25."""
    ses = [_se(_bt_fit(n, 1e-4)) for n in (0, 5, 25)]
    assert max(ses) - min(ses) < 1e-3, ses
