"""A document's standard error uses the joint fit's information referent.

Decision core-1(b): a document's standard error is
``1 / sqrt(its own accumulated Fisher information + l2_reg * W)``, where ``W``
is the fit's total comparison weight. The BT fit minimises
``sum(w*BCE)/W + 0.5*l2*||theta||^2``, so in sum units its per-document ridge is
``l2*W``; the insertion (``_insertion.incident_information``) reports the same
referent, so an inserted document's SE is commensurate with the fitted documents'.

Two guards:

* **Referent match** (tight): the estimator's SE equals
  ``1/sqrt(info_own + l2_reg * W)`` evaluated at the fitted parameters; it fails
  under a global denominator or a per-document (``+l2_reg``) ridge.
* **Weight dependence** (hand-computed): the SE follows the joint referent's
  ``l2*W`` term as unrelated comparisons join the fit, and is not the global
  information denominator.
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
def test_bt_se_matches_the_joint_fit_referent(l2_reg: float) -> None:
    for n_pairs in (0, 5, 25):
        est = _bt_fit(n_pairs, l2_reg)
        idx = est.doc_to_idx["d0"]
        winners, losers, weights, _ = est._prepare_observation_tensors()
        p = torch.sigmoid(est.theta[winners].detach() - est.theta[losers].detach())
        contrib = weights.detach() * p * (1 - p)
        info_own = float(contrib[winners == idx].sum() + contrib[losers == idx].sum())
        want = 1.0 / math.sqrt(info_own + est.l2_reg * float(weights.detach().sum()))
        got = _se(est)
        # theta_se is float32, so the referent comparison stops at its precision.
        assert abs(got - want) < 1e-5, (n_pairs, got, want)


def test_bt_se_matches_the_hand_computed_value_at_a_visible_ridge() -> None:
    """The joint ridge is pinned against a hand-computed value, not only the estimator's own formula.

    d0's evidence is 4 balanced comparisons, so its optimum is theta = 0 and its own
    information is 4 * 1 * sigmoid(0) * (1 - sigmoid(0)) = 1. The fit's total weight is
    W = 4. With l2_reg = 1.0 the SE is 1 / sqrt(1 + 1.0*4) = 1/sqrt(5) -- a per-document
    ridge (the old referent) gives 1/sqrt(2), 0.26 away, and a dropped ridge gives 1.0.
    """
    est = _bt_fit(n_padding_pairs=0, l2_reg=1.0)
    got = _se(est)
    assert abs(got - 1.0 / math.sqrt(5.0)) < 1e-3, got


def test_bt_se_follows_the_joint_fit_weight_not_the_global_information() -> None:
    """The joint referent's ridge term is ``l2*W``, so the SE follows W as unrelated balanced pairs
    join the fit while d0's own evidence stays 4 comparisons.

    The hand-computed values pin the referent at W = 4 and W = 104: a per-document ridge held the
    SE flat (the old behaviour) and a global-information denominator would shrink it toward zero.
    """
    small, large = _se(_bt_fit(0, 1.0)), _se(_bt_fit(25, 1.0))
    assert small == pytest.approx(1.0 / math.sqrt(1.0 + 4.0), abs=1e-3)
    assert large == pytest.approx(1.0 / math.sqrt(1.0 + 104.0), abs=1e-3)
