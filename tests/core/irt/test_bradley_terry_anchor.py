"""The Bradley-Terry loss has an external anchor: an independent minimiser of the documented objective.

The refit rule test compares ``bradley_terry`` against ``fit_bradley_terry``, which runs the same estimator, so
a change to the loss or its ridge would move every re-judged pool with no check noticing. Here the documented
loss is minimised again by scipy in float64 -- not by ``BradleyTerryEstimator`` -- and the estimator's fit must
agree with it, and with the pinned values, within :data:`TOLERANCE` (logits).

The documented loss (``_bradley_terry.py``): the weight-normalised binary cross-entropy of
``sigmoid(theta_winner - theta_loser)`` against the soft label, plus ``0.5 * l2 * sum(theta_raw**2)``.
"""

from __future__ import annotations

import numpy as np
import pytest
from rcp_ndcg_core.irt._bradley_terry import BradleyTerryEstimator
from scipy.optimize import minimize

DOCS = ("a", "b", "c", "d")
#: ``(winner, loser, weight, soft_label)``: one-sided, balanced and low-weight evidence.
OBSERVATIONS = (
    ("a", "b", 1.0, 0.9),
    ("a", "c", 0.5, 0.7),
    ("b", "c", 2.0, 0.6),
    ("c", "d", 1.0, 0.8),
    ("b", "d", 1.0, 0.4),
    ("a", "d", 1.0, 0.95),
)
L2 = 1e-3

#: The estimator's fit, pinned: a loss change (the ridge's normalisation, the label handling) moves it.
EXPECTED = {"a": 1.451788, "b": -0.367212, "c": -0.275314, "d": -0.809262}

#: The estimator fits in float32 by L-BFGS with its own stopping rule; the independent float64 minimiser
#: converges tighter, so the two stop within a few 1e-5 of each other on this fit.
TOLERANCE = 1e-4


def _bce(theta: np.ndarray) -> float:
    """The weight-normalised soft BCE of the documented loss (the likelihood part, shift-invariant)."""
    total = 0.0
    weight_sum = sum(weight for _w, _l, weight, _t in OBSERVATIONS)
    for winner, loser, weight, label in OBSERVATIONS:
        z = float(theta[DOCS.index(winner)] - theta[DOCS.index(loser)])
        # BCEWithLogits(label, z) = label * softplus(-z) + (1 - label) * softplus(z), softplus via logaddexp.
        total += weight * (label * float(np.logaddexp(0.0, -z)) + (1.0 - label) * float(np.logaddexp(0.0, z)))
    return total / weight_sum


def _loss(theta: np.ndarray, l2: float = L2) -> float:
    """The documented objective, written out: weight-normalised soft BCE plus the ridge on the raw abilities."""
    return _bce(theta) + 0.5 * l2 * float(np.sum(theta**2))


def _independent_minimiser() -> dict[str, float]:
    result = minimize(_loss, np.zeros(len(DOCS)), method="L-BFGS-B", options={"ftol": 1e-15, "gtol": 1e-12})
    assert result.success, result.message
    return dict(zip(DOCS, (float(value) for value in result.x), strict=True))


def _estimator_fit() -> dict[str, float]:
    estimator = BradleyTerryEstimator(doc_ids=list(DOCS), l2_reg=L2)
    for winner, loser, weight, label in OBSERVATIONS:
        estimator.add_comparison(winner, loser, weight=weight, soft_label=label)
    estimator.fit_lbfgs()
    scores = estimator.get_scores()
    assert scores is not None
    return scores


def test_the_estimator_agrees_with_an_independent_minimiser_of_the_documented_loss() -> None:
    independent = _independent_minimiser()
    fitted = _estimator_fit()
    assert fitted == pytest.approx(independent, abs=TOLERANCE)


def test_the_fit_is_pinned_to_fixed_values() -> None:
    assert _estimator_fit() == pytest.approx(EXPECTED, abs=TOLERANCE)


def test_the_loss_is_shift_invariant_and_the_ridge_pins_the_mean() -> None:
    """The documented loss depends on differences only, so the ridge's optimum has mean zero: a shift of every
    raw ability leaves the likelihood unchanged and only the ridge notices it."""
    independent = _independent_minimiser()
    shifted = {doc: value + 3.0 for doc, value in independent.items()}
    raw = np.array([independent[doc] for doc in DOCS])
    moved = np.array([shifted[doc] for doc in DOCS])
    assert _bce(raw) == pytest.approx(_bce(moved), abs=1e-6)
    assert _loss(moved) > _loss(raw) + 1e-3  # the ridge alone moves: 0.5 * 1e-3 * (36 + 6 * sum(raw))
    assert sum(independent.values()) == pytest.approx(0.0, abs=1e-6)
