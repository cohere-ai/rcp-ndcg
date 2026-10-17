"""Reliability of a fit: ECE and Brier, overall, per criterion and per family."""

from __future__ import annotations

import pytest

from rcp_ndcg.calibration import Calibration
from rcp_ndcg.calibration.diagnostics import reliability


def test_known_ece_and_brier() -> None:
    # Bin [0.2, 0.3): predicted 0.25 x4, observed 1 of 4 -> gap 0; bin [0.8, 0.9): 0.8 x2, both pass -> gap 0.2.
    result = reliability([0.25, 0.25, 0.25, 0.25, 0.8, 0.8], [1, 0, 0, 0, 1, 1], bins=10)
    assert result.count == 6
    assert result.ece == pytest.approx(2 / 6 * 0.2)
    assert result.brier == pytest.approx((0.75**2 + 3 * 0.25**2 + 2 * 0.2**2) / 6)
    filled = [b for b in result.bins if b.count]
    assert [(b.lower, b.count) for b in filled] == [(pytest.approx(0.2), 4), (pytest.approx(0.8), 2)]
    assert all(b.mean_predicted is None for b in result.bins if not b.count)


def test_perfect_and_empty() -> None:
    assert reliability([0.0, 1.0, 1.0], [0, 1, 1]).ece == 0.0
    assert reliability([], []).model_dump() == {"count": 0, "ece": None, "brier": None, "bins": []}


def test_calibrate_reports_reliability_per_criterion_and_family(fitted: Calibration) -> None:
    diagnostics = fitted.diagnostics.reliability
    (family,) = [f.key for f in fitted.family_of("rubric")]
    assert set(diagnostics.per_criterion) == set(fitted.items.criteria)
    assert set(diagnostics.per_family) == {family}
    overall = diagnostics.overall
    assert overall.count == 5 * fitted.diagnostics.fit.n_observations
    assert overall.count == diagnostics.per_family[family].overall.count
    # The fake judge answers from the 2PL model, so the fit is well calibrated on its own verdicts.
    assert overall.ece is not None and overall.ece < 0.05
