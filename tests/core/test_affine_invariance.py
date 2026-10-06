"""What a per-query affine rescaling of theta does and does not change.

2PL calibration is identified only up to a per-query affine map
`tau*theta + alpha` with `tau > 0` (see the calibration concept page,
`docs/concepts/calibration.md`), so any comparison of two
calibration fits inherits two consequences that are easy to conflate:

* Any **within-query rank** statistic (concordance, per-query Spearman) is
  *pinned* -- it cannot distinguish two fits that differ only by the affine map.
* **RCP-nDCG is not pinned**, because its gain is non-linear in theta and is
  evaluated against global criterion difficulties.

If a future change makes RCP-nDCG rank-only (or makes calibration non-affine),
these tests should fail loudly rather than silently change what the metric can
discriminate.
"""

from __future__ import annotations

import numpy as np
import pytest
from rcp_ndcg_core import gain, ndcg
from rcp_ndcg_core.metric import rank_by_score
from scipy import stats

ITEM_PARAMS = {
    "gamma": [1.169, 1.047, 0.940, 1.157, 0.687],
    "beta": [-2.572, -2.002, -0.251, 0.709, 4.116],
}

DOCS = [f"d{i}" for i in range(12)]
BASE_THETA = {
    "d0": 2.4,
    "d1": -0.7,
    "d2": 1.1,
    "d3": -3.2,
    "d4": 0.3,
    "d5": 4.9,
    "d6": -1.8,
    "d7": 0.0,
    "d8": 2.2,
    "d9": -0.1,
    "d10": 3.6,
    "d11": -2.5,
}

# (tau, alpha) pairs a Stage-B refit could plausibly produce.
AFFINE_CASES = [(1.0, 0.0), (2.0, 0.0), (0.35, 1.5), (1.7, -4.0), (0.1, 8.0)]


def _rescale(theta: dict[str, float], tau: float, alpha: float) -> dict[str, float]:
    return {d: tau * v + alpha for d, v in theta.items()}


def rcp_ndcg(ranking: list[str], thetas: dict[str, float], items: dict, *, k: int) -> float:
    return ndcg(ranking, {d: gain(t, items) for d, t in thetas.items()}, k=k)


@pytest.mark.parametrize(("tau", "alpha"), AFFINE_CASES)
def test_a_within_query_ordering_is_pinned(tau: float, alpha: float) -> None:
    """A positive affine map cannot reorder documents inside a query -- through the metric's own
    ordering (``rank_by_score``, the ``doc_id_desc`` tie rule) and through a per-query Spearman.

    This is the statement any within-query rank statistic inherits: two fits that differ only by
    the map are indistinguishable by it (zero power), so the discrimination must come from the
    gain, which the next test pins.
    """
    rescaled = _rescale(BASE_THETA, tau, alpha)
    a = np.array([BASE_THETA[d] for d in DOCS])
    b = np.array([rescaled[d] for d in DOCS])
    assert stats.spearmanr(a, b).statistic == pytest.approx(1.0, abs=1e-12)
    # The metric's own ordering (rank_by_score backs the doc_id_desc tie rule) agrees on the two fits.
    assert rank_by_score(BASE_THETA) == rank_by_score(rescaled)


def test_rcp_ndcg_is_not_pinned() -> None:
    """RCP-nDCG *does* move under rescaling -- this is what keeps the outcome live.

    Gain is non-linear in theta, so shifting the scale changes how much credit the
    top-k documents earn even though their order is fixed.
    """
    ranking = DOCS[:10]
    base = rcp_ndcg(ranking, BASE_THETA, ITEM_PARAMS, k=10)
    moved = [
        rcp_ndcg(ranking, _rescale(BASE_THETA, tau, alpha), ITEM_PARAMS, k=10)
        for tau, alpha in AFFINE_CASES
        if (tau, alpha) != (1.0, 0.0)
    ]
    assert any(abs(m - base) > 1e-6 for m in moved), (
        "RCP-nDCG was invariant to affine rescaling; it would then carry no more "
        "information than a within-query rank statistic"
    )


def test_identity_transform_changes_nothing() -> None:
    ranking = DOCS[:10]
    assert rcp_ndcg(ranking, _rescale(BASE_THETA, 1.0, 0.0), ITEM_PARAMS, k=10) == pytest.approx(
        rcp_ndcg(ranking, BASE_THETA, ITEM_PARAMS, k=10)
    )
