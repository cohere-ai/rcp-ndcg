"""The rubric-only 2PL estimator: reference values, fitting, guards and gains.

The reference values (theta EAP and the gain at it, for published item
parameters) are the contract any reimplementation must reproduce: if they pass,
gains derived downstream are trustworthy; if they fail, nothing downstream is.
"""

import numpy as np
import pytest
from rcp_ndcg_core.gain import gain
from rcp_ndcg_core.irt._criteria_2pl import (
    DEFAULT_GAMMA_BOUNDS,
    DEFAULT_MAX_ITEM_WEIGHT,
    QUAD,
    Criteria2PL,
    Criteria2PLDiagnostics,
)

QUAD_LO, QUAD_HI = float(QUAD[0]), float(QUAD[-1])

# Published item parameters of one benchmark (five criteria).
ORACLE_GAMMA = np.array(
    [4.404612065692201, 5.532794079706972, 5.146755673246227, 5.447101223049276, 3.4858621173987863]
)
ORACLE_BETA = np.array(
    [1.69811671171171803, 1.8237890998496258, 2.1027943080240308, 2.587050236003074, 3.1642576077827735]
)
ORACLE_ITEMS = {"gamma": ORACLE_GAMMA.tolist(), "beta": ORACLE_BETA.tolist(), "num_criteria": 5}

# (placements, passes per criterion) -> (theta EAP, gain at it), computed
# independently. On the shipped grid (linspace(-18, 10, 561)) every value stays
# inside the tolerances below (max 1.29e-3 on theta, 1.05e-4 on the gain); a
# deviation above 1e-2 on theta or 1e-3 on the gain is a real bug, not quadrature.
ORACLE = {
    (4, (0, 0, 0, 0, 0)): (-0.2258, 0.00004),
    (4, (4, 0, 0, 0, 0)): (1.6287, 0.15532),
    (4, (4, 4, 0, 0, 0)): (1.9886, 0.39513),
    (4, (4, 4, 4, 4, 0)): (2.8290, 0.83425),
    (4, (4, 4, 4, 4, 4)): (3.5975, 0.97267),
}

# Measured effective-theta spans the grid must host: tournament fits reach
# [-16.53, 7.80] (43% of one benchmark's documents outside the old [-5, 5]
# grid), rubric-only fits [-16.2, 8.7].
LIVE_SCALE_LO, LIVE_SCALE_HI = -16.6, 8.8


def _gains(model: Criteria2PL, n: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.array([gain(float(t), model.item_params) for t in model.eap(n, S)[0]])


class TestQuadCoverage:
    """The grid must carry the measured effective-theta scale."""

    def test_quad_covers_the_live_effective_theta_scale(self) -> None:
        assert float(QUAD[0]) <= LIVE_SCALE_LO, (float(QUAD[0]), LIVE_SCALE_LO)
        assert float(QUAD[-1]) >= LIVE_SCALE_HI, (float(QUAD[-1]), LIVE_SCALE_HI)

    def test_quad_resolution_matches_the_reference_density(self) -> None:
        """Scoring on this grid matched a dense [-18, 10] x 1121 reference to
        <= 4.5e-15 on the EAP over 27,875 real documents; the guard pins the
        spacing that achieved it."""
        spacing = float(QUAD[1] - QUAD[0])
        assert spacing <= 0.0501, spacing


class TestOracle:
    def test_theta_and_gain_match_reference_values(self) -> None:
        model = Criteria2PL(ORACLE_GAMMA, ORACLE_BETA)
        for (n, s), (want_theta, want_gain) in ORACLE.items():
            nn, ss = np.array([float(n)]), np.array([s], dtype=float)
            theta = model.eap(nn, ss)[0][0]
            assert abs(theta - want_theta) < 2e-3, (n, s, theta, want_theta)
            assert abs(gain(float(theta), ORACLE_ITEMS) - want_gain) < 2e-4, (n, s)

    def test_item_params_feed_the_metric_gain(self) -> None:
        assert Criteria2PL(ORACLE_GAMMA, ORACLE_BETA).item_params == ORACLE_ITEMS


class TestFit:
    def test_fit_recovers_a_known_difficulty_ordering(self) -> None:
        """Synthetic data with known beta: recovers order AND absolute values.

        Order alone is not enough: a structurally biased likelihood that
        preserves ordering (a sign error in the prior exponent, wrong counts
        weighting) would pass an ordering-only assert while shifting every fit.
        Absolute tolerances: ML noise on 6,000 docs at n=4 is well inside 15%
        (beta) / 25% (gamma); a structural error is not.
        """
        rng = np.random.default_rng(7)
        beta_true = np.array([1.0, 1.6, 2.2, 3.0, 3.8])
        gamma_true = np.array([4.0, 4.5, 5.0, 5.5, 6.0])
        n_docs = 6000
        # Generate under theta ~ N(0, 1) -- matching the fitted prior. A wider
        # generating spread is legitimately shrunk toward the prior scale
        # (the ability scale is pinned by the prior), so recovery is
        # only tight when the model assumption matches the generating process.
        theta = rng.normal(0, 1.0, size=n_docs)
        n = np.full(n_docs, 4.0)
        p = 1.0 / (1.0 + np.exp(-(gamma_true[None, :] * (theta[:, None] - beta_true[None, :]))))
        S = rng.binomial(4, p).astype(float)
        fitted = Criteria2PL.fit(n, S)
        assert list(np.argsort(fitted.beta)) == list(np.argsort(beta_true))
        np.testing.assert_allclose(fitted.beta, beta_true, rtol=0.15)
        np.testing.assert_allclose(fitted.gamma, gamma_true, rtol=0.25)
        assert fitted.diagnostics is not None and not fitted.diagnostics.ordinal_only

    def test_fit_deterministic(self) -> None:
        rng = np.random.default_rng(11)
        n = np.full(2000, 3.0)
        theta = rng.normal(0, 1.5, size=2000)
        beta = np.linspace(1.0, 3.5, 5)
        p = 1.0 / (1.0 + np.exp(-(4.5 * (theta[:, None] - beta[None, :]))))
        S = rng.binomial(3, p).astype(float)
        a = Criteria2PL.fit(n, S)
        b = Criteria2PL.fit(n, S)
        np.testing.assert_allclose(a.beta, b.beta, atol=1e-12)
        np.testing.assert_allclose(a.gamma, b.gamma, atol=1e-12)

    def test_rejects_a_runaway_discrimination(self) -> None:
        """A near-constant criterion must not monopolise the gain (failure mode 1).

        Reproduces a real incident (a benchmark subsample: C4 at pass rate 1e-4
        reached gamma = 7,916 and took 99.8% of the weight, collapsing the gain
        to "P(pass C4)"). The step-function regime needs the rare positive
        to fire in EVERY placement -- with s < n the binomial likelihood
        penalises a step and the fit stays identified.
        """
        rng = np.random.default_rng(3)
        n_docs = 4000
        n = np.full(n_docs, 3.0)
        S = np.zeros((n_docs, 5))
        for k in range(4):
            S[:, k] = rng.binomial(3, 0.05 + 0.05 * k, size=n_docs)
        S[:4, 4] = 3.0  # fires in every placement on 4 of 4,000 docs
        with pytest.raises(ValueError, match="degenerate 2PL fit"):
            Criteria2PL.fit(n, S)

    def test_guard_keeps_a_moderately_rare_item_bounded(self) -> None:
        """Below the runaway regime the guard must NOT fire (no false positive)."""
        rng = np.random.default_rng(3)
        n = np.full(4000, 3.0)
        S = np.zeros((4000, 5))
        for k in range(4):
            S[:, k] = rng.binomial(3, 0.05 + 0.05 * k, size=4000)
        S[:2, 4] = 1.0  # 2/4000: rare, yet the fit stays identified
        fitted = Criteria2PL.fit(n, S)
        d = fitted.diagnostics
        assert d is not None
        assert d.max_item_weight <= DEFAULT_MAX_ITEM_WEIGHT
        assert (fitted.gamma <= DEFAULT_GAMMA_BOUNDS[1]).all()

    @staticmethod
    def _near_degenerate() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(3)
        n = np.full(4000, 3.0)
        S = np.zeros((4000, 5))
        for k in range(4):
            S[:, k] = rng.binomial(3, 0.05 + 0.05 * k, size=4000)
        S[:2, 4] = 1.0
        return n, S

    def test_fit_parity_with_reference_optimizer(self) -> None:
        """A near-degenerate fit (pass-any ~2%) lands on the likelihood's optimum.

        The likelihood of this fixture is nearly flat in beta, so where L-BFGS-B
        stops depends on its path. The fit uses the exact gradient, and the pin
        is its optimum: marginal NLL 17238.707, the dense-grid optimum of this
        fixture, where the earlier finite-difference fit stopped at 17239.26 or
        higher depending on the CPU. Tolerance 1e-3: scipy cross-version noise.
        """
        n, S = self._near_degenerate()
        fitted = Criteria2PL.fit(n, S)
        np.testing.assert_allclose(
            fitted.gamma,
            [0.299802, 0.254694, 0.174329, 0.142243, 2.848666],
            atol=1e-3,
        )
        assert fitted.diagnostics is not None
        assert abs(fitted.diagnostics.max_item_weight - 0.765825) < 1e-3

    def test_fit_is_insensitive_to_last_bit_rounding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A 1e-15 relative change of the sigmoid (another CPU's rounding) must not move the fit.

        A finite-difference gradient divides such differences by its step and
        moved gamma by up to 0.13 on this fixture; the exact gradient moves it by
        less than 1e-9.
        """
        import scipy.special

        n, S = self._near_degenerate()
        reference = Criteria2PL.fit(n, S).gamma
        expit = scipy.special.expit
        monkeypatch.setattr(scipy.special, "expit", lambda z: expit(z) * (1 + 1e-15))
        np.testing.assert_allclose(Criteria2PL.fit(n, S).gamma, reference, atol=1e-6)

    def test_flags_all_saturated_instead_of_raising(self) -> None:
        """Every item pinning at the bound must flag, not raise (failure mode 2).

        The max-weight check is BLIND to this: all-equal gamma gives max weight
        1/K = 0.20, which looks healthy. Observed on two benchmarks (0.4% and 1.4%
        of judged documents pass any criterion), both hitting gamma = 20 on all
        five criteria. The ordering stays usable; only the probability
        reading is void, hence ordinal_only rather than an exception.
        """
        rng = np.random.default_rng(5)
        n_docs = 8000
        n = np.full(n_docs, 3.0)
        # 0.4% of documents pass exactly one criterion once; everything else all-zero.
        S = np.zeros((n_docs, 5))
        for i in rng.choice(n_docs, size=int(0.004 * n_docs), replace=False):
            S[i, int(rng.integers(0, 5))] = 1.0
        fitted = Criteria2PL.fit(n, S)
        d = fitted.diagnostics
        assert d is not None
        assert d.gamma_saturated, f"expected pinned gammas, got {fitted.gamma.round(2)}"
        assert d.ordinal_only
        assert d.reason is not None and "binary" in d.reason
        assert np.isfinite(_gains(fitted, n, S)).all()

    def test_low_pass_rate_flags_identifiability_floor(self) -> None:
        n = np.array([2.0] * 500)
        S = np.zeros((500, 5))
        S[:3] = 1  # 0.6% pass rate, below the 2% floor
        fitted = Criteria2PL.fit(n, S)
        d = fitted.diagnostics
        assert d is not None
        assert d.frac_passing_any < 0.02
        assert d.ordinal_only
        assert "identifiability floor" in (d.reason or "")


class TestValidation:
    def test_gains_are_bounded_and_continuous(self) -> None:
        rng = np.random.default_rng(13)
        n = np.full(500, 4.0)
        theta = rng.normal(0, 1.5, size=500)
        p = 1.0 / (1.0 + np.exp(-(ORACLE_GAMMA[None, :] * (theta[:, None] - ORACLE_BETA[None, :]))))
        S = rng.binomial(4, p).astype(float)
        g = _gains(Criteria2PL(ORACLE_GAMMA, ORACLE_BETA), n, S)
        assert (g >= 0).all() and (g <= 1).all()
        # Gains are a function of discrete (n, S) patterns, so "continuous" means:
        # not confined to the count scale, and never whole numbers (whole-number
        # gain vectors get int-cast by pytrec_eval).
        assert np.unique(g).size > np.unique(S.sum(1)).size
        assert not np.allclose(g, np.round(g))

    def test_the_eap_is_monotone_in_evidence(self) -> None:
        model = Criteria2PL(ORACLE_GAMMA, ORACLE_BETA)
        n = np.array([4.0] * 6)
        S = np.array(
            [
                [0, 0, 0, 0, 0],
                [4, 0, 0, 0, 0],
                [4, 4, 0, 0, 0],
                [4, 4, 4, 0, 0],
                [4, 4, 4, 4, 0],
                [4, 4, 4, 4, 4],
            ],
            dtype=float,
        )
        theta = model.eap(n, S)[0]
        assert (np.diff(theta) > 0).all()

    def test_rejects_bad_shapes_and_values(self) -> None:
        model = Criteria2PL(ORACLE_GAMMA, ORACLE_BETA)
        with pytest.raises(ValueError, match="non-empty 1-D"):
            model.eap(np.array([]), np.zeros((0, 5)))
        with pytest.raises(ValueError, match="0 <= s_k <= n"):
            model.eap(np.array([2.0]), np.array([[3.0, 0, 0, 0, 0]]))[0]
        with pytest.raises(ValueError, match="at least one placement"):
            model.eap(np.array([0.0]), np.zeros((1, 5)))
        with pytest.raises(ValueError, match="at least 2 criteria"):
            Criteria2PL.fit(np.array([2.0, 2.0]), np.array([[1.0], [0.0]]))
        with pytest.raises(ValueError, match="positive"):
            Criteria2PL(np.array([1.0, -1.0, 1.0, 1.0, 1.0]), ORACLE_BETA)

    def test_diagnostics_dataclass_shape(self) -> None:
        n = np.full(100, 4.0)
        S = rng_S()
        fitted = Criteria2PL.fit(n, S)
        d = fitted.diagnostics
        assert isinstance(d, Criteria2PLDiagnostics)
        assert d.n_documents == 100
        assert d.mean_placements == 4.0
        assert 0 < d.max_item_weight <= DEFAULT_MAX_ITEM_WEIGHT
        assert 0 <= d.frac_passing_any <= 1


def rng_S() -> np.ndarray:
    rng = np.random.default_rng(17)
    theta = rng.normal(0, 1.5, size=100)
    p = 1.0 / (1.0 + np.exp(-(ORACLE_GAMMA[None, :] * (theta[:, None] - ORACLE_BETA[None, :]))))
    return rng.binomial(4, p).astype(float)


def test_gamma_bounds_are_the_documented_ones() -> None:
    """The bounds encode the two observed degeneracies; changing them is a decision."""
    assert DEFAULT_GAMMA_BOUNDS == (0.05, 20.0)


class TestCoverageGaps:
    """Cases present in the real data or exposed as parameters, pinned cheaply."""

    def test_single_observation_documents(self) -> None:
        """n=1 documents exist in real pools.

        Gains must stay finite, bounded, and finite-theta: the posterior over a
        single placement is prior-dominated but must not pathologise.
        """
        model = Criteria2PL(ORACLE_GAMMA, ORACLE_BETA)
        n = np.ones(6)
        S = np.array(
            [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]],
            dtype=float,
        )
        gains = _gains(model, n, S)
        theta = model.eap(n, S)[0]
        assert np.isfinite(gains).all() and np.isfinite(theta).all()
        assert (gains >= 0).all() and (gains <= 1).all()
        assert (np.diff(gains) > 0).all()  # monotone in evidence even at n=1

    def test_non_default_criterion_count(self) -> None:
        """K != 5 must fit and stay identified (the API generalises over K)."""
        rng = np.random.default_rng(23)
        n_docs = 3000
        theta = rng.normal(0, 1.5, size=n_docs)
        beta_true = np.array([1.2, 2.1, 3.0])
        gamma_true = np.array([5.0, 6.0, 5.5])
        n = np.full(n_docs, 4.0)
        p = 1.0 / (1.0 + np.exp(-(gamma_true[None, :] * (theta[:, None] - beta_true[None, :]))))
        S = rng.binomial(4, p).astype(float)
        fitted = Criteria2PL.fit(n, S)
        assert fitted.gamma.size == 3
        assert list(np.argsort(fitted.beta)) == [0, 1, 2]
        assert fitted.diagnostics is not None and not fitted.diagnostics.ordinal_only

    def test_extreme_evidence_stays_inside_the_quadrature_grid(self) -> None:
        """All-pass documents with large n pin theta near the posterior edge.

        The EAP must stay strictly inside the shipped QUAD grid ([-18, 10]) and
        never saturate the endpoint, which would silently
        clamp the gain at its boundary value. Under the retired [-5, 5] grid
        this population was truncated (shipped-grid max 4.49 vs 4.80 dense).
        """
        model = Criteria2PL(ORACLE_GAMMA, ORACLE_BETA)
        n = np.array([50.0, 50.0])
        S = np.array([[50, 50, 50, 50, 50], [0, 0, 0, 0, 0]], dtype=float)
        theta = model.eap(n, S)[0]
        assert QUAD_LO < theta[0] < QUAD_HI and QUAD_LO < theta[1] < QUAD_HI
        # Interior with headroom to the endpoints on the shipped grid.
        margin = 0.05 * (QUAD_HI - QUAD_LO)
        assert QUAD_LO + margin < theta[0] < QUAD_HI - margin
        assert QUAD_LO + margin < theta[1] < QUAD_HI - margin
