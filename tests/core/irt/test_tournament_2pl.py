"""The tournament-mode 2PL: the estimator and :func:`rcp_ndcg_core.irt.fit_calibration` over it."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from rcp_ndcg_core.irt import fit_calibration
from rcp_ndcg_core.irt._tournament_2pl import Tournament2PLCalibrator


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _generate_synthetic_2pl_data(
    true_gamma: np.ndarray,
    true_beta: np.ndarray,
    true_tau: dict[str, float],
    true_alpha: dict[str, float],
    docs_per_query: int = 20,
    reps_per_doc: int = 250,
    rng: np.random.RandomState | None = None,
) -> tuple[list[tuple[str, str, float, dict[str, int]]], dict[str, dict[str, float]]]:
    """Generate synthetic binary observations from known 2PL parameters.

    Returns (observations, bt_scores) where observations is a list of
    (query_id, doc_id, theta_bt, criteria_dict) and bt_scores maps
    {query_id: {doc_id: theta_bt}}.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    K = len(true_gamma)
    observations: list[tuple[str, str, float, dict[str, int]]] = []
    bt_scores: dict[str, dict[str, float]] = {}

    for qid in true_tau:
        tau_j = true_tau[qid]
        alpha_j = true_alpha[qid]
        scores = rng.randn(docs_per_query)
        bt_scores[qid] = {}

        for i in range(docs_per_query):
            theta_bt = float(scores[i])
            doc_id = f"{qid}_doc{i}"
            bt_scores[qid][doc_id] = theta_bt
            cal_theta = tau_j * theta_bt + alpha_j

            for _ in range(reps_per_doc):
                logits = true_gamma * (cal_theta - true_beta)
                probs = _sigmoid(logits)
                y = (rng.rand(K) < probs).astype(int)
                criteria = {f"C{k + 1}": int(y[k]) for k in range(K)}
                observations.append((qid, doc_id, theta_bt, criteria))

    return observations, bt_scores


# ============================================================================
# Test 1: Parameter Recovery on Synthetic Data
# ============================================================================


class TestParameterRecovery:
    """Fit on synthetic data drawn from known 2PL parameters and verify recovery."""

    def test_parameter_recovery_on_synthetic_data(self) -> None:
        rng = np.random.RandomState(42)
        torch.manual_seed(42)

        K = 5
        true_gamma = np.array([1.2, 1.0, 0.8, 0.9, 1.1])
        true_gamma = true_gamma * (K / true_gamma.sum())

        true_beta = np.array([-3.0, -2.0, -0.5, 1.0, 4.0])
        true_beta = true_beta - true_beta.mean()

        true_tau = {"q0": 0.5, "q1": 1.0, "q2": 2.0}
        true_alpha = {"q0": -1.0, "q1": 0.0, "q2": 1.5}

        observations, _ = _generate_synthetic_2pl_data(
            true_gamma=true_gamma,
            true_beta=true_beta,
            true_tau=true_tau,
            true_alpha=true_alpha,
            docs_per_query=20,
            reps_per_doc=250,
            rng=rng,
        )

        cal = Tournament2PLCalibrator(
            num_criteria=K,
            l2_gamma=1e-6,
            l2_beta=1e-6,
            sigma_tau=10.0,
            sigma_alpha=10.0,
        )
        for qid, doc_id, theta_bt, criteria in observations:
            cal.add_observation(qid, doc_id, theta_bt, criteria)
        cal.finalize()
        cal.fit()

        rec_gamma = cal.gamma.detach().cpu().numpy()
        rec_beta = cal.beta.detach().cpu().numpy()
        query_params = cal.get_query_params()

        np.testing.assert_allclose(rec_gamma, true_gamma, atol=0.15)
        np.testing.assert_allclose(rec_beta, true_beta, atol=0.3)

        for qid in true_tau:
            rec_tau = query_params[qid]["tau"]
            rec_alpha = query_params[qid]["alpha"]
            assert abs(rec_tau - true_tau[qid]) < 0.3, f"tau[{qid}]: recovered {rec_tau:.3f} vs true {true_tau[qid]}"
            assert abs(rec_alpha - true_alpha[qid]) < 0.5, (
                f"alpha[{qid}]: recovered {rec_alpha:.3f} vs true {true_alpha[qid]}"
            )


# ============================================================================
# Test 2: Rank Preservation (Invariant)
# ============================================================================


class TestRankPreservation:
    """Calibrated theta must preserve BT score ordering (tau > 0 via softplus)."""

    def test_rank_preservation_invariant(self) -> None:
        rng = np.random.RandomState(123)
        torch.manual_seed(123)

        K = 3
        true_gamma = np.array([1.0, 1.0, 1.0])
        true_beta = np.array([-1.0, 0.0, 1.0])
        true_tau = {"q0": 0.5, "q1": 1.5}
        true_alpha = {"q0": -0.5, "q1": 0.5}

        observations, _ = _generate_synthetic_2pl_data(
            true_gamma=true_gamma,
            true_beta=true_beta,
            true_tau=true_tau,
            true_alpha=true_alpha,
            docs_per_query=10,
            reps_per_doc=100,
            rng=rng,
        )

        cal = Tournament2PLCalibrator(num_criteria=K)
        for qid, doc_id, theta_bt, criteria in observations:
            cal.add_observation(qid, doc_id, theta_bt, criteria)
        cal.finalize()
        cal.fit()

        seen: dict[str, dict[str, float]] = {}
        for qid, doc_id, theta_bt, _ in observations:
            seen.setdefault(qid, {})[doc_id] = theta_bt

        query_params = cal.get_query_params()
        for qid, doc_thetas in seen.items():
            sorted_docs = sorted(doc_thetas.items(), key=lambda x: x[1])
            tau, alpha = query_params[qid]["tau"], query_params[qid]["alpha"]
            cal_thetas = [tau * theta + alpha for _, theta in sorted_docs]
            for i in range(len(cal_thetas) - 1):
                assert cal_thetas[i] < cal_thetas[i + 1], (
                    f"Rank not preserved for {qid}: cal_theta[{i}]={cal_thetas[i]:.4f} "
                    f">= cal_theta[{i + 1}]={cal_thetas[i + 1]:.4f}"
                )


# ============================================================================
# Test 3: fit_calibration in tournament mode
# ============================================================================


class TestFitCalibrationTournament:
    """The public entry point over the estimator."""

    def test_returns_item_and_query_parameters(self) -> None:
        bt_scores: dict[str, dict[str, float]] = {
            "q1": {"d1": 1.5, "d2": -0.3, "d3": 0.8},
            "q2": {"d4": 2.0, "d5": -1.0},
        }
        rubric_observations: dict[str, list[tuple[str, dict[str, int]]]] = {
            "q1": [
                ("d1", {"C1": 1, "C2": 1, "C3": 0}),
                ("d2", {"C1": 0, "C2": 0, "C3": 1}),
                ("d3", {"C1": 1, "C2": 0, "C3": 1}),
            ]
            * 50,
            "q2": [
                ("d4", {"C1": 1, "C2": 1, "C3": 1}),
                ("d5", {"C1": 0, "C2": 0, "C3": 0}),
            ]
            * 50,
        }

        fit = fit_calibration(rubric_observations, mode="tournament", bt_scores=bt_scores, num_criteria=3)

        assert fit.mode == "tournament"
        assert len(list(fit.items.gamma)) == len(list(fit.items.beta)) == 3
        assert fit.items.num_criteria == 3
        assert set(fit.queries) == {"q1", "q2"}
        for qid, docs in bt_scores.items():
            tau, alpha = fit.queries[qid].tau, fit.queries[qid].alpha
            for doc_id, theta_bt in docs.items():
                assert fit.thetas[qid][doc_id] == pytest.approx(tau * theta_bt + alpha)


# ============================================================================
# Test 4: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_empty_rasch_observations_raises(self) -> None:
        bt_scores: dict[str, dict[str, float]] = {"q1": {"d1": 1.0}}
        rubric_observations: dict[str, list[tuple[str, dict[str, int]]]] = {}
        with pytest.raises(ValueError, match="No matching observations"):
            fit_calibration(rubric_observations, mode="tournament", bt_scores=bt_scores, num_criteria=3)

    def test_no_bt_rasch_overlap_raises(self) -> None:
        bt_scores: dict[str, dict[str, float]] = {"q1": {"d1": 1.0}}
        rubric_observations: dict[str, list[tuple[str, dict[str, int]]]] = {
            "q_other": [("d_other", {"C1": 1, "C2": 0, "C3": 1})],
        }
        with pytest.raises(ValueError, match="No matching observations"):
            fit_calibration(rubric_observations, mode="tournament", bt_scores=bt_scores, num_criteria=3)

    def test_single_query_works(self) -> None:
        bt_scores: dict[str, dict[str, float]] = {"q1": {"d1": 1.0, "d2": -1.0}}
        rubric_observations: dict[str, list[tuple[str, dict[str, int]]]] = {
            "q1": [("d1", {"C1": 1, "C2": 1}), ("d2", {"C1": 0, "C2": 0})] * 20,
        }
        fit = fit_calibration(rubric_observations, mode="tournament", bt_scores=bt_scores, num_criteria=2)
        assert len(fit.queries) == 1

    def test_all_observations_zero_converges(self) -> None:
        cal = Tournament2PLCalibrator(num_criteria=2)
        for i in range(20):
            cal.add_observation("q1", f"d{i}", float(i) * 0.1, {"C1": 0, "C2": 0})
        cal.finalize()
        cal.fit()
        assert len(cal.get_item_params()["gamma"]) == 2

    def test_all_observations_one_converges(self) -> None:
        cal = Tournament2PLCalibrator(num_criteria=2)
        for i in range(20):
            cal.add_observation("q1", f"d{i}", float(i) * 0.1, {"C1": 1, "C2": 1})
        cal.finalize()
        cal.fit()
        assert len(cal.get_item_params()["gamma"]) == 2

    def test_mismatched_query_ids_are_skipped(self) -> None:
        bt_scores: dict[str, dict[str, float]] = {"q1": {"d1": 1.0, "d2": -1.0}}
        rubric_observations: dict[str, list[tuple[str, dict[str, int]]]] = {
            "q1": [("d1", {"C1": 1, "C2": 0}), ("d2", {"C1": 0, "C2": 1})] * 20,
            "q_missing": [("dx", {"C1": 1, "C2": 0})] * 10,
        }
        fit = fit_calibration(rubric_observations, mode="tournament", bt_scores=bt_scores, num_criteria=2)
        assert set(fit.queries) == {"q1"}

    def test_add_observation_after_finalize_raises(self) -> None:
        cal = Tournament2PLCalibrator(num_criteria=2)
        cal.add_observation("q1", "d1", 1.0, {"C1": 1, "C2": 0})
        cal.finalize()
        with pytest.raises(RuntimeError, match="Cannot add observations after finalize"):
            cal.add_observation("q1", "d2", 0.5, {"C1": 0, "C2": 1})

    def test_a_non_finite_theta_bt_is_refused_not_fitted(self) -> None:
        """A NaN theta_bt flows straight into the observation tensor and NaNs the whole fit;
        a missing Bradley-Terry score is refused where the observation is added."""
        cal = Tournament2PLCalibrator(num_criteria=2)
        with pytest.raises(ValueError, match="finite"):
            cal.add_observation("q1", "d1", float("nan"), {"C1": 1, "C2": 0})

    def test_finalize_with_no_observations_raises(self) -> None:
        cal = Tournament2PLCalibrator(num_criteria=3)
        with pytest.raises(ValueError, match="No observations"):
            cal.finalize()


# ============================================================================
# Test 5: Constraint Verification
# ============================================================================


class TestGaussianPriorStrength:
    """The Gaussian priors enter the regulariser as squared deviations scaled by 1/(2 sigma^2) times
    the observations' per-query share -- the exponent and the strength are pinned numerically."""

    def test_the_gaussian_priors_enter_the_regulariser_squared(self) -> None:
        cal = Tournament2PLCalibrator(num_criteria=2)  # N = 4 observations, K = 2 criteria
        for query in ("q1", "q2"):
            for doc, row in (("d0", {"C1": 1, "C2": 0}), ("d1", {"C1": 0, "C2": 1})):
                cal.add_observation(query, doc, 0.5, row)
        cal.finalize()
        with torch.no_grad():
            cal.tau_raw.zero_()  # tau = softplus(0) = ln 2, both queries
            cal.alpha_param.copy_(torch.tensor([0.5, -0.5]))
            cal.gamma_raw.zero_()  # the item priors contribute nothing
            cal.beta_raw.zero_()
        # Hand-computed against the shipped formula, with query_scale = 1/(N*K) = 1/8:
        #   tau term    (1/8) * 1/(2 * 1^2) * 2 * (ln 2 - 1)^2 = 0.011769831599788852
        #   alpha term  (1/8) * 1/(2 * 2^2) * (0.5^2 + 0.5^2) = 1/128       = 0.0078125
        expected = (math.log(2.0) - 1.0) ** 2 / 8 + 0.5 / 64
        assert cal.regularization_loss().item() == pytest.approx(expected, rel=1e-4)


class TestConstraintVerification:
    """Verify structural constraints on fitted parameters."""

    @pytest.fixture()
    def fitted_calibrator(self) -> Tournament2PLCalibrator:
        rng = np.random.RandomState(99)
        torch.manual_seed(99)

        K = 4
        cal = Tournament2PLCalibrator(num_criteria=K)
        for q_idx in range(3):
            qid = f"q{q_idx}"
            for d_idx in range(15):
                theta_bt = float(rng.randn())
                criteria = {f"C{k + 1}": int(rng.rand() > 0.5) for k in range(K)}
                cal.add_observation(qid, f"d{q_idx}_{d_idx}", theta_bt, criteria)
        cal.finalize()
        cal.fit()
        return cal

    def test_gamma_always_positive(self, fitted_calibrator: Tournament2PLCalibrator) -> None:
        gamma = fitted_calibrator.gamma.detach().cpu().numpy()
        assert np.all(gamma > 0), f"gamma has non-positive values: {gamma}"

    def test_gamma_sum_equals_num_criteria(self, fitted_calibrator: Tournament2PLCalibrator) -> None:
        gamma_sum = fitted_calibrator.gamma.detach().cpu().sum().item()
        np.testing.assert_allclose(gamma_sum, fitted_calibrator.num_criteria, atol=1e-5)

    def test_beta_is_zero_mean(self, fitted_calibrator: Tournament2PLCalibrator) -> None:
        beta_mean = fitted_calibrator.beta.detach().cpu().mean().item()
        np.testing.assert_allclose(beta_mean, 0.0, atol=1e-6)

    def test_tau_always_positive(self, fitted_calibrator: Tournament2PLCalibrator) -> None:
        tau = fitted_calibrator.tau.detach().cpu().numpy()
        assert np.all(tau > 0), f"tau has non-positive values: {tau}"


# ============================================================================
# Test 6: Loss Decreases
# ============================================================================


class TestLossDecreases:
    """Total loss must decrease after fitting."""

    def test_loss_decreases_after_fit(self) -> None:
        rng = np.random.RandomState(77)
        torch.manual_seed(77)

        K = 3
        cal = Tournament2PLCalibrator(num_criteria=K)
        for q_idx in range(2):
            qid = f"q{q_idx}"
            for d_idx in range(10):
                theta_bt = float(rng.randn())
                criteria = {f"C{k + 1}": int(rng.rand() > 0.4) for k in range(K)}
                cal.add_observation(qid, f"d{q_idx}_{d_idx}", theta_bt, criteria)
        cal.finalize()

        with torch.no_grad():
            initial_loss = (cal.model_loss() + cal.regularization_loss()).item()

        cal.fit()

        with torch.no_grad():
            final_loss = (cal.model_loss() + cal.regularization_loss()).item()

        assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"
