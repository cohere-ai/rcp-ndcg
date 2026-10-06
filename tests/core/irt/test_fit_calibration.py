"""The calibration fit through its public entry point: determinism, the query scale, the rubric-only convention."""

from __future__ import annotations

import numpy as np
import pytest
from rcp_ndcg_core.irt import Priors, fit_calibration

torch = pytest.importorskip("torch")

GAMMA = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
BETA = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])


def _observations(num_queries: int = 40, num_docs: int = 60, placements: int = 3):
    """Synthetic Bradley-Terry scores and rubric verdicts (36,000 criterion answers)."""
    rng = np.random.default_rng(0)
    bt_scores, verdicts = {}, {}
    for q in range(num_queries):
        query_id = f"q{q:03d}"
        tau, alpha = rng.uniform(0.3, 2.0), rng.normal(0.0, 1.0)
        thetas = rng.normal(0.0, 1.5, num_docs)
        bt_scores[query_id] = {f"d{i:03d}": float(t) for i, t in enumerate(thetas)}
        rows = []
        for i, theta in enumerate(thetas):
            p = 1.0 / (1.0 + np.exp(-GAMMA * (tau * theta + alpha - BETA)))
            for _ in range(placements):
                passed = rng.random(GAMMA.size) < p
                rows.append((f"d{i:03d}", {f"C{c + 1}": int(v) for c, v in enumerate(passed)}))
        verdicts[query_id] = rows
    return bt_scores, verdicts


def _fit(bt_scores, verdicts):
    fit = fit_calibration(verdicts, mode="tournament", bt_scores=bt_scores)
    return fit.items, fit.queries


def test_identical_observations_in_another_order_give_an_identical_fit() -> None:
    """Refits of the same observations once drifted (tau by up to 0.11) with their order."""
    bt_scores, verdicts = _observations()
    reordered = {query_id: list(reversed(verdicts[query_id])) for query_id in reversed(list(verdicts))}

    assert _fit(bt_scores, verdicts) == _fit(bt_scores, reordered)


def test_the_fit_does_not_depend_on_the_thread_count() -> None:
    bt_scores, verdicts = _observations()
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        four = _fit(bt_scores, verdicts)
        assert torch.get_num_threads() == 4, "the fit restores the caller's thread count"
        torch.set_num_threads(1)
        one = _fit(bt_scores, verdicts)
    finally:
        torch.set_num_threads(before)

    assert four == one


def test_the_mode_is_resolved_by_the_caller() -> None:
    bt_scores, verdicts = _observations(num_queries=3, num_docs=20)
    assert fit_calibration(verdicts, mode="tournament", bt_scores=bt_scores).mode == "tournament"
    rubric_only = fit_calibration(verdicts, mode="rubric_only")
    assert rubric_only.mode == "rubric_only"
    assert rubric_only.queries == {}
    with pytest.raises(ValueError, match="needs Bradley-Terry scores"):
        fit_calibration(verdicts, mode="tournament")
    with pytest.raises(ValueError, match="mode must be"):
        fit_calibration(verdicts, mode="auto", bt_scores=bt_scores)  # type: ignore[arg-type]


def test_the_rubric_only_fit_reports_the_common_convention() -> None:
    """sum(gamma) == K and mean(beta) == 0, abilities on that scale with posterior SDs."""
    _, verdicts = _observations(num_queries=10, num_docs=40)
    fit = fit_calibration(verdicts, mode="rubric_only", priors=Priors())
    assert sum(list(fit.items.gamma)) == pytest.approx(5.0)
    assert np.mean(list(fit.items.beta)) == pytest.approx(0.0, abs=1e-12)
    assert set(fit.thetas) == set(verdicts) == set(fit.theta_se)
    assert all(se > 0 for docs in fit.theta_se.values() for se in docs.values())


@pytest.mark.parametrize("mode", ["tournament", "rubric_only"])
def test_a_missing_verdict_is_refused_not_read_as_a_fail(mode: str) -> None:
    bt_scores, verdicts = _observations(num_queries=2, num_docs=5, placements=1)
    doc_id, criteria = verdicts["q000"][0]
    verdicts["q000"][0] = (doc_id, {label: value for label, value in criteria.items() if label != "C3"})
    with pytest.raises(ValueError, match=r"missing \['C3'\]"):
        fit_calibration(verdicts, mode=mode, bt_scores=bt_scores if mode == "tournament" else None)


@pytest.mark.parametrize("mode", ["tournament", "rubric_only"])
def test_a_malformed_row_length_is_refused_not_truncated(mode: str) -> None:
    """``_check_judges`` classified any row of length != 3 as untagged, so a malformed 4-tuple
    slipped past the tagged/untagged refusal and rubric-only mode silently ignored its tail."""
    bt_scores, verdicts = _observations(num_queries=2, num_docs=5, placements=1)
    doc_id, criteria = verdicts["q000"][0]
    verdicts["q000"][0] = (doc_id, criteria, "judge-a", "extra")
    with pytest.raises(ValueError, match="doc_id, criteria"):
        fit_calibration(verdicts, mode=mode, bt_scores=bt_scores if mode == "tournament" else None)


def test_the_tournament_fit_reports_what_it_skipped() -> None:
    """Rows whose document has no BT theta, and whole queries absent from bt_scores, are skipped;
    the count was computed and then discarded unless nothing survived."""
    bt_scores, verdicts = _observations(num_queries=1, num_docs=5, placements=2)
    verdicts["q000"].append(("d_unjudged", {f"C{k + 1}": 0 for k in range(5)}))
    verdicts["q_absent"] = [("d1", {f"C{k + 1}": 1 for k in range(5)})]
    fit = fit_calibration(verdicts, mode="tournament", bt_scores=bt_scores)
    assert fit.diagnostics.n_observations == 10
    assert fit.diagnostics.skipped_observations == 2
    assert fit.diagnostics.skipped_queries == 1
