"""Anchors of the calibration fit: the rubric-only refit on public judgements, and the tournament fit's recovery.

**Rubric-only, on released data.** The fixture ``data/calibration_anchor_nanobeir_qwen.json`` holds the first 20
rubric (Stage B) windows of six NanoBEIR queries from the public release
(https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir, ``provenance/judge_criteria``, judge
Qwen3.5-397B), as the release stores them. They become judgement records, so the anchor covers the projection and
:func:`calibrate` together. The rubric-only engine (``Criteria2PL.fit``) runs L-BFGS-B on the exact gradient, so the
fit does not depend on the CPU's rounding; the tolerance absorbs scipy versions.

**Tournament mode, on synthetic data.** The tournament fit also needs each document's Bradley-Terry score, which the
public release does not carry, so its anchor on released data is kept outside the repository. Here the fit runs on
rubric answers drawn from known item and query parameters with the paper's priors: it must recover them, and every
calibrated ability must be its query's affine map ``tau * theta_BT + alpha``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rcp_ndcg_core.irt import fit_calibration

from rcp_ndcg.calibration import calibrate, count_gains

from .conftest import rubric_set

FIXTURE = Path(__file__).parent / "data" / "calibration_anchor_nanobeir_qwen.json"

RUBRIC_ONLY_GAMMA = [1.6495678696710352, 1.289852639837022, 0.723537070788223, 0.9731263134922336, 0.3639161062114862]
RUBRIC_ONLY_BETA = [-4.7186716111726446, -4.504898381938539, -0.6181378437315566, 1.9196043739084772, 7.922103462934263]


def _windows() -> dict:
    """``{(dataset, query_id): [window, ...]}``, a window as ``[(doc_id, None, verdicts), ...]`` in file order."""
    queries = json.loads(FIXTURE.read_text())["queries"]
    return {
        tuple(key.split("||", 1)): [[(row[0], None, row[1:]) for row in window] for window in query["windows"]]
        for key, query in queries.items()
    }


def test_rubric_only_refit_is_pinned() -> None:
    windows = _windows()
    fit = calibrate(rubric_set(windows))

    assert fit.mode == "rubric_only"
    assert list(fit.items.gamma) == pytest.approx(RUBRIC_ONLY_GAMMA, abs=1e-3)
    assert list(fit.items.beta) == pytest.approx(RUBRIC_ONLY_BETA, abs=1e-3)
    assert {(row.dataset, row.query_id) for row in fit.thetas} == set(windows)


def test_count_gains_on_released_rubric_windows() -> None:
    """The Count-nDCG derivation is pinned on released verdicts: the helper's gains equal a count taken straight
    from the fixture's rows (the window is the placement), per document and criterion."""
    raw = json.loads(FIXTURE.read_text())["queries"]
    expected: dict[str, dict[str, float]] = {}
    for key, query in raw.items():
        passes: dict[str, list[int]] = {}
        placements: dict[str, int] = {}
        for window in query["windows"]:
            for row in window:
                doc_id, verdicts = row[0], row[1:]
                counts = passes.setdefault(doc_id, [0] * len(verdicts))
                placements[doc_id] = placements.get(doc_id, 0) + 1
                for index, verdict in enumerate(verdicts):
                    counts[index] += verdict
        expected[key] = {doc_id: sum(counts) / (len(counts) * placements[doc_id]) for doc_id, counts in passes.items()}

    gains = count_gains(rubric_set(_windows()))
    assert set(gains) == set(expected)
    for key, docs in expected.items():
        assert gains[key] == pytest.approx(docs)


GAMMA = np.array([1.3, 1.1, 0.8, 1.0, 0.8])
BETA = np.array([-2.5, -1.5, 0.0, 1.2, 2.8])
TAU = {"q0": 0.6, "q1": 0.9, "q2": 1.2, "q3": 1.5, "q4": 2.0, "q5": 1.0}
ALPHA = {"q0": -1.0, "q1": 0.5, "q2": -0.3, "q3": 1.0, "q4": 0.0, "q5": -2.0}


def _synthetic_tournament_world(seed: int = 7) -> tuple[dict, dict]:
    """Six queries of 40 documents, 12 rubric placements each, answered by a 2PL judge with known parameters."""
    rng = np.random.default_rng(seed)
    observations, bt = {}, {}
    for query in TAU:
        bt[query] = {f"{query}-d{i}": float(theta) for i, theta in enumerate(rng.normal(0.0, 1.2, 40))}
        rows = []
        for doc, theta_bt in bt[query].items():
            passes = 1.0 / (1.0 + np.exp(-GAMMA * (TAU[query] * theta_bt + ALPHA[query] - BETA)))
            for _ in range(12):
                verdicts = rng.random(len(GAMMA)) < passes
                rows.append((doc, {f"C{k + 1}": int(v) for k, v in enumerate(verdicts)}))
        observations[query] = rows
    return observations, bt


def test_the_tournament_fit_recovers_known_parameters_and_maps_each_query_affinely() -> None:
    observations, bt = _synthetic_tournament_world()
    fit = fit_calibration(observations, bt_scores=bt, mode="tournament")

    assert fit.mode == "tournament"
    # The generating parameters satisfy the fit's identification (sum(gamma) = C, mean(beta) = 0).
    assert GAMMA.sum() == pytest.approx(len(GAMMA)) and BETA.mean() == pytest.approx(0.0)
    assert list(fit.items.gamma) == pytest.approx(GAMMA.tolist(), abs=0.15)
    assert list(fit.items.beta) == pytest.approx(BETA.tolist(), abs=0.15)
    for query in TAU:
        tau, alpha = fit.queries[query].tau, fit.queries[query].alpha
        assert tau == pytest.approx(TAU[query], abs=0.2)
        assert alpha == pytest.approx(ALPHA[query], abs=0.2)
        for doc, theta_bt in bt[query].items():
            assert fit.thetas[query][doc] == pytest.approx(tau * theta_bt + alpha, abs=1e-9)
