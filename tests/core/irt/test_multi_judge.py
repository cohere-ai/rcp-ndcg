"""Multi-judge pooling: judge identity per observation and the per-judge severity term.

Measured facts these tests encode: two real judges agree 92.0% per criterion but
differ systematically in severity (+3.5-4.1pp pass rate,
C1 +9.7pp), so pooling without a severity term biases the shared item
parameters. The severity is centred so ``sum_j n_j s_j = 0`` exactly; a single
tagged judge is structurally the no-severity model (bit-identical fit).
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch
from rcp_ndcg_core.irt import fit_calibration
from rcp_ndcg_core.irt._tournament_2pl import Tournament2PLCalibrator

torch.set_num_threads(1)

GAMMA = [1.2, 0.9, 1.5, 0.7, 1.1]
BETA = [-1.5, -0.5, 0.2, 1.0, 3.5]


def _respond(theta: float, severity: float, rng: random.Random) -> dict[str, int]:
    ps = [1.0 / (1.0 + math.exp(-(g * (theta - b) + severity))) for g, b in zip(GAMMA, BETA, strict=True)]
    return {f"C{k + 1}": 1 if rng.random() < p else 0 for k, p in enumerate(ps)}


def _two_judge_world(*, severity_b: float, n_docs: int = 30, n_queries: int = 8, seed: int = 11):
    """Two judges over one shared item set; judge b passes ``severity_b`` more often (logit shift).

    Rows are ``(query, doc, theta_bt, criteria, judge)``: the true ability doubles as
    the tournament covariate.
    """
    rng = random.Random(seed)
    obs: list[tuple[str, str, float, dict[str, int], str]] = []
    for q in range(n_queries):
        for i in range(n_docs):
            doc = f"d{i}"
            theta = -2.0 + 0.15 * i
            for severity, judge in ((0.0, "judge_a"), (severity_b, "judge_b")):
                obs.append((f"q{q}", doc, theta, _respond(theta, severity, rng), judge))
    return obs


_ROW = {"C1": 1, "C2": 0, "C3": 1, "C4": 0, "C5": 1}


class TestSeverityTerm:
    def test_planted_severity_recovered_with_exact_centering(self) -> None:
        obs = _two_judge_world(severity_b=0.45)
        cal = Tournament2PLCalibrator(num_criteria=5)
        for query_id, doc_id, theta_bt, criteria, judge in obs:
            cal.add_observation(query_id, doc_id, theta_bt, criteria, judge_id=judge)
        cal.finalize()
        cal.fit()
        severity = cal.get_judge_severity()
        assert set(severity) == {"judge_a", "judge_b"}
        # Data sanity: the generator really produced a pass-rate gap (an unlucky
        # seed can wash the planted shift out of the sample -- measured
        # 0.022/5 criteria on one 270-row seed; the fit then faithfully
        # recovers ~0, which is correct behaviour, not a broken instrument).
        mean_passes = {
            j: float(np.mean([sum(c.values()) for _q, _d, _t, c, jj in obs if jj == j])) for j in ("judge_a", "judge_b")
        }
        assert mean_passes["judge_b"] > mean_passes["judge_a"]
        # judge_b passes more often (planted logit shift +0.45) -> NEGATIVE
        # severity under ``logit = gamma*(theta - beta) - s_j``. The direction
        # is the invariant; the magnitude recovers within sampling noise and
        # the attenuation of the per-query offsets, which co-move with severity.
        gap = severity["judge_b"] - severity["judge_a"]
        assert gap < -0.1, (severity, mean_passes)
        assert abs(gap) < 1.8 * 0.45, severity
        # Centring is exact, in observation counts: sum_j n_j * s_j == 0, up to the
        # float32 rounding of the fitted severities (eps times the summed magnitudes).
        counts = {j: sum(1 for o in obs if o[4] == j) for j in ("judge_a", "judge_b")}
        weighted = sum(severity[j] * counts[j] for j in counts)
        rounding = 4 * float(np.finfo(np.float32).eps) * sum(abs(severity[j]) * counts[j] for j in counts)
        assert weighted == pytest.approx(0.0, abs=rounding)

    def test_single_judge_tag_is_bit_identical_to_untagged_fit(self) -> None:
        """P-severity-id's degenerate branch: one judge = today's model exactly."""
        rng = random.Random(11)
        obs = [
            (f"q{q}", f"d{i}", -1.0 + 0.1 * i, _respond(-1.0 + 0.1 * i, 0.0, rng)) for q in range(2) for i in range(15)
        ]

        def _fit(tagged: bool):
            cal = Tournament2PLCalibrator(num_criteria=5)
            for query_id, doc_id, theta_bt, criteria in obs:
                cal.add_observation(query_id, doc_id, theta_bt, criteria, judge_id="judge_a" if tagged else None)
            cal.finalize()
            cal.fit()
            return cal

        tagged_fit, untagged_fit = _fit(True), _fit(False)
        assert tagged_fit.get_judge_severity() == {}
        assert tagged_fit.get_item_params() == untagged_fit.get_item_params()
        a, b = tagged_fit.get_query_params(), untagged_fit.get_query_params()
        for q in a:
            for key in ("tau", "alpha"):
                assert a[q][key].hex() == b[q][key].hex()

    def test_mixing_tagged_and_untagged_refused(self) -> None:
        cal = Tournament2PLCalibrator(num_criteria=5)
        cal.add_observation("q1", "d0", 0.5, _ROW, judge_id="a")
        cal.add_observation("q1", "d1", -0.5, {"C1": 0, "C2": 0, "C3": 1, "C4": 0, "C5": 1})
        with pytest.raises(ValueError, match="judge_id"):
            cal.finalize()

    def test_untagged_fit_has_no_severity_surface(self) -> None:
        cal = Tournament2PLCalibrator(num_criteria=5)
        cal.add_observation("q1", "d0", 0.5, _ROW)
        cal.add_observation("q1", "d1", -0.5, {"C1": 0, "C2": 0, "C3": 1, "C4": 0, "C5": 1})
        cal.finalize()
        cal.fit()
        assert cal.get_judge_severity() == {}

    def test_severity_works_in_covariate_mode(self) -> None:
        rng = random.Random(5)
        cal = Tournament2PLCalibrator(num_criteria=5)
        for q in range(2):
            for i in range(15):
                theta_bt = -1.5 + 0.2 * i
                for severity, judge in ((0.0, "a"), (0.5, "b")):
                    criteria = _respond(theta_bt, severity, rng)  # b passes more
                    cal.add_observation(f"q{q}", f"d{i}", theta_bt, criteria, judge_id=judge)
        cal.finalize()
        cal.fit()
        severity = cal.get_judge_severity()
        # judge b is more lenient: its offset on ``logit = ... - s_j`` is negative.
        assert severity["b"] < severity["a"]


class TestJudgeOverlap:
    def test_disjoint_judges_refused(self) -> None:
        cal = Tournament2PLCalibrator(num_criteria=5)
        cal.add_observation("q1", "d0", 0.5, _ROW, judge_id="a")
        cal.add_observation("q2", "d1", -0.5, {"C1": 0, "C2": 0, "C3": 1, "C4": 0, "C5": 1}, judge_id="b")
        with pytest.raises(ValueError, match="overlap too little"):
            cal.check_judge_overlap(min_shared=1)

    def test_shared_items_pass_and_report_counts(self) -> None:
        cal = Tournament2PLCalibrator(num_criteria=5)
        for q in range(2):
            for i in range(5):
                for judge in ("a", "b"):
                    cal.add_observation(f"q{q}", f"d{i}", 0.1 * i, _ROW, judge_id=judge)
        report = cal.check_judge_overlap(min_shared=5)
        assert report["overlap_ok"] is True
        assert report["pairs"]["a|b"] == 10

    def test_judges_sharing_exactly_one_item_pass_the_default_check(self) -> None:
        """The shipped default threshold is ``MIN_SHARED_ITEMS = 1``: two judges sharing exactly one
        ``(query, doc)`` item are pooled, not refused."""
        cal = Tournament2PLCalibrator(num_criteria=5)
        cal.add_observation("q1", "d0", 0.5, _ROW, judge_id="a")
        cal.add_observation("q1", "d0", -0.5, _ROW, judge_id="b")  # the one shared item
        cal.add_observation("q2", "d1", -0.5, _ROW, judge_id="b")
        report = cal.check_judge_overlap()
        assert report["min_shared"] == 1
        assert report["overlap_ok"] is True and report["pairs"]["a|b"] == 1

    def test_overlap_checkable_before_finalize(self) -> None:
        cal = Tournament2PLCalibrator(num_criteria=5)
        cal.add_observation("q1", "d0", 0.5, _ROW, judge_id="a")
        with pytest.raises(ValueError, match="needs >= 2 tagged judges"):
            cal.check_judge_overlap(min_shared=1)


class TestFitCalibrationJudges:
    ROWS = [
        ("d0", {"C1": 1, "C2": 0, "C3": 1, "C4": 0, "C5": 1}, "judge_a"),
        ("d0", {"C1": 1, "C2": 1, "C3": 1, "C4": 0, "C5": 1}, "judge_b"),
        ("d1", {"C1": 0, "C2": 0, "C3": 1, "C4": 0, "C5": 1}, "judge_a"),
        ("d1", {"C1": 0, "C2": 1, "C3": 1, "C4": 0, "C5": 1}, "judge_b"),
    ]
    BT = {"q1": {"d0": 0.5, "d1": -0.5}}

    def test_pooled_rows_carry_judge_identity(self) -> None:
        fit = fit_calibration({"q1": self.ROWS}, mode="tournament", bt_scores=self.BT, judges="pooled")
        assert set(fit.judge_severity) == {"judge_a", "judge_b"}

    def test_untagged_rows_fit_a_single_judge(self) -> None:
        fit = fit_calibration({"q1": [row[:2] for row in self.ROWS]}, mode="tournament", bt_scores=self.BT)
        assert fit.judge_severity == {}

    def test_the_judges_switch_must_match_the_rows(self) -> None:
        with pytest.raises(ValueError, match='use judges="pooled"'):
            fit_calibration({"q1": self.ROWS}, mode="tournament", bt_scores=self.BT)
        with pytest.raises(ValueError, match="every row tagged"):
            fit_calibration(
                {"q1": [*self.ROWS, self.ROWS[0][:2]]}, mode="tournament", bt_scores=self.BT, judges="pooled"
            )
        with pytest.raises(ValueError, match="at least two judges"):
            fit_calibration({"q1": self.ROWS[::2]}, mode="tournament", bt_scores=self.BT, judges="pooled")

    def test_pooled_judges_need_the_tournament(self) -> None:
        with pytest.raises(ValueError, match="pooled judges need the tournament fit"):
            fit_calibration({"q1": self.ROWS}, mode="rubric_only", judges="pooled")
