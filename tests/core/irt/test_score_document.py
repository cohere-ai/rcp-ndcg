"""Scoring one document with the item parameters frozen (:func:`rcp_ndcg_core.irt.score_document`).

The operation scores a document from its own criterion responses with
``(gamma, beta)`` frozen, EAP with a prior (41.83% of real judged items have no
interior MLE, so a bare MLE would pin at the bound for two documents in five).
The scale cannot move because nothing is re-estimated; the tests pin
statelessness, the degeneracy flags, and the precision predicate against the
measured SE curve (1.19/0.84/0.69/0.53/0.42/0.34 at 1/2/3/5/8/12 placements of
K=5).
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest
from rcp_ndcg_core.irt import score_document
from rcp_ndcg_core.irt._criteria_2pl import QUAD, Criteria2PL
from rcp_ndcg_core.schemas import DocumentEstimate, ItemParams

#: Two-judge consensus item parameters of TREC DL -- the items the SE curve above was
#: measured on -- committed inline. Copied verbatim (``repr`` of the parsed floats,
#: which round-trips bit-exactly) from an ``item_params.json`` of 295 bytes (sha256
#: below) that does not ship.
CONSENSUS_ITEMS = {
    "gamma": [1.0160921812057495, 1.1171852350234985, 0.7189441919326782, 1.2219723463058472, 0.92580646276474],
    "beta": [-2.909392833709717, -1.4842638969421387, -1.6929490566253662, 0.3021891415119171, 5.784416675567627],
    "num_criteria": 5,
}
CONSENSUS_ITEMS_SHA256 = "d1b9f4fd798c10fd7c6ec68c9cc016b0ef6c885aef66abdc5462bd1a03c8eb09"
K = 5


def test_inline_consensus_items_are_the_source_file_bytes() -> None:
    """The inline copy re-serialises to the source file's exact bytes (``json.dumps(indent=2)``,
    295 bytes, no trailing newline), so no digit was lost or changed in the copy. Several of the
    tests below are insensitive to a single item parameter, so this is what pins them."""
    assert hashlib.sha256(json.dumps(CONSENSUS_ITEMS, indent=2).encode()).hexdigest() == CONSENSUS_ITEMS_SHA256


ITEMS = ItemParams(gamma=tuple(CONSENSUS_ITEMS["gamma"]), beta=tuple(CONSENSUS_ITEMS["beta"]))


def _score(n, passes, **kwargs) -> DocumentEstimate:
    """The public entry point on the consensus items."""
    return score_document(ITEMS, n, passes, **kwargs)


def consensus_calibrator() -> Criteria2PL:
    items = CONSENSUS_ITEMS
    return Criteria2PL(np.asarray(items["gamma"], float), np.asarray(items["beta"], float))


class TestDegeneracyFlags:
    def test_all_fail_is_finite_eap_with_flag(self) -> None:
        score = _score(1, [0, 0, 0, 0, 0])
        assert isinstance(score, DocumentEstimate)
        assert score.flags.degenerate == "all_fail"
        # The EAP is prior-bounded, not the MLE pin: finite and inside the grid.
        assert np.isfinite(score.theta)
        assert QUAD.min() < score.theta < QUAD.max()

    def test_all_pass_flagged(self) -> None:
        score = _score(2, [2, 2, 2, 2, 2])
        assert score.flags.degenerate == "all_pass"

    def test_interior_pattern_is_not_flagged(self) -> None:
        score = _score(3, [3, 2, 1, 0, 0])
        assert score.flags.degenerate is None

    def test_all_fail_eap_moves_toward_prior_with_more_failures(self) -> None:
        one = _score(1, [0, 0, 0, 0, 0])
        twelve = _score(12, [0, 0, 0, 0, 0])
        # All-fail evidence drags the posterior down; more of it, further down.
        assert twelve.theta < one.theta < 0.0


class TestPrecisionAndSE:
    def test_information_scales_linearly_with_placements(self) -> None:
        one = _score(1, [1, 0, 1, 0, 0])
        two = _score(2, [2, 0, 2, 0, 0])
        # Same response rate: per-placement information is a property of the
        # items and theta. The EAP moves with more data (that is estimation,
        # not scale movement), so the per-placement figure is compared at a
        # tolerance, not bit-identity; at a FIXED theta the scaling is exact
        # (that curve is the measured parity check below).
        assert abs(two.information / 2 - one.information) / one.information < 0.05

    def test_se_falls_with_placements(self) -> None:
        ses = [
            _score(n, [round(n * 0.6), round(n * 0.5), round(n * 0.4), round(n * 0.2), 0]).se
            for n in (1, 2, 3, 5, 8, 12)
        ]
        assert all(ses[i] > ses[i + 1] for i in range(len(ses) - 1)), ses

    def test_likelihood_se_reproduces_the_measured_curve_at_the_measured_theta(self) -> None:
        """Parity with the measured curve: pure-likelihood SE at theta = -2.03 with items fixed.

        The measurement found per-placement Fisher information 0.702 at the criteria-MLE
        theta median (-2.03) under the consensus items: SE
        1.19/0.84/0.69/0.53/0.42/0.34 at 1/2/3/5/8/12 placements. The score's
        ``information`` field is the same quantity evaluated at the EAP, so the
        instrument is compared at a common referent: a document whose EAP sits
        near the median must reproduce the measured per-placement SE within 15%.
        """
        cal = consensus_calibrator()
        info_at = lambda theta: float(  # noqa: E731 - the measurement's formula, inline
            (
                cal.gamma**2
                * (1.0 / (1.0 + np.exp(-(cal.gamma * (theta - cal.beta)))))
                * (1 - 1.0 / (1.0 + np.exp(-(cal.gamma * (theta - cal.beta)))))
            ).sum()
        )
        expected_se_1 = 1.0 / np.sqrt(info_at(-2.03))
        single = _score(1, [1, 0, 1, 0, 1])
        se_likelihood = 1.0 / np.sqrt(single.information)
        assert abs(se_likelihood - expected_se_1) / expected_se_1 < 0.15, (
            f"per-placement SE {se_likelihood:.4f} vs measured curve {expected_se_1:.4f} (referent: theta=-2.03)"
        )
        # At ~0.70 information per placement, SE <= 0.5 needs ceil(4/0.70) = 6 placements
        # (the measured curve gives 0.53 at 5 and 0.42 at 8).
        assert 5 <= math.ceil(1 / 0.5**2 / single.information) <= 8

    def test_precision_predicate_fires_only_when_evidence_certifies_target(self) -> None:
        weak = _score(1, [1, 0, 1, 0, 1], se_target=0.5)
        strong = _score(12, [8, 7, 6, 3, 1], se_target=0.5)
        assert weak.flags.low_information is True
        assert strong.flags.low_information is False


class TestScaleImmobility:
    def test_scoring_one_document_is_bit_identical_whatever_else_was_scored(self) -> None:
        """The scale-cannot-move property, as a state pin.

        A pointwise score is a pure function of (its own responses, frozen
        items, prior). Any cross-document state -- a cache, a running mean, a
        re-anchored frame -- would show up as a bit difference here.
        """
        cal = consensus_calibrator()
        first = cal.score_document(4, [2, 2, 1, 1, 0]).theta
        for n in (1, 2, 5, 9):
            cal.score_document(n, [0, n, 0, n, n if n > 1 else 0])
        again = cal.score_document(4, [2, 2, 1, 1, 0]).theta
        assert again.hex() == first.hex()

    def test_score_does_not_mutate_the_calibrator(self) -> None:
        cal = consensus_calibrator()
        gamma_before = cal.gamma.copy()
        beta_before = cal.beta.copy()
        cal.score_document(2, [1, 1, 0, 0, 0])
        assert np.array_equal(cal.gamma, gamma_before)
        assert np.array_equal(cal.beta, beta_before)


class TestValidation:
    def test_per_criterion_placement_counts_refused(self) -> None:
        # One placement answers every criterion, so the count is one number; a
        # per-criterion count would let "absence is a vote" in.
        with pytest.raises(ValueError, match="one whole number"):
            _score([2, 2, 1, 2, 2], [2, 1, 0, 1, 1])

    def test_s_k_above_n_refused(self) -> None:
        with pytest.raises(ValueError):
            _score(1, [2, 0, 0, 0, 0])

    def test_non_finite_pass_count_refused_not_scored(self) -> None:
        """A NaN window score upstream must refuse here, not return theta=nan as a
        clean-looking DocumentEstimate (degenerate=None, low_information=False)."""
        with pytest.raises(ValueError, match="finite"):
            _score(2, [float("nan"), 1, 1, 1, 1])

    def test_reports_an_se_in_range(self) -> None:
        assert 0 < _score(8, [5, 4, 3, 2, 0]).se < 2.0


class TestSEReferent:
    def test_posterior_se_is_the_posterior_sd_by_independent_quadrature(self) -> None:
        """Pin the reported SE to an explicit recomputation.

        The SE is the one headline number of the score with no other
        test anchor; a scalar bug (a dropped sqrt, a stray factor) would pass
        every range assert. Recomputed here directly from the posterior over
        QUAD, independent of the ``se =`` line it pins.
        """
        cal = consensus_calibrator()
        for n, passes in ((3, [2, 1, 1, 0, 0]), (8, [5, 4, 3, 2, 0]), (1, [0, 0, 0, 0, 0])):
            score = _score(n, passes)
            posterior = cal.posterior(np.asarray([float(n)]), np.asarray([passes], dtype=float))[0]
            theta = float(posterior @ QUAD)
            recomputed = float(np.sqrt(posterior @ ((QUAD - theta) ** 2)))
            assert abs(score.se - recomputed) < 1e-9, (n, score.se, recomputed)
            assert abs(score.theta - theta) < 1e-9

    def test_non_integer_placement_count_refused(self) -> None:
        """A fractional n is a caller bug (n counts placements), not data."""
        with pytest.raises(ValueError, match="integer"):
            _score(2.5, [1, 1, 1, 1, 1])


def test_the_prior_mean_translates_the_score() -> None:
    """A prior centred at m on items shifted by m is the same posterior, moved by m."""
    shift = 1.7
    moved = ItemParams(gamma=ITEMS.gamma, beta=tuple(b + shift for b in ITEMS.beta))
    for n, passes in ((3, [2, 1, 1, 0, 0]), (8, [5, 4, 3, 2, 0])):
        base, other = _score(n, passes), score_document(moved, n, passes, prior_mean=shift)
        assert other.theta == pytest.approx(base.theta + shift, abs=1e-9)
        assert other.se == pytest.approx(base.se, abs=1e-9)


class TestSeveralJudges:
    """A pooled calibration's judges each answer with their severity subtracted from every logit."""

    def test_one_judge_without_severity_is_the_plain_score(self) -> None:
        plain = _score(5, [4, 3, 2, 1, 0])
        grouped = _score([5], [[4, 3, 2, 1, 0]], severity=[0.0])
        assert (grouped.theta, grouped.se, grouped.information) == (plain.theta, plain.se, plain.information)

    def test_a_judges_severity_is_its_difficulties_shifted(self) -> None:
        severity = 0.7
        shifted = ItemParams(
            gamma=ITEMS.gamma, beta=tuple(b + severity / g for g, b in zip(ITEMS.gamma, ITEMS.beta, strict=True))
        )
        alone = score_document(shifted, 5, [4, 3, 2, 1, 0])
        grouped = _score([5], [[4, 3, 2, 1, 0]], severity=[severity])
        assert grouped.theta == pytest.approx(alone.theta, abs=1e-9)
        assert grouped.se == pytest.approx(alone.se, abs=1e-9)

    def test_two_judges_multiply_into_one_posterior(self) -> None:
        """The posterior of both judges' answers, recomputed by independent quadrature."""
        gamma, beta = np.asarray(ITEMS.gamma), np.asarray(ITEMS.beta)
        placements, passes, severity = [3, 4], [[3, 2, 2, 1, 0], [2, 1, 1, 0, 0]], [-0.4, 0.4]
        log_posterior = -0.5 * QUAD**2
        for n, row, s in zip(placements, passes, severity, strict=True):
            p = 1 / (1 + np.exp(-(gamma[:, None] * (QUAD[None, :] - beta[:, None]) - s)))
            log_posterior += (
                np.asarray(row)[:, None] * np.log(p) + (n - np.asarray(row))[:, None] * np.log(1 - p)
            ).sum(0)
        posterior = np.exp(log_posterior - log_posterior.max())
        posterior /= posterior.sum()
        theta = float(posterior @ QUAD)

        score = _score(placements, passes, severity=severity)

        assert score.theta == pytest.approx(theta, abs=1e-9)
        assert score.se == pytest.approx(float(np.sqrt(posterior @ (QUAD - theta) ** 2)), abs=1e-9)

    def test_mismatched_judge_shapes_are_refused(self) -> None:
        with pytest.raises(ValueError, match="one placement count per group"):
            _score([3, 4], [[3, 2, 2, 1, 0]], severity=[0.0, 0.1])
