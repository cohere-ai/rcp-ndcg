"""Tournament insertion by conditional MLE (:func:`rcp_ndcg_core.irt.insert_document`).

The flagship property -- existing thetas are bit-identical after a conditional
insertion -- is pinned here: re-injecting a refit-all into ``insert_document``
fails these tests. Identifiability refusals (disconnected graph, leaf evidence,
an optimum at the search bound) are exercised against the shipped refusal paths,
including a leaf fixture whose would-be SE is about 1.86.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from rcp_ndcg_core.irt import UnidentifiableInsertion, fit_bradley_terry, insert_document
from rcp_ndcg_core.irt._insertion import connected_components, select_opponents


def _soft_comparison(
    true_theta: dict[str, float], a: str, b: str, weight: float = 1.0
) -> tuple[str, str, float, float]:
    diff = true_theta[a] - true_theta[b]
    hi, lo = (a, b) if diff >= 0 else (b, a)
    p = 1.0 / (1.0 + math.exp(-abs(diff)))
    return (hi, lo, weight, min(0.99, max(0.01, p)))


def _dense_world(n_docs: int = 30, seed: int = 7, density: float = 0.9):
    """A synthetic connected tournament; every pair compared with prob ``density``."""
    random.seed(seed)
    docs = [f"d{i}" for i in range(n_docs)]
    true = {d: -2.0 + 4.0 * i / (n_docs - 1) for i, d in enumerate(docs)}
    comparisons = [
        _soft_comparison(true, docs[i], docs[j], weight=2.0)
        for i in range(n_docs)
        for j in range(i + 1, n_docs)
        if random.random() < density
    ]
    return docs, true, comparisons


def _fit_bt(doc_ids: list[str], comparisons: list[tuple[str, str, float, float]]) -> dict[str, float]:
    return fit_bradley_terry(comparisons, l2=1e-5, doc_ids=sorted({d for c in comparisons for d in c[:2]}))


def _window_comparisons(true: dict[str, float], window: list[str]) -> list[tuple[str, str, float, float]]:
    """Every pair of a window, the higher true theta winning with a hard label."""
    out = []
    for i, a in enumerate(window):
        for b in window[i + 1 :]:
            hi, lo = (a, b) if true[a] >= true[b] else (b, a)
            out.append((hi, lo, 1.0, 1.0))
    return out


def _paper_world(seed: int = 3) -> tuple[list[str], list[tuple[str, str, float, float]]]:
    """A paper-sized tournament: 100 documents in 136 windows of 10, and X in two of them."""
    rng = random.Random(seed)
    docs = [f"d{i:03d}" for i in range(100)]
    true = {d: rng.gauss(0.0, 1.0) for d in docs}
    true["X"] = 3.0
    comparisons = [c for _ in range(136) for c in _window_comparisons(true, rng.sample(docs, 10))]
    comparisons += [c for _ in range(2) for c in _window_comparisons(true, [*rng.sample(docs, 9), "X"])]
    return docs, comparisons


class TestConditionalInsertion:
    def test_existing_thetas_are_bit_identical_after_insertion(self) -> None:
        """The property the whole design rests on."""
        docs, _true, comparisons = _dense_world()
        doc = "d15"
        existing_comps = [c for c in comparisons if doc not in (c[0], c[1])]
        new_comps = [c for c in comparisons if doc in (c[0], c[1])]
        thetas = _fit_bt([d for d in docs if d != doc], existing_comps)
        frozen = {d: v for d, v in thetas.items()}
        insert_document(doc, new_comparisons=new_comps, existing_thetas=thetas, existing_comparisons=existing_comps)
        # Bit-identical: not approximately equal, not re-anchored, not refitted.
        assert all(thetas[d].hex() == frozen[d].hex() for d in frozen)

    def test_conditional_estimate_agrees_with_refit_all_minus_shift(self) -> None:
        """Cross-check: conditional theta ~ refit-all theta, shift removed."""
        docs, true, comparisons = _dense_world()
        doc = "d15"
        existing_comps = [c for c in comparisons if doc not in (c[0], c[1])]
        new_comps = [c for c in comparisons if doc in (c[0], c[1])]
        thetas = _fit_bt([d for d in docs if d != doc], existing_comps)
        result = insert_document(
            doc, new_comparisons=new_comps, existing_thetas=thetas, existing_comparisons=existing_comps, se_target=2.0
        )
        theta_refit = _fit_bt(docs, comparisons)
        shift = float(np.mean(list(theta_refit.values())) - np.mean(list(thetas.values())))
        refit_minus_shift = theta_refit[doc] - shift
        assert abs(result.theta - refit_minus_shift) < 0.5 * result.se, (
            f"conditional {result.theta:.4f} vs refit-minus-shift {refit_minus_shift:.4f} (SE {result.se:.4f})"
        )
        # And it tracks the generating truth.
        assert abs(result.theta - true[doc]) < 2.0 * result.se

    def test_reported_se_uses_per_document_information(self) -> None:
        """The SE referent is d*'s own incident information, not the fit's size."""
        docs, _true, comparisons = _dense_world()
        doc = "d15"
        existing_comps = [c for c in comparisons if doc not in (c[0], c[1])]
        new_comps = [c for c in comparisons if doc in (c[0], c[1])]
        thetas = _fit_bt([d for d in docs if d != doc], existing_comps)
        result = insert_document(
            doc, new_comparisons=new_comps, existing_thetas=thetas, existing_comparisons=existing_comps
        )
        from rcp_ndcg_core.irt import Priors
        from rcp_ndcg_core.irt._insertion import incident_information

        # Its default ridge is the calibration's Bradley-Terry penalty.
        expected = 1.0 / np.sqrt(incident_information(doc, new_comps, {**thetas, doc: result.theta}, Priors().bt_l2))
        assert abs(result.se - expected) < 1e-9


class TestInformationReferent:
    """The insertion SE's ridge term, pinned against hand-written values (not the function under test)."""

    def test_incident_information_matches_the_hand_written_referent(self) -> None:
        """Fixed thetas, no fit: the referent is ``sum w*p*(1-p)`` over incident comparisons + l2.

        theta_d = theta_a = theta_c = 0 and theta_b = ln(3), so the three incident
        comparisons contribute 2*1/4, 1*3/16 and 4*1/4 = 1.6875, and the non-incident
        pair (a, b) contributes nothing. With l2 = 0.3125 the total is exactly 2.
        """
        from rcp_ndcg_core.irt._insertion import incident_information

        theta = {"d": 0.0, "a": 0.0, "b": math.log(3.0), "c": 0.0}
        comps = [("d", "a", 2.0, 0.9), ("b", "d", 1.0, 0.8), ("c", "d", 4.0, 0.6), ("a", "b", 9.0, 0.5)]
        assert incident_information("d", comps, theta, 0.3125) == pytest.approx(2.0, abs=1e-12)
        assert incident_information("d", comps, theta, 0.0) == pytest.approx(1.6875, abs=1e-12)

    def test_conditional_insert_theta_reports_the_hand_computed_information(self) -> None:
        """One comparison with soft label 0.5: dL/dt = w(p - 0.5) + l2*t has its optimum at t = 0,

        so the returned information is ``w * 1/4 + l2`` = 1.0 -- a dropped ridge returns 0.5.
        """
        from rcp_ndcg_core.irt._insertion import conditional_insert_theta

        theta_hat, info = conditional_insert_theta(
            "d", [("a", "d", 2.0, 0.5)], {"a": 0.0}, l2_reg=0.5, weight_total=1.0
        )
        assert abs(theta_hat) < 1e-6, theta_hat
        assert abs(info - 1.0) < 1e-6, info


class TestConditionalRidge:
    """The conditional loss is in sum units, so its ridge must be the joint fit's ``l2 * W``.

    The BT estimator minimises ``sum(w*BCE)/W + 0.5*l2*||theta||^2``; the conditional loss is
    ``sum_incident(w*BCE) + 0.5*ridge*theta^2``, so the commensurate ridge is ``l2 * W`` with W the
    joint fit's comparison-weight total. A ridge of ``l2`` alone made a two-window document more
    extreme than the joint fit by more than half a logit at paper size.
    """

    def test_the_conditional_ridge_matches_the_joint_fit(self) -> None:
        """At the joint optimum the conditional gradient is zero, so the estimates must agree."""
        from rcp_ndcg_core.irt._insertion import conditional_insert_theta

        docs, comparisons = _paper_world()
        incident = [c for c in comparisons if "X" in c[:2]]
        existing = [c for c in comparisons if "X" not in c[:2]]
        full = fit_bradley_terry(comparisons, l2=1e-4, doc_ids=[*docs, "X"])
        frozen = {d: full[d] for d in docs}
        weight_total = sum(weight for _a, _b, weight, _sl in existing + incident)
        theta_hat, _info = conditional_insert_theta(
            "X", incident, frozen, l2_reg=1e-4, weight_total=weight_total
        )
        assert abs(theta_hat - full["X"]) < 1e-2, (theta_hat, full["X"])

    def test_insert_document_scales_the_ridge_like_the_joint_fit(self) -> None:
        """End to end: with the scaled ridge the two-window document is inserted, not refused."""
        docs, comparisons = _paper_world()
        incident = [c for c in comparisons if "X" in c[:2]]
        existing = [c for c in comparisons if "X" not in c[:2]]
        full = fit_bradley_terry(comparisons, l2=1e-4, doc_ids=[*docs, "X"])
        frozen = {d: full[d] for d in docs}
        result = insert_document(
            "X",
            new_comparisons=incident,
            existing_thetas=frozen,
            existing_comparisons=existing,
            l2_reg=1e-4,
            se_target=2.0,
        )
        assert abs(result.theta - full["X"]) < 1e-2, (result.theta, full["X"])


class TestIdentifiabilityPredicates:
    def test_disconnected_insertion_refused_and_reported(self) -> None:
        docs, true, comparisons = _dense_world(seed=11)
        doc = "d15"
        # Existing world: two halves, no bridge; d*'s comparisons face one half only.
        existing_comps = [c for c in comparisons if doc not in (c[0], c[1])]
        left = docs[:15]
        right = [d for d in docs[15:] if d != doc]
        half_one = [c for c in existing_comps if c[0] in left and c[1] in left]
        half_two = [c for c in existing_comps if c[0] in right and c[1] in right]
        thetas = _fit_bt([d for d in docs if d != doc], half_one + half_two)
        new_comps = [_soft_comparison(true, doc, d) for d in left[::3]]
        with pytest.raises(UnidentifiableInsertion) as excinfo:
            insert_document(
                doc, new_comparisons=new_comps, existing_thetas=thetas, existing_comparisons=half_one + half_two
            )
        assert any(f.startswith("connected:") for f in excinfo.value.report["failed_predicates"])
        assert excinfo.value.report["n_components_after"] == 2

    def test_leaf_insertion_refused_with_the_measured_se(self) -> None:
        """1 opponent / 3 comparisons -> SE ~1.86, refused."""
        docs, true, comparisons = _dense_world(seed=7)
        doc = "d15"
        existing_comps = [c for c in comparisons if doc not in (c[0], c[1])]
        thetas = _fit_bt([d for d in docs if d != doc], existing_comps)
        # One opponent, three independent comparisons (a leaf evidence set).
        opponent = docs[16] if docs[16] != doc else docs[17]
        new_comps = [_soft_comparison(true, doc, opponent, weight=1.0) for _ in range(3)]
        assert len(new_comps) == 3
        with pytest.raises(UnidentifiableInsertion) as excinfo:
            insert_document(doc, new_comparisons=new_comps, existing_thetas=thetas, existing_comparisons=existing_comps)
        report = excinfo.value.report
        assert report["n_opponents"] == 1
        assert report["n_incident_comparisons"] == 3
        # The would-be number carries its SE, behind a refusal (SE in the 1.5-2.2
        # band for one opponent's worth of evidence).
        assert report["se"] > 1.0
        assert "opponents:" in " ".join(report["failed_predicates"])
        assert "information:" in " ".join(report["failed_predicates"])

    def test_bridging_comparisons_make_the_disconnected_case_pass(self) -> None:
        docs, true, comparisons = _dense_world(seed=11)
        doc = "d15"
        existing_comps = [c for c in comparisons if doc not in (c[0], c[1])]
        left = docs[:15]
        right = [d for d in docs[15:] if d != doc]
        half_one = [c for c in existing_comps if c[0] in left and c[1] in left]
        half_two = [c for c in existing_comps if c[0] in right and c[1] in right]
        thetas = _fit_bt([d for d in docs if d != doc], half_one + half_two)
        new_comps = [_soft_comparison(true, doc, d) for d in left[::3]]
        new_comps += [_soft_comparison(true, doc, d) for d in right[::3]]
        result = insert_document(
            doc,
            new_comparisons=new_comps,
            existing_thetas=thetas,
            existing_comparisons=half_one + half_two,
            se_target=1.0,  # ten single comparisons; connectivity is the point here
        )
        assert result.flags.low_information is False and result.se > 0


class TestScaleMovementGuard:
    def test_in_place_mutation_of_frozen_thetas_caught(self) -> None:
        """Structural guard: a caller's frozen frame must survive the call.

        The mutation that turns the estimator into a refit-all is caught by the
        bit-identity test above; here the in-place contract is pinned: whatever
        the primitive does, the dict it was handed is bit-identical afterwards.
        """
        docs, _true, comparisons = _dense_world()
        doc = "d15"
        existing_comps = [c for c in comparisons if doc not in (c[0], c[1])]
        thetas = _fit_bt([d for d in docs if d != doc], existing_comps)
        before = {d: v.hex() for d, v in thetas.items()}
        insert_document(
            doc,
            new_comparisons=[c for c in comparisons if doc in (c[0], c[1])],
            existing_thetas=thetas,
            existing_comparisons=existing_comps,
        )
        assert {d: v.hex() for d, v in thetas.items()} == before


class TestComparisonGrammar:
    def test_losing_document_lands_below_opponent(self) -> None:
        """Label-flip regression (a +1.15-vs--0.36 defect): a document that
        loses every comparison to a stronger opponent must land BELOW it."""
        docs, true, comparisons = _dense_world(seed=5)
        strong = "d29"
        thetas = _fit_bt([d for d in docs if d != "d0"], [c for c in comparisons if "d0" not in (c[0], c[1])])
        # d0 loses three comparisons against the top document.
        new_comps = [_soft_comparison(true, strong, "d0") for _ in range(3)]
        result = insert_document(
            "d0",
            new_comparisons=new_comps,
            existing_thetas=thetas,
            existing_comparisons=[c for c in comparisons if "d0" not in (c[0], c[1])],
            se_target=10.0,  # evidence is intentionally thin; the SE is not the point here
            min_opponents=1,
        )
        assert result.theta < thetas[strong], (result.theta, thetas[strong])

    def test_non_incident_comparisons_refused(self) -> None:
        docs, _true, comparisons = _dense_world()
        thetas = _fit_bt(docs[:10], [c for c in comparisons if "d29" not in (c[0], c[1])])
        with pytest.raises(ValueError, match="do not involve"):
            insert_document(
                "d29",
                new_comparisons=[c for c in comparisons if "d29" not in (c[0], c[1])][:5],
                existing_thetas=thetas,
                existing_comparisons=[],
            )

    def test_a_nonsensical_se_target_is_refused(self) -> None:
        """``se_target`` is squared into the information predicate: 0.0 raised a bare
        ``ZeroDivisionError`` from inside it, NaN made it False so the predicate silently
        refused a well-connected insertion with an unrelated message, and -0.5 / inf silently
        behaved as +0.5 / a disabled predicate."""
        docs, _true, comparisons = _dense_world()
        doc = "d15"
        existing_comps = [c for c in comparisons if doc not in (c[0], c[1])]
        new_comps = [c for c in comparisons if doc in (c[0], c[1])]
        thetas = _fit_bt([d for d in docs if d != doc], existing_comps)
        for se_target in (0.0, -0.5, float("nan"), float("inf")):
            with pytest.raises(ValueError, match="se_target"):
                insert_document(
                    doc,
                    new_comparisons=new_comps,
                    existing_thetas=thetas,
                    existing_comparisons=existing_comps,
                    se_target=se_target,
                )

    def test_inserting_a_document_that_already_has_a_theta_refused(self) -> None:
        docs, _true, comparisons = _dense_world()
        thetas = _fit_bt(docs, comparisons)
        with pytest.raises(ValueError, match="already has a theta"):
            insert_document(
                "d15",
                new_comparisons=[c for c in comparisons if "d15" in (c[0], c[1])][:3],
                existing_thetas=thetas,
                existing_comparisons=[],
            )


class TestEstimatorCorrectness:
    def test_conditional_map_matches_the_analytic_optimum(self) -> None:
        """Pin the returned theta to its own likelihood optimum.

        One opponent, one comparison: the conditional MAP has a closed form,
        theta* = theta_opp + logit(sl) (ridge l2=1e-5 moves it < 1e-4). A
        constant bias in the returned theta (-0.02, say) passes every relative
        check and is caught only here.
        """

        from rcp_ndcg_core.irt._insertion import conditional_insert_theta

        thetas = {"opp": 0.5, "d*": None}
        del thetas["d*"]
        comparisons = [("opp", "d*", 1.0, 0.8)]  # opp beats d* with prob 0.8
        theta_hat, _info = conditional_insert_theta("d*", comparisons, thetas, l2_reg=1e-5, weight_total=1.0)
        expected = 0.5 + math.log(0.2 / 0.8)  # d* loses: logit(1 - sl) below the opponent
        assert abs(theta_hat - expected) < 1e-3, (theta_hat, expected)

    def test_extreme_evidence_reported_at_bound(self) -> None:
        """The interior-optimum check refuses; it is not decoration.

        A document that beats an opponent ALREADY NEAR the search box with
        hard unanimous labels pushes the conditional MAP past the box: the
        ridge creates an interior optimum only while the likelihood's
        exponential pull (measured: theta* ~ theta_opp - log(l2*theta)) can
        balance it, so an opponent near +THETA_BOUND clamps the estimate at
        the box, where the number is pinned, not identified. Must report
        at_bound and refuse."""
        docs, true, comparisons = _dense_world()
        doc = "d15"
        existing_comps = [c for c in comparisons if doc not in (c[0], c[1])]
        thetas = _fit_bt([d for d in docs if d != doc], existing_comps)
        # An opponent pinned near the search box (a degenerate existing fit's
        # top document); d* beats it unanimously with hard labels. The ridge is
        # ``l2 * W`` (W ~ 876 here), so the pull has to be this strong to clamp.
        far = max((d for d in thetas), key=lambda d: thetas[d])
        thetas[far] = 19.0
        new_comps = [(doc, far, 1.0, 1.0) for _ in range(6)]
        with pytest.raises(UnidentifiableInsertion) as excinfo:
            insert_document(
                doc,
                new_comparisons=new_comps,
                existing_thetas=thetas,
                existing_comparisons=existing_comps,
                se_target=100.0,
            )
        report = excinfo.value.report
        assert any(f.startswith("interior:") for f in report["failed_predicates"]), report
        assert report["theta"] > 19.0  # pinned at the +THETA_BOUND search box


class TestConnectivityAndSelection:
    def test_connected_components_union_find(self) -> None:
        comps = connected_components(["a", "b", "c", "d"], [("a", "b"), ("c", "d")])
        assert sorted(sorted(c) for c in comps) == [["a", "b"], ["c", "d"]]
        one = connected_components(["a", "b", "c"], [("a", "b"), ("b", "c")])
        assert len(one) == 1

    def test_select_opponents_prefers_informative_near_theta(self) -> None:
        thetas = {f"d{i}": -3.0 + 0.5 * i for i in range(12)}
        ranks = {d: i + 1 for i, d in enumerate(sorted(thetas, key=thetas.get, reverse=True))}
        picked = select_opponents("d_new", 0.4, thetas, ranks, k=4)
        assert len(picked) == 4
        # Nearest-theta opponents (d7, d8: theta 0.5, 1.0) must be in the pick.
        assert {"d7", "d8"} & set(picked)

    def test_select_opponents_spans_the_theta_range(self) -> None:
        thetas = {f"d{i}": -6.0 + 1.0 * i for i in range(13)}
        ranks = {d: i + 1 for i, d in enumerate(sorted(thetas, key=thetas.get, reverse=True))}
        picked = select_opponents("d_new", 0.0, thetas, ranks, k=6)
        thetas_picked = sorted(thetas[d] for d in picked)
        assert thetas_picked[0] <= -3.0 and thetas_picked[-1] >= 3.0  # spans the pool

    def test_select_opponents_excludes_the_inserted_document(self) -> None:
        thetas = {"d_new": 0.0, **{f"d{i}": -1.0 + 0.2 * i for i in range(10)}}
        candidates = sorted((k for k in thetas if k != "d_new"), key=thetas.get, reverse=True)
        ranks = {d: i + 1 for i, d in enumerate(candidates)}
        picked = select_opponents("d_new", 0.0, thetas, ranks, k=5)
        assert "d_new" not in picked

    def test_select_opponents_returns_best_first_by_kernel_score(self) -> None:
        """The returned order is best-first by the kernel (information x exposure), which the insertion
        consumes: here it is neither the theta order nor the rank order."""
        # p(1-p) at provisional theta 1.0 times 1/log2(rank + 1):
        # near_above 0.124048, far_below 0.104994, way_above 0.022588
        # (the theta order and the rank order are both far_below, near_above, way_above)
        thetas = {"far_below": -1.0, "near_above": 2.0, "way_above": 4.0}
        ranks = {"far_below": 1, "near_above": 2, "way_above": 3}
        picked = select_opponents("new", 1.0, thetas, ranks, k=3)
        assert picked == ["near_above", "far_below", "way_above"]
