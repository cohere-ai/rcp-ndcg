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
        from rcp_ndcg_core.irt._insertion import incident_information

        expected = 1.0 / np.sqrt(incident_information(doc, new_comps, {**thetas, doc: result.theta}, 1e-5))
        assert abs(result.se - expected) < 1e-9


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
        theta_hat, _info = conditional_insert_theta("d*", comparisons, thetas, l2_reg=1e-5)
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
        # top document); d* beats it unanimously with hard labels.
        far = max((d for d in thetas), key=lambda d: thetas[d])
        thetas[far] = 15.0
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
