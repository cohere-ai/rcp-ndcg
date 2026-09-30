"""Tournament insertion: estimate one new document's ability against a frozen Bradley-Terry fit.

When a query gains a candidate, refitting the tournament would move every
published score. Insertion instead estimates the new document's ability by
**conditional MLE**: the existing thetas and comparisons are frozen inputs, so
the estimate lands in the existing fit's frame and the existing thetas are
bit-identical afterwards (checked, not assumed).

Identifiability is checked, not assumed:

* **connected** -- the comparison graph over the existing documents plus the
  new document's comparisons has one component, and the new document faces at
  least ``min_opponents`` distinct opponents. A disconnected insertion would let
  the estimate absorb the component's arbitrary offset.
* **information** -- the new document's own Fisher information,
  ``sum_{c incident to d} w_c p_c (1 - p_c)``, reaches ``1 / se_target**2``.
* **scale** -- the output frame equals the input frame bit for bit
  (:class:`ScaleMovementError` otherwise).

A failed identifiability predicate raises :class:`UnidentifiableInsertion`
carrying the full predicate report (including the standard error that would have
been reported).

Numpy only at import; the 1-D optimiser is scipy's, imported lazily.
"""

from __future__ import annotations

import math
from typing import cast

from rcp_ndcg_core.irt._criteria_2pl import DEFAULT_SE_TARGET
from rcp_ndcg_core.metric import discount
from rcp_ndcg_core.schemas import DocumentEstimate

#: One soft pairwise comparison: ``(winner_id, loser_id, weight, soft_label)``, the
#: grammar the Bradley-Terry estimator ingests.
Comparison = tuple[str, str, float, float]

#: Default bound of the 1-D conditional MAP search (mirrors the BT fits'
#: effective theta range; a genuine optimum outside this box is an
#: identification failure and trips ``at_bound``).
THETA_BOUND = 20.0

#: Distinct existing documents an inserted document must be compared with.
MIN_OPPONENTS = 5


class UnidentifiableInsertion(Exception):
    """The insertion's evidence cannot identify the requested estimate.

    ``.report`` carries the predicate values (including the SE that would have
    been reported) so callers can measure the
    failure instead of guessing at it.
    """

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


class ScaleMovementError(Exception):
    """The existing thetas changed during an insertion: the estimator refitted them instead of freezing them.

    An insertion that moves the existing fit would attach the new document to a
    scale the published artifacts no longer describe.
    """

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


def connected_components(nodes, edges) -> list[set[str]]:
    """Union-find connected components over undirected edges (deterministic order)."""
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a not in parent or b not in parent:
            raise ValueError(f"comparison references unknown document: ({a!r}, {b!r})")
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps: dict[str, set[str]] = {}
    for node in sorted(parent):
        comps.setdefault(find(node), set()).add(node)
    return [comps[root] for root in sorted(comps)]


def incident_information(
    doc_id: str,
    comparisons: list[Comparison],
    theta: dict[str, float],
    l2_reg: float,
) -> float:
    """Fisher information incident to ``doc_id``'s ability, opponents frozen.

    Per weighted soft comparison: ``w * p * (1 - p)`` with
    ``p = sigmoid(theta_doc - theta_other)``. Per-document referent: sums ONLY
    comparisons facing ``doc_id`` -- pairs among opponents inform the
    opponents, not the inserted document.
    Includes the ``+l2_reg`` ridge term the conditional MAP carries.
    """
    info = 0.0
    for a, b, wt, _sl in comparisons:
        if a == doc_id or b == doc_id:
            other = b if a == doc_id else a
            if other not in theta:
                continue
            diff = theta[doc_id] - theta[other]
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, diff))))
            info += wt * p * (1.0 - p)
    return info + l2_reg


def conditional_insert_theta(
    doc_id: str,
    comparisons: list[Comparison],
    theta_fixed: dict[str, float],
    l2_reg: float,
) -> tuple[float, float]:
    """1-D conditional MAP of ``doc_id``'s ability; returns ``(theta, information)``.

    Loss = ``-sum_{c incident to doc} w_c * BCE(z_c, sl_c) + 0.5 * l2 * theta^2``
    where ``z_c = theta - theta_opponent`` -- the same ridge the BT estimator
    applies to every theta, so the conditional estimate is commensurate with
    the frozen ones. Minimised by bounded 1-D optimisation; the returned
    information is the observed information at the optimum (the SE's
    denominator, per-document referent).

    Regression guard baked into this function's contract: when the inserted
    document LOSES a comparison, the soft label flips (``1 - sl``) and the
    logit keeps its orientation -- flipping the logit instead silently turns
    every loss into a win.
    """
    try:
        from scipy.optimize import OptimizeResult, minimize_scalar
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError(
            "conditional_insert_theta requires scipy (bounded 1-D optimisation). "
            "Install it directly, or via the package extra: pip install 'rcp-ndcg-core[irt]'"
        ) from exc

    incident = [(a, b, wt, sl) for a, b, wt, sl in comparisons if a == doc_id or b == doc_id]
    if not incident:
        raise ValueError(f"no comparisons incident to {doc_id!r}: nothing identifies its ability")
    for a, b, _wt, _sl in incident:
        other = b if a == doc_id else a
        if other not in theta_fixed:
            raise ValueError(f"opponent {other!r} has no frozen theta")

    def neg_loss(t: float) -> float:
        loss = 0.5 * l2_reg * t * t
        for a, b, wt, sl in incident:
            if a == doc_id:
                z = t - theta_fixed[b]
            else:
                # doc is the loser: sigma(theta_a - t) = 1 - sigma(t - theta_a),
                # so BCE(theta_a - t, sl) == BCE(t - theta_a, 1 - sl).
                # Flip the TARGET only, never the logit.
                z = t - theta_fixed[a]
                sl = 1.0 - sl
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            p = min(max(p, 1e-9), 1 - 1e-9)
            loss -= wt * (sl * math.log(p) + (1.0 - sl) * math.log(1.0 - p))
        return loss

    # scipy types the return as `object`; the bounded method returns an OptimizeResult.
    res = cast(
        OptimizeResult,
        minimize_scalar(neg_loss, bounds=(-THETA_BOUND, THETA_BOUND), method="bounded", options={"xatol": 1e-7}),
    )
    theta_hat = float(res.x)
    info = incident_information(doc_id, incident, {**theta_fixed, doc_id: theta_hat}, l2_reg)
    return theta_hat, info


def select_opponents(
    doc_id: str,
    provisional_theta: float,
    theta_fixed: dict[str, float],
    candidate_ranks: dict[str, int],
    *,
    k: int = 8,
) -> list[str]:
    """Choose which existing documents the new document should be compared against.

    The kernel is the tournament's adaptive-window value without its novelty
    term (a new document has no comparison history):

    * information: a comparison informs the new document in proportion to
      ``p(1-p)`` at its provisional theta, so opponents near it are the informative ones;
    * metric exposure: ``1/log2(rank + 1)``, the nDCG discount of the opponent's
      current rank;
    * theta span: one candidate per ``k``-quantile bin of the pool, so the
      comparisons bracket the new document's theta instead of clustering.

    Args:
        doc_id: the inserted document (excluded from candidates).
        provisional_theta: the new document's current best guess (e.g. a
            rubric-only EAP); opponents are scored against it.
        theta_fixed: existing thetas (the fit the opponents come from).
        candidate_ranks: ``{doc_id: 1-based rank}`` in the existing fit.
        k: number of opponents to select.

    Returns:
        Opponent doc ids, best-first by kernel score.
    """
    candidates = [d for d in theta_fixed if d != doc_id and d in candidate_ranks]
    if not candidates:
        return []

    def score(d: str) -> float:
        diff = provisional_theta - theta_fixed[d]
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, diff))))
        return p * (1.0 - p) * discount(candidate_ranks[d])

    ordered = sorted(candidates, key=lambda d: theta_fixed[d])
    n_bins = min(k, len(ordered))
    bins: list[list[str]] = [[] for _ in range(n_bins)]
    for i, d in enumerate(ordered):
        bins[min(i * n_bins // len(ordered), n_bins - 1)].append(d)
    picked = [max(bucket, key=score) for bucket in bins if bucket]
    return sorted(picked, key=score, reverse=True)


def _bits_equal(a: dict[str, float], b: dict[str, float]) -> bool:
    """Bit-identical float maps: -0.0 != 0.0, every value compared by bit pattern."""
    return a.keys() == b.keys() and all(a[k].hex() == b[k].hex() for k in a)


def insert_document(
    doc_id: str,
    *,
    new_comparisons: list[Comparison],
    existing_thetas: dict[str, float],
    existing_comparisons: list[Comparison],
    l2_reg: float = 1e-5,
    se_target: float = DEFAULT_SE_TARGET,
    min_opponents: int = MIN_OPPONENTS,
) -> DocumentEstimate:
    """Insert one document into an existing Bradley-Terry fit, conditionally.

    The new document is scored against the frozen existing thetas from its own
    new comparisons; existing thetas are inputs, never outputs, and are
    verified bit-identical after the fit. The estimate is in the existing
    fit's frame -- the same scale as ``existing_thetas`` and any artifact they
    were published in.

    Args:
        doc_id: the document to insert; must not already carry a theta.
        new_comparisons: soft comparisons incident to ``doc_id`` (winner or
            loser). Comparisons NOT incident to ``doc_id`` are refused: they
            are re-judgements of existing documents and belong to a new fit,
            not an insertion.
        existing_thetas: the existing fit's thetas (frozen).
        existing_comparisons: the existing fit's comparisons (for the
            connectivity predicate).
        l2_reg: ridge on the inserted theta, matching the BT estimator's.
        se_target: the SE the evidence must certify (the information predicate).
        min_opponents: distinct opponents required (the connectivity predicate).

    Returns:
        The :class:`~rcp_ndcg_core.schemas.DocumentEstimate` of the inserted document, in the existing fit's
        frame (its own mean-zero anchoring, not re-anchored over the enlarged population); an insertion that
        passes every predicate has no flag set.

    Raises:
        UnidentifiableInsertion: a predicate fails; ``.report`` names the failed ones.
        ScaleMovementError: the existing thetas changed during the fit.
        ValueError: ``doc_id`` already has a theta, or a comparison does not involve it.
    """
    # --- Scale, structural: snapshot the frozen frame NOW and verify it
    # bit-identical after the fit. Comparing the dict to a later copy of
    # itself is the check; comparing it to itself at one instant would be a
    # guard that cannot fire.
    thetas_before = {k: float(v) for k, v in existing_thetas.items()}
    if doc_id in existing_thetas:
        raise ValueError(f"{doc_id!r} already has a theta in the existing fit; insertion is for NEW documents")
    if not new_comparisons:
        raise ValueError("new_comparisons is empty: no evidence about the inserted document")
    non_incident = [(a, b) for a, b, _wt, _sl in new_comparisons if doc_id not in (a, b)]
    if non_incident:
        raise ValueError(
            f"{len(non_incident)} comparison(s) do not involve {doc_id!r}; an insertion scores the new "
            "document against frozen opponents and must not ingest re-judgements of existing documents"
        )

    # --- connectivity: one component after the insertion edges join. ---
    all_nodes = sorted(set(existing_thetas) | {doc_id})
    existing_edges = sorted({tuple(sorted((a, b))) for a, b, _wt, _sl in existing_comparisons})
    new_edges = sorted({tuple(sorted((a, b))) for a, b, _wt, _sl in new_comparisons})
    comps_after = connected_components(all_nodes, existing_edges + new_edges)
    connected = len(comps_after) == 1

    opponents = sorted({b if a == doc_id else a for a, b, _wt, _sl in new_comparisons})
    opponents_ok = len(opponents) >= min_opponents

    # --- conditional MAP with every opponent frozen ---
    theta_hat, info = conditional_insert_theta(doc_id, new_comparisons, existing_thetas, l2_reg)
    se = 1.0 / math.sqrt(max(info, 1e-12))
    at_bound = abs(theta_hat) >= THETA_BOUND - 1e-3
    information_ok = info >= 1.0 / (se_target**2)

    # --- Scale, structural: the fit must not have touched the frozen frame. ---
    if not _bits_equal(thetas_before, existing_thetas):
        raise ScaleMovementError(
            "existing thetas changed during a conditional insertion -- the conditional estimator must "
            "treat opponents as frozen inputs; a fit that rewrites them is a refit-all, not an insertion",
            {
                "doc_id": doc_id,
                "n_moved": sum(1 for k in thetas_before if thetas_before[k].hex() != existing_thetas[k].hex()),
            },
        )

    if not (connected and opponents_ok and information_ok and not at_bound):
        failed = [
            name
            for name, ok in (
                ("connected: one component after insertion", connected),
                ("opponents: >= min_opponents distinct opponents", opponents_ok),
                ("information: own Fisher information >= 1/se_target^2", information_ok),
                ("interior: optimum not at the search bound", not at_bound),
            )
            if not ok
        ]
        raise UnidentifiableInsertion(
            f"insertion of {doc_id!r} is not identifiable: {'; '.join(failed)}. "
            f"Would-be theta {theta_hat:.4f} with SE {se:.4f} from {len(opponents)} opponent(s) / "
            f"{len(new_comparisons)} incident comparison(s) -- refuse, or judge more comparisons.",
            {
                "doc_id": doc_id,
                "theta": theta_hat,
                "se": se,
                "information": info,
                "n_opponents": len(opponents),
                "n_incident_comparisons": len(new_comparisons),
                "n_components_after": len(comps_after),
                "failed_predicates": failed,
            },
        )
    return DocumentEstimate(theta=theta_hat, se=se, information=info)
