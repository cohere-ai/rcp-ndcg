"""The IRT numerics of RCP-nDCG: Bradley-Terry, the 2PL calibration and the per-document primitives.

* :func:`fit_bradley_terry` -- abilities from weighted soft pairwise comparisons;
* :func:`fit_calibration` -- the 2PL fit, with the tournament (``mode="tournament"``:
  ``theta = tau_q * theta_BT + alpha_q``) or without it (``mode="rubric_only"``:
  a latent ability), for one judge or several pooled with a severity term;
* :func:`score_document` -- one document's ability from its own rubric answers,
  item parameters frozen (EAP, posterior standard deviation);
* :func:`insert_document` -- one new document's Bradley-Terry ability from its
  comparisons with documents of a frozen tournament fit (conditional MLE).

The estimator classes behind them are available from this package, lazily: :class:`BradleyTerryEstimator`
and :class:`RaschEstimator` (the live tournament refits its Bradley-Terry estimator warm after every batch of
windows, and the calibration's refit needs the standard errors, neither of which the stand-alone
:func:`fit_bradley_terry` offers) import torch when first read, exactly as the stand-alone fits do. Importing
this package itself needs numpy and pydantic only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from rcp_ndcg_core.irt._calibration import (
    AbilityPrior,
    CalibrationFit,
    FitDiagnostics,
    JudgeOverlapError,
    Priors,
    fit_calibration,
)
from rcp_ndcg_core.irt._criteria_2pl import DEFAULT_SE_TARGET
from rcp_ndcg_core.irt._insertion import (
    MIN_OPPONENTS,
    ScaleMovementError,
    UnidentifiableInsertion,
    insert_document,
)
from rcp_ndcg_core.schemas import DocumentEstimate, ItemParams

if TYPE_CHECKING:  # the lazy names below, for the type checkers; the runtime path is __getattr__
    from rcp_ndcg_core.irt._bradley_terry import BradleyTerryEstimator
    from rcp_ndcg_core.irt._rasch import RaschEstimator


def fit_bradley_terry(
    comparisons: Sequence[tuple[str, str, float, float]],
    *,
    l2: float,
    doc_ids: Sequence[str] | None = None,
) -> dict[str, float]:
    """Bradley-Terry abilities (logits, mean zero over ``doc_ids``) from weighted soft comparisons.

    Args:
        comparisons: ``(winner, loser, weight, soft_label)`` tuples, in the order they were observed:
            ``winner`` beat ``loser`` with probability ``soft_label`` in [0, 1], at likelihood weight ``weight`` > 0.
        l2: L2 penalty on the abilities (logits^-2): ``Priors.bt_l2``, the calibration's (``Priors().bt_l2`` is
            the tournament's live fit's).
        doc_ids: The documents to estimate (default: every document in ``comparisons``,
            in order of first appearance); a document without comparisons gets the mean.

    Returns:
        ``{doc_id: theta_BT}``.
    """
    from rcp_ndcg_core.irt._bradley_terry import BradleyTerryEstimator

    rows = [tuple(c) for c in comparisons]
    ids = list(doc_ids) if doc_ids is not None else list(dict.fromkeys(d for row in rows for d in row[:2]))
    estimator = BradleyTerryEstimator(doc_ids=ids, l2_reg=l2)
    for winner, loser, weight, soft_label in rows:
        estimator.add_comparison(str(winner), str(loser), weight=float(weight), soft_label=float(soft_label))
    estimator.fit_lbfgs()
    return estimator.get_scores() or {}


def score_document(
    items: ItemParams,
    placements: int | Sequence[int],
    passes: Sequence[float] | Sequence[Sequence[float]],
    *,
    prior_mean: float = 0.0,
    prior_sd: float = 1.0,
    se_target: float = DEFAULT_SE_TARGET,
    severity: Sequence[float] | None = None,
) -> DocumentEstimate:
    """One document's ability from its own rubric answers, the items frozen.

    Args:
        items: The frozen item parameters.
        placements: How many windows the document was judged in (each answers every criterion);
            with ``severity``, one count per judge.
        passes: Per criterion, in how many of them it passed (``0 <= passes_k <= placements``);
            with ``severity``, one row per judge.
        prior_mean: Mean of the ability prior (logits); the population the score joins.
        prior_sd: Standard deviation of the ability prior (logits).
        se_target: The standard error (logits) the evidence should reach.
        severity: The judges' severities (logits) of a pooled calibration, aligned with
            ``placements`` and ``passes``: judge ``j`` answers criterion ``k`` with logit
            ``gamma_k * (theta - beta_k) - severity_j``. ``None``: one judge, no severity.

    Returns:
        The :class:`~rcp_ndcg_core.schemas.DocumentEstimate`: the EAP ability and the posterior SD on
        the items' scale (``prior_mean`` included), the document's own information over every judge, and
        its flags.

    Raises:
        ValueError: a placement count is not a whole number, a pass count is outside ``[0, placements]``,
            or the per-judge shapes disagree.
    """
    from rcp_ndcg_core.irt._criteria_2pl import Criteria2PL

    # A prior N(mu, sd) on theta is a prior N(0, sd) on theta - mu with every
    # difficulty shifted by mu, so the estimator runs unchanged.
    model = Criteria2PL(np.asarray(items.gamma), np.asarray(items.beta) - prior_mean, prior_sd=prior_sd)
    score = model.score_document(
        placements,
        np.asarray(passes, dtype=float),
        se_target=se_target,
        offsets=None if severity is None else np.asarray(severity, dtype=float),
    )
    return score.model_copy(update={"theta": score.theta + prior_mean})


#: The estimator classes whose module imports torch: available from this package, imported on first read.
def __getattr__(name: str) -> Any:
    """The torch-backed estimator classes, imported when first read (PEP 562), never at package import."""
    if name == "BradleyTerryEstimator":
        from rcp_ndcg_core.irt._bradley_terry import BradleyTerryEstimator

        return BradleyTerryEstimator
    if name == "RaschEstimator":
        from rcp_ndcg_core.irt._rasch import RaschEstimator

        return RaschEstimator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_SE_TARGET",
    "MIN_OPPONENTS",
    "AbilityPrior",
    "BradleyTerryEstimator",
    "CalibrationFit",
    "FitDiagnostics",
    "JudgeOverlapError",
    "Priors",
    "RaschEstimator",
    "ScaleMovementError",
    "UnidentifiableInsertion",
    "fit_bradley_terry",
    "fit_calibration",
    "insert_document",
    "score_document",
]
