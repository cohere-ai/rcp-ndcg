"""The 2PL calibration fit: one function, two modes, one or several judges.

:func:`fit_calibration` is the single fitting entry point. Its two switches:

* ``mode`` -- ``"tournament"``: the Bradley-Terry theta of Stage A is a frozen
  covariate and every query gets an affine map onto the common scale,
  ``P(C_k = 1) = sigma(gamma_k * (tau_q * theta_BT - beta_k + alpha_q))``; the
  calibrated ability is ``tau_q * theta_BT + alpha_q``. ``"rubric_only"``: no
  tournament; ``P(C_k = 1) = sigma(gamma_k * (theta - beta_k))`` with
  ``theta ~ N(0, 1)``, fitted by marginal maximum likelihood on each document's
  placement and pass counts, and the calibrated ability is the posterior mean.
  ``rcp_ndcg.calibration.calibrate`` resolves its ``"auto"`` from the judgements
  before calling this.
* ``judges`` -- ``"single"``: one judge, untagged observations. ``"pooled"``:
  observations tagged ``(doc_id, criteria, judge_id)`` from two or more judges on
  one rubric; the fit adds one additive severity per judge on the logit, centred on
  the observation-weighted mean, and reports it in ``item_params["judge_severity"]``.

Both modes report item parameters in one convention: ``sum(gamma) == K`` and
``mean(beta) == 0``. Gains are invariant to that choice
(``gamma_k * (theta - beta_k)`` is unchanged when theta, beta and gamma are
rescaled together), so :func:`rcp_ndcg_core.gain.gain` applies to the
calibrated abilities of either mode unchanged.

The inputs are projected observations; ``rcp_ndcg.calibration.calibrate`` builds
them from judgement records. The tournament engine needs torch (imported lazily);
the rubric-only engine needs numpy and scipy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rcp_ndcg_core.schemas import ItemParams, QueryParams, items_from_mapping

FittedMode = Literal["tournament", "rubric_only"]
Judges = Literal["single", "pooled"]

#: One Stage B observation: ``(doc_id, {"C1": 0|1, ...})``, or for a pooled fit
#: ``(doc_id, {...}, judge_id)``. One entry per document per judged window.
Observation = tuple[str, Mapping[str, int]] | tuple[str, Mapping[str, int], str]


#: The fewest ``(query, doc)`` items every pair of pooled judges must share: a judge who shares none with the
#: others adds a severity but nothing that links it to their scale.
MIN_SHARED_ITEMS = 1


class JudgeOverlapError(ValueError):
    """Pooled judges share too few ``(query, doc)`` items to be put on one scale.

    ``.report`` holds ``min_shared``, ``pairs`` (``{"a|b": shared items}``) and ``items_per_judge``.
    """

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


class Priors(BaseModel):
    """The priors and penalties of the calibration fit; the defaults are the paper's.

    Attributes:
        sigma_tau: Prior standard deviation of ``tau_q - 1`` (tournament mode; dimensionless).
        sigma_alpha: Prior standard deviation of ``alpha_q`` (tournament mode; logits).
        l2_gamma: L2 penalty on ``log gamma_k`` (tournament mode).
        l2_beta: L2 penalty on ``beta_k`` (tournament mode; logits^-2).
        ability_sd: Prior standard deviation of the latent ability (rubric-only mode; logits).
        bt_l2: L2 penalty of the Bradley-Terry abilities the tournament mode refits from the tournament's
            windows (logits^-2). The default is the paper's, which the tournament's live fit uses while
            judging; the judgement store records it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    sigma_tau: float = Field(default=1.0, gt=0)
    sigma_alpha: float = Field(default=2.0, gt=0)
    l2_gamma: float = Field(default=1e-4, ge=0)
    l2_beta: float = Field(default=1e-4, ge=0)
    ability_sd: float = Field(default=1.0, gt=0)
    bt_l2: float = Field(default=1e-4, ge=0)


class AbilityPrior(BaseModel):
    """The ability prior of a rubric-only fit, on the reported scale (logits)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: float
    sd: float


class FitDiagnostics(BaseModel):
    """The fit's own summary.

    Attributes:
        n_observations: Rubric placements fitted.
        n_queries: Queries with a fitted scale map (tournament mode).
        n_documents: Documents fitted (rubric-only mode).
        mean_placements: Placements per document (rubric-only mode).
        max_item_weight: The largest criterion's share of the gain weight (rubric-only mode).
        gamma_saturated: Every discrimination sits at its bound (rubric-only mode).
        frac_passing_any: Share of documents that passed any criterion (rubric-only mode).
        ordinal_only: The gains order documents but are not calibrated probabilities (rubric-only mode).
        ordinal_reason: Why, when ``ordinal_only``.
        ability_prior: The ability prior on the reported scale (rubric-only mode); a document scored from its
            rubric answers against the calibration uses it, so the fit's own evidence scores to the fit's ability.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_observations: int
    n_queries: int | None = None
    n_documents: int | None = None
    mean_placements: float | None = None
    max_item_weight: float | None = None
    gamma_saturated: bool | None = None
    frac_passing_any: float | None = None
    ordinal_only: bool = False
    ordinal_reason: str | None = None
    ability_prior: AbilityPrior | None = None


@dataclass(frozen=True)
class CalibrationFit:
    """A fitted 2PL calibration.

    Attributes:
        mode: ``"tournament"`` or ``"rubric_only"``.
        items: The criterion parameters, reported with ``sum(gamma) == K`` and ``mean(beta) == 0``.
        queries: ``{query_id: QueryParams}`` for every fitted query of a tournament fit; empty for a
            rubric-only fit.
        judge_severity: ``{judge_id: logit offset}`` of a pooled fit; empty for a single judge.
        thetas: ``{query_id: {doc_id: calibrated ability in logits}}`` -- for a
            tournament fit, every document with a Bradley-Terry score in a fitted
            query; for a rubric-only fit, every observed document.
        theta_se: ``{query_id: {doc_id: posterior SD}}`` of a rubric-only fit (logits,
            on the reported scale); empty for a tournament fit, whose standard errors
            come from the Bradley-Terry fit (``tau_q * se_BT``).
        diagnostics: The fit's own summary.
    """

    mode: FittedMode
    items: ItemParams
    queries: dict[str, QueryParams]
    thetas: dict[str, dict[str, float]]
    diagnostics: FitDiagnostics
    judge_severity: dict[str, float] = field(default_factory=dict)
    theta_se: dict[str, dict[str, float]] = field(default_factory=dict)


def fit_calibration(
    observations: Mapping[str, Sequence[Observation]],
    *,
    mode: FittedMode,
    bt_scores: Mapping[str, Mapping[str, float]] | None = None,
    judges: Judges = "single",
    num_criteria: int = 5,
    priors: Priors | None = None,
) -> CalibrationFit:
    """Fit the 2PL calibration.

    Args:
        observations: ``{query_id: [(doc_id, {"C1": 0|1, ...}[, judge_id]), ...]}``,
            one entry per document per judged window.
        mode: ``"tournament"`` or ``"rubric_only"`` (module docstring).
        bt_scores: ``{query_id: {doc_id: theta_BT}}`` from the tournament; required by ``"tournament"``.
        judges: ``"single"`` (untagged rows) or ``"pooled"`` (every row tagged, two
            or more judges).
        num_criteria: ``K``, the criteria per observation (``C1..C<K>``).
        priors: The priors (the paper's, ``Priors()``, when ``None``).

    Returns:
        The :class:`CalibrationFit`.

    Raises:
        ValueError: An unknown switch value, a tournament fit without scores, rows
            whose judge tags do not match ``judges``, a pooled rubric-only fit, or
            no usable observation.
    """
    if mode not in ("tournament", "rubric_only"):
        raise ValueError(f'mode must be "tournament" or "rubric_only", got {mode!r}')
    if mode == "tournament" and bt_scores is None:
        raise ValueError('mode="tournament" needs Bradley-Terry scores (bt_scores)')
    _check_judges(observations, judges)
    priors = priors or Priors()
    if mode == "tournament":
        assert bt_scores is not None
        return _fit_tournament(observations, bt_scores, num_criteria=num_criteria, priors=priors)
    if judges == "pooled":
        raise ValueError(
            "pooled judges need the tournament fit: the rubric-only model has no judge-severity term, "
            "and pooling without one biases the item parameters by the judges' pass-rate gap"
        )
    return _fit_rubric_only(observations, num_criteria=num_criteria, priors=priors)


def _check_judges(observations: Mapping[str, Sequence[Observation]], judges: Judges) -> None:
    tags = {row[2] if len(row) == 3 else None for rows in observations.values() for row in rows}  # type: ignore[misc]
    if judges == "single":
        if tags - {None}:
            raise ValueError('judges="single" takes untagged (doc_id, criteria) rows; use judges="pooled"')
        return
    if judges != "pooled":
        raise ValueError(f'judges must be "single" or "pooled", got {judges!r}')
    if None in tags:
        raise ValueError('judges="pooled" needs every row tagged (doc_id, criteria, judge_id)')
    if len(tags) < 2:
        raise ValueError(f'judges="pooled" needs at least two judges, got {sorted(t for t in tags if t)}')


def _fit_tournament(
    observations: Mapping[str, Sequence[Observation]],
    bt_scores: Mapping[str, Mapping[str, float]],
    *,
    num_criteria: int,
    priors: Priors,
) -> CalibrationFit:
    from rcp_ndcg_core.irt._tournament_2pl import calibrate_2pl_from_results

    engine = calibrate_2pl_from_results(
        bt_scores=bt_scores,  # type: ignore[arg-type]
        rubric_observations=observations,  # type: ignore[arg-type]
        num_criteria=num_criteria,
        sigma_tau=priors.sigma_tau,
        sigma_alpha=priors.sigma_alpha,
        l2_gamma=priors.l2_gamma,
        l2_beta=priors.l2_beta,
    )
    items = items_from_mapping(engine.get_item_params())
    queries = {key: QueryParams(tau=p["tau"], alpha=p["alpha"]) for key, p in engine.get_query_params().items()}
    thetas = {
        query_id: {doc_id: queries[query_id].calibrated(theta) for doc_id, theta in bt_scores[query_id].items()}
        for query_id in queries
    }
    diagnostics = FitDiagnostics(n_observations=len(engine._obs_rows), n_queries=len(queries))
    return CalibrationFit("tournament", items, queries, thetas, diagnostics, judge_severity=engine.get_judge_severity())


def verdicts(criteria: Mapping[str, int], num_criteria: int) -> list[int]:
    """One placement's verdicts in criterion order ``C1..C<num_criteria>`` (0 or 1 each).

    Raises:
        ValueError: the keys are not exactly ``C1..C<num_criteria>``: a verdict missing, or one for a criterion
            the fit does not read. Nothing is defaulted: a missing verdict is not a fail.
    """
    labels = [f"C{k + 1}" for k in range(num_criteria)]
    if set(criteria) != set(labels):
        missing = sorted(set(labels) - set(criteria))
        unknown = sorted(set(criteria) - set(labels))
        raise ValueError(f"the verdicts do not match the criteria {labels}: missing {missing}, unknown {unknown}")
    return [int(criteria[label]) for label in labels]


def _fit_rubric_only(
    observations: Mapping[str, Sequence[Observation]], *, num_criteria: int, priors: Priors
) -> CalibrationFit:
    from rcp_ndcg_core.irt._criteria_2pl import Criteria2PL

    keys: list[tuple[str, str]] = []
    index: dict[tuple[str, str], int] = {}
    placements: list[float] = []
    passes: list[np.ndarray] = []
    for query_id, rows in observations.items():
        for row in rows:
            doc_id, criteria = row[0], row[1]
            key = (query_id, doc_id)
            if key not in index:
                index[key] = len(keys)
                keys.append(key)
                placements.append(0.0)
                passes.append(np.zeros(num_criteria))
            i = index[key]
            placements[i] += 1.0
            passes[i] += verdicts(criteria, num_criteria)
    if not keys:
        raise ValueError("No rubric observations registered (every query has an empty observation list)")

    n = np.asarray(placements)
    S = np.stack(passes)
    engine = Criteria2PL.fit(n, S, prior_sd=priors.ability_sd)
    theta, sd = engine.eap(n, S)

    # Report on the common convention sum(gamma) = K, mean(beta) = 0. With
    # theta' = a * theta + b, gamma' = gamma / a and beta' = a * beta + b leave every
    # logit gamma * (theta - beta) -- and so every gain -- unchanged; the posterior
    # mean moves with theta because the map is affine.
    scale = float(engine.gamma.sum()) / num_criteria
    shift = -scale * float(engine.beta.mean())
    gamma = engine.gamma / scale
    beta = scale * engine.beta + shift
    thetas: dict[str, dict[str, float]] = {}
    theta_se: dict[str, dict[str, float]] = {}
    for (query_id, doc_id), value, spread in zip(keys, scale * theta + shift, scale * sd, strict=True):
        thetas.setdefault(query_id, {})[doc_id] = float(value)
        theta_se.setdefault(query_id, {})[doc_id] = float(spread)

    diag = engine.diagnostics
    assert diag is not None  # fit() always sets it
    diagnostics = FitDiagnostics(
        n_observations=int(n.sum()),
        n_documents=diag.n_documents,
        mean_placements=diag.mean_placements,
        max_item_weight=diag.max_item_weight,
        gamma_saturated=diag.gamma_saturated,
        frac_passing_any=diag.frac_passing_any,
        ordinal_only=diag.ordinal_only,
        ordinal_reason=diag.reason,
        ability_prior=AbilityPrior(mean=shift, sd=scale * engine.prior_sd),
    )
    items = ItemParams(gamma=tuple(float(g) for g in gamma), beta=tuple(float(b) for b in beta))
    return CalibrationFit("rubric_only", items, {}, thetas, diagnostics, theta_se=theta_se)


__all__ = [
    "AbilityPrior",
    "CalibrationFit",
    "FitDiagnostics",
    "Priors",
    "FittedMode",
    "Judges",
    "Observation",
    "fit_calibration",
    "verdicts",
]
