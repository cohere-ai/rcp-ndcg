"""Is the fitted 2PL calibrated? Predicted pass probabilities against the verdicts the judge gave.

For every rubric verdict of a calibrated document, the fit predicts
``P(C_k = 1) = sigmoid(gamma_k * (theta - beta_k))`` (minus the judge's severity
in a pooled fit). :func:`reliability` bins predicted against observed and
reports the expected calibration error (ECE, population-weighted, so a bin of
three observations cannot outweigh a bin of three thousand) and the Brier
score. :func:`fit_diagnostics` reports it overall and per criterion, both
pooling every family the fit read, and per judgement family, which keeps the
families apart.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from rcp_ndcg_core.gain import pass_probabilities
from rcp_ndcg_core.irt import FitDiagnostics
from rcp_ndcg_core.schemas import ItemParams

#: Bins of the reliability curve.
DEFAULT_BINS = 10

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class ReliabilityBin(BaseModel):
    """One bin of a reliability curve: predictions in ``[lower, upper)``; the rates are ``None`` for an empty bin."""

    model_config = _FROZEN

    lower: float
    upper: float
    count: int
    mean_predicted: float | None
    observed_rate: float | None


class Reliability(BaseModel):
    """Predicted against observed pass rates: the binned curve, the ECE and the Brier score (``None`` when empty)."""

    model_config = _FROZEN

    count: int
    ece: float | None
    brier: float | None
    bins: list[ReliabilityBin]


class FamilyReliability(BaseModel):
    """One judgement family's reliability, overall and per criterion."""

    model_config = _FROZEN

    judge: str | None
    overall: Reliability
    per_criterion: dict[str, Reliability]


class FitReliability(BaseModel):
    """A fit's reliability: overall and per criterion (every family pooled), and per family."""

    model_config = _FROZEN

    overall: Reliability
    per_criterion: dict[str, Reliability]
    per_family: dict[str, FamilyReliability]


class FitWarning(BaseModel):
    """A typed warning of the fit (a :class:`~rcp_ndcg.errors.RcpNdcgWarning`'s code and message)."""

    model_config = _FROZEN

    code: str
    message: str


class Diagnostics(BaseModel):
    """What a calibration's fit reports about itself.

    Attributes:
        fit: The estimator's own summary.
        reliability: Predicted against observed verdicts.
        warnings: The fit's typed warnings (e.g. ``INVALID_WINDOWS``).
    """

    model_config = _FROZEN

    fit: FitDiagnostics
    reliability: FitReliability
    warnings: list[FitWarning] = Field(default_factory=list)


def reliability(predicted: Sequence[float], observed: Sequence[float], *, bins: int = DEFAULT_BINS) -> Reliability:
    """Binned predicted-vs-observed pass rates, the ECE and the Brier score.

    Args:
        predicted: Predicted pass probabilities in [0, 1].
        observed: The verdicts (0 or 1), aligned with ``predicted``.
        bins: Equal-width bins on [0, 1].

    Returns:
        The :class:`Reliability`; ``ece`` and ``brier`` are ``None`` without observations.
    """
    p = np.asarray(predicted, dtype=float)
    y = np.asarray(observed, dtype=float)
    if p.size == 0:
        return Reliability(count=0, ece=None, brier=None, bins=[])
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignment = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        mask = assignment == index
        count = int(mask.sum())
        mean_predicted = float(p[mask].mean()) if count else None
        observed_rate = float(y[mask].mean()) if count else None
        if mean_predicted is not None and observed_rate is not None:
            ece += count / p.size * abs(observed_rate - mean_predicted)
        rows.append(
            ReliabilityBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=count,
                mean_predicted=mean_predicted,
                observed_rate=observed_rate,
            )
        )
    return Reliability(count=int(p.size), ece=float(ece), brier=float(np.mean((p - y) ** 2)), bins=rows)


def fit_diagnostics(
    items: ItemParams,
    thetas: Mapping[str, Mapping[str, float]],
    observations: Mapping[str, Sequence[tuple]],
    *,
    family_of_judge: Mapping[str, str],
    judge_severity: Mapping[str, float] | None = None,
    single_judge: str | None = None,
    bins: int = DEFAULT_BINS,
) -> FitReliability:
    """Reliability of a fit against its own rubric verdicts, overall, per criterion and per family.

    Args:
        items: The fitted item parameters.
        thetas: ``{query: {doc_id: calibrated theta}}``.
        observations: ``{query: [(doc_id, criteria[, judge]), ...]}`` -- the verdicts the fit read.
        family_of_judge: ``{judge: family key}`` of the rubric families.
        judge_severity: Per-judge severity of a pooled fit (subtracted from every logit).
        single_judge: The judge of untagged observations (a single-judge fit).
        bins: Bins of the reliability curves.

    Returns:
        The :class:`FitReliability`: overall and per criterion pool every family; per family keeps them apart.
    """
    severity = dict(judge_severity or {})
    criteria = items.criteria
    cache: dict[float, list[float]] = {}
    predicted: dict[tuple[str, str], list[float]] = defaultdict(list)  # (family, criterion) -> predictions
    observed: dict[tuple[str, str], list[float]] = defaultdict(list)
    for query, rows in observations.items():
        query_thetas = thetas.get(query, {})
        for row in rows:
            doc_id, verdicts = row[0], row[1]
            judge = row[2] if len(row) > 2 else single_judge
            theta = query_thetas.get(doc_id)
            if theta is None:
                continue
            probabilities = cache.get(theta)
            if probabilities is None:
                probabilities = cache[theta] = pass_probabilities(theta, items)  # type: ignore[arg-type]
            offset = severity.get(judge or "", 0.0)
            if offset:
                # The severity is additive on the logit: sigmoid(logit(p) - s).
                probabilities = [q / (q + (1.0 - q) * math.exp(offset)) for q in probabilities]
            family = family_of_judge.get(judge or "", "unknown")
            for k, label in enumerate(criteria):
                predicted[(family, label)].append(probabilities[k])
                observed[(family, label)].append(float(verdicts[label]))

    families = sorted({family for family, _ in predicted})
    judges = {family: judge for judge, family in family_of_judge.items()}

    def pooled(keys: list[tuple[str, str]]) -> Reliability:
        return reliability(
            [v for key in keys for v in predicted[key]], [v for key in keys for v in observed[key]], bins=bins
        )

    return FitReliability(
        overall=pooled(list(predicted)),
        per_criterion={label: pooled([(f, label) for f in families]) for label in criteria},
        per_family={
            family: FamilyReliability(
                judge=judges.get(family),
                overall=pooled([(family, label) for label in criteria]),
                per_criterion={label: pooled([(family, label)]) for label in criteria},
            )
            for family in families
        },
    )


__all__ = [
    "DEFAULT_BINS",
    "Diagnostics",
    "FamilyReliability",
    "FitReliability",
    "FitWarning",
    "Reliability",
    "ReliabilityBin",
    "fit_diagnostics",
    "reliability",
]
