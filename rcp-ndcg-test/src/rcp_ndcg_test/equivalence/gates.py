"""The stage-2 gates and their per-``score_scale`` defaults, and the tie-robust rank agreement they use.

The defaults are the published tolerances for checking a served engine against the reference subprocess:
probability-scale scores allow |delta| of 0.02 for 99% of documents and 0.05 for all, raw-logit scores allow
|delta| of 0.05 * (1 + |reference score|), cosine-scale rerank scores allow |delta| of 0.01, vectors must agree
with a cosine of at least 1 - 1e-3 per vector (per token for a late-interaction model, after the same float16
cast), and the median per-query Kendall tau between served and reference scores must be at least 0.98.  A recipe's
``gates`` section overrides any of them.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict
from rcp_ndcg_vllm.recipe import Recipe

__all__ = ["ResolvedGates", "kendall_tau_b", "resolve_gates"]

_PROBABILITY_P99_ABS = 0.02
_PROBABILITY_MAX_ABS = 0.05
_LOGIT_REL_ABS = 0.05
_COSINE_MAX_ABS = 0.01
_VECTOR_MIN_COSINE = 1.0 - 1e-3
_TAU_MIN = 0.98
_METRICS_MAX_ABS = 2e-3


class ResolvedGates(BaseModel):
    """The gate values in force for one recipe: its overrides over the published defaults.

    Attributes:
        prob_p99_abs: Probability scale: the |delta| bound that 99% of documents must meet.
        prob_max_abs: Probability scale: the |delta| bound every document must meet.
        logit_rel_abs: Logit scale: the |delta| bound as a fraction, ``rel_abs * (1 + |reference|)``.
        cos_max_abs: Cosine-scale rerank scores: the |delta| bound every document must meet.
        vec_min_cosine: Vectors: the cosine floor per vector (per token for a late-interaction model).
        tau_min: The median per-query Kendall tau floor between served and reference scores.
        metrics_max_abs: Stage 3: the mean |delta nDCG@10| over subsets.
        embed_dtype: The transfer precision the vectors were compared in.
    """

    model_config = ConfigDict(frozen=True)

    prob_p99_abs: float
    prob_max_abs: float
    logit_rel_abs: float
    cos_max_abs: float
    vec_min_cosine: float
    tau_min: float
    metrics_max_abs: float
    embed_dtype: str


def resolve_gates(recipe: Recipe) -> ResolvedGates:
    """The gates for ``recipe``: the defaults for its ``reference.score_scale``, overridden by ``recipe.gates``."""
    overrides = recipe.gates
    return ResolvedGates(
        prob_p99_abs=overrides.prob_p99_abs if overrides.prob_p99_abs is not None else _PROBABILITY_P99_ABS,
        prob_max_abs=overrides.prob_max_abs if overrides.prob_max_abs is not None else _PROBABILITY_MAX_ABS,
        logit_rel_abs=overrides.logit_rel_abs if overrides.logit_rel_abs is not None else _LOGIT_REL_ABS,
        cos_max_abs=overrides.cos_max_abs if overrides.cos_max_abs is not None else _COSINE_MAX_ABS,
        vec_min_cosine=overrides.vec_min_cosine if overrides.vec_min_cosine is not None else _VECTOR_MIN_COSINE,
        tau_min=overrides.tau_min if overrides.tau_min is not None else _TAU_MIN,
        metrics_max_abs=overrides.metrics_max_abs if overrides.metrics_max_abs is not None else _METRICS_MAX_ABS,
        embed_dtype=str(recipe.client.get("embed_dtype", "float16")),
    )


def kendall_tau_b(served: list[float], reference: list[float]) -> float | None:
    """Kendall's tau-b between two score lists over one query's documents, or ``None`` when undefined.

    Tau-b corrects for ties, which rounding and exact score collisions make common; a query with fewer than two
    documents (or where every pair is tied) has no ordering to agree on and yields ``None``.
    """
    n = len(served)
    if n != len(reference):
        raise ValueError(f"kendall_tau_b needs equally long lists, got {n} and {len(reference)}")
    if n < 2:
        return None
    a = np.asarray(served, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    concordant = 0
    discordant = 0
    ties_a = 0
    ties_b = 0
    for i in range(n - 1):
        da = a[i] - a[i + 1 :]
        db = b[i] - b[i + 1 :]
        concordant += int(np.sum((da * db) > 0))
        discordant += int(np.sum((da * db) < 0))
        ties_a += int(np.sum(da == 0))
        ties_b += int(np.sum(db == 0))
    pairs = n * (n - 1) / 2
    denominator = float(np.sqrt((pairs - ties_a) * (pairs - ties_b)))
    if denominator == 0.0:
        return None
    return (concordant - discordant) / denominator
