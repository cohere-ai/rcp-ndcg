"""Alternative gain functions for gain ablations (not part of the library).

The paper uses one gain, the discrimination-weighted mean of the criterion pass probabilities
(:func:`rcp_ndcg_core.gain.gain`). The variants below are the alternatives compared in gain ablations. They take
the same inputs: a calibrated ability ``theta_cal`` (logits) and the 2PL item parameters
``{"gamma": [...], "beta": [...]}``.

    from gain_variants import variant_gain
    variant_gain(0.7, item_params, "mean_criterion")
"""

from __future__ import annotations

from rcp_ndcg_core.gain import gain, pass_probabilities

#: Every variant name :func:`variant_gain` accepts. ``weighted_discrimination`` (alias ``weighted_criterion``) is
#: the paper's gain.
GAIN_VARIANTS: tuple[str, ...] = (
    "criterion_C1",
    "criterion_C2",
    "criterion_C3",
    "criterion_C4",
    "criterion_C5",
    "sigmoid_theta",
    "sigmoid_mean_beta",
    "weighted_discrimination",
    "weighted_criterion",
    "mean_criterion",
    "excess_probability",
    "gated_weighted",
    "power2_weighted",
    "linear_theta",
)

_WEIGHTED = {"weighted_discrimination", "weighted_criterion", "gated_weighted", "power2_weighted"}


def variant_gain(theta_cal: float, item_params: dict, name: str) -> float:
    """The gain variant ``name`` at a calibrated ability.

    Args:
        theta_cal: Calibrated ability, in logits.
        item_params: 2PL item parameters.
        name: One of :data:`GAIN_VARIANTS`.

    Returns:
        The gain: in ``[0, 1]`` for every variant.
    """
    probs = pass_probabilities(theta_cal, item_params)  # validates the item parameters
    betas, num_criteria = item_params["beta"], len(probs)
    if name in _WEIGHTED and sum(item_params["gamma"]) <= 0:
        raise ValueError("sum(gamma) must be greater than 0 for a discrimination-weighted gain")
    if name.startswith("criterion_C"):
        try:
            k = int(name.split("C")[1]) - 1
        except (IndexError, ValueError) as exc:
            raise ValueError(f"Unknown gain variant: {name}") from exc
        if k < 0 or k >= num_criteria:
            raise ValueError(f"Gain variant {name!r} requires a criterion present in item_params")
        return float(probs[k])
    if name == "sigmoid_theta":  # one criterion with gamma 1 and beta 0
        return float(pass_probabilities(theta_cal, {"gamma": [1.0], "beta": [0.0]})[0])
    if name == "sigmoid_mean_beta":
        return float(pass_probabilities(theta_cal, {"gamma": [1.0], "beta": [sum(betas) / num_criteria]})[0])
    if name in ("weighted_discrimination", "weighted_criterion"):
        return gain(theta_cal, item_params)
    if name == "mean_criterion":
        return sum(probs) / num_criteria
    if name == "excess_probability":
        return sum(max(0.0, p - 0.5) for p in probs) / num_criteria
    if name == "gated_weighted":
        return 0.0 if theta_cal < min(betas) else gain(theta_cal, item_params)
    if name == "power2_weighted":
        return gain(theta_cal, item_params) ** 2
    if name == "linear_theta":
        lo, hi = min(betas), max(betas)
        return max(0.0, min(1.0, (theta_cal - lo) / (hi - lo))) if hi > lo else 0.5
    raise ValueError(f"Unknown gain variant: {name}")


__all__ = ["GAIN_VARIANTS", "variant_gain"]
