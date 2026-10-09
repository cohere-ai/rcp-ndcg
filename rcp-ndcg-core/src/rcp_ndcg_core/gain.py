"""Relevance gains for :func:`rcp_ndcg_core.metric.ndcg` (numpy, no torch).

* :func:`gain` -- the paper's RCP gain ``g(theta)``: the discrimination-weighted mean of the
  criterion pass probabilities at a calibrated ability.
* :func:`pass_probabilities` -- ``P(C_c | theta) = sigmoid(gamma_c * (theta - beta_c))`` per criterion.
* :func:`count_gain` -- the rubric-only Count-nDCG gain: a document's share of passed criteria.
* :func:`qrel_gain` -- the gain of a human grade.

RCP-, qrel- and Count-nDCG differ only in these gains. The item parameters ``items`` are any object with
``gamma`` and ``beta`` sequences (:class:`~rcp_ndcg_core.schemas.ItemParams`), or a mapping with the keys
``"gamma"`` and ``"beta"`` (the ``items.json`` of a calibration). The number of criteria is ``len(gamma)``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

type Gains = Mapping[str, Mapping[str, float]]
"""``{query_id: {doc_id: gain}}`` with gains in ``[0, 1]``: what :func:`rcp_ndcg.eval.evaluate` scores against."""


def item_arrays(items: Any, *, weighted: bool = True) -> tuple[Sequence[float], Sequence[float]]:
    """``(gamma, beta)`` of ``items``, validated: equal non-zero lengths, finite values, and ``sum(gamma) > 0``.

    Args:
        items: Item parameters: an :class:`~rcp_ndcg_core.schemas.ItemParams`, any object with ``gamma`` and
            ``beta`` sequences, or a ``{"gamma": [...], "beta": [...]}`` mapping (discriminations, difficulties in
            logits).
        weighted: Also require ``sum(gamma) > 0``, which the discrimination-weighted :func:`gain` divides by.

    Returns:
        ``(gamma, beta)``, as given.

    Raises:
        ValueError: missing, non-finite or mismatched parameters.
    """
    if isinstance(items, Mapping):
        try:
            gammas, betas = items["gamma"], items["beta"]
        except KeyError as exc:
            raise ValueError(f"item parameters are missing {exc.args[0]!r}") from exc
    else:
        try:
            gammas, betas = items.gamma, items.beta
        except AttributeError as exc:
            raise ValueError("item parameters need 'gamma' and 'beta' sequences") from exc
    for name, values in (("gamma", gammas), ("beta", betas)):
        if isinstance(values, str | bytes) or not isinstance(values, Sequence | np.ndarray):
            raise ValueError(f"item parameter {name!r} must be a sequence of finite numbers")
        try:
            finite = all(math.isfinite(value) for value in values)
        except TypeError as exc:
            raise ValueError(f"item parameter {name!r} must contain only finite numbers") from exc
        if not finite:
            raise ValueError(f"item parameter {name!r} must contain only finite numbers")
    if len(gammas) == 0 or len(gammas) != len(betas):
        raise ValueError(f"item parameters need as many betas as gammas (> 0): {len(gammas)} and {len(betas)}")
    if weighted and sum(gammas) <= 0:
        raise ValueError("sum(gamma) must be greater than 0 for the discrimination-weighted gain")
    return gammas, betas


@overload
def sigmoid(x: float) -> float: ...
@overload
def sigmoid(x: NDArray[np.floating]) -> NDArray[np.float64]: ...
def sigmoid(x: float | NDArray[np.floating]) -> float | NDArray[np.float64]:
    """The logistic function ``1 / (1 + exp(-x))``, computed stably: a float for a scalar, an array for an array.

    The one logistic function of the package: pass probabilities, the tournament's pair probabilities, the
    calibration fit, the insertion and the offline judge all use it.
    """
    if np.ndim(x) == 0:
        x = float(x)  # type: ignore[arg-type]
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        ez = math.exp(x)
        return ez / (1.0 + ez)
    arr = np.asarray(x)
    out = np.empty_like(arr, dtype=float)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    ex = np.exp(arr[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def pass_probabilities(theta: float | ArrayLike, items: Any) -> NDArray[np.float64]:
    """Per-criterion pass probabilities at a calibrated ability.

    Args:
        theta: Calibrated ability, in logits: a scalar, or an array of abilities.
        items: Item parameters with ``gamma`` (dimensionless) and ``beta`` (logits) sequences.

    Returns:
        ``sigmoid(gamma_c * (theta - beta_c))`` for each criterion ``c``, each in ``(0, 1)``: shape ``(C,)`` for a
        scalar ``theta``, ``theta.shape + (C,)`` for an array. Scalars compute in float64 whatever dtype the
        inputs arrive in.
    """
    gammas, betas = item_arrays(items, weighted=False)
    if np.ndim(theta) == 0:
        scalar = float(np.asarray(theta))  # a 0-d theta: computed in float64, like the array path
        weights = [float(g) for g in gammas]
        return np.array([sigmoid(g * (scalar - float(b))) for g, b in zip(weights, betas, strict=True)])
    arr = np.asarray(theta, dtype=float)
    return sigmoid(np.asarray(gammas, dtype=float) * (arr[..., None] - np.asarray(betas, dtype=float)))


def _weighted_mean(probabilities: Sequence[float], weights: Sequence[float]) -> float:
    """``sum(w * p) / sum(w)`` without an overflowing sum: the weights are divided by their
    largest absolute value first, which leaves the weighted mean unchanged and keeps every
    term finite for finite inputs."""
    scale = max(abs(w) for w in weights)
    normalised = [w / scale for w in weights]
    return sum(w * p for w, p in zip(normalised, probabilities, strict=True)) / sum(normalised)


@overload
def gain(theta: float, items: Any) -> float: ...
@overload
def gain(theta: ArrayLike, items: Any) -> float | NDArray[np.float64]: ...
def gain(theta: float | ArrayLike, items: Any) -> float | NDArray[np.float64]:
    """The RCP gain ``g(theta) = sum_c gamma_c * P(C_c | theta) / sum_c gamma_c``.

    Args:
        theta: Calibrated ability, in logits: a scalar, or an array of abilities.
        items: Item parameters with ``gamma`` and ``beta`` sequences; ``sum(gamma) > 0``.

    Returns:
        The gain in ``(0, 1)``: a float for a scalar ``theta``, an array of ``theta``'s shape otherwise.
        Scalars compute in float64 whatever dtype the inputs arrive in.
    """
    gammas, betas = item_arrays(items)
    if np.ndim(theta) == 0:
        scalar = float(np.asarray(theta))  # a 0-d theta: computed in float64, like the array path
        weights = [float(g) for g in gammas]
        probs = [sigmoid(g * (scalar - float(b))) for g, b in zip(weights, betas, strict=True)]
        total = sum(weights)
        if math.isfinite(total):
            return sum(g * p for g, p in zip(weights, probs, strict=True)) / total
        return _weighted_mean(probs, weights)
    weights = np.asarray(gammas, dtype=float)
    try:
        total = math.fsum(weights)  # raises instead of silently overflowing to inf
    except OverflowError:
        normalised = weights / weights.max()
        return pass_probabilities(theta, items) @ normalised / normalised.sum()
    return pass_probabilities(theta, items) @ weights / total


def count_gain(passes: Sequence[int], placements: int) -> float:
    """The Count-nDCG gain of one document: its share of passed rubric criteria.

    The paper's rubric-only baseline, ``G(d) = sum_c S_c / (C * n)``: no tournament, no calibration.

    Args:
        passes: ``S_c`` per criterion: the number of the document's rubric placements that passed criterion ``c``.
            ``C = len(passes)``.
        placements: ``n``, the number of rubric placements of the document.

    Returns:
        The gain, in ``[0, 1]``.

    Raises:
        ValueError: a pass count is not a whole number in ``[0, placements]``, or ``placements`` is not a
            positive whole number.
    """
    if not passes:
        raise ValueError("count_gain needs one pass count per criterion")
    n = float(placements)
    if not math.isfinite(n) or n != math.floor(n):
        raise ValueError(f"placements must be a whole number, got {placements!r}")
    if n < 1:
        raise ValueError(f"placements must be positive, got {placements}")
    bad = [
        s for s in passes if not math.isfinite(float(s)) or float(s) != math.floor(float(s)) or not 0 <= float(s) <= n
    ]
    if bad:
        raise ValueError(
            f"every pass count must be a whole number with 0 <= s_k <= placements, got {bad[:5]} "
            f"(placements={placements:g}); the gain is a share of passed criteria"
        )
    return sum(passes) / (len(passes) * n)


def qrel_gain(grade: float, scheme: Literal["linear", "exponential"] = "linear") -> float:
    """The gain of a human grade.

    Args:
        grade: The relevance grade (qrel label), a finite number.
        scheme: ``"linear"`` (the grade itself: the paper and trec_eval) or ``"exponential"`` (``2**grade - 1``).

    Returns:
        The gain.

    Raises:
        ValueError: the grade is not finite, or the exponential scheme's ``2**grade`` would overflow
            (grades >= 1024).
    """
    value = float(grade)
    if not math.isfinite(value):
        raise ValueError(f"the qrel grade must be a finite number, got {grade!r}")
    if scheme == "linear":
        return value
    if scheme == "exponential":
        if value >= 1024.0:
            raise ValueError(
                f"the exponential gain 2**grade - 1 overflows for grade {value:g}; grades must stay below "
                "1024 (the linear scheme keeps the grade itself as its gain)"
            )
        return float(2**value - 1)
    raise ValueError(f"Unknown qrel gain scheme {scheme!r}; expected 'linear' or 'exponential'")


__all__ = ["Gains", "count_gain", "gain", "item_arrays", "pass_probabilities", "qrel_gain", "sigmoid"]
