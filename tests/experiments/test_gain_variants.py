"""The gain variants of ``experiments/gain_variants.py`` (gain ablations; not part of the library).

One parametrized pin asserts every variant's formula against this module's own oracle; the monotonicity and
refusal rows below are behaviour the formulas alone do not state.
"""

from __future__ import annotations

import math

import pytest


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _oracle(name: str, theta: float, params: dict) -> float:
    """Each variant's formula, written out independently of ``experiments/gain_variants.py``."""
    probs = [_sigmoid(g * (theta - b)) for g, b in zip(params["gamma"], params["beta"], strict=True)]
    betas = params["beta"]
    if name.startswith("criterion_C"):
        return probs[int(name[len("criterion_C") :]) - 1]
    weighted = sum(g * p for g, p in zip(params["gamma"], probs, strict=True)) / sum(params["gamma"])
    formula = {
        "sigmoid_theta": lambda: _sigmoid(theta),
        "sigmoid_mean_beta": lambda: _sigmoid(theta - sum(betas) / len(betas)),
        "weighted_discrimination": lambda: weighted,
        "weighted_criterion": lambda: weighted,
        "mean_criterion": lambda: sum(probs) / len(probs),
        "excess_probability": lambda: sum(max(0.0, p - 0.5) for p in probs) / len(probs),
        "gated_weighted": lambda: 0.0 if theta < min(betas) else weighted,
        "power2_weighted": lambda: weighted**2,
        "linear_theta": lambda: max(0.0, min(1.0, (theta - min(betas)) / (max(betas) - min(betas)))),
    }[name]
    return formula()


#: Every variant name, pinned here: the completeness row below fails when the list changes without this pin.
PINNED_VARIANTS = (
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

ITEM_PARAMS = {
    "gamma": [0.6, 0.9, 1.1, 1.2, 1.3],
    "beta": [-1.5, -0.6, 0.0, 0.6, 1.5],
    "num_criteria": 5,
}
# -3.0 is below every beta (the gated branch and the clamped linear branch), 0.3 mid-range, 2.0 above.
THETAS = [-3.0, 0.3, 2.0]


@pytest.fixture()
def item_params_3() -> dict:
    return {"gamma": [1.2, 1.0, 0.8], "beta": [-2.0, 0.0, 3.0], "num_criteria": 3}


def test_the_pin_table_covers_every_variant(gain_variants) -> None:
    assert PINNED_VARIANTS == tuple(gain_variants.GAIN_VARIANTS)


@pytest.mark.parametrize("theta", THETAS)
@pytest.mark.parametrize("name", PINNED_VARIANTS)
def test_every_variant_is_its_formula(gain_variants, name: str, theta: float) -> None:
    observed = gain_variants.variant_gain(theta, ITEM_PARAMS, name)
    assert observed == pytest.approx(_oracle(name, theta, ITEM_PARAMS))
    assert 0.0 <= observed <= 1.0


def test_mean_criterion_accepts_a_nonpositive_gamma_sum(gain_variants) -> None:
    item_params = {"gamma": [-1.0, 1.0], "beta": [0.0, 1.0], "num_criteria": 2}
    assert math.isfinite(gain_variants.variant_gain(0.0, item_params, "mean_criterion"))


def test_weighted_discrimination_aliases_weighted_criterion(gain_variants) -> None:
    a = gain_variants.variant_gain(0.7, ITEM_PARAMS, "weighted_discrimination")
    b = gain_variants.variant_gain(0.7, ITEM_PARAMS, "weighted_criterion")
    assert a == b


class TestVariantMonotonicity:
    GAIN_SPECS = [
        "criterion_C1",
        "criterion_C2",
        "criterion_C3",
        "mean_criterion",
        "weighted_criterion",
        "sigmoid_theta",
        "sigmoid_mean_beta",
    ]

    @pytest.mark.parametrize("gain_spec", GAIN_SPECS)
    def test_gain_monotonically_increasing(self, gain_variants, item_params_3, gain_spec):
        thetas = [x * 0.5 for x in range(-20, 21)]  # -10.0 to 10.0
        gains = [gain_variants.variant_gain(t, item_params_3, gain_spec) for t in thetas]
        for i in range(len(gains) - 1):
            assert gains[i] <= gains[i + 1], (
                f"{gain_spec}: gain({thetas[i]})={gains[i]} > gain({thetas[i + 1]})={gains[i + 1]}"
            )


class TestRefusals:
    @pytest.mark.parametrize("bad", ["unknown_gain", "not_a_real_gain"])
    def test_an_unknown_specifier_raises(self, gain_variants, item_params_3, bad):
        with pytest.raises(ValueError, match="Unknown gain variant"):
            gain_variants.variant_gain(0.0, item_params_3, bad)
