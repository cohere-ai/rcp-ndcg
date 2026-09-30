"""The gain variants of ``experiments/gain_variants.py`` (gain ablations; not part of the library)."""

from __future__ import annotations

import math

import pytest


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


ITEM_PARAMS = {
    "gamma": [0.6, 0.9, 1.1, 1.2, 1.3],
    "beta": [-1.5, -0.6, 0.0, 0.6, 1.5],
    "num_criteria": 5,
}


@pytest.fixture()
def item_params_3() -> dict:
    return {"gamma": [1.2, 1.0, 0.8], "beta": [-2.0, 0.0, 3.0], "num_criteria": 3}


def test_the_variant_list_names_only_known_variants(gain_variants) -> None:
    for name in gain_variants.GAIN_VARIANTS:
        if name.startswith("criterion_C") and int(name[len("criterion_C") :]) > 5:
            continue
        assert math.isfinite(gain_variants.variant_gain(0.3, ITEM_PARAMS, name))


def test_mean_criterion_accepts_a_nonpositive_gamma_sum(gain_variants) -> None:
    item_params = {"gamma": [-1.0, 1.0], "beta": [0.0, 1.0], "num_criteria": 2}
    assert math.isfinite(gain_variants.variant_gain(0.0, item_params, "mean_criterion"))


class TestVariantFormulas:
    @pytest.mark.parametrize("theta", [0.0, -5.0, 5.0])
    def test_criterion_c1(self, gain_variants, item_params_3, theta):
        expected = _sigmoid(1.2 * (theta + 2.0))
        assert gain_variants.variant_gain(theta, item_params_3, "criterion_C1") == pytest.approx(expected)

    @pytest.mark.parametrize("theta", [0.0, -5.0, 5.0])
    def test_criterion_c2(self, gain_variants, item_params_3, theta):
        expected = _sigmoid(1.0 * theta)
        assert gain_variants.variant_gain(theta, item_params_3, "criterion_C2") == pytest.approx(expected)

    @pytest.mark.parametrize("theta", [0.0, -5.0, 5.0])
    def test_criterion_c3(self, gain_variants, item_params_3, theta):
        expected = _sigmoid(0.8 * (theta - 3.0))
        assert gain_variants.variant_gain(theta, item_params_3, "criterion_C3") == pytest.approx(expected)

    @pytest.mark.parametrize("theta", [0.0, -5.0, 5.0])
    def test_mean_criterion(self, gain_variants, item_params_3, theta):
        p1 = _sigmoid(1.2 * (theta + 2.0))
        p2 = _sigmoid(1.0 * theta)
        p3 = _sigmoid(0.8 * (theta - 3.0))
        expected = (p1 + p2 + p3) / 3
        assert gain_variants.variant_gain(theta, item_params_3, "mean_criterion") == pytest.approx(expected)

    @pytest.mark.parametrize("theta", [0.0, -5.0, 5.0])
    def test_weighted_criterion(self, gain_variants, item_params_3, theta):
        gammas = [1.2, 1.0, 0.8]
        p1 = _sigmoid(1.2 * (theta + 2.0))
        p2 = _sigmoid(1.0 * theta)
        p3 = _sigmoid(0.8 * (theta - 3.0))
        expected = (gammas[0] * p1 + gammas[1] * p2 + gammas[2] * p3) / sum(gammas)
        assert gain_variants.variant_gain(theta, item_params_3, "weighted_criterion") == pytest.approx(expected)

    @pytest.mark.parametrize("theta", [0.0, -5.0, 5.0])
    def test_sigmoid_theta(self, gain_variants, item_params_3, theta):
        expected = _sigmoid(theta)
        assert gain_variants.variant_gain(theta, item_params_3, "sigmoid_theta") == pytest.approx(expected)

    @pytest.mark.parametrize("theta", [0.0, -5.0, 5.0])
    def test_sigmoid_mean_beta(self, gain_variants, item_params_3, theta):
        expected = _sigmoid(theta - ((-2.0 + 0.0 + 3.0) / 3))
        assert gain_variants.variant_gain(theta, item_params_3, "sigmoid_mean_beta") == pytest.approx(expected)

    def test_unknown_specifier_raises(self, gain_variants, item_params_3):
        with pytest.raises(ValueError, match="Unknown gain variant"):
            gain_variants.variant_gain(0.0, item_params_3, "unknown_gain")


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


def test_every_listed_variant_evaluates(gain_variants) -> None:
    for spec in [
        "criterion_C1",
        "criterion_C5",
        "sigmoid_theta",
        "sigmoid_mean_beta",
        "weighted_discrimination",
        "weighted_criterion",
        "mean_criterion",
        "excess_probability",
        "linear_theta",
    ]:
        v = gain_variants.variant_gain(0.0, ITEM_PARAMS, spec)
        assert isinstance(v, float)


def test_unknown_variant_raises(gain_variants) -> None:
    with pytest.raises(ValueError, match="Unknown gain variant"):
        gain_variants.variant_gain(0.0, ITEM_PARAMS, "not_a_real_gain")


def test_weighted_discrimination_aliases_weighted_criterion(gain_variants) -> None:
    a = gain_variants.variant_gain(0.7, ITEM_PARAMS, "weighted_discrimination")
    b = gain_variants.variant_gain(0.7, ITEM_PARAMS, "weighted_criterion")
    assert a == b


def test_variants_monotone_in_theta(gain_variants) -> None:
    """For criterion_Ck, sigmoid_theta, mean_criterion, weighted_discrimination,
    higher theta strictly increases the gain."""
    for spec in [
        "criterion_C3",
        "sigmoid_theta",
        "mean_criterion",
        "weighted_discrimination",
    ]:
        thetas = [-2.0, -0.5, 0.0, 0.5, 2.0]
        gains = [gain_variants.variant_gain(t, ITEM_PARAMS, spec) for t in thetas]
        for a, b in zip(gains, gains[1:], strict=False):
            assert b >= a, f"{spec}: gain({a}) > gain({b})"
