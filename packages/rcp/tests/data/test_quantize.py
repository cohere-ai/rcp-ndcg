"""The quantization reference vectors: the serving contract's arithmetic, reproduced byte for byte.

The fixture is the committed contract (``fixtures/quantize/reference-vectors.json``): every case names the
input vector, the range and the expected bytes of the three representations. One vector holds a float64 value
that float32 cannot represent, and another holds a float32 value whose mapped position only stays below the
integer boundary when the arithmetic is widened before it runs -- so a narrowing or float32 implementation
moves a byte here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rcp.data.quantize import QuantizedVector, quantize_binary, quantize_int8, quantize_uint8
from rcp.errors import DataError

FIXTURE = Path(__file__).parent / "fixtures" / "quantize" / "reference-vectors.json"


def _cases() -> list[dict[str, Any]]:
    """The committed cases, read once per parametrization."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["name"])
def test_the_reference_vectors_reproduce_byte_for_byte(case: dict[str, Any]) -> None:
    vector = case["vector"]
    int8 = quantize_int8(vector, min_sq=case["min_sq"], max_sq=case["max_sq"])
    uint8 = quantize_uint8(vector, min_sq=case["min_sq"], max_sq=case["max_sq"])
    binary = quantize_binary(vector, min_sq=case["min_sq"], max_sq=case["max_sq"])

    assert int8.dtype == "int8" and int8.data.hex() == case["int8"]
    assert uint8.dtype == "uint8" and uint8.data.hex() == case["uint8"]
    assert binary.dtype == "binary" and binary.data.hex() == case["binary"]


def test_the_fixture_holds_a_value_float32_cannot_represent() -> None:
    """The widening is observable: the fixture is not all float32-representable values."""
    values = [value for case in _cases() for value in case["vector"]]
    assert any(float(np.float32(value)) != value for value in values)


def test_float32_arithmetic_would_move_the_reference_bytes() -> None:
    """A float32 implementation (the arithmetic in float32, not widened) gives another byte for the same
    float32 input, so the fixture proves the widening rather than merely pinning a round trip."""
    case = next(case for case in _cases() if case["name"] == "unit-range-mixed")
    vector = np.asarray(case["vector"], dtype=np.float32)
    low, high = np.float32(case["min_sq"]), np.float32(case["max_sq"])
    mapped = np.float32(255.0) * (vector - low) / (high - low) - np.float32(128.0)
    narrowed = np.clip(np.trunc(mapped), -128, 127).astype(np.int8).tobytes()
    assert narrowed.hex() != case["int8"]


def test_narrowing_the_input_to_float32_would_move_the_reference_bytes() -> None:
    """The fixture's non-float32 value must not be rounded to float32 before the arithmetic."""
    case = next(case for case in _cases() if case["name"] == "unit-range-mixed")
    narrowed = np.asarray(case["vector"], dtype=np.float32).astype(np.float64)
    mapped = 255.0 * (narrowed - case["min_sq"]) / (case["max_sq"] - case["min_sq"]) - 128.0
    bytes_after_narrowing = np.clip(np.trunc(mapped), -128, 127).astype(np.int8).tobytes()
    assert bytes_after_narrowing.hex() != case["int8"]


def test_a_quantized_vector_is_bytes_and_the_declared_dtype() -> None:
    quantized = quantize_int8([0.0], min_sq=-1.0, max_sq=1.0)

    assert isinstance(quantized, QuantizedVector)
    assert quantized == (b"\x00", "int8")
    data, dtype = quantized
    assert data == b"\x00" and dtype == "int8"


def test_uint8_is_the_int8_value_shifted_by_128() -> None:
    assert quantize_uint8([-1.0], min_sq=-1.0, max_sq=1.0).data == b"\x00"
    assert quantize_uint8([0.0], min_sq=-1.0, max_sq=1.0).data == b"\x80"
    assert quantize_uint8([1.0], min_sq=-1.0, max_sq=1.0).data == b"\xff"


def test_truncation_is_toward_zero_not_toward_minus_infinity() -> None:
    # -0.1 maps to -13.25: truncation gives -13 (0xf3), a floor would give -14 (0xf2).
    assert quantize_int8([-0.1], min_sq=-1.0, max_sq=1.0).data == b"\xf3"


def test_values_outside_the_range_clamp() -> None:
    assert quantize_int8([2.0], min_sq=-1.0, max_sq=1.0).data == b"\x7f"
    assert quantize_int8([-2.0], min_sq=-1.0, max_sq=1.0).data == b"\x80"


def test_binary_reads_the_float_sign_not_the_truncated_int8_sign() -> None:
    """-0.001 truncates to int8 zero but is negative: the binary bit is clear. -0.0 is not negative."""
    assert quantize_binary([-0.001], min_sq=-1.0, max_sq=1.0).data == b"\x00"
    assert quantize_binary([-0.0], min_sq=-1.0, max_sq=1.0).data == b"\x80"


def test_binary_packs_msb_first_and_pads_the_last_byte_with_zeros() -> None:
    # Three signs: clear, set, set -> 0110_0000.
    assert quantize_binary([-1.0, 1.0, 0.0], min_sq=-1.0, max_sq=1.0).data == b"\x60"


def test_an_empty_vector_is_empty_bytes() -> None:
    assert quantize_int8([], min_sq=-1.0, max_sq=1.0) == QuantizedVector(b"", "int8")
    assert quantize_uint8([], min_sq=-1.0, max_sq=1.0) == QuantizedVector(b"", "uint8")
    assert quantize_binary([], min_sq=-1.0, max_sq=1.0) == QuantizedVector(b"", "binary")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_sq": 1.0, "max_sq": 1.0}, "max_sq"),
        ({"min_sq": 2.0, "max_sq": 1.0}, "max_sq"),
        ({"min_sq": float("nan"), "max_sq": 1.0}, "finite"),
        ({"min_sq": -1.0, "max_sq": float("inf")}, "finite"),
    ],
)
def test_a_range_that_cannot_map_is_refused(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(DataError, match=message) as caught:
        quantize_int8([0.0], **kwargs)
    assert caught.value.hint


@pytest.mark.parametrize(
    ("vector", "message"),
    [
        ([0.0, float("nan")], "finite"),
        ([0.0, float("inf")], "finite"),
        ([[0.0, 1.0]], "1-D"),
    ],
)
def test_a_vector_that_cannot_be_quantized_is_refused(vector: list[Any], message: str) -> None:
    with pytest.raises(DataError, match=message) as caught:
        quantize_int8(vector, min_sq=-1.0, max_sq=1.0)
    assert caught.value.hint
