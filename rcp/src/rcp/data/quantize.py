"""Quantize a served embedding vector: int8, uint8 or packed binary.

One home for the serving contract's rounding rule, so a downstream that reports int8 or binary recall calls
this module instead of writing a second implementation that drifts by one byte. The input is the vector a run
compares -- the Matryoshka-truncated, L2-normalised one, never the full-width pre-selection vector -- and the
arithmetic widens it to float64 *before* the mapping, so a float32 intermediate cannot move a truncation.

The contract, in order:

* widen the vector to float64 (the served values are float32; a float64 input is not narrowed first);
* map ``255 * (v - min_sq) / (max_sq - min_sq) - 128``;
* truncate toward zero, then clamp to ``[-128, 127]``;
* ``uint8 = int8 + 128``;
* binary packs eight sign bits MSB-first, with ``v >= 0`` setting the bit (the float's sign, not the truncated
  int8's: ``-0.001`` truncates to int8 zero but is negative, so its bit is clear).

The committed reference vectors in ``tests/data/fixtures/quantize/reference-vectors.json`` pin the bytes,
including a value float32 cannot represent and a value whose truncation moves when the arithmetic runs in
float32. The return value is :class:`QuantizedVector`: the bytes and the declared dtype (``"int8"``,
``"uint8"`` or ``"binary"``), which is the value an ``embedding_types`` config entry names.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from rcp.errors import DataError

EmbeddingType = Literal["int8", "uint8", "binary"]
"""A quantized embedding representation: one entry of an embedding endpoint's ``embedding_types``."""


class QuantizedVector(NamedTuple):
    """One quantized vector: its bytes and the declared dtype that reads them.

    Attributes:
        data: The vector's bytes. ``int8`` and ``uint8`` are one byte per element, in vector order; ``binary``
            is eight sign bits per byte, MSB-first, the last byte zero-padded on the right.
        dtype: The declared representation: ``"int8"``, ``"uint8"`` or ``"binary"``.
    """

    data: bytes
    dtype: EmbeddingType


def _checked(vector: ArrayLike, min_sq: float, max_sq: float) -> tuple[np.ndarray, np.float64, np.float64]:
    """The float64 vector and the finite, ordered range; refuses what cannot be mapped."""
    low, high = np.float64(min_sq), np.float64(max_sq)
    if not (np.isfinite(low) and np.isfinite(high)):
        raise DataError(
            f"quantize: min_sq and max_sq must be finite, got min_sq={min_sq!r} and max_sq={max_sq!r}",
            hint="give the range's finite float64 bounds (min_sq < max_sq)",
        )
    if high <= low:
        raise DataError(
            f"quantize: max_sq ({max_sq!r}) must be greater than min_sq ({min_sq!r})",
            hint="give the range's bounds in order: min_sq is the low end, max_sq the high end",
        )
    try:
        values = np.asarray(vector, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataError(
            f"quantize: the vector is not a sequence of floats: {exc}",
            hint="pass a 1-D sequence or array of finite floats, one value per embedding dimension",
        ) from exc
    if values.ndim != 1:
        raise DataError(
            f"quantize: the vector must be 1-D, got shape {values.shape}",
            hint="quantize one embedding vector at a time (a (N, D) matrix is N calls, or a loop over its rows)",
        )
    if not np.all(np.isfinite(values)):
        raise DataError(
            "quantize: the vector holds a non-finite value (NaN or infinity)",
            hint="quantize a finite embedding; a non-finite value has no byte representation",
        )
    return values, low, high


def _mapped(vector: ArrayLike, min_sq: float, max_sq: float) -> np.ndarray:
    """The mapped float64 values, before truncation and clamping."""
    values, low, high = _checked(vector, min_sq, max_sq)
    return 255.0 * (values - low) / (high - low) - 128.0


def _int8_values(vector: ArrayLike, min_sq: float, max_sq: float) -> np.ndarray:
    """The int8 values: the mapped float64 truncated toward zero, then clamped."""
    return np.clip(np.trunc(_mapped(vector, min_sq, max_sq)), -128.0, 127.0).astype(np.int8)


def quantize_int8(vector: ArrayLike, *, min_sq: float, max_sq: float) -> QuantizedVector:
    """Quantize one embedding vector to signed 8-bit integers.

    Args:
        vector: The embedding vector, one float per dimension. It is widened to float64 before the arithmetic;
            a float64 input is not narrowed to float32 first.
        min_sq: The range's low bound (the value mapped to ``-128``), finite.
        max_sq: The range's high bound (the value mapped to ``127``), finite and greater than ``min_sq``.

    Returns:
        :class:`QuantizedVector` with one int8 byte per element and ``dtype == "int8"``.

    Raises:
        DataError: The range is not finite or not ordered, the vector is not 1-D, or it holds a non-finite
            value.
    """
    return QuantizedVector(_int8_values(vector, min_sq, max_sq).tobytes(), "int8")


def quantize_uint8(vector: ArrayLike, *, min_sq: float, max_sq: float) -> QuantizedVector:
    """Quantize one embedding vector to unsigned 8-bit integers: ``int8 + 128``.

    Args:
        vector: The embedding vector, one float per dimension. It is widened to float64 before the arithmetic;
            a float64 input is not narrowed to float32 first.
        min_sq: The range's low bound, finite.
        max_sq: The range's high bound, finite and greater than ``min_sq``.

    Returns:
        :class:`QuantizedVector` with one uint8 byte per element and ``dtype == "uint8"``.

    Raises:
        DataError: The range is not finite or not ordered, the vector is not 1-D, or it holds a non-finite
            value.
    """
    int8 = _int8_values(vector, min_sq, max_sq)
    return QuantizedVector((int8.astype(np.int16) + 128).astype(np.uint8).tobytes(), "uint8")


def quantize_binary(vector: ArrayLike, *, min_sq: float, max_sq: float) -> QuantizedVector:
    """Pack one embedding vector's sign bits, eight per byte, MSB-first.

    A value ``>= 0`` sets its bit; a negative value clears it. The sign is the float vector's (``-0.0`` is not
    negative), not the truncated int8 value's. The last byte is zero-padded on the right when the dimension is
    not a multiple of eight.

    Args:
        vector: The embedding vector, one float per dimension. The range is validated for a uniform call
            contract, though the sign rule itself does not use it.
        min_sq: The range's low bound, finite.
        max_sq: The range's high bound, finite and greater than ``min_sq``.

    Returns:
        :class:`QuantizedVector` with ``ceil(dim / 8)`` bytes and ``dtype == "binary"``.

    Raises:
        DataError: The range is not finite or not ordered, the vector is not 1-D, or it holds a non-finite
            value.
    """
    values, _, _ = _checked(vector, min_sq, max_sq)
    return QuantizedVector(np.packbits(values >= 0, bitorder="big").tobytes(), "binary")


__all__ = [
    "EmbeddingType",
    "QuantizedVector",
    "quantize_binary",
    "quantize_int8",
    "quantize_uint8",
]
