"""A tiny safetensors writer for tests: one header, one data buffer, no dependency.

The projection head reads a checkpoint's ``*.safetensors`` through its own minimal reader (the product does
not depend on the ``safetensors`` package), so the tests build the file's bytes here -- the same layout the
library writes: an 8-byte little-endian header length, the JSON header, then the tensors' little-endian data
in header order. Only F32/F16/BF16/I64 are needed by the tests.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_DTYPES = {"F32": "<f4", "F16": "<f2", "I64": "<i8"}


def safetensors_bytes(tensors: dict[str, np.ndarray]) -> bytes:
    """The safetensors file bytes for ``tensors`` (name -> 2-D or 1-D array)."""
    header: dict[str, dict[str, object]] = {}
    payload = bytearray()
    for name, array in tensors.items():
        array = np.ascontiguousarray(array)
        kind = next((key for key, dtype in _DTYPES.items() if array.dtype == np.dtype(dtype)), None)
        if kind is None:
            raise ValueError(f"tests/_safetensors: unsupported dtype {array.dtype}")
        start = len(payload)
        payload += array.astype(np.dtype(_DTYPES[kind])).tobytes()
        header[name] = {"dtype": kind, "shape": list(array.shape), "data_offsets": [start, len(payload)]}
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(raw_header)) + raw_header + bytes(payload)


def write_safetensors(path: str | Path, tensors: dict[str, np.ndarray]) -> Path:
    """Write ``tensors`` as a safetensors file and return the path."""
    target = Path(path)
    target.write_bytes(safetensors_bytes(tensors))
    return target
