"""The MRL head: one home for applying a selected output dimension to full-width vectors.

Matryoshka Representation Learning models ship a family of usable output widths, and there are exactly two
kinds of head in this package:

* **truncation** -- a Matryoshka-trained checkpoint's vectors are cut to ``k`` and renormalised
  (:func:`mrl_cut`, the card's own order: slice the raw output, then L2-normalise the slice). A model
  without a declared kind is never cut;
* **projection** -- the checkpoint's smaller sizes come from its own learned matrices (a
  ``*.safetensors`` file at the pinned revision, e.g. a chain of F32 matrices named by their target
  widths), applied client-side in float32 and then renormalised, because a slice of the full-width
  vector is not what the checkpoint trained (:class:`MrlProjection`).

The declared kind and set live on the endpoint config (``mrl_kind``, ``mrl_dims``, ``mrl_projection``);
this module is the one place that turns a declared ``k`` into vectors, so the embed and pooling clients
apply the head identically. Nothing here is defaulted: a ``k`` outside the declared set, a projection
kind without its source, and a ``k`` wider than the vectors are all refused with a typed error naming
the field.

The projection file is read through :mod:`rcp_ndcg.storage` (a local path, a ``gs://`` object, or an
``hf://org/model@revision/path`` Hub file) and parsed by the small safetensors reader below -- the
product does not depend on the ``safetensors`` package. Parsed matrices are cached per source: the
declared ``repo@revision`` is immutable, so one read serves every call.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Sequence
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rcp_ndcg.data.postprocess import l2_normalize
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.support.identity import FieldRole

MrlKind = Literal["truncation", "projection", "none"]
"""What kind of Matryoshka head a checkpoint has: a truncation-trained cut, a learned projection, or none."""

#: The safetensors dtype tags this reader understands, as little-endian numpy dtypes.
_SAFETENSORS_DTYPES: dict[str, str] = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U8": "u1",
    "BOOL": "?",
}


def mrl_cut(vectors: np.ndarray, mrl_dim: int) -> np.ndarray:
    """The Matryoshka cut of ``vectors`` to ``mrl_dim`` columns, renormalised.

    Cut-then-renormalise, the card's order: slicing after a normalisation
    would ship un-normalised cut vectors (the cut destroys unit-ness), so the
    slice is taken from the raw model output and the slice is normalised. A
    ``k`` wider than the vectors is refused by the head (``mrl_dim`` is
    validated against the declared set and the observed width); this function
    itself slices what it is given.

    Args:
        vectors: The model's vectors, one row per vector.
        mrl_dim: The declared Matryoshka output size.

    Returns:
        The cut, contiguous, L2-normalised (the caller's ``normalize`` step is then idempotent).
    """
    cut = np.ascontiguousarray(vectors[:, :mrl_dim])
    return l2_normalize(cut)


class MrlProjection(BaseModel):
    """Where a projection-kind head's learned matrices live, and which matrix chain each ``k`` applies.

    Attributes:
        source: The safetensors file holding the learned matrices: a storage URI
            (``hf://org/model@revision/projections.safetensors``, a ``gs://`` object, an ``https://`` URL)
            or a local path. The declared revision is part of the URI, so the bytes are immutable.
        chains: ``k`` -> the tensor names applied in order, each a matrix ``(in_width, out_width)`` in the
            file, when the file's naming needs spelling out (an integer ``k`` is accepted and normalised to
            its string form, the identity payload's canonical spelling). ``None`` (the default) derives
            each ``k``'s chain from the declared dimensions: the tensors named for every declared dimension
            at or above ``k``, widest first -- the convention of a chained projection file whose tensor
            names are their target widths.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Both fields decide what the head computes: the file and the chains it applies.
    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "source": FieldRole.CONTENT,
        "chains": FieldRole.CONTENT,
    }

    source: str = Field(min_length=1)
    chains: dict[str, tuple[str, ...]] | None = None

    @field_validator("chains", mode="before")
    @classmethod
    def _chains_keys_are_strings(cls, value: Any) -> Any:
        """YAML and JSON spell a chain's ``k`` as an integer or as a string; both normalise to the string
        form, so ``{2: ["t"]}`` and ``{"2": ["t"]}`` are the same declaration and the identity payload's
        canonical form (which requires string keys) always hashes."""
        if isinstance(value, dict):
            return {str(key): names for key, names in value.items()}
        return value

    @model_validator(mode="after")
    def _chains_are_non_empty(self) -> MrlProjection:
        """A chain with no tensor, or a non-positive ``k``, is a mistyped declaration: refused, never ignored."""
        if self.chains is not None:

            def bad_key(key: str) -> bool:
                # ``str.isdigit`` is true for superscripts and other non-ASCII digits that ``int`` cannot
                # parse, so the ASCII gate comes first and ``int(key)`` never sees one.
                return not (key.isascii() and key.isdigit()) or key != str(int(key)) or int(key) < 1

            bad = sorted(key for key, names in self.chains.items() if bad_key(key) or not names)
            if bad:
                raise ValueError(
                    f"mrl_projection.chains: {bad} must be positive integer output dimensions with at least "
                    "one tensor name each"
                )
        return self

    def chain_for(self, k: int, dims: Sequence[int]) -> tuple[str, ...]:
        """The tensor names applied in order to reach ``k`` from the full width.

        An explicit ``chains`` entry wins; otherwise the chain is the tensors named for every declared
        dimension at or above ``k``, widest first.

        Args:
            k: The selected output dimension.
            dims: The declared Matryoshka set (``mrl_dims``).

        Returns:
            The ordered tensor names.

        Raises:
            ConfigError: ``chains`` is explicit and has no entry for ``k``, or no declared dimension at or
                above ``k`` names a matrix (the set and the file disagree).
        """
        if self.chains is not None:
            names = self.chains.get(str(k))
            if names is None:
                raise ConfigError(
                    f"mrl_projection.chains has no chain for k={k}: the selected dimension would be applied "
                    "with no learned matrix",
                    hint=f"add a chains entry for {k} under mrl_projection (the tensor names in order), "
                    "or drop chains to derive each chain from mrl_dims",
                )
            return tuple(names)
        names = tuple(str(dim) for dim in sorted(set(dims), reverse=True) if dim >= k)
        if not names:
            raise ConfigError(
                f"mrl_projection derives each k's chain from mrl_dims, and no declared dimension is at or "
                f"above k={k}: the chain would be empty",
                hint="declare the full set of the card's output dimensions in mrl_dims (the widest matrix "
                "first), or spell the chain out in mrl_projection.chains",
            )
        return names


def _read_safetensors(data: bytes) -> dict[str, np.ndarray]:
    """The tensors of a safetensors buffer: the JSON header, then each tensor's slice of the data.

    The format is fixed and tiny: an 8-byte little-endian header length, the JSON header (tensor name ->
    ``{dtype, shape, data_offsets}``), then the little-endian tensor data. Every dtype the product's
    projection files use is understood; an unknown one is refused naming the tensor.

    Raises:
        DataError: the buffer is truncated, the header is not JSON, or a tensor names an unsupported dtype
            or out-of-range offsets.
    """
    if len(data) < 8:
        raise DataError(
            "the projection file is shorter than a safetensors header (8 bytes)",
            hint="point mrl_projection.source at a complete *.safetensors file",
        )
    (header_length,) = struct.unpack("<Q", data[:8])
    header_end = 8 + int(header_length)
    if header_end > len(data):
        raise DataError(
            f"the projection file's safetensors header claims {header_length} bytes but the file holds {len(data) - 8}",
            hint="point mrl_projection.source at a complete *.safetensors file",
        )
    try:
        header = json.loads(data[8:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataError(
            f"the projection file's safetensors header is not JSON: {exc}",
            hint="point mrl_projection.source at a complete *.safetensors file",
        ) from exc
    tensors: dict[str, np.ndarray] = {}
    if not isinstance(header, dict):
        raise DataError(
            "the projection file's safetensors header is not a JSON object",
            hint="point mrl_projection.source at a complete *.safetensors file",
        )
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise DataError(f"the projection file's tensor {name!r} has no safetensors entry")
        tag = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if tag == "BF16":
            dtype: Any = np.dtype("<u2")
        elif isinstance(tag, str) and tag in _SAFETENSORS_DTYPES:
            dtype = np.dtype(_SAFETENSORS_DTYPES[tag])
        else:
            raise DataError(
                f"the projection file's tensor {name!r} has dtype {tag!r}, which this reader does not know",
                hint=f"one of {sorted(_SAFETENSORS_DTYPES)} or BF16",
            )
        if (
            not isinstance(shape, list)
            or any(type(dimension) is not int or dimension < 0 for dimension in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            raise DataError(
                f"the projection file's tensor {name!r} lacks a shape of non-negative integers or a "
                "two-entry data_offsets"
            )
        start, end = int(offsets[0]), int(offsets[1])
        count = int(np.prod(shape, dtype=np.int64))
        # The span must be exactly the tensor's bytes: a span that is too wide silently reads the next
        # tensor's bytes, and one that is too narrow overflows ``np.frombuffer`` -- both refused here.
        if start < 0 or end < start or end - start != count * dtype.itemsize or header_end + end > len(data):
            raise DataError(
                f"the projection file's tensor {name!r} declares shape {shape} with data_offsets "
                f"[{start}, {end}], which is not {count} x {dtype.itemsize} bytes inside the file",
                hint="point mrl_projection.source at a complete *.safetensors file",
            )
        try:
            flat = np.frombuffer(data, dtype=dtype, count=count, offset=header_end + start)
        except ValueError as exc:
            raise DataError(f"the projection file's tensor {name!r} cannot be read: {exc}") from exc
        if tag == "BF16":
            # bfloat16 is the top half of a float32: widen the uint16 bits and shift them into place.
            array = (flat.astype(np.uint32) << 16).view(np.float32)
        else:
            array = flat
        tensors[name] = array.reshape(shape).copy()
    return tensors


_PROJECTION_FILES: dict[str, tuple[str, dict[str, np.ndarray]]] = {}
"""Parsed projection files by source URI: ``(sha256, tensors)``, read once (the declared revision is immutable)."""


def projection_tensors(source: str) -> tuple[str, dict[str, np.ndarray]]:
    """The tensors of the safetensors file at ``source``, with its SHA-256, read through the storage layer.

    The file is read once per source: the declaration carries the checkpoint's pinned revision, so the
    bytes cannot change under one process. The digest is the file's content identity (recorded beside the
    parsed tensors for callers that key on it).

    Raises:
        DataError: the source cannot be read or is not a safetensors buffer.
    """
    cached = _PROJECTION_FILES.get(source)
    if cached is not None:
        return cached
    from rcp_ndcg.storage import read_bytes

    try:
        data = read_bytes(source)
    except Exception as exc:  # the storage layer's typed errors (a missing file, a missing backend)
        raise DataError(
            f"mrl_projection.source {source!r} cannot be read: {exc}",
            hint="point it at the checkpoint's projections file (a local path, a storage URI, or an "
            "hf://org/model@revision/path.safetensors Hub file)",
        ) from exc
    digest = hashlib.sha256(data).hexdigest()
    tensors = _read_safetensors(data)
    _PROJECTION_FILES[source] = (digest, tensors)
    return digest, tensors


def clear_projection_cache() -> None:
    """Forget every parsed projection file (the tests' seam; a process never needs it)."""
    _PROJECTION_FILES.clear()


class MrlHead:
    """A configured Matryoshka head: the declared kind, set/range and (projection) learned matrices.

    Args:
        kind: ``"truncation"``, ``"projection"`` or ``"none"`` (the endpoint config's ``mrl_kind``).
        dims: The declared set of supported output dimensions (``mrl_dims``); empty for ``"none"``.
        mrl_range: The declared closed range of output dimensions (``mrl_range``), for a card whose prose
            gives ``[min, max]`` rather than a table; one of ``dims`` and ``mrl_range`` is declared.
        projection: Where the learned matrices live (``mrl_projection``); required for ``"projection"``.

    Raises:
        ConfigError: a projection kind without its source.
    """

    def __init__(
        self,
        *,
        kind: MrlKind = "none",
        dims: Sequence[int] = (),
        mrl_range: tuple[int, int] | None = None,
        projection: MrlProjection | None = None,
    ) -> None:
        if kind == "projection" and projection is None:
            raise ConfigError(
                "MrlHead kind 'projection' needs its projection source (mrl_projection): the checkpoint's "
                "learned matrices are not truncation slices",
                hint="declare mrl_projection (the safetensors source and its per-k chains), or use "
                "mrl_kind: truncation for a Matryoshka-trained checkpoint",
            )
        self.kind = kind
        self.dims = tuple(dims)
        self.mrl_range = tuple(mrl_range) if mrl_range is not None else None
        self.projection = projection

    @property
    def declaration(self) -> str:
        """The declaration as a refusal names it (``mrl_dims (...)`` or ``mrl_range [min, max]``)."""
        if self.dims:
            return f"mrl_dims {self.dims}"
        if self.mrl_range is not None:
            return f"mrl_range [{self.mrl_range[0]}, {self.mrl_range[1]}]"
        return "mrl_dims/mrl_range (none declared)"

    def supports(self, k: int) -> bool:
        """Whether ``k`` is in the declared set, or in the declared closed range."""
        if k in self.dims:
            return True
        return self.mrl_range is not None and self.mrl_range[0] <= k <= self.mrl_range[1]

    def _check(self, k: int, width: int) -> None:
        """The one selection check: a declared kind, a ``k`` in the declaration, and a ``k`` no wider."""
        if self.kind == "none":
            raise ConfigError(
                "mrl_dim selects a Matryoshka output, but mrl_kind is 'none': the checkpoint declares no "
                "Matryoshka head",
                hint="declare mrl_kind: truncation with mrl_dims or mrl_range (the card's set) or mrl_kind: "
                "projection with mrl_projection, or drop mrl_dim",
            )
        if not self.supports(k):
            raise ConfigError(
                f"mrl_dim {k} is not in the declared {self.declaration}: the run would select an output "
                "dimension the model's card does not declare",
                hint=f"select a k in {self.declaration}, or widen the declaration when the card supports it",
            )
        if k > width:
            raise ConfigError(
                f"mrl_dim {k} is wider than the vectors ({width} wide): a Matryoshka head can only narrow",
                hint=f"select a k from {self.declaration} below the checkpoint's own width, or drop mrl_dim",
            )

    def apply(self, vectors: np.ndarray, k: int) -> np.ndarray:
        """Apply the selected dimension ``k`` to full-width ``vectors`` (one row per vector).

        Truncation cuts and renormalises (:func:`mrl_cut`); projection applies the learned chain in
        float32 and renormalises. The declared ``k`` is checked against the set and the observed width.

        Args:
            vectors: The full-width vectors, ``(rows, full_width)``, any float dtype.
            k: The selected output dimension, in the declared set or closed range.

        Returns:
            The head's output, same number of rows, ``k`` wide (projection: the chain's last width).

        Raises:
            ConfigError: ``mrl_kind`` is ``"none"``, ``k`` is outside the declaration, or ``k`` is wider
                than ``vectors``.
            DataError: the projection file cannot be read or its matrices do not line up with the vectors.
        """
        array = np.asarray(vectors)
        if array.ndim != 2:
            raise DataError(f"the MRL head needs a 2-D (rows, width) buffer, got shape {array.shape}")
        self._check(k, int(array.shape[1]))
        if self.kind == "truncation":
            return mrl_cut(array, k)
        return self._project(array, k)

    def _project(self, vectors: np.ndarray, k: int) -> np.ndarray:
        """The learned chain for ``k``, in float32, renormalised (the model's own order)."""
        assert self.projection is not None  # the constructor refuses a projection kind without one
        names = self.projection.chain_for(k, self.dims)
        _, tensors = projection_tensors(self.projection.source)
        current = np.ascontiguousarray(vectors, dtype=np.float32)
        for name in names:
            matrix = tensors.get(name)
            if matrix is None:
                raise DataError(
                    f"mrl_projection.source {self.projection.source!r} has no tensor {name!r} for k={k}; "
                    f"it holds {sorted(tensors)}",
                    hint="correct mrl_projection (the source file, or the chains' tensor names) for this "
                    "checkpoint revision",
                )
            if matrix.ndim != 2 or int(matrix.shape[0]) != int(current.shape[1]):
                raise DataError(
                    f"mrl_projection tensor {name!r} has shape {tuple(matrix.shape)} and cannot map a "
                    f"{current.shape[1]}-wide vector: the chain's widths do not line up",
                    hint="check mrl_projection against the checkpoint's own matrices (each chain's first "
                    "matrix must take the full width)",
                )
            current = current @ np.asarray(matrix, dtype=np.float32)
        if int(current.shape[1]) != k:
            raise DataError(
                f"the mrl_projection chain for k={k} ends {current.shape[1]}-wide: the learned chain must "
                "produce exactly k",
                hint="check mrl_projection.chains against the checkpoint's own matrices",
            )
        return l2_normalize(current)


__all__ = [
    "MrlHead",
    "MrlKind",
    "MrlProjection",
    "clear_projection_cache",
    "mrl_cut",
    "projection_tensors",
]
