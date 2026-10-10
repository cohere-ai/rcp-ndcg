"""The project's one hashing recipe: canonical JSON bytes, then SHA-256.

Every identity in this project -- a run step's identity, a
:class:`~rcp_ndcg_core.schemas.JudgementFamily` key, a retrieval index identity, a
judgement ``record_id`` -- is "these two things are interchangeable" written as
a digest. It lives in the core because :class:`~rcp_ndcg_core.schemas.JudgementFamily`
computes its key here; the pipeline reaches it through
:mod:`rcp_ndcg.support.identity`. Two surfaces that hash *the same payload* must produce the
same digest, so there is exactly one serialiser and one digest function here.

What makes it canonical:

* keys sorted, no insertion-order dependence, compact separators;
* one declared encoder (Pydantic model, ``Path``, ``Enum``, aware
  ``datetime``/``date``, tuple) -- and **everything else is refused**.
  ``json.dumps(..., default=str)`` is how ``<object at 0x7f3c...>`` lands in a
  key that then changes between processes; a refusal is a bug report, a
  fallback ``str()`` is a silent non-deterministic identity;
* ``NaN``/``Infinity`` refused (not JSON, and ``NaN != NaN`` makes an identity
  that never matches itself), ``-0.0`` normalised to ``0.0`` (``-0.0 == 0.0``,
  so they must not hash apart);
* ``set``/``bytes`` refused -- a set has no canonical order to hash and bytes
  have no canonical JSON form; the caller must say which order/encoding it means.

``int`` and ``float`` are deliberately *not* unified: a count of ``1`` and a
fraction of ``1.0`` are different settings, so they must not share a key.

Text is UTF-8 (``ensure_ascii=False``): the bytes are the text, not an escaped
transliteration of it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import PurePath
from typing import Any

from pydantic import BaseModel


class UnhashableValueError(TypeError):
    """A payload contains a value with no declared canonical form.

    Raised instead of falling back to ``str(value)``: an identity computed from
    a repr is not reproducible, and a key that changes between processes reads
    as "nothing is cached" every time.
    """


def _refuse(where: str, value: Any, why: str) -> UnhashableValueError:
    return UnhashableValueError(
        f"cannot hash {where}: {why} (type {type(value).__name__}). "
        "Convert it in the caller so the identity says what it means."
    )


def canonical(value: Any, *, where: str = "payload") -> Any:
    """Normalise ``value`` into JSON-native data under the declared encoding."""
    # Enum first: a ``StrEnum``/``IntEnum`` member is also a str/int, and the
    # canonical form of an enum is its declared value, not the member.
    if isinstance(value, Enum):
        return canonical(value.value, where=f"{where}(enum)")
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _refuse(where, value, "non-finite floats have no JSON form")
        return 0.0 if value == 0.0 else value
    if isinstance(value, BaseModel):
        return canonical(value.model_dump(), where=where)
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise _refuse(where, value, "a naive datetime names no instant")
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        raise _refuse(where, value, "a set has no canonical order; pass a sorted list")
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise _refuse(where, value, "bytes have no canonical JSON form; pass a digest or a str")
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _refuse(f"{where}[{key!r}]", key, "mapping keys must be str")
            out[key] = canonical(item, where=f"{where}.{key}")
        return out
    if isinstance(value, Sequence):
        return [canonical(item, where=f"{where}[{index}]") for index, item in enumerate(value)]
    raise _refuse(where, value, "no declared canonical form")


def canonical_json(payload: Any) -> bytes:
    """The exact bytes every digest in this project is taken over."""
    return json.dumps(
        canonical(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def hash_payload(payload: Any) -> str:
    """Full SHA-256 hex digest of ``payload``'s canonical bytes (64 chars)."""
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def hash_text(text: str) -> str:
    """Full SHA-256 hex digest of a string -- prompt templates, rendered rubrics.

    The referent is the string itself, not a JSON encoding of it, so a template
    digest can be reproduced with ``sha256sum`` on the file's text.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_strings(items: Iterable[str]) -> str:
    """Full SHA-256 hex digest of a sequence of strings, streamed: each item is length-prefixed.

    For sequences too large to serialise as one JSON payload (a corpus's document ids or bodies). The length
    prefix makes ``["ab", "c"]`` and ``["a", "bc"]`` differ.
    """
    digest = hashlib.sha256()
    for item in items:
        encoded = item.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def combine_digests(*digests: str) -> str:
    """Full SHA-256 hex digest of several full hex digests, in order: one identity from its parts."""
    combined = hashlib.sha256()
    for digest in digests:
        combined.update(bytes.fromhex(digest))
    return combined.hexdigest()


def short(digest: str, length: int) -> str:
    """A truncated digest for human-facing display and pooling keys.

    Refuses anything that is not a full hex digest, so a short value can never
    be shortened twice and quietly become a shorter identity than it claims.
    """
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"short() takes a full sha256 hex digest, got {digest!r}")
    if not 1 <= length <= 64:
        raise ValueError(f"short() length must be in [1, 64], got {length}")
    return digest[:length]


__all__ = [
    "UnhashableValueError",
    "canonical",
    "canonical_json",
    "combine_digests",
    "hash_payload",
    "hash_strings",
    "hash_text",
    "short",
]
