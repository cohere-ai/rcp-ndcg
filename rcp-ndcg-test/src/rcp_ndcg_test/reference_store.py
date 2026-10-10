"""The stored reference outputs (owner decision 35 item 3): stage 2's reference results as artifacts.

A reference run is expensive (a model load and a full scoring/embedding pass) and deterministic for a
fixed environment: the same family reference, at the same variant revision, over the same pairs, on the
same device and dtype must produce the same scores or vectors.  This module stores those outputs under a
content-addressed key so a wave computes only the missing or stale ones and stage 2 compares the engine
against the stored reference.

The key is :func:`reference_fingerprint`: the SHA-256 of the named inputs the brief fixes -- the family's
reference hash, the variant revision, the pairs-file hash, the reference environment's lock hash, the
device and the dtype -- plus the model, the role's mode and the reference block's own facts.  Staleness
reuses the observation corpora's mechanism, one home (:mod:`rcp_ndcg_test.fingerprint`): a store entry
carries a ``manifest.json`` with its named inputs and :func:`~rcp_ndcg_test.fingerprint.fingerprint_changes`
names exactly what moved when a new run no longer matches a stored one.  An entry is immutable: the
fingerprint is its directory name, ``output.json`` is the reference's document and the manifest pins the
output's SHA-256 (a tampered entry is refused, never compared).

Where the store lives is the caller's choice (the wave runner keeps it under the wave's output so it
uploads with everything else; a previous wave's downloaded store can be passed to reuse it).  Size
arithmetic for the README: a dense 4096-dim vector is 16 KB in float32, so a 50-row pairs file with 5
documents is ~4 MB per recipe; a late-interaction model's per-token vectors are the large ones (16k
tokens x 128 dims x 4 B ~ 8 MB per text, hundreds of MB per recipe), which is why the store is a wave
artifact, not a committed file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rcp_ndcg_test.errors import HarnessError

from .fingerprint import fingerprint_changes

__all__ = [
    "REFERENCE_FINGERPRINT_SCHEMA",
    "ReferenceKey",
    "load_stored",
    "reference_fingerprint",
    "reference_fingerprint_inputs",
    "save_stored",
    "store_path",
]

REFERENCE_FINGERPRINT_SCHEMA = "rcp-ref-fp/1"
"""The reference-output key rule's version: changing the input set is a new schema, so old and new
entries never collide in one key."""

_MANIFEST = "manifest.json"
_OUTPUT = "output.json"


def _canonical(value: Any) -> str:
    """One key input's canonical string (JSON for structure, raw for strings)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(data: bytes) -> str:
    """The hex SHA-256 of some bytes."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ReferenceKey:
    """A stored reference output's key: its fingerprint and the named inputs it was computed from.

    Attributes:
        fingerprint: The 64-character hex digest (the store directory's name).
        inputs: The flat ``{name: canonical value}`` map, recorded in the entry's manifest.
    """

    fingerprint: str
    inputs: dict[str, str]


def reference_fingerprint_inputs(
    *,
    reference_sha256: str,
    revision: str,
    pairs_sha256: str,
    lock_sha256: str | None,
    device: str,
    dtype: str | None,
    model: str,
    mode: str,
    reference: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Every named input of a stored reference output's key.

    Inputs: the family reference hash (the entry file's bytes), the variant revision, the hash of the
    pairs the reference receives, the reference environment's lock hash (``None`` when the reference
    runs outside a family environment), the device and dtype, the model id, the reference mode
    (``score`` or ``embed``) and the resolved recipe's ``reference`` block (its score scale, known
    deviations, ... -- whatever shapes the output).  Output: the flat map, dotted names to canonical
    strings.  Units: none.
    """
    inputs: dict[str, str] = {
        "reference_schema": REFERENCE_FINGERPRINT_SCHEMA,
        "model": model,
        "revision": revision,
        "mode": mode,
        "reference_file": reference_sha256,
        "pairs_sha256": pairs_sha256,
        "environment_lock": lock_sha256 or "unmanaged",
        "device": device,
        "dtype": dtype or "unset",
    }
    for field, value in sorted((reference or {}).items()):
        inputs[f"reference.{field}"] = _canonical(value)
    return inputs


def reference_fingerprint(inputs: dict[str, str]) -> str:
    """The SHA-256 of a named-input map (the key's fingerprint), lowercase hex."""
    payload = REFERENCE_FINGERPRINT_SCHEMA + "\n" + json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def reference_key(**kwargs: Any) -> ReferenceKey:
    """One call of :func:`reference_fingerprint_inputs` plus :func:`reference_fingerprint`."""
    inputs = reference_fingerprint_inputs(**kwargs)
    return ReferenceKey(fingerprint=reference_fingerprint(inputs), inputs=inputs)


def store_path(root: str | Path, fingerprint: str) -> Path:
    """The directory of one stored output: ``<root>/<fingerprint>``."""
    return Path(root) / fingerprint


def _manifest_of(directory: Path) -> dict[str, Any] | None:
    """A store entry's manifest, or ``None`` when the directory holds none."""
    path = directory / _MANIFEST
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def load_stored(root: str | Path, key: ReferenceKey) -> tuple[dict[str, Any] | None, list[str]]:
    """Load a stored reference output for ``key``, or report what moved.

    Inputs: the store root and the key.  Output: ``(document, changed_inputs)``.  ``document`` is the
    stored reference document when the entry at ``<root>/<key fingerprint>`` exists, its manifest names
    exactly ``key.inputs`` and ``output.json`` hashes to the manifest's ``output_sha256``; ``None``
    otherwise.  ``changed_inputs`` names the inputs that differ from the closest stored entry of the
    same recipe and mode (via :func:`~rcp_ndcg_test.fingerprint.fingerprint_changes`; ``[]`` when the
    entry was loaded, or when nothing comparable is stored).  A corrupt entry (a hash mismatch) raises
    :class:`HarnessError` -- a tampered store is never silently recomputed and compared.  Units: none.
    """
    directory = store_path(root, key.fingerprint)
    manifest = _manifest_of(directory)
    if manifest is not None and dict(manifest.get("inputs") or {}) == key.inputs:
        output = directory / _OUTPUT
        try:
            data = output.read_bytes()
        except OSError as error:
            raise HarnessError(f"the stored reference output {output} cannot be read: {error}") from error
        expected = str(manifest.get("output_sha256") or "")
        if expected and _sha256_bytes(data) != expected:
            raise HarnessError(
                f"the stored reference output {output} does not hash to its manifest ({expected}); "
                "the store is corrupt or was hand-edited"
            )
        try:
            document = json.loads(data)
        except ValueError as error:
            raise HarnessError(f"the stored reference output {output} is not JSON: {error}") from error
        if not isinstance(document, dict):
            raise HarnessError(f"the stored reference output {output} must be a JSON object")
        return document, []
    # Not this key: find the closest stored entry of the same model and mode and name what moved (the
    # revision is deliberately not part of the filter, so a revision bump is named too).
    identity = {"model": key.inputs.get("model"), "mode": key.inputs.get("mode")}
    changed: list[str] = []
    root_path = Path(root)
    if root_path.is_dir():
        for candidate in sorted(root_path.iterdir()):
            other = _manifest_of(candidate)
            if other is None:
                continue
            other_inputs = dict(other.get("inputs") or {})
            if any(other_inputs.get(name) != value for name, value in identity.items()):
                continue
            for name in fingerprint_changes(other_inputs, key.inputs):
                if name not in changed:
                    changed.append(name)
    return None, sorted(changed)


def save_stored(
    root: str | Path,
    key: ReferenceKey,
    document: dict[str, Any],
    *,
    family: str | None = None,
    reference_python: str | None = None,
) -> Path:
    """Store one reference document under its key (immutable: an existing entry is never rewritten).

    Inputs: the store root, the key, the reference's document, and the family/ interpreter (recorded for
    review; never part of the key).  Output: the entry directory (``<root>/<fingerprint>``) with
    ``output.json`` and ``manifest.json`` (the named inputs, the output's SHA-256, the creation time and
    the family).  An entry that already exists is left untouched.  Units: none.
    """
    directory = store_path(root, key.fingerprint)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / _OUTPUT
    manifest_path = directory / _MANIFEST
    if manifest_path.is_file() and output.is_file():
        return directory
    data = json.dumps(document, indent=1, sort_keys=True).encode("utf-8") + b"\n"
    output.write_bytes(data)
    manifest = {
        "schema": REFERENCE_FINGERPRINT_SCHEMA,
        "fingerprint": key.fingerprint,
        "inputs": key.inputs,
        "output_sha256": _sha256_bytes(data),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "family": family,
        "reference_python": reference_python,
    }
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return directory
