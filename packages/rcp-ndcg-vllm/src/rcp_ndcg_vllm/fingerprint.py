"""The recipe **behaviour fingerprint** (GPU-VALIDATION.md item 8): the SHA-256 of everything that can change
what the model returns.

One function, defined here once and reused by the observation-corpus writer, the wave runner's
re-record-changed-only mode and the fake engines' conformance suite. The inputs are exactly what item 8
names: the model id and revision, the ``serve`` block (overrides, pooler config, dtype, plugin name and
version, ``mm_processor_kwargs``, ....), the template file's bytes, the tokenizer's ``tokenizer.json``
SHA-256 and the client fields that shape the request (the product's own CONTENT classification of the
endpoint config -- ``rcp_ndcg.support.identity.identity_payload`` -- plus the media caps, which decide how
much media a request carries). rcp-ndcg's own internals are not in it: the emulators speak the wire, so
refactoring the package never needs a re-recording. Recipe identity (``id``) and harness metadata
(``notes``, ``status``, ``reference``, ``gates``) are out: a renamed recipe with the same behaviour keys
the same fingerprint.

:func:`fingerprint_inputs` returns every input **named** with its value (a flat map over dotted names), so
a staleness failure can name what changed -- ``behaviour diff`` reports and the conformance suite both
read it. The model layer of an observation corpus is keyed by the fingerprint (``<recipe>/<fingerprint>/``);
the protocol layer is keyed by the engine and its version.

The tokenizer's identity is the SHA-256 of the ``tokenizer.json`` bytes, resolved in this order (no silent
default: none resolvable is an error with a hint):

1. the recipe's ``client.tokenizer`` as a local file (a recipe-relative path, the harness's own
   :func:`~rcp_ndcg_vllm.equivalence.fitting.resolved_tokenizer_spec` rule);
2. a **tokenizer store** registered with :func:`use_tokenizer_store` -- a directory with an ``index.json``
   mapping the recipe's exact spec string to a vendored copy of the ``tokenizer.json`` (optionally
   ``.gz``-compressed), whose SHA-256 is verified on every read. Observation corpora vend their tokenizers
   this way so the conformance suite recomputes fingerprints offline;
3. the product's :func:`~rcp_ndcg.data.tokenizer.load_tokenizer` (a Hub id with its pinned revision, from
   the local cache or the Hub).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rcp_ndcg.data.tokenizer import TextTokenizer, load_tokenizer
from rcp_ndcg.support.identity import identity_payload
from rcp_ndcg_vllm.equivalence.fitting import resolved_tokenizer_spec
from rcp_ndcg_vllm.errors import HarnessError
from rcp_ndcg_vllm.recipe import Recipe

__all__ = [
    "FINGERPRINT_SCHEMA",
    "behaviour_fingerprint",
    "fingerprint_changes",
    "fingerprint_inputs",
    "load_recipe_tokenizer",
    "tokenizer_sha256",
    "use_tokenizer_store",
]

FINGERPRINT_SCHEMA = "rcp-fp/1"
"""The fingerprint rule's version, part of the hashed bytes: changing the input set or their
canonicalisation is a new schema (``rcp-fp/2``), so old and new corpora never collide in one key."""

_CLIENT_FIELDS_OUT = frozenset({"model", "revision", "recipe", "tokenizer"})
"""Client config fields out of the fingerprint: they name where and what, they cannot change what the
model returns (the served name and the revision are keyed separately as ``model``/``revision``; the
tokenizer's bytes as ``tokenizer_sha256``)."""

_CLIENT_FIELDS_IN = frozenset({"max_images", "max_videos"})
"""RUNTIME client fields back in: they decide how much media a request carries, so they shape the
request bytes even though they are runtime for an identity."""

_STORES: list[Path] = []
"""Registered tokenizer stores, searched in registration order (see :func:`use_tokenizer_store`)."""


def use_tokenizer_store(directory: str | Path) -> None:
    """Register a **tokenizer store**: a directory whose ``index.json`` maps tokenizer specs to vendored
    copies (see the module docstring). Idempotent per directory; resolution uses the first match.

    Args:
        directory: The store directory. ``<directory>/index.json`` is a mapping
            ``{"<spec>": {"file": "<name>", "sha256": "<hex>"}}`` where ``file`` is relative to the store.
    """
    path = Path(directory).resolve()
    if path not in _STORES:
        _STORES.append(path)


def _canonical(value: Any) -> str:
    """One fingerprint input's value as its canonical string (JSON for structure, raw for strings)."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _template_file_sha(recipe: Recipe) -> str:
    """``sha256:<hex>`` of the recipe's template file's bytes, or ``absent``. The file's *content* is input;
    the ``serve.chat_template`` field naming it is a ``serve`` input of its own."""
    name = recipe.serve.chat_template
    if name is None:
        return "absent"
    directory = recipe._dir
    if directory is None:  # pragma: no cover - load_recipe sets the directory
        raise HarnessError(f"recipe {recipe.id}: loaded without a directory; the template file cannot be resolved")
    path = directory / name
    try:
        data = path.read_bytes()
    except OSError as error:
        raise HarnessError(f"recipe {recipe.id}: the template file {path} cannot be read: {error}") from error
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _store_lookup(spec: str) -> tuple[bytes, str] | None:
    """The vendored ``tokenizer.json`` bytes and SHA-256 for ``spec`` in a registered store, else ``None``."""
    for store in _STORES:
        index_path = store / "index.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entry = index.get(spec) if isinstance(index, dict) else None
        if not isinstance(entry, dict):
            continue
        file, expected = entry.get("file"), entry.get("sha256")
        if not isinstance(file, str) or not isinstance(expected, str):
            raise HarnessError(f"tokenizer store {index_path}: entry for {spec!r} needs 'file' and 'sha256'")
        raw = (store / file).read_bytes()
        data = gzip.decompress(raw) if file.endswith(".gz") else raw
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise HarnessError(
                f"tokenizer store {index_path}: {file} hashes to {actual}, the index declares {expected} "
                f"for {spec!r}; the vendored copy is corrupt or stale"
            )
        return data, expected
    return None


def _load_bytes(recipe: Recipe) -> tuple[bytes, str]:
    """The recipe's ``tokenizer.json`` bytes and their SHA-256 (the resolution order of the module docstring)."""
    spec = resolved_tokenizer_spec(recipe)
    path = Path(spec)
    if path.is_file():
        data = path.read_bytes()
        return data, hashlib.sha256(data).hexdigest()
    found = _store_lookup(spec)
    if found is not None:
        return found
    try:
        tokenizer = load_tokenizer(spec)
    except Exception as error:
        raise HarnessError(
            f"recipe {recipe.id}: the tokenizer {spec!r} cannot be resolved ({error}); give client.tokenizer a "
            "local tokenizer.json path, vendor one in a tokenizer store registered with use_tokenizer_store, "
            "or make it loadable through rcp_ndcg.data.tokenizer.load_tokenizer (a Hub id needs its cache)"
        ) from error
    return b"", tokenizer.sha256  # the product's loader already hashed the bytes; only their hash is input


def tokenizer_sha256(recipe: Recipe) -> str:
    """The SHA-256 of the ``tokenizer.json`` the recipe's tokenizer resolves to.

    Args:
        recipe: The loaded recipe (its ``client.tokenizer`` names the tokenizer).

    Returns:
        The hex SHA-256 of the tokenizer's bytes.

    Raises:
        HarnessError: no local file, store entry or cached Hub copy resolves (the message names the recipe,
            the spec and the three ways to make it resolvable).
    """
    return _load_bytes(recipe)[1]


def load_recipe_tokenizer(recipe: Recipe) -> TextTokenizer:
    """The recipe's real tokenizer, from the same files the fingerprint hashes.

    One resolution, two consumers: the fingerprint reads the bytes' hash, the emulators and the golden
    replay count tokens with the loaded :class:`~rcp_ndcg.data.tokenizer.TextTokenizer` (the recipe's real
    tokenizer files, on CPU). A store or cache copy is loaded through the product's own loader semantics:
    truncation and padding reset at load.

    Args:
        recipe: The loaded recipe.

    Returns:
        The :class:`~rcp_ndcg.data.tokenizer.TextTokenizer` named by ``client.tokenizer``.

    Raises:
        HarnessError: the tokenizer cannot be resolved (same message as :func:`tokenizer_sha256`).
    """
    spec = resolved_tokenizer_spec(recipe)
    try:
        return load_tokenizer(spec)
    except Exception:
        data, sha = _load_bytes(recipe)
        if not data:  # pragma: no cover - resolution only hashes when the bytes are unreadable
            raise HarnessError(f"recipe {recipe.id}: the tokenizer {spec!r} resolved to a hash, not to bytes") from None
        return TextTokenizer.from_json(data, name=f"{spec}#{sha[:12]}")


def fingerprint_inputs(recipe: Recipe) -> dict[str, str]:
    """Every fingerprint input of ``recipe``, **named**, with its canonical value.

    The inputs (GPU-VALIDATION.md item 8): ``model`` and ``revision`` (the checkpoint); one ``serve.<field>``
    per field of the ``serve`` block (``model_dump`` on the frozen schema -- overrides, pooler config,
    dtype, plugin, ``mm_processor_kwargs``, ``max_model_len``, ``limit_mm_per_prompt``, ``extra_args``);
    ``template_file`` (the template file's bytes) and ``tokenizer_sha256``; and ``client.<field>`` for
    every client field that shapes the request (the product's CONTENT classification plus the media caps).
    ``fingerprint_schema`` records which rule hashed them.

    Args:
        recipe: The loaded recipe.

    Returns:
        The flat map, dotted names to canonical strings.

    Raises:
        HarnessError: the tokenizer or the template file cannot be resolved.
    """
    inputs: dict[str, str] = {"fingerprint_schema": FINGERPRINT_SCHEMA}
    inputs["model"] = recipe.model
    inputs["revision"] = recipe.revision
    for field, value in sorted(recipe.serve.model_dump(mode="json").items()):
        inputs[f"serve.{field}"] = _canonical(value)
    inputs["template_file"] = _template_file_sha(recipe)
    inputs["tokenizer_sha256"] = tokenizer_sha256(recipe)
    client = dict(identity_payload(recipe.client))
    for field in _CLIENT_FIELDS_IN:
        value = getattr(recipe.client, field, None)
        if value is not None:
            client[field] = value
    for field, value in sorted(client.items()):
        if field not in _CLIENT_FIELDS_OUT:
            inputs[f"client.{field}"] = _canonical(value)
    return inputs


def behaviour_fingerprint(recipe: Recipe) -> str:
    """The recipe's **behaviour fingerprint**: the SHA-256 of every input that can change what the model
    returns (see :func:`fingerprint_inputs`), as lowercase hex.

    The key the observation corpora and the verified fake engines are stored under
    (``<engine>-<version>/<recipe>/<fingerprint>/``): two recipes with the same fingerprint serve the same
    model behaviour on the wire, so one corpus and one emulator verify both.

    Args:
        recipe: The loaded recipe.

    Returns:
        The 64-character hex digest of ``FINGERPRINT_SCHEMA`` and the canonical inputs.

    Raises:
        HarnessError: an input cannot be resolved (the message names recipe and input).
    """
    inputs = fingerprint_inputs(recipe)
    payload = FINGERPRINT_SCHEMA + "\n" + json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_changes(before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    """The input names whose value differs between two :func:`fingerprint_inputs` maps, sorted (added and
    removed included): the names a staleness failure prints.

    Args:
        before: The recorded inputs (from a corpus manifest).
        after: The recomputed inputs.

    Returns:
        The sorted names, ``[]`` when nothing moved.
    """
    return sorted({name for name in set(before) | set(after) if before.get(name) != after.get(name)})