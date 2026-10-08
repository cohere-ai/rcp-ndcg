"""The recipe **behaviour fingerprint** (GPU-VALIDATION.md item 8): the SHA-256 of everything that can change
what the model returns.

One function, defined here once and reused by the observation-corpus writer, the wave runner's
re-record-changed-only mode and the fake engines' conformance suite. The inputs are exactly what item 8
names: the model id and revision, the ``serve`` block (overrides, pooler config, dtype, the plugin's pip
spec, ``mm_processor_kwargs``, ...), the template file's bytes, the tokenizer's ``tokenizer.json``
SHA-256 and the client fields that change the request bytes (:data:`CLIENT_FIELDS` classifies every
client config field; client-side post-processing of the reply is out, request packing and the media caps
are in). rcp-ndcg's own internals are not in it: the emulators speak the wire, so
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
from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
from rcp_ndcg.support.identity import identity_payload
from rcp_ndcg_vllm.equivalence.fitting import resolved_tokenizer_spec
from rcp_ndcg_vllm.errors import HarnessError
from rcp_ndcg_vllm.recipe import Recipe

_CLIENT_MODELS: dict[str, type] = {
    "embed": EmbeddingEndpoint,
    "multi_vector": PoolingEndpoint,
    "rerank": RerankEndpoint,
}
"""The product endpoint model per recipe role: the plain client block validates against it when the
behaviour fingerprint reads its CONTENT fields."""

__all__ = [
    "CLIENT_FIELDS",
    "FINGERPRINT_SCHEMA",
    "behaviour_fingerprint",
    "fingerprint_changes",
    "fingerprint_inputs",
    "load_recipe_tokenizer",
    "stored_tokenizer",
    "tokenizer_sha256",
    "use_tokenizer_store",
]

FINGERPRINT_SCHEMA = "rcp-fp/3"
"""The fingerprint rule's version, part of the hashed bytes: changing the input set or their
canonicalisation is a new schema (``rcp-fp/4``), so old and new corpora never collide in one key.
Version 3 keys exactly the client fields that change the request bytes (:data:`CLIENT_FIELDS`):
client-side post-processing of the reply is out, request packing (``batch_size``) and the media caps
are in."""

CLIENT_FIELDS: dict[str, str] = {
    # request: the field changes the bytes the client sends (and with them what the model returns)
    "api": "request",  # the wire adapter: route and body shape
    "max_tokens": "request",  # the client cut: the text sent
    "query_max_tokens": "request",
    "document_max_tokens": "request",  # the per-document cap: the document text sent
    "template": "request",  # the rendered prompt
    "on_overflow": "request",  # cut, chunk or refuse: which requests are sent
    "chunk": "request",
    "empty_doc": "request",  # an empty document sent, sent as text, or never sent
    "empty_doc_text": "request",
    "empty_query": "request",
    "request_shape": "request",  # text, messages or token ids on the wire
    "add_generation_prompt": "request",  # sent on the messages route: the engine's frame gains its header
    "query_prompt": "request",
    "doc_prompt": "request",
    "dimensions": "request",  # sent in the /v1/embeddings body
    "instruction": "request",  # folded into the prompt, or sent as a field or a system message
    "use_activation": "request",  # sent in the /rerank body
    "listwise": "request",  # one N-passage request instead of one request per pair
    "embed_dtype": "request",  # sent in the /pooling body
    "image_processor": "request",  # the client resizes the media it sends
    "image_policy": "request",
    "video_policy": "request",
    "media_sides": "request",
    "max_images": "request",  # how much media one request carries
    "max_videos": "request",
    "batch_size": "request",  # request packing: a bf16 batch's numbers can depend on its composition
    # naming: keyed elsewhere or not behaviour at all
    "model": "naming",  # keyed as ``model`` from the recipe
    "revision": "naming",  # keyed as ``revision`` from the recipe
    "recipe": "naming",
    "tokenizer": "naming",  # its bytes are keyed as ``tokenizer_sha256``
    # post_processing: applied to the reply after it arrives; neither the request nor the model output moves
    "normalize": "post_processing",
    "aggregation": "post_processing",
    "dim": "post_processing",  # the width the client checks the reply against
    "mrl_dim": "post_processing",  # the client cuts and renormalises the reply
    "document_skip_token_ids": "post_processing",
    "outputs": "post_processing",  # how the client reads one input's outputs
    # transport: where, how fast and how often; never what
    "base_url": "transport",
    "api_key_env": "transport",
    "headers_env": "transport",
    "concurrency": "transport",
    "timeout_s": "transport",
    "connect_timeout_s": "transport",
    "max_retries": "transport",
    "wait_on_outage_s": "transport",
}
"""Every client config field of the three role configs, classified (GPU-VALIDATION.md item 8 keys "the
client fields that shape the request"). Only ``request`` fields are fingerprint inputs: they change the
request bytes, and with them what the model returns. ``post_processing`` fields act on the reply on the
client, ``naming`` fields are keyed elsewhere (``model``, ``revision``, ``tokenizer_sha256``) and
``transport`` fields decide where and how often a request goes. A field missing here is refused by
:func:`fingerprint_inputs`, so a new product field forces a decision instead of falling silently in or
out."""

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


def stored_tokenizer(spec: str) -> tuple[bytes, str] | None:
    """The vendored ``tokenizer.json`` of ``spec`` from the registered tokenizer stores.

    Args:
        spec: The tokenizer spec exactly as a recipe's ``client.tokenizer`` names it (``org/model@<revision>``).

    Returns:
        ``(bytes, sha256)`` from the first registered store whose index names ``spec`` (the bytes verified
        against the index's SHA-256), else ``None``.

    Raises:
        HarnessError: an index entry lacks ``file`` or ``sha256``, or the vendored copy does not hash to it.
    """
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
    found = stored_tokenizer(spec)
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
    every client field :data:`CLIENT_FIELDS` classifies as ``request`` (it changes the request bytes).
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
    client = recipe.client
    unclassified = sorted(set(client) - set(CLIENT_FIELDS))
    if unclassified:
        raise HarnessError(
            f"recipe {recipe.id}: the client fields {unclassified} are not classified in "
            "rcp_ndcg_vllm.fingerprint.CLIENT_FIELDS; decide whether each changes the request bytes "
            "('request', a fingerprint input) or not ('post_processing', 'naming', 'transport')"
        )
    # The product's canonical form of the block's CONTENT fields: the endpoint model validates the plain
    # data and identity_payload reads its CONTENT role declarations (R30: never a local re-derivation).
    content = identity_payload(_CLIENT_MODELS[recipe.role].model_validate(client))
    for field in sorted(client):
        if CLIENT_FIELDS[field] != "request":
            continue
        value = content[field] if field in content else client[field]  # RUNTIME request fields: as declared
        if value is not None:
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
