"""The recipe **behaviour fingerprint** (GPU-VALIDATION.md item 8): the SHA-256 of everything that can change
what the model returns.

One function, defined here once and reused by the observation-corpus writer, the wave runner's
re-record-changed-only mode and the fake engines' conformance suite. The inputs are item 8's, extended by
``rcp-fp/4``: the model id and revision, the engine image and its version floor (the processing the engine
performs is versioned by them), the ``serve`` block (overrides, pooler config, dtype, the plugin's pip spec
and the architectures and patches it declares, ``mm_processor_kwargs``, ...), the source hash of every
engine-side plugin module the recipe runs, the template file's bytes, the tokenizer's ``tokenizer.json``
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
   :func:`~rcp_ndcg_test.equivalence.fitting.resolved_tokenizer_spec` rule);
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
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from rcp_ndcg_vllm.models import ARCHITECTURE_MODULES, PLUGIN_ENGINE_MODULES
from rcp_ndcg_vllm.patches import PATCH_MODULES
from rcp_ndcg_vllm.recipe import Recipe, plugin_distribution_name

from rcp_ndcg.data.tokenizer import TextTokenizer, load_tokenizer
from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
from rcp_ndcg.judging.client import JudgeConfig
from rcp_ndcg.support.identity import identity_payload
from rcp_ndcg_test.equivalence.fitting import resolved_tokenizer_spec
from rcp_ndcg_test.errors import HarnessError

_CLIENT_MODELS: dict[str, type] = {
    "embed": EmbeddingEndpoint,
    "multi_vector": PoolingEndpoint,
    "rerank": RerankEndpoint,
    "judge": JudgeConfig,
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
    "plugin_module_hashes",
    "sha256_digest",
    "stored_tokenizer",
    "tokenizer_sha256",
    "use_tokenizer_store",
]

FINGERPRINT_SCHEMA = "rcp-fp/4"
"""The fingerprint rule's version, part of the hashed bytes: changing the input set or their
canonicalisation is a new schema (``rcp-fp/5``), so old and new corpora never collide in one key.
Version 4 keys the engine image and its version floor (``engine.image``, ``engine.min_version``), the
plugin code the recipe's engine runs (``plugin_sha256.<module>``: the shared entry modules, every declared
architecture's modules and every opted-in patch's module) and, as version 3 did, exactly the client fields
that change the request bytes (:data:`CLIENT_FIELDS`): client-side post-processing of the reply is out,
request packing (``batch_size``) and the media caps are in."""

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
    "title": "request",  # MTEB's title join, or the title as its own part: the document text sent
    "use_activation": "request",  # sent in the /rerank body
    "listwise": "request",  # one N-passage request instead of one request per pair
    "embed_dtype": "request",  # sent in the /pooling body
    "image_processor": "request",  # the client resizes the media it sends
    "image_policy": "request",
    "video_policy": "request",
    "media_sides": "request",
    "max_images": "request",  # how much media one request carries
    "max_videos": "request",
    "media_head_as_system": "request",  # the head rides the request as a system message
    "batch_size": "request",  # request packing: a bf16 batch's numbers can depend on its composition
    # the judge's fields (JudgeConfig): the sampling, the answer schema and the window budget change the
    # request bytes; the floating-alias switch is a config-validation rule, not behaviour
    "temperature": "request",
    "max_output_tokens": "request",  # sent as max_completion_tokens
    "extra_body": "request",  # further request fields (chat_template_kwargs, top_p, ...)
    "context_tokens": "request",  # sizes the per-window text budget: the text sent
    "decoding": "request",  # response_format json_schema, or free text
    "allow_floating_model": "naming",
    # naming: keyed elsewhere or not behaviour at all
    "model": "naming",  # keyed as ``model`` from the recipe
    "revision": "naming",  # keyed as ``revision`` from the recipe
    "recipe": "naming",
    "tokenizer": "naming",  # its bytes are keyed as ``tokenizer_sha256``
    # post_processing: applied to the reply after it arrives; neither the request nor the model output moves
    "normalize": "post_processing",
    "aggregation": "post_processing",
    "dim": "post_processing",  # the width the client checks the reply against
    "mrl_kind": "post_processing",  # the declared head kind: applied to the reply after it arrives
    "mrl_dims": "post_processing",  # the declared set: bounds which k a run may select
    "mrl_range": "post_processing",  # the declared range: bounds which k a run may select
    "mrl_projection": "post_processing",  # the learned matrices: applied client-side to the reply
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
client -- they never move the replay fingerprint (the engine's output is unchanged), but they do change
what stage 2 compares, so they are CONTENT in the endpoint's identity (the step and stored-reference key):
their home is the comparison identity, not the replay key. ``naming`` fields are keyed elsewhere
(``model``, ``revision``, ``tokenizer_sha256``) and ``transport`` fields decide where and how often a
request goes. A field missing here is refused by :func:`fingerprint_inputs`, so a new product field forces
a decision instead of falling silently in or out."""

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
    return sha256_digest(data)


def sha256_digest(data: bytes) -> str:
    """``sha256:<hex>`` of one byte string: the digest format every fingerprint input uses.

    One home for the canonical form: :func:`plugin_module_hashes` hashes the resolved plugin source with
    it, and the wave runner hashes the staged wheel's members with it, so the two sides can never drift
    apart on the prefix or the encoding.
    """
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _module_sha256(module: str) -> str:
    """``sha256:<hex>`` of one plugin module's source bytes, resolved without importing it.

    The model modules import vLLM/torch by design, so the leaf is never imported here:
    ``importlib.util.find_spec`` only imports the parent packages (which import clean), and the source file
    is read from the spec's origin.  The hash covers the code the engine runs, so a plugin fix that changes
    no recipe field still moves the fingerprint (``rcp-fp/4``).

    Raises:
        HarnessError: the module cannot be resolved to a readable source file (the message names it).
    """
    spec = importlib.util.find_spec(module)
    origin = None if spec is None else spec.origin
    if origin is None or not Path(origin).is_file():
        raise HarnessError(
            f"the plugin module {module!r} cannot be resolved to a source file; a recipe's "
            "serve.plugin_architectures and serve.patches must name the modules the engine runs"
        )
    return sha256_digest(Path(origin).read_bytes())


def plugin_module_hashes(recipe: Recipe) -> dict[str, str]:
    """The plugin-code inputs of ``recipe``: ``{module: sha256:<hex>}`` for exactly the modules it runs.

    The shipped plugin wheel's declaration is the truth: the shared engine modules every plugin recipe runs
    (:data:`~rcp_ndcg_vllm.models.PLUGIN_ENGINE_MODULES`), the modules of each declared architecture
    (:data:`~rcp_ndcg_vllm.models.ARCHITECTURE_MODULES`) and the module of every opted-in patch
    (:data:`~rcp_ndcg_vllm.patches.PATCH_MODULES`), deduplicated in declaration order.  A recipe without a
    plugin has none.  A foreign plugin spec is refused by name: its modules cannot be resolved here, and a
    name-only key is exactly the hole ``rcp-fp/4`` closes.  The wave runner hashes the same modules inside
    the staged wheel it serves (item 9) and refuses to record when the two disagree, so this function is
    the one home of both the fingerprint inputs and the wheel cross-check's module set.

    Raises:
        HarnessError: the recipe names a foreign plugin spec, or a declared module cannot be read.
    """
    if recipe.serve.plugin is None:
        return {}
    if plugin_distribution_name(recipe.serve.plugin) != "rcp-ndcg-vllm":
        raise HarnessError(
            f"recipe {recipe.id}: serve.plugin {recipe.serve.plugin!r} is not the shipped plugin, so the "
            "behaviour fingerprint cannot resolve the modules its engine runs; a foreign plugin needs its "
            "own module declaration and hashes (the shipped wheel's is rcp_ndcg_vllm.models)"
        )
    modules = [*PLUGIN_ENGINE_MODULES]
    for architecture in recipe.serve.plugin_architectures:
        modules.extend(ARCHITECTURE_MODULES[architecture])
    for patch in recipe.serve.patches:
        modules.append(PATCH_MODULES[patch])
    return {module: _module_sha256(module) for module in dict.fromkeys(modules)}


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


def fingerprint_inputs(recipe: Recipe, *, tokenizer_sha256_value: str | None = None) -> dict[str, str]:
    """Every fingerprint input of ``recipe``, **named**, with its canonical value.

    The inputs (GPU-VALIDATION.md item 8, extended by ``rcp-fp/4``): ``model`` and ``revision`` (the
    checkpoint); the engine image and its version floor (``engine.image``, ``engine.min_version`` -- a
    vLLM/transformers change can move the engine's processing without moving any recipe field); one
    ``serve.<field>`` per field of the ``serve`` block (``model_dump`` on the frozen schema -- overrides,
    pooler config, dtype, plugin, ``plugin_architectures``, ``patches``, ``mm_processor_kwargs``,
    ``max_model_len``, ``limit_mm_per_prompt``, ``extra_args``); ``plugin_sha256.<module>`` for every
    engine-side module the declared plugin architectures and patches run; ``template_file`` (the template
    file's bytes) and ``tokenizer_sha256``; and ``client.<field>`` for every client field
    :data:`CLIENT_FIELDS` classifies as ``request`` (it changes the request bytes).
    ``fingerprint_schema`` records which rule hashed them.

    Args:
        recipe: The loaded recipe.
        tokenizer_sha256_value: A tokenizer SHA-256 to use instead of resolving the tokenizer's bytes --
            for an offline comparison whose caller already pins the tokenizer spec (the spec is a
            ``model_dump`` input of its own, so the hash is a pure function of a pinned string).

    Returns:
        The flat map, dotted names to canonical strings.

    Raises:
        HarnessError: the tokenizer or the template file cannot be resolved.
    """
    inputs: dict[str, str] = {"fingerprint_schema": FINGERPRINT_SCHEMA}
    inputs["model"] = recipe.model
    inputs["revision"] = recipe.revision
    inputs["engine.image"] = recipe.engine.image
    inputs["engine.min_version"] = recipe.engine.min_version
    for field, value in sorted(recipe.serve.model_dump(mode="json").items()):
        inputs[f"serve.{field}"] = _canonical(value)
    for module, digest in plugin_module_hashes(recipe).items():
        inputs[f"plugin_sha256.{module}"] = digest
    inputs["template_file"] = _template_file_sha(recipe)
    inputs["tokenizer_sha256"] = tokenizer_sha256_value or tokenizer_sha256(recipe)
    client = recipe.client
    unclassified = sorted(set(client) - set(CLIENT_FIELDS))
    if unclassified:
        raise HarnessError(
            f"recipe {recipe.id}: the client fields {unclassified} are not classified in "
            "rcp_ndcg_test.fingerprint.CLIENT_FIELDS; decide whether each changes the request bytes "
            "('request', a fingerprint input) or not ('post_processing', 'naming', 'transport')"
        )
    # The product's canonical form of the block: the endpoint model validates the plain data (the schema
    # defaults fill in exactly what the recorded corpora's fingerprints read) and identity_payload reads
    # its CONTENT role declarations (R30: never a local re-derivation).
    endpoint = _CLIENT_MODELS[recipe.role].model_validate(client)
    dumped = endpoint.model_dump(mode="json")
    content = identity_payload(endpoint)
    for field in sorted(dumped):
        if CLIENT_FIELDS[field] != "request":
            continue
        value = content[field] if field in content else dumped[field]  # RUNTIME request fields: as declared
        if value is not None:
            inputs[f"client.{field}"] = _canonical(value)
    return inputs


def behaviour_fingerprint(recipe: Recipe, *, tokenizer_sha256_value: str | None = None) -> str:
    """The recipe's **behaviour fingerprint**: the SHA-256 of every input that can change what the model
    returns (see :func:`fingerprint_inputs`), as lowercase hex.

    The key the observation corpora and the verified fake engines are stored under
    (``<engine>-<version>/<recipe>/<fingerprint>/``): two recipes with the same fingerprint serve the same
    model behaviour on the wire, so one corpus and one emulator verify both.

    Args:
        recipe: The loaded recipe.
        tokenizer_sha256_value: See :func:`fingerprint_inputs`.

    Returns:
        The 64-character hex digest of ``FINGERPRINT_SCHEMA`` and the canonical inputs.

    Raises:
        HarnessError: an input cannot be resolved (the message names recipe and input).
    """
    inputs = fingerprint_inputs(recipe, tokenizer_sha256_value=tokenizer_sha256_value)
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
