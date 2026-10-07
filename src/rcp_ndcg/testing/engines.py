"""Verified fake engines built from observation corpora (GPU-VALIDATION.md, "the GPU run is also the test
suite's audit", items 2-4, 7 and 8).

One emulator per (engine, version, recipe, behaviour fingerprint), selected as
``fake://<engine>-<version>/<recipe>`` (e.g. ``fake://vllm-0.31.0/qwen3-embedding-0.6b``):

* the **protocol** is emulated, not replayed: routes, request validation, error statuses and bodies, the
  result ordering and framing, usage counts and token counting with the recipe's real tokenizer files --
  so an over-length request fails exactly where the engine fails (vLLM refuses at
  ``count(prompt tokens) > serve.max_model_len`` with its 400 body), and the template the engine renders
  is rendered the same way (the recipe's template, through the product's own
  :class:`~rcp_ndcg.data.templates.TemplateSpec`);
* the **model outputs** are replayed for observed inputs and come from a declared deterministic
  surrogate for unseen ones, clearly marked in the reply (``x-rcp-ndcg-emulator-source:`` ``replayed``,
  ``surrogate`` or ``mixed``) -- so a test that asserts numbers can only use observed inputs.

Replay is keyed on the **behaviour-shaping context**: the engine prompt -- the exact string(s) the
engine renders and tokenizes per request (the wire's input strings for ``/v1/embeddings`` and
``/pooling``, the pair prompts for a pointwise ``/rerank``, the N-passage prompt for a listwise one) --
plus every request field that changes what the model returns (:data:`FIELD_CLASSES`: ``use_activation``,
``dimensions``, ``add_special_tokens``, ``task``). An unobserved context answers the surrogate; a field
the emulator does not model (an ``instruction``, a ``truncate_prompt_tokens``) is refused with a marked
400; a field the engine's request model does not declare is ignored, as the engine ignores it. A corpus
whose one key holds different outputs is refused. Batch composition is not part of the key: it can move
numbers only within the engine's numeric noise, which the batching strata of a release corpus
(OBSERVATIONS-SPEC section 1) are recorded to measure; a corpus without them declares it unmeasured. The key
is recomputed from the raw records by versioned code (:data:`NORMALISATION_VERSION` strips the volatile
fields a comparison ignores -- request ids, ``created`` timestamps), never stored in the corpus.

The corpus seam: :func:`load_corpus` dispatches on the manifest's ``schema`` over a registry of loaders
(:func:`register_corpus_format`), with per-line migrations for older schemas
(:func:`register_line_migration`), so ``rcp_ndcg_vllm.observe``'s writer plugs in when it lands.

The registry resolves by (engine, version, fingerprint) and loads out-of-tree emulators through the
``rcp_ndcg.emulators`` entry-point group (entry points value: a callable returning
:class:`Emulator <VllmEmulator>` instances). Verification is recorded append-only in the corpus
directory (``verification.jsonl``) and an emulator refuses a corpus or recipe revision it was not
verified against.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import parse_qsl, urlsplit

import httpx

from rcp_ndcg.errors import ConfigError, DataError

__all__ = [
    "CORPUS_INDEX_SCHEMA",
    "LINE_SCHEMA",
    "MANIFEST_SCHEMA",
    "NORMALISATION_VERSION",
    "VERIFICATION_SCHEMA",
    "BehaviourDiff",
    "Corpus",
    "FIELD_CLASSES",
    "ROUTE_FIELDS",
    "EngineFacts",
    "EnginePrompts",
    "EmulatorRegistry",
    "Exchange",
    "PairPrompts",
    "PromptSet",
    "PromptStrategy",
    "StringsPrompts",
    "Verified",
    "append_verification",
    "behaviour_diff",
    "compare_exchange",
    "credential_findings",
    "find_corpora",
    "find_credential_patterns",
    "load_corpus",
    "NON_DETERMINISM_RULE",
    "measure_non_determinism",
    "normalise_body",
    "register_corpus_format",
    "register_line_migration",
    "route_of",
    "registry",
    "request_context",
    "request_digest",
    "transport_for",
    "verification_record",
    "verification_records",
    "verify_corpus_hashes",
    "VllmEmulator",
]

MANIFEST_SCHEMA = "rcp-ndcg.observation/1"
"""The corpus manifest's schema (OBSERVATIONS-SPEC section 3-4, as the compact corpus writes it)."""

LINE_SCHEMA = 1
"""The corpus document schema version; older documents are migrated at load (see
:func:`register_line_migration`)."""

CORPUS_INDEX_SCHEMA = "rcp-ndcg.corpus-index/1"
"""The repository corpus index's schema: the manifest hash of every committed corpus."""

VERIFICATION_SCHEMA = "rcp-ndcg.verification/1"
"""The append-only verification record's schema (written by the conformance verifier)."""

NORMALISATION_VERSION = 1
"""The derived-views version: which volatile fields :func:`normalise_body` strips before comparing two
bodies. A change here is a new version (2), never a silent edit of 1's rule."""

#: ``NORMALISATION_VERSION == 1`` strips exactly these volatile leaves (``$`` is the body root, ``*`` any
#: list element): the reply's request ids and creation timestamps. Everything else is compared.
VOLATILE_FIELDS_1: tuple[str, ...] = (
    "$.created",
    "$.id(embeddings|rerank)",
    "$.data[*].created",
    "$.data[*].permission[*].created",
    "$.data[*].permission[*].id",
)


# ---------------------------------------------------------------------------
# credential scan (OBSERVATIONS-SPEC section 6: no credential-shaped string anywhere)
# ---------------------------------------------------------------------------

_CREDENTIALS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("authorization header", re.compile(r"(?i)\bauthorization\b\s*[:=]")),
    ("bearer token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}")),
    ("basic auth", re.compile(r"(?i)\bbasic\s+[A-Za-z0-9+/=]{16,}")),
    ("api key header", re.compile(r"(?i)\bx-api-key\s*[:=]")),
    ("hugging face token", re.compile(r"\bhf_[A-Za-z0-9]{20,}")),
    ("google api key", re.compile(r"\bAIza[A-Za-z0-9_\-]{20,}")),
    ("private key block", re.compile(r"BEGIN [A-Z ]*PRIVATE KEY")),
    ("cookie header", re.compile(r"(?i)\b(?:set-cookie|cookie)\s*[:=]\s*\S+=")),
    ("secret field", re.compile(r'(?i)"(?:api[_-]?key|secret|password|token)"\s*:\s*"[^"]{8,}"')),
    ("secret field (single-quoted)", re.compile(r"(?i)'(?:api[_-]?key|secret|password|token)'\s*:\s*'[^']{8,}'")),
    (
        "secret assignment",
        re.compile(r'(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token)\s*=\s*["\'][^"\']{8,}'),
    ),
)


def credential_findings(text: str) -> list[str]:
    """The credential **patterns** found in ``text`` (never the matched values): one name per hit kind.

    Args:
        text: Any corpus content (a whole file, a JSON line).

    Returns:
        The sorted pattern names that matched (``[]`` is clean).
    """
    return sorted({name for name, pattern in _CREDENTIALS if pattern.search(text)})


def find_credential_patterns(path: str | Path) -> list[str]:
    """Every credential pattern found in a corpus file (the file is read and decompressed if ``.gz``)."""
    raw = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        raw = gzip.decompress(raw)
    return credential_findings(raw.decode("utf-8", errors="replace"))


# ---------------------------------------------------------------------------
# the corpus: raw records, derived views, hashes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Exchange:
    """One recorded request/response pair, raw first.

    Attributes:
        sequence: The order the recorder sent it.
        method: The HTTP method (``GET``, ``POST``).
        path: The route (``/v1/embeddings``).
        request_body: The request body as sent (parsed JSON, else the ``{"base64": ....}`` envelope).
        status: The response status.
        response_headers: The headers that matter (content type, server, framing).
        response: The response as received: parsed JSON, or ``{"base64", "framing_headers"}`` for a
            binary body (the exact bytes).
        repetition: ``same_process`` or ``after_restart`` (OBSERVATIONS-SPEC section 2); the shakedown
            recorder sent one repetition.
        source: The record's provenance (its file name in the source corpus).
    """

    sequence: int
    method: str
    path: str
    request_body: Any
    status: int
    response_headers: Mapping[str, str]
    response: Any
    repetition: str = "same_process"
    source: str = ""

    @property
    def response_json(self) -> Any:
        """The parsed response body, or ``None`` for a binary one."""
        return self.response if not _is_bytes_envelope(self.response) else None

    @property
    def response_bytes(self) -> bytes:
        """The raw response bytes (a JSON body re-encoded canonically, a binary body kept exact)."""
        if _is_bytes_envelope(self.response):
            import base64

            return base64.b64decode(self.response["base64"])
        return json.dumps(self.response, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _is_bytes_envelope(body: Any) -> bool:
    return isinstance(body, dict) and set(body) == {"base64", "framing_headers"}


@dataclass(frozen=True)
class Corpus:
    """An observation corpus: its manifest and its raw exchanges.

    Attributes:
        root: The corpus directory (``<engine>-<version>/<recipe>/<fingerprint>/``).
        manifest: The manifest document (a mapping), hash-checked against ``files``.
        exchanges: The raw records, in recording order.
    """

    root: Path
    manifest: Mapping[str, Any]
    exchanges: tuple[Exchange, ...]

    @property
    def engine(self) -> Mapping[str, Any]:
        """``{"name", "version", "image"}`` of the recorded engine."""
        return self.manifest["engine"]

    @property
    def behaviour_fingerprint(self) -> str:
        """The model layer's key (the fingerprint recorded at corpus-write time)."""
        return str(self.manifest["recipe"]["behaviour_fingerprint"])

    @property
    def tolerance(self) -> tuple[float, float] | None:
        """``(abs, rel)`` the measured non-determinism derives (OBSERVATIONS-SPEC section 2: measured,
        never chosen by hand), or ``None`` when the corpus holds no same-request repetition (unmeasured:
        a replay is compared exactly)."""
        return _tolerance_of(self.manifest)


def _tolerance_of(manifest: Mapping[str, Any]) -> tuple[float, float] | None:
    nondet = manifest.get("non_determinism") or {}
    if nondet.get("status") != "measured":
        return None
    return (float(nondet["tolerance_abs"]), float(nondet["tolerance_rel"]))


#: One corpus-document loader: the file lines -> :class:`Exchange` tuples.
CorpusLoader = Callable[[Path, Mapping[str, Any]], tuple[Exchange, ...]]

_CORPUS_FORMATS: dict[str, CorpusLoader] = {}
_LINE_MIGRATIONS: dict[int, Callable[[Mapping[str, Any]], Mapping[str, Any]]] = {}
_LOCK = threading.Lock()


def register_corpus_format(schema: str, loader: CorpusLoader) -> None:
    """Register a corpus **format** reader under its manifest ``schema`` (the seam where
    ``rcp_ndcg_vllm.observe``'s writer plugs in: its loader registers as its schema and
    :func:`load_corpus` dispatches to it).

    Args:
        schema: The manifest's ``schema`` value (e.g. :data:`MANIFEST_SCHEMA`).
        loader: ``(root, manifest) -> exchanges``.
    """
    with _LOCK:
        if schema in _CORPUS_FORMATS and _CORPUS_FORMATS[schema] is not loader:
            raise ConfigError(f"a corpus loader is already registered for schema {schema!r}")
        _CORPUS_FORMATS[schema] = loader


def register_line_migration(from_version: int, migrate: Callable[[Mapping[str, Any]], Mapping[str, Any]]) -> None:
    """Register a **line migration** from an older document schema to the next version.

    Readers support every document schema ever written: a line whose ``line_schema`` predates
    :data:`LINE_SCHEMA` walks the chain of registered migrations at load, so a schema bump never
    invalidates an old corpus (OBSERVATIONS-SPEC section 7). One step at a time: ``from_version + 1`` is
    the version the callable produces.

    Args:
        from_version: The schema version the callable reads.
        migrate: The transform (a raw corpus document to its successor).
    """
    with _LOCK:
        if from_version in _LINE_MIGRATIONS and _LINE_MIGRATIONS[from_version] is not migrate:
            raise ConfigError(f"a line migration is already registered for schema {from_version}")
        _LINE_MIGRATIONS[from_version] = migrate


def _migrate(document: Mapping[str, Any], source: str) -> dict[str, Any]:
    """One raw corpus document, walked through the registered migrations up to :data:`LINE_SCHEMA`."""
    version = int(document.get("line_schema", 0))
    doc = dict(document)
    while version < LINE_SCHEMA:
        step = _LINE_MIGRATIONS.get(version)
        if step is None:
            raise DataError(
                f"{source}: corpus line_schema {version} cannot reach {LINE_SCHEMA}: no migration "
                f"registered for {version} (register_line_migration)"
            )
        doc = dict(step(doc))
        version += 1
    return doc


def _read_exchanges(root: Path, manifest: Mapping[str, Any]) -> tuple[Exchange, ...]:
    """The document loader of :data:`MANIFEST_SCHEMA`: compressed JSON Lines, one exchange per line."""
    name = manifest["documents"]
    raw = (root / name).read_bytes()
    if name.endswith(".gz"):
        raw = gzip.decompress(raw)
    exchanges = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        doc = _migrate(json.loads(line), f"{root / name}:{number}")
        exchanges.append(
            Exchange(
                sequence=int(doc["sequence"]),
                method=str(doc["method"]),
                path=str(doc["path"]),
                request_body=doc["request_body"],
                status=int(doc["status"]),
                response_headers=dict(doc.get("response_headers") or {}),
                response=doc["response"],
                repetition=str(doc.get("repetition", "same_process")),
                source=str(doc.get("source", "")),
            )
        )
    return tuple(exchanges)


register_corpus_format(MANIFEST_SCHEMA, _read_exchanges)


def load_corpus(path: str | Path) -> Corpus:
    """Load an observation corpus from its directory (``manifest.json`` + the documents it names).

    The format seam: the manifest's ``schema`` selects the registered loader, and the documents walk
    their :func:`register_line_migration` chain to :data:`LINE_SCHEMA`. Hashes are **not** checked here
    (a reader need not be a verifier); call :func:`verify_corpus_hashes` for that.

    Args:
        path: The corpus directory.

    Returns:
        The :class:`Corpus` (raw records; every derived view is recomputed from them).

    Raises:
        DataError: no manifest, an unknown schema, or no loader registered for the schema.
    """
    root = Path(path)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise DataError(f"{root}: no manifest.json (a corpus directory names its manifest)")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    loader = _CORPUS_FORMATS.get(str(schema))
    if loader is None:
        raise DataError(f"{manifest_path}: schema {schema!r} has no registered corpus loader (register_corpus_format)")
    return Corpus(root=root, manifest=manifest, exchanges=loader(root, manifest))


def find_corpora(
    root: str | Path,
    *,
    recipe_id: str | None = None,
    engine_name: str | None = None,
    engine_version: str | None = None,
) -> list[Path]:
    """Every corpus directory under ``root``, resolved by **scanning manifests** -- never by deriving a
    directory name from a recomputed fingerprint, so a moved fingerprint finds the corpus it moved away
    from (and the staleness check can name what moved).

    Args:
        root: Where to scan (``tests/contract/engines``, one engine-version directory, or a GCS mirror).
        recipe_id: Keep only the corpora whose manifest names this recipe.
        engine_name: Keep only this engine (``vllm``).
        engine_version: Keep only this engine version (``0.31.0``).

    Returns:
        The corpus directories (each holds a ``manifest.json``), sorted by path.
    """
    found = []
    for manifest_path in sorted(Path(root).rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        engine = manifest.get("engine") or {}
        recipe = manifest.get("recipe") or {}
        if recipe_id is not None and recipe.get("id") != recipe_id:
            continue
        if engine_name is not None and engine.get("name") != engine_name:
            continue
        if engine_version is not None and engine.get("version") != engine_version:
            continue
        found.append(manifest_path.parent)
    return found


def verify_corpus_hashes(corpus: Corpus) -> list[str]:
    """The corpus's integrity check (OBSERVATIONS-SPEC section 4): every file hashes to the manifest.

    Args:
        corpus: The loaded corpus.

    Returns:
        A list of problems (``[]`` is intact): each names the file and what moved.
    """
    problems = []
    for name, expected in dict(corpus.manifest["files"]).items():
        path = corpus.root / name
        if not path.is_file():
            problems.append(f"{name}: named in the manifest, missing on disk")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            problems.append(f"{name}: hashes to {actual}, the manifest declares {expected}")
    return problems


# ---------------------------------------------------------------------------
# derived views: the volatile fields a comparison strips (versioned)
# ---------------------------------------------------------------------------


def normalise_body(method: str, path: str, body: Any) -> Any:
    """The body with :data:`VOLATILE_FIELDS_1` stripped, so two replies compare equal when they differ
    only in request ids and creation timestamps.

    Args:
        method: The request method (rules are route-aware).
        path: The request path (``/v1/embeddings``, ``/rerank``, ``/v1/rerank``).
        body: The parsed response body.

    Returns:
        The normalised body (a copy; the input is untouched).
    """
    if not isinstance(body, dict):
        return body
    out = dict(body)
    if "created" in out:
        out.pop("created")
    route = path.rstrip("/")
    if "id" in out and method.upper() == "POST" and route.endswith(("embeddings", "rerank", "v1/rerank")):
        out.pop("id")
    data = out.get("data")
    if isinstance(data, list) and route.endswith(("models", "embeddings")):
        out["data"] = [_normalise_item(item) for item in data]
    return out


def _normalise_item(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    fixed = {key: value for key, value in item.items() if key != "created"}
    if isinstance(fixed.get("permission"), list):
        fixed["permission"] = [
            {key: value for key, value in perm.items() if key not in ("created", "id")}
            for perm in fixed["permission"]
            if isinstance(perm, dict)
        ]
    return fixed


def _float_leaves(value: Any, prefix: str = "") -> dict[str, float]:
    leaves: dict[str, float] = {}
    if isinstance(value, bool):
        return leaves
    if isinstance(value, (int, float)):
        leaves[prefix] = float(value)
        return leaves
    if isinstance(value, dict):
        for key, item in value.items():
            leaves.update(_float_leaves(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            leaves.update(_float_leaves(item, f"{prefix}[{index}]"))
    return leaves


def compare_exchange(
    recorded: Exchange,
    method: str,
    path: str,
    status: int,
    body: Any,
    tolerance: tuple[float, float] | None,
    response_headers: Mapping[str, str] | None = None,
) -> list[str]:
    """The conformance check of one replayed exchange: identical status and body within the recorded
    non-determinism (volatile fields stripped by :data:`NORMALISATION_VERSION`), plus the headers that
    matter (content type, server) when the caller passes the replayed ones -- framing is protocol
    behaviour (GPU-VALIDATION item 1) and a wrong ``content-type`` must not pass.

    Args:
        recorded: The recorded exchange.
        method: The replayed request's method.
        path: The replayed request's path.
        status: The replayed response's status.
        body: The replayed response body (as the transport reads it).
        tolerance: ``(abs, rel)`` from the corpus's measured non-determinism, each bound applying
            jointly; ``None`` (unmeasured) compares exactly.
        response_headers: The replayed response's headers (checked against the recorded ones that
            matter; ``None`` skips the header check).

    Returns:
        A list of differences (``[]`` is conformant); each names the field that moved.
    """
    problems = []
    if status != recorded.status:
        problems.append(f"status {status} != recorded {recorded.status}")
        return problems
    if response_headers is not None:
        for name, expected in dict(recorded.response_headers).items():
            actual = response_headers.get(name)
            if actual != expected:
                problems.append(f"header {name}: {actual!r} != recorded {expected!r}")
    expected_body = normalise_body(recorded.method, recorded.path, recorded.response_json)
    actual = normalise_body(method, path, body)
    problems.extend(_diff_bodies(expected_body, actual, "", tolerance or (0.0, 0.0)))
    return problems


def _diff_bodies(expected: Any, actual: Any, path: str, tolerance: tuple[float, float]) -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        problems = []
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                problems.append(f"{path}.{key}: present in only one body")
                continue
            problems.extend(_diff_bodies(expected[key], actual[key], f"{path}.{key}", tolerance))
        return problems
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: {len(actual)} items != recorded {len(expected)}"]
        problems = []
        for index, (item, other) in enumerate(zip(expected, actual, strict=False)):
            problems.extend(_diff_bodies(item, other, f"{path}[{index}]", tolerance))
        return problems
    if isinstance(expected, bool) or isinstance(actual, bool):
        return [f"{path}: {actual!r} != recorded {expected!r}"] if expected != actual else []
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return _numeric_diff(float(expected), float(actual), path, tolerance)
    return [] if expected == actual else [f"{path}: {actual!r} != recorded {expected!r}"]


def _numeric_diff(expected: float, actual: float, path: str, tolerance: tuple[float, float]) -> list[str]:
    """One number against its record: within the measured absolute AND relative bound (jointly)."""
    absolute = abs(actual - expected)
    relative = absolute / max(abs(expected), abs(actual), 1e-300) if absolute else 0.0
    if absolute <= tolerance[0] and relative <= tolerance[1]:
        return []
    return [f"{path}: {actual!r} differs from recorded {expected!r} by {absolute:g} (relative {relative:g})"]


# ---------------------------------------------------------------------------
# the request fields per route (vLLM v0.31.0's request models), and the replay key they make
# ---------------------------------------------------------------------------

_COMMON_FIELDS = frozenset(
    {
        "model",
        "user",
        "truncate_prompt_tokens",
        "padding",
        "truncation_side",
        "request_id",
        "priority",
        "mm_processor_kwargs",
        "cache_salt",
    }
)
_ENCODE_FIELDS = frozenset(
    {"input", "messages", "add_special_tokens", "encoding_format", "embed_dtype", "endianness", "dimensions"}
)

ROUTE_FIELDS: Mapping[str, frozenset[str]] = {
    "embeddings": _COMMON_FIELDS | _ENCODE_FIELDS | {"use_activation"},
    "pooling": _COMMON_FIELDS | _ENCODE_FIELDS | {"use_activation", "task"},
    "rerank": _COMMON_FIELDS
    | {
        "query",
        "documents",
        "top_n",
        "use_activation",
        "instruction",
        "chat_template_kwargs",
        "max_tokens_per_query",
        "max_tokens_per_doc",
    },
}
"""The request fields each model route's request model declares in vLLM v0.31.0
(``vllm/entrypoints/pooling/{embed,pooling,scoring}/protocol.py`` and their mixins in
``pooling/base/protocol.py``). A field a route does not declare is **ignored** by the engine (its
``OpenAIBaseModel`` drops extra keys: ``vllm/entrypoints/serve/engine/protocol.py``, "fields were
present in the request but ignored"), so it shapes nothing and is not part of the replay key."""

FIELD_CLASSES: Mapping[str, str] = {
    # prompt: carries the engine prompt(s) -- the key's first part
    "input": "prompt",
    "query": "prompt",
    "documents": "prompt",
    # protocol: emulated, never changes a model output
    "model": "protocol",  # validated against the served name
    "user": "protocol",
    "request_id": "protocol",
    "priority": "protocol",
    "cache_salt": "protocol",  # salts the prefix cache only
    "encoding_format": "protocol",  # the framing of the same output
    "embed_dtype": "protocol",
    "endianness": "protocol",
    "top_n": "protocol",  # truncates the ranked list
    # output: changes what the model returns -- part of the replay key
    "add_special_tokens": "output",  # another tokenization of the prompt (emulated in the counts too)
    "use_activation": "output",  # raw logit or activation
    "dimensions": "output",  # the engine's Matryoshka cut
    "task": "output",  # which pooling task runs
    # unmodelled: changes the prompt or its cut in ways the emulator does not render -> a marked 400
    "messages": "unmodelled",  # the chat-style request: rendered by the engine's chat template
    "instruction": "unmodelled",  # folded into chat_template_kwargs and rendered by the template
    "chat_template_kwargs": "unmodelled",
    "truncate_prompt_tokens": "unmodelled",  # the engine cuts instead of refusing
    "truncation_side": "unmodelled",
    "max_tokens_per_query": "unmodelled",
    "max_tokens_per_doc": "unmodelled",
    "padding": "unmodelled",
    "mm_processor_kwargs": "unmodelled",
}
"""Every declared request field, classified by what it does to a reply (:data:`ROUTE_FIELDS`). Only
``output`` fields enter the replay key beside the prompts: an output is replayed only for the exact
context it was observed under, an unobserved context answers the declared surrogate, and an
``unmodelled`` field is refused with a 400 marked ``refused-unmodelled`` -- never answered from
another request's observation."""

_ROUTE_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "embeddings": {"add_special_tokens": True},
    "pooling": {"add_special_tokens": True},
}
"""The declared defaults of ``output`` fields (``CompletionRequestMixin.add_special_tokens = True``): an
absent field and its default ask the same question. Every other absent field stays distinct from any
value (``use_activation`` absent leaves the model's own default, which no record pins)."""


def route_of(path: str) -> str | None:
    """The model route a request path names (``embeddings``, ``pooling``, ``rerank``), any root prefix."""
    route = path.rstrip("/")
    for name in ("embeddings", "pooling", "rerank"):
        if route.endswith("/" + name):
            return name
    return None


def request_context(route: str, body: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """The behaviour-shaping context of one request: its ``output`` fields (defaults applied), and the
    ``unmodelled`` fields it carries.

    Args:
        route: The model route (:func:`route_of`).
        body: The parsed request body.

    Returns:
        ``(context, unmodelled)``: the ``output`` field values that key the replay (beside the prompts)
        and the sorted names of the declared fields the emulator refuses.
    """
    declared = ROUTE_FIELDS[route]
    context = dict(_ROUTE_DEFAULTS.get(route, {}))
    unmodelled = []
    for name, value in body.items():
        if name not in declared:
            continue  # the engine ignores it
        kind = FIELD_CLASSES[name]
        if kind == "output":
            context[name] = value
        elif kind == "unmodelled":
            unmodelled.append(name)
    return context, sorted(unmodelled)


#: One engine prompt: the text the engine tokenizes, or the token ids a request sent as is.
Prompt = str | tuple[int, ...]


@dataclass(frozen=True)
class PromptSet:
    """One request's engine prompts, what each produces, and the context they were asked under.

    Attributes:
        prompts: The engine prompts, in request order: strings, or token-id tuples a request sent as is.
        add_special: Per prompt: whether the engine tokenizes it with the post-processor's tokens.
        slot: What one prompt's model output is: a ``vector`` (dense embedding), a ``token_vector``
            matrix (late interaction), a ``score`` (pointwise rerank one document) or a ``score_list``
            (a listwise prompt scores the whole candidate set at once).
        positions: For ``score``: the document index each prompt scores (prompt order is document
            order); otherwise identity.
        context: The canonical JSON of the request's ``output`` fields (:func:`request_context`): the
            second part of the replay key.
    """

    prompts: tuple[Prompt, ...]
    add_special: tuple[bool, ...]
    slot: Literal["vector", "token_vector", "score", "score_list"]
    positions: tuple[int, ...] = ()
    context: str = "{}"

    def item_key(self, index: int) -> str:
        """The replay key of one output: its engine prompt and the request's context (a listwise
        prompt's one output is keyed by the whole set: :attr:`set_key`)."""
        if self.slot == "score_list":
            return self.set_key
        return json.dumps([_prompt_key(self.prompts[index]), self.context], ensure_ascii=False)

    @property
    def set_key(self) -> str:
        """The replay key of a set-level output (a listwise prompt scores its whole candidate set)."""
        return json.dumps([[_prompt_key(prompt) for prompt in self.prompts], self.context], ensure_ascii=False)

    def ids(self, index: int, tokenizer: Any) -> list[int]:
        """The token ids the engine sees for one prompt (token-id prompts as sent)."""
        prompt = self.prompts[index]
        if isinstance(prompt, tuple):
            return list(prompt)
        return list(tokenizer.ids(prompt, add_special_tokens=self.add_special[index]))

    def count(self, index: int, tokenizer: Any) -> int:
        """The prompt tokens the engine counts for one prompt."""
        prompt = self.prompts[index]
        if isinstance(prompt, tuple):
            return len(prompt)
        return tokenizer.count(prompt, add_special_tokens=self.add_special[index])

    def counted(self, tokenizer: Any) -> int:
        """The engine's ``usage.prompt_tokens`` over this request's prompts (the recorded rule: the sum
        of each engine prompt's tokens, the post-processor included per flag)."""
        return sum(self.count(index, tokenizer) for index in range(len(self.prompts)))


def _prompt_key(prompt: Prompt) -> Any:
    return {"token_ids": list(prompt)} if isinstance(prompt, tuple) else prompt


class PromptStrategy:
    """How a route turns a request body into the prompts the engine tokenizes (the model layer's key)."""

    def prompts(self, body: Mapping[str, Any]) -> PromptSet:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class StringsPrompts(PromptStrategy):
    """The pooling/embeddings routes: each request input is an engine prompt (a string, or a token-id
    list sent as is); the route adds the tokenizer's post-processor tokens unless the request's
    ``add_special_tokens`` says otherwise (vLLM's default: true)."""

    slot: Literal["vector", "token_vector"] = "vector"
    add_special: bool = True

    def prompts(self, body: Mapping[str, Any]) -> PromptSet:
        items = body.get("input")
        if isinstance(items, str) or _is_token_ids(items):
            items = [items]
        prompts = tuple(_prompt(item) for item in items or [])
        flag = body.get("add_special_tokens", self.add_special)
        flag = flag if isinstance(flag, bool) else self.add_special
        return PromptSet(prompts, (flag,) * len(prompts), self.slot, tuple(range(len(prompts))))


@dataclass(frozen=True)
class PairPrompts(PromptStrategy):
    """A pointwise ``/rerank``: the engine renders **one pair prompt per document** with the recipe's
    template (the same render the client budgets against -- byte-identical to the engine's pair builder
    per the recipes' declarations), and scores each pair independently."""

    template: Any
    tokenizer: Any

    def prompts(self, body: Mapping[str, Any]) -> PromptSet:
        query = _text(body.get("query"))
        documents = [_text(document) for document in body.get("documents") or []]
        prompts = tuple(
            self.template.render("pair", self.tokenizer, query=query, document=document) for document in documents
        )
        flag = self.template.adds_special_tokens("pair")
        return PromptSet(prompts, (flag,) * len(prompts), "score", tuple(range(len(documents))))


@dataclass(frozen=True)
class EnginePrompts(PromptStrategy):
    """A listwise ``/rerank``: the engine renders **one N-passage prompt per request** (the checkpoint's
    own builder -- a recipe reference's verbatim port -- given here as ``builder(query, documents)``)
    and scores the set in one call, so results depend on the set and its order (observed inputs cover
    whole prompts, exactly)."""

    builder: Callable[[str, Sequence[str]], str]
    add_special: bool = True

    def prompts(self, body: Mapping[str, Any]) -> PromptSet:
        query = _text(body.get("query"))
        documents = [_text(document) for document in body.get("documents") or []]
        return PromptSet((self.builder(query, documents),), (self.add_special,), "score_list", (0,))


def _is_token_ids(item: Any) -> bool:
    return isinstance(item, list) and bool(item) and all(isinstance(v, int) and not isinstance(v, bool) for v in item)


def _prompt(item: Any) -> Prompt:
    """One ``input`` item as the engine takes it: token ids as sent, anything else as its text."""
    return tuple(item) if _is_token_ids(item) else _text(item)


def _text(item: Any) -> str:
    """One wire text: a string itself, or a ``{"text": ...}`` / content-parts object by its text; any
    other shape is refused (a stringified object is never a prompt)."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if isinstance(item.get("text"), str):
            return item["text"]
        parts = item.get("content")
        if isinstance(parts, list):
            return " ".join(_text(part) for part in parts)
    raise ValueError(f"an input item of type {type(item).__name__} is not a text the emulator models")


# ---------------------------------------------------------------------------
# the model layer: observations per prompt, and the declared surrogate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelObservation:
    """One observed model output for one engine prompt (replayed verbatim for that prompt)."""

    vector: tuple[float, ...] | None = None
    matrix: tuple[tuple[float, ...], ...] | None = None
    token_ids: tuple[int, ...] | None = None
    score: float | None = None
    scores: tuple[tuple[int, float], ...] | None = None


@dataclass(frozen=True)
class EngineFacts:
    """What a served engine reports of itself (the protocol layer's inputs).

    Attributes:
        engine_name: ``vllm``.
        engine_version: The served engine version (``0.31.0``): the protocol layer's key.
        served_name: The ``--served-model-name`` (the request's ``model`` field, validated on every
            route).
        model_root: The checkpoint id (``GET /v1/models`` reports it as ``root``).
        max_model_len: The engine's cap: an over-length prompt is refused here exactly where the engine
            refuses it (prompt tokens > cap).
    """

    engine_name: str
    engine_version: str
    served_name: str
    model_root: str
    max_model_len: int


@dataclass(frozen=True)
class Verified:
    """What an emulator was verified against; anything else it **refuses**.

    Attributes:
        engine_name: The engine family.
        engine_version: The engine version whose protocol it speaks.
        recipe_id: The recipe (served model) it emulates.
        revision: The checkpoint revision the corpus pinned.
        behaviour_fingerprint: The model layer's key.
    """

    engine_name: str
    engine_version: str
    recipe_id: str
    revision: str
    behaviour_fingerprint: str


def surrogate_scores(seed: int, *parts: object, count: int) -> list[float]:
    """The declared deterministic surrogate: ``count`` scores in [-1, 1], hashed from ``seed`` and
    ``parts`` (the product's one draw, :func:`~rcp_ndcg.inference.fake.fake_uniform`), stable on every
    machine and in every call order. Marked in every reply it answers (``x-rcp-ndcg-emulator-source``)."""
    return [2.0 * _uniform(seed, "surrogate-score", *parts, index) - 1.0 for index in range(count)]


def surrogate_vector(seed: int, *parts: object, dim: int) -> list[float]:
    """The declared deterministic surrogate: one hash-seeded unit vector of width ``dim``."""
    raw = [2.0 * _uniform(seed, "surrogate-vector", *parts, index) - 1.0 for index in range(dim)]
    norm = sum(value * value for value in raw) ** 0.5 or 1.0
    return [value / norm for value in raw]


def _uniform(seed: int, *parts: object) -> float:
    """One deterministic draw in [0, 1) -- the product's single fakes draw (``fake_uniform``)."""
    from rcp_ndcg.inference.fake import fake_uniform

    return fake_uniform(seed, *parts)


def surrogate_matrix(seed: int, *parts: object, tokens: int, dim: int) -> list[list[float]]:
    """One deterministic surrogate token-vector matrix, ``tokens`` rows of :func:`surrogate_vector`."""
    return [surrogate_vector(seed, *parts, token, dim=dim) for token in range(tokens)]


# ---------------------------------------------------------------------------
# the emulator
# ---------------------------------------------------------------------------


@dataclass
class VllmEmulator:
    """The verified fake for one (engine, version, recipe, fingerprint): protocol emulated, model
    outputs replayed.

    Routing and validation follow the recorded engine exactly where the corpus records it (unknown
    request fields are ignored, an over-length prompt is refused with the engine's 400 body, results
    come back ranked) and are declared surrogate behaviour where the corpus has no record (they are
    listed in ``unverified_rules`` and reported in the verification record).

    Attributes:
        facts: The engine facts (the protocol layer's inputs).
        strategy: The request -> engine prompts derivation (the model layer's key).
        tokenizer: The recipe's real tokenizer (usage, over-length and ``/tokenize`` count with it).
        observations: Per prompt key, the replayed outputs (first observation = the replay).
        verified: What it was verified against; a different engine version or recipe revision is
            refused.
        slot: What one prompt outputs (from the strategy).
    """

    facts: EngineFacts
    strategy: PromptStrategy
    tokenizer: Any
    observations: dict[str, tuple[ModelObservation, ...]] = field(default_factory=dict)
    verified: Verified | None = None
    slot: str = "vector"
    dim: int = 64
    unverified_rules: tuple[str, ...] = (
        "malformed JSON -> 400",
        "empty input -> 400",
        "model name mismatch -> 404",
        "too many media items -> 400",
        "top_n > documents -> truncated to the documents",
        "the /pooling framings (float/base64/bytes) are emulated from the product's pooling wire but "
        "unverified: the shakedown corpus holds no /pooling recording (RC0's corpus records them)",
        "encoding_format base64 on /v1/embeddings is emulated unverified (the corpus recorded float)",
    )
    _counter: list[int] = field(default_factory=lambda: [0])
    answer_log: list[str] = field(default_factory=list)
    """The provenance of every composed model-output reply (``/embeddings``, ``/pooling``, ``/rerank``;
    errors and ``/models`` ``/tokenize`` carry no model output and are not logged): what tells a
    numbers-asserting test whether its inputs were observed (GPU-VALIDATION item 2)."""

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_corpus(
        cls,
        corpus: Corpus,
        strategy: PromptStrategy,
        tokenizer: Any,
        facts: EngineFacts,
        *,
        dim: int = 64,
    ) -> VllmEmulator:
        """Build the emulator of ``corpus``: replay table from the recorded 2xx bodies, facts given.

        Args:
            corpus: The loaded observation corpus.
            strategy: The prompts derivation matching the recipe's role and scoring.
            tokenizer: The recipe's real :class:`~rcp_ndcg.data.tokenizer.TextTokenizer`.
            facts: The engine facts (served name, cap).
            dim: The surrogate's vector width (the replayed width comes from the corpus).

        Returns:
            The emulator, keyed to the corpus's behaviour fingerprint (model layer) and the engine
            version (protocol layer).
        """
        observed: dict[str, list[tuple[int, ModelObservation]]] = {}
        for exchange in corpus.exchanges:
            route = route_of(exchange.path)
            if exchange.status != 200 or route is None:
                continue
            body = exchange.request_body if isinstance(exchange.request_body, dict) else {}
            context, unmodelled = request_context(route, body)
            if unmodelled:
                raise DataError(
                    f"{exchange.source or exchange.path} #{exchange.sequence}: the recorded request carries "
                    f"{unmodelled}, which the emulator does not model; model them before replaying this corpus"
                )
            set_ = replace(strategy.prompts(body), context=_canonical_context(context))
            for key, observation in _outputs_from_response(set_, exchange):
                observed.setdefault(key, []).append((exchange.sequence, observation))
        tolerance = _tolerance_of(corpus.manifest)
        for key, history in observed.items():
            first_sequence, first = history[0]
            for sequence, other in history[1:]:
                if _observation_differences(first, other, tolerance):
                    raise DataError(
                        f"recipe {corpus.manifest['recipe']['id']}: exchanges #{first_sequence} and #{sequence} "
                        f"answer one replay key with different outputs ({key[:160]}...): the key misses a "
                        "behaviour-shaping field, or the engine varies beyond the measured tolerance"
                    )
        merged = {key: tuple(observation for _, observation in history) for key, history in observed.items()}
        manifest_recipe = corpus.manifest["recipe"]
        return cls(
            facts=facts,
            strategy=strategy,
            tokenizer=tokenizer,
            observations=merged,
            slot=_slot_of(strategy),
            dim=dim,
            verified=Verified(
                engine_name=str(corpus.engine["name"]),
                engine_version=str(corpus.engine["version"]),
                recipe_id=str(manifest_recipe["id"]),
                revision=str(manifest_recipe["revision"]),
                behaviour_fingerprint=str(manifest_recipe["behaviour_fingerprint"]),
            ),
        )

    # -- verification guards ---------------------------------------------------

    def require_verified_for(self, recipe_id: str, revision: str, fingerprint: str, engine_version: str) -> None:
        """Refuse another engine version or recipe revision (GPU-VALIDATION.md item 3).

        Raises:
            ConfigError: any named target differs from the stored verification; the message names the
                fields that moved.
        """
        if self.verified is None:  # pragma: no cover - from_corpus always records it
            raise ConfigError("the emulator records no verification; build it with from_corpus")
        target = Verified(
            engine_name=self.verified.engine_name,
            engine_version=engine_version,
            recipe_id=recipe_id,
            revision=revision,
            behaviour_fingerprint=fingerprint,
        )
        mismatches = [
            name
            for name in ("engine_version", "recipe_id", "revision", "behaviour_fingerprint")
            if getattr(self.verified, name) != getattr(target, name)
        ]
        if mismatches:
            raise ConfigError(
                f"the emulator verified against {self.verified} refuses the requested {target}: changed {mismatches}"
            )

    # -- the HTTP surface -------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        """One request as httpx reads it: :meth:`answer` composed into the wire's shape."""
        body: Any
        try:
            raw = request.read()
            body = json.loads(raw) if raw else None
        except ValueError:
            body = None
            return _error(400, "invalid JSON body", param="body", kind="BadRequestError")
        return self.answer(str(request.url.path), request.method.upper(), body)

    def answer(self, path: str, method: str, body: Any) -> httpx.Response:
        """One request as parsed JSON: the emulated answer.

        Args:
            path: The route (any root prefix; routes match by suffix).
            method: The HTTP method.
            body: The parsed request body (``None`` for a bodyless GET).

        Returns:
            The response, marked ``x-rcp-ndcg-emulator-source`` per input observed or not.

        Raises:
            nothing: refusals are responses, as on the wire.
        """
        body = body if isinstance(body, dict) else {}
        if method == "GET" and path.rstrip("/").endswith("/models"):
            return self._models()
        if method == "GET" and path.rstrip("/").endswith("/health"):
            return _json(200, {})
        if method != "POST":
            return _error(404, f"unrecognized route {method} {path}", kind="NotFoundError")
        if path.rstrip("/").endswith("/embeddings"):
            return self._embeddings(body)
        if path.rstrip("/").endswith("/pooling"):
            return self._pooling(body)
        if path.rstrip("/").endswith("/rerank"):
            return self._rerank(body)
        if path.rstrip("/").endswith("/tokenize"):
            return self._tokenize(body)
        return _error(404, f"unrecognized route {method} {path}", kind="NotFoundError")

    # -- the routes -------------------------------------------------------------

    def _models(self) -> httpx.Response:
        created = self._volatile_time()
        permission = {
            "id": f"modelperm-{self._volatile_hex(16)}",
            "object": "model_permission",
            "created": created,
            "allow_create_engine": False,
            "allow_fine_tuning": False,
            "allow_logprobs": True,
            "allow_sampling": True,
            "allow_search_indices": False,
            "allow_view": True,
            "group": None,
            "is_blocking": False,
            "organization": "*",
        }
        return _json(
            200,
            {
                "object": "list",
                "data": [
                    {
                        "id": self.facts.served_name,
                        "object": "model",
                        "created": created,
                        "owned_by": "vllm",
                        "root": self.facts.model_root,
                        "parent": None,
                        "max_model_len": self.facts.max_model_len,
                        "permission": [permission],
                    }
                ],
            },
        )

    def _prompts_or_error(self, route: str, body: Mapping[str, Any]) -> tuple[PromptSet | None, httpx.Response | None]:
        """The request's engine prompts keyed by its context, or the refusal the engine (or, for an
        unmodelled field, the emulator -- marked) answers instead."""
        if body.get("model") not in (None, self.facts.served_name):
            return None, _error(
                404,
                f"The model `{body.get('model')}` does not exist.",
                kind="NotFoundError",
            )
        context, unmodelled = request_context(route, body)
        if unmodelled:
            refusal = _error(
                400,
                f"the verified fake engine does not model the request field(s) {unmodelled} (no recording "
                "shows the engine's behaviour for them); it refuses instead of answering from another "
                "request's observation",
                param=unmodelled[0],
                kind="EmulatorUnmodelledError",
            )
            refusal.headers["x-rcp-ndcg-emulator"] = "rcp-ndcg.testing.engines"
            refusal.headers["x-rcp-ndcg-emulator-source"] = "refused-unmodelled"
            self.answer_log.append("refused-unmodelled")
            return None, refusal
        try:
            set_ = replace(self.strategy.prompts(body), context=_canonical_context(context))
        except Exception as error:  # noqa: BLE001 - malformed requests are wire refusals
            return None, _error(400, f"invalid request: {error}", kind="BadRequestError")
        if not set_.prompts:
            return None, _error(400, "invalid request: empty input", param="input", kind="BadRequestError")
        over = [
            index for index in range(len(set_.prompts)) if set_.count(index, self.tokenizer) > self.facts.max_model_len
        ]
        if over:
            cap = self.facts.max_model_len
            return None, _error(
                400,
                f"This model's maximum context length is {cap} tokens. However, you requested 0 output "
                f"tokens and your prompt contains at least {cap + 1} input tokens, for a total of at least "
                f"{cap + 1} tokens. Please reduce the length of the input prompt or the number of requested "
                f"output tokens. (parameter=input_tokens, value={cap + 1})",
                param="input_tokens",
                kind="BadRequestError",
            )
        return set_, None

    def _observation(self, key: str) -> tuple[ModelObservation, str]:
        history = self.observations.get(key)
        if history:
            return history[0], "replayed"
        return ModelObservation(), "surrogate"

    def _embeddings(self, body: Mapping[str, Any]) -> httpx.Response:
        set_, error = self._prompts_or_error("embeddings", body)
        if error is not None:
            return error
        assert set_ is not None
        dimensions = body.get("dimensions") if isinstance(body.get("dimensions"), int) else None
        sources, data = [], []
        for index in range(len(set_.prompts)):
            key = set_.item_key(index)
            observation, source = self._observation(key)
            sources.append(source)
            if observation.vector is not None:
                vector = list(observation.vector)  # observed under this very context (dimensions included)
            else:
                vector = surrogate_vector(0, "embedding", key, dim=dimensions or self.dim)
            data.append({"object": "embedding", "index": index, "embedding": vector})
        usage = self._usage(set_)
        return self._marked(
            _json(
                200,
                {
                    "id": f"embedding-{self._volatile_hex(16)}",
                    "object": "list",
                    "created": self._volatile_time(),
                    "model": self.facts.served_name,
                    "data": data,
                    "usage": _usage_body(usage, wide=True),
                },
            ),
            sources,
        )

    def _marked(self, response: httpx.Response, sources: Sequence[str]) -> httpx.Response:
        """Compose the reply's provenance metadata and record it: the observed-inputs guard that lets a
        numbers-asserting test require every input to be replayed."""
        response = _marked(response, sources)
        self.answer_log.append(response.headers.get("x-rcp-ndcg-emulator-source", "surrogate"))
        return response

    def _pooling(self, body: Mapping[str, Any]) -> httpx.Response:
        set_, error = self._prompts_or_error("pooling", body)
        if error is not None:
            return error
        assert set_ is not None
        encoding = body.get("encoding_format") or "float"
        sources, data = [], []
        raw = bytearray()
        framing: list[dict[str, Any]] = []
        for index in range(len(set_.prompts)):
            key = set_.item_key(index)
            observation, source = self._observation(key)
            sources.append(source)
            if observation.matrix is not None:
                matrix = [list(row) for row in observation.matrix]
                token_ids = list(observation.token_ids or set_.ids(index, self.tokenizer))
            else:
                token_ids = set_.ids(index, self.tokenizer)
                matrix = surrogate_matrix(0, "pooling", key, tokens=len(token_ids), dim=self.dim)
            packed: Any
            if encoding == "base64":
                packed = _encode_matrix(matrix, body.get("embed_dtype") or "float16")
            elif encoding in ("bytes", "bytes_only"):
                packed = None
                dtype = str(body.get("embed_dtype") or "float16")
                frame = _pack_frame(matrix, dtype)
                start = len(raw)
                raw += frame
                framing.append(
                    {
                        "index": index,
                        "embed_dtype": dtype,
                        "endianness": "little",
                        "start": start,
                        "end": len(raw),
                        "shape": [len(matrix), len(matrix[0]) if matrix else 0],
                    }
                )
            else:
                packed = matrix
            if packed is not None:
                data.append({"object": "pooling", "index": index, "data": packed, "prompt_token_ids": token_ids})
        usage = self._usage(set_)
        if encoding in ("bytes", "bytes_only"):
            # the pooling wire's bytes framing (vllm/entrypoints/pooling/utils.py::encode_pooling_bytes):
            # raw frames in one body, split by the ``metadata`` header's start/end/shape; ``bytes_only``
            # sends no metadata and the product's adapter refuses it, naming the lane that will pin it.
            headers = {"content-type": "application/octet-stream"}
            if encoding == "bytes":
                headers["metadata"] = json.dumps({"data": framing, "usage": _usage_body(usage, wide=False)})
            return self._marked(httpx.Response(200, content=bytes(raw), headers=headers), sources)
        return self._marked(
            _json(
                200,
                {
                    "object": "list",
                    "model": self.facts.served_name,
                    "data": data,
                    "usage": _usage_body(usage, wide=False),
                },
            ),
            sources,
        )

    def _rerank(self, body: Mapping[str, Any]) -> httpx.Response:
        set_, error = self._prompts_or_error("rerank", body)
        if error is not None:
            return error
        assert set_ is not None
        documents = body.get("documents") or []
        sources: list[str] = []
        scored: list[dict[str, Any]] = []
        if set_.slot == "score_list":
            observation, source = self._observation(set_.set_key)
            sources.append(source)
            if observation.scores is not None:
                scored = [{"index": position, "relevance_score": score} for position, score in observation.scores]
            else:
                scored = [
                    {
                        "index": i,
                        "relevance_score": surrogate_scores(0, "rerank", set_.set_key, count=len(documents))[i],
                    }
                    for i in range(len(documents))
                ]
        else:
            for position in range(len(set_.prompts)):
                key = set_.item_key(position)
                observation, source = self._observation(key)
                sources.append(source)
                score = (
                    observation.score
                    if observation.score is not None
                    else surrogate_scores(0, "rerank", key, count=1)[0]
                )
                scored.append(
                    {"index": set_.positions[position] if set_.positions else position, "relevance_score": score}
                )
        scored.sort(key=lambda entry: -float(entry["relevance_score"]))
        top_n = body.get("top_n")
        if isinstance(top_n, int) and 0 <= top_n < len(scored):
            scored = scored[:top_n]
        results = [
            {
                "index": entry["index"],
                "relevance_score": entry["relevance_score"],
                "document": {"text": _text(documents[entry["index"]]), "multi_modal": None},
            }
            for entry in scored
        ]
        usage = self._usage(set_)
        return self._marked(
            _json(
                200,
                {
                    "id": f"score-{self._volatile_hex(16)}",
                    "model": self.facts.served_name,
                    "results": results,
                    "usage": _usage_body(usage, wide=False),
                },
            ),
            sources,
        )

    def _tokenize(self, body: Mapping[str, Any]) -> httpx.Response:
        """``POST /tokenize``: the engine's tokenization of the request's prompts (the fake engines'
        token counting IS the recipe's real tokenizer), as token ids and counts."""
        items = body.get("input", body.get("prompt", []))
        strings = [items] if isinstance(items, str) else list(items or [])
        add_special = bool(body.get("add_special_tokens", True))
        rows = [
            {"prompt": _text(item), "token_ids": self.tokenizer.ids(_text(item), add_special_tokens=add_special)}
            for item in strings
        ]
        for row in rows:
            row["count"] = len(row["token_ids"])
        return _json(
            200, {"model": self.facts.served_name, "tokenized": rows, "count": sum(row["count"] for row in rows)}
        )

    def _usage(self, set_: PromptSet) -> int:
        return set_.counted(self.tokenizer)

    def _volatile_hex(self, digits: int) -> str:
        self._counter[0] += 1
        return f"{self._counter[0]:0{digits}x}"

    def _volatile_time(self) -> int:
        return 1_700_000_000 + self._counter[0]

    # -- behaviour diff ---------------------------------------------------------

    def replays(self, key: str) -> bool:
        """Whether the observation ``key`` (a :meth:`PromptSet.item_key`) was recorded -- its outputs
        replay -- or unseen (surrogate)."""
        return key in self.observations

    def clear_answer_log(self) -> None:
        """Forget the recorded reply provenance (golden runs assert over the run's own replies)."""
        self.answer_log.clear()


class BehaviourDiff(dict):
    """The behaviour diff of two corpora of one recipe (OBSERVATIONS-SPEC section 7): a plain mapping
    with ``schema: rcp-ndcg.behaviour-diff/1``, per-input deltas and the summary by route."""


def behaviour_diff(before: Corpus, after: Corpus) -> BehaviourDiff:
    """The **behaviour diff** of two corpora of one recipe: per input, the score or vector deltas,
    changed statuses, changed refusals and changed protocol behaviour, summarised by stratum (route,
    for corpora recorded without the generator's stratum labels).

    Args:
        before: The previous corpus (the one leaving).
        after: The new corpus (the candidate).

    Returns:
        The :class:`BehaviourDiff`, JSON-ready.
    """
    old: dict[str, Exchange] = {_key(exchange): exchange for exchange in before.exchanges}
    new: dict[str, Exchange] = {_key(exchange): exchange for exchange in after.exchanges}
    inputs = []
    by_route: dict[str, dict[str, int]] = {}
    for key in sorted(set(old) | set(new)):
        left, right = old.get(key), new.get(key)
        row: dict[str, Any] = {"input": key}
        row["in"] = {"before": left is not None, "after": right is not None}
        if left is None or right is None:
            row["changed"] = True
        else:
            row["status"] = {"before": left.status, "after": right.status}
            leaves_a = _float_leaves(left.response_json)
            leaves_b = _float_leaves(right.response_json)
            deltas = {
                path: round(leaves_b[path] - leaves_a[path], 12)
                for path in sorted(set(leaves_a) & set(leaves_b))
                if abs(leaves_a[path] - leaves_b[path]) > 0
            }
            row["numeric_deltas"] = deltas
            row["protocol_changed"] = [
                problem
                for problem in _diff_bodies(
                    {k: v for k, v in (left.response_json or {}).items() if k not in ("results", "data", "usage")},
                    {k: v for k, v in (right.response_json or {}).items() if k not in ("results", "data", "usage")},
                    "",
                    (0.0, 0.0),
                )
            ]
            row["changed"] = bool(deltas or row["protocol_changed"] or left.status != right.status)
        inputs.append(row)
        route = key.split(" ", 1)[0]
        summary = by_route.setdefault(route, {"inputs": 0, "changed": 0})
        summary["inputs"] += 1
        summary["changed"] += int(bool(row["changed"]))
    return BehaviourDiff(
        {
            "schema": "rcp-ndcg.behaviour-diff/1",
            "recipe": after.manifest["recipe"]["id"],
            "fingerprints": {
                "before": before.behaviour_fingerprint,
                "after": after.behaviour_fingerprint,
            },
            "inputs": inputs,
            "summary": {
                "inputs": len(inputs),
                "changed": sum(1 for row in inputs if row["changed"]),
                "strata": by_route,
            },
        }
    )


NON_DETERMINISM_RULE = (
    "same-request repetitions only: exchanges whose request digest (method, path and the canonical "
    "request body) is identical are repetitions of one question; the measured value is the largest "
    "absolute and relative difference of any numeric leaf of their normalised response bodies between "
    "any two of them; the verification tolerance is the measured pair itself, each bound applying "
    "jointly; without a repetition the tolerance is unmeasured and a replay is compared exactly"
)
"""How :func:`measure_non_determinism` derives the verification tolerance (stored with the numbers)."""


def measure_non_determinism(corpus: Corpus) -> dict[str, Any]:
    """The measured, never assumed, non-determinism of one corpus (OBSERVATIONS-SPEC section 2).

    Only true repetitions count: exchanges with the same request digest (:data:`NON_DETERMINISM_RULE`).
    Two requests that differ in any byte -- a ``use_activation``, a ``top_n``, an ignored unknown field
    -- are different questions, so their differences measure nothing. With no repetition the block says
    ``status: unmeasured`` and carries no tolerance (``None``): the emulator then replays the one
    observation it has and conformance compares exactly; a number is never invented.

    Args:
        corpus: The raw corpus.

    Returns:
        The ``non_determinism`` manifest block: ``rule``, ``status`` (``measured``/``unmeasured``),
        ``same_request_repetitions`` (the repeated request digests), ``measured_max_abs`` and
        ``measured_max_rel``, and the tolerances they derive (``tolerance_abs``, ``tolerance_rel``).
    """
    groups: dict[str, list[Exchange]] = {}
    for exchange in corpus.exchanges:
        if exchange.status == 200 and exchange.response_json is not None:
            groups.setdefault(request_digest(exchange), []).append(exchange)
    repeated = [group for group in groups.values() if len(group) > 1]
    max_abs, max_rel = 0.0, 0.0
    for group in repeated:
        leaves = [_float_leaves(normalise_body(item.method, item.path, item.response_json)) for item in group]
        for index, first in enumerate(leaves):
            for second in leaves[index + 1 :]:
                for name in set(first) | set(second):
                    a, b = first.get(name), second.get(name)
                    if a is None or b is None:
                        continue
                    difference = abs(a - b)
                    max_abs = max(max_abs, difference)
                    max_rel = max(max_rel, difference / max(abs(a), abs(b), 1e-300))
    measured = bool(repeated)
    return {
        "rule": NON_DETERMINISM_RULE,
        "status": "measured" if measured else "unmeasured",
        "same_request_repetitions": len(repeated),
        "measured_max_abs": max_abs if measured else None,
        "measured_max_rel": max_rel if measured else None,
        "tolerance_abs": max_abs if measured else None,
        "tolerance_rel": max_rel if measured else None,
    }


def request_digest(exchange: Exchange) -> str:
    """The content address of one request: SHA-256 of its method, path and canonical body. Two
    exchanges with one digest asked the same question (a repetition)."""
    canonical = json.dumps(
        [exchange.method.upper(), exchange.path, exchange.request_body],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _key(exchange: Exchange) -> str:
    return f"{exchange.method} {exchange.path} {request_digest(exchange)[:16]}"


# ---------------------------------------------------------------------------
# the registry: (engine, version, fingerprint) + the entry-point seam
# ---------------------------------------------------------------------------


class EmulatorRegistry:
    """Registered emulators by (engine, version, recipe, fingerprint), and the entry-point seam.

    Out-of-tree emulators (a private repository's plugins) register through the
    ``rcp_ndcg.emulators`` entry-point group: each entry point resolves to a callable returning
    emulators (:class:`VllmEmulator` or any of its shape), or call :meth:`register` themselves.
    """

    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str, str, str], VllmEmulator] = {}
        self._entry_points_loaded = False

    def register(self, emulator: VllmEmulator) -> None:
        """Register one emulator (its :attr:`~VllmEmulator.verified` identity must be recorded)."""
        verified = emulator.verified
        if verified is None:  # pragma: no cover - from_corpus always records it
            raise ConfigError("an emulator is registered with the verification it carries (from_corpus)")
        key = (verified.engine_name, verified.engine_version, verified.recipe_id, verified.behaviour_fingerprint)
        with _LOCK:
            if key in self._by_key and self._by_key[key] is not emulator:
                raise ConfigError(f"an emulator is already registered for {key}")
            self._by_key[key] = emulator

    def load_entry_points(self) -> list[str]:
        """Load the ``rcp_ndcg.emulators`` entry-point group once; return the loaded names."""
        if self._entry_points_loaded:
            return []
        loaded = []
        from importlib.metadata import entry_points

        for entry in entry_points(group="rcp_ndcg.emulators"):
            provider = entry.load()
            built = provider() if callable(provider) else provider
            items = built if isinstance(built, Iterable) else (built,)
            for emulator in items:
                self.register(cast(VllmEmulator, emulator))
            loaded.append(entry.name)
        self._entry_points_loaded = True
        return loaded

    def resolve(self, engine_name: str, engine_version: str, fingerprint: str, recipe_id: str) -> VllmEmulator:
        """The emulator keyed (engine, version, fingerprint, recipe).

        Raises:
            ConfigError: nothing registered (the message names what was looked up), or the only
                candidates were verified against another fingerprint (named).
        """
        self.load_entry_points()
        key = (engine_name, engine_version, recipe_id, fingerprint)
        emulator = self._by_key.get(key)
        if emulator is not None:
            return emulator
        neighbours = [k for k in self._by_key if k[:1] == (engine_name,) and k[2] == recipe_id]
        raise ConfigError(
            f"no emulator registered for ({engine_name}, {engine_version}, {recipe_id}, {fingerprint[:12]}...)"
            + (f"; registered for the recipe: {sorted(k[3] for k in neighbours)}" if neighbours else "")
        )

    def clear(self) -> None:
        """Forget every registration (tests start from empty)."""
        with _LOCK:
            self._by_key.clear()
            self._entry_points_loaded = False


registry = EmulatorRegistry()
"""The process-wide emulator registry."""


def transport_for(url: str) -> httpx.MockTransport:
    """The in-process transport of a ``fake://<engine>-<version>/<recipe>`` URL (the routing seam
    ``rcp_ndcg.inference.fake`` hands engine-version URLs to).

    Args:
        url: The fake URL, e.g. ``fake://vllm-0.31.0/qwen3-embedding-0.6b``.

    Returns:
        A mock transport answering with the registered emulator.

    Raises:
        ConfigError: the URL names no registered emulator (the message names the lookup).
    """
    parts = urlsplit(url)
    engine, version = _split_engine_host(parts.hostname or parts.netloc)
    recipe_id = parts.path.strip("/")
    query = dict(parse_qsl(parts.query))
    fingerprint = query.get("fingerprint", "")
    if not fingerprint:
        candidates = [key for key in registry._by_key if key[:2] == (engine, version) and key[2] == recipe_id]
        if len(candidates) == 1:
            fingerprint = candidates[0][3]
    emulator = registry.resolve(engine, version, fingerprint, recipe_id) if fingerprint else None
    if emulator is None:  # pragma: no cover - resolve raises first
        raise ConfigError(f"no emulator for {url}")
    return httpx.MockTransport(emulator.handle)


def _split_engine_host(host: str) -> tuple[str, str]:
    """``vllm-0.31.0`` -> ``("vllm", "0.31.0")`` (the engine name before the version)."""
    match = re.fullmatch(r"([a-z]+)-(\d+(?:\.\d+)*(?:rc\d+)?)", host)
    if not match:
        raise ConfigError(f"{host!r} is not an engine-version host (e.g. 'vllm-0.31.0')")
    return match.group(1), match.group(2)


# ---------------------------------------------------------------------------
# the verification record (OBSERVATIONS-SPEC section 4: appended, never rewritten)
# ---------------------------------------------------------------------------


def append_verification(corpus_dir: str | Path, record: Mapping[str, Any]) -> None:
    """Append one verification record (``verification.jsonl`` beside the manifest), never rewriting.

    Args:
        corpus_dir: The corpus directory (write there; tests use ``tmp_path`` copies).
        record: The record; :data:`VERIFICATION_SCHEMA` and the verification facts are required.
    """
    path = Path(corpus_dir) / "verification.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def verification_records(corpus_dir: str | Path) -> list[dict[str, Any]]:
    """Every appended verification record (the concatenation is the emulator's verification history)."""
    path = Path(corpus_dir) / "verification.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verification_record(corpus: Corpus, problems: Sequence[str], *, verified_at: str) -> dict[str, Any]:
    """The verification record of one conformance run over ``corpus`` (OBSERVATIONS-SPEC section 4):
    which emulator verified it, against which engine version and recipe revision, with which tolerances,
    and the result.

    Args:
        corpus: The verified corpus.
        problems: The conformance differences the run found (``[]`` is a pass).
        verified_at: The run's date (``YYYY-MM-DD``).

    Returns:
        The record, ready for :func:`append_verification`.
    """
    from importlib.metadata import version

    recipe = corpus.manifest["recipe"]
    tolerance = corpus.tolerance
    return {
        "schema": VERIFICATION_SCHEMA,
        "emulator": f"rcp_ndcg.testing.engines (rcp-ndcg {version('rcp-ndcg')})",
        "engine": dict(corpus.engine),
        "recipe": {
            "id": recipe["id"],
            "revision": recipe["revision"],
            "behaviour_fingerprint": recipe["behaviour_fingerprint"],
        },
        "normalisation_version": NORMALISATION_VERSION,
        "tolerances": None if tolerance is None else {"abs": tolerance[0], "rel": tolerance[1]},
        "exchanges": len(corpus.exchanges),
        "problems": list(problems),
        "result": "pass" if not problems else "fail",
        "verified_at": verified_at,
    }


# ---------------------------------------------------------------------------
# module-local helpers
# ---------------------------------------------------------------------------


def _slot_of(strategy: PromptStrategy) -> str:
    if isinstance(strategy, EnginePrompts):
        return "score_list"
    if isinstance(strategy, PairPrompts):
        return "score"
    return getattr(strategy, "slot", "vector")


def _canonical_context(context: Mapping[str, Any]) -> str:
    return json.dumps(dict(context), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _observation_differences(
    first: ModelObservation, other: ModelObservation, tolerance: tuple[float, float] | None
) -> list[str]:
    """Where two observations of one replay key disagree (exact when the tolerance is unmeasured)."""
    plain = json.loads(json.dumps(asdict(first)))
    again = json.loads(json.dumps(asdict(other)))
    return _diff_bodies(plain, again, "", tolerance or (0.0, 0.0))


def _outputs_from_response(set_: PromptSet, exchange: Exchange) -> list[tuple[str, ModelObservation]]:
    """The model outputs one recorded 2xx exchange holds, keyed per replay key (prompt + context).

    A framed body (``bytes``/``bytes_only``) is decoded from its ``metadata`` framing when it carries
    one, base64 item payloads in the request's ``embed_dtype``; anything else is refused loudly --
    a corpus record whose outputs cannot be derived must never read as an empty replay table.
    """
    body = exchange.response_json
    pairs: list[tuple[str, ModelObservation]] = []
    if body is None:
        envelope = exchange.response if isinstance(exchange.response, dict) else {}
        header = (envelope.get("framing_headers") or {}).get("metadata")
        if header:
            import base64

            raw = base64.b64decode(envelope["base64"])
            frames = json.loads(header).get("data") or []
            dtype = str(exchange.request_body.get("embed_dtype") or "float16")
            for index, frame in enumerate(frames[: len(set_.prompts)]):
                matrix = _decode_frame(raw[int(frame["start"]) : int(frame["end"])], dtype, tuple(frame["shape"]))
                pairs.append((set_.item_key(index), ModelObservation(matrix=matrix)))
            return pairs
        raise DataError(
            f"{exchange.source or exchange.path}: a framed response without metadata cannot be decomposed "
            "into per-prompt observations"
        )
    if set_.slot in ("vector", "token_vector"):
        data = body.get("data") or []
        for index, (_prompt, item) in enumerate(zip(set_.prompts, data, strict=False)):
            if set_.slot == "vector":
                pairs.append((set_.item_key(index), ModelObservation(vector=tuple(item.get("embedding") or ()))))
                continue
            payload = item.get("data")
            ids = tuple(item.get("prompt_token_ids") or ())
            if isinstance(payload, str):
                dtype = str(exchange.request_body.get("embed_dtype") or "float16")
                matrix = _decode_base64_matrix(payload, dtype, tokens=len(ids))
            elif isinstance(payload, list):
                matrix = tuple(tuple(float(v) for v in row) for row in payload)
            elif _is_bytes_envelope(payload):
                dtype = str(exchange.request_body.get("embed_dtype") or "float16")
                import base64

                raw = base64.b64decode(payload["base64"])
                matrix = _decode_frame(raw, dtype, (len(ids), _frame_width(raw, dtype, len(ids))))
            else:
                raise DataError(f"{exchange.source or exchange.path}: an unparsable token-vector payload")
            pairs.append((set_.item_key(index), ModelObservation(matrix=matrix, token_ids=ids)))
        return pairs
    results = body.get("results") or []
    scores = {int(entry.get("index", i)): float(entry["relevance_score"]) for i, entry in enumerate(results)}
    if set_.slot == "score":
        for position, _prompt in enumerate(set_.prompts):
            pairs.append((set_.item_key(position), ModelObservation(score=scores.get(position))))
    else:
        pairs.append((set_.set_key, ModelObservation(scores=tuple(sorted(scores.items())))))
    return pairs


def _frame_width(raw: bytes, dtype: str, tokens: int) -> int:
    import numpy as np

    if tokens <= 0:
        return 0
    return len(np.frombuffer(raw, dtype=np.dtype("<f4" if dtype == "float32" else "<f2"))) // tokens


def _decode_frame(raw: bytes, dtype: str, shape: tuple[int, ...]) -> tuple[tuple[float, ...], ...]:
    import numpy as np

    array = np.frombuffer(raw, dtype=np.dtype("<f4" if dtype == "float32" else "<f2"))
    return tuple(tuple(float(v) for v in row) for row in array.reshape(shape))


def _decode_base64_matrix(payload: str, dtype: str, *, tokens: int) -> tuple[tuple[float, ...], ...]:
    import base64

    raw = base64.b64decode(payload)
    return _decode_frame(raw, dtype, (tokens, _frame_width(raw, dtype, tokens)))


def _usage_body(usage: int, *, wide: bool) -> dict[str, Any]:
    """The route's usage shape (protocol behaviour, replayed byte-for-byte): /rerank and /pooling report
    the two-key shape, /v1/embeddings the OpenAI five-key one (measured on the recorded corpus)."""
    if not wide:
        return {"prompt_tokens": usage, "total_tokens": usage}
    return {
        "completion_tokens": 0,
        "completion_tokens_details": None,
        "prompt_tokens": usage,
        "prompt_tokens_details": None,
        "total_tokens": usage,
    }


def _error(status: int, message: str, *, param: str | None = None, kind: str = "BadRequestError") -> httpx.Response:
    return _json(status, {"error": {"code": status, "message": message, "param": param, "type": kind}})


def _json(status: int, body: Any) -> httpx.Response:
    return httpx.Response(status, headers={"content-type": "application/json", "server": "uvicorn"}, json=body)


def _marked(response: httpx.Response, sources: Sequence[str]) -> httpx.Response:
    """The reply's provenance metadata: every input observed (``replayed``), none (``surrogate``) or
    ``mixed`` -- what tells a numbers-asserting test whether its inputs were observed."""
    kinds = set(sources)
    overall = sources[0] if len(kinds) == 1 else ("mixed" if kinds else "surrogate")
    response.headers["x-rcp-ndcg-emulator"] = "rcp-ndcg.testing.engines"
    response.headers["x-rcp-ndcg-emulator-source"] = overall
    return response


def _encode_matrix(matrix: Sequence[Sequence[float]], dtype: str) -> str:
    import base64

    return base64.b64encode(_pack_frame(matrix, dtype)).decode("ascii")


def _pack_frame(matrix: Sequence[Sequence[float]], dtype: str) -> bytes:
    import numpy as np

    return np.asarray(matrix, dtype=np.dtype("<f4" if dtype == "float32" else "<f2")).tobytes()
