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
(OBSERVATIONS-SPEC section 1) are recorded to measure; a corpus without them declares it unmeasured.

The corpora are read through the format's one reader, :mod:`rcp_ndcg_test.corpus` (``load_corpus``,
``integrity_mismatches``, ``normalise_body`` and ``normalise_raw``, ``credential_findings``, the append-only
verification record): this module consumes its records (:func:`exchanges_of`) and the measured
non-determinism the corpus stores (:func:`corpus_tolerance`), and adds what only the emulators need -- the
manifest scan (:func:`find_corpora`) and the verification record's content (:func:`verification_record`).

The registry resolves by (engine, version, fingerprint) and loads out-of-tree emulators through the
``rcp_ndcg.emulators`` entry-point group (entry points value: a callable returning
:class:`Emulator <VllmEmulator>` instances). Verification is recorded append-only beside the corpus
(``verification.jsonl``, :func:`rcp_ndcg_test.corpus.append_verification`; not part of the recorded
corpus, so no integrity hash covers it) and an emulator
refuses an engine version or recipe revision it was not verified against.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import parse_qsl, urlsplit

import httpx

from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg_test.corpus import (
    NORMALISATION_VERSION,
    VERIFICATION_SCHEMA,
    ObservationCorpus,
    normalise_body,
    normalise_raw,
)
from rcp_ndcg_test.errors import EmulatorUnmodelledError

__all__ = [
    "CORPUS_INDEX_SCHEMA",
    "EMULATED_ROUTES",
    "FIELD_CLASSES",
    "ROUTE_FIELDS",
    "BehaviourDiff",
    "ChatPrompts",
    "EmulatorRegistry",
    "EngineFacts",
    "EnginePrompts",
    "Exchange",
    "MediaIdentity",
    "MediaPrompt",
    "PairPrompts",
    "PromptSet",
    "PromptStrategy",
    "RequestPrompts",
    "StringsPrompts",
    "Verified",
    "VllmEmulator",
    "behaviour_diff",
    "compare_exchange",
    "corpus_tolerance",
    "exchanges_of",
    "find_corpora",
    "registry",
    "request_context",
    "route_name",
    "route_of",
    "split_engine_host",
    "surrogate_matrix",
    "surrogate_scores",
    "surrogate_vector",
    "transport_for",
    "verification_record",
]

CORPUS_INDEX_SCHEMA = "rcp-ndcg.corpus-index/1"
"""The repository corpus index's schema: the manifest hash of every committed corpus."""


# ---------------------------------------------------------------------------
# the corpus records, as the emulators read them
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Exchange:
    """One recorded exchange as the emulators read it: a view of one corpus record
    (:data:`rcp_ndcg_test.corpus.RECORD_SCHEMA`), raw first.

    Attributes:
        sequence: The order the collector sent it.
        method: The HTTP method (``GET``, ``POST``).
        path: The route (``/v1/embeddings``).
        request_body: The request body as parsed JSON (``None`` when none was sent or it is not JSON).
        status: The response status.
        response_headers: The headers that matter (content type, server, the bytes framing's metadata).
        response: The response body as parsed JSON, ``None`` when it is not JSON (a binary body).
        repetition: Which sending (``same_process_1``, ``same_process_2``, ``after_restart``).
        source: The record's provenance (its ``exchange_id``, or the source file the collector names).
        request_raw: The request's exact bytes, when the record carries them as sent.
        response_raw: The response's exact bytes, when the record carries them as received (``None`` for a
            record whose ``body_raw`` is declared reconstructed: never a re-encoding passed off as raw).
        exchange_id: The record's content address (the SHA-256 of the canonical request).
    """

    sequence: int
    method: str
    path: str
    request_body: Any
    status: int
    response_headers: Mapping[str, str]
    response: Any
    repetition: str = "same_process_1"
    source: str = ""
    request_raw: bytes | None = None
    response_raw: bytes | None = None
    exchange_id: str = ""

    @property
    def response_json(self) -> Any:
        """The parsed response body, or ``None`` for one that is not JSON."""
        return self.response

    @property
    def raw_body(self) -> bytes | None:
        """The response's exact bytes, or ``None`` when the record kept only its parsed form."""
        return self.response_raw

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> Exchange:
        """The view of one corpus record (``request``/``response`` with ``body_raw`` -- UTF-8 text or
        ``{"base64": ...}`` -- and ``body_parsed``). A record whose ``inputs.body_raw`` says
        ``reconstructed`` carries no raw bytes for the comparison."""
        request, response = record.get("request") or {}, record.get("response") or {}
        inputs = record.get("inputs") or {}
        reconstructed = str(inputs.get("body_raw") or "").startswith("reconstructed")
        source = inputs.get("source") or {}
        return cls(
            sequence=int(record.get("sequence", 0)),
            method=str(request.get("method", "")),
            path=str(request.get("path", "")),
            request_body=request.get("body_parsed"),
            status=int(response.get("status", 0)),
            response_headers=dict(response.get("headers") or {}),
            response=response.get("body_parsed"),
            repetition=str(record.get("repetition", "")),
            source=str(
                source.get("file") if isinstance(source, dict) and source.get("file") else record.get("exchange_id", "")
            ),
            request_raw=None if reconstructed else _raw(request.get("body_raw")),
            response_raw=None if reconstructed else _raw(response.get("body_raw")),
            exchange_id=str(record.get("exchange_id", "")),
        )


def _raw(body_raw: Any) -> bytes | None:
    if isinstance(body_raw, dict) and isinstance(body_raw.get("base64"), str):
        return base64.b64decode(body_raw["base64"])
    if isinstance(body_raw, str):
        return body_raw.encode("utf-8")
    return None


def exchanges_of(corpus: ObservationCorpus) -> tuple[Exchange, ...]:
    """Every record of a corpus as an :class:`Exchange`, in record order."""
    return tuple(Exchange.from_record(record) for record in corpus.records)


def corpus_tolerance(corpus: ObservationCorpus) -> tuple[float, float] | None:
    """``(abs, rel)``: the verification tolerance the corpus's measured non-determinism derived
    (``nondeterminism.json``'s ``derived`` block, with the rule that derived it), or ``None`` when no
    request was sent twice (``measured: false``: a replay is compared exactly, never with an invented
    tolerance).

    Raises:
        DataError: the corpus carries no non-determinism report.
    """
    if corpus.nondeterminism is None:
        raise DataError(f"{corpus.directory}: the corpus carries no non-determinism report (nondeterminism.json)")
    derived = corpus.nondeterminism.get("derived") or {}
    if not derived.get("measured"):
        return None
    return (float(derived["abs_tolerance"]), float(derived["rel_tolerance"]))


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
        root: Where to scan (``tests/contract/engines``, one engine-version directory, or a mirror).
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


# ---------------------------------------------------------------------------
# derived views: the volatile fields a comparison strips (rcp_ndcg_test.corpus.NORMALISATION_VERSION)
# ---------------------------------------------------------------------------


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


def compare_exchange(recorded: Exchange, replayed: httpx.Response, tolerance: tuple[float, float] | None) -> list[str]:
    """The conformance check of one replayed exchange, as the transport reads it: identical status, the
    recorded headers that matter (content type, server, the framing ``metadata`` -- volatile fields
    stripped), and the body within the recorded non-determinism -- its raw bytes too where the corpus
    recorded them (volatile values masked, :data:`NORMALISATION_VERSION`).

    Args:
        recorded: The recorded exchange.
        replayed: The emulator's reply to the recorded request.
        tolerance: ``(abs, rel)`` from the corpus's measured non-determinism, each bound applying
            jointly; ``None`` (unmeasured) compares exactly.

    Returns:
        A list of differences (``[]`` is conformant); each names what moved. An undecodable replayed
        body is a named difference, never an exception or a silent pass.
    """
    problems = []
    if replayed.status_code != recorded.status:
        return [f"status {replayed.status_code} != recorded {recorded.status}"]
    for name, expected in dict(recorded.response_headers).items():
        actual = replayed.headers.get(name)
        if name.lower() == "metadata" and actual is not None:
            if _volatile_free(actual) != _volatile_free(expected):
                problems.append(f"header metadata: {actual!r} != recorded {expected!r}")
        elif actual != expected:
            problems.append(f"header {name.lower()}: {actual!r} != recorded {expected!r}")
    content = replayed.content
    if recorded.response_json is None:  # a binary reply: the exact frames
        expected_raw = recorded.raw_body or b""
        if content != expected_raw:
            problems.append(f"body bytes: {len(content)} bytes differ from the recorded {len(expected_raw)}")
        return problems
    try:
        actual_body = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        problems.append(f"undecodable body ({replayed.headers.get('content-type')!r}): {error}")
        return problems
    expected_body = normalise_body(recorded.response_json)
    actual = normalise_body(actual_body)
    problems.extend(_diff_bodies(expected_body, actual, "", tolerance or (0.0, 0.0)))
    if recorded.response_raw is not None and not problems and tolerance is None:
        if normalise_raw(content) != normalise_raw(recorded.response_raw):
            problems.append("raw bytes differ from the recorded bytes (volatile values masked)")
    return problems


def _volatile_free(header: str) -> Any:
    try:
        value = json.loads(header)
    except ValueError:
        return header
    return (
        {key: item for key, item in value.items() if key not in ("id", "created")} if isinstance(value, dict) else value
    )


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
    "messages": "prompt",  # the chat-style request: its conversations carry the engine prompt(s)
    # unmodelled: changes the prompt or its cut in ways the emulator does not render -> a marked 400
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
another request's observation. ``messages`` is a prompt carrier: a strategy that models chat-shaped
requests (a :class:`ChatPrompts`) derives their prompts from it, and one that does not refuses with a
marked 400 (:class:`~rcp_ndcg_test.errors.EmulatorUnmodelledError`)."""

_ROUTE_DEFAULTS: Mapping[str, Mapping[str, Any]] = {
    "embeddings": {"add_special_tokens": True},
    "pooling": {"add_special_tokens": True},
}
"""The declared defaults of ``output`` fields (``CompletionRequestMixin.add_special_tokens = True``): an
absent field and its default ask the same question. Every other absent field stays distinct from any
value (``use_activation`` absent leaves the model's own default, which no record pins)."""

_CHAT_ROUTE_DEFAULT = {"add_special_tokens": False}
"""The chat-shaped (``messages``) routes' default of ``add_special_tokens``: vLLM v0.31.0's
``ChatRequestOptionsMixin`` declares it ``False`` (``pooling/base/protocol.py:230-237``), unlike the
completion-shaped routes' ``True``. A recorded chat request that omits the field and one that sends
``false`` ask the engine the same question, so they key the same replay entry."""


EMULATED_ROUTES: tuple[str, ...] = (
    "GET /health",
    "GET /v1/models",
    "POST /pooling",
    "POST /rerank",
    "POST /tokenize",
    "POST /v1/embeddings",
)
"""Every route the emulator answers. A route no recording of the corpus covers is **unobserved**: its
replies follow the engine's source and say so (``x-rcp-ndcg-emulator-route: unobserved``), and the
verification record lists it."""


def route_name(method: str, path: str) -> str | None:
    """The :data:`EMULATED_ROUTES` entry a request names (any root prefix, ``/v1`` or not), else ``None``."""
    route = "/" + path.strip("/").removeprefix("v1/").rsplit("/v1/", 1)[-1]
    for name in EMULATED_ROUTES:
        verb, suffix = name.split(" ", 1)
        bare = suffix.removeprefix("/v1")
        if method.upper() == verb and (route == bare or route.endswith(bare) and route[-len(bare) - 1] == "/"):
            return name
    return None


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
    defaults = dict(_ROUTE_DEFAULTS.get(route, {}))
    if "messages" in body:
        defaults.update(_CHAT_ROUTE_DEFAULT)
    context = defaults
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


@dataclass(frozen=True)
class MediaIdentity:
    """One media part as the replay key reads it: its content identity and the processing declared for it.

    The bytes the request carried are hashed (``sha256``) and the kind and the recipe's declared media
    processing (the image/video policy and the processor family, as the strategy's ``processing`` string)
    are recorded beside them: two requests showing the same bytes under the same declared processing ask
    the model the same question, whatever their URI says, and a different image, clip or policy is a
    different key.  The URI itself is never part of the key: it is a location, not content.
    """

    kind: str
    sha256: str
    processing: str = ""

    def key(self) -> dict[str, Any]:
        """The identity as the canonical key fragment (JSON-ready)."""
        return {"kind": self.kind, "sha256": self.sha256, "processing": self.processing}


@dataclass(frozen=True)
class MediaPrompt:
    """One engine prompt whose request also carries media: the engine's render of the text parts, the media
    parts keyed by content identity, and the tokens the engine adds for them.

    The engine renders the conversation or pair from its text parts and expands every media part into its
    vision block, so what the model reads is the render plus the media; :attr:`media_tokens` is the media
    half exactly as the engine counts it (the product's ``content_media_tokens`` under the declared
    policies, the tokenizer passed), which is what ``usage.prompt_tokens`` adds over the text render.
    """

    text: str
    media: tuple[MediaIdentity, ...] = ()
    media_tokens: int = 0
    placement: tuple[str, ...] = ()
    """The parts' kinds in the conversation's or pair's own order (``text``/``image``/``video``): the engine
    places each vision block where its part stands, so ``[text, image]`` and ``[image, text]`` render
    differently and must not share a replay key.  Empty when the strategy did not record one."""

    def key(self) -> dict[str, Any]:
        """The prompt as the canonical key fragment: the render, every media identity in part order, and the
        placement (which part stands where)."""
        return {"text": self.text, "media": [part.key() for part in self.media], "placement": list(self.placement)}


#: One engine prompt: the text the engine tokenizes, the token ids a request sent as is, or a chat/media
#: prompt (its render plus the media parts the engine expands).
Prompt = str | tuple[int, ...] | MediaPrompt


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

    def model_key(self, index: int) -> str:
        """The model-input key of one prompt: its engine prompt and the context WITHOUT the head fields.

        The surrogate's full-width draw must not depend on the requested cut: the engine's head slices the
        model's full-width output, so the ``k`` reply is ``normalize(full[:k])`` of the same draw. Only
        ``dimensions`` is dropped (the one field the head reads); the replay key still carries the whole
        context.
        """
        context = json.loads(self.context)
        context.pop("dimensions", None)
        return json.dumps([_prompt_key(self.prompts[index]), _canonical_context(context)], ensure_ascii=False)

    @property
    def set_key(self) -> str:
        """The replay key of a set-level output (a listwise prompt scores its whole candidate set)."""
        return json.dumps([[_prompt_key(prompt) for prompt in self.prompts], self.context], ensure_ascii=False)

    def ids(self, index: int, tokenizer: Any) -> list[int]:
        """The token ids the engine sees for one prompt (token-id prompts as sent; a media prompt's render)."""
        prompt = self.prompts[index]
        if isinstance(prompt, tuple):
            return list(prompt)
        text = prompt.text if isinstance(prompt, MediaPrompt) else prompt
        return list(tokenizer.ids(text, add_special_tokens=self.add_special[index]))

    def count(self, index: int, tokenizer: Any) -> int:
        """The prompt tokens the engine counts for one prompt (a media prompt's vision blocks included)."""
        prompt = self.prompts[index]
        if isinstance(prompt, tuple):
            return len(prompt)
        if isinstance(prompt, MediaPrompt):
            return tokenizer.count(prompt.text, add_special_tokens=self.add_special[index]) + prompt.media_tokens
        return tokenizer.count(prompt, add_special_tokens=self.add_special[index])

    def counted(self, tokenizer: Any) -> int:
        """The engine's ``usage.prompt_tokens`` over this request's prompts (the recorded rule: the sum
        of each engine prompt's tokens, the post-processor included per flag)."""
        return sum(self.count(index, tokenizer) for index in range(len(self.prompts)))


def _prompt_key(prompt: Prompt) -> Any:
    if isinstance(prompt, tuple):
        return {"token_ids": list(prompt)}
    if isinstance(prompt, MediaPrompt):
        return {"media_prompt": prompt.key()}
    return prompt


class PromptStrategy:
    """How a route turns a request body into the prompts the engine tokenizes (the model layer's key)."""

    def prompts(self, body: Mapping[str, Any]) -> PromptSet:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class StringsPrompts(PromptStrategy):
    """The pooling/embeddings routes: each request input is an engine prompt (a string, or a token-id
    list sent as is); the route adds the tokenizer's post-processor tokens unless the request's
    ``add_special_tokens`` says otherwise (vLLM's default: true).

    A chat-shaped body (``messages``) is refused with :class:`~rcp_ndcg_test.errors.EmulatorUnmodelledError`:
    this strategy has no chat render, and answering the request from a text prompt's observation would be a
    different question.  Wrap it in a :class:`RequestPrompts` with a :class:`ChatPrompts` for the routes
    whose media items ride ``messages``.
    """

    slot: Literal["vector", "token_vector"] = "vector"
    add_special: bool = True

    def prompts(self, body: Mapping[str, Any]) -> PromptSet:
        if "messages" in body:
            raise EmulatorUnmodelledError(
                "this prompt strategy models the completion-shaped (input) routes only; a messages request "
                "needs a chat strategy (ChatPrompts) that renders the engine's chat template"
            )
        items = body.get("input")
        if isinstance(items, str) or _is_token_ids(items):
            items = [items]
        prompts = tuple(_prompt(item) for item in items or [])
        flag = body.get("add_special_tokens", self.add_special)
        flag = flag if isinstance(flag, bool) else self.add_special
        return PromptSet(prompts, (flag,) * len(prompts), self.slot, tuple(range(len(prompts))))


@dataclass(frozen=True)
class ChatPrompts(PromptStrategy):
    """The chat-shaped (``messages``) requests: one engine prompt per conversation.

    The engine frames each conversation with its chat template and expands every media part; ``render`` is
    that frame (the wiring's, over the conversation's text parts: the served template's own render, the
    media parts dropped, so the count is the engine's text half), and ``media`` turns one media part as
    sent into its :class:`MediaIdentity` and the tokens the engine adds for it (the product's own media
    count under the recipe's declared policies).  ``media`` is ``None`` for a text-only chat strategy: a
    media part is then refused, never keyed by its URI.

    A batch (a list of conversations) yields one prompt per conversation, in order, exactly as vLLM's chat
    path frames a list of conversations.
    """

    render: Callable[[Sequence[Any], bool], str]
    slot: Literal["vector", "token_vector"] = "vector"
    media: Callable[[Any], tuple[MediaIdentity, int]] | None = None
    add_special: bool = False

    def prompts(self, body: Mapping[str, Any]) -> PromptSet:
        conversations = _conversations(body)
        generation = bool(body.get("add_generation_prompt", False))
        prompts: list[Prompt] = []
        for conversation in conversations:
            identities: list[MediaIdentity] = []
            tokens = 0
            for part in _media_parts(conversation):
                if self.media is None:
                    raise EmulatorUnmodelledError(
                        f"the request carries a media part ({part.get('type')!r}) and this chat strategy "
                        "declares no media model: build it with the recipe's declared processing"
                    )
                identity, cost = self.media(part)
                identities.append(identity)
                tokens += int(cost)
            prompts.append(
                MediaPrompt(
                    self.render(conversation, generation),
                    tuple(identities),
                    tokens,
                    _conversation_placement(conversation),
                )
            )
        flag = body.get("add_special_tokens", self.add_special)
        flag = flag if isinstance(flag, bool) else self.add_special
        return PromptSet(tuple(prompts), (flag,) * len(prompts), self.slot, tuple(range(len(prompts))))


@dataclass(frozen=True)
class RequestPrompts(PromptStrategy):
    """The role route's per-body dispatch: the completion-shaped strategy for an ``input`` body, the chat
    strategy for a ``messages`` body.

    A recipe that declares ``request_shape: text`` still sends every media item as its own ``messages``
    request (the chat route is the only shape in which the server applies its chat template), so the two
    derivations live side by side and the body's own shape selects one.  Without ``chat``, a ``messages``
    body is refused by the text strategy, which says so.
    """

    text: PromptStrategy
    chat: PromptStrategy | None = None

    def prompts(self, body: Mapping[str, Any]) -> PromptSet:
        if "messages" in body and self.chat is not None:
            return self.chat.prompts(body)
        return self.text.prompts(body)


@dataclass(frozen=True)
class PairPrompts(PromptStrategy):
    """A pointwise ``/rerank``: the engine renders **one pair prompt per document** with the recipe's
    template (the same render the client budgets against -- byte-identical to the engine's pair builder
    per the recipes' declarations), and scores each pair independently.

    A side carrying media is the wire's ``{"content": [parts]}``: its text parts are the pair template's
    span and its media parts are keyed by content identity beside it (``media``), with the engine's tokens
    for them added to the pair's count.  The query's media ride every pair prompt of the request.
    """

    template: Any
    tokenizer: Any
    media: Callable[[Any], tuple[MediaIdentity, int]] | None = None

    def prompts(self, body: Mapping[str, Any]) -> PromptSet:
        query, query_media, query_tokens, query_placement = _side(body.get("query"), self.media)
        documents = [_side(document, self.media) for document in body.get("documents") or []]
        prompts: list[Prompt] = []
        for document, media, tokens, placement in documents:
            rendered = self.template.render("pair", self.tokenizer, query=query, document=document)
            identities = (*query_media, *media)
            prompts.append(
                MediaPrompt(rendered, identities, query_tokens + tokens, (*query_placement, *placement))
                if identities
                else rendered
            )
        flag = self.template.adds_special_tokens("pair")
        return PromptSet(tuple(prompts), (flag,) * len(prompts), "score", tuple(range(len(documents))))


@dataclass(frozen=True)
class EnginePrompts(PromptStrategy):
    """A listwise ``/rerank``: the engine renders **one N-passage prompt per request** (the checkpoint's
    own builder -- a recipe reference's verbatim port -- given here as ``builder(query, documents)``)
    and scores the set in one call, so results depend on the set and its order (observed inputs cover
    whole prompts, exactly).  A media side is refused: the builder takes text passages."""

    builder: Callable[[str, Sequence[str]], str]
    add_special: bool = True

    def prompts(self, body: Mapping[str, Any]) -> PromptSet:
        query = _text(body.get("query"))
        documents = [_text(document) for document in body.get("documents") or []]
        return PromptSet((self.builder(query, documents),), (self.add_special,), "score_list", (0,))


def _conversations(body: Mapping[str, Any]) -> list[list[Any]]:
    """A ``messages`` body as its conversations: a single conversation (a list of message objects) or a
    batch (a list of conversations)."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise EmulatorUnmodelledError("a messages request needs a non-empty list of conversations")
    if all(isinstance(entry, list) for entry in messages):
        return [list(conversation) for conversation in messages]
    return [list(messages)]


def _media_parts(conversation: Sequence[Any]) -> list[dict[str, Any]]:
    """Every media part of one conversation, in message and part order (an ``image_url`` or ``video_url``
    part; a message whose content is a plain string carries none)."""
    parts: list[dict[str, Any]] = []
    for message in conversation:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "video_url"):
                parts.append(part)
    return parts


def _side(
    value: Any, media: Callable[[Any], tuple[MediaIdentity, int]] | None
) -> tuple[str, tuple[MediaIdentity, ...], int, tuple[str, ...]]:
    """One rerank side as ``(text, media identities, media tokens, placement)``: a string is its own text
    (placement ``("text",)``); a ``{"content": [parts]}`` object splits into its text parts (joined as the
    product's :attr:`~rcp_ndcg_core.content.Content.text` joins them) and its media parts (each keyed by
    ``media``), with the part kinds in their given order as the placement."""
    if isinstance(value, str):
        return value, (), 0, ("text",)
    if not isinstance(value, dict) or not isinstance(value.get("content"), list):
        raise EmulatorUnmodelledError(
            f"a rerank side of type {type(value).__name__} is neither a string nor a content-parts object; "
            "the emulator models those two shapes"
        )
    from rcp_ndcg_core.content import TEXT_JOIN

    parts = value["content"]
    texts = [
        _text(part) for part in parts if not (isinstance(part, dict) and part.get("type") in ("image_url", "video_url"))
    ]
    identities: list[MediaIdentity] = []
    tokens = 0
    placement: list[str] = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") in ("image_url", "video_url"):
            if media is None:
                raise EmulatorUnmodelledError(
                    f"the request carries a media part ({part.get('type')!r}) and this pair strategy "
                    "declares no media model: build it with the recipe's declared processing"
                )
            identity, cost = media(part)
            identities.append(identity)
            tokens += int(cost)
            placement.append("image" if part.get("type") == "image_url" else "video")
        else:
            placement.append("text")
    return TEXT_JOIN.join(texts), tuple(identities), tokens, tuple(placement)


def _conversation_placement(conversation: Sequence[Any]) -> tuple[str, ...]:
    """The kinds of one conversation's content parts, in order: ``text``, ``image`` or ``video`` per part,
    across every message (the engine places each vision block where its part stands, so the order is part of
    what the model reads)."""
    kinds: list[str] = []
    for message in conversation:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else "text"
            kinds.append("image" if kind == "image_url" else "video" if kind == "video_url" else "text")
    return tuple(kinds)


def _document_text(value: Any) -> str:
    """One rerank document as the engine echoes it in its reply: a string as given, a content-parts object
    as its text parts joined (the media contributes nothing here, as the product's ``Content.text`` reads
    it) -- the reply's ``document.text`` field, never a media part stringified."""
    if isinstance(value, dict) and isinstance(value.get("content"), list):
        from rcp_ndcg_core.content import TEXT_JOIN

        return TEXT_JOIN.join(
            _text(part)
            for part in value["content"]
            if not (isinstance(part, dict) and part.get("type") in ("image_url", "video_url"))
        )
    return _text(value)


def _is_token_ids(item: Any) -> bool:
    return isinstance(item, list) and bool(item) and all(isinstance(v, int) and not isinstance(v, bool) for v in item)


def _prompt(item: Any) -> Prompt:
    """One ``input`` item as the engine takes it: token ids as sent, anything else as its text."""
    return tuple(item) if _is_token_ids(item) else _text(item)


def _text(item: Any) -> str:
    """One wire text: a string itself, or a ``{"text": ...}`` / content-parts object by its text; any
    other shape -- a media part above all -- is refused with
    :class:`~rcp_ndcg_test.errors.EmulatorUnmodelledError` (a stringified object is never a prompt, and
    a bare ``ValueError`` would end a whole corpus build)."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        if isinstance(item.get("text"), str):
            return item["text"]
        parts = item.get("content")
        if isinstance(parts, list):
            return " ".join(_text(part) for part in parts)
        if item.get("type") in ("image_url", "video_url"):
            raise EmulatorUnmodelledError(
                f"the item is a media part ({item['type']!r}), not a text; a strategy that models media "
                "keys it by content identity (a ChatPrompts or a PairPrompts with its media callable)"
            )
    raise EmulatorUnmodelledError(f"an input item of type {type(item).__name__} is not a text the emulator models")


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
        is_matryoshka: Whether the served model's config carries the Matryoshka gate (vLLM's
            ``matryoshka_dimensions`` or ``is_matryoshka``; a recipe declares it through
            ``serve.hf_overrides``): without it, a request's ``dimensions`` is refused.
        matryoshka_dimensions: The declared Matryoshka set, when the checkpoint names one; ``None`` means
            the gate admits every integer in range (a card's prose range).
        embedding_size: The checkpoint's full output width (vLLM's ``embedding_size``); ``None`` means the
            width the corpus observed (the emulator's ``dim``).
    """

    engine_name: str
    engine_version: str
    served_name: str
    model_root: str
    max_model_len: int
    is_matryoshka: bool = False
    matryoshka_dimensions: tuple[int, ...] | None = None
    embedding_size: int | None = None


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
    ``parts`` (the product's scalar draw, :func:`~rcp_ndcg.inference.fake.fake_uniform`), stable on every
    machine and in every call order. Marked in every reply it answers (``x-rcp-ndcg-emulator-source``)."""
    from rcp_ndcg.inference.fake import fake_uniform

    return [2.0 * fake_uniform(seed, "surrogate-score", *parts, index) - 1.0 for index in range(count)]


def surrogate_vector(seed: int, *parts: object, dim: int) -> list[float]:
    """The declared deterministic surrogate: one unit vector of width ``dim`` from **one** seeded draw
    per vector -- a SHAKE-256 stream of the parts read as ``dim`` uniforms, as the offline fake draws its
    vectors -- never one hash per component. Its values are declared surrogate, never pinned."""
    import numpy as np

    stream = hashlib.shake_256("|".join(str(part) for part in (seed, "surrogate-vector", *parts)).encode("utf-8"))
    raw = (np.frombuffer(stream.digest(8 * dim), dtype=">u8") >> np.uint64(11)) * 2.0**-53 * 2.0 - 1.0
    return (raw / (float(np.linalg.norm(raw)) or 1.0)).tolist()


def surrogate_matrix(seed: int, *parts: object, tokens: int, dim: int) -> list[list[float]]:
    """One deterministic surrogate token-vector matrix, ``tokens`` rows of :func:`surrogate_vector`."""
    return [surrogate_vector(seed, *parts, token, dim=dim) for token in range(tokens)]


def _slice_normalised(vector: Sequence[float], dimensions: int) -> list[float]:
    """One vector's Matryoshka cut: the first ``dimensions`` values, L2-normalised (the engine's order).

    vLLM's head slices the raw post-projector output and then applies ``PoolerNormalize``
    (``seqwise/heads.py``): slicing an already-normalised vector without renormalising would ship a
    non-unit cut, and drawing a fresh ``dimensions``-wide vector would hide a wrong order or set.
    """
    import numpy as np

    cut = np.asarray(vector, dtype=np.float64)[:dimensions]
    norm = float(np.linalg.norm(cut))
    return (cut / norm).tolist() if norm else cut.tolist()


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
        unmodelled_records: One entry per recorded 2xx exchange the model layer could not model (the
            request's unmodelled fields, or a prompt derivation the strategy refused), each naming the
            record and the reason.  The corpus still builds: an unmodelled record is skipped and named
            here, never a silent drop and never a whole-corpus failure (a request for it later answers the
            declared surrogate, marked ``surrogate``).
        verified: What it was verified against; a different engine version or recipe revision is
            refused.
    """

    facts: EngineFacts
    strategy: PromptStrategy
    tokenizer: Any
    observations: dict[str, tuple[ModelObservation, ...]] = field(default_factory=dict)
    verified: Verified | None = None
    unmodelled_records: tuple[str, ...] = ()
    dim: int = 64
    unverified_rules: tuple[str, ...] = (
        "malformed JSON -> 400",
        "empty input -> 400",
        "model name mismatch -> 404",
        "top_n > documents -> the whole ranked list",
        "the base64 and bytes framings on a route the corpus recorded as float JSON",
    )
    """Protocol rules the emulator follows from the engine's source without a recording that shows
    them (beside :attr:`unobserved_routes`); listed in the verification record."""
    observed_routes: frozenset[str] = frozenset()
    """The :data:`EMULATED_ROUTES` the corpus recorded (any status)."""
    _counter: list[int] = field(default_factory=lambda: [0])
    answer_log: list[str] = field(default_factory=list)
    """The provenance of every composed model-output reply (``/embeddings``, ``/pooling``, ``/rerank``;
    errors and ``/models`` ``/tokenize`` carry no model output and are not logged): what tells a
    numbers-asserting test whether its inputs were observed (GPU-VALIDATION item 2)."""

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_corpus(
        cls,
        corpus: ObservationCorpus,
        strategy: PromptStrategy,
        tokenizer: Any,
        facts: EngineFacts,
        *,
        dim: int | None = None,
    ) -> VllmEmulator:
        """Build the emulator of ``corpus``: replay table from the recorded 2xx bodies, facts given.

        A recorded request the model layer cannot model (an unmodelled field, a chat body without a chat
        strategy, a media part without a media model) is **skipped and named** in
        :attr:`unmodelled_records` -- one such record never fails the whole corpus, and a request for it
        later answers the declared surrogate.  A corpus that is internally inconsistent (two records
        answering one replay key with different outputs) still raises: that is a wrong key, not an
        unmodelled request.

        Args:
            corpus: The loaded observation corpus.
            strategy: The prompts derivation matching the recipe's role and scoring.
            tokenizer: The recipe's real :class:`~rcp_ndcg.data.tokenizer.TextTokenizer`.
            facts: The engine facts (served name, cap).
            dim: The surrogate's vector width; ``None`` takes the width the corpus observed (64 when it
                observed no vector), so a batch mixing replayed and surrogate vectors is never ragged.

        Returns:
            The emulator, keyed to the corpus's behaviour fingerprint (model layer) and the engine
            version (protocol layer).
        """
        observed: dict[str, list[tuple[int, ModelObservation]]] = {}
        unmodelled_records: list[str] = []
        exchanges = exchanges_of(corpus)
        for exchange in exchanges:
            route = route_of(exchange.path)
            if exchange.status != 200 or route is None:
                continue
            body = exchange.request_body if isinstance(exchange.request_body, dict) else {}
            where = f"{exchange.source or exchange.path} #{exchange.sequence}"
            context, unmodelled = request_context(route, body)
            if unmodelled:
                unmodelled_records.append(f"{where}: the recorded request carries {unmodelled}, unmodelled")
                continue
            try:
                set_ = replace(strategy.prompts(body), context=_canonical_context(context))
            except EmulatorUnmodelledError as error:
                # A request the model layer has no derivation for (a chat body without a chat strategy, a
                # media part without a media model) is skipped and NAMED: one unmodelled record never fails
                # the whole corpus, and a replayed request for it answers the declared surrogate, marked.
                unmodelled_records.append(f"{where}: {error}")
                continue
            for key, observation in _outputs_from_response(set_, exchange):
                observed.setdefault(key, []).append((exchange.sequence, observation))
        tolerance = corpus_tolerance(corpus)
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
        routes = {route_name(exchange.method, exchange.path) for exchange in exchanges}
        widths = {
            width
            for history in merged.values()
            for observation in history
            if (width := _width(observation)) is not None
        }
        if dim is None:
            # One observed width is the model's own; a corpus that observed several (a Matryoshka cut
            # recorded beside the full width) takes the widest, the full width the head slices from.
            dim = max(widths) if widths else 64
        manifest_recipe = corpus.manifest["recipe"]
        return cls(
            facts=facts,
            strategy=strategy,
            tokenizer=tokenizer,
            observations=merged,
            observed_routes=frozenset(route for route in routes if route is not None),
            dim=dim,
            unmodelled_records=tuple(unmodelled_records),
            verified=Verified(
                engine_name=str(corpus.manifest["engine"]["name"]),
                engine_version=str(corpus.manifest["engine"]["version"]),
                recipe_id=str(manifest_recipe["id"]),
                revision=str(corpus.manifest["model"]["revision"]),
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
        response = self._answer(path, method.upper(), body if isinstance(body, dict) else {})
        name = route_name(method, path)
        response.headers["x-rcp-ndcg-emulator-route"] = "observed" if name in self.observed_routes else "unobserved"
        return response

    @property
    def unobserved_routes(self) -> tuple[str, ...]:
        """The :data:`EMULATED_ROUTES` no recording of the corpus covers (their replies follow the
        engine's source, unverified)."""
        return tuple(route for route in EMULATED_ROUTES if route not in self.observed_routes)

    def _answer(self, path: str, method: str, body: Mapping[str, Any]) -> httpx.Response:
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
            return None, self._unmodelled_refusal(
                f"the verified fake engine does not model the request field(s) {unmodelled} (no recording "
                "shows the engine's behaviour for them); it refuses instead of answering from another "
                "request's observation",
                param=unmodelled[0],
            )
        try:
            set_ = replace(self.strategy.prompts(body), context=_canonical_context(context))
        except EmulatorUnmodelledError as error:
            return None, self._unmodelled_refusal(str(error), param="input")
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

    def _unmodelled_refusal(self, message: str, *, param: str) -> httpx.Response:
        """The marked 400 for a request the emulator does not model: an unmodelled field, or a prompt
        derivation the strategy refused (:class:`~rcp_ndcg_test.errors.EmulatorUnmodelledError`).  The
        same shape on both paths, so a caller can tell "not modelled" from "the engine would refuse"."""
        refusal = _error(400, message, param=param, kind="EmulatorUnmodelledError")
        refusal.headers["x-rcp-ndcg-emulator"] = "rcp-ndcg.testing.engines"
        refusal.headers["x-rcp-ndcg-emulator-source"] = "refused-unmodelled"
        self.answer_log.append("refused-unmodelled")
        return refusal

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
        if dimensions is not None:
            refusal = self._matryoshka_refusal(dimensions)
            if refusal is not None:
                return refusal
        encoding, dtype, endianness, refusal = self._encoding(body, ("float", "base64"))
        if refusal is not None:
            return refusal
        sources, data = [], []
        for index in range(len(set_.prompts)):
            key = set_.item_key(index)
            observation, source = self._observation(key)
            sources.append(source)
            if observation.vector is not None:
                vector = list(observation.vector)  # observed under this very context (dimensions included)
            else:
                # The full-width surrogate, sliced for a requested k: the engine's head is projector ->
                # slice -> activation, so the slice is taken from the full-width vector BEFORE the L2.  A
                # replayed observation is the engine's own answer for that exact context, verbatim.
                vector = surrogate_vector(0, "embedding", set_.model_key(index), dim=self._full_width())
                if dimensions is not None:
                    vector = _slice_normalised(vector, dimensions)
            data.append({"index": index, "object": "embedding", "embedding": vector})
        if encoding == "base64":
            for item in data:
                item["embedding"] = _encode_base64([item["embedding"]], dtype, endianness)
        usage = self._usage(set_)
        return self._marked(
            _json(
                200,
                {
                    "id": f"embd-{self._volatile_hex(16)}",
                    "object": "list",
                    "created": self._volatile_time(),
                    "model": self.facts.served_name,
                    "data": data,
                    "usage": _usage_body(usage, wide=True),
                },
            ),
            sources,
        )

    def _full_width(self) -> int:
        """The checkpoint's full output width: the declared ``embedding_size``, else the observed width."""
        return int(self.facts.embedding_size or self.dim)

    def _matryoshka_refusal(self, dimensions: int) -> httpx.Response | None:
        """vLLM's three ``dimensions`` gates, in the engine's order (``pooling_params.py``).

        ``is_matryoshka`` first, then ``1 <= k <= embedding_size``, then membership in the declared
        ``matryoshka_dimensions``; the error bodies are the engine's own (``VLLMValidationError`` renders as
        ``BadRequestError`` with the exception's parameter, ``None`` here).
        """
        name = self.facts.served_name
        if not self.facts.is_matryoshka:
            return _error(
                400,
                f"Model {name!r} does not support Matryoshka embeddings; dimensions must be unset "
                f"(received dimensions={dimensions}).",
            )
        size = self._full_width()
        if not 1 <= dimensions <= size:
            return _error(400, f"Model {name!r} only supports dimensions in range [1, {size}], got {dimensions}.")
        declared = self.facts.matryoshka_dimensions
        if declared is not None and dimensions not in declared:
            return _error(
                400,
                f"Model {name!r} only supports Matryoshka dimensions {list(declared)}, got {dimensions}.",
            )
        return None

    def _marked(self, response: httpx.Response, sources: Sequence[str]) -> httpx.Response:
        """Compose the reply's provenance metadata and record it: the observed-inputs guard that lets a
        numbers-asserting test require every input to be replayed."""
        response = _marked(response, sources)
        self.answer_log.append(response.headers.get("x-rcp-ndcg-emulator-source", "surrogate"))
        return response

    def _pooling(self, body: Mapping[str, Any]) -> httpx.Response:
        """``POST /pooling``: the token-vector matrix per input, in the request's framing (vLLM
        v0.31.0's ``PoolingResponse``, or the ``bytes`` framing of
        ``vllm/entrypoints/pooling/utils.py::build_pooling_bytes_streaming_response``).

        The route refuses a per-request ``dimensions`` outright, whatever the checkpoint declares
        (``vllm/entrypoints/pooling/pooling/serving.py``: "dimensions is currently not supported")."""
        if body.get("dimensions") is not None:
            return _error(400, "dimensions is currently not supported", param="dimensions")
        set_, error = self._prompts_or_error("pooling", body)
        if error is not None:
            return error
        assert set_ is not None
        encoding, dtype, endianness, refusal = self._encoding(body, ("float", "base64", "bytes", "bytes_only"))
        if refusal is not None:
            return refusal
        sources, matrices = [], []
        for index in range(len(set_.prompts)):
            key = set_.item_key(index)
            observation, source = self._observation(key)
            sources.append(source)
            if observation.matrix is not None:
                matrices.append([list(row) for row in observation.matrix])
            else:
                tokens = len(set_.ids(index, self.tokenizer))
                matrices.append(surrogate_matrix(0, "pooling", key, tokens=tokens, dim=self.dim))
        usage = self._usage(set_)
        identity = {
            "id": f"pool-{self._volatile_hex(16)}",
            "created": self._volatile_time(),
            "model": self.facts.served_name,
        }
        if encoding in ("bytes", "bytes_only"):
            raw, framing = bytearray(), []
            for index, matrix in enumerate(matrices):
                frame = _pack_frame(matrix, dtype, endianness)
                framing.append(
                    {
                        "index": index,
                        "embed_dtype": dtype,
                        "endianness": endianness,
                        "start": len(raw),
                        "end": len(raw) + len(frame),
                        "shape": [len(matrix), len(matrix[0]) if matrix else 0],
                    }
                )
                raw += frame
            headers = {"content-type": "application/octet-stream"}
            if encoding == "bytes":  # bytes_only sends no metadata (the product's adapter refuses it)
                metadata = {**identity, "data": framing, "usage": _usage_body(usage, wide=False)}
                headers["metadata"] = json.dumps(metadata)
            return self._marked(httpx.Response(200, content=bytes(raw), headers=headers), sources)
        data = [
            {
                "index": index,
                "object": "pooling",
                "data": matrix if encoding == "float" else _encode_base64(matrix, dtype, endianness),
            }
            for index, matrix in enumerate(matrices)
        ]
        body_out = {
            "id": identity["id"],
            "object": "list",
            "created": identity["created"],
            "model": identity["model"],
            "data": data,
            "usage": _usage_body(usage, wide=True),
        }
        return self._marked(_json(200, body_out), sources)

    def _encoding(
        self, body: Mapping[str, Any], formats: tuple[str, ...]
    ) -> tuple[str, str, str, httpx.Response | None]:
        """The request's framing (vLLM's defaults: ``float``, ``float32``, ``native``); a framing the
        emulator cannot produce (an fp8 dtype) is refused marked, never approximated."""
        encoding = str(body.get("encoding_format") or "float")
        dtype = str(body.get("embed_dtype") or "float32")
        endianness = str(body.get("endianness") or "native")
        if encoding not in formats:
            return (
                encoding,
                dtype,
                endianness,
                _error(400, f"invalid encoding_format {encoding!r}", param="encoding_format"),
            )
        if dtype not in _DTYPES or endianness not in ("native", "little", "big"):
            refusal = _error(
                400,
                f"the verified fake engine does not model embed_dtype={dtype!r} / endianness={endianness!r}",
                param="embed_dtype",
                kind="EmulatorUnmodelledError",
            )
            refusal.headers["x-rcp-ndcg-emulator-source"] = "refused-unmodelled"
            self.answer_log.append("refused-unmodelled")
            return encoding, dtype, endianness, refusal
        return encoding, dtype, endianness, None

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
        if isinstance(top_n, int) and 0 < top_n < len(scored):  # vLLM: 0 (its default) means every document
            scored = scored[:top_n]
        results = [
            {
                "index": entry["index"],
                "document": {"text": _document_text(documents[entry["index"]]), "multi_modal": None},
                "relevance_score": entry["relevance_score"],
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
                    "usage": _usage_body(usage, wide=False),
                    "results": results,
                },
            ),
            sources,
        )

    def _tokenize(self, body: Mapping[str, Any]) -> httpx.Response:
        """``POST /tokenize`` (vLLM v0.31.0's ``TokenizeCompletionRequest`` -> ``TokenizeResponse``): the
        prompt's token ids with the recipe's real tokenizer, ``token_strs`` left ``null`` (the emulator
        does not model ``return_token_strs``)."""
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            return _error(400, "invalid request: 'prompt' must be a string", param="prompt")
        add_special = body.get("add_special_tokens", True)
        tokens = list(self.tokenizer.ids(prompt, add_special_tokens=bool(add_special)))
        return _json(
            200,
            {"count": len(tokens), "max_model_len": self.facts.max_model_len, "tokens": tokens, "token_strs": None},
        )

    def _usage(self, set_: PromptSet) -> int:
        return set_.counted(self.tokenizer)

    def _volatile_hex(self, digits: int) -> str:
        self._counter[0] += 1
        return f"{self._counter[0]:0{digits}x}"

    def _volatile_time(self) -> int:
        return 1_700_000_000 + self._counter[0]

    # -- the replay table -------------------------------------------------------

    def replays(self, key: str) -> bool:
        """Whether the observation ``key`` (a :meth:`PromptSet.item_key`) was recorded -- its outputs
        replay -- or unseen (surrogate)."""
        return key in self.observations


class BehaviourDiff(dict):
    """The behaviour diff of two corpora of one recipe (OBSERVATIONS-SPEC section 7): a plain mapping
    with ``schema: rcp-ndcg.behaviour-diff/1``, per-input deltas and the summary by route."""


def behaviour_diff(before: ObservationCorpus, after: ObservationCorpus) -> BehaviourDiff:
    """The **behaviour diff** of two corpora of one recipe: per input, the score or vector deltas,
    changed statuses, changed refusals and changed protocol behaviour, summarised by stratum (route,
    for corpora recorded without the generator's stratum labels).

    Args:
        before: The previous corpus (the one leaving).
        after: The new corpus (the candidate).

    Returns:
        The :class:`BehaviourDiff`, JSON-ready.
    """
    old: dict[str, Exchange] = {_key(exchange): exchange for exchange in exchanges_of(before)}
    new: dict[str, Exchange] = {_key(exchange): exchange for exchange in exchanges_of(after)}
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
            before_body = normalise_body(left.response_json)
            after_body = normalise_body(right.response_json)
            leaves_a, leaves_b = _float_leaves(before_body), _float_leaves(after_body)
            deltas = {
                path: round(leaves_b[path] - leaves_a[path], 12)
                for path in sorted(set(leaves_a) & set(leaves_b))
                if abs(leaves_a[path] - leaves_b[path]) > 0
            }
            row["numeric_deltas"] = deltas
            protocol = _diff_bodies(_frame_of(before_body), _frame_of(after_body), "", (0.0, 0.0))
            protocol += [
                f"header {name}: {left.response_headers.get(name)!r} -> {right.response_headers.get(name)!r}"
                for name in sorted(set(left.response_headers) | set(right.response_headers))
                if name.lower() != "metadata" and left.response_headers.get(name) != right.response_headers.get(name)
            ]
            if left.response_json is None or right.response_json is None:
                if left.raw_body != right.raw_body:
                    protocol.append("binary body bytes changed")
            row["protocol_changed"] = protocol
            row["changed"] = bool(deltas or protocol or left.status != right.status)
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
                "before": before.manifest["recipe"]["behaviour_fingerprint"],
                "after": after.manifest["recipe"]["behaviour_fingerprint"],
            },
            "inputs": inputs,
            "summary": {
                "inputs": len(inputs),
                "changed": sum(1 for row in inputs if row["changed"]),
                "strata": by_route,
            },
        }
    )


def _frame_of(body: Any) -> Any:
    """A normalised body without its model outputs (``results``, ``data``) and counts: the protocol part."""
    if not isinstance(body, dict):
        return body
    return {key: value for key, value in body.items() if key not in ("results", "data", "usage")}


def _key(exchange: Exchange) -> str:
    """An input across two corpora: the record's content address (the same request bytes)."""
    return f"{exchange.method} {exchange.path} {exchange.exchange_id[:16]}"


# ---------------------------------------------------------------------------
# the registry: (engine, version, fingerprint) + the entry-point seam
# ---------------------------------------------------------------------------


_LOCK = threading.Lock()


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
        registered = self.fingerprints(engine_name, engine_version, recipe_id)
        raise ConfigError(
            f"no emulator registered for ({engine_name}, {engine_version}, {recipe_id}, {fingerprint[:12]}...)"
            + (f"; registered fingerprints for that engine version and recipe: {registered}" if registered else "")
        )

    def fingerprints(self, engine_name: str, engine_version: str, recipe_id: str) -> list[str]:
        """The behaviour fingerprints registered for one recipe on one engine version, sorted."""
        return sorted(key[3] for key in self._by_key if key[:3] == (engine_name, engine_version, recipe_id))

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
    engine, version = split_engine_host(parts.hostname or parts.netloc)
    recipe_id = parts.path.strip("/")
    query = dict(parse_qsl(parts.query))
    fingerprint = query.get("fingerprint", "")
    if not fingerprint:
        registry.load_entry_points()
        candidates = registry.fingerprints(engine, version, recipe_id)
        if len(candidates) > 1:
            raise ConfigError(f"{url}: several fingerprints are registered ({candidates}); add ?fingerprint=<sha>")
        fingerprint = candidates[0] if candidates else ""
    return httpx.MockTransport(registry.resolve(engine, version, fingerprint, recipe_id).handle)


def split_engine_host(host: str) -> tuple[str, str]:
    """``vllm-0.31.0`` -> ``("vllm", "0.31.0")``: the engine name before its version, by the one
    pattern :data:`rcp_ndcg.inference.fake.RE_ENGINE_URL` routes to the emulators.

    Raises:
        ConfigError: the host is not an engine-version host.
    """
    from rcp_ndcg.inference.fake import RE_ENGINE_URL

    if not RE_ENGINE_URL.match(f"fake://{host}/"):
        raise ConfigError(f"{host!r} is not an engine-version host (e.g. 'vllm-0.31.0')")
    engine, version = host.split("-", 1)
    return engine, version


# ---------------------------------------------------------------------------
# the verification record (OBSERVATIONS-SPEC section 4: appended, never rewritten)
# ---------------------------------------------------------------------------


def verification_record(
    corpus: ObservationCorpus, problems: Sequence[str], *, verified_at: str, emulator: VllmEmulator | None = None
) -> dict[str, Any]:
    """The verification record of one conformance run over ``corpus`` (OBSERVATIONS-SPEC section 4):
    which emulator verified it, against which engine version and recipe revision, with which tolerances,
    the result, and every recorded exchange the emulator could not model (its
    :attr:`~VllmEmulator.unmodelled_records`; ``[]`` is a corpus the model layer covers completely).

    Args:
        corpus: The verified corpus.
        problems: The conformance differences the run found (``[]`` is a pass).
        verified_at: The run's date (``YYYY-MM-DD``).
        emulator: The verified emulator (its unobserved routes and unverified rules are recorded).

    Returns:
        The record, ready for :func:`rcp_ndcg_test.corpus.append_verification`.
    """
    from importlib.metadata import version

    recipe = corpus.manifest["recipe"]
    tolerance = corpus_tolerance(corpus)
    return {
        "schema": VERIFICATION_SCHEMA,
        "emulator": f"rcp_ndcg_test.engines (rcp-ndcg {version('rcp-ndcg')})",
        "engine": {"name": corpus.manifest["engine"]["name"], "version": corpus.manifest["engine"]["version"]},
        "recipe": {
            "id": recipe["id"],
            "revision": corpus.manifest["model"]["revision"],
            "behaviour_fingerprint": recipe["behaviour_fingerprint"],
        },
        "normalisation_version": NORMALISATION_VERSION,
        "tolerances": None if tolerance is None else {"abs": tolerance[0], "rel": tolerance[1]},
        "exchanges": len(corpus.records),
        "unobserved_routes": list(emulator.unobserved_routes) if emulator else [],
        "unverified_rules": list(emulator.unverified_rules) if emulator else [],
        "unmodelled_records": list(emulator.unmodelled_records) if emulator else [],
        "problems": list(problems),
        "result": "pass" if not problems else "fail",
        "verified_at": verified_at,
    }


# ---------------------------------------------------------------------------
# module-local helpers
# ---------------------------------------------------------------------------


def _width(observation: ModelObservation) -> int | None:
    """The vector width one observation shows (``None`` for a score)."""
    if observation.vector is not None:
        return len(observation.vector)
    if observation.matrix:
        return len(observation.matrix[0])
    return None


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

    Decodes vLLM v0.31.0's framings: JSON floats, ``base64`` vectors (``embed_dtype`` and
    ``endianness`` from the request, vLLM's defaults otherwise), and the ``bytes`` framing by its
    ``metadata`` header's per-item ``start``/``end``/``shape``. Anything it cannot decode -- a base64
    token matrix (vLLM sends no shape), a frame without metadata, a missing score -- is refused naming
    the record: a corpus whose outputs cannot be derived never reads as an empty or guessed table.
    """
    where = f"{exchange.source or exchange.path} #{exchange.sequence}"
    request = exchange.request_body if isinstance(exchange.request_body, dict) else {}
    dtype = str(request.get("embed_dtype") or "float32")
    endianness = str(request.get("endianness") or "native")
    body = exchange.response_json
    if body is None:  # a binary reply: the bytes framing
        header = {key.lower(): value for key, value in dict(exchange.response_headers).items()}.get("metadata")
        if not header:
            raise DataError(f"{where}: a binary response without its metadata framing cannot be decoded")
        raw = exchange.raw_body or b""
        frames = json.loads(header).get("data") or []
        if len(frames) != len(set_.prompts):
            raise DataError(f"{where}: {len(frames)} frames for {len(set_.prompts)} prompts")
        pairs = []
        for index, frame in enumerate(frames):
            chunk = raw[int(frame["start"]) : int(frame["end"])]
            values = _unpack_frame(
                chunk, str(frame.get("embed_dtype") or dtype), str(frame.get("endianness") or endianness)
            )
            shape = tuple(int(size) for size in frame["shape"])
            pairs.append((set_.item_key(index), _observation_of(set_.slot, values, shape, where)))
        return pairs
    if not isinstance(body, dict):
        raise DataError(f"{where}: a 2xx body that is not a JSON object cannot be decoded")
    if set_.slot in ("vector", "token_vector"):
        data = body.get("data")
        if not isinstance(data, list) or len(data) != len(set_.prompts):
            raise DataError(f"{where}: {len(data or [])} data items for {len(set_.prompts)} prompts")
        pairs = []
        for index, item in enumerate(sorted(data, key=lambda entry: int(entry.get("index", 0)))):
            payload = item.get("embedding" if set_.slot == "vector" else "data")
            if isinstance(payload, str):
                if set_.slot == "token_vector":
                    raise DataError(
                        f"{where}: a base64 token-vector item carries no shape (vLLM v0.31.0 sends none); "
                        "record /pooling in float or bytes framing"
                    )
                values = _unpack_frame(_b64(payload, where), dtype, endianness)
                pairs.append((set_.item_key(index), ModelObservation(vector=tuple(values))))
            elif isinstance(payload, list) and set_.slot == "vector":
                pairs.append((set_.item_key(index), ModelObservation(vector=tuple(float(v) for v in payload))))
            elif isinstance(payload, list) and all(isinstance(row, list) for row in payload):
                matrix = tuple(tuple(float(v) for v in row) for row in payload)
                pairs.append((set_.item_key(index), ModelObservation(matrix=matrix)))
            else:
                raise DataError(f"{where}: item {index} carries no decodable {set_.slot} payload")
        return pairs
    results = body.get("results")
    if not isinstance(results, list) or any("relevance_score" not in entry for entry in results):
        raise DataError(f"{where}: a rerank reply without scored results cannot be decoded")
    scores = {int(entry.get("index", i)): float(entry["relevance_score"]) for i, entry in enumerate(results)}
    if set_.slot == "score":
        return [
            (set_.item_key(position), ModelObservation(score=scores.get(position)))
            for position in range(len(set_.prompts))
        ]
    return [(set_.set_key, ModelObservation(scores=tuple(sorted(scores.items()))))]


def _observation_of(slot: str, values: list[float], shape: tuple[int, ...], where: str) -> ModelObservation:
    import math

    if math.prod(shape) != len(values):
        raise DataError(f"{where}: a frame of {len(values)} values does not fill its declared shape {list(shape)}")
    if slot == "vector" and len(shape) == 1:
        return ModelObservation(vector=tuple(values))
    if slot == "token_vector" and len(shape) == 2:
        width = shape[1]
        return ModelObservation(matrix=tuple(tuple(values[row * width : (row + 1) * width]) for row in range(shape[0])))
    raise DataError(f"{where}: a frame of shape {list(shape)} is not a {slot}")


def _b64(payload: str, where: str) -> bytes:
    import base64
    import binascii

    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise DataError(f"{where}: an undecodable base64 payload ({error})") from error


def _unpack_frame(raw: bytes, dtype: str, endianness: str) -> list[float]:
    """The inverse of :func:`_pack_frame` (refusing a dtype or a length it cannot decode)."""
    import sys

    import numpy as np

    if dtype not in _DTYPES:
        raise DataError(f"an embed_dtype the emulator cannot decode: {dtype!r}")
    width = 4 if dtype == "float32" else 2
    if len(raw) % width:
        raise DataError(f"a {dtype} frame of {len(raw)} bytes is not whole values")
    order = sys.byteorder if endianness == "native" else endianness
    marker = "<" if order == "little" else ">"
    if dtype == "bfloat16":
        upper = np.frombuffer(raw, dtype=marker + "u2").astype(np.uint32) << np.uint32(16)
        return [float(v) for v in upper.view(np.float32)]
    return [float(v) for v in np.frombuffer(raw, dtype=marker + ("f4" if dtype == "float32" else "f2"))]


def _usage_body(usage: int, *, wide: bool) -> dict[str, Any]:
    """The route's usage shape: ``/rerank`` and the ``bytes`` metadata report ``RerankUsage``'s two keys,
    ``/v1/embeddings`` and ``/pooling`` vLLM's ``UsageInfo`` (five keys, in its field order)."""
    if not wide:
        return {"prompt_tokens": usage, "total_tokens": usage}
    return {
        "prompt_tokens": usage,
        "total_tokens": usage,
        "completion_tokens": 0,
        "prompt_tokens_details": None,
        "completion_tokens_details": None,
    }


def _error(status: int, message: str, *, param: str | None = None, kind: str = "BadRequestError") -> httpx.Response:
    """vLLM's ``ErrorResponse`` (``ErrorInfo``: message, type, param, code)."""
    return _json(status, {"error": {"message": message, "type": kind, "param": param, "code": status}})


def _json(status: int, body: Any) -> httpx.Response:
    """A JSON reply rendered as the engine renders it (Starlette's ``JSONResponse``: compact separators,
    UTF-8, no ASCII escaping), so its raw bytes compare with recorded raw bytes."""
    content = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    return httpx.Response(status, headers={"content-type": "application/json", "server": "uvicorn"}, content=content)


def _marked(response: httpx.Response, sources: Sequence[str]) -> httpx.Response:
    """The reply's provenance metadata: every input observed (``replayed``), none (``surrogate``) or
    ``mixed`` -- what tells a numbers-asserting test whether its inputs were observed."""
    kinds = set(sources)
    overall = sources[0] if len(kinds) == 1 else ("mixed" if kinds else "surrogate")
    response.headers["x-rcp-ndcg-emulator"] = "rcp-ndcg.testing.engines"
    response.headers["x-rcp-ndcg-emulator-source"] = overall
    return response


_DTYPES = ("float32", "float16", "bfloat16")
"""The ``embed_dtype`` values the emulator packs exactly as vLLM's ``tensor2binary`` does
(``vllm/utils/serial_utils.py``); the fp8 ones are refused as unmodelled."""


def _pack_frame(matrix: Sequence[Sequence[float]] | Sequence[float], dtype: str, endianness: str) -> bytes:
    """Values in ``dtype`` and ``endianness`` (``native`` is the host's), flattened -- ``tensor2binary``;
    ``bfloat16`` is float32 rounded to nearest even on the upper 16 bits, as torch converts."""
    import sys

    import numpy as np

    values = np.asarray(matrix, dtype=np.float32).ravel()
    if dtype == "bfloat16":
        bits = values.view(np.uint32)
        rounded = ((bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(
            np.uint16
        )
        array: Any = rounded
    else:
        array = values.astype(np.float16 if dtype == "float16" else np.float32)
    order = sys.byteorder if endianness == "native" else endianness
    if order != sys.byteorder:
        array = array.byteswap()
    return array.tobytes()


def _encode_base64(matrix: Sequence[Sequence[float]], dtype: str, endianness: str) -> str:
    """The ``base64`` framing of one item: :func:`_pack_frame`, base64-encoded."""
    import base64

    return base64.b64encode(_pack_frame(matrix, dtype, endianness)).decode("ascii")
