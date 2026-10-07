"""The fake-engine seam: what a recipe-level fake must implement, and the registry by recipe id.

A fake engine sits *below* the product's transport: the conformance runner wraps it in an ``httpx``
transport, so the role client, the wire adapter and the transport are all the product's, and the fake
answers the routes the way an engine would -- per request path, with statuses and bodies the adapters
parse as they parse a real engine's.

The protocol is deliberately small (:class:`FakeEngine`): a recipe id and one ``handle`` method per
request. What a recipe-level fake must answer:

- the role's route(s) the recipe's ``client.api`` selects -- ``POST .../embeddings`` for
  ``openai_embeddings``, ``POST .../pooling`` for ``vllm_pooling``, ``POST .../rerank`` for ``rerank``,
  plus ``GET .../models`` (the transport's provenance probe);
- ``POST /tokenize``: the engine's tokenization ground truth (the observations spec's section 1) --
  the ids the recipe's tokenizer produces for the prompt, so anything that cross-checks the client's
  ``fit`` against the engine can do so offline.

The registry maps recipe ids to fakes. **No model-level fake ships yet**: the verified emulators are
built from the GPU recordings later (the fake-engines lane; the product's hash-based ``fake://`` fakes
stay in the product). What ships today is one test fake for the packaged fixture recipe
(:class:`FakeEmbedEngine`, recipe ``fake-embed``), so the runner is exercised end to end on CPU. A test
fake is a deterministic surrogate, clearly marked -- a test that asserts numbers may only do so against
inputs whose answer it recorded.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rcp_ndcg.inference.fake import fake_uniform

from .errors import ConformanceError

__all__ = [
    "FIXTURE_RECIPE_ID",
    "FakeEmbedEngine",
    "FakeEngine",
    "FakeReply",
    "fake_engine_for",
    "fake_http_transport",
    "fixture_path",
    "package_tokenizer_path",
    "register_fake_engine",
    "registered_fake_engines",
    "unregister_fake_engine",
]


@dataclass(frozen=True)
class FakeReply:
    """One fake route's answer: an HTTP status and the JSON body the adapter will parse.

    Attributes:
        status: The HTTP status (``200`` on success; an error status with the engine's refusal body).
        json_body: The JSON-serialisable body; ``None`` sends none.
        content_type: The reply's content type (``application/json``).
    """

    status: int
    json_body: Any = None
    content_type: str = "application/json"


@runtime_checkable
class FakeEngine(Protocol):
    """What a recipe-level fake implements to answer the role routes and ``/tokenize``.

    Attributes:
        recipe_id: The recipe id the fake answers for (the registry's key; the recipe's canonical id).
        name: A short name for logs and the reply's metadata (``"shipped-embed-fake"`` and friends).
    """

    recipe_id: str
    name: str

    def handle(self, method: str, path: str, body: Any) -> FakeReply:
        """One request in, one :class:`FakeReply` out.

        ``method`` is ``"GET"`` or ``"POST"``; ``path`` is the request path as the transport built it
        (the base URL plus the adapter's route path, e.g. ``/v1/embeddings`` -- match on the route
        suffix, as the product's own fakes do); ``body`` is the parsed JSON request body, or ``None``
        when the request carried none.
        """
        ...


class _FakeHTTPTransport:
    """An ``httpx`` transport that answers from a :class:`FakeEngine` (built by
    :func:`fake_http_transport`).

    The product's :class:`~rcp_ndcg.inference.transport.Transport` sends every call through it: the
    fake sees the requests exactly as the wire adapter built them, and the adapters parse its answers
    as they parse an engine's.
    """

    def __init__(self, engine: FakeEngine) -> None:
        self._engine = engine

    async def handle_async_request(self, request: Any) -> Any:
        """One request in, one :class:`httpx.Response` out, from the fake engine's answer."""
        import httpx

        body: Any = None
        if request.content:
            try:
                body = json.loads(request.content)
            except ValueError as error:
                return httpx.Response(
                    400,
                    json={
                        "error": {"message": f"the fake {self._engine.recipe_id!r} could not parse the body: {error}"}
                    },
                )
        reply = self._engine.handle(request.method, request.url.path, body)
        if reply.json_body is None:
            return httpx.Response(reply.status)
        return httpx.Response(reply.status, json=reply.json_body, headers={"content-type": reply.content_type})

    async def aclose(self) -> None:
        """Nothing to close (the transport holds no pool); ``httpx.AsyncClient.aclose`` calls it."""
        return None


def fake_http_transport(engine: FakeEngine) -> Any:
    """The ``httpx`` transport that routes a product transport's calls through ``engine``.

    The returned object satisfies ``httpx.AsyncBaseTransport`` (one ``handle_async_request`` coroutine);
    it is typed loosely so the package needs no hard ``httpx`` dependency of its own (the product
    carries it).
    """
    return _FakeHTTPTransport(engine)


# ---------------------------------------------------------------------------
# The registry, by recipe id
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, FakeEngine] = {}


def register_fake_engine(engine: FakeEngine) -> None:
    """Register ``engine`` for its recipe id.

    Raises:
        ConformanceError: the object is not a :class:`FakeEngine` (no ``recipe_id``, no ``name``, no
            ``handle``), or a different fake is already registered for the recipe id (the same engine
            again is accepted, so re-importing a module changes nothing).
    """
    if not isinstance(engine, FakeEngine):  # runtime-checkable: recipe_id, name and handle must exist
        raise ConformanceError(
            f"{type(engine).__name__} is not a FakeEngine: a recipe-level fake carries recipe_id, name and "
            "handle(method, path, body) -> FakeReply"
        )
    registered = _REGISTRY.get(engine.recipe_id)
    if registered is not None and registered is not engine:
        raise ConformanceError(
            f"a fake engine for recipe {engine.recipe_id!r} is already registered "
            f"({type(registered).__name__}); unregister_fake_engine first"
        )
    _REGISTRY[engine.recipe_id] = engine


def unregister_fake_engine(recipe_id: str) -> None:
    """Forget the fake registered for ``recipe_id`` (a test's cleanup; an absent one is fine)."""
    _REGISTRY.pop(recipe_id, None)


def registered_fake_engines() -> tuple[str, ...]:
    """The recipe ids with a registered fake, sorted."""
    return tuple(sorted(_REGISTRY))


def fake_engine_for(recipe_id: str) -> FakeEngine:
    """The fake registered for ``recipe_id``.

    Raises:
        ConformanceError: none is -- the hint names the registry call. Model-level fakes do not exist
            yet (they are built from the GPU recordings later).
    """
    engine = _REGISTRY.get(recipe_id)
    if engine is None:
        raise ConformanceError(
            f"no fake engine is registered for recipe {recipe_id!r} (registered: "
            f"{', '.join(registered_fake_engines()) or 'none'}); register one with "
            "rcp_ndcg_test.fakes.register_fake_engine, or run the suite against a live engine"
        )
    return engine


# ---------------------------------------------------------------------------
# The packaged fixture recipe and its shipped test fake
# ---------------------------------------------------------------------------

FIXTURE_RECIPE_ID = "fake-embed"
"""The packaged fixture recipe's id: recipe, tokenizer and fixture cases ship as package data
(``rcp_ndcg_test/fixtures/``), so the runner's end-to-end exercise runs on CPU anywhere the package is
installed."""


def fixture_path(*parts: str) -> Path:
    """A packaged fixture file's path on disk (importlib.resources, so a wheel install finds it too)."""
    root = files("rcp_ndcg_test") / "fixtures"
    for part in parts:
        root = root / part
    with as_file(root) as path:
        return Path(path)


def package_tokenizer_path() -> Path:
    """The packaged fixture tokenizer's path (the shipped fake's ``/tokenize`` loads it)."""
    return fixture_path("tokenizer.json")


class FakeEmbedEngine:
    """The shipped test fake for the packaged fixture recipe (``fake-embed``, role ``embed``).

    A deterministic surrogate, not a verified emulator: every vector is drawn by
    :func:`rcp_ndcg.inference.fake.fake_uniform` from the text and its component index, so the same
    text always returns the same unit vector and different texts differ; ``/tokenize`` answers with the
    recipe's packaged tokenizer. A test that asserts numbers may only do so against inputs whose answer
    it recorded (the fixture cases carry theirs).

    Routes: ``GET .../models``, ``POST .../embeddings`` (the OpenAI shape the ``openai_embeddings``
    adapter sends) and ``POST /tokenize``.
    """

    recipe_id = FIXTURE_RECIPE_ID
    name = "shipped-embed-fake"

    def __init__(self, *, dim: int = 8, tokenizer: Path | None = None) -> None:
        """The fake's shape: ``dim`` is the vector width; ``tokenizer`` the ``tokenizer.json`` of the
        recipe (the packaged one when ``None``), loaded lazily for ``/tokenize``."""
        self.dim = dim
        self._tokenizer_path = tokenizer
        self._tokenizer: Any = None

    def handle(self, method: str, path: str, body: Any) -> FakeReply:
        """Answer the embed role's routes and ``/tokenize`` (see :class:`FakeEngine`)."""
        if method == "GET" and path.endswith("/models"):
            return FakeReply(
                status=200,
                json_body={
                    "object": "list",
                    "data": [{"id": FIXTURE_RECIPE_ID, "owned_by": self.name, "max_model_len": 512}],
                },
            )
        if method == "POST" and path.endswith("/embeddings"):
            return self._embeddings(body)
        if method == "POST" and path.endswith("/tokenize"):
            return self._tokenize(body)
        return FakeReply(
            status=404, json_body={"error": {"message": f"the fake {self.name!r} has no route for {method} {path}"}}
        )

    def _embeddings(self, body: Any) -> FakeReply:
        """One hash-seeded unit vector per input text, in the request's order."""
        texts = _input_texts(body)
        if texts is None:
            return _bad_request("the embed fake takes string inputs ('input': text or [texts])")
        return FakeReply(
            status=200,
            json_body={
                "object": "list",
                "model": FIXTURE_RECIPE_ID,
                "data": [
                    {"object": "embedding", "index": index, "embedding": self._unit_vector(text)}
                    for index, text in enumerate(texts)
                ],
                "usage": {"prompt_tokens": sum(_word_tokens(text) for text in texts), "total_tokens": 0},
            },
        )

    def _tokenize(self, body: Any) -> FakeReply:
        """The fixture tokenizer's ids for the prompt (the engine's tokenization ground truth)."""
        prompt = body.get("prompt") if isinstance(body, dict) else None
        if not isinstance(prompt, str):
            return _bad_request("the tokenize fake takes {'prompt': str} (it honours add_special_tokens)")
        add_special_tokens = bool(body.get("add_special_tokens", True)) if isinstance(body, dict) else True
        tokenizer = self._loaded_tokenizer()
        if tokenizer is None:
            return FakeReply(
                status=503,
                json_body={"error": {"message": "the tokenize fake needs the product's [hf] extra (tokenizers)"}},
            )
        ids = tokenizer.ids(prompt, add_special_tokens=add_special_tokens)
        return FakeReply(status=200, json_body={"tokens": ids, "count": len(ids), "max_model_len": 512})

    def _loaded_tokenizer(self) -> Any:
        """The fixture tokenizer, loaded once (through the product's loader)."""
        if self._tokenizer is None:
            from rcp_ndcg.data.tokenizer import load_tokenizer

            path = self._tokenizer_path if self._tokenizer_path is not None else package_tokenizer_path()
            self._tokenizer = load_tokenizer(str(path))
        return self._tokenizer

    def _unit_vector(self, text: str) -> list[float]:
        """One deterministic unit vector for ``text`` (hash-seeded, sent as float32 lists)."""
        import numpy as np

        raw = [fake_uniform(FIXTURE_RECIPE_ID, "embedding", text, index) * 2.0 - 1.0 for index in range(self.dim)]
        norm = math.sqrt(sum(value * value for value in raw)) or 1.0
        return [float(value) for value in np.asarray([value / norm for value in raw], dtype="<f4")]


def _input_texts(body: Any) -> list[str] | None:
    """The request's ``input`` texts: one string or a list of strings; anything else is refused."""
    if not isinstance(body, dict):
        return None
    value = body.get("input")
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return list(value)
    return None


def _word_tokens(text: str) -> int:
    """The usage accounting's count (whitespace words, at least one) -- the product's fake convention."""
    return max(1, len(text.split()))


def _bad_request(message: str) -> FakeReply:
    """vLLM's validation-refusal shape: an HTTP 400 with an error body the adapters map."""
    return FakeReply(status=400, json_body={"error": {"message": message}})


register_fake_engine(FakeEmbedEngine())
"""The shipped test fake registers itself with the package (``import rcp_ndcg_test.fakes``)."""
