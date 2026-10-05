"""The offline fakes: one wire per role behind ``fake://`` URLs, so offline tests run the real transport.

They sit *below* the transport: a ``fake://`` base URL makes :class:`~rcp_ndcg.inference.transport.Transport`
send through an in-process ``httpx.MockTransport`` built here, and the transport above it is the real one, so
routing, retries, parking and usage run in every offline test. The fakes are deterministic: every draw is a
hash (:func:`fake_uniform`) of the endpoint's seed and the item's text, the same on every machine and in every
call order, and the reranker scores each document by the same hidden ability (:func:`hidden_ability`) the fake
judge reads its rubric passes and tournaments out of -- so a tiny run's rerank, judge and calibration agree.

The routes (``GET /models``, ``POST /embeddings``, ``POST /pooling``, ``POST /rerank``) speak each role's wire;
the chat completions of a later role's fake, and any third-party route, register with :func:`register_fake_route`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import numpy as np

from rcp_ndcg.errors import ConfigError

#: The URL scheme that selects a fake: ``fake://seed/<n>`` for the judge (``JudgeConfig.fake``), and the
#: per-role equivalents for the encoder, the pooler and the reranker.
FAKE_SCHEME = "fake://"

#: The dimension of the fake vectors when the URL's query gives none.
DEFAULT_DIM = 64


def fake_uniform(*parts: object) -> float:
    """A deterministic draw in [0, 1) from *parts* (the same on every machine and in every call order).

    The one draw every fake reads (the fake judge's ``llm._fake`` imports it under its old name), so a tiny
    run's rerank, judge and calibration agree.
    """
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def hidden_ability(seed: int) -> Callable[[str], float]:
    """A standard-normal ability per document text, fixed by ``seed``: how good the text is, in logits.

    The one hidden truth of the fakes: the fake judge's rubric passes, its tournament scores and the fake
    reranker's relevance scores all read it, so their orders agree on the same documents.
    """

    def ability(text: str) -> float:
        u1, u2 = max(fake_uniform(seed, "ability", text, 1), 1e-12), fake_uniform(seed, "ability", text, 2)
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    return ability


@dataclass(frozen=True)
class FakeEndpoint:
    """What a fake route handler knows about the endpoint it answers for.

    Attributes:
        url: The endpoint's full ``fake://`` URL.
        seed: The seed every draw reads: the URL's numeric path tail (``fake://seed/3``), else 0.
        model: The endpoint's served model name (the fake reports it on ``GET /models``).
        dim: The dimension of the fake vectors, from the URL's ``dim`` query (default
            :data:`DEFAULT_DIM`).
    """

    url: str
    seed: int
    model: str
    dim: int


#: What answers one fake route: the request (the handler decodes its JSON body) and the endpoint.
FakeRouteHandler = Callable[[httpx.Request, FakeEndpoint], httpx.Response]

_ROUTES: dict[tuple[str, str], FakeRouteHandler] = {}
"""The registered routes, keyed by ``(method, path)``; the built-in roles answer what nobody registered."""
_ROUTES_LOCK = threading.Lock()


def register_fake_route(method: str, path: str, handler: FakeRouteHandler) -> None:
    """Register ``handler`` as the fake's answer for ``method path`` (e.g. the chat completions of the judge's
    fake, or a third party's route).

    Args:
        method: The HTTP method the handler answers (``"POST"``).
        path: The call path the handler answers (``"/chat/completions"``).
        handler: The answer, called with the request and the :class:`FakeEndpoint`.

    Raises:
        ConfigError: A different handler is already registered for that route (the same handler again is
            accepted, so importing a module twice changes nothing).
    """
    key = (method.upper(), path if path.startswith("/") else f"/{path}")
    with _ROUTES_LOCK:
        registered = _ROUTES.get(key)
        if registered is None:
            _ROUTES[key] = handler
        elif registered is not handler:
            raise ConfigError(f"a fake route for {method} {path} is already registered")


def fake_transport(url: str, *, model: str) -> httpx.MockTransport:
    """The in-process transport of a ``fake://`` endpoint: one handler speaking each role's wire.

    The routes (each reads the endpoint's seed from the URL, and its vector dimension from the ``dim`` query,
    default 64): ``GET /models`` names the endpoint's model; ``POST /embeddings`` (OpenAI shape) answers
    deterministic hash-seeded unit vectors, cut to the request's ``dimensions`` when it carries one;
    ``POST /pooling`` (``task: token_embed``) answers ragged per-token vectors, as floats or base64-packed in
    the request's ``embed_dtype`` (default ``float16``); ``POST /rerank`` (Cohere shape) scores each document
    with its hidden ability. A route registered with :func:`register_fake_route` answers first; anything else
    is a 404 naming the route.

    Args:
        url: The endpoint's ``fake://`` URL.
        model: The endpoint's served model name.

    Returns:
        The mock transport a :class:`~rcp_ndcg.inference.transport.Transport` sends through.
    """
    endpoint = _fake_endpoint(url, model=model)
    return httpx.MockTransport(lambda request: _handle(request, endpoint))


def _fake_endpoint(url: str, *, model: str) -> FakeEndpoint:
    """The :class:`FakeEndpoint` of a ``fake://`` URL: the seed is its numeric path tail, ``dim`` its query."""
    base, _, query = url.partition("?")
    tail = base.removeprefix(FAKE_SCHEME).rstrip("/").rsplit("/", 1)[-1]
    try:
        seed = int(tail) if tail.lstrip("-").isdigit() else 0
    except ValueError:  # e.g. "--5": not a number after all; the default seed applies
        seed = 0
    dim = DEFAULT_DIM
    for pair in query.split("&") if query else ():
        name, _, value = pair.partition("=")
        if name == "dim" and value.isdigit():
            dim = int(value)
    return FakeEndpoint(url=url, seed=seed, model=model, dim=dim)


def _handle(request: httpx.Request, endpoint: FakeEndpoint) -> httpx.Response:
    """One fake request: a registered route first, then the built-in role routes, else a 404."""
    path = request.url.path
    with _ROUTES_LOCK:
        routes = dict(_ROUTES)
    for (method, route), handler in routes.items():
        if request.method.upper() == method and path.endswith(route):
            return handler(request, endpoint)
    body = _json_body(request)
    if request.method == "GET" and path.endswith("/models"):
        return _models(endpoint)
    if request.method == "POST" and path.endswith("/embeddings"):
        return _embeddings(endpoint, body)
    if request.method == "POST" and path.endswith("/pooling"):
        return _pooling(endpoint, body)
    if request.method == "POST" and path.endswith("/rerank"):
        return _rerank(endpoint, body)
    return httpx.Response(
        404, json={"error": {"message": f"the fake endpoint has no route for {request.method} {path}"}}
    )


def _json_body(request: httpx.Request) -> dict:
    """The request's JSON body, or ``{}`` when it has none or it does not parse."""
    try:
        body = json.loads(request.content)
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def _models(endpoint: FakeEndpoint) -> httpx.Response:
    """``GET /models``: the fake lists the endpoint's model, as a served engine would."""
    return httpx.Response(
        200, json={"object": "list", "data": [{"id": endpoint.model, "owned_by": "fake", "max_model_len": 32768}]}
    )


def _items(body: dict) -> list[str]:
    """The texts of a request's ``input`` (one string, or a list; non-string items by their text or as str)."""
    items = body.get("input")
    if isinstance(items, str):
        return [items]
    return [_text(item) for item in items] if isinstance(items, list) else []


def _text(item: object) -> str:
    """The text of one input item: a string itself, a part list by its text parts, else its ``str``."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict) and isinstance(item.get("text"), str):
        return item["text"]
    if isinstance(item, list):
        return " ".join(part for part in (_text(element) for element in item) if part)
    return str(item)


def _unit_vector(*parts: object, dim: int) -> list[float]:
    """A deterministic unit vector: one hash-seeded draw per component, L2-normalised."""
    raw = [fake_uniform(*parts, i) * 2.0 - 1.0 for i in range(dim)]
    norm = math.sqrt(sum(value * value for value in raw)) or 1.0
    return [value / norm for value in raw]


def _embeddings(endpoint: FakeEndpoint, body: dict) -> httpx.Response:
    """``POST /embeddings`` (OpenAI shape): one hash-seeded unit vector per input item.

    The dimension is the URL's (``?dim=``), cut to the request's Matryoshka ``dimensions`` when it carries one
    (never more components than the URL's dimension provides; the cut vector is normalised again).
    """
    texts = _items(body)
    dimensions = body.get("dimensions")
    dim = min(int(dimensions), endpoint.dim) if isinstance(dimensions, int) and dimensions else endpoint.dim
    return httpx.Response(
        200,
        json={
            "object": "list",
            "model": endpoint.model,
            "data": [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": _unit_vector(endpoint.seed, "embedding", text, dim=dim),
                }
                for index, text in enumerate(texts)
            ],
            "usage": {"prompt_tokens": sum(len(text.split()) for text in texts), "total_tokens": 0},
        },
    )


def _tokens(text: str) -> int:
    """The fake's token count of a text (its whitespace words, at least one)."""
    return max(1, len(text.split()))


def _pooling(endpoint: FakeEndpoint, body: dict) -> httpx.Response:
    """``POST /pooling`` (vLLM, ``task: token_embed``): ragged per-token vectors, one slice per item.

    The vectors cross as nested floats, or as one base64 string per item when ``encoding_format: base64`` --
    packed in ``embed_dtype`` (``float16`` by default, the owner's transfer precision), row-major, so the
    adapter reshapes by the item's ``prompt_token_ids`` count.
    """
    texts = _items(body)
    encoding = body.get("encoding_format", "float")
    dtype = np.dtype(body.get("embed_dtype") or "float16")
    data = []
    for index, text in enumerate(texts):
        count = _tokens(text)
        matrix = np.asarray(
            [_unit_vector(endpoint.seed, "token", text, token, dim=endpoint.dim) for token in range(count)],
            dtype=np.float32,
        ).reshape(count, endpoint.dim)
        embedding: str | list[list[float]]
        if encoding == "base64":
            embedding = base64.b64encode(matrix.astype(dtype).tobytes()).decode("ascii")
        else:
            embedding = matrix.tolist()
        data.append(
            {
                "index": index,
                "embedding": embedding,
                "prompt_token_ids": [
                    int(fake_uniform(endpoint.seed, "token_id", text, token) * 100_000) for token in range(count)
                ],
            }
        )
    return httpx.Response(
        200,
        json={
            "object": "list",
            "model": endpoint.model,
            "data": data,
            "usage": {"prompt_tokens": sum(_tokens(text) for text in texts), "total_tokens": 0},
        },
    )


def _documents(body: dict) -> list[str]:
    """The texts a rerank request's ``documents`` name: a string itself, a mapping by its ``text`` (or its
    ``id``), anything else by its ``str``."""
    documents = body.get("documents")
    return [_text(document) for document in documents] if isinstance(documents, list) else []


def _rerank(endpoint: FakeEndpoint, body: dict) -> httpx.Response:
    """``POST /rerank`` (Cohere shape): each document scored by its hidden ability, best first.

    The score of a document is :func:`hidden_ability` of its text (its ``id`` when it carries one) -- the same
    draw the fake judge reads, so the tiny run's rerank and judge agree. ``top_n`` keeps the best entries.
    """
    ability = hidden_ability(endpoint.seed)
    documents = _documents(body)
    scored = sorted(
        ({"index": index, "relevance_score": ability(text)} for index, text in enumerate(documents)),
        key=lambda entry: -entry["relevance_score"],
    )
    top_n = body.get("top_n")
    if isinstance(top_n, int) and 0 <= top_n < len(scored):
        scored = scored[:top_n]
    return httpx.Response(200, json={"id": "fake", "model": endpoint.model, "results": scored})


__all__ = [
    "DEFAULT_DIM",
    "FAKE_SCHEME",
    "FakeEndpoint",
    "FakeRouteHandler",
    "fake_transport",
    "fake_uniform",
    "hidden_ability",
    "register_fake_route",
]
