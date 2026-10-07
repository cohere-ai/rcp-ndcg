"""The recorder: the product's role clients, observed through an ``httpx`` transport hook, for the contract
fixtures.

Against a served recipe it drives the product's role client
(:class:`~rcp_ndcg.inference.clients.EmbeddingClient`, ``PoolingClient`` or ``RerankClient``) built from
:func:`~rcp_ndcg_vllm.recipe.client_config` with the recipe's real budget -- the client prompts, fits and
settles, the adapter renders, the transport sends -- and records the engine behaviour the adapters must map:
the provenance ``GET /v1/models``, and on the role route an over-length prompt (measured against the engine's
own ``serve.max_model_len`` cap, sent bare: the client would cut it before the engine saw it) and an unknown
request field.  Written under ``<out>/<engine>-<version>/<recipe-id>/``::

    {"route": "http://engine/v1/embeddings", "request": {"url": ..., "body": {...}}, "status": 200,
     "headers": {"content-type": ..., "server": ...}, "body": ...}

Bytes bodies are base64-encoded with their framing headers kept.  No secret and no hostname is written: the URL
carries the placeholder host ``http://engine``.  The role route's request is the product's, byte for byte: it
crosses the capturing transport the client's own transport wraps (the product's injection point), never a
hand-built copy of the adapter's body.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx
from rcp_ndcg_vllm.recipe import Recipe

from rcp_ndcg.data.templates import TemplateSpec
from rcp_ndcg_test.errors import HarnessError

from .equivalence.wire import role_client

__all__ = ["record"]

_PLACEHOLDER = "http://engine"
_TIMEOUT_S = 120.0
_SNIPPET_TEXT = "What is the capital of France?"
_SNIPPET_DOCUMENTS = ["Paris is the capital of France.", "Berlin is the capital of Germany."]


def _engine_root(base_url: str) -> str:
    """The engine's root: a base URL ending in the version segment loses it (the routes are root-relative)."""
    root = base_url.rstrip("/")
    if root.endswith(("/v1", "/v2")):
        root = root.rsplit("/", 1)[0]
    return root


def record(
    recipe: Recipe,
    base_url: str,
    out_dir: str | Path,
    *,
    timeout_s: float = _TIMEOUT_S,
) -> list[Path]:
    """Record the engine's request set for ``recipe``; return the written file paths.

    Every exchange becomes ``<out>/<engine>-<version>/<recipe-id>/<method>-<route>-<status>.json``; a failed
    exchange is recorded like any other (its status and body are the fixture), except a connection error, which
    stops the recording with :class:`HarnessError`.  The set: ``GET /v1/models``, the role route (the product's
    request, through the role client), the over-length 400 and the unknown-field 400 on the role route.
    """
    root = _engine_root(base_url)
    out = Path(out_dir) / f"{recipe.engine.name}-{_version(recipe.engine.image)}" / recipe.id
    out.mkdir(parents=True, exist_ok=True)
    exchanges: list[dict[str, Any]] = []
    with httpx.Client(base_url=root, timeout=timeout_s) as bare:
        _record_one(bare, exchanges, "GET", "/v1/models", None)
        _record_role_request(recipe, base_url, exchanges)
        _record_errors(bare, recipe, exchanges)
    return [
        _write_exchange(out / f"{index:02d}-{_slug(exchange)}.json", exchange)
        for index, exchange in enumerate(exchanges)
    ]


def _record_one(http: httpx.Client, exchanges: list[dict[str, Any]], method: str, route: str | None, body: Any) -> None:
    """One bare exchange against ``route``; a connection error stops the recording, a status does not.

    The bare client records the engine-behaviour probes only (the provenance route and the error bodies): the
    role route's request comes from the product's role client, never from a hand-built copy.
    """
    if route is None:
        return
    try:
        response = http.request(method, route, json=body)
    except httpx.HTTPError as error:
        raise HarnessError(f"recording {method} {route} failed: {error}") from error
    exchanges.append(_exchange(f"{_PLACEHOLDER}{route}", method, body, response))


def _record_role_request(recipe: Recipe, base_url: str, exchanges: list[dict[str, Any]]) -> None:
    """The role route through the product's role client: the captured exchange is the product's request."""
    client, capture = role_client(recipe, base_url)
    if recipe.role == "rerank":
        client.rerank(_SNIPPET_TEXT, _SNIPPET_DOCUMENTS, instruction=_default_instruction(recipe))
    else:
        from rcp_ndcg_core.content import Content

        from rcp_ndcg.inference.types import EncodeRole

        role = EncodeRole.QUERY if "query" in _declared(recipe) else EncodeRole.DOCUMENT
        text = _SNIPPET_TEXT if role is EncodeRole.QUERY else _SNIPPET_DOCUMENTS[0]
        client.encode([Content.from_text(text)], role)
    for exchange in capture.exchanges:
        exchange["url"] = _placeholder(exchange["url"])
        exchanges.append(exchange)


def _default_instruction(recipe: Recipe) -> str | None:
    """The recipe's default instruction, sent as the request field when the mode sends one."""
    instruction = recipe.client.get("default_instruction")
    mode = recipe.client.get("instruction")
    return instruction if (mode == "field" and instruction) else None


def _declared(recipe: Recipe) -> list[str]:
    """The recipe's declared shapes (the side the recorder probes follows them)."""
    template = TemplateSpec.model_validate(recipe.client.get("template"))
    return [str(shape) for shape in template.shapes()] if template is not None else ["document"]


def _record_errors(http: httpx.Client, recipe: Recipe, exchanges: list[dict[str, Any]]) -> None:
    """The error bodies the adapters map, on the recipe's role route: an over-length prompt and an unknown field.

    The over-length input is measured against the engine's own cap, ``serve.max_model_len`` -- the number the
    engine enforces -- so the probe genuinely crosses it and records the engine's 400.  The probes go bare (a
    deliberate refusal shape): the product's clients cut before the engine would refuse, and these fixtures
    document exactly what the adapters must map.
    """
    over_length = "a " * (recipe.serve.max_model_len * 2)
    route, over_length_body = _role_request(recipe.role, recipe.id, over_length)
    _record_one(http, exchanges, "POST", route, over_length_body)
    _, in_budget_body = _role_request(recipe.role, recipe.id, _SNIPPET_TEXT)
    _record_one(http, exchanges, "POST", route, {**in_budget_body, "unknown_field": "map-the-error"})


def _role_request(role: str, model: str, text: str) -> tuple[str, dict[str, Any]]:
    """The role route and its request with ``text`` as the input (the route the role's adapter speaks)."""
    routes: dict[str, tuple[str, dict[str, Any]]] = {
        "embed": (
            "/v1/embeddings",
            {"model": model, "input": [text], "encoding_format": "float"},
        ),
        "multi_vector": (
            "/pooling",
            {"model": model, "input": [text], "task": "token_embed", "encoding_format": "float"},
        ),
        "rerank": ("/rerank", {"model": model, "query": text, "documents": _SNIPPET_DOCUMENTS}),
    }
    return routes[role]


def _exchange(url: str, method: str, request_body: Any, response: httpx.Response) -> dict[str, Any]:
    """One recorded exchange from the raw httpx response."""
    response.read()
    raw = response.content
    try:
        response_json: Any = response.json()
    except ValueError:
        response_json = None
    return {
        "url": _placeholder(url),
        "method": method,
        "request_body": request_body,
        "status": response.status_code,
        "headers": {key: response.headers.get(key, "") for key in ("content-type", "server")},
        "response_bytes": base64.b64encode(raw).decode("ascii"),
        "response_json": response_json,
    }


def _placeholder(url: str) -> str:
    """The engine URL with the placeholder host (no hostname is ever recorded); the path is kept."""
    for scheme in ("https://", "http://"):
        if url.startswith(scheme):
            return _PLACEHOLDER + url[url.find("/", len(scheme)) :]
    return _PLACEHOLDER + "/" + url.removeprefix("/")


def _version(image: str) -> str:
    """The engine version from the image tag: ``vllm/vllm-openai:v0.31.0`` -> ``0.31.0``."""
    _, _, tag = image.rpartition(":")
    return tag.removeprefix("v") or "unknown"


def _write_exchange(path: Path, exchange: dict[str, Any]) -> Path:
    """One exchange file; a JSON body is kept decoded, a binary body base64 with its framing headers."""
    body: Any
    if exchange["response_json"] is not None:
        body = exchange["response_json"]
    else:
        framing = {
            key: value
            for key, value in exchange["headers"].items()
            if key.lower() in ("content-type", "content-length")
        }
        body = {"base64": exchange["response_bytes"], "framing_headers": framing}
    document = {
        "route": exchange["url"],
        "request": {"url": exchange["url"], "body": exchange["request_body"]},
        "status": exchange["status"],
        "headers": exchange["headers"],
        "body": body,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _slug(exchange: dict[str, Any]) -> str:
    """The exchange's file stem: the method and the URL path (``post-v1-embeddings-400``)."""
    path = exchange["url"].removeprefix(_PLACEHOLDER).strip("/")
    return f"{exchange['method'].lower()}-{path.replace('/', '-')}-{exchange['status']}"
