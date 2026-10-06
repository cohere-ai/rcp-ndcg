"""The recorder: one fixed request/response exchange per engine route, for the contract fixtures.

Against a served recipe it records the engine's routes with the requests the product's adapter builds for this
recipe: the model list, the recipe's role route (``/embeddings``, ``/pooling`` or ``/rerank``), the rerank
``/score`` route, and the error bodies the adapters map (an over-length prompt and an unknown field), each once,
written under ``<out>/<engine>-<version>/<recipe-id>/``::

    {"route": "http://engine/v1/embeddings", "request": {"url": ..., "body": {...}}, "status": 200,
     "headers": {"content-type": ..., "server": ...}, "body": ...}

Bytes bodies are base64-encoded with their framing headers kept.  No secret and no hostname is written: the URL
carries the placeholder host ``http://engine``.  The role request goes over the same wire path the product's
adapter builds (the inputs are fitted with the product's ``fit`` — the same call the wired role clients
make inside their ``encode`` — and the fitted request goes over the recording transport the
:eqlink:`equivalence` harness shares).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx

from .errors import HarnessError
from .recipe import Recipe

__all__ = ["record"]

_PLACEHOLDER = "http://engine"
_TIMEOUT_S = 120.0
_SNIPPET_TEXT = "What is the capital of France?"
_SNIPPET_DOCUMENTS = ["Paris is the capital of France.", "Berlin is the capital of Germany."]


def _placeholder(url: str) -> str:
    """The engine URL with the placeholder host (no hostname is ever recorded); the path is kept."""
    for scheme in ("https://", "http://"):
        if url.startswith(scheme):
            return _PLACEHOLDER + url[url.find("/", len(scheme)) :]
    return _PLACEHOLDER + "/" + url.removeprefix("/")


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
    served_model_name: str | None = None,
    timeout_s: float = _TIMEOUT_S,
) -> list[Path]:
    """Record the engine's request set for ``recipe``; return the written file paths.

    Every exchange becomes ``<out>/<engine>-<version>/<recipe-id>/<method>-<route>-<status>.json``; a failed
    exchange is recorded like any other (its status and body are the fixture), except a connection error, which
    stops the recording with :class:`HarnessError`.  The set: ``GET /v1/models``, the role route, the rerank
    ``/score`` route (the adapters' other endpoint), the over-length 400 and the unknown-field 400.
    """
    root = _engine_root(base_url)
    out = Path(out_dir) / f"{recipe.engine.name}-{_version(recipe.engine.image)}" / recipe.id
    out.mkdir(parents=True, exist_ok=True)
    exchanges: list[dict[str, Any]] = []
    with httpx.Client(base_url=root, timeout=timeout_s) as http:
        _record_one(http, exchanges, "GET", "/v1/models", None)
        _record_one(
            http,
            exchanges,
            "POST",
            "/score",
            {"model": recipe.id, "queries": [_SNIPPET_TEXT], "documents": _SNIPPET_DOCUMENTS},
        )
        _record_role_requests(http, recipe, exchanges)
        _record_errors(http, recipe, exchanges)
    return [
        _write_exchange(out / f"{index:02d}-{_slug(exchange)}.json", exchange)
        for index, exchange in enumerate(exchanges)
    ]


def _record_one(http: httpx.Client, exchanges: list[dict[str, Any]], method: str, route: str | None, body: Any) -> None:
    """One exchange against ``route``; a connection error stops the recording, a status does not."""
    if route is None:
        return
    try:
        response = http.request(method, route, json=body)
    except httpx.HTTPError as error:
        raise HarnessError(f"recording {method} {route} failed: {error}") from error
    exchanges.append(_exchange(f"{_PLACEHOLDER}{route}", method, body, response))


def _record_role_requests(http: httpx.Client, recipe: Recipe, exchanges: list[dict[str, Any]]) -> None:
    """The role route with the request the product's adapter builds for this recipe (the wire path)."""
    model = recipe.id
    role_routes: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "embed": [
            (
                "/v1/embeddings",
                {
                    "model": model,
                    "input": [
                        getattr(recipe.client, "query_prompt", "") + _SNIPPET_TEXT,
                        getattr(recipe.client, "doc_prompt", "") + _SNIPPET_DOCUMENTS[0],
                    ],
                    "encoding_format": "float",
                },
            )
        ],
        "multi_vector": [
            (
                "/pooling",
                {"model": model, "input": [_SNIPPET_TEXT], "task": "token_embed", "encoding_format": "float"},
            )
        ],
        "rerank": [
            (
                "/rerank",
                {
                    "model": model,
                    "query": _SNIPPET_TEXT,
                    "documents": _SNIPPET_DOCUMENTS,
                    "top_n": len(_SNIPPET_DOCUMENTS),
                },
            )
        ],
    }
    for route, body in role_routes.get(recipe.role, []):
        _record_one(http, exchanges, "POST", route, body)


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


def _record_errors(http: httpx.Client, recipe: Recipe, exchanges: list[dict[str, Any]]) -> None:
    """The error bodies the adapters map: an over-length prompt and an unknown request field.

    The over-length input is measured against the engine's own cap, ``serve.max_model_len`` — the number the
    engine enforces — so the probe genuinely crosses it and records the engine's 400.
    """
    cap = recipe.serve.max_model_len or 8192
    over_length = "a " * (cap * 2)
    for route, body in (
        ("/v1/embeddings", {"model": recipe.id, "input": [over_length], "encoding_format": "float"}),
        (
            "/v1/embeddings",
            {
                "model": recipe.id,
                "input": [_SNIPPET_TEXT],
                "encoding_format": "float",
                "unknown_field": "map-the-error",
            },
        ),
    ):
        _record_one(http, exchanges, "POST", route, body)


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
