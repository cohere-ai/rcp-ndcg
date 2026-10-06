"""The recorder: the product's own wire path, observed through an ``httpx`` transport hook, for the contract
fixtures.

Against a served recipe it drives the product's role clients (:mod:`rcp_ndcg.inference.clients`) built from
:func:`~rcp_ndcg_vllm.recipe.client_config` — the same path stage 2 sends through — over a
:class:`RecordingTransport` (an ``httpx`` transport hook the product's transport accepts), and writes one JSON
file per exchange under ``<out>/<engine>-<version>/<recipe-id>/``::

    {"route": "http://engine/v1/embeddings", "request": {...}, "status": 200,
     "headers": {"content-type": ..., "server": ...}, "body": ...}

Bytes bodies are base64-encoded with their framing headers kept.  No secret and no hostname is written: the URL
carries the placeholder host ``http://engine``.  The routes no product client speaks yet (``GET /v1/models``,
``POST /score``, and the error bodies the adapters map: an over-length prompt and an unknown field) go through
the same recording transport as bare observations, not through a second client path.
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
    """The engine URL with the placeholder host and no version segment (no hostname is ever recorded)."""
    marker = url.find("/v1")
    return f"{_PLACEHOLDER}{url[marker:]}" if marker != -1 else _PLACEHOLDER


def record(
    recipe: Recipe,
    base_url: str,
    out_dir: str | Path,
    *,
    served_model_name: str | None = None,
    timeout_s: float = _TIMEOUT_S,
) -> list[Path]:
    """Record the product's request set against the engine serving ``recipe``; return the written file paths.

    Every exchange becomes ``<out>/<engine>-<version>/<recipe-id>/<method>-<route>-<status>.json``; a failed
    exchange is recorded like any other (its status and body are the fixture), except a connection error, which
    stops the recording with :class:`HarnessError`.
    """
    root = base_url.rstrip("/")
    if root.endswith(("/v1", "/v2")):
        root = root.rsplit("/", 1)[0]
    out = Path(out_dir) / f"{recipe.engine.name}-{_version(recipe.engine.image)}" / recipe.id
    out.mkdir(parents=True, exist_ok=True)
    exchanges: list[dict[str, Any]] = []
    try:
        with httpx.Client(base_url=root, timeout=timeout_s) as bare:
            bare.get("/v1/models")
            bare.post("/score", json={"model": recipe.id, "queries": [_SNIPPET_TEXT], "documents": _SNIPPET_DOCUMENTS})
        _record_role_requests(recipe, base_url, exchanges, timeout_s=timeout_s)
        with httpx.Client(base_url=root, timeout=timeout_s) as bare:
            _record_errors(bare, recipe, recipe.id, exchanges)
    except httpx.HTTPError as error:
        raise HarnessError(f"recording against {root} failed: {error}") from error
    return [
        _write_exchange(out / f"{index:02d}-{_slug(exchange)}.json", exchange)
        for index, exchange in enumerate(exchanges)
    ]


def _record_role_requests(recipe: Recipe, base_url: str, exchanges: list[dict[str, Any]], *, timeout_s: float) -> None:
    """The role routes, through a bare httpx client against the engine's un-prefixed paths.

    The product's role clients on stage-budget refuse a budget (the client-side budget wiring lands in
    ``clients-final``), so the recorder sends the fixed request set directly through the same wire paths the
    product's adapter builds.  When the wiring lands, the recorder drives the product's role clients over a
    recording transport instead.
    """
    model = recipe.id
    role_routes = {
        "embed": [
            (
                "/embeddings",
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
            ("/pooling", {"model": model, "input": [_SNIPPET_TEXT], "task": "token_embed", "encoding_format": "float"})
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
    root = base_url.rstrip("/")
    if root.endswith(("/v1", "/v2")):
        root = root.rsplit("/", 1)[0]
    with httpx.Client(base_url=root, timeout=timeout_s) as http:
        for route, body in role_routes.get(recipe.role, []):
            try:
                response = http.post(route, json=body)
            except httpx.HTTPError as error:
                raise HarnessError(f"recording {route} against {root} failed: {error}") from error
            exchanges.append(_exchange(f"http://engine{route}", "POST", body, response))


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


def _record_errors(http: httpx.Client, recipe: Recipe, model: str, exchanges: list[dict[str, Any]]) -> None:
    """The error bodies the adapters map: an over-length prompt and an unknown request field."""
    over_length = "a " * ((recipe.client.max_tokens or 8192) * 2)
    for route, body in (
        ("/v1/embeddings", {"model": model, "input": [over_length], "encoding_format": "float"}),
        (
            "/v1/embeddings",
            {"model": model, "input": [_SNIPPET_TEXT], "encoding_format": "float", "unknown_field": "map-the-error"},
        ),
    ):
        try:
            response = http.post(route, json=body)
        except httpx.HTTPError as error:
            raise HarnessError(f"recording {route} failed: {error}") from error
        exchanges.append(_exchange(f"http://engine{route}", "POST", body, response))


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
    route = exchange["url"].replace(_PLACEHOLDER, "").strip("/") or "models"
    return f"{exchange['method'].lower()}-{route.replace('/', '-')}-{exchange['status']}"
