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

from ..errors import HarnessError
from ..recipe import Recipe, client_config

__all__ = ["RecordingTransport", "record"]

_PLACEHOLDER = "http://engine"
_TIMEOUT_S = 120.0
_SNIPPET_TEXT = "What is the capital of France?"
_SNIPPET_DOCUMENTS = ["Paris is the capital of France.", "Berlin is the capital of Germany."]


class RecordingTransport(httpx.AsyncBaseTransport):
    """An ``httpx`` transport that records every exchange and delegates to the real one.

    The observation seam: the product's :class:`~rcp_ndcg.inference.transport.Transport` accepts it as its
    ``httpx_transport``, so every request the product's clients send crosses here once — the product's wire
    path, not a second one.
    """

    def __init__(self, exchanges: list[dict[str, Any]], real: httpx.AsyncBaseTransport | None = None) -> None:
        super().__init__()
        self.exchanges = exchanges
        self._real = real or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Send through the real transport and record the exchange."""
        raw = request.read()
        try:
            body: Any = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            body = {"base64": base64.b64encode(raw).decode("ascii")}
        response = await self._real.handle_async_request(request)
        await response.aread()
        self.exchanges.append(
            {
                "url": _placeholder(str(request.url)),
                "method": request.method,
                "request_body": body,
                "status": response.status_code,
                "headers": {key: response.headers.get(key, "") for key in ("content-type", "server")},
                "response_bytes": base64.b64encode(response.content).decode("ascii"),
                "response_json": _response_json(response),
            }
        )
        return response


def _placeholder(url: str) -> str:
    """The engine URL with the placeholder host and no version segment (no hostname is ever recorded)."""
    marker = url.find("/v1")
    return f"{_PLACEHOLDER}{url[marker:]}" if marker != -1 else _PLACEHOLDER


def _response_json(response: httpx.Response) -> Any:
    """The response body as decoded JSON, or ``None`` when the body is not JSON."""
    if "json" not in response.headers.get("content-type", ""):
        return None
    try:
        return response.json()
    except ValueError:
        return None


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
            _record_errors(bare, recipe, recipe.id)
    except httpx.HTTPError as error:
        raise HarnessError(f"recording against {root} failed: {error}") from error
    return [
        _write_exchange(out / f"{index:02d}-{_slug(exchange)}.json", exchange)
        for index, exchange in enumerate(exchanges)
    ]


def _record_role_requests(recipe: Recipe, base_url: str, exchanges: list[dict[str, Any]], *, timeout_s: float) -> None:
    """The role routes, through the product's clients over a recording transport."""
    from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
    from rcp_ndcg.inference.transport import Transport

    exchanges.clear()
    hook = RecordingTransport(exchanges)
    classes: dict[str, type] = {"embed": EmbeddingEndpoint, "multi_vector": PoolingEndpoint, "rerank": RerankEndpoint}
    config = dict(client_config(recipe, base_url=base_url))
    config["max_tokens"] = None
    if recipe.client.template is not None:
        config["query_prompt"] = ""
        config["doc_prompt"] = ""
    endpoint = classes[recipe.role](**config)
    from rcp_ndcg_core.content import Content as WireContent

    from rcp_ndcg.inference.types import EncodeRole

    if recipe.role == "rerank":
        from rcp_ndcg.inference.clients import RerankClient

        client = RerankClient(endpoint, sender=Transport(endpoint, httpx_transport=hook))
        try:
            client.rerank(_SNIPPET_TEXT, _SNIPPET_DOCUMENTS)
        finally:
            client.close()
        return
    if recipe.role == "multi_vector":
        from rcp_ndcg.inference.clients import PoolingClient

        client = PoolingClient(endpoint, sender=Transport(endpoint, httpx_transport=hook))
        client.encode([WireContent.from_text(_SNIPPET_TEXT)], EncodeRole.DOCUMENT)
        return
    from rcp_ndcg.inference.clients import EmbeddingClient

    client = EmbeddingClient(endpoint, sender=Transport(endpoint, httpx_transport=hook))
    client.encode([WireContent.from_text(_SNIPPET_TEXT)], EncodeRole.DOCUMENT)


def _record_errors(http: httpx.Client, recipe: Recipe, model: str) -> None:
    """The error bodies the adapters map: an over-length prompt and an unknown request field."""
    over_length = "a " * ((recipe.client.max_tokens or 8192) * 2)
    http.post(
        "/v1/embeddings",
        json={"model": model, "input": [over_length], "encoding_format": "float"},
    )
    http.post(
        "/v1/embeddings",
        json={
            "model": model,
            "input": [_SNIPPET_TEXT],
            "encoding_format": "float",
            "unknown_field": "map-the-error",
        },
    )


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
