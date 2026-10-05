"""The recorder: one fixed request set per route, recorded for the contract fixtures.

Against a served recipe it sends a small, fixed request set — ``GET /v1/models``, ``POST /v1/embeddings`` (float
and base64), ``POST /pooling`` (float, base64, bytes), ``POST /rerank``, ``POST /score``, and the two error bodies
the rcp-ndcg adapters map (an over-length prompt and an unknown field) — and writes one JSON file per exchange
under ``<out>/<engine>-<version>/<recipe-id>/``::

    {"route": "/v1/embeddings", "request": {"url": "http://engine/v1/embeddings", ...},
     "status": 200, "headers": {"content-type": ..., "server": ...}, "body": ...}

Bytes bodies are base64-encoded with their framing headers kept.  No secret and no hostname is written: the URL
carries the placeholder host ``http://engine``, and nothing else in the file identifies the machine.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx

from .errors import HarnessError
from .recipe import Recipe, effective_embed_dtype

__all__ = ["record"]

_PLACEHOLDER = "http://engine"
_TIMEOUT_S = 120.0
_SNIPPET_TEXT = "What is the capital of France?"
_SNIPPET_DOCUMENTS = ["Paris is the capital of France.", "Berlin is the capital of Germany."]


def record(
    recipe: Recipe,
    base_url: str,
    out_dir: str | Path,
    *,
    served_model_name: str | None = None,
    timeout_s: float = _TIMEOUT_S,
) -> list[Path]:
    """Record the fixed request set against the engine serving ``recipe``; return the written file paths.

    Every exchange becomes ``<out>/<engine>-<version>/<recipe-id>/<method>-<route>-<variant>.json``; a failed
    exchange is recorded like any other (its status and body are the fixture), except a connection error, which
    stops the recording with :class:`HarnessError`.
    """
    model = served_model_name or recipe.id
    root = base_url.rstrip("/")
    if root.endswith(("/v1", "/v2")):
        root = root.rsplit("/", 1)[0]
    out = Path(out_dir) / f"{recipe.engine.name}-{_version(recipe.engine.image)}" / recipe.id
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with httpx.Client(base_url=root, timeout=timeout_s) as http:
        exchanges = _exchanges(recipe, model)
        for file_name, route, request, request_body in exchanges:
            url = f"{_PLACEHOLDER}{route}"
            recorded_request: dict[str, Any] = {"url": url, **request}
            try:
                if request_body is None:
                    response = http.get(route)
                else:
                    body = {"model": model, **request_body}
                    recorded_request["body"] = body
                    response = http.post(route, json=body)
            except httpx.HTTPError as error:
                raise HarnessError(f"recording {route} against {root} failed: {error}") from error
            written.append(_write_exchange(out / file_name, route, recorded_request, response))
    return written


def _exchanges(recipe: Recipe, model: str) -> list[tuple[str, str, dict[str, Any], dict[str, Any] | None]]:
    """The fixed request set: (file name, route, url-level request fields, JSON body or None for GET)."""
    over_length = "a " * (recipe.client.max_tokens * 2)
    pooling: dict[str, Any] = {
        "input": [_SNIPPET_TEXT],
        "task": "token_embed",
        "encoding_format": "float",
        "embed_dtype": effective_embed_dtype(recipe),
    }
    pooling_b64 = {**pooling, "encoding_format": "base64"}
    pooling_bytes = {**pooling, "encoding_format": "bytes"}
    return [
        ("get-v1-models.json", "/v1/models", {}, None),
        (
            "post-v1-embeddings-float.json",
            "/v1/embeddings",
            {},
            {"input": _texts_of(recipe), "encoding_format": "float"},
        ),  # fmt: skip
        (
            "post-v1-embeddings-base64.json",
            "/v1/embeddings",
            {},
            {"input": _texts_of(recipe), "encoding_format": "base64"},
        ),  # fmt: skip
        ("post-pooling-float.json", "/pooling", {}, pooling),
        ("post-pooling-base64.json", "/pooling", {}, pooling_b64),
        ("post-pooling-bytes.json", "/pooling", {}, pooling_bytes),
        (
            "post-rerank.json",
            "/rerank",
            {},
            {"query": _SNIPPET_TEXT, "documents": _SNIPPET_DOCUMENTS, "top_n": len(_SNIPPET_DOCUMENTS)},
        ),  # fmt: skip
        ("post-score.json", "/score", {}, {"queries": [_SNIPPET_TEXT], "documents": _SNIPPET_DOCUMENTS}),  # fmt: skip
        (
            "post-v1-embeddings-overlength.json",
            "/v1/embeddings",
            {},
            {"input": [over_length], "encoding_format": "float"},
        ),  # fmt: skip
        (
            "post-v1-embeddings-unknown-field.json",
            "/v1/embeddings",
            {},
            {"input": _texts_of(recipe), "encoding_format": "float", "unknown_field": "sent-to-map-the-error"},
        ),  # fmt: skip
    ]


def _texts_of(recipe: Recipe) -> list[str]:
    """The two recorded input texts, with the recipe's client-side prompts applied as the adapters would send."""
    return [recipe.client.query_prompt + _SNIPPET_TEXT, recipe.client.doc_prompt + _SNIPPET_DOCUMENTS[0]]


def _version(image: str) -> str:
    """The engine version from the image tag: ``vllm/vllm-openai:v0.31.0`` -> ``0.31.0``."""
    _, _, tag = image.rpartition(":")
    return tag.removeprefix("v") or "unknown"


def _write_exchange(path: Path, route: str, request: dict[str, Any], response: httpx.Response) -> Path:
    """One exchange file; a JSON body is kept decoded, a binary body base64 with its framing headers."""
    headers = {key: response.headers.get(key, "") for key in ("content-type", "server")}
    content_type = response.headers.get("content-type", "")
    if "json" in content_type or not response.content:
        try:
            body: Any = response.json()
        except ValueError:
            body = {"text": response.text}
    else:
        framing = {
            key: value
            for key, value in response.headers.items()
            if key.lower() in ("content-type", "content-length", "content-encoding")
        }
        body = {"base64": base64.b64encode(response.content).decode("ascii"), "framing_headers": framing}
    document = {"route": route, "request": request, "status": response.status_code, "headers": headers, "body": body}
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
