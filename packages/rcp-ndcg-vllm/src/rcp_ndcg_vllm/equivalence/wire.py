"""The client seam: the product's role clients, observed through one ``httpx`` transport hook.

Stage 1 and stage 2 both talk to the engine through the product's role clients
(:class:`~rcp_ndcg.inference.clients.EmbeddingClient`, :class:`~rcp_ndcg.inference.clients.PoolingClient`,
:class:`~rcp_ndcg.inference.clients.RerankClient`), built from :func:`~rcp_ndcg_vllm.recipe.client_config` with
the recipe's real budget: the client prompts, fits and settles exactly as the served path does. The harness
never re-derives a render, a cut or a settlement.

The observation seam is the product's own injection point: a
:class:`~rcp_ndcg_vllm.equivalence.wire.CapturingTransport` is handed to the client's
:class:`~rcp_ndcg.inference.transport.Transport` as its ``httpx_transport``, so every request the client
produces crosses it once -- raw request and reply bodies for the recorder, and the request texts stage 1
tokenizes (the engine is the tokenization truth, R29).  Without an engine the transport answers through the
product's own offline fake (``rcp_ndcg.inference.fake``): the requests are still the client's.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import json
from collections.abc import Iterator
from typing import Any

import httpx
from rcp_ndcg_core.content import TEXT_JOIN

from ..errors import HarnessError
from ..recipe import Recipe, client_config
from .fitting import resolved_tokenizer_spec

__all__ = ["CapturingTransport", "Capture", "patched_wire", "role_client"]

_WIRE_PATCH: contextvars.ContextVar[dict[str, dict[str, Any]] | None] = contextvars.ContextVar(
    "rcp_ndcg_vllm_wire_patch", default=None
)


@contextlib.contextmanager
def patched_wire(patch: dict[str, dict[str, Any]]) -> Iterator[None]:
    """Within the block, every JSON request a role client sends to a path ending in one of ``patch``'s keys
    carries that key's fields too -- the negative controls' wire breakages (an engine-side cut, a misread
    ``embed_dtype``; :mod:`rcp_ndcg_vllm.observe.controls`).  The capture records the patched bytes: what crossed
    the wire.  Never used outside the controls."""
    token = _WIRE_PATCH.set(patch)
    try:
        yield
    finally:
        _WIRE_PATCH.reset(token)


_CAPTURE_BASE = "fake://harness"
"""The ``fake://`` base URL a client probes through when the harness gave no engine: the product's offline
fake answers, so the client completes and its requests are captured."""


def _openai_base(base_url: str) -> str:
    """The OpenAI-style base URL the embed wire speaks: the root plus the version segment (``.../v1``)."""
    root = base_url.rstrip("/")
    return root if root.endswith(("/v1", "/v2")) else f"{root}/v1"


def _capture_base(recipe: Recipe) -> str:
    """The probe URL for ``recipe``: the offline fake's, with the declared vector width in its query."""
    dim = getattr(recipe.client, "dim", None)
    return f"{_CAPTURE_BASE}?dim={dim}" if dim else _CAPTURE_BASE


class CapturingTransport(httpx.AsyncBaseTransport):
    """An ``httpx`` transport that records every request and reply, then answers through ``delegate``.

    The product's injection point: :class:`~rcp_ndcg.inference.transport.Transport` takes it as its
    ``httpx_transport`` and wraps it in its own client (timeouts, pool limits, routing and retries stay the
    product's).  There is no second request path: the requests are the role client's, byte for byte.
    """

    def __init__(self, delegate: httpx.AsyncBaseTransport | httpx.BaseTransport) -> None:
        super().__init__()
        self.exchanges: list[dict[str, Any]] = []
        self._delegate = delegate
        self._patch = _WIRE_PATCH.get()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Send through the delegate and record the exchange (request body decoded, reply bytes kept)."""
        raw = request.read()
        try:
            body: Any = json.loads(raw) if raw else None
        except (ValueError, UnicodeDecodeError):
            body = {"base64": base64.b64encode(raw).decode("ascii")}
        patch = next(
            (fields for suffix, fields in (self._patch or {}).items() if request.url.path.endswith(suffix)), None
        )
        if patch and isinstance(body, dict):
            body = {**body, **patch}
            raw = json.dumps(body).encode("utf-8")
            headers = {key: value for key, value in request.headers.items() if key.lower() != "content-length"}
            request = httpx.Request(request.method, request.url, headers=headers, content=raw)
        async_handle = getattr(self._delegate, "handle_async_request", None)
        if async_handle is not None:
            response = await async_handle(request)
        else:
            # A sync-only delegate (an httpx.MockTransport): answered off the event loop's thread.
            sync_handle = getattr(self._delegate, "handle_request", None)
            if sync_handle is None:  # pragma: no cover - a delegate with neither handler
                raise HarnessError("the capture transport's delegate speaks neither the sync nor the async wire")
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, sync_handle, request)
        await response.aread()
        try:
            response_json: Any = response.json()
        except ValueError:
            response_json = None
        self.exchanges.append(
            {
                "url": str(request.url),
                "method": request.method,
                "request_bytes": base64.b64encode(raw).decode("ascii"),
                "request_body": body,
                "status": response.status_code,
                "headers": {
                    **{key: response.headers.get(key, "") for key in ("content-type", "server")},
                    # A /pooling ``bytes`` reply's framing: its vectors do not decode without it.
                    **({"metadata": response.headers["metadata"]} if "metadata" in response.headers else {}),
                },
                "response_bytes": base64.b64encode(response.content).decode("ascii"),
                "response_json": response_json,
            }
        )
        return response


class Capture:
    """The captured wire of one client session, with the request texts extracted per role.

    Attributes:
        role: The recipe's role: which wire shape the captured bodies speak.
        exchanges: Every request the client produced, in order (the recorder writes these as fixtures).
    """

    def __init__(self, recipe: Recipe) -> None:
        self.role = recipe.role
        self.exchanges: list[dict[str, Any]] = []

    def texts(self, exchange: dict[str, Any]) -> dict[str, Any]:
        """The texts one captured request carries: ``input`` for the embed roles, ``query``/``documents``
        for the rerank wire (the spans the engine assembles -- for a reranker, the settled query span).

        A ``messages`` body (the chat-style input: a media item's route, or ``request_shape: messages``) is
        read as vLLM v0.31.0 reads it: a list of messages is ONE conversation and a list of conversations a
        batch.  It yields one ``input`` per conversation -- its messages' text parts joined in order with
        ``TEXT_JOIN`` (``"\n"``, as the engine joins them): the content the client sent, which the engine's
        chat template frames -- ``conversations``, the conversations as sent (the served template renders
        them), ``add_generation_prompt``, the flag the engine renders them with (false when the body sends
        none, the chat routes' default), and ``media``: per conversation, the placeholders of its media
        parts in order (their part ``type``, e.g. ``image_url``), which ride beside the text and are never
        part of it.  A ``token_ids``
        body yields its id lists as sent.
        """
        body = exchange.get("request_body") or {}
        if self.role == "rerank":
            documents = body.get("documents", [])
            if isinstance(documents, str):
                documents = [documents]
            return {"query": body.get("query"), "documents": [str(document) for document in documents]}
        if "input" not in body and isinstance(body.get("messages"), list) and body["messages"]:
            messages = body["messages"]
            conversations = messages if all(isinstance(entry, list) for entry in messages) else [messages]
            texts: list[str] = []
            media: list[list[str]] = []
            for conversation in conversations:
                parts: list[dict[str, Any]] = []
                for message in conversation:
                    content = message.get("content") if isinstance(message, dict) else None
                    raw = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
                    # A bare string part is a text part (vLLM's chat_utils reads it so).
                    parts += [{"type": "text", "text": part} if isinstance(part, str) else part for part in raw]
                # The engine joins text parts with "\n" (vLLM's chat_utils): the product's TEXT_JOIN.
                texts.append(TEXT_JOIN.join(str(part.get("text", "")) for part in parts if part.get("type") == "text"))
                media.append([str(part.get("type")) for part in parts if part.get("type") != "text"])
            return {
                "input": texts,
                "media": media,
                "conversations": conversations,
                # The chat routes' default is false (vllm/entrypoints/pooling/base/protocol.py:230-237).
                "add_generation_prompt": bool(body.get("add_generation_prompt", False)),
            }
        inputs = body.get("input", body.get("texts"))
        return {"input": [inputs] if isinstance(inputs, str) else list(inputs or [])}


def role_client(
    recipe: Recipe,
    base_url: str | None,
    *,
    census: Any | None = None,
) -> tuple[Any, Capture]:
    """The product's role client for the recipe, sending through a :class:`CapturingTransport`.

    Inputs: the recipe and the engine's base URL (``None`` probes through the product's offline fake: no
    engine needed, the requests are still the client's).  Outputs: the client (its config is
    :func:`~rcp_ndcg_vllm.recipe.client_config`'s, with the recipe's real budget -- prompts, text budget,
    ``query_max_tokens``, the reranker's settle-once) and the :class:`Capture` whose ``exchanges`` carry every
    request and reply, in order.
    """
    from rcp_ndcg.inference.clients import EmbeddingClient, PoolingClient, RerankClient
    from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
    from rcp_ndcg.inference.transport import Transport

    url = base_url if base_url else _capture_base(recipe)
    if base_url and recipe.role == "embed":
        url = _openai_base(url)
    data = client_config(recipe, base_url=url)
    data["tokenizer"] = resolved_tokenizer_spec(recipe)
    classes = {"embed": EmbeddingEndpoint, "multi_vector": PoolingEndpoint, "rerank": RerankEndpoint}
    config = classes[recipe.role](**data)
    if base_url:
        delegate: httpx.AsyncBaseTransport | httpx.BaseTransport = httpx.AsyncHTTPTransport()
    else:
        from rcp_ndcg.inference.fake import FAKE_SCHEME, fake_transport

        delegate = fake_transport(
            url.removeprefix(FAKE_SCHEME), model=str(data.get("model") or recipe.id), tokenizer=data["tokenizer"]
        )
    capturing = CapturingTransport(delegate)
    sender = Transport(config, httpx_transport=capturing)
    clients = {"embed": EmbeddingClient, "multi_vector": PoolingClient, "rerank": RerankClient}
    client = clients[recipe.role](config, sender=sender, census=census)
    capture = Capture(recipe)
    capture.exchanges = capturing.exchanges
    return client, capture
