"""The offline pooling fake shared by the adapter and client tests: a ``/pooling`` look-alike over
``httpx.MockTransport``, plus a plain in-process sender for concurrency and bridge behaviour."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import numpy as np
from rcp_ndcg_core.content import Content

from rcp_ndcg.inference.types import Call, EncodeRole, PoolRequest, Reply

_FRAME_DTYPES = {"float16": "<f2", "float32": "<f4"}


class PoolingServer:
    """A ``/pooling`` look-alike: one token matrix per input text, framed the way the request asked.

    The usage line is the real one: a ``token_embed`` answer has one vector per prompt token, so
    ``prompt_tokens`` is the total row count of the batch. Texts the vector table does not name (a client's
    role-prefixed render) decode to ``default``; a media message decodes to ``media_vector``.
    """

    MEDIA_KEY = "<media>"

    def __init__(
        self,
        vectors: dict[str, np.ndarray],
        *,
        default: np.ndarray | None = None,
        media_vector: np.ndarray | None = None,
        encoding: str = "base64",
        usage: bool = True,
        reverse: bool = False,
    ) -> None:
        self.vectors = vectors
        self.default = default
        self.media_vector = media_vector
        self.encoding = encoding
        self.usage = usage
        self.reverse = reverse

    def _array_of(self, key: str) -> np.ndarray:
        if key == self.MEDIA_KEY:
            if self.media_vector is None:
                raise KeyError("the request carries media and the server fake has no media_vector")
            return self.media_vector
        if key in self.vectors:
            return self.vectors[key]
        if self.default is None:
            raise KeyError(f"the server fake has no vectors for {key!r} and no default")
        return self.default

    def response_body(self, body: dict[str, Any]) -> dict[str, Any]:
        """The ``PoolingResponse`` the inputs of ``body`` decode to, in the request's own encoding."""
        if "input" in body:
            # A token-ids input (3) keys by its stringified ids, as the client sent them.
            keys = [key if isinstance(key, str) else str(key) for key in body["input"]]
        else:
            keys = []
            for message in body.get("messages", []):
                if message.get("role") == "system":
                    continue  # the declared media head frames the item; it is not an input
                content = message["content"]
                if isinstance(content, str):
                    keys.append(content or self.MEDIA_KEY)
                    continue
                texts = [part["text"] for part in content if part["type"] == "text"]
                keys.append(texts[0] if texts else self.MEDIA_KEY)
        rows = sum(len(np.asarray(self._array_of(key))) for key in keys)
        dtype = _FRAME_DTYPES[body["embed_dtype"]]
        items = []
        for index, key in enumerate(keys):
            array = np.asarray(self._array_of(key), dtype=dtype)
            data: Any = (
                base64.b64encode(array.tobytes()).decode("ascii")
                if self.encoding == "base64"
                else array.astype(np.float32).tolist()
            )
            items.append({"index": index, "object": "pooling", "data": data})
        if self.reverse:
            items.reverse()
        response: dict[str, Any] = {"object": "list", "model": body.get("model"), "data": items}
        if self.usage:
            response["usage"] = {"prompt_tokens": rows, "total_tokens": rows}
        return response


class RecordingSender:
    """The sender the tests inject: real ``httpx`` requests over ``httpx.MockTransport``, canned replies.

    ``send`` mirrors the transport's contract (JSON bodies decoded, everything else passed as bytes), so
    ``interpret`` reads what a server actually sends.
    """

    def __init__(self, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self._handler = handler

    async def send(self, calls: Sequence[Call]) -> list[Reply]:
        replies: list[Reply] = []
        for call in calls:
            request = httpx.Request(call.method, f"http://pool.test{call.path}", json=call.json)
            self.requests.append(request)
            response = self._handler(request)
            if response.headers.get("content-type", "").startswith("application/json"):
                body: Any = json.loads(response.content)
            else:
                body = response.content
            replies.append(Reply(status=response.status_code, body=body, headers=dict(response.headers)))
        return replies

    def run(self, coroutine: Any) -> Any:
        return asyncio.run(coroutine)


def server_sender(server: PoolingServer) -> RecordingSender:
    """A sender answering every pooling request from ``server``."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=server.response_body(json.loads(request.content)))
    )
    return RecordingSender(lambda request: transport.handle_request(request))


def send(sender: RecordingSender, calls: Sequence[Call]) -> list[Reply]:
    """Run ``sender.send`` to completion on a fresh loop (the tests' sync bridge)."""
    return asyncio.run(sender.send(calls))


def b64(array: np.ndarray) -> str:
    """The little-endian flattened frame of ``array`` (what ``tensor2binary`` sends for a 2-D tensor)."""
    return base64.b64encode(np.asarray(array, dtype="<f2").tobytes()).decode("ascii")


def request(
    contents: Sequence[Content],
    *,
    embed_dtype: str = "float16",
    dim: int | None = None,
    outputs: str = "per_token",
) -> PoolRequest:
    return PoolRequest(
        contents=tuple(contents),
        role=EncodeRole.DOCUMENT,
        embed_dtype=embed_dtype,  # type: ignore[arg-type]
        dim=dim,
        outputs=outputs,  # type: ignore[arg-type]
    )
