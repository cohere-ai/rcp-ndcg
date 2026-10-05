"""Shared offline plumbing for the embedding tests: a fake ``Sender`` and the profiles' reply shapes.

The real transport (and the ``fake://`` fakes below it) serve the auth tests, so these tests speak to the
frozen ``Sender`` seam: a sender that records every call, replies from a handler, counts requests in
flight, and bridges synchronous calls with ``asyncio.run``.

``DEFAULT_TOKENIZER`` is the saved test tokenizer the ``conftest`` fixture fills in: a self-hosted role
config must declare its budget (``tokenizer`` + ``max_tokens``), so ``endpoint`` declares the default one
for the served wire adapters; a hosted profile needs neither.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rcp_ndcg.inference.config import SELF_HOSTED_APIS, EmbeddingEndpoint
from rcp_ndcg.inference.types import Call, Reply, Usage

#: The default explicit budget's tokenizer, filled by the ``conftest`` fixture (a saved ``tokenizer.json``).
DEFAULT_TOKENIZER = ""


def vendor_payload(api: str, vectors: list[list[float]]) -> dict[str, Any]:
    """The reply body ``api`` answers a batch of ``vectors`` with (the shapes of today's ``api_dense``)."""
    if api in ("openai_embeddings", "voyage"):
        return {"data": [{"index": index, "embedding": vector} for index, vector in enumerate(vectors)]}
    if api == "cohere":
        return {"embeddings": {"float": vectors}}
    if api == "gemini":
        return {"embeddings": [{"values": vector} for vector in vectors]}
    raise ValueError(f"unknown embedding api {api!r}")


def openai_data(vectors: list[list[float]], *, indices: list[int] | None = None) -> dict[str, Any]:
    """An OpenAI-shaped ``data`` list, optionally with the entries' ``index`` fields in a given order."""
    order = indices if indices is not None else list(range(len(vectors)))
    return {"data": [{"index": index, "embedding": vectors[index]} for index in order]}


class FakeSender:
    """A :class:`~rcp_ndcg.inference.transport.Sender` for offline tests: records what it sent, replies from a
    handler, and counts the requests in flight (with an optional per-call delay to force overlap)."""

    def __init__(
        self,
        handler: Callable[[Call], Reply],
        *,
        delay_fn: Callable[[Call], float] | None = None,
    ) -> None:
        self.handler = handler
        self.delay_fn = delay_fn
        self.calls: list[Call] = []
        self.in_flight = 0
        self.peak = 0

    async def send(self, calls: Any) -> list[Reply]:
        self.calls.extend(calls)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            for call in calls:
                if self.delay_fn is not None:
                    await asyncio.sleep(self.delay_fn(call))
            return [self.handler(call) for call in calls]
        finally:
            self.in_flight -= 1

    async def probe(self) -> list[Any]:
        return []

    @property
    def usage(self) -> Usage:
        return Usage()

    def run(self, coroutine: Any) -> Any:
        return asyncio.run(coroutine)


def endpoint(api: str = "openai_embeddings", **overrides: Any) -> EmbeddingEndpoint:
    """An embedding endpoint for tests: model ``m``, any profile, any overrides.

    A self-hosted profile (``openai_embeddings``) declares the explicit budget by default (the saved test
    tokenizer and a cap of 8192 tokens); a hosted profile declares neither, unless the test overrides."""
    if api in SELF_HOSTED_APIS and DEFAULT_TOKENIZER:
        overrides.setdefault("tokenizer", DEFAULT_TOKENIZER)
        overrides.setdefault("max_tokens", 8192)
    return EmbeddingEndpoint(api=api, model="m", **overrides)
