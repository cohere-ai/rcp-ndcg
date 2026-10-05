"""The embedding role client: every content decision between an :class:`~rcp_ndcg.inference.config.EmbeddingEndpoint`
and the wire.

The client owns what the model reads and how the requests go out -- the adapters and the transport own
everything around that:

* **prompts per side** -- ``query_prompt`` / ``doc_prompt`` prepended to every item of that side
  (:meth:`_prepare`, the one seam the text-budget mechanism plugs into when it lands);
* **the Matryoshka cut** -- ``dimensions`` sent to the adapter only when the config sets one;
* **normalisation** -- the vectors L2-normalised when ``normalize`` (the default), so an inner product is a
  cosine (normalising twice is harmless);
* **batching and concurrency** -- items sliced into ``batch_size``-sized requests, at most ``concurrency``
  requests in flight, reassembled in the input's order;
* **credentials** -- the key read from the config's ``api_key_env``, else the adapter profile's own variables,
  sent in the profile's header (``Authorization: Bearer``; Gemini's ``x-goog-api-key``).

Text is sent as given: cutting to ``max_tokens`` waits for the text-budget mechanism (a config that sets
``max_tokens`` is refused, never silently ignored), and media is refused by the adapter (these endpoints are
text-only for now).
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Coroutine, Sequence
from dataclasses import replace
from typing import Any, Protocol, runtime_checkable

import numpy as np
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import ConfigError, CredentialsError
from rcp_ndcg.inference.adapters import embeddings as _shipped_adapters  # noqa: F401  # registers them
from rcp_ndcg.inference.adapters import get_adapter, known_adapters
from rcp_ndcg.inference.adapters.base import Adapter
from rcp_ndcg.inference.config import EmbeddingEndpoint
from rcp_ndcg.inference.transport import Sender, Transport
from rcp_ndcg.inference.types import Embeddings, EmbedRequest, EncodeRole, l2_normalize


@runtime_checkable
class _SyncSender(Sender, Protocol):
    """A :class:`~rcp_ndcg.inference.transport.Sender` that also runs coroutines to completion: the sync bridge
    :meth:`EmbeddingClient.encode` needs (the real :class:`~rcp_ndcg.inference.transport.Transport`, or a fake
    with the same shape in tests)."""

    def run(self, coroutine: Coroutine[Any, Any, Embeddings]) -> Embeddings:
        """Run ``coroutine`` to completion and return its result."""
        ...


def _profile(adapter: type[Adapter[Any, Any]], attribute: str, default: Any) -> Any:
    """A profile attribute an adapter may leave out (``MAX_BATCH``, ``DEFAULT_BASE_URL``, ...)."""
    return getattr(adapter, attribute, default)


def _check_batch_size(adapter: type[Adapter[Any, Any]], size: int) -> None:
    """Refuse a request size above the profile's published cap, instead of silently capping it."""
    max_batch = _profile(adapter, "MAX_BATCH", None)
    if max_batch is not None and size > max_batch:
        raise ConfigError(
            f"the {getattr(adapter, 'name', '?')} embedding API takes at most {max_batch} texts per request; "
            f"batch_size is {size}",
            hint=f"set batch_size to {max_batch} or less, or leave it unset",
        )


class EmbeddingClient:
    """The content decisions of one embedding endpoint, on the :class:`~rcp_ndcg.inference.transport.Sender` seam.

    Args:
        config: The endpoint: the wire adapter (``api``), the model, the per-side prompts, ``normalize``,
            ``dimensions``, ``batch_size``, ``concurrency`` and the credentials.
        sender: What sends the calls; a :class:`~rcp_ndcg.inference.transport.Transport` built on ``config``
            when ``None``. A transport is built on a copy of the config with the profile's public base URL
            filled in (a hosted API: ``base_url: null``) and ``api_key_env`` cleared -- the client resolves the
            key itself, in the profile's header, so the transport adds no second one.

    Raises:
        ConfigError: ``api`` names no registered adapter, or one of another role; ``max_tokens`` is set
            (cutting waits for the text-budget mechanism); ``batch_size`` exceeds the profile's cap.
    """

    def __init__(self, config: EmbeddingEndpoint, *, sender: Sender | None = None) -> None:
        adapter_cls = get_adapter(config.api)
        role = getattr(adapter_cls, "role", None)
        if role != "embed":
            raise ConfigError(
                f"api {config.api!r} is not an embedding adapter",
                hint=f"api: {config.api!r} is a {role} adapter; an EmbeddingEndpoint needs an embed one",
                details={"known": list(known_adapters())},
            )
        if config.max_tokens is not None:
            raise ConfigError(
                "max_tokens needs the text-budget mechanism, which is not wired yet",
                hint="drop max_tokens and send text that already fits, or wait for the text-budget wiring",
            )
        if config.dimensions is not None and not _profile(adapter_cls, "SUPPORTS_DIMENSIONS", True):
            raise ConfigError(
                f"the {config.api} embedding API takes no dimensions parameter; the cut would be silently ignored",
                hint="drop dimensions, or use api: openai_embeddings for a Matryoshka cut",
            )
        _check_batch_size(adapter_cls, config.batch_size)

        self.config = config
        self._adapter_cls: type[Adapter[Any, Any]] = adapter_cls
        self.adapter: Adapter[Any, Any] = adapter_cls()
        self.endpoint = self._resolved_endpoint(config, adapter_cls)
        self._sender: _SyncSender
        if sender is None:
            self._sender = Transport(self.endpoint)
        elif isinstance(sender, _SyncSender):
            self._sender = sender
        else:
            raise TypeError(
                "sender must be a Transport (it bridges the synchronous encode onto its event loop) "
                "or an object with the same shape (send/probe/usage/run)"
            )

    # -- the public calls ---------------------------------------------------
    def encode(self, contents: Sequence[Content], role: EncodeRole, *, batch_size: int | None = None) -> Embeddings:
        """Embed ``contents`` as ``role``, synchronously, through the sender's sync bridge.

        Args:
            contents: The queries or documents, in order.
            role: Which side of the retrieval pair these are (the prompts differ per side).
            batch_size: The request size for this call; the config's ``batch_size`` when ``None``. At most the
                profile's cap.

        Returns:
            One float32 vector per content, in the input's order, L2-normalised when ``normalize``.
        """
        return self._sender.run(self.aencode(contents, role, batch_size=batch_size))

    async def aencode(
        self, contents: Sequence[Content], role: EncodeRole, *, batch_size: int | None = None
    ) -> Embeddings:
        """Embed ``contents`` as ``role``, asynchronously: the batch requests in flight at once, in order.

        Same arguments and result as :meth:`encode`; the sync method runs this on the sender's event loop.
        """
        prepared = self._prepare(contents, role)
        size = self._request_size(batch_size)
        if not prepared:
            return Embeddings.empty(0)
        auth = self._auth_headers()

        requests = [
            EmbedRequest(contents=tuple(prepared[offset : offset + size]), role=role, dimensions=self.config.dimensions)
            for offset in range(0, len(prepared), size)
        ]
        calls = [
            [_with_auth(call, auth) for call in self.adapter.calls(request, model=self.config.model)]
            for request in requests
        ]

        gate = asyncio.Semaphore(self.config.concurrency)

        async def one(index: int) -> Embeddings:
            async with gate:
                replies = await self._sender.send(calls[index])
            return self.adapter.interpret(requests[index], replies)

        parts = await asyncio.gather(*(one(index) for index in range(len(requests))))
        vectors = np.concatenate([part.as_matrix() for part in parts])
        if self.config.normalize:
            vectors = l2_normalize(vectors)
        return Embeddings.single(vectors)

    # -- the content decisions ---------------------------------------------
    def _prepare(self, contents: Sequence[Content], role: EncodeRole) -> tuple[Content, ...]:
        """The content decisions, in one place: today the per-side prompt, later the text budget.

        Args:
            contents: The items as given.
            role: Which side of the retrieval pair they are (``query_prompt`` vs ``doc_prompt``).

        Returns:
            The items to send: each with the side's prompt prepended (as a text part, so media survives
            untouched), otherwise unchanged.
        """
        prompt = self.config.query_prompt if role is EncodeRole.QUERY else self.config.doc_prompt
        return tuple(content.with_text_prefix(prompt) for content in contents)

    def _request_size(self, batch_size: int | None) -> int:
        """The request size of one call: ``batch_size``, else the config's; above the profile's cap is refused."""
        size = self.config.batch_size if batch_size is None else batch_size
        if size < 1:
            raise ConfigError(f"batch_size must be at least 1, got {size}")
        _check_batch_size(self._adapter_cls, size)
        return size

    # -- credentials --------------------------------------------------------
    def _auth_headers(self) -> dict[str, str]:
        """The credential headers of this call: the key from the config's or the profile's environment
        variables, in the profile's header.

        Raises:
            CredentialsError: The profile requires a key and none of its variables is set; the hint names them.
        """
        if self.config.api_key_env is not None:
            # The config names the variable itself: silence would drop the key, so an unset one is an error
            # whatever the profile's own rule is.
            names = (self.config.api_key_env,)
            required = True
        else:
            names = _profile(self._adapter_cls, "API_KEY_ENV", ())
            required = _profile(self._adapter_cls, "KEY_REQUIRED", False)
        key = next((os.environ[name] for name in names if os.environ.get(name)), None)
        if key is None and required:
            raise CredentialsError(
                f"the {self.config.api} embedding API needs an API key ({' or '.join(names)} is not set)",
                hint=f"export {' or '.join(names)}=...",
            )
        if not key:
            return {}
        header = _profile(self._adapter_cls, "AUTH_HEADER", None)
        return {header: key} if header else {"Authorization": f"Bearer {key}"}

    # -- the endpoint the transport sees ------------------------------------
    @staticmethod
    def _resolved_endpoint(config: EmbeddingEndpoint, adapter: type[Adapter[Any, Any]]) -> EmbeddingEndpoint:
        """The config as the transport receives it: the profile's public URL when ``base_url`` is ``None``,
        and no ``api_key_env`` (the client resolved the key; the transport sends no duplicate)."""
        updates: dict[str, Any] = {}
        if config.base_url is None:
            default = _profile(adapter, "DEFAULT_BASE_URL", None)
            if not default:
                raise ConfigError(
                    f"api {config.api!r} has no public base URL", hint="set base_url to the endpoint's URL"
                )
            updates["base_url"] = default
        if config.api_key_env is not None:
            updates["api_key_env"] = None
        return config.model_copy(update=updates) if updates else config


def _with_auth(call: Any, auth: dict[str, str]) -> Any:
    """The call with the credential headers added (the adapter's own headers, if any, stay first)."""
    if not auth:
        return call
    return replace(call, headers={**call.headers, **auth})


__all__ = ["EmbeddingClient"]
