"""The transport: one endpoint's HTTP behaviour, shared by every role.

The transport owns everything around a request so that no adapter repeats it: the least-busy replica pick, the
bounded concurrency, the retries with their backoff and ``Retry-After``, the set-aside of a failing replica, the
parking while every replica is down, the shared status map, and the calls-and-tokens usage. Its behaviour is
assigned by RFC-0001 section 4.3 and arrives with lane L1; every method below is the frozen signature the other
lanes build against.
"""

from __future__ import annotations

from collections.abc import Coroutine, Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

import httpx

from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.inference.types import Call, EngineInfo, Reply, Usage

T = TypeVar("T")
"""The result type of a coroutine the sync bridge runs."""


@runtime_checkable
class Sender(Protocol):
    """What sends calls and returns replies: the seam a fake transport or a recording one implements.

    Implementations hold the credentials, the pool and the accounting; ``send`` sends the calls of one request
    to one replica without interleaving other requests between them.
    """

    async def send(self, calls: Sequence[Call]) -> list[Reply]:
        """Send ``calls`` to the endpoint and return their replies, one per call, in order.

        Raises:
            BackendUnavailableError: Every replica stayed unavailable for longer than the endpoint's
                ``wait_on_outage_s``.
            RequestRejectedError: The endpoint refused this request after other requests got through.
            CredentialsError: The endpoint refused the credentials (HTTP 401 or 403).
            ProviderError: The endpoint has no such route or model (HTTP 404), or a request failed after its
                retries for a reason no status map covers.
        """
        ...

    async def probe(self) -> list[EngineInfo]:
        """Ask each replica what it serves (``GET <base_url>/models``), best effort; never raises."""
        ...

    @property
    def usage(self) -> Usage:
        """Calls and tokens accumulated so far (calls, failed calls, input and output tokens)."""
        ...


class Transport:
    """One :class:`~rcp_ndcg.inference.endpoint.Endpoint`'s sending: routing, retries, parking, accounting.

    Implements :class:`Sender`; the role clients hold one per endpoint and share it. Safe to share across the
    coroutines of one event loop.
    """

    def __init__(self, endpoint: Endpoint, *, httpx_transport: httpx.AsyncBaseTransport | None = None) -> None:
        """A transport for ``endpoint``'s replicas.

        Args:
            endpoint: The endpoint whose replicas are routed; ``base_url`` is one URL or a replica list.
            httpx_transport: A caller-supplied ``httpx.AsyncBaseTransport`` (a mock in tests); wrapped with the
                endpoint's timeouts and pool limits, never replacing them (unlike today's judge client, where a
                supplied client replaced both).
        """
        self.endpoint = endpoint
        self._httpx_transport = httpx_transport
        self.usage = Usage()
        raise NotImplementedError("lane L1")

    async def send(self, calls: Sequence[Call]) -> list[Reply]:
        """Send ``calls`` for one request to one live replica and return their replies, one per call, in order.

        The behaviour RFC-0001 section 4.3 assigns (today's judge client, over ``httpx`` instead of the OpenAI
        SDK):

        * **routing** -- each request goes to the live replica with the fewest requests in flight from this
          transport, then the fewest sent;
        * **bounded concurrency** -- at most the endpoint's ``concurrency`` requests in flight over all
          replicas, and an httpx pool sized to it;
        * **credentials** -- ``Authorization: Bearer <api_key_env>`` unless the adapter's profile says
          otherwise, plus one header per :attr:`~rcp_ndcg.inference.endpoint.Endpoint.headers_env` entry, its
          value read from the environment only;
        * **retries** -- a connection error or timeout, or an HTTP 408, 429 or 5xx reply, is retried up to the
          endpoint's ``max_retries`` with an exponential backoff (5 s doubling, capped at 60 s) or the server's
          ``Retry-After``, whichever is longer;
        * **set-aside and parking** -- a replica that failed is taken out of rotation for a backoff that doubles
          while it keeps failing; while every replica is set aside, requests wait until the first comes back or
          ``wait_on_outage_s`` passes (the outage clock starts when the request holds a concurrency slot), then
          ``BackendUnavailableError``;
        * **the rejection rule** -- a replica that served other requests since this request first failed there
          is up, and this failure belongs to this request: ``RequestRejectedError``;
        * **the shared status map** -- 401 and 403 raise ``CredentialsError``; 404 raises a non-retryable
          ``ProviderError``; 408, 429 and 5xx are outages; 400 and 422 are returned as replies, for the
          adapter's ``interpret`` to map onto its role-specific errors.

        Raises:
            BackendUnavailableError: Every replica stayed down for longer than ``wait_on_outage_s``.
            RequestRejectedError: The endpoint refused this request after serving others.
            CredentialsError: The endpoint refused the credentials (HTTP 401 or 403).
            ProviderError: The endpoint has no such route or model (HTTP 404), or the request failed after its
                retries with no mapped status.
        """
        raise NotImplementedError("lane L1")

    async def probe(self) -> list[EngineInfo]:
        """Ask each replica what it serves (``GET <base_url>/models``), best effort.

        Returns:
            One :class:`~rcp_ndcg.inference.types.EngineInfo` per replica that answered, with the entry of the
            endpoint's model; a replica that could not be read is reported with its ``error``. Never raises:
            an endpoint that says nothing about itself is recorded, and the requests themselves park while it
            is down.
        """
        raise NotImplementedError("lane L1")

    def run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        """Run ``coroutine`` to completion on this transport's event loop: the sync bridge for the retrieval
        API's synchronous callers (as judging already does with ``asyncio.run``).

        All of a synchronous call's requests run on one loop through this bridge; the pool is bound to it.

        Args:
            coroutine: The awaitable to run (a role client's call).

        Returns:
            The coroutine's result, as its own type ``T``.
        """
        raise NotImplementedError("lane L1")

    def aclose(self) -> None:
        """Close the underlying client and its connection pool; safe to call more than once.

        Synchronous, so a caller of :meth:`run` can clean up without an event loop of its own: the async close
        runs on the transport's own loop through :meth:`run`.
        """
        raise NotImplementedError("lane L1")


__all__ = ["Sender", "Transport"]
