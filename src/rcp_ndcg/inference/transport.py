"""The transport: one endpoint's HTTP behaviour, shared by every role.

The transport owns everything around a request so that no adapter repeats it: the least-busy replica pick, the
bounded concurrency, the retries with their backoff and ``Retry-After``, the set-aside of a failing replica, the
parking while every replica is down, the shared status map, the credentials, the calls-and-tokens usage, and the
sync bridge the retrieval API's synchronous callers send through. Its behaviour is the judge client's
(:mod:`rcp_ndcg.llm.client`), over ``httpx`` instead of the OpenAI SDK, with the same numbers.

A role client holds one transport per endpoint, next to its wire adapter
(:func:`~rcp_ndcg.inference.adapters.base.get_adapter`), and sends one request like this::

    replies = await transport.send(adapter.calls(request, model=transport.endpoint.model))
    for reply in replies:
        transport.add_usage(adapter.usage(reply))
    result = adapter.interpret(request, replies)

The transport counts the calls and the failed calls itself; the tokens cross the adapter, which is where the
API's field names are known, and come back through :meth:`Transport.add_usage`.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import Coroutine, Mapping, Sequence
from types import TracebackType
from typing import Any, ClassVar, Protocol, Self, TypeVar, runtime_checkable

import httpx

from rcp_ndcg.errors import (
    BackendUnavailableError,
    ConfigError,
    CredentialsError,
    RequestRejectedError,
    status_error,
    status_is_unavailable,
)
from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.inference.fake import FAKE_SCHEME, fake_transport
from rcp_ndcg.inference.probe import read_replica
from rcp_ndcg.inference.types import Call, EngineInfo, Reply, TokenCount, Usage
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

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


class _Replica:
    """One replica URL and what the transport knows about it."""

    __slots__ = ("backoff", "down_until", "engine", "in_flight", "sent", "successes", "url")

    def __init__(self, url: str) -> None:
        self.url = url
        self.in_flight = 0
        self.sent = 0
        self.successes = 0
        self.down_until = 0.0
        """``time.monotonic()`` before which no request is sent to it (0: live)."""
        self.backoff: float | None = None
        """The next time it is set aside, for how long (``None``: the transport's first backoff)."""
        self.engine: EngineInfo | None = None


class _Unavailable(Exception):
    """One attempt failed the way the shared status map calls unavailable: a connection error, a timeout, or an
    HTTP 408, 429 or 5xx reply. Retried on the replica up to ``max_retries``, then the replica is set aside."""

    def __init__(self, describe: str, *, cause: BaseException | None = None, retry_after: float | None = None) -> None:
        super().__init__(describe)
        self.describe = describe
        """What happened, for a message: ``ConnectError: refused`` or ``HTTP 503: <body>`` (never a header)."""
        self.cause = cause
        """The underlying exception, chained into the errors the transport raises."""
        self.retry_after = retry_after
        """The server's ``Retry-After``, when the failed reply carried one."""


async def _sleep(seconds: float) -> None:
    """One backoff sleep; a module function so tests can replace the clock."""
    await asyncio.sleep(seconds)


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """The server's ``Retry-After`` in seconds, when it gives one as a number (the retrieval clients' policy)."""
    raw = headers.get("retry-after")
    try:
        return float(raw) if raw else None
    except (TypeError, ValueError):
        return None


def _body_text(response: httpx.Response) -> str:
    """The response body as text for an error message, cut to 500 characters (a header value is never in it)."""
    return response.text[:500]


def _reply(response: httpx.Response) -> Reply:
    """The :class:`~rcp_ndcg.inference.types.Reply` of one response: JSON decoded, ``application/octet-stream``
    (and anything that does not parse) kept as bytes."""
    content_type = response.headers.get("content-type", "")
    if "octet-stream" in content_type:
        body: Any = response.content
    else:
        try:
            body = response.json()
        except ValueError:
            body = response.content
    return Reply(status=response.status_code, body=body, headers=dict(response.headers))


class Transport:
    """One :class:`~rcp_ndcg.inference.endpoint.Endpoint`'s sending: routing, retries, parking, accounting.

    Implements :class:`Sender`; the role clients hold one per endpoint and share it. Safe to share across the
    coroutines of one event loop; a synchronous caller runs its requests through :meth:`run`, which owns one
    event loop and one pool for them all.

    The behaviour is the judge client's, over ``httpx`` instead of the OpenAI SDK, with the same numbers:

    * **routing** -- each request goes to the live replica with the fewest requests in flight from this
      transport, then the fewest sent;
    * **bounded concurrency** -- at most the endpoint's ``concurrency`` requests in flight over all replicas,
      and an httpx pool sized to it;
    * **credentials** -- ``Authorization: Bearer`` from ``api_key_env`` when the endpoint names one, plus one
      header per :attr:`~rcp_ndcg.inference.endpoint.Endpoint.headers_env` entry, every value read from the
      environment at send time and never logged;
    * **the shared status map** -- a connection error or timeout, or an HTTP 408, 429 or 5xx reply, is
      *unavailable*: retried up to ``max_retries`` with an exponential backoff (1 s doubling, capped at 60 s,
      the retrieval clients' policy) or the server's ``Retry-After``, then the replica is set aside. 401 and
      403 raise ``CredentialsError``, 404 a non-retryable ``ProviderError`` naming the URL and model; every
      other 4xx (400, 413, 422, ...) is returned as a :class:`~rcp_ndcg.inference.types.Reply` for the
      adapter's ``interpret`` to map onto its role-specific errors;
    * **set-aside and parking** -- a replica that failed is taken out of rotation for a backoff that doubles
      while it keeps failing (5 s to 60 s, the judge's numbers). While every replica is set aside, requests
      wait until the first comes back or ``wait_on_outage_s`` passes (the outage clock starts when the request
      holds a concurrency slot), then ``BackendUnavailableError`` whose message states how long the endpoint
      was unavailable;
    * **the rejection rule** -- a replica that served other requests since this request first failed there is
      up, and this failure belongs to this request: ``RequestRejectedError``.
    """

    #: First time a replica is set aside after it failed (seconds); doubles up to the maximum. The judge's
    #: numbers, kept.
    BACKOFF_S: ClassVar[float] = 5.0
    MAX_BACKOFF_S: ClassVar[float] = 60.0
    #: The first within-request retry delay (seconds), doubling per attempt up to :attr:`RETRY_MAX_BACKOFF_S`
    #: or the server's ``Retry-After``; the retrieval clients' policy.
    RETRY_BACKOFF_S: ClassVar[float] = 1.0
    RETRY_MAX_BACKOFF_S: ClassVar[float] = 60.0

    def __init__(self, endpoint: Endpoint, *, httpx_transport: httpx.AsyncBaseTransport | None = None) -> None:
        """A transport for ``endpoint``'s replicas.

        Args:
            endpoint: The endpoint whose replicas are routed; ``base_url`` is one URL or a replica list, and a
                ``fake://`` URL sends through the offline fakes (:mod:`rcp_ndcg.inference.fake`) unless
                ``httpx_transport`` is supplied, which answers instead of them.
            httpx_transport: A caller-supplied ``httpx.AsyncBaseTransport`` (a mock in tests), wrapped in the
                transport's own ``httpx.AsyncClient`` with the endpoint's timeouts and pool limits -- never
                replacing them, unlike the judge client of today, where a supplied client replaced both. The
                pool limits size the transport's own pool (the default httpx transport); a supplied transport
                pools as it pleases.
        """
        if not endpoint.urls:
            raise ConfigError(
                f"the endpoint {endpoint.model!r} has no base_url to send to: give one URL, or a list of replica URLs"
            )
        self.endpoint = endpoint
        self._replicas = [_Replica(url) for url in endpoint.urls]
        self._httpx_transport: httpx.AsyncBaseTransport | None = httpx_transport
        if httpx_transport is None and any(url.startswith(FAKE_SCHEME) for url in endpoint.urls):
            self._httpx_transport = fake_transport(endpoint.urls[0], model=endpoint.model)
        self._usage = Usage()
        self._pool: httpx.AsyncClient | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._own_loop: asyncio.AbstractEventLoop | None = None
        """The sync bridge's private loop (:meth:`run`); the pool of its calls is bound to it."""
        self._background_loop: asyncio.AbstractEventLoop | None = None
        self._background_thread: threading.Thread | None = None
        self._last_error: BaseException | None = None

    # ------------------------------------------------------------------
    # Sending
    # ------------------------------------------------------------------

    async def send(self, calls: Sequence[Call]) -> list[Reply]:
        """Send ``calls`` for one request to one live replica and return their replies, one per call, in order.

        A request's calls all go to the one replica the request was routed to; a retried request is sent whole
        again, so no partial result comes back. A reply the status map returns (any 4xx outside the map) is the
        adapter's to interpret. The credentials and ``headers_env`` are read once here, so a missing variable
        fails before the request is queued (and counts as a failed call, as the judge's does).

        Raises:
            BackendUnavailableError: Every replica stayed down for longer than ``wait_on_outage_s``.
            RequestRejectedError: The endpoint refused this request after serving others.
            CredentialsError: The endpoint refused the credentials (HTTP 401 or 403), or a configured
                environment variable (``api_key_env``, ``headers_env``) is not set.
            ProviderError: The endpoint has no such route or model (HTTP 404); non-retryable.
            ValueError: ``calls`` is empty.
        """
        if not calls:
            raise ValueError("send() needs at least one call")
        calls = list(calls)
        try:
            headers = self._base_headers()
        except CredentialsError:
            self._usage = self._usage + Usage(failed_calls=len(calls))  # the request failed before it was queued
            raise
        #: Per replica: its successes when this request first failed there.
        failed_at: dict[int, int] = {}
        #: When this request first found the endpoint unavailable; its outage clock (never the queueing time).
        outage_since: float | None = None
        async with self._gate():
            while True:
                replica = self._pick()
                if replica is None:
                    outage_since = time.monotonic() if outage_since is None else outage_since
                    await self._park(outage_since)
                    continue
                replica.in_flight += 1
                replica.sent += 1
                try:
                    replies = await self._send_on(replica, calls, headers)
                except _Unavailable as exc:
                    index = self._replicas.index(replica)
                    if index in failed_at and replica.successes > failed_at[index]:
                        # Other requests got through on this replica since this one first failed there: the
                        # replica is up, and this failure belongs to this request.
                        raise RequestRejectedError(
                            f"the endpoint refused this request after serving others ({exc.describe})"
                        ) from exc.cause
                    failed_at.setdefault(index, replica.successes)
                    outage_since = time.monotonic() if outage_since is None else outage_since
                    self._set_aside(replica, exc)
                    continue
                except Exception:
                    self._usage = self._usage + Usage(failed_calls=len(calls))
                    raise
                finally:
                    replica.in_flight -= 1
                replica.successes += 1
                replica.down_until, replica.backoff = 0.0, None
                self._usage = self._usage + Usage(calls=len(calls))
                return replies

    async def _send_on(self, replica: _Replica, calls: Sequence[Call], headers: Mapping[str, str]) -> list[Reply]:
        """One request's calls to one replica, with the within-request retries; no parking (the caller routes)."""
        for attempt in range(self.endpoint.max_retries + 1):
            try:
                return [await self._one(replica, call, headers) for call in calls]
            except _Unavailable as exc:
                if attempt == self.endpoint.max_retries:
                    raise
                delay = min(self.RETRY_BACKOFF_S * 2**attempt, self.RETRY_MAX_BACKOFF_S)
                if exc.retry_after is not None:
                    delay = min(exc.retry_after, self.RETRY_MAX_BACKOFF_S)
                logger.warning("%s unavailable (%s); retrying in %.1fs", replica.url, exc.describe, delay)
                await _sleep(delay)
        raise AssertionError("unreachable")

    async def _one(self, replica: _Replica, call: Call, base_headers: Mapping[str, str]) -> Reply:
        """One call, once, to one replica; the shared status map decides reply or error."""
        headers = {**base_headers, **call.headers}
        path = call.path if call.path.startswith("/") else f"/{call.path}"
        request = self._client().build_request(call.method, self._url(replica, path), json=call.json, headers=headers)
        try:
            response = await self._client().send(request)
        except httpx.TransportError as exc:
            raise _Unavailable(f"{type(exc).__name__}: {exc}", cause=exc) from exc
        status = response.status_code
        if status_is_unavailable(status):
            text = _body_text(response)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:  # always: every unavailable status is an error status
                raise _Unavailable(
                    f"HTTP {status}: {text}", cause=exc, retry_after=_retry_after(response.headers)
                ) from exc
            raise AssertionError(f"unreachable: HTTP {status} is an error status")
        error = status_error(status, url=replica.url, path=path, model=self.endpoint.model, body=_body_text(response))
        if error is not None:
            raise error
        return _reply(response)

    def _url(self, replica: _Replica, path: str) -> str:
        """The request URL: the replica's base URL then the call's path; their queries, if any, joined."""
        if not path.startswith("/"):
            path = f"/{path}"
        base, _, base_query = replica.url.partition("?")
        path_only, _, path_query = path.partition("?")
        query = "&".join(part for part in (base_query, path_query) if part)
        return f"{base}{path_only}?{query}" if query else f"{base}{path_only}"

    def _base_headers(self) -> dict[str, str]:
        """The endpoint's credentials and gateway headers of one send; every value is read from the environment
        only, at send time, and never logged."""
        headers: dict[str, str] = {}
        api_key_env = self.endpoint.api_key_env
        if api_key_env is not None:
            value = os.environ.get(api_key_env)
            if not value:
                raise CredentialsError(
                    f"the endpoint needs an API key in ${api_key_env}, which is not set",
                    hint=f"export {api_key_env}=...  (keys are read from the environment, never from configs)",
                    details={"variable": api_key_env},
                )
            headers["Authorization"] = f"Bearer {value}"
        for header, variable in self.endpoint.headers_env.items():
            value = os.environ.get(variable)
            if not value:
                raise CredentialsError(
                    f"the endpoint needs the {header} header from ${variable}, which is not set",
                    hint=f"export {variable}=...  (header values are read from the environment, never from configs)",
                    details={"variable": variable, "header": header},
                )
            headers[header] = value
        return headers

    # ------------------------------------------------------------------
    # Routing, set-aside, parking
    # ------------------------------------------------------------------

    def _gate(self) -> asyncio.Semaphore:
        """The concurrency semaphore of the running loop; a new loop (a new pass, the sync bridge's own) gets a
        new one and a new pool, which are bound to the loop they first ran on."""
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._loop is not loop:
            self._semaphore = asyncio.Semaphore(self.endpoint.concurrency)
            self._loop = loop
            self._pool = None  # an HTTP client is bound to the loop it was first used on
        return self._semaphore

    def _pick(self) -> _Replica | None:
        """The live replica with the fewest requests in flight (then the fewest sent); ``None`` when all are down."""
        now = time.monotonic()
        live = [replica for replica in self._replicas if replica.down_until <= now]
        return min(live, key=lambda replica: (replica.in_flight, replica.sent)) if live else None

    def _set_aside(self, replica: _Replica, exc: _Unavailable) -> None:
        """Take ``replica`` out of rotation after it failed, for a backoff that doubles while it keeps failing."""
        self._last_error = exc.cause or exc
        now = time.monotonic()
        if replica.down_until > now:
            return  # already set aside, by a request sent before it went down
        wait = replica.backoff or self.BACKOFF_S
        replica.down_until = now + wait
        replica.backoff = min(wait * 2, self.MAX_BACKOFF_S)
        others = sum(other.down_until <= now for other in self._replicas)
        logger.warning(
            "%s unavailable (%s); not sending to it for %.0fs%s",
            replica.url,
            exc.describe,
            wait,
            f", {others} other replica(s) live" if len(self._replicas) > 1 else "",
        )

    async def _park(self, outage_since: float) -> None:
        """Wait while every replica is set aside, until the first comes back or ``wait_on_outage_s`` is spent.

        ``outage_since`` is when this request first found the endpoint unavailable (its first failure, or the
        moment it found every replica set aside), measured once the request held a concurrency slot: time spent
        queued behind other requests is not outage.

        Raises:
            BackendUnavailableError: the endpoint has been unavailable to the request for ``wait_on_outage_s``.
        """
        now = time.monotonic()
        waited = now - outage_since
        limit = self.endpoint.wait_on_outage_s
        where = ", ".join(replica.url for replica in self._replicas)
        last = self._last_error
        if limit is not None and waited >= limit:
            raise BackendUnavailableError(
                f"{where} was unavailable for {waited:.1f}s (wait_on_outage_s={limit}); "
                f"last error: {type(last).__name__}: {last}"
            ) from last
        wake = min(replica.down_until for replica in self._replicas) - now
        if limit is not None:
            wake = min(wake, limit - waited)
        logger.warning("%s unavailable; waiting %.0fs before re-sending (waited %.0fs)", where, wake, waited)
        await _sleep(max(wake, 0.0))

    # ------------------------------------------------------------------
    # The HTTP pool
    # ------------------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        """The HTTP client every replica shares, pooled to the concurrency.

        httpx pools 100 connections by default: a larger concurrency would queue on the pool, not the endpoint.
        """
        if self._pool is None:
            concurrency = self.endpoint.concurrency
            self._pool = httpx.AsyncClient(
                timeout=self._timeout(),
                limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
                transport=self._httpx_transport,
            )
        return self._pool

    def _timeout(self) -> httpx.Timeout:
        """The endpoint's per-request and connect timeouts."""
        return httpx.Timeout(self.endpoint.timeout_s, connect=self.endpoint.connect_timeout_s)

    # ------------------------------------------------------------------
    # The provenance probe
    # ------------------------------------------------------------------

    async def probe(self) -> list[EngineInfo]:
        """Ask each replica what it serves (``GET <base_url>/models``), best effort; the result is :attr:`engines`.

        Never raises: an endpoint that cannot be read is recorded with its ``error``, and the role goes on (the
        requests themselves park while it is down).
        """
        self._gate()  # binds the HTTP client to this event loop, as a request would
        await asyncio.gather(*(self._probe_replica(replica) for replica in self._replicas))
        return self.engines

    async def _probe_replica(self, replica: _Replica) -> None:
        fingerprint = replica.engine.system_fingerprint if replica.engine is not None else None
        try:
            replica.engine = await read_replica(
                self._client(),
                replica.url,
                model=self.endpoint.model,
                headers=self._base_headers(),
                timeout=self._timeout(),
                system_fingerprint=fingerprint,
            )
        except Exception as exc:  # best effort: what the endpoint says is recorded, never required
            replica.engine = EngineInfo(
                url=replica.url, system_fingerprint=fingerprint, error=f"{type(exc).__name__}: {exc}"
            )

    @property
    def engines(self) -> list[EngineInfo]:
        """What each replica reported (:meth:`probe`), with the fingerprint of its first completion."""
        return [replica.engine for replica in self._replicas if replica.engine is not None]

    def note_system_fingerprint(self, url: str, fingerprint: str) -> None:
        """Record a replica's ``system_fingerprint`` -- the role client reads it from its first reply's body --
        onto the replica's engine record, so :attr:`engines` carries it and :meth:`probe` preserves it."""
        for replica in self._replicas:
            if replica.url == url:
                engine = replica.engine or EngineInfo(url=url)
                if engine.system_fingerprint is None:
                    replica.engine = engine.model_copy(update={"system_fingerprint": fingerprint})
                return

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    @property
    def usage(self) -> Usage:
        """Calls and tokens accumulated so far (calls, failed calls, input and output tokens).

        :meth:`send` counts the calls and the failed calls itself: a request the transport raises on is a failed
        call, whether before it was queued (a missing credentials variable) or after it was sent (the status
        map's typed errors); a request the rejection rule refuses, one parked out by ``wait_on_outage_s``, and a
        reply the status map returns (even one the adapter refuses) are not. The tokens arrive through
        :meth:`add_usage`.
        """
        return self._usage

    def add_usage(self, tokens: TokenCount | None) -> None:
        """Add one reply's tokens to :attr:`usage`: the role clients call it once per reply, with the adapter's
        ``usage(reply)`` (``None`` when the API reports no tokens, which adds nothing)."""
        if tokens is None:
            return
        self._usage = Usage(
            calls=self._usage.calls,
            failed_calls=self._usage.failed_calls,
            input_tokens=self._usage.input_tokens + (tokens.input_tokens or 0),
            output_tokens=self._usage.output_tokens + (tokens.output_tokens or 0),
        )

    # ------------------------------------------------------------------
    # The sync bridge
    # ------------------------------------------------------------------

    def run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        """Run ``coroutine`` to completion on this transport's event loop: the sync bridge for the retrieval
        API's synchronous callers (as judging already does with ``asyncio.run``).

        All of a synchronous call's requests run on one loop through this bridge, and the pool is bound to it:
        called repeatedly, the loop and the pool are reused. When a loop is already running in this thread (a
        notebook), the coroutine runs on a private background thread of the transport instead of failing.

        Args:
            coroutine: The awaitable to run (a role client's call).

        Returns:
            The coroutine's result, as its own type ``T``.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            # A loop is already running in this thread (a notebook): run on the transport's own background
            # thread instead of failing, as run_until_complete here would.
            return asyncio.run_coroutine_threadsafe(coroutine, self._background()).result()
        loop = self._own_loop
        if loop is None:
            loop = self._own_loop = asyncio.new_event_loop()
        return loop.run_until_complete(coroutine)

    def _background(self) -> asyncio.AbstractEventLoop:
        """The transport's private background loop, started once, for :meth:`run` inside a running loop."""
        if self._background_loop is None:
            self._background_loop = asyncio.new_event_loop()
            self._background_thread = threading.Thread(
                target=self._background_loop.run_forever, name="rcp-ndcg-transport", daemon=True
            )
            self._background_thread.start()
        return self._background_loop

    def aclose(self) -> None:
        """Close the underlying client and its connection pool; safe to call more than once.

        Synchronous, so a caller of :meth:`run` can clean up without an event loop of its own: the async close
        runs on the pool's own loop. Called from the loop the pool serves (an async caller), the close is
        scheduled instead of blocking that loop on itself. A later :meth:`run` builds a fresh pool.
        """
        pool, loop = self._pool, self._loop
        self._pool = None
        self._semaphore = None
        self._loop = None
        if pool is None or loop is None:
            return
        try:
            running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is running:
            asyncio.run_coroutine_threadsafe(pool.aclose(), loop)  # blocking here would deadlock this loop
            return
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(pool.aclose(), loop).result()
            return
        loop.run_until_complete(pool.aclose())

    def close(self) -> None:
        """The sync twin of :meth:`aclose` (which is synchronous too): closes the pool; safe to call twice."""
        self.aclose()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["Sender", "Transport"]
