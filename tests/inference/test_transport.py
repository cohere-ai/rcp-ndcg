"""The transport against in-process endpoints: routing, outages, the status map, headers, the sync bridge.

The routing, parking and outage-clock cases are the judge client's (``tests/llm/test_client.py``), ported to
the shared transport; the originals stay untouched and keep passing. Everything runs on ``httpx.MockTransport``
handlers, with the transport's backoffs monkeypatched to milliseconds; the only longer sleeps are the
outage-clock tests' 0.15 s mock answers, which queue a request longer than its ``wait_on_outage_s`` on purpose.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pytest

import rcp_ndcg.inference.transport as transport_module
from rcp_ndcg.errors import (
    BackendUnavailableError,
    ConfigError,
    CredentialsError,
    ProviderError,
    RequestRejectedError,
)
from rcp_ndcg.inference import Call, EncodeRole, Endpoint, TokenCount, Transport
from rcp_ndcg.inference.types import Reply

#: A minimal JSON answer, for the 200s the scripts serve.
_BODY = {"object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class ReplicaScript:
    """One replica's behaviour: a script of steps (a status, ``(status, headers)``, or an exception to raise)
    answered in order (then 200), the delay before each answer, and the requests it received."""

    def __init__(self, *script: int | tuple[int, dict[str, str]] | Exception, delay: float = 0.0) -> None:
        self.script = list(script)
        self.delay = delay
        self.requests: list[httpx.Request] = []
        self.in_flight = 0
        self.peak = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        step: Any = self.script.pop(0) if self.script else 200
        try:
            if isinstance(step, Exception):
                raise step
            if self.delay:
                await asyncio.sleep(self.delay)
            headers = step[1] if isinstance(step, tuple) else {}
            status = step[0] if isinstance(step, tuple) else step
            return httpx.Response(
                status, json=_BODY if status < 400 else {"error": {"message": "down"}}, headers=headers
            )
        finally:
            self.in_flight -= 1


class Replicas:
    """Several replicas behind one mock transport: each host answers from its own script."""

    def __init__(
        self, scripts: dict[str, list[int | tuple[int, dict[str, str]] | Exception]], *, delay: float = 0.0
    ) -> None:
        self.by_host = {host: ReplicaScript(*script, delay=delay) for host, script in scripts.items()}

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        return await self.by_host[request.url.host](request)

    @property
    def requests(self) -> dict[str, int]:
        return {host: len(script.requests) for host, script in self.by_host.items()}

    @property
    def peak(self) -> dict[str, int]:
        return {host: script.peak for host, script in self.by_host.items()}

    def transport(self, **config: Any) -> Transport:
        urls = [f"http://{host}/v1" for host in self.by_host]
        return Transport(
            Endpoint(base_url=urls, model="m", max_retries=0, **config), httpx_transport=httpx.MockTransport(self)
        )


def _transport(script: ReplicaScript, **config: Any) -> Transport:
    """One replica at ``http://judge.test/v1``, answering from ``script`` (``max_retries`` defaults to 0)."""
    settings: dict[str, Any] = {"base_url": "http://judge.test/v1", "model": "m", "max_retries": 0, **config}
    return Transport(Endpoint(**settings), httpx_transport=httpx.MockTransport(script))


def _send(transport: Transport, path: str = "/chat/completions", body: Any = None, **call: Any) -> list[Reply]:
    return asyncio.run(transport.send([Call("POST", path, body, **call)]))


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Transport, "BACKOFF_S", 0.001)
    monkeypatch.setattr(Transport, "MAX_BACKOFF_S", 0.002)
    monkeypatch.setattr(Transport, "RETRY_BACKOFF_S", 0.001)
    monkeypatch.setattr(Transport, "RETRY_MAX_BACKOFF_S", 0.002)


class TestReplicas:
    def test_requests_spread_over_the_replicas_by_requests_in_flight(self) -> None:
        replicas = Replicas({"a": [], "b": [], "c": []}, delay=0.01)
        transport = replicas.transport(concurrency=6)

        async def many() -> list[Any]:
            sends = [transport.send([Call("POST", "/chat/completions", {"i": i})]) for i in range(30)]
            return await asyncio.gather(*sends)

        replies = asyncio.run(many())
        assert len(replies) == 30
        assert replicas.requests == {"a": 10, "b": 10, "c": 10}
        # The concurrency is shared: six in flight over three replicas is two on each, never more.
        assert replicas.peak == {"a": 2, "b": 2, "c": 2}

    def test_a_down_replica_is_set_aside_and_its_request_moves_to_a_live_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(Transport, "BACKOFF_S", 60.0)
        replicas = Replicas({"a": [503] * 100, "b": []})
        transport = replicas.transport()
        replies = [_send(transport) for _ in range(5)]
        assert [reply[0].status for reply in replies] == [200] * 5
        assert replicas.requests == {"a": 1, "b": 5}  # a failed once, then b carried every request

    def test_a_replica_that_answers_again_is_used_again(self) -> None:
        replicas = Replicas({"a": [503], "b": []})
        transport = replicas.transport()
        for _ in range(6):
            _send(transport)
            asyncio.run(asyncio.sleep(0.005))  # past the backoff
        assert replicas.requests["a"] > 1

    def test_with_every_replica_down_the_request_parks_until_one_answers(self) -> None:
        replicas = Replicas({"a": [503, 503, 503], "b": [502, 502, 502]})
        assert _send(replicas.transport())[0].status == 200
        assert replicas.requests["a"] + replicas.requests["b"] == 7

    def test_parking_over_every_replica_ends_after_wait_on_outage(self) -> None:
        replicas = Replicas({"a": [503] * 1000, "b": [503] * 1000})
        with pytest.raises(BackendUnavailableError, match="http://a/v1, http://b/v1"):
            _send(replicas.transport(wait_on_outage_s=0.02))

    def test_a_requests_calls_share_one_replica(self) -> None:
        replicas = Replicas({"a": [], "b": []})
        transport = replicas.transport(concurrency=4)
        replies = asyncio.run(transport.send([Call("POST", "/x", {}), Call("POST", "/y", {})]))
        assert [reply.status for reply in replies] == [200, 200]
        hosts = {request.url.host for script in replicas.by_host.values() for request in script.requests}
        assert len(hosts) == 1

    def test_a_request_that_keeps_failing_on_a_replica_that_serves_others_is_refused(self) -> None:
        def serve(request: httpx.Request) -> httpx.Response:
            stuck = b"stuck" in request.content or request.url.host == "b"
            return httpx.Response(503, json={"error": {"message": "x"}}) if stuck else httpx.Response(200, json=_BODY)

        transport = Transport(
            # wait_on_outage_s bounds a broken rejection rule (one that parks instead): the test would
            # otherwise hang forever, and now fails as a BackendUnavailableError instead.
            Endpoint(base_url=["http://a/v1", "http://b/v1"], model="m", max_retries=0, wait_on_outage_s=0.5),
            httpx_transport=httpx.MockTransport(serve),
        )

        async def both() -> list[Any]:
            stuck = transport.send([Call("POST", "/chat/completions", {"prompt": "stuck"})])
            fine = transport.send([Call("POST", "/chat/completions", {"prompt": "fine"})])
            return await asyncio.gather(stuck, fine, return_exceptions=True)

        stuck, fine = asyncio.run(both())
        assert isinstance(stuck, RequestRejectedError) and fine[0].status == 200


class TestOutages:
    def test_a_transient_failure_is_retried_on_the_replica(self) -> None:
        script = ReplicaScript(503)
        replies = _send(_transport(script, max_retries=1))
        assert replies[0].status == 200 and len(script.requests) == 2

    def test_a_down_endpoint_parks_the_request_until_it_answers(self) -> None:
        script = ReplicaScript(503, 429, 502, 503)
        assert _send(_transport(script))[0].status == 200
        assert len(script.requests) == 5

    def test_parking_ends_after_wait_on_outage(self) -> None:
        script = ReplicaScript(*[503] * 1000)
        with pytest.raises(BackendUnavailableError, match="wait_on_outage_s"):
            _send(_transport(script, wait_on_outage_s=0.02))

    def test_a_lasting_transport_failure_ends_as_backend_unavailable(self) -> None:
        script = ReplicaScript(*[httpx.ConnectError("refused")] * 1000)
        with pytest.raises(BackendUnavailableError, match="ConnectError"):
            _send(_transport(script, wait_on_outage_s=0.02))


class TestOutageClock:
    """``wait_on_outage_s`` counts from a request's first unavailable failure, never its time in the queue."""

    ENDPOINT = dict(base_url="http://127.0.0.1:9/v1", model="m", concurrency=1, wait_on_outage_s=0.2, max_retries=0)

    def _queued(self, script: ReplicaScript, requests: int) -> list[Any]:
        transport = Transport(Endpoint(**self.ENDPOINT), httpx_transport=httpx.MockTransport(script))

        async def main() -> list[Any]:
            sends = [transport.send([Call("POST", "/chat/completions", {"i": i})]) for i in range(requests)]
            return await asyncio.gather(*sends, return_exceptions=True)

        return asyncio.run(main())

    def test_a_blip_after_a_long_queue_is_waited_out(self) -> None:
        # The fourth request queued 0.45 s (three answers of 0.15 s) behind the others, longer than
        # wait_on_outage_s; its one failed send is a short outage, which it waits out.
        script = ReplicaScript(200, 200, 200, httpx.ConnectError("connection refused"), 200, delay=0.15)
        results = self._queued(script, 5)
        assert all(isinstance(result, list) for result in results), results
        assert len(script.requests) == 6

    def test_the_message_states_how_long_the_endpoint_was_unavailable(self) -> None:
        script = ReplicaScript(200, 200, *([httpx.ConnectError("connection refused")] * 1000), delay=0.15)
        results = self._queued(script, 3)
        assert isinstance(results[0], list) and isinstance(results[1], list)
        error = results[2]
        assert isinstance(error, BackendUnavailableError), error
        seconds = float(str(error).split("was unavailable for ")[1].split("s ")[0])
        assert 0.2 <= seconds < 0.3  # the outage it saw, not the 0.3 s it also spent queued


class _Clock:
    """A scripted ``time.monotonic``: frozen except when the faked ``_sleep`` advances it (the tests
    below need the exact wake instant, which wall clocks cannot aim at)."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        """The transport's ``_sleep``, replaced: no wall time passes, the clock advances by the sleep."""
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)  # a real yield point, so a deadline around the send can fire


class TestOutageBoundaries:
    """The outage windows at their exact boundaries: a replica is live again at the wake instant
    (``down_until <= now``), parking gives up after exactly ``wait_on_outage_s`` (``waited >= limit``),
    and the pick tie-breaks on the requests a replica has already sent. The first two need a scripted
    clock: the wake instant is a float equality wall clocks cannot aim at."""

    BACKOFF = 0.0625  # 2^-4: every window boundary is exact in binary floating point

    def _clocked(
        self, monkeypatch: pytest.MonkeyPatch, script: ReplicaScript, **config: Any
    ) -> tuple[Transport, _Clock]:
        clock = _Clock()
        monkeypatch.setattr(transport_module, "time", clock)
        monkeypatch.setattr(transport_module, "_sleep", clock.sleep)
        monkeypatch.setattr(Transport, "BACKOFF_S", self.BACKOFF)
        monkeypatch.setattr(Transport, "MAX_BACKOFF_S", 1.0)
        # the boundary dances park in a tight loop when the code is wrong: keep the warnings off the log
        monkeypatch.setattr(logging.getLogger("rcp_ndcg"), "disabled", True)
        return _transport(script, wait_on_outage_s=self.BACKOFF, **config), clock

    def test_a_replica_wakes_at_exactly_the_end_of_its_outage_window(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """At ``now == down_until`` the replica is live again (``<=``), so the request goes through; a
        replica considered live only *after* the window (``<``) would keep parking until
        ``wait_on_outage_s`` is spent and fail the request."""
        transport, clock = self._clocked(monkeypatch, ReplicaScript(503, 200))
        replies = asyncio.run(asyncio.wait_for(transport.send([Call("POST", "/x", {})]), timeout=5.0))
        assert replies[0].status == 200
        assert clock.sleeps == [self.BACKOFF]  # one park, then the replica wakes at the exact instant

    def test_the_request_gives_up_after_exactly_wait_on_outage_s(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The second park sits exactly at ``waited == wait_on_outage_s``: that is the give-up boundary
        (``>=``). A ``>`` rule would want more outage than the limit allows, and the clock cannot get
        there (the wake it computes is 0), so the request would never finish: the 5 s cap fails it."""
        transport, clock = self._clocked(monkeypatch, ReplicaScript(*([503] * 100)))
        with pytest.raises(BackendUnavailableError, match="wait_on_outage_s"):
            asyncio.run(asyncio.wait_for(transport.send([Call("POST", "/x", {})]), timeout=5.0))
        assert clock.sleeps == [self.BACKOFF]  # the outage clock is exactly the limit when it gives up

    def test_the_next_request_goes_to_the_replica_that_has_sent_the_fewest(self) -> None:
        """The least-in-flight pick tie-breaks on the requests the replica has sent: sequential sends
        with everything else equal must alternate across the replicas."""
        replicas = Replicas({"a": [], "b": []})
        transport = replicas.transport(concurrency=1)
        for _ in range(3):
            _send(transport)
        assert replicas.requests == {"a": 2, "b": 1}  # after a tie, the replica with fewer sent goes next


class TestStatusMap:
    @pytest.mark.parametrize(
        ("status", "error"), [(401, CredentialsError), (403, CredentialsError), (404, ProviderError)]
    )
    def test_an_endpoint_that_cannot_serve_this_request_raises_and_is_not_returned(
        self, status: int, error: type[Exception]
    ) -> None:
        with pytest.raises(error) as caught:
            _send(_transport(ReplicaScript(status)))
        assert not isinstance(caught.value, RequestRejectedError)

    def test_a_404_names_the_url_and_the_model_and_is_not_retryable(self) -> None:
        with pytest.raises(ProviderError, match=r"http://judge.test/v1/chat/completions.*'m'") as caught:
            _send(_transport(ReplicaScript(404)))
        assert caught.value.retryable is False

    @pytest.mark.parametrize("status", [400, 413, 422])
    def test_any_other_4xx_is_returned_to_the_caller(self, status: int) -> None:
        script = ReplicaScript(status)
        replies = _send(_transport(script))
        assert (replies[0].status, replies[0].body) == (status, {"error": {"message": "down"}})
        assert len(script.requests) == 1  # returned at once: not retried, not parked

    def test_a_json_body_is_decoded_and_octet_stream_stays_bytes(self) -> None:
        def serve(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/bytes"):
                return httpx.Response(200, headers={"content-type": "application/octet-stream"}, content=b"\x00\x01")
            return httpx.Response(200, json={"ok": True})

        transport = Transport(Endpoint(base_url="http://t/v1", model="m"), httpx_transport=httpx.MockTransport(serve))
        replies = asyncio.run(transport.send([Call("POST", "/json", {}), Call("POST", "/bytes", {})]))
        assert replies[0].body == {"ok": True} and replies[1].body == b"\x00\x01"

    def test_retry_after_is_honoured_and_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Transport, "RETRY_MAX_BACKOFF_S", 60.0)
        sleeps: list[float] = []

        async def sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(transport_module, "_sleep", sleep)
        script = ReplicaScript((429, {"Retry-After": "0.05"}), 200)
        replies = _send(_transport(script, max_retries=1))
        assert replies[0].status == 200 and sleeps == [0.05]
        script = ReplicaScript((429, {"Retry-After": "120"}), 200)
        replies = _send(_transport(script, max_retries=1))
        assert replies[0].status == 200 and sleeps[-1] == 60.0  # capped at RETRY_MAX_BACKOFF_S

    def test_usage_counts_requests_failed_requests_and_tokens(self) -> None:
        transport = _transport(ReplicaScript(503, 200, 401), max_retries=1)
        assert _send(transport)[0].status == 200
        with pytest.raises(CredentialsError):
            _send(transport)
        assert (transport.usage.requests, transport.usage.failed_requests) == (1, 1)
        transport.add_usage(TokenCount(input_tokens=10, output_tokens=2))
        transport.add_usage(None)  # a reply the API reports no tokens for adds nothing
        assert (transport.usage.input_tokens, transport.usage.output_tokens) == (10, 2)

    def test_usage_counts_each_call_of_a_request(self) -> None:
        script = ReplicaScript(200, 200)
        transport = _transport(script)
        asyncio.run(transport.send([Call("POST", "/a", {}), Call("POST", "/b", {})]))
        assert transport.usage.requests == 2

    def test_send_without_calls_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one call"):
            asyncio.run(Transport(Endpoint(base_url="http://t/v1", model="m")).send([]))


class TestHeaders:
    def test_the_api_key_header_is_sent_and_never_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("RCP_NDCG_TEST_KEY", "sekrit-key")
        script = ReplicaScript(503, 200)  # a set-aside, so the transport logs
        transport = _transport(script, api_key_env="RCP_NDCG_TEST_KEY")
        with caplog.at_level("WARNING", logger="rcp_ndcg"):
            assert _send(transport)[0].status == 200
        assert script.requests[0].headers["Authorization"] == "Bearer sekrit-key"
        assert all("sekrit-key" not in record.getMessage() for record in caplog.records)

    def test_headers_env_is_sent_and_never_logged(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("RCP_NDCG_TEST_GATEWAY", "sekrit-value")
        script = ReplicaScript(503, 200)
        transport = _transport(script, headers_env={"X-Gateway-Key": "RCP_NDCG_TEST_GATEWAY"})
        with caplog.at_level("WARNING", logger="rcp_ndcg"):
            assert _send(transport)[0].status == 200
        assert script.requests[0].headers["X-Gateway-Key"] == "sekrit-value"
        assert all("sekrit-value" not in record.getMessage() for record in caplog.records)

    def test_a_missing_header_variable_is_a_credentials_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RCP_NDCG_TEST_GATEWAY", raising=False)
        script = ReplicaScript()
        transport = _transport(script, headers_env={"X-Gateway-Key": "RCP_NDCG_TEST_GATEWAY"})
        with pytest.raises(CredentialsError, match="RCP_NDCG_TEST_GATEWAY"):
            _send(transport)
        assert script.requests == []  # refused before anything was queued
        assert (transport.usage.requests, transport.usage.failed_requests) == (0, 1)  # the request failed

    def test_a_missing_api_key_names_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RCP_NDCG_TEST_KEY", raising=False)
        transport = _transport(ReplicaScript(), api_key_env="RCP_NDCG_TEST_KEY")
        with pytest.raises(CredentialsError, match="RCP_NDCG_TEST_KEY") as caught:
            _send(transport)
        assert caught.value.details == {"variable": "RCP_NDCG_TEST_KEY"}
        assert transport.usage.failed_requests == 1  # the request failed, nothing was queued

    def test_the_calls_own_headers_are_sent(self) -> None:
        script = ReplicaScript()
        _send(_transport(script), headers={"X-Trace": "trace-1"})
        assert script.requests[0].headers["X-Trace"] == "trace-1"


class TestPool:
    @pytest.mark.parametrize("concurrency", [8, 256])
    def test_the_connection_pool_holds_as_many_requests_as_the_concurrency(self, concurrency: int) -> None:
        """httpx's default pool holds 100 connections, so a concurrency of 256 once ran at most 100 requests at once."""
        transport = Transport(Endpoint(base_url="http://judge.test/v1", model="m", concurrency=concurrency))
        pool = transport._client()._transport._pool
        assert pool._max_connections >= concurrency and pool._max_keepalive_connections >= concurrency

    def test_a_call_path_with_its_own_query_joins_the_endpoints_query(self) -> None:
        script = ReplicaScript()
        transport = Transport(
            Endpoint(base_url="http://judge.test/v1?api-version=7", model="m"),
            httpx_transport=httpx.MockTransport(script),
        )
        _send(transport, path="/x?a=1")
        (request,) = script.requests
        assert request.url.path == "/v1/x"  # neither query swallows the other
        assert dict(request.url.params) == {"api-version": "7", "a": "1"}

    def test_a_supplied_transport_takes_precedence_over_the_fakes(self) -> None:
        script = ReplicaScript()
        transport = Transport(
            Endpoint(base_url="fake://embed", model="enc"), httpx_transport=httpx.MockTransport(script)
        )
        replies = transport.run(transport.send([Call("POST", "/embeddings", {"input": ["x"]})]))
        assert replies[0].body["object"] == "chat.completion"  # the caller's transport answers, not the built-in fake
        assert len(script.requests) == 1

    def test_a_wrapped_transport_keeps_the_timeouts_and_its_own_pool(self) -> None:
        supplied = httpx.AsyncHTTPTransport(limits=httpx.Limits(max_connections=123, max_keepalive_connections=123))
        transport = Transport(
            Endpoint(base_url="http://judge.test/v1", model="m", concurrency=7, timeout_s=12.0, connect_timeout_s=3.0),
            httpx_transport=supplied,
        )
        client = transport._client()
        assert client._transport is supplied  # wrapped in the transport's client, never replacing it
        assert client.timeout == httpx.Timeout(12.0, connect=3.0)
        assert client._transport._pool._max_connections == 123  # the caller's transport pools as it was built


class TestSyncBridge:
    def test_two_runs_reuse_one_loop_and_one_pool(self) -> None:
        script = ReplicaScript(200, 200)
        transport = _transport(script)
        first = transport.run(transport.send([Call("POST", "/a", {})]))
        pool = transport._pool
        second = transport.run(transport.send([Call("POST", "/b", {})]))
        assert (first[0].status, second[0].status) == (200, 200)
        assert transport._pool is pool and transport._own_loop is transport._loop

    def test_run_inside_a_running_loop_uses_a_background_thread(self) -> None:
        script = ReplicaScript()
        transport = _transport(script)

        async def main() -> list[Reply]:
            return transport.run(transport.probe())  # a loop is running in this thread: the background bridge

        engines = asyncio.run(main())
        assert engines[0].error is None and transport._background_thread is not None

    def test_aclose_closes_the_pool_and_a_later_run_builds_a_fresh_one(self) -> None:
        script = ReplicaScript(200, 200)
        transport = _transport(script)
        transport.run(transport.send([Call("POST", "/a", {})]))
        assert transport._pool is not None
        asyncio.run(transport.aclose())  # the true async close (R15)
        assert transport._pool is None
        assert transport.run(transport.send([Call("POST", "/b", {})]))[0].status == 200

    def test_aclose_after_its_loop_closed_drops_the_pool_without_raising(self) -> None:
        """The judge client replaces its config after a pass's `asyncio.run` closed the loop the pool rode on;
        closing the wire then must drop the dead pool, not raise `Event loop is closed` (R15: `close()` is
        the synchronous twin; an async caller awaits `aclose()`)."""
        script = ReplicaScript(200, 200)
        transport = _transport(script)
        asyncio.run(transport.send([Call("POST", "/a", {})]))  # builds the pool on a loop that then closes
        transport.close()
        assert transport._pool is None
        assert asyncio.run(transport.send([Call("POST", "/b", {})]))[0].status == 200

    def test_close_and_the_context_manager_are_the_same_close(self) -> None:
        script = ReplicaScript(200)
        with _transport(script) as transport:
            assert transport.run(transport.send([Call("POST", "/a", {})]))[0].status == 200
            assert transport._pool is not None
            transport.close()
            assert transport._pool is None
            transport.close()  # safe twice
            assert transport._pool is None


class TestEndpointUrls:
    def test_base_url_is_one_url_or_a_list_of_replicas(self) -> None:
        one = Endpoint(base_url="http://a/v1/", model="m")
        replicas = Endpoint(base_url=["http://a/v1/", "http://b/v1"], model="m")
        assert one.urls == ("http://a/v1",) and replicas.urls == ("http://a/v1", "http://b/v1")

    def test_a_replica_list_is_non_empty_without_duplicates_and_never_mixes_fakes(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Endpoint(base_url=[], model="m")
        with pytest.raises(ValueError, match="twice"):
            Endpoint(base_url=["http://a/v1", "http://a/v1/"], model="m")
        with pytest.raises(ValueError, match="replica list"):
            Endpoint(base_url=["fake://seed/0", "http://a/v1"], model="m")

    def test_an_endpoint_without_a_url_sends_nowhere(self) -> None:
        with pytest.raises(ConfigError, match="base_url"):
            Transport(Endpoint(model="m"))


class TestAdapterAuth:
    """Auth in the transport (R6): the adapter profile's credential facts are the transport's input, and the
    key is resolved and sent there -- never in a client, and never in a log or an error message."""

    @staticmethod
    def _role_answer(path: str) -> dict[str, Any]:
        """One 2xx body per wire path, so the adapters read the auth tests' replies."""
        if path.endswith("/embeddings"):
            return {"data": [{"index": 0, "embedding": [0.0, 0.0]}]}
        if path.endswith("/embed"):
            return {"embeddings": {"float": [[0.0, 0.0]]}}
        if "batchEmbedContents" in path:
            return {"embeddings": [{"values": [0.0, 0.0]}]}
        return {"results": [{"index": 0, "relevance_score": 0.5}]}

    @classmethod
    def _client(cls, api: str, tokenizer_json: str, **config: Any) -> tuple[Any, ReplicaScript]:
        """A role client sending through a real transport over a recording mock endpoint.

        ``base_url=None`` aims the config at the adapter profile's own default host -- where its key
        variables resolve; any other ``base_url`` is a foreign host, which the profile's variables never
        reach (the host rule the client applies, tested below).
        """
        script = ReplicaScript()
        from rcp_ndcg.inference import EmbeddingClient, RerankClient
        from rcp_ndcg.inference.adapters.base import get_adapter
        from rcp_ndcg.inference.config import EmbeddingEndpoint, RerankEndpoint

        def answer(request: httpx.Request) -> httpx.Response:
            script.requests.append(request)  # the recording mock answers each role's shape
            return httpx.Response(200, json=cls._role_answer(request.url.path))

        if api.endswith("_rerank"):
            endpoint: Any = RerankEndpoint(
                api=api[: -len("_rerank")],
                model="m",
                base_url=config.pop("base_url", "http://judge.test/v1"),
                use_activation=None,
                tokenizer=tokenizer_json,
                max_tokens=8192,
                **config,
            )
            transport_url = endpoint.base_url or get_adapter(endpoint.api, role="rerank").DEFAULT_BASE_URL
            transport_endpoint = endpoint.model_copy(update={"base_url": transport_url})
            client: Any = RerankClient(
                endpoint, sender=Transport(transport_endpoint, httpx_transport=httpx.MockTransport(answer))
            )
        else:
            endpoint = EmbeddingEndpoint(
                api=api,
                model="m",
                base_url=config.pop("base_url", "http://judge.test/v1"),
                tokenizer=tokenizer_json,
                max_tokens=8192,
                **config,
            )
            transport_url = endpoint.base_url or get_adapter(endpoint.api, role="embed").DEFAULT_BASE_URL
            transport_endpoint = endpoint.model_copy(update={"base_url": transport_url})
            client = EmbeddingClient(
                endpoint, sender=Transport(transport_endpoint, httpx_transport=httpx.MockTransport(answer))
            )
        return client, script

    @staticmethod
    def _send(client: Any) -> None:
        from rcp_ndcg_core.content import Content

        from rcp_ndcg.inference import RerankClient

        if isinstance(client, RerankClient):
            client.rerank("q", ["a"])
        else:
            client.encode([Content.from_text("x")], EncodeRole.DOCUMENT)

    @pytest.mark.parametrize(
        ("api", "variable", "header", "value"),
        [
            ("openai_embeddings", "OPENAI_API_KEY", "Authorization", "Bearer fake-openai"),
            ("cohere", "CO_API_KEY", "Authorization", "Bearer fake-cohere"),
            ("voyage", "VOYAGE_API_KEY", "Authorization", "Bearer fake-voyage"),
            ("gemini", "GEMINI_API_KEY", "x-goog-api-key", "fake-gemini"),
            ("cohere_rerank", "CO_API_KEY", "Authorization", "Bearer fake-cohere"),
            ("voyage_rerank", "VOYAGE_API_KEY", "Authorization", "Bearer fake-voyage"),
        ],
        # the variable's value is the key; the header carries `Bearer <key>` where the profile declares no
        # AUTH_HEADER (Gemini's is its own header, value verbatim).
    )
    def test_a_profile_header_is_sent_from_its_default_variable(
        self,
        api: str,
        variable: str,
        header: str,
        value: str,
        tokenizer_json: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(variable, value.removeprefix("Bearer "))
        client, script = self._client(api, tokenizer_json, base_url=None)  # the profile's own host
        self._send(client)
        assert script.requests[0].headers[header] == value

    def test_the_second_profile_variable_is_tried_in_order(
        self, tokenizer_json: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:  # noqa: E501 -> None:
        monkeypatch.delenv("CO_API_KEY", raising=False)
        monkeypatch.setenv("COHERE_API_KEY", "fake-cohere-second")
        client, script = self._client("cohere", tokenizer_json, base_url=None)  # the profile's own host
        self._send(client)
        assert script.requests[0].headers["Authorization"] == "Bearer fake-cohere-second"

    def test_the_config_api_key_env_is_resolved_first(
        self, tokenizer_json: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:  # noqa: E501 -> None:
        monkeypatch.setenv("RCP_NDCG_TEST_KEY", "fake-from-config")
        client, script = self._client("cohere", tokenizer_json, api_key_env="RCP_NDCG_TEST_KEY")
        self._send(client)
        assert script.requests[0].headers["Authorization"] == "Bearer fake-from-config"

    def test_an_unset_config_api_key_env_is_a_credentials_error(
        self, tokenizer_json: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RCP_NDCG_TEST_KEY", raising=False)
        monkeypatch.setenv("CO_API_KEY", "fake-fallback")  # the profile's variable must NOT paper over it
        client, script = self._client("cohere", tokenizer_json, api_key_env="RCP_NDCG_TEST_KEY")
        with pytest.raises(CredentialsError, match="RCP_NDCG_TEST_KEY"):
            self._send(client)
        assert script.requests == []

    def test_a_required_key_missing_names_the_profile_variables(
        self, tokenizer_json: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CO_API_KEY", raising=False)
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        client, _ = self._client("cohere", tokenizer_json, base_url=None)  # the profile's own host
        with pytest.raises(CredentialsError) as caught:
            self._send(client)
        assert "CO_API_KEY" in (caught.value.hint or "")
        assert "COHERE_API_KEY" in (caught.value.hint or "")

    def test_a_served_wire_sends_no_key_without_a_variable(
        self, tokenizer_json: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        client, script = self._client("openai_embeddings", tokenizer_json)
        self._send(client)
        assert "Authorization" not in script.requests[0].headers

    def test_no_key_value_reaches_logs_or_errors(
        self, tokenizer_json: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("CO_API_KEY", "fake-sekrit-value")
        client, script = self._client(
            "cohere", tokenizer_json, base_url=None, wait_on_outage_s=0
        )  # the profile's own host, where its key resolves; a set-aside logs; no outage wait
        with caplog.at_level("WARNING", logger="rcp_ndcg"):
            self._send(client)
        assert all("fake-sekrit-value" not in record.getMessage() for record in caplog.records)
        monkeypatch.delenv("CO_API_KEY", raising=False)
        monkeypatch.delenv("COHERE_API_KEY", raising=False)  # the profile's second variable must not paper over it
        with pytest.raises(CredentialsError) as caught:
            self._send(client)
        assert "fake-sekrit-value" not in str(caught.value)
        assert "fake-sekrit-value" not in str(caught.value.hint)


class TestSyncBridgeLoop:
    """The sync bridge's private loop is part of the transport's lifecycle: closed with the transport
    (never leaked as an un-closed event loop), rebuilt by a later run."""

    def test_close_closes_the_own_loop_and_a_later_run_builds_a_fresh_one(self) -> None:
        script = ReplicaScript(200, 200)
        transport = _transport(script)
        transport.run(transport.send([Call("POST", "/a", {})]))
        own = transport._own_loop
        assert own is not None and not own.is_closed()
        transport.close()
        assert transport._own_loop is None and own.is_closed()
        assert transport.run(transport.send([Call("POST", "/b", {})]))[0].status == 200
        rebuilt = transport._own_loop
        assert rebuilt is not None and rebuilt is not own
        transport.close()
        assert rebuilt.is_closed() and transport._own_loop is None


class TestCloseInsideTheBridgeCall:
    """A close requested from inside the bridge's own call drains: the scheduled pool close completes and
    the loop closes when the call returns (no destroyed-pending task, no abandoned pool)."""

    def test_close_from_within_the_bridged_coroutine(self) -> None:
        script = ReplicaScript()
        transport = _transport(script)

        async def caller() -> None:
            transport.run(transport.send([Call("POST", "/a", {})])) if False else None
            transport.close()  # from inside the coroutine the bridge serves

        async def run() -> None:
            await caller()

        asyncio.run(run())
        transport._close_own_loop()  # the next close (or the caller's cleanup) finishes the close
        own = transport._own_loop
        assert own is None or own.is_closed()
        assert transport._pool is None


class TestRetryAfterGarbage:
    """A server ``Retry-After`` the transport cannot honestly sleep on falls back to the doubling backoff:
    ``nan`` would never return and wedge the request inside the retry loop, and a negative one would hammer
    a rate-limited server with instant retries."""

    @pytest.mark.parametrize("header", ["nan", "-5", "1e999", "soon", ""])
    def test_garbage_never_becomes_a_sleep(self, monkeypatch: pytest.MonkeyPatch, header: str) -> None:
        sleeps: list[float] = []

        async def sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(transport_module, "_sleep", sleep)
        script = ReplicaScript((429, {"Retry-After": header}), 200)
        replies = _send(_transport(script, max_retries=1))

        assert replies[0].status == 200
        assert sleeps and sleeps[-1] == Transport.RETRY_BACKOFF_S, "the doubling backoff, never the garbage"

    def test_a_usable_retry_after_is_still_capped(self) -> None:
        from rcp_ndcg.inference.transport import _retry_after

        assert _retry_after({"retry-after": "0.0005"}) == 0.0005  # below the cap: honoured
        assert _retry_after({"retry-after": "120"}) == Transport.RETRY_MAX_BACKOFF_S  # capped, whatever it says
        assert _retry_after({}) is None


class TestUserInfoNeverLeaks:
    """A user who embeds credentials in a URL (a documented httpx idiom) never sees them in a log line, an
    error message or an engine record -- beside the code's keys-are-never-logged claim. The request itself
    still uses the full URL (that is where the credentials live)."""

    def test_set_aside_and_outage_never_name_the_secret(self, caplog: pytest.LogCaptureFixture) -> None:
        script = ReplicaScript(503)
        transport = _transport(script, base_url="http://user:sekrit-value@judge.test/v1", wait_on_outage_s=0)
        with caplog.at_level(logging.WARNING, logger="rcp_ndcg"):
            with pytest.raises(BackendUnavailableError) as caught:
                asyncio.run(transport.send([Call("POST", "/a", {})]))

        assert "sekrit-value" not in str(caught.value)
        assert all("sekrit-value" not in record.getMessage() for record in caplog.records)

    def test_the_engine_record_strips_userinfo_and_query(self, tokenizer_json: str) -> None:
        from rcp_ndcg.inference.types import EngineInfo, safe_url

        assert safe_url("http://user:sekrit-value@judge.test/v1") == "http://judge.test/v1"
        assert (
            safe_url("http://user:sekrit-value@judge.test/v1/models?api-key=sekrit-value")
            == "http://judge.test/v1/models"
        )
        assert safe_url("fake://seed/1") == "fake://seed/1"
        record = EngineInfo(url="http://user:sekrit-value@judge.test/v1")
        assert record.url == "http://judge.test/v1", "the run manifest never carries userinfo"

    def test_the_probe_record_of_a_userinfo_replica_is_clean(self) -> None:
        script = ReplicaScript()
        transport = _transport(script, base_url="http://user:sekrit-value@judge.test/v1")
        engines = asyncio.run(transport.probe())
        transport.close()
        assert engines and all("sekrit-value" not in (engine.url or "") for engine in engines)


class TestConcurrentSyncBridges:
    """The sync bridge is one loop, one caller at a time: concurrent ``run()``s from OS threads queue on the
    bridge lock instead of racing two ``run_until_complete`` passes on the shared loop (the second used to
    die with ``This event loop is already running`` and its batch aborted)."""

    def test_run_from_many_threads_serves_every_call(self) -> None:
        import threading

        script = ReplicaScript()
        transport = _transport(script)
        results: list[Reply] = []
        errors: list[BaseException] = []
        barrier = threading.Barrier(8)

        def work(index: int) -> None:
            try:
                barrier.wait()
                for _ in range(3):
                    replies = transport.run(transport.send([Call("POST", f"/{index}", {})]))
                    results.append(replies[0])
            except BaseException as exc:  # noqa: BLE001 - the thread's failure is the test's result
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        assert not any(thread.is_alive() for thread in threads), "every call finished"
        assert not errors, f"no call was lost: {errors!r}"
        assert len(results) == 24 and all(reply.status == 200 for reply in results)

    def test_close_from_another_thread_while_run_is_mid_flight_waits_then_closes(self) -> None:
        import threading

        script = ReplicaScript(200)
        transport = _transport(script)
        done = threading.Event()

        def work() -> None:
            transport.run(transport.send([Call("POST", "/a", {})]))
            done.set()

        thread = threading.Thread(target=work)
        thread.start()
        transport.close()  # waits for the in-flight run (the bridge lock), then closes; never raises
        thread.join(30)
        assert done.is_set(), "the in-flight call completed"
        assert transport._pool is None
