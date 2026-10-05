"""The transport against in-process endpoints: routing, outages, the status map, headers, the sync bridge.

The routing, parking and outage-clock cases are the judge client's (``tests/llm/test_client.py``), ported to
the shared transport; the originals stay untouched and keep passing. Everything runs on ``httpx.MockTransport``
handlers, with the transport's backoffs monkeypatched to milliseconds; the only longer sleeps are the
outage-clock tests' 0.15 s mock answers, which queue a request longer than its ``wait_on_outage_s`` on purpose.
"""

from __future__ import annotations

import asyncio
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
from rcp_ndcg.inference import Call, Endpoint, TokenCount, Transport
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

    def test_usage_counts_calls_failed_calls_and_tokens(self) -> None:
        transport = _transport(ReplicaScript(503, 200, 401), max_retries=1)
        assert _send(transport)[0].status == 200
        with pytest.raises(CredentialsError):
            _send(transport)
        assert (transport.usage.calls, transport.usage.failed_calls) == (1, 1)
        transport.add_usage(TokenCount(input_tokens=10, output_tokens=2))
        transport.add_usage(None)  # a reply the API reports no tokens for adds nothing
        assert (transport.usage.input_tokens, transport.usage.output_tokens) == (10, 2)

    def test_usage_counts_each_call_of_a_request(self) -> None:
        script = ReplicaScript(200, 200)
        transport = _transport(script)
        asyncio.run(transport.send([Call("POST", "/a", {}), Call("POST", "/b", {})]))
        assert transport.usage.calls == 2

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
        assert (transport.usage.calls, transport.usage.failed_calls) == (0, 1)  # the request failed

    def test_a_missing_api_key_names_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RCP_NDCG_TEST_KEY", raising=False)
        transport = _transport(ReplicaScript(), api_key_env="RCP_NDCG_TEST_KEY")
        with pytest.raises(CredentialsError, match="RCP_NDCG_TEST_KEY") as caught:
            _send(transport)
        assert caught.value.details == {"variable": "RCP_NDCG_TEST_KEY"}
        assert transport.usage.failed_calls == 1  # the request failed, nothing was queued

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
        transport.aclose()
        assert transport._pool is None
        assert transport.run(transport.send([Call("POST", "/b", {})]))[0].status == 200

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
