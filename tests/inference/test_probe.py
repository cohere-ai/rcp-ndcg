"""The provenance probe against in-process endpoints: what each replica reports, and what it cannot say.

The cases are the judge client's probe tests (``tests/llm/test_client.py``), ported to the shared transport;
the originals stay untouched and keep passing.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from rcp_ndcg.inference import Endpoint, Transport


class ReplicaScript:
    """One replica's behaviour: a script of statuses answered in order (then 200), and its requests."""

    def __init__(self, *script: int) -> None:
        self.script = list(script)
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status = self.script.pop(0) if self.script else 200
        return httpx.Response(status, json={"ok": True} if status < 400 else {"error": {"message": "down"}})


def _transport(handler: Any, **config: Any) -> Transport:
    """One replica at ``http://judge.test/v1``, model ``m``, answering from ``handler``."""
    return Transport(
        Endpoint(base_url="http://judge.test/v1", model="m", max_retries=0, **config),
        httpx_transport=httpx.MockTransport(handler),
    )


def _models(*entries: dict, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(200, json={"object": "list", "data": list(entries)}, headers=headers or {})


def test_the_probe_records_what_each_replica_reports() -> None:
    def serve(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            entries = [{"id": "other", "owned_by": "x"}, {"id": "m", "owned_by": "vllm", "max_model_len": 131072}]
            return httpx.Response(200, json={"object": "list", "data": entries}, headers={"server": "uvicorn"})
        return httpx.Response(200, json={"ok": True})

    transport = _transport(serve)
    (engine,) = asyncio.run(transport.probe())
    assert (engine.model, engine.owned_by, engine.max_model_len) == ("m", "vllm", 131072)
    assert engine.headers == {"server": "uvicorn"} and engine.error is None


def test_a_probe_keeps_the_system_fingerprint_the_client_reported() -> None:
    def serve(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": "m", "owned_by": "vllm"}]})
        return httpx.Response(200, json={"ok": True})

    transport = _transport(serve)
    transport.note_system_fingerprint("http://judge.test/v1", "vllm-0.30.0")
    (engine,) = asyncio.run(transport.probe())
    assert engine.system_fingerprint == "vllm-0.30.0" and engine.owned_by == "vllm"


def test_an_endpoint_that_cannot_be_read_is_recorded_not_raised() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    transport = Transport(
        Endpoint(base_url=["http://a/v1", "http://b/v1"], model="m"),
        httpx_transport=httpx.MockTransport(refuse),
    )
    engines = asyncio.run(transport.probe())
    assert [engine.url for engine in engines] == ["http://a/v1", "http://b/v1"]
    assert all(engine.error and "ConnectError" in engine.error for engine in engines)


def test_a_server_that_does_not_serve_the_endpoints_model_is_named(caplog: pytest.LogCaptureFixture) -> None:
    transport = _transport(
        lambda request: httpx.Response(200, json={"object": "list", "data": [{"id": "org/weights"}]})
    )
    with caplog.at_level("WARNING", logger="rcp_ndcg"):
        (engine,) = asyncio.run(transport.probe())
    assert engine.model == "org/weights"
    assert any("--served-model-name m" in record.getMessage() for record in caplog.records)


def test_the_probe_sends_the_credentials_and_never_logs_their_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("RCP_NDCG_TEST_KEY", "sekrit-key")
    script = ReplicaScript()
    transport = _transport(script, api_key_env="RCP_NDCG_TEST_KEY")
    with caplog.at_level("WARNING", logger="rcp_ndcg"):
        (engine,) = asyncio.run(transport.probe())
    assert engine.error is None
    assert script.requests[0].headers["Authorization"] == "Bearer sekrit-key"
    assert all("sekrit-key" not in record.getMessage() for record in caplog.records)


def test_a_probe_without_its_api_key_is_recorded_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RCP_NDCG_TEST_KEY", raising=False)
    transport = _transport(ReplicaScript(), api_key_env="RCP_NDCG_TEST_KEY")
    (engine,) = asyncio.run(transport.probe())
    assert engine.error and "CredentialsError" in engine.error


def test_a_probe_in_a_later_event_loop_reads_the_endpoint_again() -> None:
    """Each pass may run in its own event loop; the probe must not reuse connections of a closed one."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Models(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: a connection outlives the first event loop

        def do_GET(self) -> None:
            body = json.dumps({"object": "list", "data": [{"id": "m", "owned_by": "engine"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Models)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        transport = Transport(Endpoint(base_url=f"http://127.0.0.1:{server.server_port}/v1", model="m"))
        for _ in range(2):
            (engine,) = asyncio.run(transport.probe())
            assert engine.error is None and engine.owned_by == "engine", engine.error
    finally:
        server.shutdown()
