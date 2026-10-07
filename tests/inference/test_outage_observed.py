"""The observed outage behaviour as a transport test (GPU-VALIDATION.md T4's offline counterpart).

T4's outage scenario observes, on the node, that a run whose judge engine is killed **parks** and
recovers when the engine is back, and that an outage past ``wait_on_outage_s`` fails with
:class:`~rcp_ndcg.errors.BackendUnavailableError`.  That observation is recorded here against the real
socket stack: a real HTTP server on an ephemeral port is killed (its socket dies like an engine's), and
the real :class:`~rcp_ndcg.inference.Transport` parks through the outage and speaks to the server when a
new one binds the same port.  The in-process outage-clock cases live in ``tests/inference/test_transport.py``;
these cover what only a real dead socket shows: connection refusals while the engine is down.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from rcp_ndcg.errors import BackendUnavailableError
from rcp_ndcg.inference import Call, Endpoint, Transport

#: The one chat completion the stub judge answers (the judge route's shape).
_ANSWER = {
    "object": "chat.completion",
    "model": "judge",
    "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
}


class JudgeHandler(BaseHTTPRequestHandler):
    """One judge route: a 2xx chat completion, whatever the request body."""

    def log_message(self, *args: object) -> None:  # noqa: A001, ARG002 - stdlib signature
        return

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        body = json.dumps(_ANSWER).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Engine:
    """A judge engine as the outage scenario kills it: one server on a fixed port, started, killed, back.

    ``kill`` closes its listener (the connection refusals the transport parks through); ``start`` binds a
    new server on the same port, as a restarted engine does.
    """

    def __init__(self, port: int) -> None:
        self.port = port
        self.server: ThreadingHTTPServer | None = None

    def start(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", self.port), JudgeHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def kill(self) -> None:
        assert self.server is not None
        self.server.shutdown()
        self.server.server_close()
        self.server = None


@pytest.fixture
def engine() -> Engine:
    running = Engine(_free_port())
    running.start()
    yield running
    if running.server is not None:
        running.kill()


@pytest.fixture(autouse=True)
def _fast_parking(monkeypatch: pytest.MonkeyPatch) -> None:
    """The park's re-probe interval from the root suite's rule: the outage clock tests let milliseconds
    stand for the seconds the node observes (the semantics are identical, the sleeps shorter)."""
    monkeypatch.setattr(Transport, "BACKOFF_S", 0.02)
    monkeypatch.setattr(Transport, "MAX_BACKOFF_S", 0.05)


def _send(engine: Engine, **endpoint: Any) -> Any:
    transport = Transport(
        Endpoint(base_url=f"http://127.0.0.1:{engine.port}/v1", model="judge", max_retries=0, **endpoint)
    )
    (reply,) = asyncio.run(transport.send([Call("POST", "/chat/completions", {"model": "judge"})]))
    return reply


def test_a_killed_engine_parks_the_request_until_it_is_back(engine: Engine) -> None:
    """The observed outage (T4 scenario 2a): the engine is killed mid-run, the request parks through the
    outage and completes when the engine restarts on its port."""
    engine.kill()
    restart = threading.Timer(0.4, engine.start)
    restart.start()
    try:
        started = time.monotonic()
        reply = _send(engine, wait_on_outage_s=30.0)
    finally:
        restart.cancel()
    elapsed = time.monotonic() - started
    assert reply.body["choices"], reply.body
    assert elapsed >= 0.35  # parked for at least the outage


def test_an_outage_past_wait_on_outage_s_fails_as_backend_unavailable(engine: Engine) -> None:
    """The observed expiry (T4 scenario 2b): the engine stays down past ``wait_on_outage_s`` (0.2 s here),
    and the request fails with :class:`~rcp_ndcg.errors.BackendUnavailableError`, whose message states how
    long the endpoint was unavailable."""
    engine.kill()
    started = time.monotonic()
    with pytest.raises(BackendUnavailableError) as raised:
        _send(engine, wait_on_outage_s=0.2)
    waited = time.monotonic() - started
    assert waited >= 0.2
    assert isinstance(raised.value, BackendUnavailableError)
    assert "unavailable" in str(raised.value)
