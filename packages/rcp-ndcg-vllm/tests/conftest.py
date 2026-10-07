"""Shared fixtures: a stub engine on an ephemeral port, the recipe roots and pairs."""

from __future__ import annotations

import ipaddress
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
STUB = TESTS / "stub_engine.py"
FIXTURES = TESTS / "fixtures"
RECIPES = FIXTURES / "recipes"
TOKENIZER = FIXTURES / "tokenizer.json"


@pytest.fixture(autouse=True)
def _no_dns(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No network in tests: every hostname resolution is recorded and refused (an attempt swallowed
    by an ``except Exception`` still fails the test here), then the run fails if any name was tried.
    IP literals keep resolving (the stub engines are on 127.0.0.1); the explicitly marked network
    tests run without the watchdog when ``RCP_NDCG_NETWORK_TESTS=1`` enables them."""
    if os.environ.get("RCP_NDCG_NETWORK_TESTS"):
        yield
        return
    original = socket.getaddrinfo
    tried: list[str] = []

    def _watched(host: object, *args: object, **kwargs: object) -> object:
        try:
            ipaddress.ip_address(str(host))
        except ValueError:
            tried.append(str(host))
            raise OSError(f"the test tried to resolve {host!r}: tests use no network") from None
        return original(host, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(socket, "getaddrinfo", _watched)
    yield
    assert not tried, f"tests use no network (tried to resolve: {sorted(set(tried))})"


class StubEngine:
    """One stub engine subprocess on an ephemeral port."""

    def __init__(self, process: subprocess.Popen[bytes], port: int) -> None:
        self.process = process
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"

    def stop(self) -> None:
        """Terminate the engine, escalating to SIGKILL if it ignores SIGTERM."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def start_stub(*extra: str, env: dict[str, str] | None = None) -> StubEngine:
    """Start ``tests/stub_engine.py --port 0`` (plus extra flags) and return its announced port."""
    process = subprocess.Popen(
        [sys.executable, str(STUB), "--port", "0", *extra],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**{key: value for key, value in __import__("os").environ.items()}, **(env or {})},
    )
    line = process.stdout.readline().decode() if process.stdout else ""
    if not line.startswith("RCPS_STUB_PORT="):
        process.kill()
        raise AssertionError(f"the stub engine did not announce its port: {line!r}")
    return StubEngine(process, int(line.strip().split("=", 1)[1]))


@pytest.fixture
def stub() -> Iterator[StubEngine]:
    """One clean stub engine."""
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        yield engine
    finally:
        engine.stop()


def sandbox_path(fakes: Path, *tools: str) -> str:
    """A hermetic ``PATH`` for a node-script test: the test's fakes, then a sandbox of the named system tools.

    No system bin directory is on it, so a tool the machine happens to have (a real ``gcloud`` in
    ``/usr/bin``, as on GitHub's runner images) can never stand in for one the test means to be absent. Each
    named tool is symlinked from the caller's ``PATH`` into ``<fakes>/../sandbox-bin``; a tool the machine
    lacks fails the test here, by name.
    """
    sandbox = fakes.parent / "sandbox-bin"
    sandbox.mkdir(exist_ok=True)
    for tool in tools:
        target = shutil.which(tool)
        assert target is not None, f"the sandbox needs {tool!r}, and this machine has none on PATH"
        link = sandbox / tool
        if not link.is_symlink():
            link.symlink_to(target)
    return f"{fakes}:{sandbox}"


def write_pairs(path: Path, pairs: list[dict]) -> Path:
    """One pairs JSONL file."""
    import json

    path.write_text("".join(json.dumps(row) + "\n" for row in pairs), encoding="utf-8")
    return path


def sample_pairs(documents: int = 4) -> list[dict]:
    """Two queries with a few short documents each (the fixture tokenizer's words)."""
    return [
        {
            "query": "capital of france",
            "documents": [f"document {index} about cities and rivers in europe {index}" for index in range(documents)],
            "instruction": "Follow the task.",
        },
        {
            "query": "second query about retrieval models",
            "documents": [f"another document {index} with tokens a b c {index}" for index in range(documents)],
        },
    ]
