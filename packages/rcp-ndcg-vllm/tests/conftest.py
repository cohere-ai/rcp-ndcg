"""Shared fixtures: a stub engine on an ephemeral port, the recipe roots and pairs."""

from __future__ import annotations

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
