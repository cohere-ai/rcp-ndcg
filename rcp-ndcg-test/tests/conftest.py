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

from tests._checkout import checkout_guard as _checkout_guard
from tests._checkout import entries as _checkout_entries

TESTS = Path(__file__).resolve().parent
STUB = TESTS / "stub_engine.py"
FIXTURES = TESTS / "fixtures"
RECIPES = FIXTURES / "recipes"
TOKENIZER = FIXTURES / "tokenizer.json"
ROOT = TESTS.parent.parent

#: The session-start baseline of the checkout tree (see :func:`pytest_sessionstart`).
_CHECKOUT_BASELINE: pytest.StashKey[set[str] | None] = pytest.StashKey()


def _snapshot_exempt(config: pytest.Config) -> bool:
    """A ``--update-snapshots`` run writes the generated files into the tree on purpose."""
    return bool(config.getoption("--update-snapshots", default=False)) or bool(
        os.environ.get("RCP_NDCG_UPDATE_SNAPSHOTS")
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Take the checkout guard's baseline before collection, so an import-time leak is caught too."""
    config = session.config
    config.stash[_CHECKOUT_BASELINE] = None if _snapshot_exempt(config) else _checkout_entries(ROOT)


@pytest.fixture(scope="session", autouse=True)
def _tests_leave_the_checkout_clean(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail the session when a test leaves a new file or directory in the checkout (tests write to tmp_path).

    The baseline is taken in ``pytest_sessionstart`` (before collection) and compared at session end, so an
    empty directory -- invisible to ``git status`` -- and an import-time write are both caught. The scan roots
    at the workspace root, not at this suite's subtree: the two test trees share one checkout, and a leak from
    either is a leak from the checkout. A ``--update-snapshots`` run is exempt (it writes on purpose).
    """
    baseline = request.config.stash[_CHECKOUT_BASELINE]
    if baseline is None:
        yield
        return
    with _checkout_guard(ROOT, before=baseline):
        yield


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


@pytest.fixture(autouse=True)
def _hub_is_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No network in tests: every wave's Hub question (the model's bytes) answers "unknown" here, for every
    test that runs a wave; the tests that need a size monkeypatch their own value on top of this. A Hub file
    read (the checkpoint's chat template) answers from the local cache only -- offline mode, so a missing
    file fails at once instead of retrying a refused connection."""
    from rcp_ndcg_test.jobs import weights

    monkeypatch.setattr(weights, "model_disk_bytes", lambda model, revision=None: None)
    if not os.environ.get("RCP_NDCG_NETWORK_TESTS"):
        import huggingface_hub.constants as constants

        monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)


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

    No system bin directory is on it, so a tool the machine has on its ``PATH`` (a real ``gcloud`` in
    ``/usr/bin``) is not found through it. Each named tool is symlinked from the caller's ``PATH`` into
    ``<fakes>/../sandbox-bin``; a tool the machine lacks fails the test here, by name. This closes the
    ``PATH`` only: a script that adds directories itself must be told not to -- the node scripts' Cloud SDK
    search (``gcs_sdk_on_path`` in ``jobs/gcs.sh``, which would put e.g. ``/usr/lib/google-cloud-sdk/bin``
    first) is switched off with ``RCP_GCLOUD_SDK_DIRS`` set empty.
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


def hub_cache(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    repo: str,
    revision: str,
    files: dict[str, str],
    absent: tuple[str, ...] = (),
) -> Path:
    """An offline Hub cache under ``root`` holding ``files`` for ``repo`` at ``revision`` (the layout
    huggingface_hub reads a commit-hash revision from without any request); the Hub is offline meanwhile.
    ``absent`` names the files the cache records as not in the repository (the ``.no_exist`` markers an online
    lookup leaves); a file neither held nor marked is unknown. Calling it again with another root replaces the
    cache."""
    import huggingface_hub.constants as constants

    model_dir = root / "hub" / ("models--" + repo.replace("/", "--"))
    snapshot = model_dir / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (snapshot / name).write_text(text, encoding="utf-8")
    for name in absent:
        marker = model_dir / ".no_exist" / revision / name
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(root / "hub"))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    return snapshot
