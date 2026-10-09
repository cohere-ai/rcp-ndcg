"""``_served.fetch_tokenizer`` in the shared cache: concurrent workers, one pinned file.

The recipe job runs with ``-n 4`` against one ``RCP_NDCG_VLLM_TOKENIZER_CACHE``, so two workers can fetch
the same tokenizer file at once. The cache must end up holding the pinned bytes whatever the interleaving:
a download that fails the pin must never clobber a pinned file another worker wrote, and no reader may ever
see a partial one. These tests drive the race with a fake ``urlopen`` -- no network.
"""

from __future__ import annotations

import hashlib
import threading
import urllib.request
from pathlib import Path

import pytest

from ._served import fetch_tokenizer

PINNED = b"the pinned tokenizer bytes"
SHA256 = hashlib.sha256(PINNED).hexdigest()
CORRUPT = b"a truncated or mangled download"
URL = "https://example.invalid/model/resolve/deadbeef/tokenizer.json"
NAME = "model/tokenizer.json"


class _Response:
    """A ``urlopen`` stand-in: ``read`` announces itself, waits for the gate, then returns the bytes."""

    def __init__(self, data: bytes, gate: tuple[threading.Event, threading.Event] | None = None) -> None:
        self._data = data
        self._gate = gate

    def read(self) -> bytes:
        if self._gate is not None:
            started, release = self._gate
            started.set()
            assert release.wait(timeout=30), "the race never resolved"
        return self._data

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def _fetch(tmp_path: Path) -> Path:
    return fetch_tokenizer(URL, NAME, tmp_path, sha256=SHA256)


def test_a_download_that_fails_the_pin_keeps_the_pinned_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Worker B starts (cache empty), worker A writes the pinned file, then B's corrupt bytes arrive.

    The cache must still hold the pinned file when B returns, and B must return it instead of failing:
    with a plain ``write_bytes`` B's late download overwrote A's file and B's own pin assert fired (the
    observed CI flake).
    """
    monkeypatch.setenv("RCP_NDCG_VLLM_TOKENIZER_CACHE", str(tmp_path / "cache"))
    started, release = threading.Event(), threading.Event()
    responses = [_Response(CORRUPT, gate=(started, release)), _Response(PINNED)]

    def fake_urlopen(url: str, timeout: int | None = None) -> _Response:
        return responses.pop(0)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    failures: list[BaseException] = []

    def late_worker() -> None:
        try:
            _fetch(tmp_path)
        except BaseException as error:  # noqa: BLE001 - the test reports whatever the worker raised
            failures.append(error)

    worker = threading.Thread(target=late_worker)
    worker.start()
    assert started.wait(timeout=30), "the late worker never started its download"
    pinned = _fetch(tmp_path)  # worker A: the pinned file lands first
    assert pinned.read_bytes() == PINNED
    release.set()
    worker.join(timeout=30)
    assert not worker.is_alive()
    assert not failures, failures
    assert pinned.read_bytes() == PINNED, "the late corrupt download clobbered the pinned file"


def test_a_fetch_leaves_no_partial_file_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The write is atomic: the cache directory holds the target and nothing else after a fetch."""
    monkeypatch.setenv("RCP_NDCG_VLLM_TOKENIZER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: _Response(PINNED))
    target = _fetch(tmp_path)
    assert target.read_bytes() == PINNED
    assert [path.name for path in target.parent.iterdir()] == [target.name]


def test_a_reader_never_sees_a_partial_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache path changes only by rename: while a fetch writes its temporary file, a reader sees either
    nothing or the complete pinned file -- never a half-written one (a plain write_bytes fails this)."""
    monkeypatch.setenv("RCP_NDCG_VLLM_TOKENIZER_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: _Response(PINNED))
    target = tmp_path / "cache" / NAME
    half = len(PINNED) // 2
    mid, release = threading.Event(), threading.Event()

    def half_write(path: Path, data: bytes) -> int:
        with path.open("wb") as handle:
            handle.write(data[:half])
            handle.flush()
            mid.set()
            assert release.wait(timeout=30), "the reader never looked"
            handle.write(data[half:])
        return len(data)

    monkeypatch.setattr(Path, "write_bytes", half_write)
    seen: list[bytes | None] = []

    def reader() -> None:
        assert mid.wait(timeout=30), "the writer never reached the middle of its write"
        try:
            seen.append(target.read_bytes())
        except FileNotFoundError:
            seen.append(None)
        release.set()

    thread = threading.Thread(target=reader)
    thread.start()
    fetched = _fetch(tmp_path)
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert seen in ([None], [PINNED]), f"a reader saw a partial download: {seen!r}"
    assert fetched.read_bytes() == PINNED
