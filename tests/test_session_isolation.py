"""The test session is hermetic: caches go to the session scratch, never to an exported cache directory."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.conftest import ROOT

PROBE = """\
import os
from pathlib import Path


def test_cache_dirs_are_the_session_scratch() -> None:
    cache = Path(os.environ["RCP_NDCG_CACHE_DIR"])
    assert "rcp-tests-" in cache.as_posix(), cache
    assert "RCP_NDCG_MEDIA_CACHE" not in os.environ
    assert "RCP_NDCG_RUNS_DIR" not in os.environ


def test_a_bare_socket_connect_is_refused_by_the_guard() -> None:
    import socket

    import pytest

    sock = socket.socket()
    try:
        with pytest.raises(OSError, match="tests use no network"):
            sock.connect(("rcp-ndcg.invalid", 80))
        # connect_ex returns an errno instead of raising, so the guard must refuse it too: without its wrapper
        # the call answers -2 (libc resolved the name in the C call) and the raises block below fails.
        with pytest.raises(OSError, match="tests use no network"):
            sock.connect_ex(("rcp-ndcg.invalid", 80))
    finally:
        sock.close()
"""


#: The socket probe's failure is the guard's own bookkeeping (it records every host it refused and fails the
#: session at teardown): the message names the guard and the host, where a libc resolution names neither.
SOCKET_REFUSAL = "tests use no network (tried to resolve: ['rcp-ndcg.invalid'])"
#: A guard hole in the probe's own body reads as one of pytest's failure lines, never as the teardown record:
#: a missing ``connect``/``connect_ex`` wrapper lets libc resolve the name, whose ``gaierror`` (or the errno it
#: returns) does not carry the guard's message.
PROBE_BODY_FAILURE = ("Failed: DID NOT RAISE", "Regex pattern did not match")


def test_an_exported_cache_dir_does_not_escape_the_session_scratch(tmp_path: Path) -> None:
    """AGENTS.md: tests write only to tmp_path; a developer's RCP_NDCG_CACHE_DIR must not win.

    The probe runs in a subprocess with the three cache variables exported, and with this suite's
    ``tests/conftest.py`` loaded as a plugin, so it sees exactly what a test process sees when the
    developer's shell exports a shared cache directory (the documented container setup).  The same session
    also refuses a bare ``socket.socket().connect`` and ``connect_ex`` (CPython resolves that name inside the
    C call, not through ``getaddrinfo``): the probe's refusal names the guard, never a libc resolution, and a
    missing wrapper shows up as a failure inside the probe's body rather than as the guard's record.
    """
    probe = tmp_path / "test_probe.py"
    probe.write_text(PROBE, encoding="utf-8")
    env = {
        **os.environ,
        "RCP_NDCG_CACHE_DIR": str(tmp_path / "outer-cache"),
        "RCP_NDCG_MEDIA_CACHE": str(tmp_path / "outer-media"),
        "RCP_NDCG_RUNS_DIR": str(tmp_path / "outer-runs"),
    }
    command = [sys.executable, "-m", "pytest", str(probe), "-q", "-p", "no:cacheprovider", "-p", "tests.conftest"]
    result = subprocess.run(
        [*command, "-k", "cache_dirs_are_the_session_scratch"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    refused = subprocess.run(
        [*command, "-k", "bare_socket_connect"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert SOCKET_REFUSAL in refused.stdout, refused.stdout + refused.stderr
    assert not any(failure in refused.stdout for failure in PROBE_BODY_FAILURE), refused.stdout + refused.stderr
