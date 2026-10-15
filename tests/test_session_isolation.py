"""The test session is hermetic: caches go to the session scratch, never to an exported cache directory."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tests.conftest import ROOT

PROBE = '''\
import os
from pathlib import Path


def test_cache_dirs_are_the_session_scratch() -> None:
    cache = Path(os.environ["RCP_NDCG_CACHE_DIR"])
    assert "rcp-ndcg-tests-" in cache.as_posix(), cache
    assert "RCP_NDCG_MEDIA_CACHE" not in os.environ
    assert "RCP_NDCG_RUNS_DIR" not in os.environ
'''


def test_an_exported_cache_dir_does_not_escape_the_session_scratch(tmp_path: Path) -> None:
    """AGENTS.md: tests write only to tmp_path; a developer's RCP_NDCG_CACHE_DIR must not win.

    The probe runs in a subprocess with the three cache variables exported, and with this suite's
    ``tests/conftest.py`` loaded as a plugin, so it sees exactly what a test process sees when the
    developer's shell exports a shared cache directory (the documented container setup).
    """
    probe = tmp_path / "test_probe.py"
    probe.write_text(PROBE, encoding="utf-8")
    env = {
        **os.environ,
        "RCP_NDCG_CACHE_DIR": str(tmp_path / "outer-cache"),
        "RCP_NDCG_MEDIA_CACHE": str(tmp_path / "outer-media"),
        "RCP_NDCG_RUNS_DIR": str(tmp_path / "outer-runs"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(probe), "-q", "-p", "no:cacheprovider", "-p", "tests.conftest"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
