"""Importing the package reads no ``.env``, needs no project root and writes nothing."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROBE = """
import os, rcp_ndcg, rcp_ndcg.cli.main, rcp_ndcg.support.paths as paths
paths.cache_dir(); paths.runs_dir()
print(os.environ.get("RCP_NDCG_PROBE"), rcp_ndcg.__version__)
"""


def test_import_in_a_clean_directory_creates_nothing_and_reads_no_env_file(tmp_path: Path) -> None:
    """``import rcp_ndcg`` once called ``dotenv.load_dotenv()`` and walked up from the cwd for a project root."""
    cwd = tmp_path / "cwd"
    home = tmp_path / "home"
    cwd.mkdir()
    home.mkdir()
    (cwd / ".env").write_text("RCP_NDCG_PROBE=loaded\n", encoding="utf-8")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("RCP_NDCG_") and key not in ("XDG_CACHE_HOME", "PYTHONDONTWRITEBYTECODE")
    }
    env.update(HOME=str(home), PYTHONDONTWRITEBYTECODE="1")

    out = subprocess.run([sys.executable, "-c", PROBE], cwd=cwd, env=env, capture_output=True, text=True, check=True)

    loaded, version = out.stdout.split()
    assert loaded == "None"
    assert version
    assert sorted(path.name for path in cwd.iterdir()) == [".env"]
    assert list(home.iterdir()) == []
