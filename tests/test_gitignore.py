"""The ignore rules match only the repository root's scratch directories."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout"
)


def _ignored(path: str) -> bool:
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=ROOT, check=False).returncode == 0


@pytest.mark.parametrize("name", ["logs", ".cache", "results", "build", "dist"])
def test_scratch_directories_are_ignored_only_at_the_root(name: str) -> None:
    """Unanchored patterns also dropped same-named package directories from ``git archive``."""
    assert _ignored(f"{name}/anything.txt")
    assert not _ignored(f"src/rcp_ndcg/{name}/module.py")
