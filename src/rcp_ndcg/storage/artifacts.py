"""Artifact provenance: a file's content hash, and the code version that produced it (for the run manifest)."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from rcp_ndcg import storage

if TYPE_CHECKING:
    from rcp_ndcg.runs.layout import RunLayout


class ArtifactRef(BaseModel):
    """A file a stage read or wrote.

    The hash is what makes resume trustworthy: a stage can be skipped only if
    its inputs are byte-identical to what the recorded output was produced
    from.  Comparing paths and mtimes would silently reuse stale work.
    """

    model_config = ConfigDict(extra="forbid")

    path: str
    """Run-relative where possible, so a run directory can be moved or copied."""

    sha256: str


class CodeVersion(BaseModel):
    """Where the code came from, for reproduction."""

    model_config = ConfigDict(extra="forbid")

    package_version: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    python_version: str
    platform: str


def artifact_ref(uri: str | Path, *, layout: RunLayout | None = None) -> ArtifactRef:
    """Describe a file: its run-relative path and content hash.

    Hashing streams the file, so a multi-GB observations JSONL costs memory
    proportional to the chunk size rather than the file.
    """
    uri = str(uri)
    digest = hashlib.sha256()
    with storage.open_path(uri, "rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return ArtifactRef(path=layout.relative(uri) if layout else uri, sha256=digest.hexdigest())


def code_version() -> CodeVersion:
    """Package version, git commit if we are in a checkout, interpreter, platform."""
    from rcp_ndcg import __version__

    commit, dirty = _git_state()
    return CodeVersion(
        package_version=__version__,
        git_commit=commit,
        git_dirty=dirty,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )


def _git_state() -> tuple[str | None, bool | None]:
    """Current commit and whether the tree is dirty, or ``(None, None)``.

    A dirty tree is recorded rather than rejected: it is the normal state
    during development, and a run that cannot say which commit it came from is
    better than a run that refuses to start.
    """
    package_dir = Path(__file__).resolve().parents[1]
    try:
        toplevel = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=package_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        # Only this package's own checkout counts: an installed copy inside some other repository (a venv in a
        # user's project) must not record that repository's commit.
        if Path(toplevel).resolve() / "src" / "rcp_ndcg" != package_dir:
            return None, None
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=package_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=package_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return commit or None, bool(status)
    except (OSError, subprocess.SubprocessError):
        return None, None


__all__ = ["ArtifactRef", "CodeVersion", "artifact_ref", "code_version"]
