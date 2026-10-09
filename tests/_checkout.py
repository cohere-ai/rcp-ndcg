"""The checkout guard's tree scan: every file and directory a test run must not leave behind.

``tests/conftest.py`` snapshots :func:`entries` at session start and compares it at session end, so a test
that writes into the checkout (instead of ``tmp_path``) fails the run. Directories are entries themselves:
an empty ``logs/slurm/`` is the case the guard was written for, and ``git status`` never shows an empty
directory.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["UNTRACKED_DIRS", "entries"]

#: Directories the scan never descends into: not the checkout's tracked content (environments, caches,
#: bytecode), and written by a tool on purpose.
UNTRACKED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".basedpyright",
        ".ipynb_checkpoints",
        "node_modules",
    }
)


def entries(root: Path) -> set[str]:
    """Every file and directory under ``root`` as a relative path, skipping :data:`UNTRACKED_DIRS`."""
    found: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in UNTRACKED_DIRS]
        for name in dirnames:
            found.add(str(Path(dirpath, name).relative_to(root)))
        found.update(str(Path(dirpath, name).relative_to(root)) for name in filenames)
    return found
