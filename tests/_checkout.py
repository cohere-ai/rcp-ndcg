"""The checkout guard: the scan and the comparison that fail a test run leaving files behind.

``tests/conftest.py`` takes a baseline of :func:`entries` in ``pytest_sessionstart`` (before collection, so an
import-time leak is caught too) and runs the test session under :func:`checkout_guard`, which compares the tree
again at session end. Directories are entries themselves: an empty ``logs/slurm/`` is the case the guard was
written for, and ``git status`` never shows an empty directory.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Set
from contextlib import contextmanager
from pathlib import Path

__all__ = ["UNTRACKED_DIRS", "checkout_guard", "entries"]

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


@contextmanager
def checkout_guard(root: Path, *, before: Set[str] | None = None) -> Iterator[None]:
    """Fail on exit when ``root`` holds a file or directory that was not there at entry.

    ``before`` is the baseline to compare against (the session-start snapshot); without it the entry state is
    used. The failure names every added path, so the offending test run is diagnosable from the error alone.
    """
    baseline = entries(root) if before is None else set(before)
    yield
    added = sorted(entries(root) - baseline)
    if added:
        raise AssertionError(
            "the tests left new files or directories in the checkout (they write under tmp_path): " + ", ".join(added)
        )
