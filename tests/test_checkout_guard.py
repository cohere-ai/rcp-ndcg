"""The checkout guard: the tree scan and the session guard behind the fixture that fails a leaking run."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._checkout import UNTRACKED_DIRS, checkout_guard, entries


def test_a_new_file_and_an_empty_directory_are_entries(tmp_path: Path) -> None:
    """The case the guard exists for: an empty ``logs/slurm/`` is invisible to ``git status`` but not here."""
    before = entries(tmp_path)
    (tmp_path / "logs" / "slurm").mkdir(parents=True)
    (tmp_path / "stray.txt").write_text("x")
    added = entries(tmp_path) - before
    assert added == {"logs", "logs/slurm", "stray.txt"}


def test_cache_and_environment_directories_are_skipped(tmp_path: Path) -> None:
    """A tool writes ``__pycache__`` and the caches on purpose; the guard must not report them."""
    names = ("__pycache__", ".git", ".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".basedpyright")
    assert set(names) <= UNTRACKED_DIRS
    for name in names:
        (tmp_path / name / "inner").mkdir(parents=True)
        (tmp_path / name / "inner" / "file").write_text("x")
    assert entries(tmp_path) == set()


def test_removing_an_entry_leaves_the_difference_empty(tmp_path: Path) -> None:
    """The guard compares ``after - before``: a path that existed before the session is not a finding."""
    (tmp_path / "logs").mkdir()
    before = entries(tmp_path)
    (tmp_path / "logs").rmdir()
    assert entries(tmp_path) - before == set()


def test_the_guard_fails_a_leaking_run_and_names_the_paths(tmp_path: Path) -> None:
    """The comparison itself, not only the scan: a guard that compares nothing must fail this test."""
    with pytest.raises(AssertionError, match="logs/slurm, stray.txt"):
        with checkout_guard(tmp_path):
            (tmp_path / "logs" / "slurm").mkdir(parents=True)
            (tmp_path / "stray.txt").write_text("x")


def test_the_guard_passes_a_clean_run(tmp_path: Path) -> None:
    # A directory, not a file: on the network-backed filesystem this lane's tmp lives on, unlinking an open file
    # can leave an NFS ``.nfs*`` placeholder, which is a real leftover the guard must report.
    with checkout_guard(tmp_path):
        (tmp_path / "inside").mkdir()
        (tmp_path / "inside").rmdir()


def test_the_guard_uses_the_given_baseline(tmp_path: Path) -> None:
    """The session baseline comes from ``pytest_sessionstart`` (before collection): a file written between the
    baseline and the guard's entry -- an import-time leak -- must still be a finding, not a new entry state."""
    baseline = entries(tmp_path)
    (tmp_path / "import_time.txt").write_text("x")
    with pytest.raises(AssertionError, match="import_time.txt"):
        with checkout_guard(tmp_path, before=baseline):
            pass
