"""The checkout guard: the tree scan behind the session fixture that fails a run leaving files behind."""

from __future__ import annotations

from pathlib import Path

from tests._checkout import UNTRACKED_DIRS, entries


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
    """The fixture compares ``after - before``: a path that existed before the session is not a finding."""
    (tmp_path / "logs").mkdir()
    before = entries(tmp_path)
    (tmp_path / "logs").rmdir()
    assert entries(tmp_path) - before == set()
