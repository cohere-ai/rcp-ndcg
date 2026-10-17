"""The rcp-ndcg-test copy of the checkout guard: the same pins as the root suite's ``tests/test_checkout_guard.py``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import _checkout as _checkout_module
from tests._checkout import UNTRACKED_DIRS, checkout_guard, entries


def test_the_tested_guard_is_this_suites_copy() -> None:
    """Both distributions carry a copy; a shadowed import must not let the other copy's tests pass for this one."""
    assert Path(_checkout_module.__file__).parent == Path(__file__).parent


def test_a_new_file_and_an_empty_directory_are_entries(tmp_path: Path) -> None:
    before = entries(tmp_path)
    (tmp_path / "logs" / "slurm").mkdir(parents=True)
    (tmp_path / "stray.txt").write_text("x")
    assert entries(tmp_path) - before == {"logs", "logs/slurm", "stray.txt"}


def test_cache_and_environment_directories_are_skipped(tmp_path: Path) -> None:
    names = ("__pycache__", ".git", ".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".basedpyright")
    assert set(names) <= UNTRACKED_DIRS
    for name in names:
        (tmp_path / name / "inner").mkdir(parents=True)
        (tmp_path / name / "inner" / "file").write_text("x")
    assert entries(tmp_path) == set()


def test_the_guard_fails_a_leaking_run_and_names_the_paths(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="logs/slurm, stray.txt"):
        with checkout_guard(tmp_path):
            (tmp_path / "logs" / "slurm").mkdir(parents=True)
            (tmp_path / "stray.txt").write_text("x")


def test_the_guard_uses_the_given_baseline(tmp_path: Path) -> None:
    baseline = entries(tmp_path)
    (tmp_path / "import_time.txt").write_text("x")
    with pytest.raises(AssertionError, match="import_time.txt"):
        with checkout_guard(tmp_path, before=baseline):
            pass
