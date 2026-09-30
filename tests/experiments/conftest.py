"""The ``experiments/`` scripts are not a package: the fixtures import them by name for one test at a time."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[2] / "experiments"


@pytest.fixture
def experiment(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], ModuleType]:
    """``experiment("human_study")`` imports ``experiments/human_study.py``; the path and the modules are undone."""
    monkeypatch.syspath_prepend(str(EXPERIMENTS))
    before = set(sys.modules)
    yield importlib.import_module
    for name in set(sys.modules) - before:
        del sys.modules[name]


@pytest.fixture
def gain_variants(experiment: Callable[[str], ModuleType]) -> ModuleType:
    return experiment("gain_variants")
