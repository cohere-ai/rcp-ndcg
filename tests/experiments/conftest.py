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
    """``experiment("human_study")`` imports ``experiments/human_study.py``; the path and the module are undone.

    Only this folder's own modules are removed from ``sys.modules`` afterwards: they are plain scripts with
    import-time state, while a third-party module (scipy, or numpy's C-extension submodules) cannot be re-imported
    a second time in the same process.
    """
    monkeypatch.syspath_prepend(str(EXPERIMENTS))
    before = set(sys.modules)
    yield importlib.import_module
    for name in set(sys.modules) - before:
        module = sys.modules[name]
        origin = getattr(module, "__file__", None)
        if origin is not None and Path(origin).resolve().is_relative_to(EXPERIMENTS):
            del sys.modules[name]


@pytest.fixture
def gain_variants(experiment: Callable[[str], ModuleType]) -> ModuleType:
    return experiment("gain_variants")
