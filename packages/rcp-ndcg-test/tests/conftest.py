"""Shared fixtures: the fixture recipes' root and the two test-local fakes, registered per test module."""

from __future__ import annotations

from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
FIXTURES = TESTS / "fixtures"
RECIPES = FIXTURES / "recipes"
CASES = FIXTURES / "cases"


@pytest.fixture
def fixture_recipes_root() -> Path:
    """The tests' fixture recipes root (fake-rerank, fake-pool; the packaged fake-embed ships in src)."""
    return RECIPES
