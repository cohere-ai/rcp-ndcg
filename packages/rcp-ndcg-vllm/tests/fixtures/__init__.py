"""Test fixtures for the recipe schema: valid recipes per role, written as a recipe lane would write them."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

FIXTURES = Path(__file__).parent
REV = "0123456789abcdef0123456789abcdef01234567"
OTHER_REV = "fedcba9876543210fedcba9876543210fedcba98"


def recipe_dirs() -> dict[str, Path]:
    """Every fixture recipe directory, by id."""
    root = FIXTURES / "recipes"
    return {p.name: p for p in sorted(root.iterdir()) if p.is_dir()}


def recipe_data(recipe_id: str) -> dict[str, Any]:
    """The raw YAML mapping of one fixture recipe (a deep copy, safe to mutate in a test)."""
    data: Any = yaml.safe_load((recipe_dirs()[recipe_id] / "recipe.yaml").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return copy.deepcopy(data)
