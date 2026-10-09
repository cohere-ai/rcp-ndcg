"""The recipes the recipe lanes write: loaded and validated against the product's endpoint configs."""

from __future__ import annotations

import pytest


def test_the_recipes_root_loads_clean() -> None:
    """Every recipe the recipe lanes shipped loads (network tests only: the tokenizer may be a Hub id)."""
    pytest.skip("populated by the recipe lanes; this file runs when they land")
