"""Typed errors of the package.

The package raises typed errors; it never prints.  Recipe problems are
:class:`RecipeError`. Harness problems moved with the harness to `rcp_ndcg_test.errors`.
"""

from __future__ import annotations

__all__ = ["RecipeError"]


class RecipeError(ValueError):
    """A recipe failed to load or validate.

    Carries the recipe directory or file path and a message that names the offending field and the value that
    broke the rule, so a recipe author can fix the YAML without reading the schema's source.
    """
