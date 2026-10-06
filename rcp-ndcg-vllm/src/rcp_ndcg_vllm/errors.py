"""Typed errors of the package.

The package raises typed errors; it never prints.  Recipe problems are
:class:`RecipeError`, engine and harness problems :class:`HarnessError`.
"""

from __future__ import annotations

__all__ = ["HarnessError", "RecipeError"]


class RecipeError(ValueError):
    """A recipe failed to load or validate.

    Carries the recipe directory or file path and a message that names the offending field and the value that
    broke the rule, so a recipe author can fix the YAML without reading the schema's source.
    """


class HarnessError(RuntimeError):
    """The equivalence harness or the wave runner could not complete a step (a missing extra, a bad reference
    entry, an engine that never came up).  The message names the step and, where one exists, the fix."""
