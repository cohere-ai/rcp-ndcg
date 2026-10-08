"""Typed errors of the package.

The package raises typed errors; it never prints. Case-format problems are
:class:`CaseError`, conformance-run problems :class:`ConformanceError`.
"""

from __future__ import annotations

__all__ = ["CaseError", "ConformanceError"]


class CaseError(ValueError):
    """A case file (or the strata grid of a recipe's case directory) breaks a rule of the case format.

    Carries the file path (or the recipe id, for a coverage gap) and a message that names the offending
    field and the value that broke the rule, so a case author can fix the YAML without reading the
    model's source.
    """


class ConformanceError(RuntimeError):
    """The conformance runner could not complete a case (no engine for the recipe at the target, a recipe
    whose client config the role client refuses). The message names the case or recipe and, where one
    exists, the fix."""
