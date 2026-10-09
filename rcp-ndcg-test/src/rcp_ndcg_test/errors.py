"""Typed errors of the package.

The package raises typed errors; it never prints. Case-format problems are
:class:`CaseError`, conformance-run problems :class:`ConformanceError`, and the moved harness raises
:class:`HarnessError`. The recipe loader's :class:`RecipeError` keeps its home in
:mod:`rcp_ndcg_vllm.errors` and is re-exported here so the harness's callers take both from one place.
"""

from __future__ import annotations

from rcp_ndcg_vllm.errors import RecipeError as RecipeError

__all__ = ["CaseError", "ConformanceError", "EmulatorUnmodelledError", "HarnessError", "RecipeError"]


class HarnessError(RuntimeError):
    """The equivalence harness, the recorder or the wave runner could not complete a step (a missing
    tokenizer, a bad reference entry, an engine that never came up). The message names the step and, where
    one exists, the fix."""


class EmulatorUnmodelledError(HarnessError):
    """A verified fake engine was asked for something its model layer does not model (a request shape it
    has no prompt derivation for, a media part it cannot key, a request field it does not render).

    Typed so the emulator can answer it as a marked wire refusal and ``from_corpus`` can skip-and-name the
    record it could not model -- never a bare ``ValueError`` and never a whole-corpus failure."""


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
