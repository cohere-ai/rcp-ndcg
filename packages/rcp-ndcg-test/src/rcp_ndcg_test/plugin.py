"""The pytest integration: the conformance suite as parametrised tests.

Opt-in (a plain import; no entry point, so installing the package never injects tests into another
suite). The one factory, :func:`conformance_params`, builds one :class:`pytest.param` per case that can
run under the requested target; the companion :class:`CaseRun` runs it and asserts. A suite writes:

.. code-block:: python

    import pytest
    from rcp_ndcg_test.plugin import conformance_params

    @pytest.mark.parametrize("run", conformance_params("fake"))
    def test_conformance(run):
        run.assert_passes()

A skipped case (``values: null``, pending the GPU wave) becomes ``pytest.skip`` with its reason; a
failed case an ``AssertionError`` with the worst delta; a pass keeps its detail on the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pytest

from .cases import Case, load_cases
from .conformance import CaseResult, run_case
from .errors import CaseError

if TYPE_CHECKING:
    from rcp_ndcg_vllm.recipe import Recipe

__all__ = ["CaseRun", "conformance_params"]


@dataclass(frozen=True)
class CaseRun:
    """One (recipe, case, target) triple, ready to run.

    Build through :func:`conformance_params`; ``run``/``assert_passes`` is what a parametrised test
    body calls.

    Attributes:
        recipe: The loaded :class:`rcp_ndcg_vllm.recipe.Recipe` the case belongs to.
        case: The validated case.
        target: What answers (``engine`` or ``fake``).
        base_url: The engine's URL (an engine target); ``None`` for a fake.
        fake_engine: The fake engine (a fake target); ``None`` for an engine target.
    """

    recipe: Recipe
    case: Case
    target: Literal["engine", "fake"]
    base_url: str | None
    fake_engine: Any | None

    def run(self) -> CaseResult:
        """Send the case through the product's role client and return the typed result."""
        return run_case(
            self.recipe,
            self.case,
            target=self.target,
            base_url=self.base_url,
            fake_engine=self.fake_engine,
        )

    def assert_passes(self) -> CaseResult:
        """The test body's assertion: a skip becomes ``pytest.skip``, a failure an ``AssertionError``."""
        result = self.run()
        if result.skipped is not None:
            pytest.skip(f"{result.case_id}: {result.skipped}")
        if result.failed:
            raise AssertionError(f"{result.case_id}: {result.detail}")
        return result


def conformance_params(
    target: Literal["engine", "fake"] = "fake",
    *,
    recipes_root: str | Path | None = None,
    cases_root: str | Path | None = None,
    base_url: str | None = None,
    check_lengths: bool = False,
) -> list[Any]:
    """One :class:`pytest.param` per case that can run under ``target``, in a stable order.

    Inputs: the ``target`` (``fake`` resolves the recipe's fake through the registry; ``engine`` needs
    ``base_url``), where the recipe directories and the case directories live (the product's
    ``recipes/`` and the package's ``cases/`` when ``None``), and ``check_lengths`` for the load (the
    conformance run itself needs no tokenizer; a recipe whose tokenizer lives on the Hub does).

    Outputs: params whose id is ``<recipe>/<case-slug>`` and whose value is a :class:`CaseRun`. A case
    directory whose recipe has no case at the target is silently absent from the list (no fake
    registered, or no recipe at the recipes root yet): a suite that must not collect green against a
    half-merged tree asserts on :func:`rcp_ndcg_test.cases.load_cases`'s bundle instead.

    Raises:
        CaseError: the cases root or a case file does not validate, or an engine target has no
            ``base_url``.
    """
    import rcp_ndcg_test.fakes  # noqa: F401  # importing registers the shipped fixture fake

    if target == "engine" and base_url is None:
        raise CaseError("target 'engine' needs base_url (the live engine's URL)")
    bundle = load_cases(cases_root, recipes_root=recipes_root, check_lengths=check_lengths)
    params: list[Any] = []
    for case in bundle.cases:
        recipe = bundle.recipes.get(case.recipe)
        if recipe is None:
            continue  # no recipe (yet) at the recipes root: the root-suite validation test owns that gap
        engine = None
        if target == "fake":
            from .fakes import fake_engine_for

            engine = fake_engine_for(recipe.id)  # absent: the recipe runs against an engine or not at all
        params.append(
            pytest.param(
                CaseRun(recipe=recipe, case=case, target=target, base_url=base_url, fake_engine=engine),
                id=case.id,
            )
        )
    return params
