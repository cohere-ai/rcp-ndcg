"""The product suite's hook on the ``rcp-ndcg-test`` case tree: a malformed case fails CI.

The case files live at ``packages/rcp-ndcg-test/cases/<recipe-id>/`` (the cases lanes write them). Every
load applies the file-level rules of the case format; for a case directory whose recipe exists at the
serving-recipes package's ``recipes/`` root, the recipe-backed rules run too (role, modality, template
shapes, strata coverage). The measured token lengths run with the network-marked test below: a recipe
whose tokenizer lives on the Hub cannot be loaded offline, and the check records itself as skipped
instead of passing silently.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "packages" / "rcp-ndcg-test" / "cases"


def test_every_case_file_validates() -> None:
    """Every case in the repository loads; the recipe-backed rules run wherever a recipe exists."""
    from rcp_ndcg_test.cases import load_cases
    from rcp_ndcg_vllm.recipe import default_recipes_root

    bundle = load_cases(CASES, recipes_root=default_recipes_root(), check_lengths=False)
    directories = {directory.name for directory in CASES.iterdir() if directory.is_dir()}
    # A case directory whose recipe has not merged yet is recorded, never silently dropped.
    assert set(bundle.recipes) | set(bundle.recipes_missing) == directories


def test_every_recipe_backed_case_validates_with_its_recipe() -> None:
    """For every recipe that both exists and has cases: the full recipe-backed validation, no lengths."""
    from rcp_ndcg_test.cases import load_cases
    from rcp_ndcg_vllm.recipe import iter_recipes

    for recipe in iter_recipes():
        if not (CASES / recipe.id).is_dir():
            continue
        bundle = load_cases(CASES, recipe, check_lengths=False)
        assert bundle.cases, f"recipe {recipe.id} has a case directory that loaded no cases"


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("RCP_NDCG_NETWORK_TESTS"), reason="set RCP_NDCG_NETWORK_TESTS=1 (HF Hub)")
def test_every_long_input_measures_its_stratum() -> None:
    """The long inputs measure against the recipe's max_tokens, with the product's tokenizer.

    Recipes whose tokenizer lives on the Hub need the cache or the network, so this runs only in the
    network-marked run; the offline tests above still enforce the strata coverage.
    """
    from rcp_ndcg_test.cases import load_cases
    from rcp_ndcg_vllm.recipe import iter_recipes

    for recipe in iter_recipes():
        if not (CASES / recipe.id).is_dir():
            continue
        bundle = load_cases(CASES, recipe)
        assert bundle.skipped_checks == (), f"recipe {recipe.id}: every length check must have run"


def test_the_shipped_fixture_suite_runs_the_plugin_end_to_end() -> None:
    """The pytest plugin's params run the packaged fixture suite (offline, against the shipped fake)."""
    from rcp_ndcg_test.fakes import fixture_path
    from rcp_ndcg_test.plugin import conformance_params

    params = conformance_params("fake", recipes_root=fixture_path("recipes"), cases_root=fixture_path("cases"))
    assert {param.id for param in params} >= {"fake-embed/short-single", "fake-embed/long-over"}
    for param in params:
        result = param.values[0].run()
        # The pending fixture case skips inside assert_passes; the run itself must pass or skip.
        assert result.passed or result.skipped is not None, result.detail
        assert not result.failed
