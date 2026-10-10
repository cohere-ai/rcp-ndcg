"""The catalog table in ``README.md`` against ``iter_recipes()``: one pin, so a row cannot drift.

The README's "The recipes" table is the rendered catalog (``docs/reference/recipes.md`` delegates to it).
Every row is checked against the resolved recipe: family, id, model, role, input, whether the recipe needs
its model plugin, and the status state. The test is offline (package data and PyYAML only).
"""

from __future__ import annotations

import re
from pathlib import Path

from rcp_ndcg_vllm.recipe import iter_families, iter_recipes

README = Path(__file__).resolve().parents[1] / "README.md"

_ROW = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
    r"\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$"
)


def _catalog_rows() -> dict[str, tuple[str, str, str, str, str, str]]:
    rows: dict[str, tuple[str, str, str, str, str, str]] = {}
    for line in README.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line)
        if match is None:
            continue
        family, recipe_id, model, role, inputs, plugin, status = (group.strip() for group in match.groups())
        rows[recipe_id] = (family, model, role, inputs, plugin, status)
    return rows


def test_the_readme_catalog_matches_iter_recipes() -> None:
    """Every variant has exactly one row and every row names its resolved recipe's facts."""
    rows = _catalog_rows()
    recipes = {recipe.id: recipe for recipe in iter_recipes()}
    assert set(rows) == set(recipes), (
        f"catalog rows out of step with iter_recipes(): missing {sorted(set(recipes) - set(rows))}, "
        f"extra {sorted(set(rows) - set(recipes))}"
    )
    families = {family.id: family for family in iter_families()}
    for recipe_id, (family_id, model, role, inputs, plugin, status) in rows.items():
        recipe = recipes[recipe_id]
        assert family_id == next(f.id for f in families.values() if recipe_id in {v.id for v in f.variants})
        assert model == recipe.model, f"{recipe_id}: README model {model!r} != recipe {recipe.model!r}"
        assert role == recipe.role, f"{recipe_id}: README role {role!r} != recipe {recipe.role!r}"
        assert inputs == ", ".join(recipe.input), f"{recipe_id}: README input {inputs!r} != {recipe.input}"
        assert (plugin != "—") == (recipe.serve.plugin is not None), (
            f"{recipe_id}: README plugin {plugin!r} does not match serve.plugin {recipe.serve.plugin!r}"
        )
        assert status == recipe.status.state, f"{recipe_id}: README status {status!r} != {recipe.status.state!r}"
