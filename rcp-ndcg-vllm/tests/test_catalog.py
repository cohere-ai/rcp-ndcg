"""The catalog table in ``README.md`` against ``iter_recipes()``: one pin, so a row cannot drift.

The README's "The recipes" table is the rendered catalog (``docs/reference/recipes.md`` delegates to it).
Every row is checked against the resolved recipe: family, id, model, role, input, the MRL cell, whether the
recipe needs its model plugin, the reference cell and the status state. The test is offline (package data and
PyYAML only).
"""

from __future__ import annotations

import re
from pathlib import Path

from rcp_ndcg_vllm.recipe import iter_families, iter_recipes

README = Path(__file__).resolve().parents[1] / "README.md"
RECIPES_ROOT = Path(__file__).resolve().parents[1] / "src" / "rcp_ndcg_vllm" / "recipes"

_ROW = re.compile(
    r"^\|\s*`([^`]+)`\s*\|\s*`([^`]+)`\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|"
    r"\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$"
)


def _catalog_rows() -> dict[str, tuple[str, str, str, str, str, str, str, str]]:
    rows: dict[str, tuple[str, str, str, str, str, str, str, str]] = {}
    for line in README.read_text(encoding="utf-8").splitlines():
        match = _ROW.match(line)
        if match is None:
            continue
        family, recipe_id, model, role, inputs, mrl, plugin, reference, status = (
            group.strip() for group in match.groups()
        )
        rows[recipe_id] = (family, model, role, inputs, mrl, plugin, reference, status)
    return rows


def _reference_header(family_id: str) -> str:
    """The first line of a family's ``reference.py``: the reference implementation names its own code path there."""
    return (RECIPES_ROOT / family_id / "reference.py").read_text(encoding="utf-8").splitlines()[0]


def test_the_readme_catalog_matches_iter_recipes() -> None:
    """Every variant -- retrieval and judge -- has exactly one row, and every row names its resolved facts."""
    rows = _catalog_rows()
    recipes = {recipe.id: recipe for recipe in iter_recipes()}
    assert set(rows) == set(recipes), (
        f"catalog rows out of step with iter_recipes(): missing {sorted(set(recipes) - set(rows))}, "
        f"extra {sorted(set(rows) - set(recipes))}"
    )
    families = {family.id: family for family in iter_families()}
    family_of = {variant.id: family.id for family in families.values() for variant in family.variants}
    for recipe_id, (family_id, model, role, inputs, mrl, plugin, reference, status) in rows.items():
        recipe = recipes[recipe_id]
        assert family_id == family_of[recipe_id]
        assert model == recipe.model, f"{recipe_id}: README model {model!r} != recipe {recipe.model!r}"
        assert role == recipe.role, f"{recipe_id}: README role {role!r} != recipe {recipe.role!r}"
        assert inputs == ", ".join(recipe.input), f"{recipe_id}: README input {inputs!r} != {recipe.input}"
        assert (plugin != "—") == (recipe.serve.plugin is not None), (
            f"{recipe_id}: README plugin {plugin!r} does not match serve.plugin {recipe.serve.plugin!r}"
        )
        assert status == recipe.status.state, f"{recipe_id}: README status {status!r} != {recipe.status.state!r}"
        # The reference cell: a judge recipe carries no equivalence reference ("—"); a retrieval family's
        # reference.py names its own code path in its header -- the paper's code (ported from experiments/paper/)
        # or the model card's published usage.
        if role == "judge":
            assert reference == "—", f"{recipe_id}: a judge recipe's reference cell is {reference!r}"
        else:
            assert reference in {"paper", "card"}, f"{recipe_id}: unknown reference cell {reference!r}"
            from_the_paper = "the paper" in _reference_header(family_id)
            assert (reference == "paper") == from_the_paper, (
                f"{recipe_id}: reference {reference!r} does not match the family reference's header"
            )
        # The MRL cell is free-form prose (a set, a range, "none" or "—"); it must name the declared kind
        # when the recipe declares one, and the declared dims must all appear.
        kind = recipe.client.get("mrl_kind")
        if kind is not None:
            assert kind in mrl, f"{recipe_id}: README MRL {mrl!r} does not name mrl_kind {kind!r}"
            dims = tuple(recipe.client.get("mrl_dims") or ()) + tuple(recipe.client.get("mrl_range") or ())
            for value in dims:
                assert str(value) in mrl, f"{recipe_id}: README MRL {mrl!r} omits {value}"
        else:
            assert mrl, f"{recipe_id}: README MRL cell is empty"
