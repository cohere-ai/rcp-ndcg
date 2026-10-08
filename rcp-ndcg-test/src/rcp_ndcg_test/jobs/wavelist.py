"""The wave list and its recipes: one parse of the ids, one tolerant load (one home per concept).

A wave names its recipes as ``@file`` (one id per line, ``#`` comments) or ``a,b``; both
:mod:`rcp_ndcg_test.jobs.plugins` and :mod:`rcp_ndcg_test.jobs.run_wave` read such a list and load its
recipes.  The load is tolerant on purpose ("one failing recipe never stops the wave", end to end): a
recipe that fails validation is returned as a failure with the validation message, so the collector
reports and skips it and the wave report marks it failed -- it never kills the job that merely lists it.
"""

from __future__ import annotations

from pathlib import Path

from rcp_ndcg_vllm.recipe import Recipe, default_recipes_root, load_recipe

from rcp_ndcg_test.errors import HarnessError, RecipeError

__all__ = ["load_wave", "parse_ids"]


def parse_ids(value: str) -> list[str]:
    """Parse a wave-list argument into recipe ids.

    Inputs: ``@file`` (one id per line, ``#`` comments, blanks ignored) or a comma-separated list.
    Outputs: the ids in order; empty input gives an empty list (the caller decides what "every recipe"
    means).  Raises :class:`HarnessError` when ``@file`` names no file.  Units: none.
    """
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.is_file():
            raise HarnessError(f"no wave list at {path}")
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return [item.strip() for item in value.split(",") if item.strip()]


def load_wave(recipe_ids: list[str], recipes_root: str | Path | None = None) -> tuple[list[Recipe], dict[str, str]]:
    """Load a wave's recipes tolerantly.

    Inputs: the recipe ids (directories under ``recipes_root``; empty means every recipe directory
    there) and the recipe root (default: the package's ``recipes/``).  Outputs: ``(recipes, failed)`` --
    the valid recipes in wave order, and the invalid ones' ``recipe id -> validation message`` (the
    caller reports and records them; the wave marks them failed).  An unknown id is a failed entry with
    its load message.  Raises :class:`HarnessError` for a wave request no recipe can answer: a missing
    recipe root.  Units: none.
    """
    root = Path(recipes_root) if recipes_root is not None else default_recipes_root()
    if not root.is_dir():
        raise HarnessError(f"no recipe root at {root}")
    ids = (
        list(recipe_ids)
        if recipe_ids
        else sorted(
            directory.name
            for directory in root.iterdir()
            if directory.is_dir() and (directory / "recipe.yaml").is_file()
        )
    )
    recipes: list[Recipe] = []
    failed: dict[str, str] = {}
    for recipe_id in ids:
        try:
            recipes.append(load_recipe(root / recipe_id))
        except RecipeError as error:
            failed[recipe_id] = str(error)
    return recipes, failed
