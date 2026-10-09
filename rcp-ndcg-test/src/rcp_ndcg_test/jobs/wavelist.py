"""The wave list and its recipes: one parse of the ids, one tolerant load, one generator (one home per concept).

A wave names its recipes as ``@file`` (one id per line, ``#`` comments) or ``a,b``; both
:mod:`rcp_ndcg_test.jobs.plugins` and :mod:`rcp_ndcg_test.jobs.run_wave` read such a list and load its
recipes.  The load is tolerant on purpose ("one failing recipe never stops the wave", end to end): a
recipe that fails validation is returned as a failure with the validation message, so the collector
reports and skips it and the wave report marks it failed -- it never kills the job that merely lists it.

The committed lists' one home is ``rcp-ndcg-test/wave-lists/`` (owner decision, 2026-10-09: the wave
lists are committed under the tooling home, decision 20); ``rc_build.sh`` stages them as
``<stage>/wave-lists/``.  :func:`write_wave_lists` generates them from the shipped recipes through
:func:`rcp_ndcg_vllm.recipe.iter_recipes`, so a list cannot drift from the recipe catalog: regenerate
after a catalog change with ``python -m rcp_ndcg_test.jobs.wavelist --out rcp-ndcg-test/wave-lists``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rcp_ndcg_vllm.recipe import (
    Recipe,
    default_recipes_root,
    iter_recipes,
    load_family,
    resolve_recipe,
)

from rcp_ndcg_test.errors import HarnessError, RecipeError

__all__ = ["ALL_RETRIEVAL", "DEFAULT_WAVE_LISTS", "load_wave", "main", "parse_ids", "write_wave_lists"]

DEFAULT_WAVE_LISTS = Path("rcp-ndcg-test/wave-lists")
"""The committed wave lists' one home in the checkout (``rc_build.sh`` stages it as ``<stage>/wave-lists/``)."""

ALL_RETRIEVAL = "all-retrieval.txt"
"""The wave naming every shipped recipe id: one id per line, sorted (``#`` comments allowed)."""


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
    recipes: list[Recipe] = []
    failed: dict[str, str] = {}
    if recipe_ids:
        ids = list(recipe_ids)
    else:
        # every variant of every family under the root, in id order (the wave's "all recipes"); a
        # family that does not load is a failed entry under its directory id (one failing family never
        # stops the wave), never an abort
        ids = []
        for directory in sorted(p for p in root.iterdir() if p.is_dir() and (p / "family.yaml").is_file()):
            try:
                ids.extend(variant.id for variant in load_family(directory).variants)
            except RecipeError as error:
                failed[directory.name] = str(error)
        ids.sort()
    for recipe_id in ids:
        try:
            recipes.append(resolve_recipe(recipe_id, root=root))
        except RecipeError as error:
            failed[recipe_id] = str(error)
    return recipes, failed


def write_wave_lists(out_dir: str | Path = DEFAULT_WAVE_LISTS, recipes_root: str | Path | None = None) -> list[Path]:
    """Write the committed wave lists from the shipped recipes; return the written paths.

    Inputs: the output directory (default: :data:`DEFAULT_WAVE_LISTS`, the lists' one home in the
    checkout) and the recipe root (default: the shipped package data).  Output: the paths written --
    :data:`ALL_RETRIEVAL`, one shipped retrieval recipe id per line (judge recipes are not wave members),
    sorted, generated through :func:`rcp_ndcg_vllm.recipe.iter_recipes` so the list cannot drift from the recipe catalog.  Raises
    :class:`RecipeError` when the recipe root holds no recipe.  Units: none.
    """
    recipes = [recipe for recipe in iter_recipes(recipes_root) if recipe.role != "judge"]
    if not recipes:
        raise RecipeError(
            f"no recipes under {recipes_root if recipes_root is not None else 'the package data'}; "
            "the wave lists are generated from the recipe catalog"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ALL_RETRIEVAL
    path.write_text("".join(f"{recipe.id}\n" for recipe in recipes), encoding="utf-8")
    return [path]


def main(argv: list[str] | None = None) -> int:
    """The CLI: ``python -m rcp_ndcg_test.jobs.wavelist --out <dir>`` regenerates the committed lists."""
    parser = argparse.ArgumentParser(
        prog="python -m rcp_ndcg_test.jobs.wavelist",
        description="Generate the committed wave lists from the shipped recipes.",
    )
    parser.add_argument("--out", default=str(DEFAULT_WAVE_LISTS), help="the wave-lists directory to write")
    parser.add_argument("--recipes-root", default=None, help="recipe root (default: the package's recipes)")
    args = parser.parse_args(argv)
    try:
        written = write_wave_lists(args.out, args.recipes_root)
    except (HarnessError, RecipeError) as error:
        print(f"error: {error}")
        return 2
    for path in written:
        print(f"{path}: {len(path.read_text(encoding='utf-8').splitlines())} ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
