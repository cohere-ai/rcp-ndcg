"""The plugin wheels a wave's recipes install into the engine environment (node-runtime item 11).

A recipe's ``serve.plugin`` names the pip package the engine imports through ``vllm.general_plugins``;
the bootstrap installs it into the engine environment with ``--no-deps`` (the image ships torch, vLLM
and transformers, and a plugin vendors its own model code), under the freeze-diff guard: after the
installs, the engine environment's ``pip freeze`` may differ only in the plugins' own distributions,
or the bootstrap fails before any engine starts. A spec that names a file is installed from the staged
tree; anything else is installed as named (a wheelhouse wheel, or a name on an index - the item-9
fallback, declared). ``io_processor_plugin`` names a package resolved through the checkpoint's own
config, so it is recorded here, not installed by name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rcp_ndcg_vllm.recipe import Recipe, load_recipe

from ..errors import RecipeError

__all__ = ["collect", "main"]


def collect(recipes_root: str | Path, recipe_ids: list[str]) -> list[str]:
    """The pip specs of the listed recipes' ``serve.plugin`` wheels (deduped, in recipe order).

    A spec that names a file inside its recipe directory (``plugin.wheel``) or under the recipes root
    is returned as the recipe-relative path (``<recipe-id>/<file>``), so the bootstrap finds it in the
    staged tree; anything else passes through as named (the bootstrap then installs it from the
    wheelhouse or an index - the item-9 fallback, never silently).
    """
    specs: list[str] = []
    seen: set[str] = set()
    root = Path(recipes_root)
    for recipe_id in recipe_ids:
        recipe = _load(root / recipe_id)
        spec = recipe.serve.plugin
        if spec is None:
            continue
        candidate = Path(spec)
        if not candidate.is_absolute() and (root / recipe_id / spec).is_file():
            spec = f"{recipe_id}/{spec}"
        if spec not in seen:
            seen.add(spec)
            specs.append(spec)
    return specs


def _load(path: Path) -> Recipe:
    try:
        return load_recipe(path)
    except RecipeError as error:
        raise SystemExit(f"plugins: cannot load the recipe {path}: {error}") from error


def main(argv: list[str] | None = None) -> int:
    """The CLI the bootstrap calls: ``python -m rcp_ndcg_test.jobs.plugins collect``."""
    parser = argparse.ArgumentParser(
        prog="python -m rcp_ndcg_test.jobs.plugins",
        description="Collect the plugin wheels a wave's recipes declare (serve.plugin).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_collect = sub.add_parser("collect", help="print one plugin spec per line")
    p_collect.add_argument("--recipes-root", required=True, help="root of the recipe directories")
    p_collect.add_argument("--recipes", required=True, help="@file with one recipe id per line, or comma-separated")
    args = parser.parse_args(argv)
    recipe_ids = _recipe_ids(args.recipes)
    for spec in collect(args.recipes_root, recipe_ids):
        print(spec)
    return 0


def _recipe_ids(value: str) -> list[str]:
    """``@file`` (one id per line, # comments) or ``a,b`` into a list; empty means no recipes."""
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.is_file():
            print(f"plugins: no wave list at {path}", file=sys.stderr)
            raise SystemExit(2)
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
