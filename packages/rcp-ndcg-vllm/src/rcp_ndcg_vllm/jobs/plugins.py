"""The plugin wheels a wave's recipes install into the engine environment (node-runtime item 11).

A recipe's ``serve.plugin`` names the pip package the engine imports through ``vllm.general_plugins``;
the bootstrap installs it into the engine environment with ``--no-deps`` (the image ships torch, vLLM
and transformers, and a plugin vendors its own model code), under the freeze-diff guard: after the
installs, the engine environment's ``pip freeze`` may differ only in the plugins' own distributions,
or the bootstrap fails before any engine starts. A spec that names a file is installed from the staged
tree; anything else is installed as named from the staged wheelhouse only
(``--no-index --find-links <stage>/wheelhouse``, never an index), and a plugin found nowhere fails
the recipes that name it - with its exact name in the wave report - without stopping the wave.
``io_processor_plugin`` names a package resolved through the checkpoint's own config, so it is
recorded here, not installed by name.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..errors import HarnessError
from ..recipe import Recipe
from .wavelist import load_wave, parse_ids

__all__ = ["collect", "main", "spec_of"]


def collect(recipes_root: str | Path, recipe_ids: list[str]) -> list[str]:
    """The pip specs of the listed recipes' ``serve.plugin`` wheels (deduped, in recipe order).

    A spec that names a file inside its recipe directory (``plugin.wheel``) or under the recipes root
    is returned as the recipe-relative path (``<recipe-id>/<file>``), so the bootstrap finds it in the
    staged tree; anything else passes through as named (the bootstrap then installs it from the staged
    wheelhouse only).  A recipe that fails to validate is skipped here and never raises: the CLI
    reports it on stderr and the wave marks it failed with the validation message, so one invalid
    recipe never fails the job that merely lists it.
    """
    root = Path(recipes_root)
    recipes, _failed = load_wave(recipe_ids, root)
    return _specs(recipes, root)


def spec_of(recipe: Recipe, root: Path) -> str | None:
    """One recipe's ``serve.plugin`` pip spec as the collector emits it: its staged file as
    ``<recipe-id>/<file>`` when the file exists in the recipe's directory, else the bare name.  One
    home for the form: ``collect`` prints it, the bootstrap installs it and ``run_wave`` matches its
    failures on it.  Units: none.
    """
    spec = recipe.serve.plugin
    if spec is None:
        return None
    candidate = Path(spec)
    if not candidate.is_absolute() and (root / recipe.id / spec).is_file():
        return f"{recipe.id}/{spec}"
    return spec


def _specs(recipes: list[Recipe], root: Path) -> list[str]:
    """The recipes' plugin specs, deduped in recipe order."""
    specs: list[str] = []
    seen: set[str] = set()
    for recipe in recipes:
        spec = spec_of(recipe, root)
        if spec is not None and spec not in seen:
            seen.add(spec)
            specs.append(spec)
    return specs


def main(argv: list[str] | None = None) -> int:
    """The CLI the bootstrap calls: ``python -m rcp_ndcg_vllm.jobs.plugins collect``.

    Prints one plugin spec per line for the valid recipes and reports every skipped (invalid) recipe
    with its validation message on stderr; an invalid recipe never fails the job, so the exit code is
    0 unless the wave list itself cannot be read.
    """
    parser = argparse.ArgumentParser(
        prog="python -m rcp_ndcg_vllm.jobs.plugins",
        description="Collect the plugin wheels a wave's recipes declare (serve.plugin).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_collect = sub.add_parser("collect", help="print one plugin spec per line")
    p_collect.add_argument("--recipes-root", required=True, help="root of the recipe directories")
    p_collect.add_argument("--recipes", required=True, help="@file with one recipe id per line, or comma-separated")
    args = parser.parse_args(argv)
    try:
        recipe_ids = parse_ids(args.recipes)
        recipes, failed = load_wave(recipe_ids, args.recipes_root)
    except HarnessError as error:
        print(f"plugins: {error}", file=sys.stderr)
        return 2
    root = Path(args.recipes_root)
    for recipe_id, message in failed.items():
        print(f"plugins: skipping the recipe {recipe_id}: {message}", file=sys.stderr)
    for spec in _specs(recipes, root):
        print(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
