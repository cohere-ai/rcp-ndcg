"""``rcp-ndcg-vllm serve <recipe-id> [--port ...] [--dry-run]``: the one command of the lean serving package.

It builds the exact ``vllm serve`` argv of the shipped recipe (the template path from the package data, the
media flags, the pooler config) with :func:`~rcp_ndcg_vllm.recipe.serve_argv` and execs it — the engine image
runs this instead of a hand-written command line (``serve: {command: ["rcp-ndcg-vllm", "serve", "<id>"]}`` in a
run config). ``--dry-run`` prints the shell-quoted argv instead. A recipe that declares ``serve.plugin`` the
engine environment does not carry is refused before anything starts, with the exact ``pip install --no-deps``
line that prepares it.

The module needs only the stdlib and this package's own imports (pydantic, PyYAML): it runs inside a stock
engine image where rcp-ndcg is absent.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import shlex
import shutil
import sys

from .errors import RecipeError
from .recipe import default_recipes_root, load_recipe, serve_argv

__all__ = ["main"]


def _recipe(recipe_id: str):
    """The shipped recipe ``recipe_id``; ``RecipeError`` listing the known ids when it is not shipped."""
    root = default_recipes_root()
    directory = root / recipe_id
    if not (directory / "recipe.yaml").is_file():
        known = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "recipe.yaml").is_file())
        raise RecipeError(f"no shipped recipe {recipe_id!r}; the shipped recipes are: {', '.join(known)}")
    return load_recipe(directory)


def _check_plugin(spec: str | None) -> None:
    """Refuse a declared ``serve.plugin`` (a pip spec) missing from this environment, with the install line."""
    if spec is None:
        return
    name = spec.split("==", 1)[0].split("[", 1)[0].strip()
    try:
        importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(
            f"recipe names plugin {spec!r}, which this environment does not carry.\n"
            f"install it first: pip install --no-deps {spec}"
        ) from None


def main(argv: list[str] | None = None) -> int:
    """The ``rcp-ndcg-vllm`` console: ``serve <recipe-id> [--port ...] [--dry-run]``.

    Inputs: the console argv (default :data:`sys.argv`).  Outputs: the exit status — 0 from ``--dry-run`` and
    never from a real serve (:func:`os.execvp` replaces this process); 2 for usage, 1 for a bad recipe or a
    missing plugin, each message on stderr.
    """
    parser = argparse.ArgumentParser(prog="rcp-ndcg-vllm", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="build the `vllm serve` argv of a recipe and exec it")
    serve.add_argument("recipe_id", metavar="recipe-id", help="the shipped recipe to serve (its id)")
    serve.add_argument("--port", type=int, default=8000, help="the port the engine serves on (default 8000)")
    serve.add_argument("--dry-run", action="store_true", help="print the shell-quoted argv; exec nothing")
    args = parser.parse_args(argv)

    try:
        recipe = _recipe(args.recipe_id)
        _check_plugin(recipe.serve.plugin)
        _check_plugin(recipe.serve.io_processor_plugin)
        command = serve_argv(recipe, port=args.port, served_model_name=recipe.id)
    except RecipeError as error:
        print(f"rcp-ndcg-vllm: {error}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(shlex.join(command))
        return 0
    if shutil.which(command[0]) is None:
        print(
            f"rcp-ndcg-vllm: {command[0]} is not on PATH; run inside the engine image (image {recipe.engine.image})",
            file=sys.stderr,
        )
        return 1
    os.execvp(command[0], command)
    return 0  # unreachable: exec replaces this process
