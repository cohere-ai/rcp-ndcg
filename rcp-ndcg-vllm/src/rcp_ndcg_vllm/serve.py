"""``rcp-ndcg-vllm serve <recipe> [--variant <id>] [--port ...] [--set <path>=<value>] [--dry-run]``: the one
command of the lean serving package.

It builds the exact ``vllm serve`` argv of the recipe (the template path from the package data, the media
flags, the pooler config) with :func:`~rcp_ndcg_vllm.recipe.serve_argv` and execs it — the engine image
runs this instead of a hand-written command line (``serve: {command: ["rcp-ndcg-vllm", "serve", "<id>"]}`` in a
run config). ``--dry-run`` prints the argv, the recipe's identity and the applied deployment overrides instead;
a real serve logs the identity and the overrides on stderr before exec (the argv is the engine's own record,
which the corpus provenance keeps).

``<recipe>`` is a shipped recipe id, or a **family directory of the operator's own** (``./my-family/``, with
``--variant <id>`` when it declares more than one size): such a recipe loads through the same schema, is marked
unshipped and ``unverified``, and is identified by the content hash of its resolved form (never a shipped id).

``--set <path>=<value>`` sets a **deployment field**: the recipe schema declares which fields those are
(:data:`~rcp_ndcg_vllm.recipe.FIELD_ROLES`), and only a DEPLOYMENT one may be named — ``resources.gpus``,
``serve.host``, ``serve.port``, ``serve.max_model_len`` (at or above the client's largest token budget),
``serve.gpu_memory_utilization`` (a finite fraction above 0), ``serve.max_num_seqs``,
``serve.max_num_batched_tokens``. A content field
(model, revision, dtype, the pooler config, the templates, the hf overrides, a patch) is refused by name: a
different revision or content is a different variant. A recipe that declares ``serve.plugin`` the engine
environment does not carry is refused before anything starts, with the exact ``pip install --no-deps`` line that
prepares it.

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
from .recipe import Recipe, load_recipe, parse_deployment_overrides, serve_argv

__all__ = ["build_parser", "run_console"]

_DEFAULT_PORT = 8000
"""The port a serve without ``--port`` and without a ``serve.port`` override serves on."""


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


def build_parser() -> argparse.ArgumentParser:
    """The console's ``argparse`` tree (``tests/contract`` pins it: docs-release Q3's public console tree)."""
    assert __doc__ is not None  # the module docstring is the console description
    parser = argparse.ArgumentParser(prog="rcp-ndcg-vllm", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="build the `vllm serve` argv of a recipe and exec it")
    serve.add_argument(
        "recipe_id",
        metavar="recipe-id",
        help="the recipe to serve: a shipped recipe id, or a family directory (or family.yaml) of your own",
    )
    serve.add_argument(
        "--variant",
        metavar="variant-id",
        default=None,
        help="the variant to serve when recipe-id is a family directory with more than one size",
    )
    serve.add_argument(
        "--port",
        type=int,
        default=None,
        help="the port the engine serves on; --set serve.port=... overrides it (default 8000)",
    )
    serve.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="a deployment field of the recipe, e.g. --set serve.max_num_seqs=64 (the recipe schema declares "
        "which fields are deployment; a content field is refused)",
    )
    serve.add_argument("--dry-run", action="store_true", help="print the argv; exec nothing")
    return parser


def _overrides_text(overrides: dict[str, object]) -> str:
    """The applied overrides as one line's worth of text (``none`` when there are none)."""
    return ", ".join(f"{path}={value}" for path, value in overrides.items()) or "none"


def _served_lines(recipe: Recipe, overrides: dict[str, object]) -> list[str]:
    """The two record lines a serve prints: the recipe's identity, and the deployment overrides applied."""
    return [f"identity: {recipe.identity}", f"deployment overrides: {_overrides_text(overrides)}"]


def run_console(argv: list[str] | None = None) -> int:
    """The ``rcp-ndcg-vllm`` console: ``serve <recipe-id> [--variant ...] [--port ...] [--set ...] [--dry-run]``.

    Inputs: the console argv (default :data:`sys.argv`).  Outputs: the exit status — 0 from ``--dry-run`` and
    never from a real serve (:func:`os.execvp` replaces this process); 2 for usage, 1 for a bad recipe, a
    refused deployment override or a missing plugin, each message on stderr.  ``--dry-run`` prints the argv
    first (one shell-quoted line), then the identity and the applied overrides; a real serve logs those two
    lines on stderr.
    """
    args = build_parser().parse_args(argv)

    try:
        recipe = load_recipe(args.recipe_id, variant=args.variant)
        _check_plugin(recipe.serve.plugin)
        _check_plugin(recipe.serve.io_processor_plugin)
        overrides = parse_deployment_overrides(args.set)
        port = args.port if args.port is not None else _DEFAULT_PORT
        command = serve_argv(recipe, port=port, served_model_name=recipe.id, deployment=overrides)
    except RecipeError as error:
        print(f"rcp-ndcg-vllm: {error}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(shlex.join(command))
        for line in _served_lines(recipe, overrides):
            print(line)
        return 0
    print(f"rcp-ndcg-vllm: serving {recipe.id}", file=sys.stderr)
    for line in _served_lines(recipe, overrides):
        print(f"rcp-ndcg-vllm: {line}", file=sys.stderr)
    if shutil.which(command[0]) is None:
        print(
            f"rcp-ndcg-vllm: {command[0]} is not on PATH; run inside the engine image (image {recipe.engine.image})",
            file=sys.stderr,
        )
        return 1
    os.execvp(command[0], command)
    return 0  # unreachable: exec replaces this process
