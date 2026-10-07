"""Change handling over the observation corpora (OBSERVATIONS-SPEC section 7): what to re-record and
what moved.

Two commands, one library:

* ``changed`` -- the **re-record-changed-only** selection: every recipe of the recipes root, its
  behaviour fingerprint recomputed from the repository, reported against the committed corpora:
  ``new`` (no corpus), ``changed`` (a fingerprint moved -- the inputs that moved are named) or
  ``unchanged``. The wave runner's ``--changed-since <corpus-index>`` mode records exactly the
  ``changed`` and ``new`` ones (and, for a new engine version, the protocol layer once).
* ``diff`` -- the **behaviour diff** of two corpora of one recipe (per input, the score or vector
  deltas, changed statuses, changed refusals and changed protocol behaviour, summarised by stratum),
  written here for review before a new corpus replaces the old in the repository.

Run from a checkout of the repository::

    python -m rcp_ndcg_vllm.changes changed --recipes-root packages/rcp-ndcg-vllm/recipes \
        --corpora-root tests/contract/engines/vllm-0.31.0
    python -m rcp_ndcg_vllm.changes diff --before <corpus-dir> --after <corpus-dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rcp_ndcg_vllm.errors import HarnessError

from .fingerprint import behaviour_fingerprint, fingerprint_changes, fingerprint_inputs
from .recipe import load_recipe

__all__ = ["behaviour_report", "changed_recipes", "main"]


def changed_recipes(recipes_root: str | Path, corpora_root: str | Path) -> dict[str, dict[str, Any]]:
    """What to re-record: every recipe's state against the committed corpora.

    Args:
        recipes_root: The recipe directories (each with its ``recipe.yaml``).
        corpora_root: The committed corpora root (``<engine>-<version>/<recipe>/<fingerprint>/``;
            recursion picks up every manifest).

    Returns:
        ``{recipe_id: {"state": "new"|"changed"|"unchanged", "behaviour_fingerprint", "changed_inputs",
        "recorded_fingerprints"}}}`` -- a recipe that does not load is reported ``unloadable`` with its
        error (one failing recipe never stops the selection), never fatal.

    Raises:
        HarnessError: a recipe cannot be loaded or fingerprinted (the message names recipe and input).
    """
    recorded: dict[str, dict[str, dict[str, str]]] = {}
    for manifest_path in sorted(Path(corpora_root).rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recipe_id = manifest["recipe"]["id"]
        fingerprint = manifest["recipe"]["behaviour_fingerprint"]
        recorded.setdefault(recipe_id, {})[fingerprint] = dict(manifest["recipe"].get("fingerprint_inputs") or {})
    states: dict[str, dict[str, Any]] = {}
    for directory in sorted(Path(recipes_root).iterdir()):
        if not (directory / "recipe.yaml").is_file():
            continue
        recipe_id = directory.name
        known = recorded.get(recipe_id, {})
        try:
            recipe = load_recipe(directory)
            inputs = fingerprint_inputs(recipe)
            fingerprint = behaviour_fingerprint(recipe)
        except Exception as error:  # noqa: BLE001 - one failing recipe never stops the selection
            states[recipe_id] = {
                "state": "unloadable",
                "error": str(error).splitlines()[0],
                "changed_inputs": [],  # nothing is known to have changed: the recipe does not load
                "recorded_inputs": sorted({name for previous in known.values() for name in previous}),
                "recorded_fingerprints": sorted(known),
            }
            continue
        changed_inputs: list[str] = []
        for previous, previous_inputs in known.items():
            if previous == fingerprint:
                continue
            changed_inputs = sorted(set(changed_inputs) | set(fingerprint_changes(previous_inputs, inputs)))
        if fingerprint in known:
            state = "unchanged"
        elif known:
            state = "changed"
        else:
            state = "new"
        states[recipe_id] = {
            "state": state,
            "behaviour_fingerprint": fingerprint,
            "changed_inputs": changed_inputs,
            "recorded_fingerprints": sorted(known),
        }
    return states


def behaviour_report(before: str | Path, after: str | Path) -> Any:
    """The behaviour diff of two corpora of one recipe (see :func:`rcp_ndcg.testing.engines.behaviour_diff`).

    Args:
        before: The previous corpus directory.
        after: The candidate corpus directory.

    Returns:
        The JSON-ready behaviour-diff document (per-input deltas and the summary by stratum).
    """
    from rcp_ndcg.testing.engines import behaviour_diff, load_corpus

    return behaviour_diff(load_corpus(before), load_corpus(after))


def main(argv: list[str] | None = None) -> int:
    """The ``changes`` command: ``changed`` (the re-record selection) and ``diff`` (the behaviour diff)."""
    parser = argparse.ArgumentParser(prog="python -m rcp_ndcg_vllm.changes", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    selection = sub.add_parser("changed", help="the recipes whose fingerprint changed since their corpus")
    selection.add_argument("--recipes-root", required=True)
    selection.add_argument("--corpora-root", required=True)
    diff = sub.add_parser("diff", help="the behaviour diff of two corpora of one recipe")
    diff.add_argument("--before", required=True)
    diff.add_argument("--after", required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "changed":
            report: Any = changed_recipes(arguments.recipes_root, arguments.corpora_root)
        else:
            report = behaviour_report(arguments.before, arguments.after)
    except HarnessError as error:
        print(f"changes: {error}", file=sys.stderr)
        return 3
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main() with argv
    raise SystemExit(main())
