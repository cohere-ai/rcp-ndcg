"""Change handling over the observation corpora (OBSERVATIONS-SPEC section 7): what to re-record and
what moved.

Two commands, one library:

* ``changed`` -- the **re-record-changed-only** selection against the corpora COMMITTED in the
  repository: every recipe of the recipes root, its behaviour fingerprint recomputed, reported as
  ``new`` (no corpus), ``changed`` (a fingerprint moved -- the inputs that moved are named) or
  ``unchanged`` -- what a reviewer re-records before a release. The wave runner's own
  ``--changed-since <wave.json>`` selection is another scope: it compares a previous WAVE's index (its
  fingerprints and engine versions, :func:`rcp_ndcg_test.observe.corpus.changed_since`), so a wave
  re-records what moved since that wave; both key a corpus by the same behaviour fingerprint.
* ``diff`` -- the **behaviour diff** of two corpora of one recipe (per input, the score or vector
  deltas, changed statuses, changed refusals and changed protocol behaviour, summarised by stratum),
  written here for review before a new corpus replaces the old in the repository.

Run from a checkout of the repository::

    python -m rcp_ndcg_test.changes changed --recipes-root rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes \
        --corpora-root tests/contract/engines/vllm-0.31.0
    python -m rcp_ndcg_test.changes diff --before <corpus-dir> --after <corpus-dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from rcp_ndcg_vllm.recipe import Recipe, load_family, load_recipes_of

from rcp_ndcg_test.errors import HarnessError

from .fingerprint import behaviour_fingerprint, fingerprint_changes, fingerprint_inputs

__all__ = [
    "StaleCorpusError",
    "behaviour_report",
    "changed_recipes",
    "main",
    "recipe_state",
    "resolve_corpus",
    "waiver_covers",
]


class StaleCorpusError(HarnessError):
    """No committed corpus carries the recipe's current behaviour fingerprint: the message names the
    fingerprint inputs that moved against every recorded corpus of the recipe (GPU-VALIDATION item 8:
    staleness is a failure that names what changed, never a silent pass)."""


def _recorded(corpora_root: str | Path, recipe_id: str) -> dict[str, tuple[Path, dict[str, str]]]:
    """The recipe's committed corpora by recorded fingerprint: ``{fingerprint: (directory, inputs)}``,
    found by scanning manifests (:func:`rcp_ndcg_test.engines.find_corpora`) and read through the one
    corpus reader (:func:`rcp_ndcg_test.corpus.load_corpus`) with its integrity check: a corpus whose
    hashes do not hold is refused (:class:`HarnessError` naming it and the mismatches), never compared."""
    from rcp_ndcg_test.corpus import integrity_mismatches, load_corpus
    from rcp_ndcg_test.engines import find_corpora

    recorded: dict[str, tuple[Path, dict[str, str]]] = {}
    for directory in find_corpora(corpora_root, recipe_id=recipe_id):
        corpus = load_corpus(directory)
        mismatches = integrity_mismatches(corpus)
        if mismatches:
            raise HarnessError(f"corpus {directory}: its hashes do not hold ({'; '.join(mismatches)})")
        manifest = corpus.manifest
        fingerprint = str(manifest["recipe"]["behaviour_fingerprint"])
        recorded[fingerprint] = (directory, dict(manifest["recipe"].get("fingerprint_inputs") or {}))
    return recorded


def recipe_state(recipe: Recipe, corpora_root: str | Path) -> dict[str, Any]:
    """One recipe against its committed corpora: the fingerprint recomputed from the repository and
    compared with every recorded one.

    Args:
        recipe: The loaded recipe (from the repository, or an edited copy).
        corpora_root: The corpora root (scanned for manifests; directory names are never trusted).

    Returns:
        ``{"state": "unchanged"|"changed"|"new", "behaviour_fingerprint", "corpus" (the directory whose
        recorded fingerprint is the current one, else ``None``), "changed_inputs" (the input names that
        moved against any recorded corpus, ``[]`` when unchanged or new), "changed_by_corpus"
        (``{recorded fingerprint: names}``), "recorded_fingerprints"}``.

    Raises:
        HarnessError: the recipe cannot be fingerprinted (the message names recipe and input).
    """
    inputs = fingerprint_inputs(recipe)
    fingerprint = behaviour_fingerprint(recipe)
    recorded = _recorded(corpora_root, recipe.id)
    if fingerprint in recorded:
        return {
            "state": "unchanged",
            "behaviour_fingerprint": fingerprint,
            "corpus": str(recorded[fingerprint][0]),
            "changed_inputs": [],
            "changed_by_corpus": {},
            "recorded_fingerprints": sorted(recorded),
        }
    by_corpus = {previous: fingerprint_changes(old, inputs) for previous, (_, old) in sorted(recorded.items())}
    return {
        "state": "changed" if recorded else "new",
        "behaviour_fingerprint": fingerprint,
        "corpus": None,
        "changed_inputs": sorted({name for names in by_corpus.values() for name in names}),
        "changed_by_corpus": by_corpus,
        "recorded_fingerprints": sorted(recorded),
    }


def resolve_corpus(recipe: Recipe, corpora_root: str | Path) -> Path:
    """The committed corpus of the recipe's **current** behaviour fingerprint.

    Args:
        recipe: The loaded recipe.
        corpora_root: The corpora root (scanned for manifests).

    Returns:
        The corpus directory whose recorded fingerprint equals the recomputed one.

    Raises:
        StaleCorpusError: none does; the message names, per recorded corpus, the inputs that moved
            (or says no corpus of the recipe is committed).
    """
    state = recipe_state(recipe, corpora_root)
    if state["corpus"] is not None:
        return Path(state["corpus"])
    if state["state"] == "new":
        raise StaleCorpusError(f"recipe {recipe.id}: no committed corpus under {corpora_root}; record one")
    moved = "; ".join(f"{previous[:12]}...: {names}" for previous, names in state["changed_by_corpus"].items())
    raise StaleCorpusError(
        f"recipe {recipe.id}: stale corpus -- the behaviour fingerprint moved to "
        f"{state['behaviour_fingerprint'][:12]}...; changed fingerprint inputs against the recorded "
        f"corpora: {moved}. Re-record the recipe (python -m rcp_ndcg_test.changes changed) or add a "
        "dated waiver"
    )


def waiver_covers(waiver: Mapping[str, Any], recipe_id: str, changed: list[str], *, today: date) -> bool:
    """Whether one staleness waiver covers a recipe's changed inputs on ``today``.

    A waiver names the recipe, every changed input it covers (a superset of ``changed``: an input it
    does not name stays failing), why (``reason``), when it was granted (``date``) and an ``expires``
    date not before ``today``. The release checklist requires the waiver file to be empty.

    Args:
        waiver: One entry of the waiver file.
        recipe_id: The stale recipe.
        changed: The changed input names the staleness check reported.
        today: The day of the check.

    Returns:
        ``True`` only when every condition holds.
    """
    try:
        expires = date.fromisoformat(str(waiver.get("expires", "")))
        date.fromisoformat(str(waiver.get("date", "")))
    except ValueError:
        return False
    return (
        waiver.get("recipe_id") == recipe_id
        and set(waiver.get("changed_inputs") or ()) >= set(changed)
        and bool(str(waiver.get("reason") or "").strip())
        and expires >= today
    )


def _family_recipes(directory: Path) -> list[Recipe]:
    """The resolved recipes of one family directory (its variants, in file order)."""
    return load_recipes_of(load_family(directory), directory)


def changed_recipes(recipes_root: str | Path, corpora_root: str | Path) -> dict[str, dict[str, Any]]:
    """What to re-record: every recipe's :func:`recipe_state` against the committed corpora.

    Args:
        recipes_root: The family directories (each with its ``family.yaml``; one state per variant id).
        corpora_root: The committed corpora root (scanned for manifests).

    Returns:
        ``{recipe_id: recipe_state}`` -- a recipe that does not load is reported ``unloadable`` with its
        error and no changed inputs (one failing recipe never stops the selection), never fatal.
    """
    states: dict[str, dict[str, Any]] = {}
    for directory in sorted(Path(recipes_root).iterdir()):
        if not (directory / "family.yaml").is_file():
            continue
        try:
            recipes = _family_recipes(directory)
        except Exception as error:  # noqa: BLE001 - one failing family never stops the selection
            # The family itself does not load: its variants cannot be named, so the directory is
            # reported under the family's id with the load error (the caller sees a failed family,
            # never a silently empty selection).
            states[directory.name] = {
                "state": "unloadable",
                "error": str(error).splitlines()[0],
                "changed_inputs": [],
                "recorded_fingerprints": [],
            }
            continue
        for recipe in recipes:
            try:
                states[recipe.id] = recipe_state(recipe, corpora_root)
            except Exception as error:  # noqa: BLE001 - one failing recipe never stops the selection
                states[recipe.id] = {
                    "state": "unloadable",
                    "error": str(error).splitlines()[0],
                    "changed_inputs": [],  # nothing is known to have changed: the recipe does not load
                    "recorded_fingerprints": sorted(_recorded(corpora_root, recipe.id)),
                }
    return states


def behaviour_report(before: str | Path, after: str | Path) -> Any:
    """The behaviour diff of two corpora of one recipe (see :func:`rcp_ndcg_test.engines.behaviour_diff`).

    Args:
        before: The previous corpus directory.
        after: The candidate corpus directory.

    Returns:
        The JSON-ready behaviour-diff document (per-input deltas and the summary by stratum).
    """
    from rcp_ndcg_test.corpus import load_corpus
    from rcp_ndcg_test.engines import behaviour_diff

    return behaviour_diff(load_corpus(before), load_corpus(after))


def main(argv: list[str] | None = None) -> int:
    """The ``changes`` command: ``changed`` (the re-record selection) and ``diff`` (the behaviour diff)."""
    parser = argparse.ArgumentParser(prog="python -m rcp_ndcg_test.changes", description=__doc__)
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
