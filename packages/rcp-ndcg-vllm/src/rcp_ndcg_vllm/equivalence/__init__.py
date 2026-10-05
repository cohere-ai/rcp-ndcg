"""The three-stage equivalence harness: prompts (CPU), scores and vectors (the gates), and metrics.

Public entry points: :func:`run` (all requested stages, writes the report, returns the document) plus the stage
functions in :mod:`.stages` and :mod:`.metrics`.  The CLI lives at :mod:`rcp_ndcg_vllm.equivalence.__main__`::

    python -m rcp_ndcg_vllm.equivalence --recipe <dir> --base-url <url> --pairs <file> --out <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..errors import HarnessError
from ..recipe import Recipe, load_recipe
from .gates import ResolvedGates, kendall_tau_b, resolve_gates
from .metrics import stage3_metrics
from .prompt import TokenizerAdapter, load_tokenizer
from .reference import Reference, load_reference
from .report import write_report
from .stages import load_pairs, stage1_prompts, stage2_scores

__all__ = [
    "Reference",
    "ResolvedGates",
    "TokenizerAdapter",
    "kendall_tau_b",
    "load_pairs",
    "load_reference",
    "load_tokenizer",
    "resolve_gates",
    "run",
    "stage1_prompts",
    "stage2_scores",
    "stage3_metrics",
]


def run(
    recipe: Recipe,
    *,
    base_url: str | None,
    pairs: list[dict[str, Any]],
    out_dir: str | Path,
    stages: list[int],
    tokenizer: TokenizerAdapter | None = None,
    reference: Reference | None = None,
    rankings_dir: str | Path | None = None,
    served_model_name: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the requested stages of the equivalence check for one recipe; write and return the report document.

    Inputs: the recipe, the engine's base URL (only stage 2 needs it), the sampled pairs, the output directory,
    the stage numbers to run (1 and 2 by default; 3 needs ``rankings_dir``) and optional injected
    tokenizer/reference for tests.  ``limit`` truncates the pairs (a quick check).  Output: the report document;
    ``passed`` is true only when every requested stage passed.  Also writes ``equivalence.json`` and
    ``EQUIVALENCE.md`` under ``out_dir``.
    """
    if reference is None:
        if recipe.reference.kind == "stored_scores":
            reference = _stored_scores_stub(recipe)
        else:
            reference = load_reference(_recipe_dir(recipe), recipe.reference.entry)
    if limit is not None:
        pairs = pairs[:limit]
    document: dict[str, Any] = {
        "recipe": recipe.id,
        "image": recipe.engine.image,
        "base_url": base_url,
        "pairs": len(pairs),
        "stages": stages,
    }
    if 1 in stages:
        document["stage1"] = stage1_prompts(
            recipe, pairs, reference, tokenizer if tokenizer is not None else load_tokenizer(recipe)
        )
    if 2 in stages:
        if base_url is None:
            raise HarnessError("stage 2 needs the engine's --base-url")
        document["stage2"] = stage2_scores(
            recipe, base_url, pairs, reference, served_model_name=served_model_name or recipe.id
        )
    if 3 in stages:
        if rankings_dir is None:
            raise HarnessError("stage 3 needs --rankings-dir with <subset>.{served,reference}.jsonl rankings")
        document["stage3"] = stage3_metrics(rankings_dir, resolve_gates(recipe))
    document["passed"] = all(
        bool(document[stage]["passed"]) for stage in ("stage1", "stage2", "stage3") if stage in document
    )
    write_report(out_dir, document)
    return document


def _recipe_dir(recipe: Recipe) -> str:
    directory = recipe._dir
    if directory is None:
        raise HarnessError(f"recipe {recipe.id} was not loaded from a directory; use load_recipe")
    return str(directory)


def _stored_scores_stub(recipe: Recipe) -> Reference:
    """A reference for ``stored_scores`` recipes: no module exists, so stages that need one refuse clearly."""

    def _refuse(*args: object) -> None:  # pragma: no cover - only reached on misuse
        raise HarnessError(
            f"recipe {recipe.id} uses reference.kind=stored_scores: no reference module exists; run stage 3, or "
            "compare served scores against the stored scores file directly"
        )

    return Reference(_Refusing(_refuse), "<stored-scores>")


class _Refusing:
    """A module-like object whose every call refuses; see :func:`_stored_scores_stub`."""

    def __init__(self, refuse: Any) -> None:
        self._refuse = refuse

    def __getattr__(self, name: str) -> Any:
        return self._refuse


def main(argv: list[str] | None = None) -> int:
    """The CLI: ``python -m rcp_ndcg_vllm.equivalence``; exit code 0 only if every requested gate passes."""
    parser = argparse.ArgumentParser(
        prog="python -m rcp_ndcg_vllm.equivalence",
        description="Check a served recipe against its reference implementation (stages 1, 2, 3).",
    )
    parser.add_argument("--recipe", required=True, help="recipe directory (with recipe.yaml)")
    parser.add_argument("--base-url", default=None, help="engine root URL, e.g. http://127.0.0.1:8100 (stage 2)")
    parser.add_argument("--pairs", default=None, help='JSONL pairs file: {"query", "documents"} per line')
    parser.add_argument("--out", required=True, help="output directory for equivalence.json and EQUIVALENCE.md")
    parser.add_argument("--stages", default="1,2", help="stages to run, comma-separated (default: 1,2)")
    parser.add_argument("--rankings-dir", default=None, help="rankings directory for stage 3")
    parser.add_argument("--limit", type=int, default=None, help="check only the first N pairs (a quick run)")
    args = parser.parse_args(argv)
    stages = sorted({int(stage.strip()) for stage in args.stages.split(",") if stage.strip()})
    if not stages or any(stage not in (1, 2, 3) for stage in stages):
        parser.error("--stages must be a comma-separated list drawn from 1, 2, 3")
    try:
        recipe = load_recipe(args.recipe)
        if args.pairs is None:
            parser.error("--pairs is required (stage 1 and stage 2 read the same sampled pairs)")
        pairs = load_pairs(args.pairs)
        document = run(
            recipe,
            base_url=args.base_url,
            pairs=pairs,
            out_dir=args.out,
            stages=stages,
            rankings_dir=args.rankings_dir,
            limit=args.limit,
        )
    except HarnessError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"equivalence for {document['recipe']}: {'PASS' if document['passed'] else 'FAIL'}")
    return 0 if document["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
