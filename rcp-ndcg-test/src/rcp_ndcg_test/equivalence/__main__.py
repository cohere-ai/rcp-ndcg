"""The equivalence CLI: ``python -m rcp_ndcg_test.equivalence ...``; see :func:`rcp_ndcg_test.equivalence`."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from rcp_ndcg_vllm.recipe import load_recipe

from rcp_ndcg_test.errors import HarnessError, RecipeError

from . import run


def main(argv: list[str] | None = None) -> int:
    """The CLI: ``python -m rcp_ndcg_test.equivalence``; exit code 0 only if every requested gate passes."""
    parser = argparse.ArgumentParser(
        prog="python -m rcp_ndcg_test.equivalence",
        description="Check a served recipe against its reference implementation (stages 1, 2, 3).",
    )
    parser.add_argument("--recipe", required=True, help="recipe directory (with family.yaml)")
    parser.add_argument(
        "--base-url",
        default=None,
        help="engine root URL (stage 1's /tokenize check and stage 2)",
    )
    parser.add_argument("--pairs", default=None, help='JSONL pairs file: {"query", "documents"} per line')
    parser.add_argument("--out", required=True, help="output directory for equivalence.json and EQUIVALENCE.md")
    parser.add_argument("--stages", default="1,2", help="stages to run, comma-separated (default: 1,2)")
    parser.add_argument("--rankings-dir", default=None, help="rankings directory for stage 3")
    parser.add_argument(
        "--reference-python",
        default=None,
        help="the python that runs the recipe's reference (its environment carries torch/transformers); "
        "required for stage 2, used by stage 1's render comparison when given",
    )
    parser.add_argument("--limit", type=int, default=None, help="check only the first N pairs (a quick run)")
    parser.add_argument("--device", default="cpu", help="device for the reference subprocess in stage 2 (default: cpu)")
    args = parser.parse_args(argv)
    stages = sorted({int(stage.strip()) for stage in args.stages.split(",") if stage.strip()})
    if not stages or any(stage not in (1, 2, 3) for stage in stages):
        parser.error("--stages must be a comma-separated list drawn from 1, 2, 3")
    if 2 in stages and args.reference_python is None:
        parser.error("--reference-python is required for stage 2 (the reference runs in its own environment)")
    if (1 in stages or 2 in stages) and args.pairs is None:
        parser.error("--pairs is required for stages 1 and 2 (the same sampled pairs feed both)")
    try:
        recipe = load_recipe(args.recipe)
        document: dict[str, Any] = run(
            recipe,
            base_url=args.base_url,
            pairs_path=args.pairs or "",
            out_dir=args.out,
            stages=stages,
            reference_python=args.reference_python,
            rankings_dir=args.rankings_dir,
            limit=args.limit,
            device=args.device,
        )
    except (HarnessError, RecipeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"equivalence for {document['recipe']}: {'PASS' if document['passed'] else 'FAIL'}")
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
