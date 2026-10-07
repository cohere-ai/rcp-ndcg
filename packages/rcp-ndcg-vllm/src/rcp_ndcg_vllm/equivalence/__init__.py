"""The three-stage equivalence harness: prompts (CPU), scores and vectors (the gates), and metrics.

Public entry points: :func:`run` (all requested stages, writes the report, returns the document) plus the stage
functions in :mod:`.stages` and :mod:`.metrics`.  The CLI lives at :mod:`rcp_ndcg_vllm.equivalence.__main__`::

    python -m rcp_ndcg_vllm.equivalence --recipe <dir> --base-url <url> --pairs <file> --out <dir> \\
        --reference-python <python>

``--reference-python`` is required when stage 2 runs (the reference never imports into the harness's process);
with it, stage 1 also compares the reference's ``render`` against the product's ``fit``.
"""

from __future__ import annotations

from typing import Any

from ..errors import HarnessError
from ..recipe import load_recipe
from .gates import ResolvedGates, kendall_tau_b, resolve_gates
from .metrics import stage3_metrics
from .stages import load_pairs, stage1_prompts, stage2_scores

__all__ = [
    "ResolvedGates",
    "kendall_tau_b",
    "load_pairs",
    "load_recipe",
    "resolve_gates",
    "run",
    "stage1_prompts",
    "stage2_scores",
    "stage3_metrics",
]


def run(
    recipe: Any,
    *,
    base_url: str | None,
    pairs_path: str,
    out_dir: str,
    stages: list[int],
    reference_python: str | None = None,
    rankings_dir: str | None = None,
    served_model_name: str | None = None,
    limit: int | None = None,
    device: str = "cpu",
    recorder: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the requested stages of the equivalence check for one recipe; write and return the report document.

    Inputs: the recipe, the engine's base URL (only stage 2 needs it), the pairs file, the output directory,
    the stage numbers to run (1 and 2 by default; 3 needs ``rankings_dir``), the reference interpreter
    (``--reference-python``; required for stage 2, and used by stage 1's render comparison when given) and the
    rankings directory for stage 3; ``recorder`` collects stage 2's captured exchanges.  Output: the report
    document; ``passed`` is true only when every requested stage passed.  Also writes ``equivalence.json`` and ``EQUIVALENCE.md`` under ``out_dir``.
    """
    from .report import write_report

    document: dict[str, Any] = {
        "recipe": recipe.id,
        "image": recipe.engine.image,
        "base_url": base_url,
        "stages": stages,
    }
    if 1 in stages:
        document["stage1"] = stage1_prompts(recipe, pairs_path, reference_python, base_url=base_url, limit=limit)
    if 2 in stages:
        if base_url is None:
            raise HarnessError("stage 2 needs the engine's --base-url")
        document["stage2"] = stage2_scores(
            recipe,
            pairs_path,
            reference_python or "",
            base_url=base_url,
            served_model_name=served_model_name or recipe.id,
            device=device,
            recorder=recorder,
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
