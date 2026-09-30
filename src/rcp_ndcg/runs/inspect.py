"""Read-only questions about runs: which runs exist, what happened in one, its evaluation, why a query scored.

What ``run list``, ``run show``, ``eval compare --run`` and ``eval explain`` (and their MCP tools) read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rcp_ndcg.errors import MissingInputError
from rcp_ndcg.runs.layout import RunLayout, discover_runs
from rcp_ndcg.runs.manifest import RunManifest
from rcp_ndcg.support import paths

if TYPE_CHECKING:
    from rcp_ndcg.data import Dataset
    from rcp_ndcg.eval import QueryExplanation
    from rcp_ndcg.runs.run import StepProgress, StepState


def list_runs(runs_dir: str | Path | None = None, *, limit: int = 50) -> list[dict[str, Any]]:
    """Summaries of the runs under ``runs_dir`` (default :func:`rcp_ndcg.support.paths.runs_dir`), newest first (an
    unreadable manifest is listed as such)."""
    if runs_dir is None:
        runs_dir = paths.runs_dir()
    rows = []
    for run_id in discover_runs(runs_dir)[:limit]:
        layout = RunLayout.at(Path(runs_dir) / run_id)
        try:
            manifest = RunManifest.load(layout)
        except Exception as exc:
            rows.append({"run_id": run_id, "status": "unreadable", "error": str(exc)})
            continue
        judges = sorted({family.judge_model for family in manifest.families.values()})
        rows.append(
            {
                "run_id": manifest.run_id,
                "status": manifest.status.value,
                "created_at": manifest.created_at.isoformat(),
                "dataset": manifest.dataset.name if manifest.dataset else None,
                "judges": judges,
                "metrics": manifest.metrics,
                "cost_usd": manifest.cost_usd,
                "steps": {record.name: record.status.value for record in manifest.steps},
            }
        )
    return rows


def step_states(layout: RunLayout, manifest: RunManifest) -> list[StepState]:
    """Every step the run's config plans, in run order, with its recorded status (``pending`` before it starts)
    and, for a judging step, the windows judged so far against the windows its schedule plans."""
    from rcp_ndcg.runs.config import JUDGE_STEPS, STEPS
    from rcp_ndcg.runs.run import StepState

    wanted = set(manifest.config.get("steps") or []) | {record.name for record in manifest.steps}
    names = [step for step in STEPS if step in wanted]
    states = []
    for name in names:
        record = manifest.step(name)
        progress = _judge_progress(layout, manifest, name) if name in JUDGE_STEPS else None
        if record is None:
            states.append(StepState(name=name, status="pending", progress=progress))
        else:
            states.append(
                StepState(
                    name=name,
                    status=record.status.value,
                    duration_s=record.duration_s,
                    error=record.error,
                    progress=progress,
                )
            )
    return states


def _judge_progress(layout: RunLayout, manifest: RunManifest, stage: str) -> StepProgress:
    """Windows stored in the stage's judgement file, against the windows its schedule plans for the run's pools.

    ``planned`` is known once the stage has claimed its store (the schedule) and the pools exist
    (``candidates.parquet``): the schedule's calls per query summed over the judged queries (the first ``limit``,
    each cut to ``depth``), without chunking, where a query shows more units than documents.
    """
    from rcp_ndcg.runs.run import StepProgress

    store = Path(layout.judgements) / f"{stage}.jsonl"
    done = 0
    if store.exists():
        with store.open(encoding="utf-8") as handle:
            done = sum(1 for line in handle if line.strip())
    return StepProgress(done=done, planned=_planned_windows(layout, manifest, stage))


def _planned_windows(layout: RunLayout, manifest: RunManifest, stage: str) -> int | None:
    from rcp_ndcg.llm.schedule import RubricSchedule, TournamentSchedule
    from rcp_ndcg.llm.store import JudgementStore

    claimed = JudgementStore(layout.judgements).identities().get(stage) if Path(layout.judgements).exists() else None
    if claimed is None or not Path(layout.candidates).exists():
        return None
    kind = TournamentSchedule if stage == "tournament" else RubricSchedule
    schedule = kind.model_validate(claimed["identity"]["schedule"])
    from rcp_ndcg.data import load_rankings

    candidates = manifest.config.get("candidates") or {}
    depth = int(candidates.get("depth", 150))
    limit = manifest.config.get("limit")
    pools = list(load_rankings(layout.candidates, format="parquet").queries().values())
    sizes = [min(len(pool), depth) for pool in (pools[:limit] if limit else pools)]
    return sum(schedule.calls_per_query(size) for size in sizes)


def get_run(run_dir: str | Path) -> dict[str, Any]:
    """One run's manifest and which artifacts of the layout exist."""
    layout = _layout(run_dir)
    manifest = RunManifest.load(layout)
    artifacts = {
        "candidates": layout.candidates,
        "tournament": layout.path("judgements", "tournament.jsonl"),
        "rubric": layout.path("judgements", "rubric.jsonl"),
        "calibration": layout.path("calibration", "items.json"),
        "metrics": layout.metrics,
        "comparison": layout.comparison,
    }
    return {
        "manifest": json.loads(manifest.model_dump_json()),
        "artifacts": {name: Path(path).exists() for name, path in artifacts.items()},
        "run_dir": layout.root,
    }


def evaluation_report(run_dir: str | Path):
    """The run's :class:`~rcp_ndcg.eval.EvalReport`, recomputed from its artifacts (every system, its calibration)."""
    return _pipeline(run_dir).evaluation_report()


def explain_query(run_dir: str | Path, query_id: str, *, k: int = 10) -> tuple[QueryExplanation, Dataset]:
    """One query side by side across the run's systems (:func:`rcp_ndcg.eval.explain`), with the run's dataset
    (for the query and document texts)."""
    from rcp_ndcg.calibration import Calibration
    from rcp_ndcg.eval import explain

    pipeline = _pipeline(run_dir)
    calibration = Calibration.load(pipeline.layout.calibration)
    return explain(pipeline.evaluation_report(), query_id, calibration=calibration, k=k), pipeline.dataset


def _layout(run_dir: str | Path) -> RunLayout:
    layout = RunLayout.at(run_dir)
    if not Path(layout.manifest).exists():
        raise MissingInputError(
            f"no manifest at {layout.root}", hint="pass a run directory (runs/<run_id>), not the runs root"
        )
    return layout


def _pipeline(run_dir: str | Path):
    """The pipeline of a finished run, reopened read-only; refuses a run without a calibration."""
    from rcp_ndcg.runs.pipeline import Pipeline

    layout = _layout(run_dir)
    for required in (layout.candidates, layout.path("calibration", "items.json")):
        if not Path(required).exists():
            raise MissingInputError(
                f"{layout.run_id} is missing {layout.relative(required)}",
                hint="finish the run's calibrate step first",
            )
    return Pipeline.resume(layout.root)


__all__ = [
    "evaluation_report",
    "explain_query",
    "get_run",
    "list_runs",
    "step_states",
]
