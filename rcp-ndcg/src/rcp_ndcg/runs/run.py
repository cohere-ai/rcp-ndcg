"""A run as an object: :class:`Run` over a run directory, and the steps that fill it.

:func:`prepare` builds the :class:`~rcp_ndcg.runs.pipeline.Pipeline` of a new run, :func:`reopen` that of a run
directory, and :func:`execute_run` runs one in this process, writing the pipeline's log to ``logs/run.log``. The
one entry point for running a config is :func:`rcp_ndcg.run` (:func:`rcp_ndcg.runs.execution.run`), which also
hands a run to a job runner (SLURM, Kubernetes, a plugin).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from rcp_ndcg.errors import DataError, MissingInputError
from rcp_ndcg.runners.base import JobStatus
from rcp_ndcg.runs.config import RunConfig
from rcp_ndcg.runs.layout import RunLayout
from rcp_ndcg.runs.manifest import RunManifest, RunStatus, StepStatus
from rcp_ndcg.runs.mirror import MirrorState, check_target, mirrored, read_state
from rcp_ndcg.support.logging import BASE_LOGGER_NAME
from rcp_ndcg.support.serve import EngineRole, EngineURLs

#: The run statuses after which nothing more happens without a new command.
TERMINAL = frozenset({RunStatus.COMPLETED, RunStatus.PARTIAL, RunStatus.FAILED, RunStatus.CANCELLED})


class StepProgress(BaseModel):
    """How far a judging step has got, in judge windows.

    Attributes:
        done: Windows judged and stored so far.
        planned: The windows the step's schedule asks at most for its pools (the tournament's adaptive phase can
            stop early); ``None`` until the pools and the schedule are known.
    """

    done: int
    planned: int | None = None


class StepState(BaseModel):
    """One planned step of a run: its status, time, error and, for a judging step, its progress."""

    name: str
    status: StepStatus = Field(description="pending (planned, not started), running, completed, failed or cancelled.")
    duration_s: float | None = None
    error: str | None = None
    progress: StepProgress | None = Field(default=None, description="Judge windows done and planned (judging steps).")


class JobState(BaseModel):
    """One job a runner was handed for the run."""

    name: str
    handle: str
    status: JobStatus = Field(
        description="pending, running, completed, failed, cancelled or unknown (a runner that cannot say)."
    )


class RunState(BaseModel):
    """Where a run stands: its status, every step, its judge requests, and its jobs when a runner holds it."""

    run_id: str
    run_dir: str
    status: RunStatus = Field(description="submitted, running, completed, partial, failed or cancelled.")
    done: bool = Field(
        description="Whether the run is done: its status is terminal and no job of it is still running (or "
        "unknown to its runner)."
    )
    steps: list[StepState] = Field(description="Every planned step in run order; one not started yet is pending.")
    requests: int = Field(description="Judge requests made so far.")
    metrics: dict[str, float] = Field(
        description="`<system>/<metric>@<k>` -> value of every system but the judge's own order (the full report "
        "with confidence intervals: metrics/report.json)."
    )
    runner: str | None = Field(default=None, description="The job runner holding the run, if any.")
    jobs: list[JobState] = Field(default_factory=list)
    mirror: MirrorState | None = Field(default=None, description="The mirror's last upload and lag, if mirrored.")
    note: str | None = Field(
        default=None,
        description="Why the state is not the local manifest's own: read from a newer mirror, or derived from jobs "
        "that ended without recording it; and a submission that failed.",
    )


#: The names of a run's artifacts, in :meth:`Run.artifacts` and ``run show``.
ARTIFACTS = ("config", "candidates", "tournament", "rubric", "calibration", "report", "comparison", "log", "jobs")


class Run:
    """A run directory: its manifest, its config and its artifacts.

    Args:
        run_dir: The run directory (``<runs dir>/<run_id>``).

    Raises:
        MissingInputError: The directory holds no manifest.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.layout = RunLayout.at(run_dir)
        if not Path(self.layout.manifest).exists():
            raise MissingInputError(
                f"no manifest at {self.layout.root}", hint="pass a run directory (runs/<run_id>), not the runs root"
            )

    @property
    def dir(self) -> str:
        """The run directory."""
        return self.layout.root

    @property
    def manifest(self) -> RunManifest:
        """The manifest, read now."""
        return RunManifest.load(self.layout)

    @property
    def config(self) -> RunConfig:
        """The run's resolved config."""
        return RunConfig.from_data(self.manifest.config)

    def artifacts(self) -> dict[str, str]:
        """``{name: path}`` of the layout's artifacts that exist (the names: :data:`ARTIFACTS`)."""
        layout = self.layout
        paths = {
            "config": layout.config,
            "candidates": layout.candidates,
            "tournament": layout.path("judgements", "tournament.jsonl"),
            "rubric": layout.path("judgements", "rubric.jsonl"),
            "calibration": layout.calibration,
            "report": layout.metrics,
            "comparison": layout.comparison,
            "log": layout.log,
            "jobs": layout.jobs,
        }
        present = {name: path for name, path in paths.items() if Path(path).exists()}
        if "calibration" in present and not Path(layout.path("calibration", "items.json")).exists():
            del present["calibration"]
        return present

    def jobs(self) -> dict[str, Any] | None:
        """The jobs record (``logs/jobs.json``): ``{"runner", "options", "jobs": [{"name", "handle"}]}``.

        Raises:
            DataError: the file does not parse, or does not have the record's shape (a hand edit, another
                writer); the message names the file, so ``run status``, ``run logs`` and ``run cancel`` say
                what is broken.
        """
        path = Path(self.layout.jobs)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise self._damaged_jobs(path, f"it is not valid JSON: {exc}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("runner"), str):
            raise self._damaged_jobs(path, "it names no runner")
        if not isinstance(record.get("options", {}), dict):
            raise self._damaged_jobs(path, "its options are not a mapping")
        jobs = record.get("jobs")
        if not isinstance(jobs, list) or not jobs:
            raise self._damaged_jobs(path, "it lists no jobs")
        for job in jobs:
            if not isinstance(job, dict) or not isinstance(job.get("name"), str) or "handle" not in job:
                raise self._damaged_jobs(path, f"a job entry is not {{name, handle}}: {job!r}")
            if job["handle"] is not None and not isinstance(job["handle"], str):
                raise self._damaged_jobs(path, f"a job handle is not a string: {job['handle']!r}")
        return record

    def _damaged_jobs(self, path: Path, why: str) -> DataError:
        """The typed error a damaged job record raises, naming the file and the way out."""
        return DataError(
            f"{path} is not a job record: {why}",
            hint="the job record is damaged; delete it and submit the run again, or restore the run directory "
            "from its mirror",
        )

    def status(self) -> RunState:
        """The run's state as its manifest records it (a runner's live job states and a newer mirror:
        :func:`rcp_ndcg.runs.execution.status`)."""
        return self.state(self.manifest)

    def state(self, manifest: RunManifest) -> RunState:
        """The run's state as ``manifest`` records it (this run's manifest, or its mirror's copy)."""
        from rcp_ndcg.runs.inspect import step_states

        jobs = self.jobs()
        return RunState(
            run_id=manifest.run_id,
            run_dir=self.dir,
            status=manifest.status,
            done=manifest.status in TERMINAL,
            steps=step_states(self.layout, manifest),
            requests=manifest.usage.requests,
            metrics=manifest.metrics,
            mirror=read_state(self.layout.mirror_state),
            runner=jobs["runner"] if jobs else None,
            jobs=[
                JobState(name=j["name"], handle=j["handle"] or "", status=JobStatus.UNKNOWN)
                for j in (jobs or {}).get("jobs", [])
            ],
        )

    def log(self, *, tail: int | None = None) -> str:
        """The pipeline's log (``logs/run.log``); ``tail`` keeps the last lines.

        Raises:
            MissingInputError: The run has no log (it was not run in this process's way yet).
        """
        if not Path(self.layout.log).exists():
            raise MissingInputError(f"{self.layout.run_id} has no log yet", hint="the run has not started")
        lines = Path(self.layout.log).read_text(encoding="utf-8").splitlines(keepends=True)
        return "".join(lines if tail is None else lines[-tail:] if tail > 0 else [])

    def __repr__(self) -> str:
        return f"Run({self.dir!r})"


@contextmanager
def _logged(layout: RunLayout) -> Iterator[None]:
    """Also write the package's log records to the run's ``logs/run.log`` while the block runs."""
    Path(layout.log).parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(layout.log, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    logger = logging.getLogger(BASE_LOGGER_NAME)
    previous = logger.level
    logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
        handler.close()


def prepare(
    config: RunConfig | str | Path,
    *,
    overrides: Sequence[str] = (),
    runs_dir: str | None = None,
    label: str | None = None,
    steps: Sequence[str] | None = None,
):
    """The :class:`~rcp_ndcg.runs.pipeline.Pipeline` of a new run of ``config`` (nothing written yet).

    Args:
        config: A :class:`RunConfig`, or the path of a run config YAML.
        overrides: ``key=value`` overrides applied to the config.
        runs_dir: Where the run directory is created; default ``$RCP_NDCG_RUNS_DIR``, else ``runs``.
        label: The readable fragment of the run id (overrides ``label``).
        steps: Run only these steps (overrides ``steps``).
    """
    from rcp_ndcg.runs.pipeline import Pipeline

    if isinstance(config, RunConfig):
        data = config.resolved()
    else:
        data = RunConfig.load(config, overrides=overrides).resolved()
        overrides = ()
    updates = {"label": label, "steps": list(steps) if steps else None}
    data.update({key: value for key, value in updates.items() if value is not None})
    return Pipeline(RunConfig.from_data(data, overrides=overrides), runs_dir=runs_dir)


def reopen(
    run_dir: str | Path,
    *,
    overrides: Sequence[str] = (),
    steps: Sequence[str] | None = None,
    engines: Mapping[EngineRole, EngineURLs] | None = None,
):
    """The :class:`~rcp_ndcg.runs.pipeline.Pipeline` of an existing run directory, with changes applied.

    Args:
        run_dir: The run directory.
        overrides: ``key=value`` overrides applied to the recorded config (each value a YAML literal). They are
            kept only when the resumed run succeeds (:meth:`~rcp_ndcg.runs.pipeline.Pipeline.run`).
        steps: Run only these steps now; the run's recorded ``steps`` are not changed.
        engines: The engines overlay of this invocation (a phase's ``RCP_NDCG_ENGINES``, or ``run resume
            --engine``); ``None`` reads the environment variable.
    """
    from rcp_ndcg.runs.pipeline import Pipeline

    return Pipeline.resume(run_dir, overrides=list(overrides), only=steps, engines=engines)


def execute_run(pipeline, *, resume: bool = True) -> Run:
    """Run a prepared pipeline in this process, logging to the run's ``logs/run.log``; return the :class:`Run`.

    With a ``mirror`` in the config, the run directory is restored from it first and mirrored while the pipeline
    runs (:func:`rcp_ndcg.runs.mirror.mirrored`). A mirror no installed filesystem serves is refused before the
    run directory is created.
    """
    config = pipeline.config
    if config.mirror is not None:
        check_target(config.mirror)
    layout = pipeline.layout.ensure()
    with ExitStack() as stack:
        if config.mirror is not None:
            stack.enter_context(
                mirrored(
                    layout.root, config.mirror, interval_s=config.mirror_interval_s, state_file=layout.mirror_state
                )
            )
        stack.enter_context(_logged(layout))
        pipeline.run(resume=resume)
    return Run(layout.root)


def mark(run: Run, status: RunStatus) -> None:
    """Record ``status`` as the run's status in its manifest.

    A ``cancelled`` or ``failed`` run closes the steps it recorded as running, with the same status.
    """
    manifest = run.manifest
    manifest.status = status
    if status in (RunStatus.CANCELLED, RunStatus.FAILED):
        closing = StepStatus.CANCELLED if status is RunStatus.CANCELLED else StepStatus.FAILED
        for record in manifest.steps:
            if record.status is StepStatus.RUNNING:
                manifest.finish_step(record.name, status=closing, error=f"the run was {status.value}")
    manifest.save(run.layout)


__all__ = [
    "ARTIFACTS",
    "TERMINAL",
    "JobState",
    "Run",
    "RunState",
    "StepProgress",
    "StepState",
    "execute_run",
    "mark",
    "prepare",
    "reopen",
]
