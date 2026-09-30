"""Runs handed to job runners, and :func:`run`, the one entry point for running a config.

A runner transports one job per run: its argv is ``rcp-ndcg run resume --run <run_dir>``, so the job re-enters
the same pipeline, resume logic included, wherever it lands. The runner's options and the job are validated first;
only then is the run directory created (config and manifest written, status ``submitted``), and the job record
``logs/jobs.json`` names the runner, its options and the handles; ``run status``, ``run logs`` and ``run cancel``
ask the runner through it. A runner whose jobs do not see this host's files (``run_root``: Kubernetes) gets the run
through its mirror: the prepared directory is uploaded to the mirror before the job is submitted, and the job
restores it into ``run_root``.

A run config's ``serve:`` (:class:`~rcp_ndcg.support.serve.ServeConfig`) goes into the job, and the runner starts the
judge's engine beside it; the local runner and a run in this process start none, and refuse it.

Three keys of a run config's ``runner.options`` (typed per runner: :mod:`rcp_ndcg.runs.config`) describe the job
rather than the runner (:data:`JOB_OPTIONS`): ``resources`` (:class:`~rcp_ndcg.runners.Resources`: ``gpus``,
``cpus``, ``memory_gb``, ``time_limit_s``), ``image`` and ``env``. They go into the run's
:class:`~rcp_ndcg.runners.JobSpec`; the other options configure the runner::

    runner:
      name: slurm
      options:
        partition: gpu
        resources: {gpus: 1, cpus: 8, memory_gb: 64, time_limit_s: 86400}
        env: {HF_HOME: /scratch/hf}
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rcp_ndcg.errors import ConfigError, MissingInputError
from rcp_ndcg.runners.base import JobRunner, JobSpec, JobStatus
from rcp_ndcg.runners.registry import get_runner
from rcp_ndcg.runs.config import RunConfig
from rcp_ndcg.runs.layout import RunLayout, slugify
from rcp_ndcg.runs.manifest import RunStatus
from rcp_ndcg.runs.run import JobState, Run, RunState, execute_run, mark, prepare, reopen

if TYPE_CHECKING:
    from rcp_ndcg.llm.cost import CostEstimate


#: The ``runner.options`` keys that describe the run's job (its :class:`JobSpec` fields), not the runner.
JOB_OPTIONS: tuple[str, ...] = ("resources", "image", "env")


def _split_options(options: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(runner options, job fields)`` of a run config's ``runner.options``."""
    runner = {key: value for key, value in options.items() if key not in JOB_OPTIONS}
    job = {key: value for key, value in options.items() if key in JOB_OPTIONS}
    return runner, job


def run_argv(run_dir: str, mirror: str | None = None, outage_timeout_s: int | None = None) -> tuple[str, ...]:
    """The command a job runs to execute a prepared run directory (restored from ``mirror`` when it is missing).

    A job that starts its judge's engine passes ``outage_timeout_s`` (``serve.outage_timeout_s``): its judge then
    stops waiting for an engine that stopped answering after that many seconds (``judge.wait_on_outage_s``).
    """
    wait = ("--set", f"judge.wait_on_outage_s={outage_timeout_s}") if outage_timeout_s is not None else ()
    return ("rcp-ndcg", "run", "resume", "--run", run_dir, *(("--mirror", mirror) if mirror else ()), *wait)


def job_for(
    pipeline: Any, runner: str, options: Mapping[str, Any] | None = None
) -> tuple[JobRunner, JobSpec, dict[str, Any]]:
    """The runner a prepared run would be handed to, its job, and the runner's options; nothing is written.

    Args:
        pipeline: A prepared :class:`~rcp_ndcg.runs.pipeline.Pipeline` (:func:`rcp_ndcg.runs.run.prepare`).
        runner: A runner name (``local``, ``slurm``, ``kubernetes`` or an installed plugin).
        options: Options added to the config's ``runner.options`` (those apply when the config names ``runner``).

    Raises:
        ConfigError: the runner's options or the job's fields do not validate, or a runner whose jobs do not see
            this host's files is given a run without a mirror.
    """
    from pydantic import ValidationError

    config: RunConfig = pipeline.config
    layout: RunLayout = pipeline.layout
    configured = config.runner.option_values() if config.runner.name == runner else {}
    runner_options, fields = _split_options({**configured, **(options or {})})
    if runner == "local" and "log_dir" not in runner_options:
        runner_options = {**runner_options, "log_dir": layout.logs_dir}  # keep the job's output with the run
    backend = get_runner(runner, **runner_options)
    run_root = getattr(backend, "run_root", None)
    if run_root is not None and config.mirror is None:
        raise ConfigError(
            f"the {runner} runner's jobs do not see this host's files, so the run reaches its job through a mirror, "
            "and this run has none",
            hint="set mirror: <any fsspec URI the job can reach> (e.g. s3://bucket/runs/name), or pass --mirror",
        )
    run_dir = layout.root if run_root is None else f"{run_root}/{layout.run_id}"
    try:
        job = JobSpec(
            name=slugify(f"rcp-{layout.run_id}", max_length=60),
            argv=run_argv(run_dir, config.mirror, config.serve.outage_timeout_s if config.serve else None),
            serve=config.serve,
            **fields,
        )
    except ValidationError as exc:
        raise ConfigError(f"runner.options: {exc}", hint=f"the job's keys are {', '.join(JOB_OPTIONS)}") from exc
    return backend, job, runner_options


def render_run(pipeline: Any, runner: str, options: Mapping[str, Any] | None = None) -> dict[str, str]:
    """What ``runner`` would submit for a prepared run (an sbatch script, Kubernetes manifests); nothing is written.

    Raises:
        ConfigError: as :func:`job_for`, or the runner cannot render its jobs.
    """
    backend, job, _ = job_for(pipeline, runner, options)
    render = getattr(backend, "render", None)
    if render is None:
        raise ConfigError(f"the {runner!r} runner cannot show what it would submit (it has no render method)")
    return render([job])


def submit_run(pipeline: Any, runner: str, options: Mapping[str, Any] | None = None) -> Run:
    """Create the run directory of a prepared pipeline and hand it to ``runner`` as one job.

    Everything is validated before anything is written: a bad option leaves no run directory behind.

    Args:
        pipeline: A prepared :class:`~rcp_ndcg.runs.pipeline.Pipeline` (:func:`rcp_ndcg.runs.run.prepare`).
        runner: A runner name (``local``, ``slurm``, ``kubernetes`` or an installed plugin).
        options: Options added to the config's ``runner.options`` (those apply when the config names ``runner``).

    Returns:
        The :class:`~rcp_ndcg.runs.run.Run` (status ``submitted`` until the job starts it).

    Raises:
        ConfigError: the runner's options or the job's fields do not validate (e.g. an unknown resource), or the
            runner cannot run the job as configured (e.g. ``serve:`` on the local runner).
    """
    from rcp_ndcg.runs.mirror import Mirror

    backend, job, runner_options = job_for(pipeline, runner, options)
    render = getattr(backend, "render", None)
    if render is not None:
        render([job])  # a runner refuses what it cannot run while nothing is written yet
    config: RunConfig = pipeline.config
    layout = pipeline.layout.ensure()
    pipeline._write_config()
    pipeline.manifest.status = RunStatus.SUBMITTED
    pipeline.manifest.save(layout)
    record = {"runner": runner, "options": runner_options, "jobs": [{"name": job.name, "handle": None}]}
    Path(layout.jobs).write_text(json.dumps(record, indent=2), encoding="utf-8")
    if getattr(backend, "run_root", None) is not None and config.mirror is not None:
        Mirror(layout.root, config.mirror).flush()  # the job restores the run from here
    (handle,) = backend.submit([job])
    record["jobs"][0]["handle"] = handle
    Path(layout.jobs).write_text(json.dumps(record, indent=2), encoding="utf-8")
    return Run(layout.root)


def _backend(run: Run) -> tuple[Any, dict[str, Any]]:
    record = run.jobs()
    if record is None:
        raise MissingInputError(f"{run.layout.run_id} was not handed to a runner", hint="it runs in-process")
    return get_runner(record["runner"], **record.get("options", {})), record


def status(run_dir: str | Path) -> RunState:
    """The run's state, with each job's live status asked from its runner when a runner holds the run."""
    run = Run(run_dir)
    state = run.status()
    if state.runner is None:
        return state
    backend, record = _backend(run)
    jobs = []
    for job in record["jobs"]:
        live = backend.status(job["handle"]) if job["handle"] else JobStatus.UNKNOWN
        jobs.append(JobState(name=job["name"], handle=job["handle"] or "", status=JobStatus(live).value))
    return state.model_copy(update={"jobs": jobs})


def logs(run_dir: str | Path, *, tail: int | None = None) -> str:
    """The run's log: its jobs' output from the runner, else the pipeline's ``logs/run.log``."""
    run = Run(run_dir)
    if run.jobs() is None:
        return run.log(tail=tail)
    backend, record = _backend(run)
    return "".join(backend.logs(job["handle"], tail=tail) for job in record["jobs"] if job["handle"])


def cancel(run_dir: str | Path) -> RunState:
    """Cancel the run's jobs and record the run as ``cancelled``.

    Raises:
        MissingInputError: The run was not handed to a runner (an in-process run stops with Ctrl-C).
    """
    run = Run(run_dir)
    backend, record = _backend(run)
    for job in record["jobs"]:
        if job["handle"]:
            backend.cancel(job["handle"])
    if run.manifest.status in (RunStatus.SUBMITTED, RunStatus.RUNNING):
        mark(run, RunStatus.CANCELLED)
    return status(run_dir)


def run(
    config: RunConfig | str | Path,
    *,
    runner: str | None = None,
    resume: bool = True,
    budget_usd: float | None = None,
    estimate: bool = False,
    overrides: Sequence[str] = (),
    runs_dir: str | None = None,
) -> Run | CostEstimate:
    """Run a config: in this process (``runner="local"``), or as one job of another runner.

    Args:
        config: A :class:`RunConfig`, the path of a run config YAML, or an existing run directory (resumed).
        runner: ``"local"`` runs in this process; another name hands the run to that job runner. ``None`` takes
            the config's ``runner.name`` for a new run. An existing run directory is resumed in this process
            (that is what a submitted job does).
        resume: Reuse the steps a resumed run already completed with the same identity and inputs.
        budget_usd: The spend ceiling of the judging steps, USD.
        estimate: Price the judging steps instead of running anything.
        overrides: ``key=value`` config overrides.
        runs_dir: Where a new run's directory is created; default ``$RCP_NDCG_RUNS_DIR``, else ``runs``.

    Returns:
        The :class:`~rcp_ndcg.runs.run.Run`; with ``estimate=True``, the :class:`~rcp_ndcg.llm.CostEstimate` of
        its judging steps (calls, tokens, USD, wall time) instead, and nothing runs.
    """
    existing = not isinstance(config, RunConfig) and Path(RunLayout.at(config).manifest).exists()
    if existing:
        pipeline = reopen(config, overrides=overrides, budget_usd=budget_usd)  # type: ignore[arg-type]
    else:
        pipeline = prepare(config, overrides=overrides, budget_usd=budget_usd, runs_dir=runs_dir)
    if estimate:
        return pipeline.estimate()
    if existing and runner not in (None, "local"):
        raise ConfigError(
            "a run directory is resumed in this process",
            hint="use runner='local'",
            cli_hint="resume it with `rcp-ndcg run resume --run DIR` (it runs in this process)",
        )
    if runner is None:
        runner = "local" if existing else pipeline.config.runner.name
    if runner == "local":
        if not existing:
            refuse_serving_here(pipeline.config)
        return execute_run(pipeline, resume=resume)
    return submit_run(pipeline, runner)


def refuse_serving_here(config: RunConfig) -> None:
    """Refuse to start a run with a ``serve:`` section in this process, which starts no engine.

    A job of the ``slurm`` or ``kubernetes`` runner resumes its run here with the engine already up beside it, so
    only a new run is refused.

    Raises:
        ConfigError: ``config`` has a ``serve:`` section.
    """
    if config.serve is not None:
        raise ConfigError(
            "this run has a serve: section, and a run in this process (or on the local runner) starts no engine",
            hint="start the engine yourself (docs/concepts/serving.md) and pass its URL with --judge-url and "
            "--judge-model; or hand the run to a runner that starts it: --runner slurm | kubernetes",
        )


__all__ = [
    "JOB_OPTIONS",
    "cancel",
    "job_for",
    "logs",
    "refuse_serving_here",
    "render_run",
    "run",
    "run_argv",
    "status",
    "submit_run",
]
