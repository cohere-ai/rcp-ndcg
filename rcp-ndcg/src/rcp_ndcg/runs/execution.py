"""Runs handed to job runners, and :func:`run`, the one entry point for running a config.

A runner transports one job per run: the job runs ``rcp-ndcg run resume --run <run_dir>``, phase by phase, so it
re-enters the same pipeline, resume logic included, wherever it lands. The runner's options and the job are
validated first; only then is the run directory created (config and manifest written, status ``submitted``), and
the job record ``logs/jobs.json`` names the runner, its options and the handles; ``run status``, ``run logs`` and
``run cancel`` ask the runner through it. A runner whose jobs do not see this host's files (``run_root``:
Kubernetes) gets the run through its mirror: the prepared directory is uploaded to the mirror before the job is
submitted, and the job restores it into ``run_root``.

A run config's ``serve:`` (:class:`~rcp_ndcg.support.serve.ServeByRole`) becomes the job's
:class:`~rcp_ndcg.runners.JobPhase` s (:func:`rcp_ndcg.support.serve.plan_phases`): the job runs the run in
phases, each starting only the engines its steps use and handing their URLs to the coordinator through
``RCP_NDCG_ENGINES``. The local runner starts no engine; a run in this process starts none either, and both
refuse a phase that would start one.

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
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rcp_ndcg.errors import ConfigError, MissingInputError
from rcp_ndcg.runners.base import JobOptions, JobRunner, JobSpec, JobStatus
from rcp_ndcg.runners.registry import get_runner
from rcp_ndcg.runs.config import RunConfig
from rcp_ndcg.runs.layout import MANIFEST_NAME, RunLayout, slugify
from rcp_ndcg.runs.manifest import RunManifest, RunStatus
from rcp_ndcg.runs.run import JobState, Run, RunState, execute_run, mark, prepare, reopen
from rcp_ndcg.storage import publish_bytes
from rcp_ndcg.support.resources import Resources
from rcp_ndcg.support.serve import Phase, ServeByRole
from rcp_ndcg.support.urls import redact_urls, safe_url

if TYPE_CHECKING:
    from rcp_ndcg.judging.cost import CostEstimate


#: The ``runner.options`` keys that describe the run's job (its :class:`JobSpec` fields), not the runner.
JOB_OPTIONS: tuple[str, ...] = ("resources", "image", "env")


def _split_options(options: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(runner options, job fields)`` of a run config's ``runner.options``."""
    runner = {key: value for key, value in options.items() if key not in JOB_OPTIONS}
    job = {key: value for key, value in options.items() if key in JOB_OPTIONS}
    return runner, job


def _set_job_fields(options: Mapping[str, Any]) -> list[str]:
    """The job fields (``JOB_OPTIONS``) a config's ``runner.options`` declares, defaults excluded.

    A resolved config carries its runner options' defaults (an empty ``env``, ``resources: {gpus: 0}``): those
    are not a declaration, so only a non-empty ``env``, an ``image`` and a non-default ``resources`` count. A
    ``resources`` value that is not even a mapping counts too: the job would refuse it, and silently dropping it
    when another runner is in use would hide the typo.
    """
    found: list[str] = []
    if options.get("env"):
        found.append("env")
    if options.get("image") is not None:
        found.append("image")
    resources = options.get("resources")
    if isinstance(resources, Resources):
        resources = resources.model_dump()  # an in-memory default instance is no more a declaration than {"gpus": 0}
    if resources not in (None, {}):
        if not isinstance(resources, Mapping):
            found.append("resources")
        else:
            try:
                declared = Resources.model_validate(resources).model_dump(exclude_defaults=True)
            except ValueError:
                declared = dict(resources)  # not a Resources mapping: the job would refuse it anyway
            if declared:
                found.append("resources")
    return sorted(found)


def run_argv(run_dir: str, mirror: str | None = None, only: Sequence[str] = ()) -> tuple[str, ...]:
    """The command one phase of a job runs to execute a prepared run directory (restored from ``mirror`` when it
    is missing), narrowed to the phase's steps with ``--only``.

    A phase's engines reach the command through ``RCP_NDCG_ENGINES``, which the coordinator applies as a runtime
    overlay; nothing about them is passed on the command line.
    """
    argv = ["rcp-ndcg", "run", "resume", "--run", run_dir]
    if mirror:
        argv += ["--mirror", mirror]
    argv += [flag for step in only for flag in ("--only", step)]
    return tuple(argv)


def job_for(
    pipeline: Any, runner: str, options: Mapping[str, Any] | None = None
) -> tuple[JobRunner, JobSpec, dict[str, Any]]:
    """The runner a prepared run would be handed to, its job, and the runner's options; nothing is written.

    Every path the job or its record names is absolute (the run directory, a public runner's ``log_dir``, ``cwd``
    and ``workdir``), so the job finds the run wherever it starts, and ``run status``, ``run logs`` and
    ``run cancel`` find the job from any working directory.

    Args:
        pipeline: A prepared :class:`~rcp_ndcg.runs.pipeline.Pipeline` (:func:`rcp_ndcg.runs.run.prepare`).
        runner: A runner name (``local``, ``slurm``, ``kubernetes`` or an installed plugin).
        options: Options added to the config's ``runner.options`` (those apply when the config names ``runner``).

    Raises:
        ConfigError: the runner's options or the job's fields do not validate; or a runner whose jobs do not see
            this host's files is given a run without a mirror, or a run that reads inputs from this host.
        DependencyError: no installed filesystem serves the run's mirror.
    """
    from rcp_ndcg.runs.mirror import check_target
    from rcp_ndcg.support.serve import ServeByRole, plan_phases

    config: RunConfig = pipeline.config
    layout: RunLayout = pipeline.layout
    root = os.path.abspath(layout.root)
    configured = config.runner.option_values() if config.runner.name == runner else {}
    if config.runner.name != runner:
        # The job fields (resources, image, env) describe the job and are only read when the config names the
        # runner in use; silently dropping them once gave a job a different resource request, image or env than
        # its config declared (a lost time limit holds GPUs indefinitely). Only fields the config *set* count:
        # a resolved config carries its options' defaults, and those are no one's declaration.
        dropped = _set_job_fields(config.runner.option_values())
        if dropped:
            raise ConfigError(
                f"the config names the {config.runner.name!r} runner and sets its job fields "
                f"({', '.join(dropped)}), and this run is handed to the {runner!r} runner",
                hint=f"the job fields describe the job and are read when the config names the runner in use: set "
                f"runner.name: {runner} and put {', '.join(dropped)} under its options, or drop them",
            )
    runner_options, fields = _split_options({**configured, **(options or {})})
    if runner == "local":  # keep the job's output with the run, and run it where it was started
        runner_options = {"log_dir": str(Path(root) / "logs"), "cwd": os.getcwd(), **runner_options}
    backend = get_runner(runner, **runner_options)
    parsed = getattr(backend, "options", None)
    if isinstance(parsed, JobOptions):
        runner_options = parsed.resolved()
        backend = get_runner(runner, **runner_options)
    run_root = getattr(backend, "run_root", None)
    if run_root is not None:
        _refuse_host_inputs(config, runner)
    if config.mirror is not None:
        check_target(config.mirror)
    run_dir = root if run_root is None else f"{run_root}/{layout.run_id}"
    serve = config.serve or ServeByRole()
    phases = plan_phases(config.ordered_steps, serve, config.engine_uses())
    job = _phased_job(
        runner,
        backend,
        name=slugify(f"rcp-{layout.run_id}", max_length=60),
        run_dir=run_dir,
        mirror=config.mirror,
        phases=phases,
        serve=serve,
        fields=fields,
    )
    return backend, job, runner_options


def _phased_job(
    runner: str,
    backend: Any,
    *,
    name: str,
    run_dir: str,
    mirror: str | None,
    phases: Sequence[Phase],
    serve: ServeByRole,
    fields: dict[str, Any],
) -> JobSpec:
    """The :class:`~rcp_ndcg.runners.JobSpec` of the run's phase plan: one :class:`~rcp_ndcg.runners.JobPhase`
    per planned phase, with the engines it starts and its coordinator argv. A runner that renders phases takes
    the phases (the job's commands are their ``argv``); one that does not takes the whole-run command as the
    job's ``argv`` (which is also every engine-free phase, in order).

    Raises:
        ConfigError: the job's fields do not validate, or ``runner`` cannot render a job that starts engines
            (only a runner that renders phases runs a phase that starts engines).
    """
    from pydantic import ValidationError

    from rcp_ndcg.runners.base import JobPhase

    if any(phase.engines for phase in phases) and not getattr(backend, "renders_phases", False):
        raise ConfigError(
            f"the {runner} runner does not start a phase's engines (it renders no phases)",
            hint="start the engines yourself (docs/concepts/runs.md) and resume with --engine <role>=<url>[,<url>]",
        )
    if phases and getattr(backend, "renders_phases", False):
        commands: dict[str, Any] = {
            "phases": tuple(
                JobPhase(
                    engines={role: getattr(serve, role) for role in phase.engines},
                    argv=run_argv(run_dir, mirror, only=list(phase.steps)),
                )
                for phase in phases
            )
        }
    else:  # no phases planned, or a runner that runs the whole-run command and renders no phases
        commands = {"argv": run_argv(run_dir, mirror)}
    try:
        return JobSpec(name=name, **fields, **commands)
    except ValidationError as exc:
        raise ConfigError(f"runner.options: {exc}", hint=f"the job's keys are {', '.join(JOB_OPTIONS)}") from exc


def _refuse_host_inputs(config: RunConfig, runner: str) -> None:
    """Refuse a run whose job does not see this host's files, when it needs a mirror or reads inputs from here."""
    if config.mirror is None:
        raise ConfigError(
            f"the {runner} runner's jobs do not see this host's files, so the run reaches its job through a mirror, "
            "and this run has none",
            hint="set mirror: <any fsspec URI the job can reach> (e.g. s3://bucket/runs/name), or pass --mirror",
        )
    local = config.local_inputs()
    if local:
        raise ConfigError(
            f"the {runner} runner's jobs do not see this host's files, and this run reads {len(local)} input(s) "
            f"from them: {'; '.join(local)}",
            hint="put the inputs where the job can read them and name them by URI (hf://, s3://, gs://, https://); "
            "a judge or prompt by its shipped name, or inline in the run config",
            details={"inputs": local},
        )


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


def stage_run(pipeline: Any) -> RunLayout:
    """Stage ``pipeline``'s run directory for a job that will run it: the layout, the resolved ``run.yaml`` (the
    pipeline's own writer) and the manifest in status ``submitted``.

    The one staging step: :func:`submit_run` calls it before it hands the job to a runner, and a driver that
    runs the job itself (the GPU end-to-end driver, whose job re-enters ``run resume`` on this directory)
    stages exactly the same way.

    Args:
        pipeline: The run's :class:`~rcp_ndcg.runs.pipeline.Pipeline` (from :func:`~rcp_ndcg.runs.run.prepare`).

    Returns:
        The run's layout, its directories created.
    """
    layout = pipeline.layout.ensure()
    pipeline._write_config()
    pipeline.manifest.status = RunStatus.SUBMITTED
    pipeline.manifest.save(layout)
    return layout


def submit_run(pipeline: Any, runner: str, options: Mapping[str, Any] | None = None) -> Run:
    """Hand a prepared run to ``runner`` as one job: a new run's directory is created first, an existing run's
    directory is submitted again (``run resume --runner``).

    Everything is validated before anything is written: a bad option leaves no run directory behind. A run that
    is submitted again must have no job still pending or running. If the submission itself fails (the scheduler's
    tool is missing, it refuses the job), the run is recorded ``failed`` with the error in its job record.

    Args:
        pipeline: A prepared :class:`~rcp_ndcg.runs.pipeline.Pipeline` (:func:`rcp_ndcg.runs.run.prepare`), or a
            reopened one (:func:`rcp_ndcg.runs.run.reopen`).
        runner: A runner name (``local``, ``slurm``, ``kubernetes`` or an installed plugin).
        options: Options added to the config's ``runner.options`` (those apply when the config names ``runner``).

    Returns:
        The :class:`~rcp_ndcg.runs.run.Run` (status ``submitted`` until the job starts it).

    Raises:
        ConfigError: the runner's options or the job's fields do not validate (e.g. an unknown resource), the
            runner cannot run the job as configured (e.g. ``serve:`` on the local runner), or a job of the run is
            still pending or running.
    """
    from rcp_ndcg.runs.mirror import Mirror

    layout = pipeline.layout
    if Path(layout.manifest).exists():
        _refuse_live_jobs(Run(layout.root))
    backend, job, runner_options = job_for(pipeline, runner, options)
    render = getattr(backend, "render", None)
    if render is not None:
        render([job])  # a runner refuses what it cannot run while nothing is written yet
    config: RunConfig = pipeline.config
    layout = stage_run(pipeline)
    record: dict[str, Any] = {
        "runner": runner,
        "options": runner_options,
        "jobs": [{"name": job.name, "handle": None}],
        # The submission is in flight: a process killed before the handle is written leaves this flag, and a
        # resubmission must refuse rather than start a second job over the first.
        "submitting": True,
    }
    _write_record(layout, record)
    try:
        if getattr(backend, "run_root", None) is not None and config.mirror is not None:
            # The job restores the run from here.
            Mirror(layout.root, config.mirror, state_file=layout.mirror_state).flush()
        (handle,) = backend.submit([job])
    except BaseException as exc:
        # A scheduler error names the request's URL (kubectl, an fsspec store): a credential in it must not
        # reach logs/jobs.json (host-local, owner-only and never mirrored), which `run status` prints as its note.
        record["error"] = redact_urls(f"{type(exc).__name__}: {exc}")
        record.pop("submitting", None)
        _write_record(layout, record)
        mark(Run(layout.root), RunStatus.FAILED)
        raise
    record["jobs"][0]["handle"] = handle
    record.pop("submitting", None)
    _write_record(layout, record)
    return Run(layout.root)


def _write_record(layout: RunLayout, record: dict[str, Any]) -> None:
    """Write ``logs/jobs.json`` atomically and owner-only: it names the runner, its options and the job handles,
    and it is host-local state (the mirror never uploads or restores it)."""
    publish_bytes(layout.jobs, json.dumps(record, indent=2).encode("utf-8"), mode=0o600)


def _refuse_live_jobs(run: Run) -> None:
    """Refuse a resubmission the record cannot prove is safe: a job still pending or running, a job the runner
    cannot find (its handle may be live), a handle-less record, a submission that never recorded its handle, or a
    runner that cannot report a job's status at all.

    A submission that failed before it produced a handle records its ``error`` and is resubmittable: nothing was
    started for it. ``logs/jobs.json`` is host-local and never mirrored, so a restore cannot replace this
    record with another host's handle-less copy.
    """
    record = run.jobs()
    if record is None:
        return
    if record.get("submitting"):
        raise ConfigError(
            f"the job record of {run.layout.run_id} shows a submission that never recorded its handle",
            hint="the job may be live: check the scheduler and cancel it before resubmitting",
            cli_hint=f"check the scheduler and cancel it before resubmitting: rcp-ndcg run status --run {run.dir}",
        )
    backend, _ = _backend(run)
    for job in record["jobs"]:
        if not job["handle"]:
            if record.get("error"):
                continue  # a submission that failed before a handle existed: nothing runs for it
            raise ConfigError(
                f"job {job['name']} of {run.layout.run_id} has no handle, and the record names no submission error",
                hint="the job may be live: check the scheduler and cancel it before resubmitting",
                cli_hint=f"check the scheduler and cancel it before resubmitting: rcp-ndcg run status --run {run.dir}",
            )
        try:
            state = JobStatus(backend.status(job["handle"]))
        except Exception as exc:  # noqa: BLE001 - a plugin runner raises what it raises
            raise ConfigError(
                f"the {record['runner']} runner could not report job {job['name']} ({job['handle']}) of "
                f"{run.layout.run_id}: {exc}",
                hint="its handle may be live: check the scheduler and cancel it before resubmitting",
                cli_hint=f"check the scheduler and cancel it before resubmitting: rcp-ndcg run status --run {run.dir}",
            ) from exc
        if state is JobStatus.UNKNOWN:
            raise ConfigError(
                f"the {record['runner']} runner cannot find job {job['name']} ({job['handle']}) of {run.layout.run_id}",
                hint="its handle may be live: check the scheduler and cancel it before resubmitting",
                cli_hint=f"check the scheduler and cancel it before resubmitting: rcp-ndcg run status --run {run.dir}",
            )
        if state in (JobStatus.PENDING, JobStatus.RUNNING):
            raise ConfigError(
                f"job {job['name']} ({job['handle']}) of {run.layout.run_id} is still {state}",
                hint="wait for it to end, or cancel it first",
                cli_hint=f"wait for it to end, or cancel it first: rcp-ndcg run cancel --run {run.dir}",
            )


def _backend(run: Run) -> tuple[Any, dict[str, Any]]:
    record = run.jobs()
    if record is None:
        raise MissingInputError(f"{run.layout.run_id} was not handed to a runner", hint="it runs in-process")
    return get_runner(record["runner"], **record.get("options", {})), record


#: What a runner's job status says about a run whose manifest has not reached a terminal status.
_ENDED = {JobStatus.FAILED: RunStatus.FAILED, JobStatus.CANCELLED: RunStatus.CANCELLED}


def status(run_dir: str | Path) -> RunState:
    """The run's state, with each job's live status asked from its runner when a runner holds the run.

    A run with a mirror is read from the mirror's manifest when that one is newer (a job elsewhere, e.g. a
    Kubernetes pod, ran the run further than this directory shows). When every job has ended and the manifest is
    still ``submitted`` or ``running``, the job ended without recording it (its engine failed, the scheduler stopped
    it, its submission failed): the state then says ``failed`` or ``cancelled``, with ``done`` true and a ``note``
    that says why. Nothing is written.
    """
    run = Run(run_dir)
    manifest = run.manifest
    notes: list[str] = []
    config = RunConfig.from_data(manifest.config)
    if config.mirror is not None:
        manifest = _newer_from_mirror(run, config.mirror, manifest, notes)
    state = run.state(manifest)
    record = run.jobs()
    if record is None:
        return state.model_copy(update={"note": " ".join(notes) or None})
    if record.get("error"):
        notes.append(f"the submission failed: {record['error']}")
    backend, _ = _backend(run)
    jobs = []
    for job in record["jobs"]:
        if job["handle"]:
            try:
                live = JobStatus(backend.status(job["handle"]))
            except Exception as exc:  # noqa: BLE001 - a plugin runner raises what it raises (a damaged session
                # file, a missing tool): a status command reports what this host holds and says so, never aborts.
                live = JobStatus.UNKNOWN
                notes.append(f"the {record['runner']} runner could not report job {job['name']}: {exc}")
        else:
            live = JobStatus.UNKNOWN
        if live is JobStatus.UNKNOWN and not record.get("error"):
            notes.append(
                f"the {record['runner']} runner reports job {job['name']} ({job['handle'] or 'no handle'}) as "
                "unknown: check the scheduler and the job's log"
            )
        jobs.append(JobState(name=job["name"], handle=job["handle"] or "", status=live))
    update: dict[str, Any] = {"jobs": jobs}
    ended = [JobStatus(job.status) for job in jobs]
    if (
        manifest.status in (RunStatus.SUBMITTED, RunStatus.RUNNING)
        and ended
        and all(status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED) for status in ended)
    ):
        derived = next((_ENDED[s] for s in (JobStatus.FAILED, JobStatus.CANCELLED) if s in ended), RunStatus.FAILED)
        update.update(status=derived, done=True)
        notes.append(
            f"every job of the run has ended ({', '.join(s.value for s in ended)}), and the run's manifest was left "
            f"{manifest.status.value}: the job stopped before it recorded the end; see `rcp-ndcg run logs`, and "
            "resume the run to carry on"
        )
    if state.done:
        live = any(status in (JobStatus.PENDING, JobStatus.RUNNING) for status in ended)
        # A multi-phase job's manifest is `partial` (terminal) between phases. The run is not done while the job
        # is still running, and not done either when the runner cannot say whether the job ended (a missing
        # `sacct`, a TTL-deleted Job): a poller must not stop at a phase boundary.
        unresolved = state.status is RunStatus.PARTIAL and JobStatus.UNKNOWN in ended
        if live or unresolved:
            update["done"] = False
            why = "its job is still running" if live else "the runner cannot say whether its job ended"
            notes.append(f"the run's manifest is {manifest.status.value}, and {why}: the run is not done")
    update["note"] = " ".join(notes) or None
    return state.model_copy(update=update)


def _newer_from_mirror(run: Run, remote: str, manifest: RunManifest, notes: list[str]) -> RunManifest:
    """The mirror's manifest when it was updated after ``manifest``, else ``manifest``.

    A mirror whose manifest names another run is ignored with a note: the mirror is run-scoped
    (:meth:`~rcp_ndcg.runs.mirror.Mirror.restore` refuses it the same way), and ``run status`` must not adopt
    another run's id, steps or metrics from a shared prefix.
    """
    from rcp_ndcg.runs.mirror import Mirror

    try:
        payload = Mirror(run.dir, remote).read(MANIFEST_NAME)
        mirrored = RunManifest.model_validate_json(payload) if payload is not None else None
    except Exception as exc:  # noqa: BLE001 - any mirror client failure (gcsfs HttpError, auth RefreshError, ...)
        # A read-only status question falls back to the local state, whatever the store's client raised.
        notes.append(
            f"the mirror {safe_url(remote)} could not be read ({type(exc).__name__}: {redact_urls(str(exc))}); "
            "this is the local state."
        )
        return manifest
    if mirrored is not None and mirrored.run_id != manifest.run_id:
        notes.append(
            f"the mirror {safe_url(remote)} holds the run {mirrored.run_id!r}, and this run is "
            f"{manifest.run_id!r}: ignoring it (the mirror is run-scoped)"
        )
        return manifest
    if mirrored is None or mirrored.updated_at <= manifest.updated_at:
        return manifest
    notes.append(
        f"read from the mirror {safe_url(remote)}, which is ahead of this directory (restore it with run resume "
        "--mirror)."
    )
    return mirrored


def logs(run_dir: str | Path, *, tail: int | None = None) -> str:
    """The run's log: its jobs' output from the runner, else the pipeline's ``logs/run.log``."""
    run = Run(run_dir)
    if run.jobs() is None:
        return run.log(tail=tail)
    backend, record = _backend(run)
    return "".join(backend.logs(job["handle"], tail=tail) for job in record["jobs"] if job["handle"])


def cancel(run_dir: str | Path) -> RunState:
    """Cancel the run's jobs that are pending or running, and record the run as ``cancelled`` if any was.

    A job that has already ended is left as it is, and so is the run's status (``run status`` reports how it
    ended). The steps the run recorded as running are closed as cancelled.

    Raises:
        MissingInputError: The run was not handed to a runner (an in-process run stops with Ctrl-C), a job was
            never submitted, the runner cannot find it or cannot report its status (its handle may be live), or
            it has no handle. Nothing is recorded then.
        RunnerError: The runner failed to stop a job.
    """
    run = Run(run_dir)
    backend, record = _backend(run)
    live: list[tuple[str, JobStatus]] = []
    for job in record["jobs"]:
        if not job["handle"]:
            if record.get("submitting") or not record.get("error"):
                raise MissingInputError(
                    f"job {job['name']} of {run.layout.run_id} has no handle, and the record does not say the "
                    "submission failed: it may be live",
                    hint="check the scheduler; nothing was cancelled from here",
                )
            raise MissingInputError(
                f"job {job['name']} of {run.layout.run_id} was never submitted, so there is nothing to cancel",
                hint="see why in `run status` (its note); the run is not running",
            )
        try:
            state = JobStatus(backend.status(job["handle"]))
        except Exception as exc:  # noqa: BLE001 - a plugin runner raises what it raises
            raise MissingInputError(
                f"the {record['runner']} runner could not report job {job['name']} ({job['handle']}) of "
                f"{run.layout.run_id}: {exc}",
                hint="check the job on the scheduler; nothing was cancelled and the run's record is left as it was",
            ) from exc
        if state is JobStatus.UNKNOWN:
            raise MissingInputError(
                f"the {record['runner']} runner cannot find job {job['name']} ({job['handle']}) of "
                f"{run.layout.run_id}; nothing was cancelled",
                hint="check the job on the scheduler; the run's record is left as it was",
                details={"runner": record["runner"], "handle": job["handle"]},
            )
        live.append((job["handle"], state))
    stopped = 0
    for handle, state in live:
        if state in (JobStatus.PENDING, JobStatus.RUNNING):
            backend.cancel(handle)
            stopped += 1
    if stopped and run.manifest.status in (RunStatus.SUBMITTED, RunStatus.RUNNING):
        mark(run, RunStatus.CANCELLED)
    return status(run_dir)


def run(
    config: RunConfig | str | Path,
    *,
    runner: str | None = None,
    resume: bool = True,
    estimate: bool = False,
    overrides: Sequence[str] = (),
    runs_dir: str | None = None,
) -> Run | CostEstimate:
    """Run a config: in this process (``runner="local"``), or as one job of another runner.

    Args:
        config: A :class:`RunConfig`, the path of a run config YAML, or an existing run directory (resumed).
        runner: ``"local"`` runs in this process; another name hands the run to that job runner. ``None`` takes
            the config's ``runner.name`` for a new run, and resumes an existing run directory in this process
            (that is what a submitted job does). An existing run directory with a runner is submitted again, with
            the options its job record holds (after a restore from its mirror, when it has one).
        resume: Reuse the steps a resumed run already completed with the same identity and inputs.
        estimate: Estimate the judging steps instead of running anything.
        overrides: ``key=value`` config overrides.
        runs_dir: Where a new run's directory is created; default ``$RCP_NDCG_RUNS_DIR``, else ``runs``.

    Returns:
        The :class:`~rcp_ndcg.runs.run.Run`; with ``estimate=True``, the :class:`~rcp_ndcg.judging.CostEstimate` of
        its judging steps (calls, tokens, wall time) instead, and nothing runs.
    """
    existing = not isinstance(config, RunConfig) and Path(RunLayout.at(config).manifest).exists()
    resubmit = existing and runner not in (None, "local") and not estimate
    if existing:
        if resubmit:
            restore_for_resubmission(config)  # type: ignore[arg-type]
        pipeline = reopen(config, overrides=overrides)  # type: ignore[arg-type]
    else:
        pipeline = prepare(config, overrides=overrides, runs_dir=runs_dir)
    if estimate:
        return pipeline.estimate()
    if resubmit:
        assert runner is not None
        return submit_run(pipeline, runner, recorded_options(pipeline.layout.root, runner))
    if runner is None:
        runner = "local" if existing else pipeline.config.runner.name
    if runner == "local":
        if not existing:
            refuse_serving_here(pipeline.config)
        return execute_run(pipeline, resume=resume)
    return submit_run(pipeline, runner)


def restore_for_resubmission(run_dir: str | Path, mirror: str | None = None) -> None:
    """Bring a run directory up to date from its mirror (``mirror``, else the run's own) before it is submitted
    again: a job elsewhere may have run it further than this directory shows."""
    from rcp_ndcg.runs.mirror import restore

    remote = mirror or Run(run_dir).config.mirror
    if remote is not None:
        restore(run_dir, remote)


def recorded_options(run_dir: str | Path, runner: str) -> dict[str, Any] | None:
    """The runner options of the run's last job record, when it was handed to ``runner`` (``None`` otherwise)."""
    record = Run(run_dir).jobs()
    return dict(record.get("options", {})) if record is not None and record["runner"] == runner else None


def refuse_serving_here(config: RunConfig) -> None:
    """Refuse to start a run with a ``serve:`` section in this process, which starts no engine.

    A job of a runner that starts engines resumes its run here with the engines already up beside it, so only a
    new run is refused.

    Raises:
        ConfigError: ``config`` has a ``serve:`` section.
    """
    if config.serve is not None:
        raise ConfigError(
            "this run has a serve: section, and a run in this process (or on the local runner) starts no engine",
            hint="start the engine(s) yourself (docs/concepts/runs.md), drop serve:, and pass the URLs with "
            "RCP_NDCG_ENGINES (or run resume --engine <role>=<url>[,<url>])",
        )


__all__ = [
    "JOB_OPTIONS",
    "cancel",
    "job_for",
    "logs",
    "recorded_options",
    "refuse_serving_here",
    "render_run",
    "restore_for_resubmission",
    "run",
    "run_argv",
    "stage_run",
    "status",
    "submit_run",
]
