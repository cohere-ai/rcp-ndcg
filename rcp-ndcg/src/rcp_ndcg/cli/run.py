"""``rcp-ndcg run``: run a config end to end, resume it, follow it, and list and show runs.

``run start`` runs a whole evaluation from one config (retrieve, rerank, tournament, rubric, calibrate, evaluate);
its artifacts land in ``<runs dir>/<run_id>/`` and the manifest there drives resume. By default it runs in this
process; ``--runner slurm|kubernetes|local|<plugin>`` hands the run to that job runner as one job that runs
``rcp-ndcg run resume`` on the prepared run directory. ``run status``, ``run logs`` and ``run cancel`` follow it
there. ``run list`` and ``run show`` read the manifests and never write.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal

import click
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rcp_ndcg.cli.command import command
from rcp_ndcg.errors import ConfigError, RcpNdcgWarning, UsageError
from rcp_ndcg.judging.cost import CostEstimate
from rcp_ndcg.judging.judges import judge_names
from rcp_ndcg.runs.manifest import RunManifest, RunStatus
from rcp_ndcg.runs.run import RunState
from rcp_ndcg.support.paths import RUNS_DIR_ENV, runs_dir
from rcp_ndcg.support.serve import EngineRole, EngineURLs
from rcp_ndcg.support.urls import redact_urls


class RunListRequest(BaseModel):
    runs_dir: str | None = Field(default=None, description="Where the runs live (default: ./runs).")
    limit: int = Field(default=50, ge=1, description="At most this many runs, newest first.")


class RunListRow(BaseModel):
    """One run of ``run list``: the manifest's summary. A manifest that does not parse is listed with
    ``status: "unreadable"`` (the one value outside :class:`~rcp_ndcg.runs.manifest.RunStatus`) and its error."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: RunStatus | Literal["unreadable"]
    created_at: str | None = Field(default=None, description="When the run was created (a readable manifest).")
    dataset: str | None = None
    judges: list[str] | None = None
    metrics: dict[str, float] | None = None
    requests: int | None = None
    steps: dict[str, str] | None = None
    error: str | None = Field(default=None, description="Why the manifest is unreadable (status 'unreadable').")


class RunList(BaseModel):
    """The runs under a runs directory, newest first."""

    runs_dir: str
    runs: list[RunListRow]


class RunShowRequest(BaseModel):
    run: str = Field(description="The run directory (<runs dir>/<run_id>).")


class RunSummary(BaseModel):
    """One run: its manifest and which artifacts exist."""

    run_dir: str
    manifest: RunManifest
    artifacts: dict[str, bool]


def _list_text(result: RunList) -> str:
    if not result.runs:
        return f"no runs under {result.runs_dir}"
    lines = [f"{'run_id':<40} {'status':<10} {'dataset':<20} {'requests':>9}  metrics"]
    for row in result.runs:
        metrics = ", ".join(f"{key}={value:.4f}" for key, value in (row.metrics or {}).items())
        lines.append(
            f"{row.run_id:<40} {str(row.status):<10} {str(row.dataset or '-'):<20} "
            f"{f'{row.requests:,}' if row.requests is not None else '-':>9}  {metrics}"
        )
    return "\n".join(lines)


def _show_text(result: RunSummary) -> str:
    manifest = result.manifest
    lines = [
        f"Run {manifest.run_id} ({manifest.status})",
        f"  directory   {result.run_dir}",
        f"  created     {manifest.created_at}",
        f"  code        {manifest.code.package_version} @ {manifest.code.git_commit or 'unknown'}"
        + (" (dirty)" if manifest.code.git_dirty else ""),
        f"  requests    {manifest.usage.requests:,} judge requests",
        "  steps",
    ]
    for record in manifest.steps:
        duration = record.duration_s
        lines.append(
            f"    {record.name:<12} {str(record.status):<10} "
            f"{f'{duration:.1f}s' if duration else '':>8}" + (f"  {record.error}" if record.error else "")
        )
    present = [name for name, exists in result.artifacts.items() if exists]
    lines.append(f"  artifacts   {', '.join(present) or 'none'}")
    for key, value in manifest.metrics.items():
        lines.append(f"    {key:<24} {value:.4f}")
    return "\n".join(lines)


@command(
    "run list",
    request=RunListRequest,
    result=RunList,
    envvars={"runs_dir": RUNS_DIR_ENV},
    text=_list_text,
)
def run_list(request: RunListRequest) -> RunList:
    """List the runs under a runs directory, newest first, with status, judge requests and metrics."""
    from rcp_ndcg.runs import inspect

    directory = request.runs_dir or str(runs_dir())
    rows = inspect.list_runs(directory, limit=request.limit)
    for row in rows:
        if row.get("status") == "unreadable":
            warnings.warn(
                RcpNdcgWarning("UNREADABLE_RUN", f"{row['run_id']}: manifest unreadable ({row.get('error')})"),
                stacklevel=1,
            )
    return RunList(runs_dir=directory, runs=[RunListRow.model_validate(row) for row in rows])


@command("run show", request=RunShowRequest, result=RunSummary, text=_show_text)
def run_show(request: RunShowRequest) -> RunSummary:
    """Show one run: its manifest (config, steps, usage, provenance) and which artifacts exist."""
    from rcp_ndcg.runs import inspect

    return RunSummary.model_validate(inspect.get_run(request.run))


# ----------------------------------------------------------------------------------------------------------------
# start and resume
# ----------------------------------------------------------------------------------------------------------------

Step = Literal["retrieve", "rerank", "tournament", "rubric", "calibrate", "evaluate"]


_MIRROR_HELP = (
    "Mirror the run directory to any fsspec URI (e.g. s3://bucket/runs) while it runs, and restore it from there "
    "on resume (sets the config's mirror)."
)


class RunJudgeFields(BaseModel):
    judge: str | None = Field(
        default=None,
        description="Replace the config's judge: fake, a judge recipe id (recipe:<id> or a bare id), "
        f"a shipped vendor profile ({', '.join(judge_names())}), or a judge config YAML.",
    )
    judge_url: str | None = Field(default=None, description="Replace the judge by an ad-hoc endpoint (.../v1).")
    judge_model: str | None = Field(default=None, description="The served model name, with --judge-url.")

    def judge_overrides(self) -> list[str]:
        if self.judge_model is not None and self.judge_url is None:
            raise UsageError(
                "--judge-model names the model served at --judge-url, and there is none",
                hint="pass --judge-url with it, or set the model of a judge config with --set judge.model=...",
            )
        if self.judge_url is not None:
            if self.judge is not None:
                raise UsageError(
                    "pass --judge or --judge-url, not both",
                    hint="drop --judge, or point a judge config at the endpoint with --set judge.base_url=...",
                )
            if self.judge_model is None:
                raise UsageError(
                    "--judge-url needs --judge-model (the served model name)",
                    hint="pass the endpoint and its model: --judge-url .../v1 --judge-model ID",
                )
            return ["judge=" + json.dumps({"base_url": self.judge_url, "model": self.judge_model})]
        if self.judge is not None:
            return [f"judge={json.dumps(self.judge)}"]
        return []


class RunStartRequest(RunJudgeFields):
    config: str = Field(description="The run config YAML.")
    set: list[str] = Field(
        default_factory=list,
        description="Override one config field (dotted KEY=VALUE, repeatable; VALUE is a YAML literal: 5, true, "
        "[a, b], {k: v}); applied after --judge/--judge-url, so --set judge.context_tokens=... configures an "
        "ad-hoc judge.",
    )
    runs_dir: str | None = Field(default=None, description="Where the run directory is created (default: ./runs).")
    label: str | None = Field(default=None, description="A readable fragment of the run id.")
    mirror: str | None = Field(default=None, description=_MIRROR_HELP)
    only: list[Step] = Field(
        default_factory=list, description="Give the new run only these steps (repeatable; its config's steps)."
    )
    runner: str | None = Field(
        default=None, description="Hand the run to a job runner: local, slurm, kubernetes or an installed plugin."
    )
    detach: bool = Field(
        default=False,
        description="Return at once and follow the run with `run status` (a slurm or kubernetes run always "
        "does); without --runner, the run becomes a background job of the local runner.",
    )
    estimate: bool = Field(
        default=False,
        description="Print the judging steps' calls, tokens and time; run nothing. Refuses what the run would refuse.",
    )
    dry_run: bool = Field(
        default=False,
        description="Print the step plan and, with a runner, what it would submit (sbatch script or manifests); "
        "run nothing. Refuses what the run would refuse.",
    )


class PlanStep(BaseModel):
    step: str
    status: Literal["would run", "would skip"]


class RunStartResult(BaseModel):
    """What ``run start`` or ``run resume`` did: ran the run, handed it to a runner, planned it or estimated it."""

    mode: Literal["ran", "submitted", "plan", "estimate"]
    run_id: str | None = Field(description="The run's id; null for a plan or an estimate of a new run.")
    run_dir: str | None = Field(description="The run directory; null for a plan or an estimate of a new run.")
    state: RunState | None = None
    plan: list[PlanStep] | None = None
    rendered: dict[str, str] | None = Field(
        default=None, description="For a plan handed to a runner: what it would submit, per job (script or YAML)."
    )
    estimate: CostEstimate | None = None


def _finish(
    pipeline: Any,
    request: Any,
    *,
    resume: bool,
    runner: str | None = None,
    options: dict[str, Any] | None = None,
) -> RunStartResult:
    """Estimate, plan, hand the run to ``runner`` (with ``options``) or run it in this process.

    ``run resume`` passes a runner only with ``--runner``: without it, it is what a submitted job executes, so it
    runs here.
    """
    from rcp_ndcg.runs.run import execute_run

    run_id = pipeline.layout.run_id
    existing = pipeline.layout.root if Path(pipeline.layout.manifest).exists() else None
    planned_id = run_id if existing else None  # a plan or an estimate of a new run names no id: none exists
    if request.estimate:
        # The real command's refusals first (identity, settings); nothing is written.
        estimate = pipeline.estimate(resume=resume)
        return RunStartResult(mode="estimate", run_id=planned_id, run_dir=existing, estimate=estimate)
    detach = bool(getattr(request, "detach", False))
    if runner is None and detach:
        runner = "local"
    if request.dry_run:
        pipeline.preflight(resume=resume)  # the real command's refusals; nothing is written
        rendered = None
        if runner is not None:
            from rcp_ndcg.runs.execution import render_run

            rendered = render_run(pipeline, runner, {"detach": True} if runner == "local" and detach else options)
        plan = pipeline.plan(resume=resume)
        return RunStartResult(mode="plan", run_id=planned_id, run_dir=existing, plan=plan, rendered=rendered)
    if runner is not None:
        from rcp_ndcg.runs.execution import status, submit_run

        run = submit_run(pipeline, runner, {"detach": True} if runner == "local" and detach else options)
        return RunStartResult(mode="submitted", run_id=run_id, run_dir=run.dir, state=status(run.dir))
    run = execute_run(pipeline, resume=resume)
    return RunStartResult(mode="ran", run_id=run_id, run_dir=run.dir, state=run.status())


def _start_text(result: RunStartResult) -> str:
    if result.estimate is not None:
        e = result.estimate
        lines = [f"Estimate (input tokens {e.input_token_count}, output tokens assumed; see the assumptions)", ""]
        for name, stage in e.stages.items():
            lines.append(
                f"  {name:<12} {stage.calls:>9,} calls  ~{stage.input_tokens:>12,} input  "
                f"~{stage.output_tokens:>11,} output"
            )
        lines += ["", f"  {'total':<12} {e.calls:>9,} calls  ~{e.input_tokens:>12,} input  ~{e.output_tokens:>11,} "
                  "output", f"  about {e.wall_s:,.0f} s", "", "Assumptions:"]  # fmt: skip
        return "\n".join(lines + [f"  - {item}" for item in e.assumptions])
    if result.plan is not None:
        lines = [f"Plan for {result.run_id or 'a new run'}", ""]
        lines += [f"  {row.step:<12} {row.status}" for row in result.plan]
        for name, text in (result.rendered or {}).items():
            lines += ["", f"# {name}: what the runner would submit", text.rstrip()]
        return "\n".join(lines)
    state = result.state
    assert state is not None
    lines = [f"Run {result.run_id} {state.status} ({result.mode})", f"  directory  {result.run_dir}"]
    lines.append(f"  requests   {state.requests:,} judge requests")
    for step in state.steps:
        duration = f"{step.duration_s:.1f}s" if step.duration_s else ""
        progress = step.progress
        windows = f"  {progress.done}/{progress.planned or '?'} windows" if progress is not None else ""
        lines.append(
            f"    {step.name:<12} {step.status:<10} {duration:>8}{windows}" + (f"  {step.error}" if step.error else "")
        )
    for job in state.jobs:
        lines.append(f"  job {job.name} ({state.runner}): {job.handle} {job.status}")
    for key, value in state.metrics.items():
        lines.append(f"    {key:<24} {value:.4f}")
    return "\n".join(lines)


_START = {"result": RunStartResult, "output_schema": "rcp-ndcg.run-start.v1", "text": _start_text}


@command(
    "run start",
    request=RunStartRequest,
    positional=("config",),
    envvars={"runs_dir": RUNS_DIR_ENV},
    read_only=False,
    **_START,
)
def run_start(request: RunStartRequest) -> RunStartResult:
    """Run a config end to end (retrieve, rerank, tournament, rubric, calibrate, evaluate), here or on a runner."""
    from rcp_ndcg.runs.run import prepare

    mirror = [f"mirror={request.mirror}"] if request.mirror else []
    pipeline = prepare(
        request.config,
        overrides=[*request.judge_overrides(), *request.set, *mirror],
        runs_dir=request.runs_dir,
        label=request.label,
        steps=request.only or None,
    )
    configured = pipeline.config.runner.name
    runner = request.runner or (configured if configured != "local" else None)
    if runner is None and not request.detach and not request.estimate:
        from rcp_ndcg.runs.execution import refuse_serving_here

        refuse_serving_here(pipeline.config)
    return _finish(pipeline, request, resume=True, runner=runner)


_ENGINE_ROLES = ("judge", "encoder", "reranker")


def _engine_overlays(specs: list[str]) -> dict[EngineRole, EngineURLs]:
    """``--engine role=url[,url]`` entries as the engines overlay the run applies at runtime.

    Raises:
        UsageError: an entry without ``=`` or with an unknown role, a role given twice, or a role with no URL.
    """
    engines: dict[EngineRole, EngineURLs] = {}
    for spec in specs:
        role, sep, urls = spec.partition("=")
        role = role.strip()
        if not sep or role not in _ENGINE_ROLES:
            raise UsageError(
                f"--engine {spec!r} is not <role>=<url>[,<url>] with role one of {', '.join(_ENGINE_ROLES)}",
                hint="e.g. --engine judge=http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1",
                cli_hint="e.g. --engine judge=http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1",
            )
        if role in engines:
            raise UsageError(
                f"--engine names the {role} role twice",
                hint=f"give one --engine per role, with all of its URLs: --engine {role}=url1,url2",
            )
        replicas = tuple(url.strip() for url in urls.split(",") if url.strip())
        if not replicas:
            raise UsageError(
                f"--engine {role}= has no URL",
                hint=f"give at least one replica base URL (.../v1): --engine {role}=http://127.0.0.1:8000/v1",
            )
        try:
            engines[role] = EngineURLs(urls=replicas)  # type: ignore[index]
        except (ValidationError, ConfigError) as exc:
            # The spec's URLs are named redacted (safe_url): a URL may embed credentials.
            reason = exc.errors(include_url=False)[0]["msg"] if isinstance(exc, ValidationError) else exc.message
            raise UsageError(
                f"--engine {redact_urls(spec)!r}: {reason}",
                hint=f"give each replica once: --engine {role}=http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1",
            ) from None
    return engines


class RunResumeRequest(RunJudgeFields):
    run: str = Field(description="The run directory.")
    engine: list[str] = Field(
        default_factory=list,
        description="Point one role's model at engine URLs instead of its config's base_url: role=url[,url] "
        "(repeatable; role is judge, encoder or reranker). A runtime overlay: applied to the run's configs while "
        "this invocation runs, never written into its config.",
    )
    mirror: str | None = Field(
        default=None,
        description="Restore what the run directory lacks, or holds in an older state, from this mirror (any fsspec "
        "URI) first, also with --dry-run or --estimate, and keep mirroring to it.",
    )
    set: list[str] = Field(
        default_factory=list,
        description="Override one config field (dotted KEY=VALUE, repeatable; VALUE is a YAML literal: 5, true, "
        "[a, b], {k: v}); applied after --judge/--judge-url. The run keeps the change only if the resume succeeds.",
    )
    only: list[Step] = Field(
        default_factory=list,
        description="Run only these of the run's steps now (repeatable); the run's recorded steps stay as they are.",
    )
    resume: bool = Field(default=True, description="Skip the steps already done with the same identity and inputs.")
    estimate: bool = Field(
        default=False, description="Estimate the judging steps; run nothing. Refuses what the resume would refuse."
    )
    dry_run: bool = Field(
        default=False,
        description="Print which steps would run and which are current; run nothing. Refuses what the resume would "
        "refuse (exit 11 for a changed judging identity).",
    )
    runner: str | None = Field(
        default=None,
        description="Submit the run again as one job of this runner (slurm, kubernetes, local or a plugin), with "
        "the runner options of its last job and its serve: section, e.g. after its job failed; the job resumes it. "
        "Without it, the run resumes in this process.",
    )


@command("run resume", request=RunResumeRequest, read_only=False, **_START)
def run_resume(request: RunResumeRequest) -> RunStartResult:
    """Continue a run directory in this process: the steps whose identity and inputs are unchanged are skipped.

    A changed step runs again, except a judging step whose stored judgements have another identity: it is refused
    (exit 11) and the run is left as it was. A resume whose --set change fails leaves the run's
    config as it was too; --only never changes the run's steps. --engine overlays the role configs at runtime
    (the URLs of engines the job or you started), never the run's recorded config. With --runner, the run is
    submitted again as one job of that runner instead (its --set changes are recorded before the job starts).
    """
    from rcp_ndcg.runs.execution import recorded_options, restore_for_resubmission
    from rcp_ndcg.runs.mirror import restore
    from rcp_ndcg.runs.run import reopen

    if request.runner is not None and (request.only or not request.resume or request.engine):
        raise UsageError(
            "--runner submits the whole run again, as recorded; --only, --no-resume and --engine apply to a "
            "resume in this process",
            hint="drop them, or resume in this process without --runner",
        )
    mirror = []
    if request.runner is not None:
        restore_for_resubmission(request.run, request.mirror)  # a job elsewhere may have run it further
    elif request.mirror:
        restore(request.run, request.mirror)
    if request.mirror:
        mirror = [f"mirror={request.mirror}"]
    engines = _engine_overlays(request.engine)
    pipeline = reopen(
        request.run,
        overrides=[*request.judge_overrides(), *request.set, *mirror],
        steps=request.only or None,
        engines=engines or None,
    )
    if request.runner is None:
        return _finish(pipeline, request, resume=request.resume)
    options = recorded_options(request.run, request.runner)
    return _finish(pipeline, request, resume=True, runner=request.runner, options=options)


# ----------------------------------------------------------------------------------------------------------------
# status, logs, cancel
# ----------------------------------------------------------------------------------------------------------------


class RunRef(BaseModel):
    run: str = Field(description="The run directory.")


def _state_text(state: RunState) -> str:
    return _start_text(RunStartResult(mode="ran", run_id=state.run_id, run_dir=state.run_dir, state=state))


@command("run status", request=RunRef, result=RunState, output_schema="rcp-ndcg.run-status.v1", text=_state_text)
def run_status(request: RunRef) -> RunState:
    """Where a run stands: every step's status, its judge requests, and its jobs' live status on a runner."""
    from rcp_ndcg.runs.execution import status

    return status(request.run)


class RunLogsRequest(RunRef):
    tail: int | None = Field(default=None, ge=0, description="Only the last N lines.")


class RunLog(BaseModel):
    """A run's log: its jobs' output from the runner, or the pipeline's logs/run.log."""

    run_dir: str
    text: str


@command("run logs", request=RunLogsRequest, result=RunLog, text=lambda result: result.text.rstrip())
def run_logs(request: RunLogsRequest) -> RunLog:
    """Print a run's log: its jobs' output from the runner, else the pipeline's logs/run.log."""
    from rcp_ndcg.runs.execution import logs

    return RunLog(run_dir=request.run, text=logs(request.run, tail=request.tail))


@command(
    "run cancel",
    request=RunRef,
    result=RunState,
    output_schema="rcp-ndcg.run-status.v1",
    text=_state_text,
    read_only=False,
)
def run_cancel(request: RunRef) -> RunState:
    """Cancel a run's jobs on its runner and record the run as cancelled."""
    from rcp_ndcg.runs.execution import cancel

    return cancel(request.run)


@click.group(name="run", help="Run a config end to end, resume it, follow it; list and show runs.")
def run_group() -> None:
    """``rcp-ndcg run``."""


for _command in (run_start, run_resume, run_status, run_logs, run_cancel, run_list, run_show):
    run_group.add_command(_command)


__all__ = [
    "PlanStep",
    "RunList",
    "RunListRequest",
    "RunListRow",
    "RunLog",
    "RunLogsRequest",
    "RunRef",
    "RunResumeRequest",
    "RunShowRequest",
    "RunStartRequest",
    "RunStartResult",
    "RunSummary",
    "run_group",
]
