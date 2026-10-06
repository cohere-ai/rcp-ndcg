"""SlurmRunner -- one ``sbatch`` script per :class:`JobSpec`.

The rendered script carries the job's resources as ``#SBATCH`` lines and
runs the worker scripts (:func:`~rcp_ndcg.runners.script.worker_script`)
either directly on the node or inside a container:

* ``container_runtime: none`` (default) -- the node's environment; ``setup``
  lines (``module load``, ``source .venv/bin/activate``) run first. An engine
  (a phase's :class:`~rcp_ndcg.support.serve.EngineConfig`) runs its command on the node too, so it takes no
  image.
* ``container_runtime: apptainer`` -- ``apptainer exec --nv <image> bash -c ...``
  (``image`` is a ``.sif`` path or an image reference, run as ``docker://``).
* ``container_runtime: pyxis`` -- ``srun --container-image=<image> bash -c ...``
  (NVIDIA enroot + pyxis).

In a container the command runs in a stock image with uv and Python 3.12 (the
default :data:`~rcp_ndcg.runners.script.COORDINATOR_IMAGE`), which installs this
release when the job starts; on the node it runs as is.

A job with phases runs them in order in one allocation: the ``sbatch`` asks for
the maximum nodes and GPUs over the phases, and each phase that starts engines
runs one supervision block (:func:`~rcp_ndcg.runners.script.supervise`), with
its own engine heredocs and its coordinator command. Each role's engines run as
one background step (``srun --overlap``, one replica per node) under the block:
the engines start once, the script waits until every role has a replica answering
its readiness path, exports their URLs in ``RCP_NDCG_ENGINES``, runs the phase's
coordinator with them, and ends the phase when either ends. The engine steps run
with ``--kill-on-bad-exit=1`` and ``--wait``, so one replica that exits ends the
whole step, and with it the job (non-zero); the next phase starts only after the
previous coordinator exited 0 and its engine step was stopped. When the job's
largest phase needs more than one node, every role's replicas are pinned to a
deterministic slice of the allocation's nodes (``srun --nodelist``), read from
``SLURM_JOB_NODELIST`` when the job starts; on a one-node allocation the engine
answers on ``localhost``. Runs, stores and caches live on the cluster's shared
filesystem, which the submitting host sees too.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.runners._cli import run_cli
from rcp_ndcg.runners.base import JobHandle, JobOptions, JobPhase, JobSpec, JobStatus, RunnerError, tail_lines
from rcp_ndcg.runners.script import (
    COORDINATOR_IMAGE,
    EngineStep,
    engine_script,
    engines_env_command,
    engines_env_spec,
    engines_env_value,
    heredoc,
    supervise,
    worker_script,
)
from rcp_ndcg.support.serve import ENGINES_ENV, ServeConfig

#: Seconds the engine step keeps its other replicas after one exits with status 0 (``srun --wait``).
ENGINE_STEP_WAIT_S = 10

#: SLURM job states -> :class:`JobStatus`.
_STATES = {
    "PENDING": JobStatus.PENDING,
    "CONFIGURING": JobStatus.PENDING,
    "REQUEUED": JobStatus.PENDING,
    "RESIZING": JobStatus.PENDING,
    "SUSPENDED": JobStatus.PENDING,
    "RUNNING": JobStatus.RUNNING,
    "COMPLETING": JobStatus.RUNNING,
    "STAGE_OUT": JobStatus.RUNNING,
    "COMPLETED": JobStatus.SUCCEEDED,
    "CANCELLED": JobStatus.CANCELLED,
    "FAILED": JobStatus.FAILED,
    "TIMEOUT": JobStatus.FAILED,
    "NODE_FAIL": JobStatus.FAILED,
    "OUT_OF_MEMORY": JobStatus.FAILED,
    "PREEMPTED": JobStatus.FAILED,
    "BOOT_FAIL": JobStatus.FAILED,
    "DEADLINE": JobStatus.FAILED,
}


def slurm_time(seconds: int) -> str:
    """Seconds -> SLURM ``--time`` (``D-HH:MM:SS``)."""
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"


def _aggregate(states: Sequence[JobStatus]) -> JobStatus:
    """One status for the records the scheduler reports for a job."""
    if not states:
        return JobStatus.UNKNOWN
    for status in (JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.RUNNING, JobStatus.PENDING):
        if status in states:
            return status
    return JobStatus.SUCCEEDED if all(s is JobStatus.SUCCEEDED for s in states) else JobStatus.UNKNOWN


def _pyxis_image(image: str) -> str:
    """An image reference as pyxis takes it: a registry host is separated from the path by ``#``."""
    head, slash, rest = image.partition("/")
    if slash and "#" not in image and ("." in head or ":" in head or head == "localhost"):
        return f"{head}#{rest}"
    return image


def _apptainer_image(image: str) -> str:
    """An image reference as apptainer takes it: a ``.sif`` path or a URI as is, else ``docker://<image>``."""
    return image if "://" in image or image.endswith(".sif") else f"docker://{image}"


class SlurmOptions(JobOptions):
    """The ``slurm`` runner's options.

    Attributes:
        image: The coordinator's container image, for a container runtime (default
            :data:`~rcp_ndcg.runners.script.COORDINATOR_IMAGE`); a job's own image wins.
        partition: ``--partition``.
        account: ``--account``.
        qos: ``--qos``.
        log_dir: Where ``--output`` goes (``<log_dir>/<job>-<id>.out``).
        workdir: The directory the coordinator runs in (default: sbatch's working directory).
        setup: Shell lines run first (``module load``, venv activation).
        container_runtime: ``none``, ``apptainer`` or ``pyxis``; applies to the coordinator and the engines.
        container_mounts: ``host:container`` bind mounts (the run directory, an HF cache).
        sbatch_args: Further ``#SBATCH`` arguments, verbatim (e.g. ``--constraint=a100``).
    """

    PATHS = ("log_dir", "workdir", "wheelhouse", "constraints")

    image: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    log_dir: str = "logs/slurm"
    workdir: str | None = None
    setup: list[str] = Field(default_factory=list)
    container_runtime: Literal["none", "apptainer", "pyxis"] = "none"
    container_mounts: list[str] = Field(default_factory=list)
    sbatch_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _an_install_source_needs_a_container(self) -> Self:
        """The coordinator installs itself only in a container; on the node the environment has the release."""
        if (self.wheelhouse or self.constraints) and self.container_runtime == "none":
            raise ValueError(
                f"container_runtime is {self.container_runtime!r}, and the coordinator runs on the node, which "
                "already provides the release: a wheelhouse or constraints file applies when the job runs it "
                "in a container (container_runtime: apptainer | pyxis)"
            )
        return self


class SlurmRunner:
    """Submit jobs with ``sbatch``; query with ``squeue``/``sacct``; cancel with ``scancel``.

    Options (``get_runner("slurm", **options)``): :class:`SlurmOptions`; its ``resources`` and ``env`` are every
    job's defaults.
    """

    name = "slurm"
    #: This runner renders a job's phases (a job whose phases start engines is handed to it).
    renders_phases = True

    def __init__(self, **options: Any) -> None:
        self.options = SlurmOptions.parse(self.name, options)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _in_container(self, image: str, script_var: str, *, srun: Sequence[str]) -> str:
        """The line that runs ``$<script_var>`` in ``image`` (or on the node), under ``srun`` when it is given."""
        mounts = self.options.container_mounts
        step = " ".join(srun)
        if self.options.container_runtime == "apptainer":
            binds = "".join(f" --bind {shlex.quote(m)}" for m in mounts)
            launcher = f"apptainer exec --nv{binds} {shlex.quote(_apptainer_image(image))}"
            return f'{step + " " if step else ""}{launcher} bash -c "${script_var}"'
        if self.options.container_runtime == "pyxis":
            flags = f" --container-mounts={shlex.quote(','.join(mounts))}" if mounts else ""
            container = f"--container-image={shlex.quote(_pyxis_image(image))}{flags}"
            return f'{step or "srun"} {container} bash -c "${script_var}"'
        return f'{step + " " if step else ""}bash -c "${script_var}"'

    def _engine_start(self, role: str, serve: ServeConfig, nodelist: str) -> str:
        """The line that starts one role's engine replicas, one per node, in the background (without ``&``).

        Args:
            role: The engine role (``judge``, ``encoder`` or ``reranker``); it names the heredoc variable and
                the error messages.
            nodelist: The ``--nodelist`` value that pins the replicas to their nodes; empty for a one-node
                allocation, where the step lands on the only node.

        Raises:
            ConfigError: ``serve.image`` is set with ``container_runtime: none`` (the command runs on the node, so
                the image cannot be honoured), or missing with a container runtime.
        """
        var = f"ENGINE_{role.upper()}"
        if self.options.container_runtime == "none" and serve.image is not None:
            raise ConfigError(
                f"the {role} engine's image is {serve.image!r}, and the slurm runner's container_runtime is none: "
                "the engine's command would run on the node, and the image would be ignored",
                hint="set runner.options.container_runtime: apptainer | pyxis to run the engine in its image, or "
                f"drop the {role} engine's image to run the command on the node",
            )
        if self.options.container_runtime != "none" and serve.image is None:
            raise ConfigError(
                f"the slurm runner's container_runtime is {self.options.container_runtime}, and the {role} engine "
                "names no image",
                hint="set the engine's image (pin the tag)",
            )
        # One replica that exits, even with status 0, ends the step: at once for a failure, else after 10 s.
        step = ["srun", "--overlap", f"--nodes={serve.replicas}", "--ntasks-per-node=1", "--kill-on-bad-exit=1"]
        step.append(f"--wait={ENGINE_STEP_WAIT_S}")
        if serve.resources.gpus:
            step.append(f"--gres=gpu:{serve.resources.gpus}")
        if nodelist:
            step.append(f"--nodelist={nodelist}")
        return self._in_container(serve.image or "", var, srun=step)  # on the node itself for runtime none

    def _phase_worker(
        self,
        job: JobSpec,
        argv: Sequence[str],
        *,
        install: bool,
        engines_env: str | None = None,
        cuda: str | None = None,
    ) -> str:
        """The worker script of one phase: the job's environment under the phase's own, and the phase's command.

        ``engines_env`` is the ``RCP_NDCG_ENGINES`` the phase's coordinator sees; ``"{}"`` for a phase without
        engines, so no engine of an earlier phase reaches it. ``cuda`` is the coordinator's
        ``CUDA_VISIBLE_DEVICES`` (its own GPU request, reserved ahead of the engines' slices on the phase's first
        node); ``None`` leaves the scheduler's value.
        """
        env = {ENGINES_ENV: engines_env} if engines_env is not None else None
        if cuda is not None:
            env = {**(env or {}), "CUDA_VISIBLE_DEVICES": cuda}
        return worker_script(
            job.with_argv(argv),
            install=install,
            workdir=self.options.workdir,
            env=env,
            wheelhouse=self.options.wheelhouse,
            constraints=self.options.constraints,
        )

    def _phase_lines(
        self, job: JobSpec, index: int, phase: JobPhase, *, image: str, container: bool, one_node: bool
    ) -> list[str]:
        """One phase's block: its worker heredoc, its engines' heredocs and hosts, and its supervision.

        Args:
            index: The phase's 1-based position in the job; it names the worker heredoc (``WORKER_<index>``).
            one_node: The whole allocation is one node, so every engine answers on ``localhost`` and the URLs are
                known at render time. Otherwise each role's replicas are pinned to a deterministic slice of the
                allocation's nodes (``srun --nodelist``), and the ``RCP_NDCG_ENGINES`` JSON is built when the job
                starts from the nodes the script read into ``RCP_NDCG_HOSTS``.
        """
        worker_var = f"WORKER_{index}"
        if not phase.engines:
            worker = self._phase_worker(job, phase.argv, install=container, engines_env="{}")
            command = self._in_container(image, worker_var, srun=[]) if container else f'bash -c "${worker_var}"'
            return [*heredoc(worker_var, worker), command]
        # The placement order: engines by descending GPU count, then role name — the largest engine shares the
        # coordinator's node, which keeps the per-node GPU sum tight; roles sorted, replicas in order.
        roles = sorted(phase.engines, key=lambda role: (-phase.engines[role].resources.gpus, role))
        # On a one-node allocation every phase has at most one engine (the largest phase's replica total sizes the
        # allocation), so roles never share a port there; on larger ones each role gets its own node slice.
        lines = [*heredoc(worker_var, self._phase_worker(job, phase.argv, install=container))]
        steps: list[EngineStep] = []
        offset = 0
        # The replicas are placed one per node (roles by descending GPU count), so no two engine processes share
        # a node and SLURM's per-step CUDA_VISIBLE_DEVICES -- unique per step (gres.html, "GPU Management") --
        # already partitions the devices; a node's --gres is the sum of what runs on it (the coordinator's own
        # request on the first node, see render_job).
        for role in roles:
            serve = phase.engines[role]
            upper = role.upper()
            if one_node:
                hosts, nodelist = "127.0.0.1", ""
            else:
                lines.append(f'HOSTS_{upper}=("${{RCP_NDCG_HOSTS[@]:{offset}:{serve.replicas}}}")')
                hosts = f'"${{HOSTS_{upper}[@]}}"'
                nodelist = f'"$(IFS=,; echo "${{HOSTS_{upper}[*]}}")"'
            offset += serve.replicas
            start = self._engine_start(role, serve, nodelist)
            steps.append(EngineStep(serve=serve, role=role, start=start, hosts=hosts))
            # An engine that declares no GPUs sees no device -- a step without --gres is allocated all of the
            # job's GRES (srun(1)), which would hand it every device on its node.
            lines += heredoc(f"ENGINE_{upper}", engine_script(serve, cuda="" if serve.resources.gpus == 0 else None))
        if one_node:
            engines_env = shlex.quote(
                engines_env_value(
                    phase.engines,
                    {role: [f"http://127.0.0.1:{phase.engines[role].port}/v1"] for role in roles},
                )
            )
        else:
            engines_env = engines_env_command(
                phase.engines, {role: f'IFS=,; echo "${{HOSTS_{role.upper()}[*]}}"' for role in roles}
            )
        srun = ["srun", "--overlap", "--nodes=1", "--ntasks=1"]  # one task beside the engine steps
        if not one_node:
            # The coordinator's own GPU request is part of the first node's sum: pin it there.
            srun.append("--nodelist=${RCP_NDCG_HOSTS[0]}")
        if job.resources.gpus:
            # Its devices are the node's first res.gpus (reserved ahead of the engines' slices on Kubernetes): the
            # step asks for them itself, and SLURM's per-step CUDA_VISIBLE_DEVICES -- unique per step (gres.html,
            # "GPU Management") -- then names the disjoint remainder, not a co-located engine's devices.
            srun.append(f"--gres=gpu:{job.resources.gpus}")
        coordinator = self._in_container(image, worker_var, srun=srun) if container else f'bash -c "${worker_var}"'
        return [*lines, *supervise(steps, coordinator=coordinator, engines_env=engines_env)]

    def render_job(self, job: JobSpec) -> str:
        """The ``sbatch`` script for ``job`` (the runner's ``resources`` and ``env`` under the job's own)."""
        job = self.options.defaults_for(job)
        res = job.resources
        phases = job.phases
        engines = [e for phase in phases for e in phase.engines.values()]
        nodes = max((sum(e.replicas for e in phase.engines.values()) for phase in phases), default=0)
        out = f"{self.options.log_dir}/%x-%j.out"
        directives = [f"--job-name={job.name}", f"--output={out}"]
        if nodes:
            directives += [f"--nodes={nodes}", "--ntasks-per-node=1"]
        else:
            directives.append("--ntasks=1")
        for flag, value in (
            ("partition", self.options.partition),
            ("account", self.options.account),
            ("qos", self.options.qos),
        ):
            if value:
                directives.append(f"--{flag}={value}")
        # The GPUs of one node are the sum of the engines placed on it, and the coordinator's own request sits
        # on the phase's first node -- co-located engines partition the node's devices, they do not share them.
        # The placement puts the GPU-largest engine on the first node (see _phase_lines), so a phase's first-node
        # request is its coordinator plus its largest engine; every other node hosts one replica. The job asks
        # for the maximum of that over the phases (SLURM's --gres is per node).
        gpus = max(
            [
                (res.gpus if phase.engines else res.gpus)
                + max((e.resources.gpus for e in phase.engines.values()), default=0)
                for phase in phases
            ],
            default=res.gpus,
        )
        if gpus:
            directives.append(f"--gres=gpu:{gpus}")
        # One node holds a replica and, on the first node, the coordinator: their CPUs and memory add up, per
        # phase; the job asks for the maximum over the phases.
        cpus = max(
            [
                (res.cpus or 0) + max((e.resources.cpus or 0 for e in phase.engines.values()), default=0)
                for phase in phases
            ],
            default=res.cpus or 0,
        )
        if cpus:
            directives.append(f"--cpus-per-task={cpus}")
        memory_gb = max(
            [
                (res.memory_gb or 0) + max((e.resources.memory_gb or 0 for e in phase.engines.values()), default=0)
                for phase in phases
            ],
            default=res.memory_gb or 0,
        )
        if any(e.resources.memory_gb is None for e in engines):
            directives.append("--mem=0")  # an engine of unstated memory gets the node's: loading weights needs it
        elif memory_gb:
            directives.append(f"--mem={int(memory_gb * 1024 + 0.5)}M")
        if res.time_limit_s:
            directives.append(f"--time={slurm_time(res.time_limit_s)}")
        directives += self.options.sbatch_args

        container = self.options.container_runtime != "none"
        image = job.image or self.options.image or COORDINATOR_IMAGE
        lines = ["#!/usr/bin/env bash", *(f"#SBATCH {d}" for d in directives), "set -euo pipefail"]
        lines += self.options.setup
        if not phases:
            lines += heredoc(
                "WORKER",
                worker_script(
                    job,
                    install=container,
                    workdir=self.options.workdir,
                    wheelhouse=self.options.wheelhouse,
                    constraints=self.options.constraints,
                ),
            )
            lines.append(self._in_container(image, "WORKER", srun=[]) if container else 'bash -c "$WORKER"')
            return "\n".join(lines) + "\n"
        if nodes > 1:  # every role's replicas are pinned to their slice of the allocation's nodes
            lines.append('mapfile -t RCP_NDCG_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")')
            lines += engines_env_spec()
        for index, phase in enumerate(phases, 1):
            lines += self._phase_lines(job, index, phase, image=image, container=container, one_node=nodes == 1)
        return "\n".join(lines) + "\n"

    def render(self, jobs: Sequence[JobSpec]) -> dict[str, str]:
        """Scripts as submitted."""
        return {job.name: self.render_job(job) for job in jobs}

    # ------------------------------------------------------------------
    # Scheduler calls
    # ------------------------------------------------------------------

    def submit(self, jobs: Sequence[JobSpec]) -> list[JobHandle]:
        """``sbatch --parsable`` each job in order; returns the SLURM job ids.

        Raises:
            RunnerError: ``sbatch`` fails.
        """
        handles: list[JobHandle] = []
        Path(self.options.log_dir).mkdir(parents=True, exist_ok=True)
        for job in jobs:
            job_id = run_cli(["sbatch", "--parsable"], input_text=self.render_job(job)).strip().split(";")[0]
            if not job_id:
                raise RunnerError(f"sbatch returned no job id for {job.name!r}")
            handles.append(job_id)
        return handles

    def status(self, handle: JobHandle) -> JobStatus:
        """``squeue`` while the job is queued or running, ``sacct`` once it has left the queue.

        A non-zero ``squeue`` is what standard Slurm answers for a job that has left the queue (``Invalid job
        id specified``), so it means "not in queue" and the ``sacct`` fallback runs -- it is not an error."""
        states: list[JobStatus] = []
        try:
            queue = run_cli(["squeue", "-h", "-o", "%i %T", "-j", handle])
        except RunnerError:
            queue = ""
        for line in queue.splitlines():
            job_id, _, state = line.strip().partition(" ")
            if job_id == handle:
                states.append(_STATES.get(state.strip(), JobStatus.UNKNOWN))
        if not states:
            for line in run_cli(["sacct", "-n", "-X", "-P", "-o", "JobID,State", "-j", handle]).splitlines():
                job_id, _, state = line.strip().partition("|")
                if job_id.split(".")[0] == handle and state:
                    states.append(_STATES.get(state.split()[0], JobStatus.UNKNOWN))
        return _aggregate(states)

    def logs(self, handle: JobHandle, *, tail: int | None = None) -> str:
        """The job's output file in ``log_dir``.

        Raises:
            RunnerError: no output file for ``handle`` in ``log_dir``.
        """
        log_dir = Path(self.options.log_dir)
        files = sorted(log_dir.glob(f"*-{handle}.out"))
        if not files:
            raise RunnerError(f"no output for SLURM job {handle} in {self.options.log_dir}")
        text = "".join(f"==> {f.name} <==\n{f.read_text(encoding='utf-8', errors='replace')}" for f in files)
        return tail_lines(text, tail)

    def cancel(self, handle: JobHandle) -> None:
        run_cli(["scancel", handle])


__all__ = ["SlurmOptions", "SlurmRunner", "slurm_time"]
