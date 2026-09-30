"""SlurmRunner -- one ``sbatch`` script per :class:`JobSpec`.

The rendered script carries the job's resources as ``#SBATCH`` lines and
runs the worker script (:func:`~rcp_ndcg.runners.script.worker_script`)
either directly on the node or inside a container:

* ``container_runtime: none`` (default) -- the node's environment; ``setup``
  lines (``module load``, ``source .venv/bin/activate``) run first.
* ``container_runtime: apptainer`` -- ``apptainer exec --nv <image> bash -c ...``
  (``image`` is a ``.sif`` path or an image reference, run as ``docker://``).
* ``container_runtime: pyxis`` -- ``srun --container-image=<image> bash -c ...``
  (NVIDIA enroot + pyxis).

In a container the command runs in a stock image with uv and Python 3.12 (the
default :data:`~rcp_ndcg.runners.script.COORDINATOR_IMAGE`), which installs this
release when the job starts; on the node it runs as is.

A job with engine replicas (``JobSpec.serve``) starts them in the same allocation,
one per node, as one background step (``srun --overlap``) under the supervision
script (:func:`~rcp_ndcg.runners.script.supervise`): the engines start once, the
script waits until a replica answers its readiness path, runs the coordinator
with the replicas' URLs, and ends when either ends. The engine step runs with
``--kill-on-bad-exit=1`` and ``--wait``, so one replica that exits ends the whole
step, and with it the job (non-zero). One replica is ``localhost``; several are
the allocation's nodes, read from ``SLURM_JOB_NODELIST`` when the job starts. Runs,
stores and caches live on the cluster's shared filesystem, which the submitting
host sees too.
"""

from __future__ import annotations

import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from rcp_ndcg.runners._cli import run_cli
from rcp_ndcg.runners.base import JobHandle, JobOptions, JobSpec, JobStatus, RunnerError, tail_lines
from rcp_ndcg.runners.script import COORDINATOR_IMAGE, engine_script, heredoc, supervise, worker_script
from rcp_ndcg.support.serve import JUDGE_URLS_ENV, ServeConfig

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


class SlurmRunner:
    """Submit jobs with ``sbatch``; query with ``squeue``/``sacct``; cancel with ``scancel``.

    Options (``get_runner("slurm", **options)``): :class:`SlurmOptions`.
    """

    name = "slurm"

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

    def _engine_lines(self, serve: ServeConfig, coordinator: str) -> list[str]:
        """Start one engine per node in the background, run ``coordinator`` beside it, and end when either ends."""
        serve.check_nodes()
        # One replica that exits, even with status 0, ends the step: at once for a failure, else after 10 s.
        step = ["srun", "--overlap", f"--nodes={serve.replicas}", "--ntasks-per-node=1", "--kill-on-bad-exit=1"]
        step.append(f"--wait={ENGINE_STEP_WAIT_S}")
        if serve.resources.gpus:
            step.append(f"--gres=gpu:{serve.resources.gpus}")
        engine = self._in_container(serve.image, "ENGINE", srun=step)  # on the node itself for runtime none
        if serve.replicas == 1:
            hosts = ["HOSTS=(127.0.0.1)"]
        else:
            hosts = ['mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")']
        return [
            *heredoc("ENGINE", engine_script(serve)),
            *hosts,
            f'URLS=(); for host in "${{HOSTS[@]}}"; do URLS+=("http://$host:{serve.port}/v1"); done',
            f'{JUDGE_URLS_ENV}="$(IFS=,; echo "${{URLS[*]}}")"',
            f"export {JUDGE_URLS_ENV}",
            *supervise(serve, engine=engine, coordinator=coordinator, hosts='"${HOSTS[@]}"'),
        ]

    def render_job(self, job: JobSpec) -> str:
        """The ``sbatch`` script for ``job``."""
        res = job.resources
        serve = job.serve
        out = f"{self.options.log_dir}/%x-%j.out"
        directives = [f"--job-name={job.name}", f"--output={out}"]
        if serve is None:
            directives.append("--ntasks=1")
        else:
            directives += [f"--nodes={serve.replicas}", "--ntasks-per-node=1"]
        for flag, value in (
            ("partition", self.options.partition),
            ("account", self.options.account),
            ("qos", self.options.qos),
        ):
            if value:
                directives.append(f"--{flag}={value}")
        gpus = max(res.gpus, serve.resources.gpus if serve is not None else 0)
        if gpus:
            directives.append(f"--gres=gpu:{gpus}")
        # One node holds a replica and, on the first node, the coordinator: their CPUs and memory add up.
        engine = serve.resources if serve is not None else None
        cpus = (res.cpus or 0) + (engine.cpus or 0 if engine else 0)
        if cpus:
            directives.append(f"--cpus-per-task={cpus}")
        memory_gb = (res.memory_gb or 0) + (engine.memory_gb or 0 if engine else 0)
        if engine is not None and engine.memory_gb is None:
            directives.append("--mem=0")  # an engine of unstated memory gets the node's: loading weights needs it
        elif memory_gb:
            directives.append(f"--mem={int(memory_gb * 1024 + 0.5)}M")
        if res.time_limit_s:
            directives.append(f"--time={slurm_time(res.time_limit_s)}")
        directives += self.options.sbatch_args

        container = self.options.container_runtime != "none"
        image = job.image or self.options.image or COORDINATOR_IMAGE
        worker = worker_script(job, install=container, workdir=self.options.workdir)
        lines = ["#!/usr/bin/env bash", *(f"#SBATCH {d}" for d in directives), "set -euo pipefail"]
        lines += self.options.setup
        lines += heredoc("WORKER", worker)
        if serve is None:
            lines.append(self._in_container(image, "WORKER", srun=[]) if container else 'bash -c "$WORKER"')
        else:
            srun = ["srun", "--overlap", "--nodes=1", "--ntasks=1"]  # one task beside the engine step
            coordinator = self._in_container(image, "WORKER", srun=srun) if container else 'bash -c "$WORKER"'
            lines += self._engine_lines(serve, coordinator)
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
        """``squeue`` while the job is queued or running, ``sacct`` once it has left the queue."""
        states: list[JobStatus] = []
        for line in run_cli(["squeue", "-h", "-o", "%i %T", "-j", handle]).splitlines():
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
