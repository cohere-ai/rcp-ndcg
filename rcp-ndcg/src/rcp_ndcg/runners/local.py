"""LocalRunner -- run each :class:`JobSpec` on the calling host, one after another.

Each job runs its worker script (:func:`~rcp_ndcg.runners.script.worker_script`)
with ``bash``. Jobs run in submission order. :meth:`LocalRunner.submit` returns when
they have all finished, or at once with ``detach``: the jobs then run in a
background session of their own.

With a ``log_dir``, every job leaves files there that :meth:`LocalRunner.status`,
:meth:`LocalRunner.logs` and :meth:`LocalRunner.cancel` read from any process:
``<name>.log`` (its output), ``<name>.exit`` (its exit code, once it finished)
and, for a detached job, ``<name>.session`` (the process group running it).

The command runs in the calling environment (an activated venv with
``rcp-ndcg`` installed). The local runner starts no engine: a job whose phases would start one is refused; start
the engine yourself and point the run at it (``run resume --engine <role>=<url>``).
"""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import signal
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

from pydantic import Field, model_validator

from rcp_ndcg.errors import ConfigError, ExitCode, RcpNdcgError, error_class
from rcp_ndcg.runners.base import JobHandle, JobOptions, JobSpec, JobStatus, RunnerError, tail_lines
from rcp_ndcg.runners.script import STOP_GRACE_S, worker_script

log = logging.getLogger(__name__)


class LocalOptions(JobOptions):
    """The ``local`` runner's options.

    Attributes:
        cwd: The working directory; default the caller's.
        log_dir: Each job's output goes to ``<log_dir>/<name>.log`` instead of the terminal.
        detach: Return at once and run the jobs in the background (needs ``log_dir``).
    """

    PATHS = ("cwd", "log_dir")

    cwd: str | None = None
    log_dir: str | None = None
    detach: bool = False
    # The install source is inherited but never accepted: the local runner installs nothing, so its schema
    # carries the refusal instead of advertising the fields (a schema-driven client would emit configs that
    # always fail).
    wheelhouse: str | None = Field(
        default=None,
        min_length=1,
        description="not taken: the local runner installs nothing (it runs the coordinator in this host's environment)",
    )
    constraints: str | None = Field(
        default=None,
        min_length=1,
        description="not taken: the local runner installs nothing (it runs the coordinator in this host's environment)",
    )

    @model_validator(mode="after")
    def _detached_jobs_are_followed_through_files(self) -> Self:
        if self.detach and not self.log_dir:
            raise ValueError("a detached local runner needs `log_dir`: its jobs are followed through their files")
        if self.wheelhouse or self.constraints:
            raise ValueError(
                "the local runner installs nothing (it runs the coordinator in this host's environment): "
                "drop wheelhouse and constraints, or hand the run to a runner that installs it "
                "(kubernetes, or slurm with a container runtime)"
            )
        return self


class LocalRunner:
    """Run jobs on the calling host.

    Options (``get_runner("local", **options)``): :class:`LocalOptions`; its ``resources`` and ``env`` are every
    job's defaults (the local runner requests no resources: the job gets what the host has).
    """

    name = "local"

    #: The local runner runs a job's phases (its argv covers every engine-free phase of the plan) but starts no
    #: engine: a phase with engines is refused, with a hint about running the engines yourself.
    renders_phases = True

    def __init__(self, **options: Any) -> None:
        self.options = LocalOptions.parse(self.name, options)
        self.cwd = self.options.cwd or str(Path.cwd())
        self.log_dir = Path(self.options.log_dir) if self.options.log_dir else None
        self.detach = self.options.detach
        self._statuses: dict[str, JobStatus] = {}

    def render(self, jobs: Sequence[JobSpec]) -> dict[str, str]:
        """The worker script each job runs: a phased job's phases in order (their commands are the job's
        commands), a plain job its ``argv``. Submits nothing.

        Raises:
            ConfigError: a job has a phase that starts an engine, which the local runner does not start.
        """
        for job in jobs:
            serving = sorted(role for phase in job.phases for role in phase.engines)
            if serving:
                raise ConfigError(
                    f"job {job.name!r} starts engine(s) for role(s) {', '.join(serving)}, and the local runner "
                    "starts no engine",
                    hint="start the engine(s) yourself (docs/concepts/runs.md) and pass the URLs: "
                    "rcp-ndcg run resume --run <run dir> --engine <role>=<url>[,<url>]",
                )
        return {
            job.name: worker_script(self.options.defaults_for(job), install=False, workdir=self.cwd) for job in jobs
        }

    def submit(self, jobs: Sequence[JobSpec]) -> list[JobHandle]:
        """Run every job in order, stopping at the first failure; with ``detach``, start them and return.

        Returns:
            The job names, which are the handles.

        Raises:
            RcpNdcgError: a job exited non-zero (later jobs are not started): the class of its exit code
                (``MissingInputError`` for 4, ...), else :class:`RunnerError`. A detached submission never raises
                it; :meth:`status` reports the failure.
        """
        scripts = self.render(jobs)
        if self.detach:
            return self._start_detached(jobs, scripts)
        handles: list[JobHandle] = []
        for job in jobs:
            handles.append(job.name)
            self._statuses[job.name] = JobStatus.RUNNING
            log.info("[local] running %s", job.name)
            rc = self._run(job.name, scripts[job.name], dict(os.environ))
            if rc != 0:
                self._statuses[job.name] = JobStatus.FAILED
                raise self._failure(job.name, rc)
            self._statuses[job.name] = JobStatus.COMPLETED
        return handles

    def _failure(self, name: str, rc: int) -> RcpNdcgError:
        """A job's non-zero exit as the error class of its exit code (an ``rcp-ndcg`` command's own), so a caller
        sees the job's failure, not a runner's; a code no class uses is a :class:`RunnerError` that is not
        retryable."""
        where = f"its output is in {self.log_dir / f'{name}.log'}" if self.log_dir else "its output is above"
        known = error_class(rc)
        details = {"job": name, "exit_code": rc}
        if known is None:
            return RunnerError(f"local job {name!r} exited {rc}", hint=where, details=details, retryable=False)
        return known(f"local job {name!r} exited {rc} ({ExitCode(rc).name})", hint=where, details=details)

    def _run(self, name: str, script: str, env: dict[str, str]) -> int:
        if self.log_dir is None:
            return subprocess.run(["bash", "-c", script], cwd=self.cwd, env=env, check=False).returncode
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.log_dir / f"{name}.log").open("a", encoding="utf-8") as fh:
            rc = subprocess.run(
                ["bash", "-c", script], cwd=self.cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, check=False
            ).returncode
        (self.log_dir / f"{name}.exit").write_text(f"{rc}\n", encoding="utf-8")
        return rc

    def _start_detached(self, jobs: Sequence[JobSpec], scripts: dict[str, str]) -> list[JobHandle]:
        """Start one background shell that runs the jobs in order, each writing its log and then its exit code."""
        assert self.log_dir is not None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        steps = []
        for job in jobs:
            for stale in (".exit", ".session", ".cancelled"):
                (self.log_dir / f"{job.name}{stale}").unlink(missing_ok=True)
            log_path = shlex.quote(str(self.log_dir / f"{job.name}.log"))
            exit_path = shlex.quote(str(self.log_dir / f"{job.name}.exit"))
            steps.append(
                f"bash -c {shlex.quote(scripts[job.name])} >> {log_path} 2>&1; rc=$?; "
                f"echo $rc > {exit_path}; [ $rc -eq 0 ] || exit $rc"
            )
        # The launcher leads a new session and exits at once; the jobs run in its background child, which
        # outlives it in the same process group, so nothing here waits for them or leaves a zombie behind.
        launcher = subprocess.Popen(
            ["bash", "-c", f"( {'; '.join(steps)} ) </dev/null >/dev/null 2>&1 &"],
            cwd=self.cwd,
            env=dict(os.environ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        launcher.wait()
        for job in jobs:
            (self.log_dir / f"{job.name}.session").write_text(f"{launcher.pid}\n", encoding="utf-8")
        log.info("[local] started %s in the background (session %d)", [job.name for job in jobs], launcher.pid)
        return [job.name for job in jobs]

    def status(self, handle: JobHandle) -> JobStatus:
        """The job's state: this runner's own record, else its exit code, else whether its session still runs."""
        if handle in self._statuses:
            return self._statuses[handle]
        if self.log_dir is None:
            return JobStatus.UNKNOWN
        exit_path = self.log_dir / f"{handle}.exit"
        if exit_path.is_file():
            return JobStatus.COMPLETED if exit_path.read_text(encoding="utf-8").strip() == "0" else JobStatus.FAILED
        session = self._session(handle)
        if session is None:
            return JobStatus.UNKNOWN
        if not _session_alive(session):
            return JobStatus.CANCELLED if (self.log_dir / f"{handle}.cancelled").exists() else JobStatus.FAILED
        return JobStatus.RUNNING if (self.log_dir / f"{handle}.log").exists() else JobStatus.PENDING

    def _session(self, handle: JobHandle) -> int | None:
        assert self.log_dir is not None
        path = self.log_dir / f"{handle}.session"
        return int(path.read_text(encoding="utf-8")) if path.is_file() else None

    def logs(self, handle: JobHandle, *, tail: int | None = None) -> str:
        """The job's log file; only kept when ``log_dir`` is configured.

        Raises:
            RunnerError: no ``log_dir`` (output went to the terminal) or no log for ``handle``.
        """
        if self.log_dir is None:
            raise RunnerError("the local runner keeps no logs unless `log_dir` is set; output went to the terminal")
        path = self.log_dir / f"{handle}.log"
        if not path.is_file():
            raise RunnerError(f"no log for local job {handle!r} at {path}")
        return tail_lines(path.read_text(encoding="utf-8"), tail)

    def cancel(self, handle: JobHandle) -> None:
        """Stop a detached job that has not finished (a foreground job has finished when :meth:`submit` returns).

        The job's process group gets ``SIGTERM``, then ``SIGKILL`` after :data:`~rcp_ndcg.runners.script.STOP_GRACE_S`
        seconds if any process of the group is still alive, and the group is then checked: a coordinator that
        ignores ``SIGTERM`` must not leave the run recorded ``cancelled`` while it keeps running.

        Raises:
            RunnerError: the job's files are not in ``log_dir`` (no ``log_dir``, another one, or no such job),
                so it cannot be found or stopped; or the process group survived ``SIGKILL``.
        """
        state = self.status(handle)
        if state in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            return
        session = self._session(handle) if self.log_dir is not None else None
        if session is None:
            raise RunnerError(
                f"no local job {handle!r} to cancel: its session file is not in {self.log_dir}",
                hint="a local job is followed through the files of its log_dir; pass the runner that started it",
                retryable=False,
            )
        assert self.log_dir is not None
        (self.log_dir / f"{handle}.cancelled").touch()
        if not _session_alive(session):  # it ended between the status read and here
            return
        _stop_group(session, grace_s=STOP_GRACE_S)


def _stop_group(session: int, *, grace_s: float = STOP_GRACE_S) -> None:
    """``SIGTERM`` a process group, ``SIGKILL`` what is left after ``grace_s``, and check it is gone.

    Raises:
        RunnerError: the group still has a process after ``SIGKILL``.
    """
    with contextlib.suppress(ProcessLookupError):
        os.killpg(session, signal.SIGTERM)
    if _wait_gone(session, grace_s):
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(session, signal.SIGKILL)
    if not _wait_gone(session, grace_s):
        raise RunnerError(
            f"the process group {session} of the cancelled job did not stop after SIGKILL",
            hint="a process in the group is unkillable (a D-state or another user's process); check the host",
            retryable=False,
        )


def _wait_gone(session: int, timeout_s: float) -> bool:
    """Whether every process of the group ``session`` is gone within ``timeout_s`` seconds."""
    deadline = time.monotonic() + timeout_s
    while _session_alive(session):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def _session_alive(session: int) -> bool:
    """Whether any process of the process group ``session`` still runs."""
    try:
        os.killpg(session, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # it exists, under another user
    return True


__all__ = ["LocalRunner"]
