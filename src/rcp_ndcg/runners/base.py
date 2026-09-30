"""The job-execution interface: :class:`JobSpec` in, :class:`JobRunner` out.

A :class:`JobSpec` is *what* to run: one command line (argv) with its image,
environment and resource request, and optionally the engine replicas to start
beside it (:class:`~rcp_ndcg.support.serve.ServeConfig`). A :class:`JobRunner` decides
*where*: the calling host (:class:`~rcp_ndcg.runners.local.LocalRunner`), a SLURM
cluster (:class:`~rcp_ndcg.runners.slurm.SlurmRunner`) or Kubernetes
(:class:`~rcp_ndcg.runners.kubernetes.KubernetesRunner`). Further runners are
separate packages that register under the ``rcp_ndcg.runners`` entry-point group
(:mod:`rcp_ndcg.runners.registry`) and implement the same four methods. A runner
transports commands; it knows nothing about what they compute.

Each public runner validates its options with one model (:class:`JobOptions` and
the runner's own fields), which a run config's ``runner.options`` is typed by.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from rcp_ndcg.errors import ConfigError, ProviderError
from rcp_ndcg.support.resources import Resources
from rcp_ndcg.support.serve import ServeConfig

#: An opaque job reference returned by :meth:`JobRunner.submit` (a SLURM job id,
#: ``<namespace>/<job>`` on Kubernetes, the job name locally).
JobHandle = str


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class RunnerError(ProviderError):
    """A scheduler command failed, a job could not be submitted as specified, or a job failed (exit 6)."""


class JobSpec(BaseModel):
    """One unit of work for a runner: a named command."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
    """Lowercase alphanumerics and ``-``: the job's name on the scheduler (a runner whose scheduler
    limits the length, like Kubernetes' 63, shortens it deterministically)."""
    argv: tuple[str, ...]
    """The command each task runs, never pre-quoted (the runner quotes it exactly once)."""
    image: str | None = None
    """Container image; ``None`` uses the runner's configured image (or none, for host execution)."""
    resources: Resources = Resources()
    env: Mapping[str, str] = Field(default_factory=dict)
    """Environment for the command."""
    serve: ServeConfig | None = None
    """Engine replicas the runner starts beside the command; their URLs reach it as ``RCP_NDCG_JUDGE_URLS``."""

    @field_validator("argv")
    @classmethod
    def _non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("argv must not be empty")
        return value


class JobOptions(BaseModel):
    """What a run's one job asks of any runner: its resources and its environment.

    Each public runner's options model extends it with the runner's own fields; a run config's ``runner.options``
    is that model, so a typo fails when the config is read.
    """

    model_config = ConfigDict(extra="forbid")

    resources: Resources = Resources()
    env: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def parse(cls, runner: str, options: Mapping[str, Any]) -> Self:
        """``options`` validated for the ``runner`` runner.

        Raises:
            ConfigError: an unknown option, or a value of the wrong type.
        """
        try:
            return cls.model_validate(dict(options))
        except ValidationError as exc:
            raise ConfigError(f"invalid options for the {runner!r} runner: {exc}") from exc


class JobRunner(Protocol):
    """Where jobs run. Instantiated with the runner's options (``get_runner(name, **options)``).

    A runner whose jobs do not see the submitting host's files sets ``run_root``: the directory a run is restored
    into inside the job, from the run's mirror (:mod:`rcp_ndcg.runs.mirror`). Runners without it share the files.
    """

    name: str

    def submit(self, jobs: Sequence[JobSpec]) -> list[JobHandle]:
        """Submit ``jobs`` in order; one handle per job."""
        ...

    def status(self, handle: JobHandle) -> JobStatus:
        """The job's current status. Non-blocking."""
        ...

    def logs(self, handle: JobHandle, *, tail: int | None = None) -> str:
        """The job's output so far; ``tail`` keeps the last lines."""
        ...

    def cancel(self, handle: JobHandle) -> None:
        """Stop the job; a finished job is left as it is."""
        ...


def tail_lines(text: str, tail: int | None) -> str:
    """The last ``tail`` lines of ``text`` (all of it for ``None``)."""
    if tail is None:
        return text
    lines = text.splitlines(keepends=True)
    return "".join(lines[-tail:]) if tail > 0 else ""


__all__ = [
    "JobHandle",
    "JobOptions",
    "JobRunner",
    "JobSpec",
    "JobStatus",
    "Resources",
    "RunnerError",
    "tail_lines",
]
