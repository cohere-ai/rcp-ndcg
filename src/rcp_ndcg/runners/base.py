"""The job-execution interface: :class:`JobSpec` in, :class:`JobRunner` out.

A :class:`JobSpec` is *what* to run: one command line (argv) with its image,
environment and resource request, or the phases to run in order
(:class:`JobPhase`, each with the engine replicas to start by role and the
command to run while they serve) -- exactly one of the two: a job without phases
runs its ``argv``, a phased job's commands are its phases' ``argv``. A
:class:`JobRunner` decides
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

import os
import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, ClassVar, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from rcp_ndcg.errors import ConfigError, ProviderError
from rcp_ndcg.support.resources import Environment, EnvName, Resources
from rcp_ndcg.support.serve import EngineConfig, EngineRole

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


class JobPhase(BaseModel):
    """One phase of a job: the engines it starts, by role, and the command it runs while they serve.

    The runner starts the phase's engines, waits until each answers its readiness path, exports their URLs to the
    command in ``RCP_NDCG_ENGINES`` (:data:`~rcp_ndcg.support.serve.ENGINES_ENV`), runs ``argv``, stops the engines,
    and only then starts the next phase. A phase without engines runs ``argv`` directly.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    engines: Mapping[EngineRole, EngineConfig] = Field(default_factory=dict)
    """The engines this phase starts, by the role whose config each one serves; empty for none."""
    argv: tuple[str, ...]
    """The coordinator command of this phase, never pre-quoted (the runner quotes it exactly once)."""

    @field_validator("argv")
    @classmethod
    def _non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("argv must not be empty")
        return value


class JobSpec(BaseModel):
    """One unit of work for a runner: one command line, or the phases to run in order.

    Exactly one of ``argv`` and ``phases``: a job without phases runs ``argv``; a phased job's commands are
    its phases' ``argv``, and it carries no ``argv`` of its own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
    """Lowercase alphanumerics and ``-``: the job's name on the scheduler (a runner whose scheduler
    limits the length, like Kubernetes' 63, shortens it deterministically)."""
    argv: tuple[str, ...] | None = None
    """The command each task runs, never pre-quoted (the runner quotes it exactly once); the command of a job
    without phases. A phased job takes none: its commands are its phases' ``argv``."""
    image: str | None = None
    """Container image; ``None`` uses the runner's configured image (or none, for host execution)."""
    resources: Resources = Resources()
    env: Mapping[EnvName, str] = Field(default_factory=dict)
    """Environment for the command; each name a shell identifier."""
    phases: tuple[JobPhase, ...] = ()
    """The job's phases, run in order in one allocation; when set, ``argv`` must be left unset (a phased
    job's commands are its phases' ``argv``).

    Each phase starts its engines, waits until each role has a replica answering its readiness path, exports
    their URLs to the phase's command in ``RCP_NDCG_ENGINES`` (:data:`~rcp_ndcg.support.serve.ENGINES_ENV`), runs
    it, and stops the engines before the next phase starts; a phase without engines runs its command directly.
    """

    @field_validator("argv")
    @classmethod
    def _non_empty(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is not None and not value:
            raise ValueError("argv must not be empty")
        return value

    @model_validator(mode="after")
    def _argv_xor_phases(self) -> Self:
        if self.phases and self.argv is not None:
            raise ValueError("a phased job takes no argv: its commands are its phases' argv")
        if not self.phases and self.argv is None:
            raise ValueError("a job takes argv, or phases")
        return self

    def with_argv(self, argv: Sequence[str]) -> JobSpec:
        """The same job running ``argv``: the spec a runner renders for one phase of this job (its image,
        resources and environment; no phases -- a phase's command lives in the job's ``phases``).

        Raises:
            ValueError: ``argv`` is empty, or a string (``model_copy`` would skip the model's own check: an
                empty command would crash the rendered script, and a string would be char-split into words).
        """
        if isinstance(argv, str):
            raise ValueError("argv must be a sequence of words, not a string")
        if not argv:
            raise ValueError("argv must not be empty")
        return self.model_copy(update={"argv": tuple(argv), "phases": ()})


class JobOptions(BaseModel):
    """What every job of a runner asks for by default: its resources and its environment.

    Each public runner's options model extends it with the runner's own fields; a run config's ``runner.options``
    is that model, so a typo fails when the config is read. The runner applies them to each job it renders
    (:meth:`defaults_for`): a job's own resources and environment win, field by field and name by name.
    """

    model_config = ConfigDict(extra="forbid")

    #: The options that name a path on the submitting host (made absolute by :meth:`resolved`).
    PATHS: ClassVar[tuple[str, ...]] = ()

    resources: Resources = Resources()
    env: Environment = Field(default_factory=dict)
    wheelhouse: str | None = Field(default=None, min_length=1)
    """Where the coordinator installs the release from instead of PyPI: a directory of staged wheels, or an
    ``http(s)://`` URL of one, readable on the node; rendered as ``uvx --find-links <wheelhouse> --no-index``
    -- every package comes from the wheelhouse (a pre-release, or an air-gapped node). A relative local path is
    recorded absolute by the runners whose job records read paths on the submitting host (slurm); the value is
    always rendered verbatim, so it must be readable where the job runs (a container: mounted in). On an
    air-gapped node set ``constraints`` too: the default release constraints URL is fetched at job start even
    under ``--no-index``. Only runners that install the release take one (a container on SLURM, Kubernetes);
    where the coordinator runs in an environment that already has it, setting one is refused. The wheels are
    built and staged as ``docs/concepts/serving.md`` ("The coordinator installs itself") describes.
    """
    constraints: str | None = Field(default=None, min_length=1)
    """A constraints file (path or URL) replacing the release's, which pins every dependency to the version
    the release was tested with; default the release's own, attached to its GitHub release."""

    @field_validator("wheelhouse", "constraints")
    @classmethod
    def _readable_by_uv(cls, value: str | None) -> str | None:
        """Refuse what ``uv`` cannot read: ``--find-links`` and ``--constraints`` take a local path, a
        ``file://`` URL or an ``http(s)://`` URL with a host -- a bucket scheme (``gs://``, ``s3://``, also
        single-slashed like ``gs:/x``) is treated as a filesystem path and fails at job start, so the config
        refuses it and names the fix."""
        if value is None:
            return value
        if value != value.strip() or not value.strip():
            raise ValueError(f"{value!r} is empty or has surrounding whitespace")
        if re.match(r"[A-Za-z][A-Za-z0-9+.\-]*:/", value) and not value.lower().startswith(
            ("http://", "https://", "file://")
        ):
            raise ValueError(
                f"{value!r} names a scheme uv cannot read: --find-links and --constraints take a local path, a "
                "file:// URL or an http(s):// URL; stage the wheelhouse as an https URL, or mount it on the node",
            )
        if value.lower().startswith(("http://", "https://")) and not re.match(r"^https?://[^/]+", value, re.IGNORECASE):
            raise ValueError(
                f"{value!r} names no host: --find-links and --constraints take a local path, a file:// URL or an "
                "http(s):// URL of one"
            )
        return value

    def resolved(self) -> dict[str, Any]:
        """The options set away from their defaults, every local path among them absolute (:data:`PATHS`).

        A PATHS value that is a URL (it names its location with ``://``) is kept as it is. What a job record
        keeps: the runner re-created from it finds the job's files from any working directory.
        """
        data = self.model_dump(mode="json", exclude_defaults=True)
        for name in self.PATHS:
            value = getattr(self, name)
            if value is not None and "://" not in value:
                data[name] = os.path.abspath(os.path.expanduser(value))
        return data

    def defaults_for(self, job: JobSpec) -> JobSpec:
        """``job`` with these resources and environment under its own (the job's values win)."""
        resources = {**self.resources.model_dump(exclude_unset=True), **job.resources.model_dump(exclude_unset=True)}
        return job.model_copy(update={"resources": Resources(**resources), "env": {**self.env, **job.env}})

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
    "JobPhase",
    "JobRunner",
    "JobSpec",
    "JobStatus",
    "Resources",
    "RunnerError",
    "tail_lines",
]
