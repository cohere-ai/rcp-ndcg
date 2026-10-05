"""The engines a job starts by role: :class:`ServeConfig` (an engine), and the serve-by-role types that phase
planning builds on.

A leaf model, like :class:`~rcp_ndcg.support.resources.Resources`: a job's phases declare their engines
(``JobPhase.engines``) and the job runners render them, without the run layer importing the runners; the single
``serve:`` section of a run config is no longer submitted (it is refused).

rcp-ndcg's contract with a model is one OpenAI-compatible URL. An engine does not change that: the package never
builds, translates or reads an engine's flags. It starts the user's image with the user's command, waits until
``GET <readiness_path>`` answers, and hands the replicas' URLs to the run's judge as its ``base_url`` list.

A job that owns its engine never outlives it: an engine that dies, or does not answer within
``startup_timeout_s``, ends the job with a non-zero exit, and the judge of such a job stops waiting for an engine
that stopped answering after ``outage_timeout_s``. Resuming is cheap (the stores are asked only for the windows they
lack): ``rcp-ndcg run resume --run <dir> --runner slurm|kubernetes`` submits the run again, engine included.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.support.resources import Environment, Resources

#: The environment variable a rendered job used to set to the replica URLs (comma-separated), read by
#: ``rcp-ndcg run resume --judge-urls``. DEPRECATED, to be deleted: the runners export ``ENGINES_ENV`` now, and
#: ``rcp-ndcg run resume`` gains ``--engine role=url``; only the CLI's old flag still reads it.
JUDGE_URLS_ENV = "RCP_NDCG_JUDGE_URLS"

#: The environment variable a phase's runner sets to the engines of the current phase, as JSON
#: ``{"encoder": {"urls": [...], "wait_on_outage_s": 900}, ...}``; the coordinator
#: applies it as a runtime overlay. It is never written into ``run.yaml`` and never reaches an identity, since
#: ``base_url`` and ``wait_on_outage_s`` are runtime fields. It has replaced ``JUDGE_URLS_ENV`` in the runners;
#: the old variable remains only for ``rcp-ndcg run resume --judge-urls`` until that flag goes.
ENGINES_ENV = "RCP_NDCG_ENGINES"


class ServeConfig(BaseModel):
    """The engine replicas a run's job starts for its judge.

    Attributes:
        image: The engine's container image, e.g. ``vllm/vllm-openai:<tag>`` or ``lmsysorg/sglang:<tag>``; pin the
            tag. Kubernetes and the SLURM runner's container runtimes need it; with the SLURM runner's
            ``container_runtime: none`` the command runs on the node, and an image is refused (it would be ignored).
        command: The command that starts one replica, verbatim: an argv list, or one shell-quoted string. It must
            serve the judge's ``model`` name on ``port`` on all interfaces (``--host 0.0.0.0``) when there are
            several replicas.
        env: Environment of the engine (e.g. ``HF_HOME``).
        resources: What one replica needs (``gpus``, ``cpus``, ``memory_gb``).
        replicas: Independent engine replicas, one URL each; the judge sends each request to the one with the
            fewest requests in flight.
        port: The port each replica serves on.
        readiness_path: The path that answers once a replica serves (``GET``, any 2xx).
        nodes_per_replica: Nodes one replica spans; only 1 is implemented (another value fails validation).
        startup_timeout_s: Seconds a job waits for a replica to answer ``readiness_path`` before it fails; an engine
            that exits before it answers fails the job at once. 0 fails unless the first probe answers.
        outage_timeout_s: Seconds the judge of the job waits while every replica is down before it fails with
            ``BackendUnavailableError`` (it becomes the judge's ``wait_on_outage_s`` in the job).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: str | None = Field(default=None, min_length=1)
    command: tuple[str, ...] = Field(min_length=1)
    env: Environment = Field(default_factory=dict)
    resources: Resources = Resources()
    replicas: int = Field(default=1, ge=1)
    port: int = Field(default=8000, ge=1, le=65535)
    readiness_path: str = Field(default="/v1/models", pattern=r"^/[A-Za-z0-9/_.~%=&?-]*$")
    nodes_per_replica: int = Field(default=1, ge=1)
    startup_timeout_s: int = Field(default=1800, ge=0)
    outage_timeout_s: int = Field(default=900, ge=0)

    @field_validator("nodes_per_replica")
    @classmethod
    def _one_node_per_replica(cls, value: int) -> int:
        if value != 1:
            raise ValueError(
                f"nodes_per_replica={value}: a replica spanning several nodes is not implemented; run one replica "
                "per node (nodes_per_replica: 1) and scale out with replicas"
            )
        return value

    @field_validator("command", mode="before")
    @classmethod
    def _a_string_is_split(cls, value: Any) -> Any:
        return tuple(shlex.split(value)) if isinstance(value, str) else value

    def url(self, host: str) -> str:
        """The OpenAI-compatible base URL of the replica on ``host``."""
        return f"http://{host}:{self.port}/v1"


#: A role an engine can serve. The role fixes which config the engine serves: ``judge`` serves the run's judge,
#: ``encoder`` serves the retrieval config's encoder, ``reranker`` its reranker.
EngineRole = Literal["judge", "encoder", "reranker"]

#: The engine a role runs: today's single-engine ``ServeConfig`` (an alias, so every existing use keeps
#: working while ``serve:`` grows a role per engine).
EngineConfig = ServeConfig


class ServeByRole(BaseModel):
    """The engines of a run, by role: at most one engine per role, each an :class:`EngineConfig`.

    Attributes:
        judge: The engine serving the judge, when the job starts one (``None``: the judge is reached at the
            URLs its config already holds).
        encoder: The engine serving the retrieval config's encoder.
        reranker: The engine serving the retrieval config's reranker.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    judge: EngineConfig | None = None
    encoder: EngineConfig | None = None
    reranker: EngineConfig | None = None


@dataclass(frozen=True)
class Phase:
    """One phase of a phased run: the engines it starts, and the steps it runs.

    Attributes:
        engines: The roles whose engines the phase starts and stops around its steps.
        steps: The run's steps of this phase, in the order they run.
    """

    engines: frozenset[EngineRole]
    steps: tuple[str, ...]


class EngineURLs(BaseModel):
    """The URLs of one role's engines in the current phase, as ``ENGINES_ENV`` carries them.

    Attributes:
        urls: The replica base URLs of the role's engine, one per replica; at least one.
        wait_on_outage_s: How long a request waits while every replica of the role is down before
            :class:`~rcp_ndcg.errors.BackendUnavailableError`; ``None`` waits indefinitely. Runtime: applied as
            an overlay, never written into a config and never reaching an identity.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    urls: tuple[str, ...] = Field(min_length=1)
    wait_on_outage_s: float | None = Field(default=None, ge=0)

    @field_validator("urls")
    @classmethod
    def _real_urls(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        urls = tuple(url.rstrip("/") for url in value)
        if any(not url for url in urls):
            raise ValueError("urls: every entry must be a non-empty base URL")
        return urls


def parse_engines_env(text: str) -> dict[EngineRole, EngineURLs]:
    """Parse an ``ENGINES_ENV`` value into one :class:`EngineURLs` per role it names.

    Args:
        text: The variable's value: JSON of the shape ``{"encoder": {"urls": [...], "wait_on_outage_s":
            900}, ...}``; an empty object is a phase without engines.

    Returns:
        One :class:`EngineURLs` per role named in ``text``.

    Raises:
        ConfigError: ``text`` is not JSON, not an object, or names a role that is not one of ``judge``,
            ``encoder`` or ``reranker``; a value that does not validate as :class:`EngineURLs` is named the
            same way.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{ENGINES_ENV} is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})",
            hint=f"{ENGINES_ENV} holds the engines of the current phase as JSON, e.g. "
            '"{"encoder": {"urls": ["http://127.0.0.1:8000/v1"], "wait_on_outage_s": 900}}"',
            details={"variable": ENGINES_ENV},
        ) from exc
    if not isinstance(parsed, dict):
        raise ConfigError(
            f"{ENGINES_ENV} must be a JSON object of role -> urls, got {type(parsed).__name__}",
            hint='expect roles "judge", "encoder" or "reranker", each with its "urls"',
            details={"variable": ENGINES_ENV},
        )
    known = ("judge", "encoder", "reranker")
    unknown = sorted(str(role) for role in parsed if role not in known)
    if unknown:
        raise ConfigError(
            f"{ENGINES_ENV} names unknown engine role(s) {unknown}",
            hint=f"known roles: {', '.join(known)}",
            details={"variable": ENGINES_ENV, "unknown": unknown, "known": list(known)},
        )
    from pydantic import ValidationError

    from rcp_ndcg.support.config import config_error

    try:
        return {role: EngineURLs.model_validate(entry) for role, entry in parsed.items()}
    except ValidationError as exc:
        raise config_error(exc) from exc


def plan_phases(
    steps: Sequence[str],
    serve: ServeByRole,
    uses: Mapping[str, frozenset[EngineRole]],
) -> list[Phase]:
    """The phase plan of a run: a pure function of the steps, the engines and their use.

    A phased run starts each phase's engines, waits for readiness, runs the phase's steps, and stops its
    engines, so the job's GPUs are the maximum over phases instead of the sum over engines. Consecutive steps
    that use the same engines share a phase; steps that call no model of a served engine (``calibrate``,
    ``evaluate``, BM25 retrieval, hosted APIs) form phases without engines. The paper's run becomes four
    phases: ``retrieve`` (encoder), ``rerank`` (reranker), ``tournament`` with ``rubric`` (judge), then
    ``calibrate`` with ``evaluate`` (none).

    Args:
        steps: The run's steps, in the order they run (``retrieve``, ``rerank``, ``tournament``, ``rubric``,
            ``calibrate``, ``evaluate``).
        serve: The engines the run starts, by role. A role with no engine is never in a phase, whatever
            ``uses`` says: the step reaches it over the URLs its config already holds.
        uses: Per step, the engine roles the step calls; a step absent from the mapping uses no engine. A role
            a step uses but ``serve`` does not start is ignored for that step.

    Returns:
        One :class:`Phase` per group of consecutive steps sharing their engines, in order, each with the roles
        of the engines it starts and the steps it runs.

    Raises:
        NotImplementedError: The behaviour is the serve-phases work's; this is the frozen signature.
    """
    raise NotImplementedError("lane L4a")


__all__ = [
    "ENGINES_ENV",
    "EngineConfig",
    "EngineRole",
    "EngineURLs",
    "Phase",
    "ServeByRole",
    "ServeConfig",
    "parse_engines_env",
    "plan_phases",
]
