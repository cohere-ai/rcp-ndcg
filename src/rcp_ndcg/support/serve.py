"""The engines a run starts beside its job, by role: :class:`ServeByRole`, the phase plan, and the
``RCP_NDCG_ENGINES`` runtime overlay that carries the engines' URLs to the steps.

A leaf model, like :class:`~rcp_ndcg.support.resources.Resources`: a job's phases declare their engines
(``JobPhase.engines``, built from the run config's ``serve:`` by :func:`plan_phases`) and the job runners render
them, without the run layer importing the runners.

rcp-ndcg's contract with a model is one OpenAI-compatible URL. An engine does not change that: the package never
builds, translates or reads an engine's flags. It starts the user's image with the user's command, waits until
``GET <readiness_path>`` answers, and hands the replicas' URLs to the run's step through the environment
(:data:`ENGINES_ENV`), which applies them as a runtime overlay on the role's config.

A run that serves engines runs in **phases** (:func:`plan_phases`): the job starts each phase's engines, waits for
readiness, runs the phase's steps, and stops its engines, so the job's GPUs are the maximum over phases instead of
the sum over engines.

A job that owns its engines never outlives them: an engine that dies, or does not answer within
``startup_timeout_s``, ends the job with a non-zero exit, and the step that calls such an engine stops waiting for
one that stopped answering after the phase's ``wait_on_outage_s`` (the engine config's ``outage_timeout_s``).
Resuming is cheap (the stores are asked only for the windows they lack): ``rcp-ndcg run resume --run <dir>
--runner slurm|kubernetes`` submits the run again, engines included.
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
from rcp_ndcg.support.urls import safe_url

#: The environment variable a phase's runner sets to the engines of the current phase, as JSON
#: ``{"encoder": {"urls": [...], "wait_on_outage_s": 900}, ...}``; the coordinator
#: applies it as a runtime overlay. It is never written into ``run.yaml`` and never reaches an identity, since
#: ``base_url`` and ``wait_on_outage_s`` are runtime fields. Its command-line spelling is
#: ``rcp-ndcg run resume --engine role=url[,url]``.
ENGINES_ENV = "RCP_NDCG_ENGINES"


class ServeConfig(BaseModel):
    """The engine replicas a run's job starts for one role's config (the judge, the retrieval encoder, the
    reranker).

    Attributes:
        image: The engine's container image, e.g. ``vllm/vllm-openai:<tag>`` or ``lmsysorg/sglang:<tag>``; pin the
            tag. Kubernetes and the SLURM runner's container runtimes need it; with the SLURM runner's
            ``container_runtime: none`` the command runs on the node, and an image is refused (it would be ignored).
        command: The command that starts one replica, verbatim: an argv list, or one shell-quoted string. It must
            serve the role config's ``model`` name on ``port`` on all interfaces (``--host 0.0.0.0``) when there
            are several replicas.
        env: Environment of the engine (e.g. ``HF_HOME``).
        resources: What one replica needs (``gpus``, ``cpus``, ``memory_gb``).
        replicas: Independent engine replicas, one URL each; a client that reaches several sends each request to
            the live one with the fewest requests in flight (the judge; a retrieval role takes one replica).
        port: The port each replica serves on.
        readiness_path: The path that answers once a replica serves (``GET``, any 2xx).
        nodes_per_replica: Nodes one replica spans; only 1 is implemented (another value fails validation).
        startup_timeout_s: Seconds a job waits for a replica to answer ``readiness_path`` before it fails; an engine
            that exits before it answers fails the job at once. 0 fails unless the first probe answers.
        outage_timeout_s: Seconds the step that calls this engine waits while every replica is down before it
            fails with ``BackendUnavailableError`` (the phase carries it to the role config's
            ``wait_on_outage_s``).
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
        if len(set(urls)) < len(urls):
            # Typed, never a ValueError: pydantic would render the raw input (credentials embedded in a URL
            # included) into the ValidationError's text; the message names the URLs through safe_url.
            raise ConfigError(
                f"urls lists a replica twice: {[safe_url(url) for url in urls]}", hint="list each replica URL once"
            )
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
        of the engines it starts and the steps it runs. A run with no steps plans no phase.
    """
    planned: list[Phase] = []
    for step in steps:
        roles = uses.get(step, frozenset[EngineRole]())
        kept: list[EngineRole] = [role for role in roles if getattr(serve, role, None) is not None]
        wanted = frozenset(kept)
        if planned and planned[-1].engines == wanted:
            planned[-1] = Phase(engines=wanted, steps=(*planned[-1].steps, step))
        else:
            planned.append(Phase(engines=wanted, steps=(step,)))
    return planned


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
