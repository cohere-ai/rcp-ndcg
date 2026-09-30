"""The judge's engine, started beside a run's job: the one home of :class:`ServeConfig`.

A leaf model, like :class:`~rcp_ndcg.support.resources.Resources`: the run config declares it (``serve:``) and the
job runners render it, without the run layer importing the runners.

rcp-ndcg's contract with a model is one OpenAI-compatible URL. ``serve:`` does not change that: the package never
builds, translates or reads an engine's flags. It starts the user's image with the user's command, waits until
``GET <readiness_path>`` answers, and hands the replicas' URLs to the run's judge as its ``base_url`` list.

A job that owns its engine never outlives it: an engine that dies, or does not answer within
``startup_timeout_s``, ends the job with a non-zero exit, and the judge of such a job stops waiting for an engine
that stopped answering after ``outage_timeout_s``. Retrying is the scheduler's business, and resuming is cheap:
``rcp-ndcg run resume`` asks only the windows its stores lack.
"""

from __future__ import annotations

import shlex
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rcp_ndcg.support.resources import Resources

#: The environment variable a rendered job sets to the replica URLs (comma-separated); ``rcp-ndcg run resume``
#: reads it as its ``--judge-urls``.
JUDGE_URLS_ENV = "RCP_NDCG_JUDGE_URLS"


class ServeConfig(BaseModel):
    """The engine replicas a run's job starts for its judge.

    Attributes:
        image: The engine's container image, e.g. ``vllm/vllm-openai:<tag>`` or ``lmsysorg/sglang:<tag>``; pin the
            tag. With the SLURM runner's ``container_runtime: none``, the command runs on the node instead.
        command: The command that starts one replica, verbatim: an argv list, or one shell-quoted string. It must
            serve the judge's ``model`` name on ``port`` on all interfaces (``--host 0.0.0.0``) when there are
            several replicas.
        env: Environment of the engine (e.g. ``HF_HOME``).
        resources: What one replica needs (``gpus``, ``cpus``, ``memory_gb``).
        replicas: Independent engine replicas, one URL each; the judge sends each request to the one with the
            fewest requests in flight.
        port: The port each replica serves on.
        readiness_path: The path that answers once a replica serves (``GET``, any 2xx).
        nodes_per_replica: Nodes one replica spans. Reserved: only 1 is implemented.
        startup_timeout_s: Seconds a job waits for a replica to answer ``readiness_path`` before it fails; an engine
            that exits before it answers fails the job at once. 0 fails unless the first probe answers.
        outage_timeout_s: Seconds the judge of the job waits while every replica is down before it fails with
            ``BackendUnavailableError`` (it becomes the judge's ``wait_on_outage_s`` in the job).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    image: str = Field(min_length=1)
    command: tuple[str, ...] = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    resources: Resources = Resources()
    replicas: int = Field(default=1, ge=1)
    port: int = Field(default=8000, ge=1, le=65535)
    readiness_path: str = Field(default="/v1/models", pattern=r"^/[A-Za-z0-9/_.~%=&?-]*$")
    nodes_per_replica: int = Field(default=1, ge=1)
    startup_timeout_s: int = Field(default=1800, ge=0)
    outage_timeout_s: int = Field(default=900, ge=0)

    @field_validator("command", mode="before")
    @classmethod
    def _a_string_is_split(cls, value: Any) -> Any:
        return tuple(shlex.split(value)) if isinstance(value, str) else value

    def check_nodes(self) -> None:
        """Where a replica spanning several nodes would be rendered: not implemented.

        Raises:
            NotImplementedError: ``nodes_per_replica`` is above 1.
        """
        if self.nodes_per_replica > 1:
            raise NotImplementedError(
                f"serve.nodes_per_replica={self.nodes_per_replica}: a replica spanning several nodes is not "
                "implemented; run one replica per node (nodes_per_replica: 1) and scale out with serve.replicas"
            )

    def url(self, host: str) -> str:
        """The OpenAI-compatible base URL of the replica on ``host``."""
        return f"http://{host}:{self.port}/v1"


__all__ = ["JUDGE_URLS_ENV", "ServeConfig"]
