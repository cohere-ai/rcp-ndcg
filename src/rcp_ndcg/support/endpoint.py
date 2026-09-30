"""An endpoint a model is served from: where, which model, and how the client talks to it.

The one home of the fields every served model shares: the judge (:class:`~rcp_ndcg.llm.JudgeConfig`) and the
hosted retrieval providers build on :class:`Endpoint` instead of repeating them.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rcp_ndcg.support.identity import FieldRole


class Endpoint(BaseModel):
    """A served model and how to reach it.

    Attributes:
        base_url: The endpoint, e.g. ``http://localhost:8000/v1``; ``None`` for a provider's own public API.
        model: The served model name, sent as the request's ``model``.
        revision: The checkpoint commit the served weights resolved to; recorded in identities, so two
            checkpoints served under one name are never mistaken for each other.
        api_key_env: Environment variable holding the API key; ``None`` sends no key.
        concurrency: Requests in flight at once.
        timeout_s: Per-request timeout, seconds.
        connect_timeout_s: TCP connect timeout, seconds.
        max_retries: Retries of a transient failure before it counts as an outage.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The model and its checkpoint decide what is computed; where and how fast it is asked do not.
    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "model": FieldRole.CONTENT,
        "revision": FieldRole.CONTENT,
        "base_url": FieldRole.RUNTIME,
        "api_key_env": FieldRole.RUNTIME,
        "concurrency": FieldRole.RUNTIME,
        "timeout_s": FieldRole.RUNTIME,
        "connect_timeout_s": FieldRole.RUNTIME,
        "max_retries": FieldRole.RUNTIME,
    }

    base_url: str | None = None
    model: str = Field(min_length=1)
    revision: str | None = None
    api_key_env: str | None = None
    concurrency: int = Field(default=64, ge=1)
    timeout_s: float = Field(default=600.0, gt=0)
    connect_timeout_s: float = Field(default=5.0, gt=0)
    max_retries: int = Field(default=2, ge=0)

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str | None) -> str | None:
        return value.rstrip("/") if value is not None else None


__all__ = ["Endpoint"]
