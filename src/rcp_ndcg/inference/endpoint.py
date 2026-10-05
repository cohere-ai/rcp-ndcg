"""The endpoint: where a model is served, which checkpoint, and how the client talks to it.

The one home of the fields every role shares (the judge, the encoders, the rerankers); a role's config subclasses
:class:`Endpoint` and adds only the fields its wire protocol needs. Where and how fast a model is asked are
runtime fields that never enter an identity; which model, which checkpoint and which wire adapter decide what is
computed, and do.
"""

from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rcp_ndcg.support.identity import FieldRole

#: An environment variable name: a letter or underscore, then letters, digits or underscores.
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Endpoint(BaseModel):
    """A served model and how to reach it.

    Attributes:
        api: The wire adapter that speaks the endpoint's protocol
            (:data:`~rcp_ndcg.inference.adapters.base.ADAPTER_ENTRY_POINTS`); ``None`` until a role config sets
            its default. Content: which service computes the numbers.
        base_url: The endpoint, e.g. ``http://localhost:8000/v1``; ``None`` for a provider's own public API.
            A role that reaches replicas over several URLs lists them in its own config (the judge: one URL or a
            replica list).
        model: The served model name, sent as the request's ``model``.
        revision: The checkpoint commit the served weights resolved to; recorded in identities, so two
            checkpoints served under one name are never mistaken for each other.
        api_key_env: Environment variable holding the API key; ``None`` sends no key.
        headers_env: Header name -> environment variable name (e.g. ``{"X-Gateway-Key": "GATEWAY_KEY"}``); the
            values are read from the environment only, never from a config, and the variable names are validated.
        concurrency: Requests in flight at once.
        timeout_s: Per-request timeout, seconds.
        connect_timeout_s: TCP connect timeout, seconds.
        max_retries: Retries of a transient failure before it counts as an outage.
        wait_on_outage_s: How long a request waits while every replica is down before
            :class:`~rcp_ndcg.errors.BackendUnavailableError`, counted from the request's first unavailable
            failure (time spent queued behind the concurrency limit never counts); ``None`` waits indefinitely.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The model, its checkpoint and the wire adapter decide what is computed; where and how fast it is asked
    #: do not.
    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "api": FieldRole.CONTENT,
        "model": FieldRole.CONTENT,
        "revision": FieldRole.CONTENT,
        "base_url": FieldRole.RUNTIME,
        "api_key_env": FieldRole.RUNTIME,
        "headers_env": FieldRole.RUNTIME,
        "concurrency": FieldRole.RUNTIME,
        "timeout_s": FieldRole.RUNTIME,
        "connect_timeout_s": FieldRole.RUNTIME,
        "max_retries": FieldRole.RUNTIME,
        "wait_on_outage_s": FieldRole.RUNTIME,
    }

    api: str | None = None
    base_url: str | None = None
    model: str = Field(min_length=1)
    revision: str | None = None
    api_key_env: str | None = None
    headers_env: dict[str, str] = Field(default_factory=dict)
    concurrency: int = Field(default=64, ge=1)
    timeout_s: float = Field(default=600.0, gt=0)
    connect_timeout_s: float = Field(default=5.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    wait_on_outage_s: float | None = Field(default=None, ge=0)

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str | None) -> str | None:
        return value.rstrip("/") if value is not None else None

    @field_validator("headers_env")
    @classmethod
    def _env_names_are_names(cls, value: dict[str, str]) -> dict[str, str]:
        """Refuse an environment variable name that cannot name a variable: the values are read from the
        environment only, so a mistyped name would silently send no header."""
        for header, variable in value.items():
            if _ENV_NAME.fullmatch(variable) is None:
                raise ValueError(
                    f"headers_env[{header!r}]: {variable!r} is not an environment variable name "
                    "(a letter or underscore, then letters, digits or underscores)"
                )
        return value


__all__ = ["Endpoint"]
