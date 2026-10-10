"""The endpoint: where a model is served, which checkpoint, and how the client talks to it.

The one home of the fields every role shares (the judge, the encoders, the rerankers); a role's config subclasses
:class:`Endpoint` and adds only the fields its wire protocol needs. Where and how fast a model is asked are
runtime fields that never enter an identity; which model, which checkpoint and which wire adapter decide what is
computed, and do.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator
from rcp_ndcg_core.records import DocumentTitle

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.inference.fake import FAKE_SCHEME
from rcp_ndcg.support.identity import FieldRole
from rcp_ndcg.support.urls import safe_url

#: An environment variable name: a letter or underscore, then letters, digits or underscores.
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Endpoint(BaseModel):
    """A served model and how to reach it.

    Attributes:
        api: The wire adapter that speaks the endpoint's protocol
            (:data:`~rcp_ndcg.inference.adapters.base.ADAPTER_ENTRY_POINTS`); ``None`` until a role config sets
            its default. Content: which service computes the numbers.
        base_url: The endpoint, e.g. ``http://localhost:8000/v1``; ``None`` for a provider's own public API.
            One URL, or a list of replica URLs of the same served model (each request goes to the live replica
            with the fewest in flight; the offline fakes are one URL, never a list). A role that reaches
            replicas over several URLs lists them in its own config (the judge: one URL or a replica list).
        model: The served model name, sent as the request's ``model``.
        revision: The checkpoint commit the served weights resolved to; recorded in identities, so two
            checkpoints served under one name are never mistaken for each other.
        api_key_env: Environment variable holding the API key, resolved by the transport; when ``None``
            (the default), the wire adapter profile's own variables are tried in order, in the profile's
            header (a hosted profile requires one). A variable the config names must be set. An empty name
            is refused (it would silently send no header).
        headers_env: Header name -> environment variable name (e.g. ``{"X-Gateway-Key": "GATEWAY_KEY"}``); the
            values are read from the environment only, never from a config, and the variable names are validated.
        concurrency: Requests in flight at once.
        timeout_s: Per-request timeout, seconds.
        connect_timeout_s: TCP connect timeout, seconds.
        max_retries: Retries of a transient failure before it counts as an outage.
        wait_on_outage_s: How long a request waits while every replica is down before
            :class:`~rcp_ndcg.errors.BackendUnavailableError`, counted from the request's first unavailable
            failure (time spent queued behind the concurrency limit never counts); the default 1800 s covers
            an engine restart plus a large model's load, and ``None`` waits indefinitely.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The model, its checkpoint and the wire adapter decide what is computed; where and how fast it is asked
    #: do not.
    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "api": FieldRole.CONTENT,
        "model": FieldRole.CONTENT,
        "revision": FieldRole.CONTENT,
        "title": FieldRole.CONTENT,
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
    base_url: str | list[str] | None = None
    model: str = Field(min_length=1)
    revision: str | None = None
    title: DocumentTitle | None = None
    """How a document's title reaches the model: ``join`` (the default, MTEB's dataloader rule:
    ``(title + " " + body).strip()``, the body alone without a title), or ``separate`` (the title as its own
    leading text part) for a model or recipe that takes it that way. Content: the model reads a different
    string. ``None`` (the default) is the same as ``join`` and declares nothing, so a config that does not
    choose keeps the identity and the behaviour fingerprint it had."""
    api_key_env: str | None = Field(default=None, min_length=1)
    headers_env: dict[str, str] = Field(default_factory=dict)
    concurrency: int = Field(default=64, ge=1)
    timeout_s: float = Field(default=600.0, gt=0)
    connect_timeout_s: float = Field(default=5.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    wait_on_outage_s: float | None = Field(default=1800.0, ge=0)

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str | list[str] | None) -> str | list[str] | None:
        """Strip trailing slashes; a replica list is non-empty, lists no replica twice, and never mixes the
        offline fakes with real replicas (a fake is one in-process endpoint, a list is a deployed gateway)."""
        if value is None:
            return None
        if isinstance(value, str):
            if not value:
                raise ValueError("base_url: give one URL or a non-empty list of replica URLs")
            return value.rstrip("/")
        if not value:
            raise ValueError("base_url: give one URL or a non-empty list of replica URLs")
        urls = [url.rstrip("/") for url in value]
        # Typed refusals, never a ValueError: pydantic would render the raw input (credentials embedded in a
        # URL included) into the ValidationError's text; the message names the URLs through safe_url.
        if len(set(urls)) < len(urls):
            raise ConfigError(
                f"base_url lists a replica twice: {[safe_url(url) for url in urls]}",
                hint="list each replica URL once",
            )
        if len(urls) > 1 and any(url.startswith(FAKE_SCHEME) for url in urls):
            raise ConfigError(
                f"the offline fakes ({FAKE_SCHEME}) are one URL, not a replica list: {[safe_url(url) for url in urls]}",
                hint=f"give the {FAKE_SCHEME} URL alone, or only real replica URLs",
            )
        return urls

    @property
    def urls(self) -> tuple[str, ...]:
        """The replica URLs: ``base_url`` as a tuple (one element for a single URL or a gateway); empty when the
        endpoint names none (a hosted profile's own URL, resolved by its wire adapter)."""
        if self.base_url is None:
            return ()
        return (self.base_url,) if isinstance(self.base_url, str) else tuple(self.base_url)

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

    def identity_extra(self) -> dict[str, Any]:
        """The identity fields beyond :func:`rcp_ndcg.support.identity.identity_payload`; default: none.

        A role config that declares a ``tokenizer`` field -- the judge's, and the embedding, pooling and
        rerank configs -- carries the tokenizer's content identity here: ``{"tokenizer_sha256": <sha>}``,
        the SHA-256 of its ``tokenizer.json``, through the one helper
        :func:`rcp_ndcg.data.tokenizer.tokenizer_identity`. What cuts (or budgets) the text enters every
        identity by that digest, never by how it is named -- the name is RUNTIME and is only recorded beside
        the identity as a source. A step identity (``runs/pipeline.py``) merges this into the config's
        ``identity_payload``; the judge's own identity payload keeps its existing keys and is unchanged by
        this method.

        Returns:
            ``{"tokenizer_sha256": <sha>}`` when the config declares and names a tokenizer, else ``{}``.

        Raises:
            DependencyError: ``tokenizers`` (or, for a Hub id, ``huggingface_hub``) is not installed.
            MissingInputError: The local file, or the repository's ``tokenizer.json`` at the named revision,
                does not exist; an unknown repository raises the Hub client's own error.
        """
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return {}
        from rcp_ndcg.data.tokenizer import tokenizer_identity

        return tokenizer_identity(tokenizer)


__all__ = ["Endpoint"]
