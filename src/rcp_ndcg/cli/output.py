"""The machine-output contract: the ``rcp-ndcg.cli.v1`` envelope that every ``--json`` command prints.

With ``--json``, stdout carries exactly one JSON document on every exit path, success or failure, usage errors
included; logs and progress go to stderr. Success::

    {"schema": "rcp-ndcg.cli.v1", "command": "data inspect", "ok": true,
     "data": {"schema": "rcp-ndcg.dataset-summary.v1", ...},
     "warnings": [{"code": "UNREADABLE_RUN", "message": "..."}],
     "meta": {"version": "0.1.0", "elapsed_s": 0.42, "run_dir": null}}

Failure: ``ok`` is false, ``data`` is absent and ``error`` is :meth:`rcp_ndcg.errors.RcpNdcgError.to_dict`.
Without ``--json`` the same result is rendered as text on stdout and failures as ``error [CODE]: ...`` on
stderr.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Literal, NoReturn

import click
from pydantic import BaseModel, ConfigDict, Field

from rcp_ndcg.errors import ExitCode, RcpNdcgError, WarningCode

CLI_SCHEMA = "rcp-ndcg.cli.v1"

#: The ``error.code`` of a failure: the name of every exit code but success (:class:`~rcp_ndcg.errors.ExitCode`).
ErrorCode = Literal[tuple(code.name for code in ExitCode if code is not ExitCode.SUCCESS)]  # type: ignore[valid-type]


class CliWarning(BaseModel):
    """A typed warning: ``code`` is from the closed list :data:`rcp_ndcg.errors.WarningCode`."""

    code: WarningCode
    message: str


class CliError(BaseModel):
    """Why a command failed and what to do next (the exit-code table of :mod:`rcp_ndcg.errors`)."""

    code: ErrorCode
    exit_code: int = Field(ge=1, le=max(ExitCode))
    message: str
    hint: str | None = None
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class CliMeta(BaseModel):
    """Facts about the invocation: package version, wall time in seconds, and the run directory the result names
    (its ``run_dir``), if any."""

    version: str
    elapsed_s: float
    run_dir: str | None = None


class CliEnvelope(BaseModel):
    """The one JSON document a ``--json`` command prints on stdout."""

    model_config = ConfigDict(populate_by_name=True)

    schema_id: Literal["rcp-ndcg.cli.v1"] = Field(default=CLI_SCHEMA, alias="schema")
    command: str = Field(description="The command path without the program name, e.g. 'data inspect'.")
    ok: bool
    data: Any = Field(default=None, description="The result, present when ok; its 'schema' names its JSON Schema.")
    error: CliError | None = Field(default=None, description="Present when not ok.")
    warnings: list[CliWarning] = Field(default_factory=list)
    meta: CliMeta


def _version() -> str:
    from rcp_ndcg import __version__

    return __version__


def success_document(
    command: str, data: Any, *, warnings: list[dict[str, str]], elapsed_s: float, run_dir: str | None = None
) -> dict[str, Any]:
    """The success envelope as a JSON-ready dict."""
    meta = CliMeta(version=_version(), elapsed_s=round(elapsed_s, 3), run_dir=run_dir)
    envelope = CliEnvelope(command=command, ok=True, data=data, warnings=warnings, meta=meta)  # type: ignore[arg-type]
    return envelope.model_dump(mode="json", by_alias=True, exclude={"error"})


def error_document(
    command: str, error: RcpNdcgError, *, warnings: list[dict[str, str]] | None = None, elapsed_s: float = 0.0
) -> dict[str, Any]:
    """The failure envelope as a JSON-ready dict."""
    meta = CliMeta(version=_version(), elapsed_s=round(elapsed_s, 3))
    failure = CliError.model_validate(error.to_dict())
    envelope = CliEnvelope(command=command, ok=False, error=failure, warnings=warnings or [], meta=meta)  # type: ignore[arg-type]
    return envelope.model_dump(mode="json", by_alias=True, exclude={"data"})


def print_json(document: Any) -> None:
    """Write *document* to stdout as one JSON document."""
    click.echo(json.dumps(document, indent=2, default=str))


def print_error(error: RcpNdcgError) -> None:
    """The human rendering of a failure, on stderr."""
    click.echo(f"error [{error.code}]: {error.message}", err=True)
    if error.hint:
        click.echo(f"hint: {error.hint}", err=True)


def print_warnings(warnings: list[dict[str, str]]) -> None:
    """The human rendering of typed warnings, on stderr."""
    for warning in warnings:
        click.echo(f"warning [{warning['code']}]: {warning['message']}", err=True)


def fail(command: str, error: RcpNdcgError, *, as_json: bool, elapsed_s: float = 0.0) -> NoReturn:
    """Report *error* in the agreed shape and exit with its code."""
    if as_json:
        print_json(error_document(command, error, elapsed_s=elapsed_s))
    else:
        print_error(error)
    sys.exit(int(error.exit_code))


__all__ = [
    "CLI_SCHEMA",
    "CliEnvelope",
    "CliError",
    "CliMeta",
    "CliWarning",
    "ErrorCode",
    "error_document",
    "fail",
    "print_error",
    "print_json",
    "print_warnings",
    "success_document",
]
