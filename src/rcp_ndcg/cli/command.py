"""One decorator turns a library call into a CLI command, and the same declaration serves MCP.

A command is declared once::

    @command("run list", request=RunListRequest, result=RunList)
    def run_list(request: RunListRequest) -> RunList:
        '''List the runs under a runs directory, newest first.'''
        return RunList(...)

The decorator builds the click options from the request model's fields (a flag is a field, so the MCP input
schema is the same model), adds ``--json``, calls the function, renders the result as text or as the
``rcp-ndcg.cli.v1`` envelope (:mod:`rcp_ndcg.cli.output`), and maps every exception through
:func:`rcp_ndcg.errors.classify` onto a typed error and its exit code. The declaration is kept as a
:class:`CommandSpec` on the click command (``.spec``); :func:`execute` runs it for the CLI and for the MCP
server alike, and :func:`rcp_ndcg.cli.introspect.command_specs` collects them all.
"""

from __future__ import annotations

import re
import sys
import time
import types
import typing
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import click
from pydantic import BaseModel, ValidationError

from rcp_ndcg.cli.output import (
    error_document,
    print_error,
    print_json,
    print_warnings,
    success_document,
)
from rcp_ndcg.errors import RcpNdcgError, RcpNdcgWarning, UsageError, classify
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CommandSpec:
    """A declared command: what it takes, what it returns and which library call does the work.

    Attributes:
        name: The command path without the program name (``"data inspect"``).
        request: The pydantic model of its inputs; its fields are the flags and the MCP input schema.
        result: The pydantic model of its output, or ``None`` for a free-form JSON document.
        handler: ``handler(request) -> result``.
        output_schema: The id of the JSON Schema that ``data`` conforms to (``"rcp-ndcg.run-list.v1"``).
        read_only: Whether the call leaves no artifacts behind.
        spends: Whether the call can spend money (judge calls); such tools need ``--allow-spend`` over MCP.
        summary: The one-line description (help text, MCP tool description).
    """

    name: str
    request: type[BaseModel]
    result: type[BaseModel] | None
    handler: Callable[[Any], Any]
    output_schema: str
    read_only: bool = True
    spends: bool = False
    summary: str = ""


@dataclass
class Outcome:
    """What one execution produced: the result as JSON-ready ``data``, or the typed ``error``."""

    command: str
    result: Any = None
    data: Any = None
    error: RcpNdcgError | None = None
    warnings: list[dict[str, str]] = field(default_factory=list)
    elapsed_s: float = 0.0

    def document(self) -> dict[str, Any]:
        """The ``rcp-ndcg.cli.v1`` envelope of this outcome."""
        if self.error is not None:
            return error_document(self.command, self.error, warnings=self.warnings, elapsed_s=self.elapsed_s)
        run_dir = self.data.get("run_dir") if isinstance(self.data, dict) else None
        return success_document(
            self.command,
            self.data,
            warnings=self.warnings,
            elapsed_s=self.elapsed_s,
            run_dir=str(run_dir) if run_dir is not None else None,
        )


def schema_name(model: type[BaseModel]) -> str:
    """The schema name of a result model: its class name in kebab case (``RunList`` -> ``run-list``)."""
    return re.sub(r"(?<!^)(?=[A-Z])", "-", model.__name__).lower()


def _validation_error(exc: ValidationError) -> UsageError:
    problems = [
        {"field": ".".join(str(part) for part in err["loc"]) or "<input>", "message": err["msg"]}
        for err in exc.errors(include_url=False)
    ]
    listed = "; ".join(f"{problem['field']}: {problem['message']}" for problem in problems)
    return UsageError(
        f"invalid arguments: {listed}", hint="see --help for the expected values", details={"errors": problems}
    )


def _data(spec: CommandSpec, result: Any) -> Any:
    """The result as JSON-ready data, tagged with its schema id (a model that carries its own ``schema`` keeps it)."""
    if isinstance(result, BaseModel):
        payload = result.model_dump(mode="json", by_alias=True)
        return {"schema": spec.output_schema, **payload} if spec.result is not None else payload
    return result


def execute(spec: CommandSpec, arguments: Mapping[str, Any]) -> Outcome:
    """Validate *arguments* against the request model, run the handler and capture its typed warnings.

    Every exception becomes a typed error on the outcome (malformed arguments are :class:`UsageError`); only
    ``KeyboardInterrupt`` and ``SystemExit`` propagate.

    Args:
        spec: The command.
        arguments: Field values of ``spec.request``.

    Returns:
        The :class:`Outcome`, with ``elapsed_s`` in seconds.
    """
    started = time.perf_counter()
    outcome = Outcome(command=spec.name)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            try:
                request = spec.request.model_validate(dict(arguments))
            except ValidationError as exc:
                raise _validation_error(exc) from exc
            outcome.result = spec.handler(request)
            outcome.data = _data(spec, outcome.result)
        except Exception as exc:  # noqa: BLE001 -- every failure is reported as a typed error
            outcome.error = classify(exc).for_cli()
            logger.debug(f"{spec.name} failed: {type(exc).__name__}: {exc}", exc_info=exc)
    for record in seen:
        if isinstance(record.message, RcpNdcgWarning):
            outcome.warnings.append(record.message.to_dict())
        else:  # not ours: hand it back to the normal warning filters
            warnings.warn_explicit(record.message, record.category, record.filename, record.lineno)
    outcome.elapsed_s = time.perf_counter() - started
    return outcome


# ----------------------------------------------------------------------------------------------------------------
# click options from a request model
# ----------------------------------------------------------------------------------------------------------------


def _unwrap(annotation: Any) -> tuple[Any, bool]:
    """``(inner type, multiple)`` of a field annotation, with ``None`` removed from unions."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        members = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(members) == 1:
            return _unwrap(members[0])
        raise TypeError(f"cannot build a CLI option for the union {annotation!r}")
    if origin in (list, tuple, Sequence, set, frozenset):
        args = [arg for arg in typing.get_args(annotation) if arg is not Ellipsis]
        return (args[0] if args else str), True
    return annotation, False


def _click_type(inner: Any) -> click.ParamType:
    if typing.get_origin(inner) is Literal:
        return click.Choice([str(value) for value in typing.get_args(inner)])
    if isinstance(inner, type) and issubclass(inner, Enum):
        return click.Choice([str(member.value) for member in inner])
    if inner is bool:
        return click.BOOL
    if inner is int:
        return click.INT
    if inner is float:
        return click.FLOAT
    if inner is Path:
        return click.Path(path_type=str)
    if inner is str:
        return click.STRING
    raise TypeError(f"cannot build a CLI option for the type {inner!r}; give the field a scalar type")


def _params(
    request: type[BaseModel], *, positional: Sequence[str], envvars: Mapping[str, str], hidden: Sequence[str]
) -> list[click.Parameter]:
    params: list[click.Parameter] = []
    for name, info in request.model_fields.items():
        inner, multiple = _unwrap(info.annotation)
        click_type = _click_type(inner)
        required = info.is_required()
        default = None if required else info.get_default(call_default_factory=True)
        if isinstance(default, Enum):
            default = default.value
        if multiple:
            default = tuple(default or ())
        if name in positional:
            params.append(click.Argument([name], type=click_type, required=required, nargs=-1 if multiple else 1))
            continue
        flag = "--" + name.replace("_", "-")
        common: dict[str, Any] = {
            "help": info.description,
            "hidden": name in hidden,
            "envvar": envvars.get(name),
            "show_envvar": name in envvars,
        }
        if inner is bool and not multiple:
            decls = [f"{flag}/--no-{flag[2:]}" if default else flag, name]
            params.append(click.Option(decls, is_flag=True, default=bool(default), show_default=True, **common))
            continue
        option: dict[str, Any] = {"type": click_type, "required": required, "multiple": multiple, **common}
        if not required:  # click treats an explicit default (even None) as a value, which defeats `required`
            option.update(default=default, show_default=default not in (None, ()))
        params.append(click.Option([flag, name], **option))
    return params


def _summary(handler: Callable[..., Any]) -> str:
    doc = (handler.__doc__ or "").strip()
    return doc.split("\n\n", 1)[0].replace("\n", " ") if doc else ""


def _render_text(data: Any) -> str:
    import yaml

    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120).rstrip()


def _run_cli(spec: CommandSpec, arguments: dict[str, Any], *, as_json: bool, text: Callable[[Any], str] | None) -> None:
    try:
        outcome = execute(spec, arguments)
    except KeyboardInterrupt:
        outcome = Outcome(command=spec.name, error=classify(KeyboardInterrupt()))
    if outcome.error is not None:
        if as_json:
            print_json(outcome.document())
        else:
            print_warnings(outcome.warnings)
            print_error(outcome.error)
        sys.exit(int(outcome.error.exit_code))
    if as_json:
        print_json(outcome.document())
        return
    print_warnings(outcome.warnings)
    click.echo(text(outcome.result) if text is not None else _render_text(outcome.data))


def command(
    name: str,
    *,
    request: type[BaseModel],
    result: type[BaseModel] | None,
    output_schema: str | None = None,
    text: Callable[[Any], str] | None = None,
    positional: Sequence[str] = (),
    envvars: Mapping[str, str] | None = None,
    hidden_fields: Sequence[str] = (),
    read_only: bool = True,
    spends: bool = False,
) -> Callable[[Callable[[Any], Any]], click.Command]:
    """Declare a command over a library call.

    Args:
        name: The command path without the program name; the last word is the click command's name.
        request: The pydantic model of the inputs. Each field becomes ``--field-name`` (``True`` defaults become
            ``--x/--no-x``, lists become repeatable options); fields named in *positional* become arguments.
        result: The pydantic model of the output; ``None`` for a free-form JSON document, which then needs an
            explicit *output_schema*.
        output_schema: The schema id of ``data``; default ``rcp-ndcg.<kebab name of result>.v1``.
        text: Renders the result for a person; default a YAML dump of ``data``.
        positional: Fields taken as positional arguments.
        envvars: Field name -> environment variable read when the flag is absent (shown in ``--help``).
        hidden_fields: Fields whose options are hidden from ``--help`` (debug switches).
        read_only: Whether the call leaves no artifacts behind (MCP ``readOnlyHint``).
        spends: Whether the call can spend money (MCP tools need ``--allow-spend`` and ``budget_usd``).

    Returns:
        A decorator turning ``handler(request) -> result`` into a :class:`click.Command` that carries the
        :class:`CommandSpec` as ``.spec`` and its schema id as ``.output_schema``.
    """
    if output_schema is None:
        if result is None:
            raise ValueError(f"{name}: a command without a result model needs an explicit output_schema")
        output_schema = f"rcp-ndcg.{schema_name(result)}.v1"
    resolved_schema = output_schema

    def decorator(handler: Callable[[Any], Any]) -> click.Command:
        spec = CommandSpec(
            name=name,
            request=request,
            result=result,
            handler=handler,
            output_schema=resolved_schema,
            read_only=read_only,
            spends=spends,
            summary=_summary(handler),
        )
        params = _params(request, positional=positional, envvars=envvars or {}, hidden=hidden_fields)
        params.append(
            click.Option(["--json", "as_json"], is_flag=True, help="Print one rcp-ndcg.cli.v1 JSON document on stdout.")
        )

        def callback(as_json: bool, **values: Any) -> None:
            arguments = {key: list(value) if isinstance(value, tuple) else value for key, value in values.items()}
            _run_cli(spec, arguments, as_json=as_json, text=text)

        cmd = click.Command(
            name=name.split()[-1],
            callback=callback,
            params=params,
            help=handler.__doc__,
            short_help=spec.summary,
        )
        cmd.spec = spec  # type: ignore[attr-defined]
        cmd.output_schema = resolved_schema  # type: ignore[attr-defined]
        return cmd

    return decorator


__all__ = ["CommandSpec", "Outcome", "command", "execute", "schema_name"]
