"""The CLI describing itself: every command, flag, type, default and output schema, as data.

Two views of one tree:

* :func:`command_index` -- the compact view ``rcp-ndcg schema show commands`` prints and the MCP ``describe`` tool
  returns: each leaf command with its help, its flags (type, choices, default, whether required or repeatable,
  help) and its output schema (schema ``rcp-ndcg.command-index.v1``).
* :func:`describe_commands` -- the full click tree (``schema show commands --full``, schema
  ``rcp-ndcg.commands.v1``), which the public-surface snapshot test pins. Shape only there: no help prose unless
  asked for, so rewording a help text never changes the snapshot.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

import click
from pydantic import BaseModel, Field

from rcp_ndcg.cli.command import CommandSpec


class ParamInfo(BaseModel):
    """One argument or option of a command."""

    param_type_name: Literal["argument", "option"]
    opts: list[str]
    secondary_opts: list[str]
    type: dict[str, Any] = Field(description="'name', plus 'choices' for a choice.")
    required: bool
    multiple: bool
    nargs: int
    default: Any
    envvar: str | list[str] | None
    hidden: bool
    is_flag: bool | None = None
    help: str | None = None


class CommandInfo(BaseModel):
    """One command or group of the tree."""

    group: bool
    hidden: bool
    deprecated: bool
    no_args_is_help: bool
    params: dict[str, ParamInfo]
    output_schema: str | None = Field(description="The schema id of the command's --json data, if declared.")
    help: str | None = None


class CommandTree(BaseModel):
    """The whole CLI, keyed by command path (``"rcp-ndcg data inspect"``)."""

    commands: dict[str, CommandInfo]


def _default(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, tuple | list):
        return [_default(item) for item in value]
    if type(value).__name__ == "Sentinel":  # click's "no default" marker
        return None
    return repr(value)


def _param(param: click.Parameter, *, include_help: bool = False) -> ParamInfo:
    type_info: dict[str, Any] = {"name": param.type.name}
    if isinstance(param.type, click.Choice):
        type_info["choices"] = [str(choice) for choice in param.type.choices]
    return ParamInfo(
        param_type_name="option" if isinstance(param, click.Option) else "argument",
        opts=list(param.opts),
        secondary_opts=list(param.secondary_opts),
        type=type_info,
        required=param.required,
        multiple=param.multiple,
        nargs=param.nargs,
        default=_default(param.default) if not callable(param.default) else "<callable>",
        envvar=param.envvar if param.envvar is None or isinstance(param.envvar, str) else list(param.envvar),
        hidden=bool(getattr(param, "hidden", False)),
        is_flag=param.is_flag if isinstance(param, click.Option) else None,
        help=getattr(param, "help", None) if include_help else None,
    )


def walk(root: click.Command | None = None) -> Iterator[tuple[str, click.Command]]:
    """Every command of the tree with its path, loading lazily registered groups on the way.

    Raises:
        RuntimeError: A group lists a command it cannot load.
    """
    if root is None:
        from rcp_ndcg.cli.main import cli as root

    def visit(cmd: click.Command, path: str, ctx: click.Context) -> Iterator[tuple[str, click.Command]]:
        yield path, cmd
        if isinstance(cmd, click.Group):
            for name in cmd.list_commands(ctx):
                sub = cmd.get_command(ctx, name)
                if sub is None:
                    raise RuntimeError(f"{path} {name}: listed but cannot be loaded")
                yield from visit(sub, f"{path} {name}", click.Context(sub, info_name=name, parent=ctx))

    yield from visit(root, "rcp-ndcg", click.Context(root, info_name="rcp-ndcg"))


def describe_commands(*, include_help: bool) -> CommandTree:
    """The CLI tree as a :class:`CommandTree` (schema ``rcp-ndcg.commands.v1``).

    Args:
        include_help: Add each command's short help and each flag's help (left out of the snapshot, which pins
            shape only).
    """
    commands = {}
    for path, cmd in walk():
        commands[path] = CommandInfo(
            group=isinstance(cmd, click.Group),
            hidden=bool(cmd.hidden),
            deprecated=bool(cmd.deprecated),
            no_args_is_help=bool(cmd.no_args_is_help),
            params={param.name: _param(param, include_help=include_help) for param in cmd.params if param.name},
            output_schema=getattr(cmd, "output_schema", None),
            help=cmd.get_short_help_str(limit=200) if include_help else None,
        )
    return CommandTree(commands=commands)


class FlagInfo(BaseModel):
    """One flag (or positional argument) of a command, as an agent needs it."""

    flag: str = Field(description="How to pass it: '--name', '--name/--no-name' for a switch, or NAME (positional).")
    type: str = Field(description="text, integer, float, boolean, path or choice.")
    choices: list[str] | None = None
    required: bool
    repeatable: bool
    default: Any = None
    help: str | None = None


class CommandEntry(BaseModel):
    """One command, as an agent needs it."""

    help: str | None
    output_schema: str | None = Field(description="The schema id of the command's --json data.")
    flags: list[FlagInfo]


class CommandIndex(BaseModel):
    """Every leaf command, keyed by path (``"data inspect"``): its help, flags and output schema."""

    commands: dict[str, CommandEntry]


def _flag(param: click.Parameter) -> FlagInfo:
    if isinstance(param, click.Option):
        flag = "/".join([*param.opts, *param.secondary_opts]) if param.secondary_opts else param.opts[0]
    else:
        flag = (param.name or "").upper()
    info = _param(param, include_help=True)
    return FlagInfo(
        flag=flag,
        type="boolean" if info.is_flag else str(info.type["name"]),
        choices=info.type.get("choices"),
        required=info.required,
        repeatable=info.multiple or info.nargs == -1,
        default=info.default,
        help=info.help,
    )


def command_index() -> CommandIndex:
    """The leaf commands with their help, flags and output schemas (schema ``rcp-ndcg.command-index.v1``).

    Hidden flags and ``--help`` are left out; the full click tree is :func:`describe_commands`.
    """
    commands = {}
    for path, cmd in walk():
        if isinstance(cmd, click.Group) or cmd.hidden:
            continue
        flags = [
            _flag(param)
            for param in cmd.params
            if param.name and param.name != "help" and not getattr(param, "hidden", False)
        ]
        commands[path.removeprefix("rcp-ndcg ")] = CommandEntry(
            help=cmd.get_short_help_str(limit=200) or None,
            output_schema=getattr(cmd, "output_schema", None),
            flags=flags,
        )
    return CommandIndex(commands=commands)


def command_specs() -> dict[str, CommandSpec]:
    """Every declared command (:func:`rcp_ndcg.cli.command.command`) by path, all command modules loaded."""
    specs = {}
    for path, cmd in walk():
        spec = getattr(cmd, "spec", None)
        if isinstance(spec, CommandSpec):
            specs[path.removeprefix("rcp-ndcg ")] = spec
    return specs


__all__ = [
    "CommandEntry",
    "CommandIndex",
    "CommandInfo",
    "CommandTree",
    "FlagInfo",
    "ParamInfo",
    "command_index",
    "command_specs",
    "describe_commands",
    "walk",
]
