"""The root ``rcp-ndcg`` command registry."""

from __future__ import annotations

import importlib.util

import click
import pytest

from rcp_ndcg.cli.main import _LAZY_SUBCOMMANDS, cli


@pytest.mark.parametrize("name", sorted(_LAZY_SUBCOMMANDS))
def test_every_registered_subcommand_points_at_an_existing_module(name):
    module, _, _ = _LAZY_SUBCOMMANDS[name]
    assert importlib.util.find_spec(module) is not None, f"`rcp-ndcg {name}` names a missing module {module}"


def _commands(group, prefix=()):
    ctx = click.Context(group)
    for name in group.list_commands(ctx):
        command = group.get_command(ctx, name)
        if command is None:
            continue
        yield (*prefix, name), command
        if isinstance(command, click.Group):
            yield from _commands(command, (*prefix, name))


def test_no_command_declares_an_option_twice():
    """``evaluate`` once declared --qrel-gain twice; click only warns about it."""
    for path, command in _commands(cli):
        flags = [flag for param in command.params for flag in (*param.opts, *param.secondary_opts)]
        duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
        assert not duplicates, f"`rcp-ndcg {' '.join(path)}` declares {duplicates} more than once"


@pytest.mark.parametrize("name", sorted(_LAZY_SUBCOMMANDS))
def test_the_root_help_of_a_group_is_the_groups_own(name):
    # The root lists each group without importing it, so its line is a copy; it must not drift from the group.
    module, attribute, short = _LAZY_SUBCOMMANDS[name]
    group = getattr(importlib.import_module(module), attribute)
    assert short == group.get_short_help_str(200).rstrip(".")


def test_short_help_is_one_line_of_single_spaces():
    """A docstring's continuation indentation once leaked into the short help ('per-criterion     probabilities')."""
    for path, command in _commands(cli):
        short = command.get_short_help_str(limit=10_000)
        assert "  " not in short and "\n" not in short, f"`rcp-ndcg {' '.join(path)}`: {short!r}"


def test_mcp_group_serves_only() -> None:
    """`mcp tools` is gone: the MCP surface is `mcp serve`; a tool call goes through an MCP client or
    `rcp_ndcg.mcp.call_tool()` in Python."""
    from rcp_ndcg.cli.mcp import mcp_group

    assert set(mcp_group.list_commands(click.Context(mcp_group))) == {"serve"}


def test_the_command_index_renders_a_switch_as_flag_and_no_flag() -> None:
    """A boolean that defaults on is a `--x/--no-x` switch, what `describe` and `schema show commands` expose."""
    from rcp_ndcg.cli.introspect import command_index

    flags = {flag.flag: flag for flag in command_index().commands["run resume"].flags}

    assert "--resume/--no-resume" in flags
    assert flags["--resume/--no-resume"].type == "boolean"
