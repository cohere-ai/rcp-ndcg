"""Every ``rcp-ndcg`` command in the repository's Markdown exists and parses.

A command in a fenced shell block must parse completely: its subcommand path exists, every flag belongs to it, choice
values are valid and required options are given. A command quoted inline in prose (``rcp-ndcg judge rubric``) must
name an existing command, and the flags it quotes must belong to it.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import click
import pytest

from rcp_ndcg.cli.main import cli
from tests.docs._markdown import ROOT, code_blocks, markdown_files, prose

SHELL = {"bash", "sh", "shell", "console", "zsh"}
_STOP = {"#", "|", "&&", "||", ";", ">", "2>", "&"}
_INLINE = re.compile(r"`(rcp-ndcg(?: [^`]*)?)`")


def _commands_in_shell(text: str) -> list[list[str]]:
    """The argument lists after ``rcp-ndcg`` in a shell block (continuation lines joined, comments dropped)."""
    commands: list[list[str]] = []
    for line in text.replace("\\\n", " ").splitlines():
        try:
            tokens = shlex.split(line, comments=True)
        except ValueError:
            continue
        if "rcp-ndcg" not in tokens:
            continue
        args = tokens[tokens.index("rcp-ndcg") + 1 :]
        cut = next((i for i, token in enumerate(args) if token in _STOP), len(args))
        commands.append(args[:cut])
    return commands


def _resolve(args: list[str]) -> tuple[click.Command, list[str], str]:
    """Walk the command tree: the leaf command, its remaining arguments and its name. Raises ``LookupError``."""
    command: click.Command = cli
    name = "rcp-ndcg"
    rest = list(args)
    while isinstance(command, click.Group):
        while rest and rest[0].startswith("-"):  # options of the group itself (-v, --env-file PATH, --version)
            option = rest.pop(0)
            param = next((p for p in command.params if option.split("=")[0] in p.opts), None)
            if param is None:
                raise LookupError(f"{name} has no option {option}")
            takes_value = isinstance(param, click.Option) and not (param.is_flag or param.count)
            if takes_value and "=" not in option and rest:
                rest.pop(0)
        if not rest:
            return command, rest, name
        sub = command.get_command(click.Context(command), rest[0])
        if sub is None:
            raise LookupError(f"{name} has no command {rest[0]!r}")
        command, name, rest = sub, f"{name} {rest[0]}", rest[1:]
    return command, rest, name


def _check_full(args: list[str]) -> str | None:
    try:
        command, rest, name = _resolve(args)
    except LookupError as exc:
        return str(exc)
    if isinstance(command, click.Group):
        return None if not rest or rest == ["--help"] else f"{name}: unexpected {rest}"
    try:
        command.make_context(name, rest)
    except click.exceptions.Exit:
        return None
    except click.ClickException as exc:
        return f"{name} {' '.join(rest)}: {exc.format_message()}"
    return None


def _check_inline(args: list[str]) -> str | None:
    words = [a for a in args if not a.startswith("-") and re.fullmatch(r"[a-z][a-z-]*", a)]
    flags = [a.split("=")[0] for a in args if a.startswith("--")]
    command: click.Command = cli
    name = "rcp-ndcg"
    for word in words:
        if not isinstance(command, click.Group):
            break
        sub = command.get_command(click.Context(command), word)
        if sub is None:
            return f"{name} has no command {word!r}"
        command, name = sub, f"{name} {word}"
    known = {opt for p in command.params for opt in [*p.opts, *getattr(p, "secondary_opts", [])]} | {"--help"}
    unknown = [f for f in flags if f not in known and not (command is cli and f == "--version")]
    return f"{name}: no option {unknown}" if unknown else None


def _problems(path: Path) -> list[str]:
    text = (ROOT / path).read_text(encoding="utf-8")
    problems: list[str] = []
    for block in code_blocks(text):
        if block.language in SHELL:
            for args in _commands_in_shell(block.text):
                if (problem := _check_full(args)) is not None:
                    problems.append(f"line {block.line}: {problem}")
    for quoted in _INLINE.findall(prose(text, keep_inline_code=True).replace("\n", " ")):
        if (problem := _check_inline(quoted.split()[1:])) is not None:
            problems.append(f"`{quoted}`: {problem}")
    return problems


FILES = [f for f in markdown_files() if "rcp-ndcg" in (ROOT / f).read_text(encoding="utf-8")]


@pytest.mark.parametrize("path", FILES, ids=str)
def test_every_documented_command_exists_and_parses(path: Path) -> None:
    assert _problems(path) == []


def test_a_wrong_command_is_reported() -> None:
    assert _check_full(["eval", "score", "--suite", "nanobeir"])  # --rankings is required
    assert _check_full(["eval", "scores", "--rankings", "r.parquet"])
    assert _check_full(["judge", "serve", "--engine", "tgi"])
    assert _check_full(["-v", "run", "list", "--json"]) is None
    assert _check_inline(["run", "start", "--resume"])
    assert _check_inline(["calibration", "show", "--calibration"]) is None
