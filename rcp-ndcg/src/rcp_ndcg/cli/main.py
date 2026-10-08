"""The ``rcp-ndcg`` root command: global options, the lazily loaded command groups, and the outermost guard.

``rcp-ndcg <noun> <verb>``. Global options come before the noun::

    rcp-ndcg -v --log-file run.log --env-file .env data inspect --dataset suite:nanobeir --json

* ``-v`` logs at DEBUG on stderr; ``-vv`` also prints tracebacks; ``-q`` shows errors only. The default level
  is ``$RCP_NDCG_LOG_LEVEL`` or INFO.
* ``--log-file PATH`` receives every record at DEBUG, tracebacks included. Nothing is logged to a file
  otherwise.
* ``--env-file PATH`` loads environment variables from a file (existing variables win). No file is read
  implicitly.

The guard around the whole tree keeps the output contract on every exit path: with ``--json`` a usage error
or an unexpected failure still prints exactly one ``rcp-ndcg.cli.v1`` document, and no path prints a
traceback unless ``-vv`` is given. ``--help`` and ``--version`` are the exception: they print plain text and
exit 0, envelope or not.
"""

from __future__ import annotations

import importlib
import signal
import sys
import threading
from collections.abc import Sequence
from typing import Any

import click

from rcp_ndcg import __version__
from rcp_ndcg.cli.output import fail
from rcp_ndcg.errors import RcpNdcgError, UsageError, classify
from rcp_ndcg.support.logging import configure_logging, get_logger
from rcp_ndcg.support.paths import log_level

logger = get_logger(__name__)

#: name -> (module, attribute, short help). Groups are imported when invoked, so ``--help`` and a command of one
#: group never pay for the imports of another (torch stays out of ``rcp-ndcg data``).
_LAZY_SUBCOMMANDS: dict[str, tuple[str, str, str]] = {
    "data": ("rcp_ndcg.cli.data", "data_group", "Fetch, inspect, validate and convert datasets"),
    "retrieval": (
        "rcp_ndcg.cli.retrieval",
        "retrieval_group",
        "First-stage retrieval, system reranking and rank fusion",
    ),
    "judge": (
        "rcp_ndcg.cli.judge",
        "judge_group",
        "Judge candidate pools with an LLM: the tournament and the rubric; re-parse a store",
    ),
    "calibration": (
        "rcp_ndcg.cli.calibration",
        "calibration_group",
        "Fit the 2PL calibration, extend it without refitting, and show it",
    ),
    "eval": ("rcp_ndcg.cli.eval", "eval_group", "Score rankings, compare systems and explain queries"),
    "run": ("rcp_ndcg.cli.run", "run_group", "Run a config end to end, resume it, follow it; list and show runs"),
    "schema": (
        "rcp_ndcg.cli.schema",
        "schema_group",
        "List, show and export the JSON Schemas of configs, artifacts and outputs",
    ),
    "mcp": ("rcp_ndcg.cli.mcp", "mcp_group", "Serve the commands as MCP tools over stdio"),
    "doctor": (
        "rcp_ndcg.cli.doctor",
        "doctor_cmd",
        "Check the environment: versions, installed extras, directories, credentials present, endpoint reachable",
    ),
}


class _RootGroup(click.Group):
    """The root group: resolves :data:`_LAZY_SUBCOMMANDS` on demand and guards every exit path."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(_LAZY_SUBCOMMANDS)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        target = _LAZY_SUBCOMMANDS.get(cmd_name)
        if target is None:
            return None
        module_path, attr, _ = target
        return getattr(importlib.import_module(module_path), attr)

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows = [(name, _LAZY_SUBCOMMANDS[name][2]) for name in self.list_commands(ctx)]
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)

    def main(  # type: ignore[override]
        self, args: Sequence[str] | None = None, prog_name: str | None = None, **extra: Any
    ) -> Any:
        argv = list(sys.argv[1:] if args is None else args)
        as_json = "--json" in argv
        extra.pop("standalone_mode", None)
        previous = _interrupt_on_sigterm()
        try:
            code = super().main(argv, prog_name=prog_name or "rcp-ndcg", standalone_mode=False, **extra)
        except click.exceptions.Exit as exc:
            sys.exit(exc.exit_code)
        except click.Abort:
            fail(_command_of(None, argv), classify(KeyboardInterrupt()), as_json=as_json)
        except click.ClickException as exc:
            if not as_json:
                exc.show()
                sys.exit(exc.exit_code)
            error = UsageError(exc.format_message(), hint="see --help for the expected arguments")
            if not isinstance(exc, click.UsageError):
                error = RcpNdcgError(exc.format_message(), hint="this is a bug: please report it")
            fail(_command_of(getattr(exc, "ctx", None), argv), error, as_json=True)
        except Exception as exc:  # noqa: BLE001 -- the outermost guard: typed, logged, never a bare traceback
            logger.debug("unexpected failure", exc_info=exc)  # the traceback: --log-file, or stderr with -vv
            fail(_command_of(None, argv), classify(exc), as_json=as_json)
        finally:
            if previous is not None:
                signal.signal(signal.SIGTERM, previous)
        sys.exit(code if isinstance(code, int) else 0)


def _interrupt_on_sigterm() -> Any:
    """Stop on ``SIGTERM`` as on ``SIGINT``: a scheduler's stop is reported as INTERRUPTED (exit 9), after the
    ``finally`` blocks ran (a mirror's last flush, a store's last records). Returns the previous handler."""
    if threading.current_thread() is not threading.main_thread():
        return None

    def interrupt(*_: object) -> None:
        raise KeyboardInterrupt

    return signal.signal(signal.SIGTERM, interrupt)


def _command_of(ctx: click.Context | None, argv: list[str]) -> str:
    """The command path for the envelope: from the failing context, else the leading words of argv.

    A global option's value (``--env-file f.env``) is not a command word: the root group's value-taking options
    are skipped together with their values, so the envelope's ``command`` names the command."""
    if ctx is not None:
        _, _, path = ctx.command_path.partition(" ")  # without the program name
        return path or "rcp-ndcg"
    takes_value = {
        option
        for param in cli.params
        if isinstance(param, click.Option) and not param.is_flag and not param.count
        for option in param.opts
    }
    words = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token.startswith("-"):
            if words:
                break
            skip_next = token in takes_value
            continue
        words.append(token)
    return " ".join(words[:2]) or "rcp-ndcg"


@click.group(
    name="rcp-ndcg",
    cls=_RootGroup,
    help="RCP-nDCG: calibrated relevance for retrieval evaluation.",
    context_settings={"show_default": True, "help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, "--version", package_name="rcp-ndcg", prog_name="rcp-ndcg")
@click.option(
    "-v", "--verbose", "verbose", count=True, show_default=False, help="-v: DEBUG logs on stderr; -vv: also tracebacks."
)
@click.option("-q", "--quiet", is_flag=True, help="Only errors on stderr.")
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False),
    default=None,
    help="Also write every log record, tracebacks included, to this file.",
)
@click.option(
    "--env-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Load environment variables from this file first (variables already set win).",
)
def cli(verbose: int, quiet: bool, log_file: str | None, env_file: str | None) -> None:
    """Top-level ``rcp-ndcg`` command group."""
    if env_file is not None:
        import dotenv

        dotenv.load_dotenv(env_file, override=False)
    level = "ERROR" if quiet else ("DEBUG" if verbose else log_level())
    configure_logging(level, log_file=log_file, tracebacks=verbose >= 2)


def main() -> None:
    """Console-script entry point."""
    cli.main()


if __name__ == "__main__":
    main()


__all__ = ["cli", "main"]
