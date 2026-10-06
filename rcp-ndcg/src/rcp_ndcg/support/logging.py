"""Logging: named loggers for the library, one stderr (and optional file) configuration for the CLI.

The library only creates loggers (:func:`get_logger`) and installs no handler of its own, with one exception: while
a run executes, its records also go to the run's ``logs/run.log`` (:func:`rcp_ndcg.runs.run.execute_run`). The CLI
calls :func:`configure_logging` once per invocation from its ``-v/-vv``, ``-q`` and ``--log-file`` options.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rcp_ndcg.errors import ConfigError

BASE_LOGGER_NAME = "rcp_ndcg"
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class _StderrHandler(logging.Handler):
    """Writes through ``tqdm.write`` when tqdm is loaded, so log lines do not break progress bars.

    The stream is looked up at emit time: a test runner that swaps ``sys.stderr`` gets the lines.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            tqdm = sys.modules.get("tqdm")
            if tqdm is not None:
                tqdm.tqdm.write(message, file=sys.stderr)
            else:
                sys.stderr.write(message + "\n")
        except Exception:  # noqa: BLE001 -- logging must never raise into the caller
            self.handleError(record)


class _NoTracebackFormatter(logging.Formatter):
    """Formats a record without its traceback (the log file keeps it; stderr shows it only on request)."""

    def format(self, record: logging.LogRecord) -> str:
        if record.exc_info is None and record.exc_text is None:
            return super().format(record)
        plain = logging.makeLogRecord(record.__dict__)
        plain.exc_info = None
        plain.exc_text = None
        return super().format(plain)


def get_logger(name: str) -> logging.Logger:
    """The logger for *name*, under the package's ``rcp_ndcg`` logger (the prefix is added once)."""
    if name == BASE_LOGGER_NAME or name.startswith(f"{BASE_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{BASE_LOGGER_NAME}.{name}")


def configure_logging(
    level: int | str, *, log_file: str | Path | None = None, tracebacks: bool = False
) -> logging.Logger:
    """Send the package's log records to stderr at *level*, and every record to *log_file* if given.

    Calling it again replaces the handlers it installed before, so repeated CLI invocations in one process do
    not duplicate lines.

    Args:
        level: The stderr level (a :mod:`logging` level or its name).
        log_file: A file that receives every record at ``DEBUG``, tracebacks included; its directory is
            created. ``None`` writes no file.
        tracebacks: Whether stderr shows the tracebacks attached to records (the file always has them).

    Returns:
        The package's root logger.
    """
    logger = logging.getLogger(BASE_LOGGER_NAME)
    for handler in [h for h in logger.handlers if getattr(h, "_rcp_ndcg", False)]:
        logger.removeHandler(handler)
        handler.close()
    stderr_level = logging.getLevelName(level.upper()) if isinstance(level, str) else level
    if isinstance(level, str) and not isinstance(stderr_level, int):
        raise ConfigError(
            f"unknown log level {level!r}; expected one of {', '.join(sorted(logging.getLevelNamesMapping()))} "
            "(or the numeric value)"
        )
    stderr = _StderrHandler(level=stderr_level)
    stderr.setFormatter(logging.Formatter(_FORMAT) if tracebacks else _NoTracebackFormatter(_FORMAT))
    handlers: list[logging.Handler] = [stderr]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        handlers.append(file_handler)
    for handler in handlers:
        handler._rcp_ndcg = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.setLevel(min(h.level for h in handlers))
    logger.propagate = False
    return logger


__all__ = ["BASE_LOGGER_NAME", "configure_logging", "get_logger"]
