"""Where the package reads and writes by default, resolved lazily from the environment.

Nothing here runs at import and nothing creates a directory: callers create what they write into. The
variables:

* ``RCP_NDCG_CACHE_DIR``: caches (media, remote objects, synced trees). Default ``$XDG_CACHE_HOME/rcp-ndcg``,
  else ``~/.cache/rcp-ndcg``.
* ``RCP_NDCG_RUNS_DIR``: where run directories live. Default ``runs`` in the working directory.
* ``RCP_NDCG_LOG_LEVEL``: the stderr log level when neither ``-v`` nor ``-q`` is given. Default ``INFO``.
"""

from __future__ import annotations

import os
from pathlib import Path

from rcp_ndcg.errors import ConfigError

CACHE_DIR_ENV = "RCP_NDCG_CACHE_DIR"
RUNS_DIR_ENV = "RCP_NDCG_RUNS_DIR"
LOG_LEVEL_ENV = "RCP_NDCG_LOG_LEVEL"

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def cache_dir() -> Path:
    """The cache root: ``$RCP_NDCG_CACHE_DIR``, else the user cache directory (not created)."""
    override = os.environ.get(CACHE_DIR_ENV)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "rcp-ndcg"


def runs_dir() -> Path:
    """The runs root: ``$RCP_NDCG_RUNS_DIR``, else ``runs`` relative to the working directory (not created)."""
    return Path(os.environ.get(RUNS_DIR_ENV) or "runs").expanduser()


def log_level() -> str:
    """The default stderr log level: ``$RCP_NDCG_LOG_LEVEL`` (a logging level name), else ``INFO``.

    Raises:
        ConfigError: The variable names no logging level.
    """
    level = (os.environ.get(LOG_LEVEL_ENV) or "INFO").upper()
    if level not in _LOG_LEVELS:
        raise ConfigError(f"${LOG_LEVEL_ENV}={level!r} is not a log level", hint=f"use one of {', '.join(_LOG_LEVELS)}")
    return level


__all__ = [
    "CACHE_DIR_ENV",
    "LOG_LEVEL_ENV",
    "RUNS_DIR_ENV",
    "cache_dir",
    "log_level",
    "runs_dir",
]
