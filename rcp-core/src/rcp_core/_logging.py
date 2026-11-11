"""Logger naming for the core package.

Core stays import-light and must not depend on ``rcp``, but its records belong under the ``rcp`` logger
tree, so the handlers :func:`rcp.support.logging.configure_logging` installs apply to them:
``rcp_core.records`` logs as ``rcp.core.records``.
"""

from __future__ import annotations

import logging

_PACKAGE = "rcp_core"


def get_logger(name: str) -> logging.Logger:
    """The logger of the core module *name* (``rcp_core.<module>``), under ``rcp.core``."""
    return logging.getLogger(f"rcp.core{name.removeprefix(_PACKAGE)}")


__all__ = ["get_logger"]
