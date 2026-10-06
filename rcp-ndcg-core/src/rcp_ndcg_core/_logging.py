"""Logger naming for the core package.

Core stays import-light and must not depend on ``rcp_ndcg``, but its records belong under the ``rcp_ndcg`` logger
tree, so the handlers :func:`rcp_ndcg.support.logging.configure_logging` installs apply to them:
``rcp_ndcg_core._records`` logs as ``rcp_ndcg.core._records``.
"""

from __future__ import annotations

import logging

_PACKAGE = "rcp_ndcg_core"


def get_logger(name: str) -> logging.Logger:
    """The logger of the core module *name* (``rcp_ndcg_core.<module>``), under ``rcp_ndcg.core``."""
    return logging.getLogger(f"rcp_ndcg.core{name.removeprefix(_PACKAGE)}")


__all__ = ["get_logger"]
