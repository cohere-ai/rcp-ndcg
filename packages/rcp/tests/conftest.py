"""A distribution's test session: the fixtures, guards and hooks that live in :mod:`tests._session`."""

import sys
from pathlib import Path

# The repository root, so ``tests._session`` is importable from a package's test tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests._session import *  # noqa: E402,F401,F403
