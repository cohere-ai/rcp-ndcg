"""Path setup for the plugin tests.

The plugin is not installed in the dev venv (it is built and installed on the GPU
wave), so the tests import it from this package's ``src`` tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
