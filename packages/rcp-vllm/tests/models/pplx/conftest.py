"""Path setup for the folded-model tests.

The package's model modules import torch and vLLM, which the dev venv does not carry: the tests
import them through this package's ``src`` tree (the lazy registration imports nothing until then).
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
