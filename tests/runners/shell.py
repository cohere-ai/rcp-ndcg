"""Check rendered shell scripts with shellcheck when it is installed (``pip install shellcheck-py``)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def assert_shellcheck_clean(script: str) -> None:
    """Assert shellcheck finds nothing of severity warning or above in ``script``; a no-op without shellcheck."""
    beside = Path(sys.executable).with_name("shellcheck")  # shellcheck-py installed in this environment
    binary = str(beside) if beside.is_file() else shutil.which("shellcheck")
    if binary is None:
        return
    result = subprocess.run(
        [binary, "--shell=bash", "--severity=warning", "-"], input=script, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
