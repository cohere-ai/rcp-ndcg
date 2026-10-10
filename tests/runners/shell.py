"""Check rendered shell scripts with shellcheck when it is installed (``pip install shellcheck-py``)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def heredoc_body(script: str, var: str) -> str:
    """The body of the ``read -r -d '' <var> <<'RCP_NDCG_<var>'`` heredoc in ``script`` (with its newline)."""
    tag = f"RCP_NDCG_{var}"
    lines = script.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(f"read -r -d '' {var} <<"))
    end = next(index for index in range(start + 1, len(lines)) if lines[index] == tag)
    return "\n".join(lines[start + 1 : end]) + "\n"


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
