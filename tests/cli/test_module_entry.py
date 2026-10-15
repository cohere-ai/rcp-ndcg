"""``python -m rcp_ndcg.cli``: the module entry point runs the CLI; importing the module does not."""

from __future__ import annotations

import subprocess
import sys


def test_importing_the_module_entry_point_does_not_run_the_cli() -> None:
    """``__main__.py`` called ``main()`` unguarded, so any import of the module executed click and
    printed the command help. The module is importable surface and must stay inert; the console
    script (``rcp_ndcg.cli.main:main``) and ``python -m rcp_ndcg.cli`` are what run the CLI."""
    result = subprocess.run(
        [sys.executable, "-c", "import rcp_ndcg.cli.__main__"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "" and result.stderr == "", (result.stdout, result.stderr)


def test_running_the_module_entry_point_still_runs_the_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "rcp_ndcg.cli", "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "Usage:" in result.stdout
