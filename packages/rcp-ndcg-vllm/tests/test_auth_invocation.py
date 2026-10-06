"""The node scripts run the mounted GCS auth script without needing its execute bit (job-CLI mounts carry none)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

BOOTSTRAP = Path(__file__).resolve().parents[1] / "jobs" / "bootstrap.sh"


def _auth_function() -> str:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    match = re.search(r"^auth\(\) \{\n.*?^\}\n", text, flags=re.S | re.M)
    assert match, "bootstrap.sh defines auth()"
    return match.group(0)


def _run_auth(tmp_path: Path, body: str) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "gcs_auth.sh"
    script.write_text(body, encoding="utf-8")
    script.chmod(0o644)  # mounted read-only, without the execute bit
    program = f'{_auth_function()}\nAUTH_SCRIPT="{script}"\nauth\n'
    return subprocess.run(["bash", "-c", program], capture_output=True, text=True, cwd=tmp_path, check=False)


def test_a_non_executable_auth_script_runs(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    result = _run_auth(tmp_path, f"touch '{marker}'\n")
    assert result.returncode == 0, result.stderr
    assert marker.exists()


def test_a_failing_auth_script_reports_only_its_exit_code(tmp_path: Path) -> None:
    result = _run_auth(tmp_path, "echo SECRET-VALUE; exit 7\n")
    assert result.returncode == 1
    assert "exit code 7" in result.stderr
    assert "SECRET-VALUE" not in result.stdout + result.stderr


MOUNTED = ("AUTH_SCRIPT", "BOOTSTRAP_SH", "GCS_SH", "GCS_HELPER_PY", "REPORT_PY", "HOST_PY")
NODE_SCRIPTS = (BOOTSTRAP.parent / "bootstrap.sh", BOOTSTRAP.parents[1] / "src" / "rcp_ndcg_vllm" / "jobs" / "wave0.sh")


def test_no_node_script_executes_a_mounted_file_directly() -> None:
    """A mounted file is run through ``bash``/``python3`` or sourced, never as a program (no execute bit)."""
    direct = re.compile(r'(^|[;&|(!]|\bthen|\bdo|\bif)\s*"\$\{?(' + "|".join(MOUNTED) + r')\}?"')
    offenders = [
        f"{path.name}:{number}: {line.strip()}"
        for path in NODE_SCRIPTS
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if direct.search(line)
    ]
    assert not offenders, offenders
