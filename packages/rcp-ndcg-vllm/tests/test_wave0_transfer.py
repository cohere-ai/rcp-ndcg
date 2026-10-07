"""The GCS transfer in wave 0: with neither CLI on PATH, the report upload takes the python path."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.conftest import sandbox_path

WAVE0_SH = Path(__file__).resolve().parents[1] / "src" / "rcp_ndcg_vllm" / "jobs" / "wave0.sh"
JOBS = Path(__file__).resolve().parents[1] / "jobs"
REPORT_PY = JOBS / "report.py"
WAVE0_HOST = JOBS / "wave0_host.py"
BOOTSTRAP_SH = JOBS / "bootstrap.sh"

PINNED_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"

_SHELL_TOOLS = ("bash", "mkdir", "mktemp", "rm", "date", "wc", "tail")
"""The system tools wave0.sh and the scripts it calls run up to the bootstrap's expected failure; nothing else
of the machine is on the test's PATH (a real gcloud there would stand in for the absent one)."""


def _fake_tools(tmp_path: Path) -> dict[str, Path]:
    """A bin dir whose python3 logs its argv, simulates the tools install and the gcs.py transfer (no test
    reaches a bucket), and passes everything else through to the real one."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.log"
    fake_python = bin_dir / "python3"
    real = Path(sys.executable)
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$*" >>"{log}"\n'
        'if [[ "$1" == "-m" && "$2" == "pip" ]]; then exit 0; fi\n'  # the tools install is simulated
        'if [[ "$1" == */gcs.py ]]; then exit 0; fi\n'  # and so is the transfer: no test reaches a bucket
        f'exec "{real}" "$@"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    fake_smi = bin_dir / "nvidia-smi"
    fake_smi.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "$1" == "--list-gpus" ]]; then for i in 0 1 2 3 4 5 6 7; do echo "GPU $i"; done; exit 0; fi\n'
        'if [[ "$2" == "--query-gpu=index,name,memory.total" ]]; then'
        ' for i in 0 1 2 3 4 5 6 7; do echo "$i, NVIDIA B200, 180000"; done; exit 0; fi\n'
        'echo "580.0"\n',
        encoding="utf-8",
    )
    fake_smi.chmod(0o755)
    return {"bin": bin_dir, "log": log}


def _fake_stage(tmp_path: Path) -> Path:
    """A minimal staged RC (the manifest hashes what it lists; the bootstrap fails at import vllm)."""
    stage = tmp_path / "rc-stage"
    (stage / "wheelhouse").mkdir(parents=True)
    (stage / "recipes").mkdir()
    (stage / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "rcp-ndcg.rc-manifest.v1",
                "rc_name": "fake",
                "version": "0.0.1",
                "commit": "0" * 40,
                "files": [],
            }
        ),
        encoding="utf-8",
    )
    (stage / "requirements-constraints.txt").write_text("# fake\n", encoding="utf-8")
    (stage / "requirements-reference.txt").write_text("# fake\n", encoding="utf-8")
    return stage


def test_wave0_without_the_clis_takes_the_python_transfer_path(tmp_path: Path) -> None:
    """No gcloud and no gsutil: the auth script runs first, then the gcsfs helper; the path is recorded."""
    tools = _fake_tools(tmp_path)
    stage = _fake_stage(tmp_path)
    (tmp_path / "token").write_text("hf_fake_0123456789abcdef\n", encoding="utf-8")
    (tmp_path / "auth.sh").write_text("# placeholder auth: not the real script\n", encoding="utf-8")
    (tmp_path / "auth.sh").chmod(0o755)
    out_uri = f"gs://YOUR-BUCKET/waves/wave0-{tmp_path.name}"

    completed = subprocess.run(
        [
            "bash",
            str(WAVE0_SH),
            str(stage),
            out_uri,
        ],  # fmt: skip
        capture_output=True,
        text=True,
        env={
            "PATH": sandbox_path(tools["bin"], *_SHELL_TOOLS),
            "HOME": str(tmp_path),
            "TMPDIR": str(tmp_path),  # the run's work directories land under tmp_path, never /tmp
            "HF_TOKEN": "hf_fake_0123456789abcdef",
            "WAVE0_MIN_SHM_GIB": "0",
            "WAVE0_MIN_FREE_GIB": "0",
            "RCP_GCS_AUTH_FILE": str(tmp_path / "auth.sh"),
            "RCP_REPORT_PY": str(REPORT_PY),
            "RCP_HOST_PY": str(WAVE0_HOST),
            "RCP_BOOTSTRAP_SH": str(BOOTSTRAP_SH),
            "RCP_GCS_HELPER_SH": str(JOBS / "gcs.sh"),
            "RCP_GCS_HELPER_PY": str(JOBS / "gcs.py"),
            "RCP_REFERENCE_DEPS_PY": str(JOBS / "reference_deps.py"),
            "RCP_IMAGE": "vllm/vllm-openai:v0.31.0",
            "RCP_IMAGE_DIGEST": "sha256:" + "0" * 64,
        },
    )
    calls = tools["log"].read_text(encoding="utf-8")
    assert "gcloud" not in calls and "gsutil" not in calls  # neither CLI ran
    assert "-m pip install" in calls  # the gcsfs tools install went to its own --target directory
    assert "gcs.py cp" in calls  # the report upload ran through the python helper
    # The run itself cannot finish on a CPU box (no vllm); the fail-fast contract still holds.
    assert completed.returncode == 1
    report = _last_report_text(completed.stdout)
    assert report["failed_step"] == "bootstrap"  # the CPU box has no vllm: the expected failure
    assert report["host"]["transfer"] == "python"  # the path that ran, recorded
    assert report["failed_step"] == "bootstrap"
    assert "PYNVML" not in json.dumps(report)  # no token-shaped or unexpected fields
    assert "hf_fake_0123456789abcdef" not in json.dumps(report)  # the token's value never leaks


def _last_report_text(stdout: str) -> dict:
    """The emitted report: the last JSON object on stdout (the fail-fast emit)."""
    import re

    starts = [match.start() for match in re.finditer(r"^\{$", stdout, flags=re.MULTILINE)]
    assert starts, "the report was never emitted"
    blob = stdout[starts[-1] :]
    decoder = json.JSONDecoder()
    document, _ = decoder.raw_decode(blob)
    return document
