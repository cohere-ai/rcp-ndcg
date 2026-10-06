"""The job scripts: bootstrap.sh is the superseded stub; submit.sh's argv under KJOBS=echo."""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

JOBS = Path(__file__).resolve().parents[1] / "jobs"
BOOTSTRAP = JOBS / "bootstrap.sh"
SUBMIT = JOBS / "submit.sh"

needs_shellcheck = pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck is not installed")


@needs_shellcheck
def test_both_scripts_pass_shellcheck() -> None:
    for script in (BOOTSTRAP, SUBMIT):
        completed = subprocess.run(["shellcheck", str(script)], capture_output=True, text=True)
        assert completed.returncode == 0, f"{script.name}: {completed.stdout}{completed.stderr}"


def test_bootstrap_is_the_superseded_stub() -> None:
    """bootstrap.sh installs nothing and fails clearly: the rc-build node bootstrap owns node setup."""
    completed = subprocess.run(["bash", str(BOOTSTRAP)], capture_output=True, text=True)
    assert completed.returncode == 2
    assert "rc-build" in completed.stderr
    assert "pip" not in (BOOTSTRAP.read_text(encoding="utf-8"))


def _scratch_auth(tmp_path: Path) -> Path:
    """A placeholder auth file in scratch; submit.sh only names its path, never reads it."""
    path = tmp_path / "gcs_auth.sh"
    path.write_text("# placeholder: not the real script\n", encoding="utf-8")
    return path


def _env(tmp_path: Path) -> dict[str, str]:
    """The submit environment: config and auth paths from scratch, no machine paths."""
    config = tmp_path / "config.yaml"
    config.write_text("worker: {cpu: 1}\n", encoding="utf-8")
    return {
        "PATH": "/usr/bin:/bin",
        "KJOBS": "echo",
        "RCP_KJOBS_CONFIG": str(config),
        "RCP_GCS_AUTH_FILE": str(_scratch_auth(tmp_path)),
        "EXTRA_DIRS": "",
    }


def test_submit_prints_the_expected_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """KJOBS=echo prints the staged upload and the job submission without running either."""
    if (
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"], cwd=JOBS.parent.parent.parent, capture_output=True
        ).returncode
        != 0
    ):
        pytest.skip("submit.sh stages `git archive HEAD`, which needs a checkout (a fresh archive has none)")
    recipes = tmp_path / "recipes.txt"
    recipes.write_text("fixture-embed\n", encoding="utf-8")
    monkeypatch.chdir(JOBS.parent.parent.parent)
    completed = subprocess.run(
        ["bash", str(SUBMIT), "testwave", str(recipes), "gs://YOUR-BUCKET/stage", "gs://YOUR-BUCKET/waves"],
        capture_output=True,
        text=True,
        env=_env(tmp_path),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 2, lines
    submission_words = shlex.split(lines[1])
    assert "submit" in submission_words
    assert "app=rcp-testwave" in submission_words
    worker = next(word for word in submission_words if word.startswith("worker.command="))
    assert worker.startswith("worker.command=/bin/bash /etc/rcp/files/bootstrap/bootstrap.sh ")
    assert f"files.bootstrap.from_file={JOBS / 'bootstrap.sh'}" in submission_words
    assert f"files.gcsauth.from_file={tmp_path / 'gcs_auth.sh'}" in submission_words
    assert "files.gcsauth.mount_path=/etc/rcp/gcs_auth.sh" in submission_words
    config_flag = submission_words[submission_words.index("-f") + 1]
    assert config_flag == str(tmp_path / "config.yaml")  # RCP_KJOBS_CONFIG, not a default path


def test_submit_fails_with_a_usage_message_without_the_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No defaults: the script refuses to run without RCP_KJOBS_CONFIG / RCP_GCS_AUTH_FILE."""
    recipes = tmp_path / "recipes.txt"
    recipes.write_text("fixture-embed\n", encoding="utf-8")
    monkeypatch.chdir(JOBS.parent.parent.parent)
    env = _env(tmp_path)
    for missing in ("RCP_KJOBS_CONFIG", "RCP_GCS_AUTH_FILE"):
        broken = {key: value for key, value in env.items() if key != missing}
        completed = subprocess.run(
            ["bash", str(SUBMIT), "w", str(recipes), "gs://b/stage", "gs://b/waves"],
            capture_output=True,
            text=True,
            env=broken,
        )
        assert completed.returncode != 0, (missing, completed.stdout)
        assert missing in completed.stderr
