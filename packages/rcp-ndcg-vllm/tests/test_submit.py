"""The shell scripts: shellcheck, and submit.sh's argv under KJOBS=echo with a scratch auth file."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

JOBS = Path(__file__).resolve().parents[1] / "jobs"
BOOTSTRAP = JOBS / "bootstrap.sh"
SUBMIT = JOBS / "submit.sh"

needs_shellcheck = pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck is not installed")


@needs_shellcheck
def test_bootstrap_passes_shellcheck() -> None:
    completed = subprocess.run(["shellcheck", str(BOOTSTRAP)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@needs_shellcheck
def test_submit_passes_shellcheck() -> None:
    completed = subprocess.run(["shellcheck", str(SUBMIT)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr


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
    if subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=JOBS.parent.parent.parent,
                      capture_output=True).returncode != 0:  # fmt: skip
        pytest.skip("submit.sh stages `git archive HEAD`, which needs a checkout (a fresh archive has none)")
    recipes = tmp_path / "recipes.txt"
    recipes.write_text("fixture-embed\n", encoding="utf-8")
    monkeypatch.chdir(JOBS.parent.parent.parent)  # the repository root: git archive needs the checkout
    completed = subprocess.run(
        [
            "bash",
            str(SUBMIT),
            "testwave",
            str(recipes),
            "gs://YOUR-BUCKET/stage",
            "gs://YOUR-BUCKET/waves",
        ],  # fmt: skip
        capture_output=True,
        text=True,
        env=_env(tmp_path),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert len(lines) == 2, lines  # the upload, then the job submission
    upload_words = shlex.split(lines[0])
    submission_words = shlex.split(lines[1])
    assert "storage" in upload_words and "cp" in upload_words
    assert any("gs://YOUR-BUCKET/stage/testwave/code.tar.gz" in word for word in upload_words)
    assert "submit" in submission_words
    assert "app=rcp-testwave" in submission_words
    worker = next(word for word in submission_words if word.startswith("worker.command="))
    assert worker.startswith("worker.command=/bin/bash /etc/rcp/files/bootstrap/bootstrap.sh ")
    assert "gs://YOUR-BUCKET/stage/testwave/code.tar.gz" in worker
    assert "gs://YOUR-BUCKET/waves/testwave" in worker
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


def test_bootstrap_derives_its_light_dependencies_from_the_manifests(tmp_path: Path) -> None:
    """bootstrap.sh reads the packages' pyproject.toml files (tomllib) instead of a hand-mirrored list."""

    if not (JOBS.parent.parent.parent / "pyproject.toml").is_file():
        pytest.skip("the derivation reads all three manifests; a fresh archive of this package has only one")
    script = (JOBS / "bootstrap.sh").read_text(encoding="utf-8")
    start = script.index('python3 - "$1" <<')
    body = script[script.index("\n", start) + 1 :]
    body = body[: body.index("\nPY\n")]
    script_path = tmp_path / "light_deps.py"
    script_path.write_text(body, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script_path), str(JOBS.parent.parent.parent)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert completed.returncode == 0, completed.stderr
    derived = completed.stdout.split()
    assert "bm25s==0.2.13" in derived  # pinned, exactly as the root manifest pins it
    assert any(dependency.startswith("sentence-transformers") for dependency in derived)
    assert all("torch" not in dependency for dependency in derived)  # the image provides it
    assert "pandas" in " ".join(derived) and "scipy" in " ".join(derived)
