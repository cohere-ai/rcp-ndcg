"""The job scripts: the freeze-diff guard, submit.sh's argv under KJOBS=echo, and the token-leak grep.

The scripts run on the node (bootstrap.sh, wave0.sh) or wherever the operator runs them (rc_build.sh,
submit.sh); what can be exercised on CPU is their plan, their guards and their refusals - never a node.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

JOBS = Path(__file__).resolve().parents[1] / "jobs"
BOOTSTRAP = JOBS / "bootstrap.sh"
SUBMIT = JOBS / "submit.sh"
RC_BUILD = JOBS / "rc_build.sh"
REPORT_PY = JOBS / "report.py"
WAVE0_SH = Path(__file__).resolve().parents[1] / "src" / "rcp_ndcg_vllm" / "jobs" / "wave0.sh"
WAVE0_HOST = JOBS / "wave0_host.py"

SCRIPTS = (BOOTSTRAP, SUBMIT, RC_BUILD, WAVE0_SH)

needs_shellcheck = pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck is not installed")


@needs_shellcheck
def test_every_script_passes_shellcheck() -> None:
    for script in (BOOTSTRAP, SUBMIT, RC_BUILD, WAVE0_SH):
        completed = subprocess.run(["shellcheck", str(script)], capture_output=True, text=True)
        assert completed.returncode == 0, f"{script.name}: {completed.stdout}{completed.stderr}"


@pytest.mark.parametrize("script", [BOOTSTRAP, SUBMIT, RC_BUILD, WAVE0_SH])
def test_every_script_parses(script: Path) -> None:
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


def test_bootstrap_without_arguments_prints_usage_and_fails() -> None:
    completed = subprocess.run(["bash", str(BOOTSTRAP)], capture_output=True, text=True)
    assert completed.returncode == 2
    assert "usage" in completed.stderr


# --- the freeze-diff guard (the engine environment may gain exactly the declared plugins) --------------


@pytest.fixture()
def bootstrap_functions() -> str:
    """The guard functions, by sourcing bootstrap.sh (its main runs nothing when sourced)."""
    return str(BOOTSTRAP)


def test_bootstrap_freeze_guard_passes_an_unchanged_environment(bootstrap_functions: str, tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    allowed = tmp_path / "allowed"
    before.write_text("pkg-a==1.0\npkg-b==2.0\n", encoding="utf-8")
    shutil.copy(before, after)
    allowed.write_text("my-plugin\n", encoding="utf-8")
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{bootstrap_functions}" && freeze_diff_guard "$1" "$2" "$3"',
            "bash",
            str(before),
            str(after),
            str(allowed),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_bootstrap_freeze_guard_fires_beyond_the_plugin(bootstrap_functions: str, tmp_path: Path) -> None:
    """A fake pip freeze that installs a rogue package fails the guard, with the package named."""
    before = tmp_path / "before"
    after = tmp_path / "after"
    allowed = tmp_path / "allowed"
    before.write_text("pkg-a==1.0\npkg-b==2.0\n", encoding="utf-8")
    after.write_text("pkg-a==1.0\npkg-b==2.0\nrogue==9.9\n", encoding="utf-8")
    allowed.write_text("my-plugin\n", encoding="utf-8")
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{bootstrap_functions}" && freeze_diff_guard "$1" "$2" "$3"',
            "bash",
            str(before),
            str(after),
            str(allowed),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "changed beyond the declared plugins" in completed.stderr
    assert "rogue" in completed.stderr


def test_bootstrap_freeze_guard_allows_exactly_the_plugin(bootstrap_functions: str, tmp_path: Path) -> None:
    """The plugin's two freeze shapes pass: name==version and the local-wheel direct URL."""
    before = tmp_path / "before"
    after = tmp_path / "after"
    allowed = tmp_path / "allowed"
    before.write_text("pkg-a==1.0\npkg-b==2.0\n", encoding="utf-8")
    allowed.write_text("my-plugin\n", encoding="utf-8")
    for installed in (
        "My_Plugin @ file:///opt/wheels/My_Plugin-1.2.3-py3-none-any.whl\n",
        "my_plugin==1.2.3\n",
    ):
        after.write_text("pkg-a==1.0\npkg-b==2.0\n" + installed, encoding="utf-8")
        completed = subprocess.run(
            [
                "bash",
                "-c",
                f'source "{bootstrap_functions}" && freeze_diff_guard "$1" "$2" "$3"',
                "bash",
                str(before),
                str(after),
                str(allowed),
            ],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (installed, completed.stderr)


def test_bootstrap_freeze_guard_fires_on_an_upgrade_beyond_the_plugin(bootstrap_functions: str, tmp_path: Path) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    allowed = tmp_path / "allowed"
    before.write_text("pkg-a==1.0\n", encoding="utf-8")
    after.write_text("pkg-a==1.1\nmy_plugin==1.2.3\n", encoding="utf-8")
    allowed.write_text("my-plugin\n", encoding="utf-8")
    completed = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{bootstrap_functions}" && freeze_diff_guard "$1" "$2" "$3"',
            "bash",
            str(before),
            str(after),
            str(allowed),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1


def test_freeze_name_of_parses_wheel_names_and_specs(bootstrap_functions: str) -> None:
    """A wheel filename, a plain name, a versioned spec and a staged path all give the canonical name."""
    cases = [
        ("My_Plugin-1.2.3-py3-none-any.whl", "my-plugin"),  # a wheel filename: the first dash splits
        ("/opt/staged/fixture-plug/plugin_wheel-1.0.0-py3-none-any.whl", "plugin-wheel"),
        ("some-plugin", "some-plugin"),  # a pip spec keeps its dashes
        ("my-plugin==1.2.3", "my-plugin"),
        ("/staged/org/pkg-2.0", "pkg-2-0"),  # consistent with how its freeze line canonicalises
    ]
    for argument, expected in cases:
        completed = subprocess.run(
            ["bash", "-c", f'source "{bootstrap_functions}" && freeze_name_of "$1"', "bash", argument],
            capture_output=True,
            text=True,
        )
        assert completed.stdout.strip() == expected, argument


# --- submit.sh: the operator's submission, KJOBS=echo prints the plan ----------------------------------


FAKE_TOKEN = "hf_fake_0123456789abcdef"


def _scratch_auth(tmp_path: Path) -> Path:
    """A placeholder auth file in tmp_path; submit.sh only names its path, never reads it."""
    path = tmp_path / "gcs_auth.sh"
    path.write_text("# placeholder: not the real script\n", encoding="utf-8")
    return path


def _env(tmp_path: Path) -> dict[str, str]:
    """The submit environment: config, auth and token paths from tmp_path, no machine paths."""
    config = tmp_path / "config.yaml"
    config.write_text("worker: {cpu: 4}\n", encoding="utf-8")
    token = tmp_path / "hf-token"
    token.write_text(f"{FAKE_TOKEN}\n", encoding="utf-8")
    return {
        "PATH": "/usr/bin:/bin",
        "KJOBS": "echo",
        "RCP_KJOBS_CONFIG": str(config),
        "RCP_GCS_AUTH_FILE": str(_scratch_auth(tmp_path)),
        "RCP_HF_TOKEN_FILE": str(token),
    }


def _submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *args: str, env_overrides: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    monkeypatch.chdir(JOBS.parent.parent.parent)
    env = _env(tmp_path)
    env.update(env_overrides or {})
    return subprocess.run(
        ["bash", str(SUBMIT), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def test_submit_prints_the_expected_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """KJOBS=echo prints one kjobs-go submit per wave, with the priority class and shm overrides."""
    completed = _submit(
        tmp_path, monkeypatch,
        "--priority", "dev-high", "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves", "wave-a",
    )  # fmt: skip
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.startswith("kjobs-go") or line.startswith("echo ")]
    words = shlex.split(lines[0])
    assert "submit" in words
    assert "app=rcp-wave-a" in words
    assert "priority_class=dev-high" in words
    assert "worker.shared_memory=128Gi" in words
    command = next(word for word in words if word.startswith("worker.command="))
    # The recipe wave: bootstrap.sh's wave mode, with the wave's list resolved on the node.
    assert command == (
        "worker.command=/bin/bash /etc/rcp/files/bootstrap/bootstrap.sh"
        " wave gs://YOUR-BUCKET/rc0 gs://YOUR-BUCKET/waves/wave-a --wave wave-a"
    )
    assert f"files.bootstrap.from_file={JOBS / 'bootstrap.sh'}" in words
    assert f"files.report.from_file={REPORT_PY}" in words
    assert f"files.gcsauth.from_file={tmp_path / 'gcs_auth.sh'}" in words
    config_flag = words[words.index("-f") + 1]
    assert config_flag == str(tmp_path / "config.yaml")  # RCP_KJOBS_CONFIG, not a default path


def test_submit_chains_waves_beyond_max_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """At most --max-jobs jobs in flight: wave i for i >= N depends on wave i-N (the plan shows it)."""
    completed = _submit(
        tmp_path,
        monkeypatch,
        "--max-jobs",
        "2",
        "gs://YOUR-BUCKET/rc0",
        "gs://YOUR-BUCKET/waves",
        "w1",
        "w2",
        "w3",
        "w4",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    submissions = [line for line in completed.stdout.splitlines() if "submit" in line and "app=rcp-" in line]
    assert len(submissions) == 4
    deps = [shlex.split(line) for line in submissions]
    depends = [next((word for word in words if word.startswith("depends_on=")), None) for words in deps]
    assert depends == [None, None, "depends_on=rcp-w1", "depends_on=rcp-w2"]


def test_submit_never_prints_the_token_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The token file's value reaches the argv only in a real run; echo mode prints the substitution."""
    completed = _submit(tmp_path, monkeypatch, "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves", "wave-a")
    assert completed.returncode == 0
    for stream in (completed.stdout, completed.stderr):
        assert FAKE_TOKEN not in stream, "the token's value reached the script's output"
    assert "secret.HF_TOKEN=" in completed.stdout  # the placeholder is part of the plan


def test_submit_wave0_mounts_the_wave0_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    completed = _submit(
        tmp_path,
        monkeypatch,
        "--script",
        "wave0",
        "gs://YOUR-BUCKET/rc0",
        "gs://YOUR-BUCKET/waves",
        "wave0",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    words = shlex.split(next(line for line in completed.stdout.splitlines() if line.startswith("echo ")))
    assert any(word.startswith("files.wave0.from_file=") and word.endswith("wave0.sh") for word in words)
    assert any(word.startswith("files.wave0host.from_file=") for word in words)
    command = next(word for word in words if word.startswith("worker.command="))
    assert "/etc/rcp/files/wave0/wave0.sh" in command


def test_submit_fails_with_a_usage_message_without_the_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No defaults: the script refuses to run without RCP_KJOBS_CONFIG, RCP_GCS_AUTH_FILE, RCP_HF_TOKEN_FILE."""
    for missing in ("RCP_KJOBS_CONFIG", "RCP_GCS_AUTH_FILE", "RCP_HF_TOKEN_FILE"):
        env = _env(tmp_path)
        env.pop(missing)
        monkeypatch.chdir(JOBS.parent.parent.parent)
        completed = subprocess.run(
            ["bash", str(SUBMIT), "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves", "wave-a"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert completed.returncode != 0, (missing, completed.stdout)
        assert missing in completed.stderr


def test_submit_refuses_an_unknown_script(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    completed = _submit(
        tmp_path, monkeypatch, "--script", "deploy", "gs://YOUR-BUCKET/rc0", "gs://YOUR-BUCKET/waves", "wave-a"
    )
    assert completed.returncode == 2
    assert "bootstrap or wave0" in completed.stderr
