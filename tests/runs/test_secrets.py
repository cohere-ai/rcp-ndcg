"""Secrets never reach a recorded or uploaded file in clear, and a literal secret value is refused.

The package's rule is that credentials are read from the environment at use time and never written down
(``docs/concepts/judges.md``). A run config's ``runner.options.env`` and ``serve.<role>.env`` used to keep their
values verbatim in ``run.yaml``/``manifest.json`` -- both mirrored to the object store -- and a mirror URI's
userinfo reached logs, the state file and ``run status``.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from rcp_ndcg.runners import JobOptions, ServeConfig
from rcp_ndcg.runs import Pipeline, RunConfig
from rcp_ndcg.runs.execution import _write_record
from rcp_ndcg.runs.layout import RunLayout
from rcp_ndcg.runs.mirror import Mirror, read_state, restore
from tests.runs.conftest import STEPS, tiny_config

SECRET = "hf_SUPERSECRET_0123456789"


def _config(**fields) -> RunConfig:
    return RunConfig.model_validate({"dataset": "jsonl:rows.jsonl", "judge": "fake", "steps": STEPS, **fields})


def test_a_literal_secret_in_runner_options_is_refused_with_a_hint() -> None:
    with pytest.raises(ValidationError) as caught:
        _config(runner={"name": "slurm", "options": {"env": {"HF_TOKEN": SECRET}}})
    message = str(caught.value)
    assert "HF_TOKEN" in message
    assert "secret" in message.lower() and ("environment" in message.lower() or "secrets" in message.lower())


@pytest.mark.parametrize(
    "name",
    ["HF_TOKEN", "OPENAI_API_KEY", "MY_SECRET", "DB_PASSWORD", "SERVICE_PASSWD", "GOOGLE_APPLICATION_CREDENTIALS"],
)
def test_a_secret_looking_name_is_refused_in_a_job_and_an_engine_environment(name: str) -> None:
    with pytest.raises(ValidationError, match="secret"):
        JobOptions(env={name: SECRET})
    with pytest.raises(ValidationError, match="secret"):
        ServeConfig(image="vllm/vllm-openai:v0.31.0", command=("vllm", "serve", "m"), env={name: SECRET})


@pytest.mark.parametrize("name", ["HF_HOME", "HF_HUB_OFFLINE", "UV_CACHE_DIR", "MONKEY", "KEYSTONE", "TOKENIZER"])
def test_a_name_that_only_looks_secret_by_substring_is_allowed(name: str) -> None:
    """``MONKEY`` and ``KEYSTONE`` hold "key"; the rule is a word, not a substring, so ordinary names pass."""
    assert JobOptions(env={name: "1"}).env == {name: "1"}


def test_the_recorded_config_redacts_a_secret_that_reached_it_by_another_route() -> None:
    """A plugin runner's options are free-form, so the recording path redacts rather than trusts the boundary;
    the live config keeps the value the operator wrote (the job runs with it)."""
    config = _config(runner={"name": "mine", "options": {"env": {"HF_TOKEN": SECRET, "HF_HOME": "/cache"}}})
    recorded = config.recorded()
    assert recorded["runner"]["options"]["env"] == {"HF_TOKEN": "<redacted>", "HF_HOME": "/cache"}
    assert SECRET not in json.dumps(recorded)
    assert config.resolved()["runner"]["options"]["env"]["HF_TOKEN"] == SECRET


def test_run_yaml_and_the_manifest_never_record_a_secret(data: Path, tmp_path: Path) -> None:
    config = tiny_config(data, runner={"name": "mine", "options": {"env": {"HF_TOKEN": SECRET, "HF_HOME": "/cache"}}})
    pipeline = Pipeline(config, runs_dir=str(tmp_path / "runs"))
    pipeline.layout.ensure()
    pipeline._write_config()
    pipeline.manifest.save(pipeline.layout)
    run_yaml = Path(pipeline.layout.config).read_text(encoding="utf-8")
    manifest = Path(pipeline.layout.manifest).read_text(encoding="utf-8")
    assert SECRET not in run_yaml and "<redacted>" in run_yaml
    assert SECRET not in manifest and "<redacted>" in manifest
    assert yaml.safe_load(run_yaml)["runner"]["options"]["env"]["HF_HOME"] == "/cache"


def test_a_mirror_uri_is_recorded_redacted_and_the_live_config_keeps_it() -> None:
    """Only the written form redacts: ``prepare`` builds the pipeline from :meth:`resolved`, so the job's argv
    and the submitting host's flush keep the credential the store needs."""
    config = _config(mirror="s3://key:secret@bucket/runs/x")
    assert config.resolved()["mirror"] == "s3://key:secret@bucket/runs/x"
    assert config.recorded()["mirror"] == "s3://bucket/runs/x"


def test_the_job_argv_keeps_the_full_mirror_uri(data: Path, tmp_path: Path) -> None:
    """The live pipeline and the job's argv carry the credential; only the recorded copy redacts it."""
    from rcp_ndcg.runs.execution import job_for
    from rcp_ndcg.runs.run import prepare

    config = tiny_config(data, mirror="memory://key:secret@bucket/runs/x")
    pipeline = prepare(config, runs_dir=str(tmp_path / "runs"))
    assert pipeline.config.mirror == "memory://key:secret@bucket/runs/x"
    _, job, _ = job_for(pipeline, "local")
    commands = [phase.argv for phase in job.phases] if job.phases else [job.argv]
    assert any("memory://key:secret@bucket/runs/x" in argv for argv in commands)
    assert config.recorded()["mirror"] == "memory://bucket/runs/x"


def test_the_mirror_state_logs_and_status_redact_credentials(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    remote = "memory://key:secret@mirror/run"
    state_file = tmp_path / "logs" / "mirror.json"
    mirror = Mirror(tmp_path, remote, state_file=state_file)
    with caplog.at_level("INFO"):
        mirror.flush()
    assert read_state(state_file).remote == "memory://mirror/run"
    assert "key:secret" not in state_file.read_text(encoding="utf-8")
    assert "key:secret" not in caplog.text


def test_the_restore_log_redacts_credentials(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    remote = "memory://key:secret@mirror/run"
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    Mirror(tmp_path, remote).flush()
    (tmp_path / "manifest.json").unlink()
    with caplog.at_level("INFO"):
        restore(tmp_path, remote)
    assert "key:secret" not in caplog.text


def test_a_restore_temp_with_an_old_mode_does_not_leak_it(tmp_path: Path) -> None:
    """The restore temp path is predictable; a stale one from a killed restore must not hand its 0644 to the
    restored record (``O_CREAT``'s mode applies only when it creates the file)."""
    remote = "memory://mirror/run"
    (tmp_path / "manifest.json").write_text('{"v": 1}', encoding="utf-8")
    Mirror(tmp_path, remote).flush()
    (tmp_path / "manifest.json").unlink()
    stale = tmp_path / "manifest.json.restore.tmp"
    stale.write_text("old", encoding="utf-8")
    stale.chmod(0o644)
    restore(tmp_path, remote)
    assert stat.S_IMODE((tmp_path / "manifest.json").stat().st_mode) == 0o600


def test_a_valid_state_files_error_is_redacted(tmp_path: Path) -> None:
    """A state file written before the redaction keeps a flush error naming the credentialed URI; ``run status``
    reads it through ``read_state``, so the error is redacted there too."""
    state_file = tmp_path / "mirror.json"
    state_file.write_text(
        '{"remote": "memory://bucket/x", "last_upload_at": "2026-01-01T00:00:00+00:00", "last_error": '
        '"DataError: flush to memory://key:secret@mirror/run failed"}',
        encoding="utf-8",
    )
    state = read_state(state_file)
    assert state is not None and "key:secret" not in (state.last_error or "")


def test_a_torn_state_file_warning_redacts_credentials(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A state file written before the redaction still holds the full URI; a torn one must not print it."""
    state = tmp_path / ".mirror.json"
    state.write_text('{"remote":"memory://key:secret@mirror/run","x":', encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert read_state(state) is None
    assert "key:secret" not in caplog.text
    assert "torn" in caplog.text


def test_a_run_directory_and_its_records_are_owner_only(tmp_path: Path) -> None:
    """A SLURM cluster shares the filesystem: the run directory must not be world-traversable, and the files the
    mirror uploads (the config, the job record, the mirror state) are written owner-only."""
    layout = RunLayout.at(tmp_path / "runs" / "r1").ensure()
    assert stat.S_IMODE(Path(layout.root).stat().st_mode) == 0o700
    assert stat.S_IMODE(Path(layout.logs_dir).stat().st_mode) == 0o700
    _write_record(layout, {"runner": "slurm", "options": {}, "jobs": []})
    assert stat.S_IMODE(Path(layout.jobs).stat().st_mode) == 0o600
    Path(layout.config).write_text("mirror: null\n", encoding="utf-8")
    Mirror(layout.root, "memory://mirror/r1", state_file=layout.mirror_state).flush()
    assert stat.S_IMODE(Path(layout.mirror_state).stat().st_mode) == 0o600


def test_an_existing_run_directory_is_tightened(tmp_path: Path) -> None:
    """A run created before the owner-only rule must not stay world-traversable when it is resumed."""
    root = tmp_path / "runs" / "r1"
    root.mkdir(parents=True)
    root.chmod(0o755)
    RunLayout.at(root).ensure()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
