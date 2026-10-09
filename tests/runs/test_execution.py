"""A run handed to a job runner: absolute paths, status when the job ends, cancel, resubmission, a failed submit."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from rcp_ndcg import storage
from rcp_ndcg.cli.main import cli
from rcp_ndcg.runs.run import Run
from tests.runs.conftest import tiny_config


def _invoke(*args: str) -> tuple[int, dict]:
    result = CliRunner().invoke(cli, [*args, "--json"])
    document = json.loads(result.stdout)
    return result.exit_code, document.get("data") if result.exit_code == 0 else document["error"]


def _ok(*args: str) -> dict:
    code, data = _invoke(*args)
    assert code == 0, data
    return data


class _Scheduler:
    """A job runner whose jobs' status the test sets; it records what it was handed and runs nothing."""

    submitted: list[tuple[str, ...]] = []
    cancelled: list[str] = []
    state = "pending"
    run_root: str | None = None

    def __init__(self, **options) -> None:
        self.options = options

    def submit(self, jobs) -> list[str]:
        _Scheduler.submitted.extend(job.argv for job in jobs)
        return [f"h{len(_Scheduler.submitted)}"]

    def status(self, handle: str) -> str:
        return _Scheduler.state

    def logs(self, handle: str, *, tail: int | None = None) -> str:
        return ""

    def cancel(self, handle: str) -> None:
        _Scheduler.cancelled.append(handle)
        _Scheduler.state = "cancelled"


class _PodScheduler(_Scheduler):
    """A runner whose jobs do not see this host's files (as Kubernetes): the run reaches them through its mirror."""

    run_root = "/scratch/runs"


@pytest.fixture
def scheduler(monkeypatch: pytest.MonkeyPatch) -> type[_Scheduler]:
    _Scheduler.submitted, _Scheduler.cancelled, _Scheduler.state = [], [], "pending"
    runners = {"sched": _Scheduler, "pod": _PodScheduler}

    def get_runner(name, **options):
        return runners[name](**options)

    monkeypatch.setattr("rcp_ndcg.runs.execution.get_runner", get_runner)
    return _Scheduler


@pytest.fixture(autouse=True)
def _empty_memory_fs():
    memory = storage.filesystem("memory://")
    memory.store.clear()
    yield
    memory.store.clear()


def _config(data: Path, tmp_path: Path, **fields) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(tiny_config(data, **fields).resolved()), encoding="utf-8")
    return path


def _submit(config: Path, tmp_path: Path, runner: str = "sched", *extra: str) -> dict:
    return _ok("run", "start", str(config), "--runs-dir", str(tmp_path / "runs"), "--runner", runner, *extra)


class TestPaths:
    def test_every_path_a_job_or_its_record_names_is_absolute(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A relative run directory reached the job's argv: after `cd <workdir>` the job found no manifest."""
        monkeypatch.chdir(tmp_path)
        slurm = {"name": "slurm", "options": {"workdir": "/shared/work", "log_dir": "slurm-logs"}}
        plan = _ok("run", "start", str(_config(data, tmp_path, runner=slurm)), "--runs-dir", "runs", "--dry-run")
        (script,) = plan["rendered"].values()
        assert f"--run {tmp_path / 'runs'}/" in script and "cd /shared/work" in script
        assert f"#SBATCH --output={tmp_path / 'slurm-logs'}/%x-%j.out" in script

    def test_a_detached_run_is_followed_and_cancelled_from_any_directory(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """From another directory `run status` said 'unknown', `run logs` failed, and `run cancel` killed nothing
        but still recorded the run as cancelled."""
        monkeypatch.setattr("rcp_ndcg.runs.execution.run_argv", lambda *args, **kwargs: ("sleep", "30"))
        monkeypatch.chdir(tmp_path)
        started = _ok("run", "start", str(_config(data, tmp_path)), "--runs-dir", "runs", "--detach")
        record = Run(started["run_dir"]).jobs()
        assert all(Path(record["options"][key]).is_absolute() for key in ("log_dir", "cwd"))

        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        run_dir = str(tmp_path / started["run_dir"])
        assert _ok("run", "status", "--run", run_dir)["jobs"][0]["status"] in ("pending", "running")
        assert _ok("run", "logs", "--run", run_dir)["text"] == ""
        cancelled = _ok("run", "cancel", "--run", run_dir)
        assert (cancelled["status"], cancelled["done"], cancelled["jobs"][0]["status"]) == (
            "cancelled",
            True,
            "cancelled",
        )
        session = int(Path(record["options"]["log_dir"], f"{record['jobs'][0]['name']}.session").read_text())
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                os.killpg(session, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            pytest.fail("the job still runs after run cancel")

    def test_a_job_the_runner_cannot_find_is_not_recorded_cancelled(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("rcp_ndcg.runs.execution.run_argv", lambda *args, **kwargs: ("sleep", "30"))
        started = _ok("run", "start", str(_config(data, tmp_path)), "--runs-dir", str(tmp_path / "runs"), "--detach")
        record = Run(started["run_dir"]).jobs()
        name = record["jobs"][0]["name"]
        session = Path(record["options"]["log_dir"], f"{name}.session")
        pid = int(session.read_text())
        session.unlink()  # its runner can no longer find it
        try:
            code, error = _invoke("run", "cancel", "--run", started["run_dir"])
            assert code == 4 and "cannot find" in error["message"]
            assert Run(started["run_dir"]).manifest.status.value == "submitted"
        finally:
            os.killpg(pid, 15)


class TestStatus:
    @pytest.mark.parametrize(
        ("job", "derived"), [("failed", "failed"), ("cancelled", "cancelled"), ("completed", "failed")]
    )
    def test_a_run_whose_job_ended_without_recording_it_is_done(
        self, data: Path, tmp_path: Path, scheduler, job: str, derived: str
    ) -> None:
        """An engine that died before readiness left the run 'submitted', done false, forever."""
        started = _submit(_config(data, tmp_path), tmp_path)
        assert (started["state"]["status"], started["state"]["done"]) == ("submitted", False)
        scheduler.state = job
        state = _ok("run", "status", "--run", started["run_dir"])
        assert (state["status"], state["done"]) == (derived, True)
        assert "every job of the run has ended" in state["note"]
        assert Run(started["run_dir"]).manifest.status.value == "submitted", "run status writes nothing"

    def test_a_pod_run_is_read_from_its_mirror_and_restored_from_it(
        self, data: Path, tmp_path: Path, scheduler
    ) -> None:
        """The pod completed the run in the mirror; here it stayed 'submitted', and a restore kept the old manifest."""
        remote = "memory://mirror/pod-run"
        storage.filesystem("memory://").pipe("/data/rows.jsonl", data.read_bytes())
        config = _config(data, tmp_path, mirror=remote, runner={"name": "pod"})
        rows = yaml.safe_load(config.read_text())
        config.write_text(yaml.safe_dump({**rows, "dataset": "jsonl:memory://data/rows.jsonl"}))
        started = _submit(config, tmp_path, "pod")
        assert started["state"]["mirror"]["last_upload_at"] is not None, "the submitter's upload shows"

        pod = tmp_path / "pod" / Path(started["run_dir"]).name  # what the job does in its pod
        _ok("run", "resume", "--run", str(pod), "--mirror", remote)
        scheduler.state = "completed"

        state = _ok("run", "status", "--run", started["run_dir"])
        assert (state["status"], state["done"]) == ("completed", True) and "mirror" in state["note"]
        assert {step["status"] for step in state["steps"]} == {"completed"}
        _ok("run", "resume", "--run", started["run_dir"], "--mirror", remote, "--dry-run")
        assert Run(started["run_dir"]).manifest.status.value == "completed"

    def test_a_submission_that_fails_leaves_a_failed_run_saying_why(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """sbatch missing left a 'submitted' run with no handle, and was reported retryable."""
        monkeypatch.setenv("PATH", str(tmp_path / "no-tools"))
        # The runner creates its log_dir before it calls sbatch; keep it under tmp_path, never the checkout.
        slurm = {"name": "slurm", "options": {"log_dir": str(tmp_path / "slurm")}}
        code, error = _invoke(
            "run",
            "start",
            str(_config(data, tmp_path, runner=slurm)),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--runner",
            "slurm",
        )
        assert (code, error["retryable"]) == (6, False) and "sbatch" in error["message"]
        (run_dir,) = (tmp_path / "runs").iterdir()
        assert Run(run_dir).manifest.status.value == "failed"
        state = _ok("run", "status", "--run", str(run_dir))
        assert state["done"] is True and "submission failed" in state["note"]

    def test_a_live_job_keeps_the_run_not_done_between_phases(self, data: Path, tmp_path: Path, scheduler) -> None:
        """A job's phases run ``run resume --only <steps>``; after phase 1 the manifest is ``partial``
        (terminal), so a poller read ``done=true`` while the job still ran the next phase."""
        from rcp_ndcg.runs.manifest import RunStatus

        started = _submit(_config(data, tmp_path), tmp_path)
        run = Run(started["run_dir"])
        manifest = run.manifest
        manifest.status = RunStatus.PARTIAL
        manifest.save(run.layout)
        scheduler.state = "running"
        state = _ok("run", "status", "--run", started["run_dir"])
        assert (state["status"], state["done"]) == ("partial", False)
        assert "still running" in state["note"]

    def test_a_job_the_runner_cannot_find_is_reported_with_a_note(self, data: Path, tmp_path: Path, scheduler) -> None:
        """An unmapped SLURM state (or a TTL-deleted Job) left the run 'submitted' with note null forever."""
        started = _submit(_config(data, tmp_path), tmp_path)
        scheduler.state = "unknown"
        state = _ok("run", "status", "--run", started["run_dir"])
        assert (state["status"], state["done"], state["jobs"][0]["status"]) == ("submitted", False, "unknown")
        assert "unknown" in state["note"]

    def test_a_runner_that_cannot_report_a_job_falls_back_with_a_note(
        self, data: Path, tmp_path: Path, scheduler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing accounting CLI (``sacct``) once aborted ``run status`` with a provider error instead of
        reporting the job it cannot ask about."""
        from rcp_ndcg.runners import RunnerError

        started = _submit(_config(data, tmp_path), tmp_path)

        def missing(self, handle: str) -> str:
            raise RunnerError("sacct: command not found")

        monkeypatch.setattr(_Scheduler, "status", missing)
        state = _ok("run", "status", "--run", started["run_dir"])
        assert state["jobs"][0]["status"] == "unknown" and "sacct: command not found" in state["note"]


class TestJobRecord:
    def test_a_torn_job_record_is_a_typed_error_naming_the_file(
        self, data: Path, tmp_path: Path, scheduler
    ) -> None:
        """A torn ``logs/jobs.json`` once made run status, run logs and run cancel fail with a hint that named
        no file, forever."""
        started = _submit(_config(data, tmp_path), tmp_path)
        jobs = Path(Run(started["run_dir"]).layout.jobs)
        jobs.write_text('{"runner": "sched", "jobs": [', encoding="utf-8")
        code, error = _invoke("run", "status", "--run", started["run_dir"])
        assert code == 12 and "jobs.json" in error["message"]

    def test_the_job_record_is_written_atomically(
        self, data: Path, tmp_path: Path, scheduler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The record is rewritten on every submission; a plain ``write_text`` truncates first, so a reader
        racing it saw a partial JSON."""
        published: list[str] = []
        real = storage.publish_bytes

        def record(target, payload: bytes) -> None:
            published.append(str(target))
            real(target, payload)

        monkeypatch.setattr(storage, "publish_bytes", record)
        started = _submit(_config(data, tmp_path), tmp_path)
        assert str(Run(started["run_dir"]).layout.jobs) in published


class TestResubmission:
    def test_a_run_whose_job_failed_is_submitted_again_with_its_runner(
        self, data: Path, tmp_path: Path, scheduler
    ) -> None:
        """`run resume` only ran in this process (no engine), and rcp.run(dir, runner=...) was refused."""
        import rcp_ndcg

        started = _submit(_config(data, tmp_path, runner={"name": "sched", "options": {"queue": "a"}}), tmp_path)
        running = _invoke("run", "resume", "--run", started["run_dir"], "--runner", "sched")
        assert running[0] == 3 and "still pending" in running[1]["message"]  # its job has not ended
        for flag in (("--only", "tournament"), ("--no-resume",)):
            assert _invoke("run", "resume", "--run", started["run_dir"], "--runner", "sched", *flag)[0] == 2

        scheduler.state = "failed"
        again = _ok("run", "resume", "--run", started["run_dir"], "--runner", "sched")
        assert again["mode"] == "submitted" and again["run_dir"] == started["run_dir"]
        assert len(scheduler.submitted) == 2 and scheduler.submitted[1] == scheduler.submitted[0]
        assert Run(started["run_dir"]).jobs() == {**Run(started["run_dir"]).jobs(), "options": {"queue": "a"}}
        assert Run(started["run_dir"]).manifest.status.value == "submitted"

        scheduler.state = "failed"
        assert rcp_ndcg.run(started["run_dir"], runner="sched").manifest.status.value == "submitted"
        assert len(scheduler.submitted) == 3

    def test_a_null_handle_blocks_a_resubmission(self, data: Path, tmp_path: Path, scheduler) -> None:
        """A record with ``handle: null`` was invisible to the live-job guard: a second job could start over a
        live one."""
        started = _submit(_config(data, tmp_path, runner={"name": "sched"}), tmp_path)
        run = Run(started["run_dir"])
        record = run.jobs()
        record["jobs"][0]["handle"] = None
        Path(run.layout.jobs).write_text(json.dumps(record), encoding="utf-8")
        code, error = _invoke("run", "resume", "--run", started["run_dir"], "--runner", "sched")
        assert code == 3 and "handle" in error["message"]

    def test_a_submission_that_never_recorded_its_handle_is_refused(
        self, data: Path, tmp_path: Path, scheduler
    ) -> None:
        """A process killed between ``submit`` and the record rewrite left a handle-less record that was
        indistinguishable from a failed submission."""
        started = _submit(_config(data, tmp_path, runner={"name": "sched"}), tmp_path)
        run = Run(started["run_dir"])
        record = run.jobs()
        record["submitting"] = True
        Path(run.layout.jobs).write_text(json.dumps(record), encoding="utf-8")
        code, error = _invoke("run", "resume", "--run", started["run_dir"], "--runner", "sched")
        assert code == 3 and "never recorded its handle" in error["message"]

    def test_a_submission_that_failed_before_a_handle_can_be_resubmitted(
        self, data: Path, tmp_path: Path, scheduler
    ) -> None:
        """A submission that failed (``sbatch`` missing) records the error and no handle: that run is not live
        and must stay resubmittable."""
        started = _submit(_config(data, tmp_path, runner={"name": "sched"}), tmp_path)
        run = Run(started["run_dir"])
        record = run.jobs()
        record["jobs"][0]["handle"] = None
        record["error"] = "RunnerError: sbatch: command not found"
        Path(run.layout.jobs).write_text(json.dumps(record), encoding="utf-8")
        scheduler.state = "failed"
        again = _ok("run", "resume", "--run", started["run_dir"], "--runner", "sched")
        assert again["mode"] == "submitted"


class TestCancel:
    def test_cancel_closes_the_steps_that_were_running(self, data: Path, tmp_path: Path, scheduler) -> None:
        started = _submit(_config(data, tmp_path), tmp_path)
        run = Run(started["run_dir"])
        manifest = run.manifest
        manifest.status = manifest.status.RUNNING
        manifest.start_step("tournament", identity={})
        manifest.save(run.layout)
        scheduler.state = "running"
        state = _ok("run", "cancel", "--run", started["run_dir"])
        assert state["status"] == "cancelled" and scheduler.cancelled == ["h1"]
        assert [(step["name"], step["status"]) for step in state["steps"]][0] == ("tournament", "cancelled")

    def test_a_job_that_already_ended_is_not_cancelled(self, data: Path, tmp_path: Path, scheduler) -> None:
        started = _submit(_config(data, tmp_path), tmp_path)
        scheduler.state = "failed"
        state = _ok("run", "cancel", "--run", started["run_dir"])
        assert scheduler.cancelled == [] and state["status"] == "failed"
        assert Run(started["run_dir"]).manifest.status.value == "submitted"


def test_stage_run_writes_the_run_directory_a_runner_is_handed(data: Path, tmp_path: Path) -> None:
    """The one staging of a run directory before a job runs it (``submit_run`` and any driver that runs the
    job itself): the layout, the resolved ``run.yaml`` and the manifest in status ``submitted``."""
    from rcp_ndcg.runs.execution import stage_run
    from rcp_ndcg.runs.manifest import RunStatus
    from rcp_ndcg.runs.run import prepare

    pipeline = prepare(tiny_config(data), runs_dir=str(tmp_path))
    layout = stage_run(pipeline)
    assert layout.root == pipeline.layout.root
    assert yaml.safe_load(Path(layout.config).read_text(encoding="utf-8")) == pipeline.config.resolved()
    assert Run(layout.root).manifest.status is RunStatus.SUBMITTED


def test_the_job_fields_of_another_runner_are_refused_not_dropped(
    data: Path, tmp_path: Path, scheduler
) -> None:
    """A config that names ``local`` and sets ``env``/``resources``, handed to ``slurm``, lost them silently:
    the job asked for no GPUs, no time limit and no environment."""
    from rcp_ndcg.errors import ConfigError
    from rcp_ndcg.runs.execution import job_for
    from rcp_ndcg.runs.run import prepare

    config = tiny_config(
        data,
        runner={"name": "local", "options": {"env": {"HF_HOME": "/shared/hf"}, "resources": {"gpus": 2}}},
    )
    with pytest.raises(ConfigError, match="job fields") as refused:
        job_for(prepare(config, runs_dir=str(tmp_path / "runs")), "slurm")
    assert "env, resources" in refused.value.message and "local" in refused.value.message
    # The runner the config names keeps its job fields (they are its own), and defaults are not a declaration.
    same = tiny_config(data, runner={"name": "sched", "options": {"env": {"HF_HOME": "/shared/hf"}}})
    _, job, _ = job_for(prepare(same, runs_dir=str(tmp_path / "runs")), "sched")
    assert dict(job.env) == {"HF_HOME": "/shared/hf"}
    plain = tiny_config(data)
    _, plain_job, _ = job_for(prepare(plain, runs_dir=str(tmp_path / "runs")), "sched")
    assert dict(plain_job.env) == {} and plain_job.resources.gpus == 0
