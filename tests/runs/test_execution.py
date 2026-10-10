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
    reason: str | None = None
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

    def note(self, handle: str) -> str | None:
        return _Scheduler.reason


class _PodScheduler(_Scheduler):
    """A runner whose jobs do not see this host's files (as Kubernetes): the run reaches them through its mirror."""

    run_root = "/scratch/runs"


@pytest.fixture
def scheduler(monkeypatch: pytest.MonkeyPatch) -> type[_Scheduler]:
    _Scheduler.submitted, _Scheduler.cancelled, _Scheduler.state = [], [], "pending"
    _Scheduler.reason = None
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

    def test_a_stuck_job_note_reaches_run_status(self, data: Path, tmp_path: Path, scheduler) -> None:
        """An unschedulable pod is reported pending; the scheduler's own reason is the run's note."""
        started = _submit(_config(data, tmp_path), tmp_path)
        scheduler.reason = "0/8 nodes are available: 8 Insufficient nvidia.com/gpu."
        state = _ok("run", "status", "--run", started["run_dir"])
        assert state["jobs"][0]["status"] == "pending"
        assert "Insufficient nvidia.com/gpu" in (state["note"] or "")

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

    def test_a_runner_error_that_is_not_typed_falls_back_with_a_note(
        self, data: Path, tmp_path: Path, scheduler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plugin runner raises what it raises: an ``OSError`` (or a ``ValueError`` from a damaged session
        file) once turned ``run status`` into an INTERNAL error."""
        started = _submit(_config(data, tmp_path), tmp_path)

        def broken(self, handle: str) -> str:
            raise OSError("the scheduler's tool is not there")

        monkeypatch.setattr(_Scheduler, "status", broken)
        state = _ok("run", "status", "--run", started["run_dir"])
        assert state["jobs"][0]["status"] == "unknown" and "the scheduler's tool is not there" in state["note"]

    def test_a_mirror_client_error_falls_back_with_a_note(
        self, data: Path, tmp_path: Path, scheduler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``gcsfs.retry.HttpError`` (a 403, an expired credential) is not an ``OSError``: it once escaped the
        handler and made ``run status`` abort instead of reporting the local state."""
        from rcp_ndcg.runs.mirror import Mirror

        started = _submit(_config(data, tmp_path, mirror="gs://bucket/run", runner={"name": "sched"}), tmp_path)

        def refuse(self, relative: str) -> bytes:
            raise RuntimeError("credential refresh failed, 403")

        monkeypatch.setattr(Mirror, "read", refuse)
        state = _ok("run", "status", "--run", started["run_dir"])
        assert "could not be read" in state["note"] and "403" in state["note"]

    def test_a_partial_manifest_with_an_unknown_job_is_not_done(self, data: Path, tmp_path: Path, scheduler) -> None:
        """A phase boundary leaves the manifest ``partial``; when the runner cannot say whether the job ended
        (a missing accounting CLI, a TTL-deleted Job), ``done`` must not be true while the next phase may run."""
        from rcp_ndcg.runs.manifest import RunStatus

        started = _submit(_config(data, tmp_path), tmp_path)
        run = Run(started["run_dir"])
        manifest = run.manifest
        manifest.status = RunStatus.PARTIAL
        manifest.save(run.layout)
        scheduler.state = "unknown"
        state = _ok("run", "status", "--run", started["run_dir"])
        assert (state["status"], state["done"]) == ("partial", False)
        assert "cannot say" in state["note"]

    def test_run_status_does_not_adopt_another_runs_mirror_manifest(
        self, data: Path, tmp_path: Path, scheduler
    ) -> None:
        """B9's run scope must hold for ``run status`` too: a shared mirror prefix once made it report the other
        run's id, status and metrics, while ``restore`` refused the same mirror."""
        from datetime import UTC, datetime, timedelta

        from rcp_ndcg.runs.manifest import RunManifest

        remote = "memory://mirror/foreign"
        started = _submit(_config(data, tmp_path, mirror=remote, runner={"name": "sched"}), tmp_path)
        run = Run(started["run_dir"])
        foreign = run.manifest.model_copy(
            update={"run_id": "20260101-000000-other", "updated_at": datetime.now(UTC) + timedelta(days=1)}
        )
        storage.write_bytes(f"{remote}/manifest.json", foreign.model_dump_json().encode("utf-8"))
        state = _ok("run", "status", "--run", started["run_dir"])
        assert state["run_id"] == run.layout.run_id
        assert "holds the run" in state["note"] and "20260101-000000-other" in state["note"]
        assert RunManifest.load(run.layout).run_id == run.layout.run_id, "run status writes nothing"


class TestJobRecord:
    def test_a_torn_job_record_is_a_typed_error_naming_the_file(self, data: Path, tmp_path: Path, scheduler) -> None:
        """A torn ``logs/jobs.json`` once made run status, run logs and run cancel fail with a hint that named
        no file, forever."""
        started = _submit(_config(data, tmp_path), tmp_path)
        jobs = Path(Run(started["run_dir"]).layout.jobs)
        jobs.write_text('{"runner": "sched", "jobs": [', encoding="utf-8")
        code, error = _invoke("run", "status", "--run", started["run_dir"])
        assert code == 12 and "jobs.json" in error["message"]

    def test_a_job_record_of_the_wrong_shape_is_a_typed_error(self, data: Path, tmp_path: Path, scheduler) -> None:
        """Valid JSON of the wrong shape (``{}``, ``[]``, ``null``, a job without a handle) once reached the
        callers as a ``KeyError``/``AttributeError`` (INTERNAL) instead of the typed DATA error."""
        started = _submit(_config(data, tmp_path), tmp_path)
        jobs = Path(Run(started["run_dir"]).layout.jobs)
        for payload in ("{}", "[]", "null", '{"runner": "sched", "jobs": [{"name": "j"}]}'):
            jobs.write_text(payload, encoding="utf-8")
            code, error = _invoke("run", "status", "--run", started["run_dir"])
            assert code == 12 and "jobs.json" in error["message"], payload

    def test_the_job_record_is_written_atomically(
        self, data: Path, tmp_path: Path, scheduler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The record is rewritten on every submission; a plain ``write_text`` truncates first, so a reader
        racing it saw a partial JSON."""
        published: list[str] = []
        real = storage.publish_bytes

        def record(target, payload: bytes, *, mode: int | None = None) -> None:
            published.append(str(target))
            real(target, payload, mode=mode)

        monkeypatch.setattr("rcp_ndcg.runs.execution.publish_bytes", record)
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

    def test_a_runner_that_cannot_report_a_job_blocks_a_resubmission(
        self, data: Path, tmp_path: Path, scheduler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The live-job guard must refuse when the runner cannot say whether the job ended: an untyped
        ``OSError`` once escaped as INTERNAL instead of the typed config error."""
        started = _submit(_config(data, tmp_path, runner={"name": "sched"}), tmp_path)

        def broken(self, handle: str) -> str:
            raise OSError("the runner's tool is not there")

        monkeypatch.setattr(_Scheduler, "status", broken)
        code, error = _invoke("run", "resume", "--run", started["run_dir"], "--runner", "sched")
        assert code == 3 and "could not report" in error["message"]


class TestCancel:
    def test_cancel_of_a_handle_less_record_says_it_may_be_live(self, data: Path, tmp_path: Path, scheduler) -> None:
        """A record with no handle was reported as 'never submitted ... the run is not running' -- a claim the
        record cannot make when the submission was in flight."""
        started = _submit(_config(data, tmp_path, runner={"name": "sched"}), tmp_path)
        run = Run(started["run_dir"])
        record = run.jobs()
        record["jobs"][0]["handle"] = None
        record["submitting"] = True
        Path(run.layout.jobs).write_text(json.dumps(record), encoding="utf-8")
        code, error = _invoke("run", "cancel", "--run", started["run_dir"])
        assert code == 4 and "may be live" in error["message"]

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

    def test_a_runner_that_cannot_report_a_job_makes_cancel_a_typed_error(
        self, data: Path, tmp_path: Path, scheduler, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plugin runner raising ``OSError`` once made ``run cancel`` exit 1 INTERNAL (\"this is a bug\")
        instead of reporting that the job could not be asked about."""
        started = _submit(_config(data, tmp_path, runner={"name": "sched"}), tmp_path)

        def broken(self, handle: str) -> str:
            raise OSError("the runner's tool is not there")

        monkeypatch.setattr(_Scheduler, "status", broken)
        code, error = _invoke("run", "cancel", "--run", started["run_dir"])
        assert code == 4 and "could not report" in error["message"]


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


def test_the_job_fields_of_another_runner_are_refused_not_dropped(data: Path, tmp_path: Path, scheduler) -> None:
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
    # All three fields are named, and the check covers every one of them (a fourth would need this list).
    from rcp_ndcg.runs.execution import JOB_OPTIONS

    assert set(JOB_OPTIONS) == {"resources", "image", "env"}
    slurm = tiny_config(
        data,
        runner={
            "name": "slurm",
            "options": {
                "container_runtime": "pyxis",
                "env": {"HF_HOME": "/hf"},
                "image": "registry.example.com/rcp:v1",
                "resources": {"gpus": 2},
            },
        },
    )
    with pytest.raises(ConfigError, match="env, image, resources"):
        job_for(prepare(slurm, runs_dir=str(tmp_path / "runs")), "kubernetes")
    # A malformed resources value counts too: silently dropping it would hide the typo, not just the field.
    bad = tiny_config(data, runner={"name": "mine", "options": {"resources": "2"}})
    with pytest.raises(ConfigError, match="resources"):
        job_for(prepare(bad, runs_dir=str(tmp_path / "runs")), "sched")
    # An in-memory default Resources() instance is no more a declaration than its dumped {"gpus": 0}.
    from rcp_ndcg.runners import Resources

    default_instance = tiny_config(data, runner={"name": "mine", "options": {"resources": Resources()}})
    _, default_job, _ = job_for(prepare(default_instance, runs_dir=str(tmp_path / "runs")), "sched")
    assert default_job.resources == Resources()


def test_a_submission_error_is_recorded_with_url_credentials_redacted(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scheduler error names the request's URL; a credential in it must not reach ``logs/jobs.json`` (host-local,
    never mirrored) or ``run status``'s note."""
    from rcp_ndcg.runners.base import RunnerError

    class _Refusing:
        def __init__(self, **options) -> None:
            self.options = options

        def submit(self, jobs):
            raise RunnerError("the store refused s3://key:secret@bucket/runs/x")

    monkeypatch.setattr("rcp_ndcg.runs.execution.get_runner", lambda name, **options: _Refusing(**options))
    code, error = _invoke(
        "run", "start", str(_config(data, tmp_path)), "--runs-dir", str(tmp_path / "runs"), "--runner", "sched"
    )
    assert code == 6, error
    (run_dir,) = (tmp_path / "runs").iterdir()
    record = json.loads((run_dir / "logs" / "jobs.json").read_text(encoding="utf-8"))
    assert "key:secret" not in json.dumps(record) and "bucket/runs/x" in record["error"]
    state = _ok("run", "status", "--run", str(run_dir))
    assert "key:secret" not in (state["note"] or "")


def test_the_run_directory_and_the_records_the_mirror_uploads_are_owner_only(
    data: Path, tmp_path: Path, scheduler
) -> None:
    """On a shared cluster filesystem every user could read the config, the job record and the mirror state."""
    import stat as stat_module

    started = _submit(_config(data, tmp_path), tmp_path)
    run_dir = Path(started["run_dir"])
    assert stat_module.S_IMODE(run_dir.stat().st_mode) == 0o700
    assert stat_module.S_IMODE((run_dir / "logs").stat().st_mode) == 0o700
    for name in ("run.yaml", "manifest.json", "logs/jobs.json"):
        assert stat_module.S_IMODE((run_dir / name).stat().st_mode) == 0o600, name
