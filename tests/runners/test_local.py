"""LocalRunner: jobs run on this host, in order."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.runners import JobSpec, JobStatus, LocalRunner, RunnerError, ServeConfig, get_runner


def _py(code: str) -> tuple[str, ...]:
    return (sys.executable, "-c", code)


def test_jobs_run_in_order_and_report_success(tmp_path: Path) -> None:
    runner = LocalRunner()
    log = tmp_path / "order.txt"
    specs = [
        JobSpec(name="first", argv=_py(f"open(r'{log}', 'a').write('1')")),
        JobSpec(name="second", argv=_py(f"open(r'{log}', 'a').write('2')")),
    ]
    assert runner.submit(specs) == ["first", "second"]
    assert log.read_text() == "12"
    assert runner.status("first") is JobStatus.COMPLETED
    assert runner.status("never-submitted") is JobStatus.UNKNOWN


def test_a_failing_job_stops_the_submission(tmp_path: Path) -> None:
    runner = LocalRunner()
    marker = tmp_path / "after.txt"
    specs = [
        JobSpec(name="fail", argv=_py("raise SystemExit(7)")),
        JobSpec(name="after", argv=_py(f"open(r'{marker}', 'w').write('x')")),
    ]
    with pytest.raises(RunnerError, match="exited 7"):
        runner.submit(specs)
    assert runner.status("fail") is JobStatus.FAILED
    assert not marker.exists()


def test_a_job_sees_its_env(tmp_path: Path) -> None:
    out = tmp_path / "env.txt"
    code = f"import os; open(r'{out}', 'w').write(os.environ['WHO'] + os.environ['EXTRA'])"
    LocalRunner(env={"EXTRA": "+runner"}).submit([JobSpec(name="env", argv=_py(code), env={"WHO": "job"})])
    assert out.read_text() == "job+runner"


def test_logs_are_kept_only_with_a_log_dir(tmp_path: Path) -> None:
    runner = LocalRunner(log_dir=str(tmp_path / "logs"))
    runner.submit([JobSpec(name="talk", argv=_py("print('a'); print('b'); print('c')"))])
    assert runner.logs("talk") == "a\nb\nc\n"
    assert runner.logs("talk", tail=1) == "c\n"
    with pytest.raises(RunnerError, match="log_dir"):
        LocalRunner().logs("talk")


TERMINAL = frozenset({JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED})


def _wait(runner: LocalRunner, handle: str, *, until: frozenset[JobStatus] = TERMINAL) -> JobStatus:
    deadline = time.monotonic() + 60
    while (state := runner.status(handle)) not in until:
        assert time.monotonic() < deadline, f"{handle} still {state}"
        time.sleep(0.05)
    return state


class TestDetached:
    def test_submit_returns_at_once_and_any_runner_follows_the_jobs(self, tmp_path: Path) -> None:
        gate, done = tmp_path / "gate", tmp_path / "done.txt"
        wait_for_gate = f"import pathlib, time\nwhile not pathlib.Path(r'{gate}').exists(): time.sleep(0.02)"
        runner = LocalRunner(log_dir=str(tmp_path / "logs"), detach=True)
        specs = [
            JobSpec(name="first", argv=_py(f"{wait_for_gate}\nprint('first ran')")),
            JobSpec(name="second", argv=_py(f"open(r'{done}', 'w').write('2')")),
        ]
        assert runner.submit(specs) == ["first", "second"]

        follower = LocalRunner(log_dir=str(tmp_path / "logs"))  # e.g. `run status` in another process
        assert follower.status("first") in (JobStatus.PENDING, JobStatus.RUNNING)
        assert follower.status("second") is JobStatus.PENDING and not done.exists()
        gate.touch()
        assert _wait(follower, "second") is JobStatus.COMPLETED
        assert follower.status("first") is JobStatus.COMPLETED
        assert follower.logs("first") == "first ran\n" and done.read_text() == "2"

    def test_a_failing_job_fails_and_stops_the_rest(self, tmp_path: Path) -> None:
        runner = LocalRunner(log_dir=str(tmp_path / "logs"), detach=True)
        runner.submit([JobSpec(name="fail", argv=_py("raise SystemExit(3)")), JobSpec(name="after", argv=_py(""))])
        assert _wait(runner, "fail") is JobStatus.FAILED
        assert (tmp_path / "logs" / "fail.exit").read_text().strip() == "3"
        assert _wait(runner, "after", until=frozenset({JobStatus.FAILED})) is JobStatus.FAILED  # never started

    def test_cancel_stops_the_session(self, tmp_path: Path) -> None:
        runner = LocalRunner(log_dir=str(tmp_path / "logs"), detach=True)
        runner.submit([JobSpec(name="sleeper", argv=_py("import time; time.sleep(60)"))])
        _wait(runner, "sleeper", until=frozenset({JobStatus.RUNNING}))
        runner.cancel("sleeper")
        assert _wait(runner, "sleeper") is JobStatus.CANCELLED

    def test_cancel_escalates_to_sigkill_for_a_job_that_ignores_sigterm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A coordinator that traps or ignores SIGTERM once survived ``run cancel`` while the run was recorded
        ``cancelled``: there was no grace period, no SIGKILL and no check that the group died."""
        monkeypatch.setattr("rcp_ndcg.runners.local.STOP_GRACE_S", 1)
        runner = LocalRunner(log_dir=str(tmp_path / "logs"), detach=True)
        ready = tmp_path / "ready"
        code = (
            "import pathlib, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            f"pathlib.Path(r'{ready}').write_text('ready'); time.sleep(60)"
        )
        runner.submit([JobSpec(name="stubborn", argv=_py(code))])
        _wait(runner, "stubborn", until=frozenset({JobStatus.RUNNING}))
        deadline = time.monotonic() + 10
        while not ready.exists():
            assert time.monotonic() < deadline, "the job never installed its SIGTERM handler"
            time.sleep(0.02)
        session = int((tmp_path / "logs" / "stubborn.session").read_text())
        started = time.monotonic()
        runner.cancel("stubborn")
        assert time.monotonic() - started >= 1.0, "the group was not given its grace before SIGKILL"
        with pytest.raises(ProcessLookupError):
            os.killpg(session, 0)
        assert runner.status("stubborn") is JobStatus.CANCELLED

    def test_a_detached_runner_needs_a_log_dir(self) -> None:
        with pytest.raises(ConfigError, match="needs `log_dir`"):
            LocalRunner(detach=True)


def test_a_finished_job_is_known_to_a_later_runner(tmp_path: Path) -> None:
    LocalRunner(log_dir=str(tmp_path)).submit([JobSpec(name="ok", argv=_py(""))])
    assert LocalRunner(log_dir=str(tmp_path)).status("ok") is JobStatus.COMPLETED


def test_a_damaged_session_file_is_unknown_not_a_crash(tmp_path: Path) -> None:
    """A torn ``<name>.session`` once made ``status`` raise ``ValueError`` (INTERNAL) and ``cancel`` abort."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "job.session").write_text("not a pid\n", encoding="utf-8")
    runner = LocalRunner(log_dir=str(logs))
    assert runner.status("job") is JobStatus.UNKNOWN
    with pytest.raises(RunnerError, match="missing or damaged"):
        runner.cancel("job")


def test_a_session_file_of_pid_zero_or_one_is_never_signalled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A damaged or edited session file once made ``cancel`` signal an arbitrary process group: a numeric
    prefix was trusted, so ``0`` killed the cancelling process's own group and ``1`` every process of the host."""
    logs = tmp_path / "logs"
    logs.mkdir()
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    runner = LocalRunner(log_dir=str(logs))
    for payload in ("0", "1", "-5", ""):
        (logs / "job.session").write_text(payload, encoding="utf-8")
        assert runner.status("job") is JobStatus.UNKNOWN
        with pytest.raises(RunnerError, match="missing or damaged"):
            runner.cancel("job")
    assert calls == []


def test_render_is_the_script_that_runs(tmp_path: Path) -> None:
    spec = JobSpec(name="show", argv=("echo", "a b"), env={"K": "v w"})
    script = LocalRunner(cwd=str(tmp_path)).render([spec])["show"]
    assert script.splitlines()[-2:] == ["export K='v w'", "exec echo 'a b'"]


def test_a_phase_that_starts_an_engine_is_refused() -> None:
    """The local runner runs a phased job's phases (its commands) and refuses the ones with engines."""
    from rcp_ndcg.runners import JobPhase

    engine = ServeConfig(command=["vllm", "serve", "m"])
    phases = (
        JobPhase(engines={"encoder": engine}, argv=("echo", "first")),
        JobPhase(argv=("echo", "rest")),
    )
    with pytest.raises(ConfigError, match=r"role\(s\) encoder") as caught:
        LocalRunner().render([JobSpec(name="j", phases=phases)])
    assert "--engine" in (caught.value.hint or "")


def test_engine_free_phases_run_in_order(tmp_path: Path) -> None:
    """A phased job's commands are its phases' argv: the local runner runs each in order (its plan has no
    engines: the engine-free phases cover exactly the run's steps)."""
    from rcp_ndcg.runners import JobPhase

    phases = (JobPhase(argv=_py("print('one')")), JobPhase(argv=_py("print('two')")))
    job = JobSpec(name="phased", phases=phases)
    runner = LocalRunner(log_dir=str(tmp_path))
    runner.submit([job])
    assert runner.status("phased") is JobStatus.COMPLETED
    assert runner.logs("phased") == "one\ntwo\n"


def test_options_are_config_errors() -> None:
    with pytest.raises(ConfigError, match="invalid options for the 'local' runner"):
        LocalRunner(mode="container")
    with pytest.raises(ConfigError, match="invalid options for the 'local' runner"):
        get_runner("local", partition="gpu")
    with pytest.raises(ConfigError, match="unknown runner 'teleport'"):
        get_runner("teleport")


def test_a_plugin_runner_is_found_through_its_entry_point(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one lookup: every runner, built-in or plugin, is an ``rcp_ndcg.runners`` entry point."""
    from importlib.metadata import EntryPoint

    import rcp_ndcg.runners.registry as registry

    plugin = EntryPoint(name="teleport", value="tests.runners.test_local:TeleportRunner", group="rcp_ndcg.runners")
    installed = registry.entry_points(group="rcp_ndcg.runners")
    monkeypatch.setattr(registry, "entry_points", lambda group: [*installed, plugin])

    runner = get_runner("teleport", site="moon")

    assert isinstance(runner, TeleportRunner) and runner.site == "moon"
    assert isinstance(get_runner("local"), LocalRunner)


def test_a_duplicate_entry_point_name_never_shadows_a_built_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dict comprehension kept the last entry point: a plugin publishing ``local`` could replace the built-in
    (and receive the built-in's typed options) with no diagnostic."""
    from importlib.metadata import EntryPoint

    import rcp_ndcg.runners.registry as registry

    plugin = EntryPoint(name="local", value="tests.runners.test_local:TeleportRunner", group="rcp_ndcg.runners")
    installed = registry.entry_points(group="rcp_ndcg.runners")
    monkeypatch.setattr(registry, "entry_points", lambda group: [*installed, plugin])
    with pytest.raises(ConfigError, match="more than one") as refused:
        get_runner("local")
    assert "rename one entry point" in (refused.value.hint or "")
    # an installed name resolves to the shipped class: the public names are never shadowed
    monkeypatch.setattr(registry, "entry_points", lambda group: list(installed))
    assert isinstance(get_runner("local"), LocalRunner)


class TeleportRunner:
    name = "teleport"

    def __init__(self, *, site: str) -> None:
        self.site = site


def test_a_job_runs_in_the_callers_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A relative run directory (`--runs-dir runs`) is resolved where the command was typed, also in a checkout.
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "rcp-ndcg"\n', encoding="utf-8")
    here = tmp_path / "sub"
    here.mkdir()
    monkeypatch.chdir(here)
    LocalRunner().submit([JobSpec(name="where", argv=_py("import os; open('cwd.txt', 'w').write(os.getcwd())"))])
    assert (here / "cwd.txt").read_text() == str(here)


def test_a_failing_job_is_raised_as_its_own_exit_codes_error(tmp_path: Path) -> None:
    """An evaluate step that failed MissingInputError (exit 4) came back as a retryable PROVIDER error."""
    from rcp_ndcg.errors import MissingInputError

    runner = LocalRunner(log_dir=str(tmp_path / "logs"))
    with pytest.raises(MissingInputError, match=r"exited 4 \(MISSING_INPUT\)") as failed:
        runner.submit([JobSpec(name="evaluate", argv=_py("raise SystemExit(4)"))])
    assert failed.value.retryable is False and failed.value.details == {"job": "evaluate", "exit_code": 4}
    assert str(tmp_path / "logs" / "evaluate.log") in (failed.value.hint or "")
    with pytest.raises(RunnerError, match="exited 137") as killed:
        runner.submit([JobSpec(name="killed", argv=_py("raise SystemExit(137)"))])
    assert killed.value.retryable is False
