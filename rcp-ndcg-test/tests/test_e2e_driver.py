"""The T4 driver on CPU: the rendered script's supervision with stub engines (the root suite's
``tests/runners/test_supervise.py`` stubs, in the same shape), the process-boundary probe, the ``srun``
shim and the run-scoped judge the outage scenario kills and restarts.

These are the T4 driver's "CPU tests of the driver with stub engines"
(docs/how-to/validate-a-recipe.md): the phases' engines are
bash stubs standing in for ``vllm serve`` processes (today) and for the verified fake engines'
``rcp_ndcg_test.engines`` emulators (when lane ``fake-engines`` lands -- the engine commands in these
tests are the one seam that changes).  No GPU, no network.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from rcp_ndcg_test.e2e import (
    ManagedEngine,
    check_process_boundary,
    engine_python_facts,
    read_probe,
    recovers_outage_s,
    run_job_script,
    t0_smoke,
    write_probe,
    write_srun_shim,
)
from rcp_ndcg_test.errors import HarnessError

from rcp_ndcg.runners import JobSpec, Resources, ServeConfig
from rcp_ndcg.runners.base import JobPhase
from rcp_ndcg.runners.slurm import SlurmRunner

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the job scripts target Linux nodes")

#: The root suite's supervision stubs (tests/runners/test_supervise.py), in the same shape: one engine stub
#: per phase, a coordinator stub, and the readiness probe as a marker-file check on the URL's port.  The
#: engine commands here are the seam where the verified fake engines (rcp_ndcg_test.engines) plug in.
ENGINE = """#!/usr/bin/env bash
phase="${PHASE:-1}"
echo $$ > "$STUBS/engine-$phase.pid"
echo "engine-$phase start" >> "$STUBS/order"
if [[ "$phase" == 2 ]]; then
  if kill -0 "$(cat "$STUBS/engine-1.pid" 2>/dev/null)" 2>/dev/null; then
    echo "engine-2 saw phase 1 running" >> "$STUBS/order"
  else
    echo "engine-2 saw phase 1 stopped" >> "$STUBS/order"
  fi
fi
case "${ENGINE_MODE_OVERRIDE:-${ENGINE_MODE:-ready}}" in
  crash) echo "engine: CUDA out of memory" >&2; exit 3 ;;
  ready) echo $$ > "$STUBS/ready-${PORT:-8000}"; exec sleep 60 ;;
  dies) echo $$ > "$STUBS/ready-${PORT:-8000}"; sleep 0.3; exit 7 ;;
esac
"""
COORDINATOR = """#!/usr/bin/env bash
echo "coord $RCP_NDCG_ENGINES" >> "$STUBS/order"
case "$COORDINATOR_MODE" in
  done) exit 0 ;;
  fails) exit 4 ;;
  runs) exec sleep 60 ;;
esac
"""
PYTHON3 = (
    '#!/usr/bin/env bash\nurl="${@: -1}"; port="${url##*:}"; port="${port%%/*}"\n'
    'pid_file="$STUBS/ready-$port"\n[ -f "$pid_file" ] || exit 1\n'
    'pid="$(cat "$pid_file")"\nkill -0 "$pid" 2>/dev/null\n'
)


@pytest.fixture
def stubs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("rcp_ndcg.runners.script.PROBE_INTERVAL_S", 0.1)  # the root suite's stubs fixture
    monkeypatch.setattr("rcp_ndcg.runners.script.STOP_GRACE_S", 3)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("vllm", ENGINE), ("rcp-ndcg", COORDINATOR), ("uvx", COORDINATOR), ("python3", PYTHON3)):
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    return tmp_path


def _serve(phase: str, port: int) -> ServeConfig:
    return ServeConfig(
        command=("vllm", "serve", "stub"),
        env={"PHASE": phase, "PORT": str(port)},
        resources=Resources(gpus=1),
        port=port,
        startup_timeout_s=5,
    )


def _job() -> JobSpec:
    resume = ("rcp-ndcg", "run", "resume", "--run", "/runs/x")
    phases = (
        JobPhase(engines={"encoder": _serve("1", 8100)}, argv=(*resume, "--only", "retrieve")),
        JobPhase(engines={"reranker": _serve("2", 8110)}, argv=(*resume, "--only", "rerank")),
        JobPhase(
            engines={"judge": _serve("3", 8120)},
            argv=(*resume, "--only", "tournament", "--only", "rubric"),
        ),
        JobPhase(engines={}, argv=(*resume, "--only", "calibrate", "--only", "evaluate")),
    )
    return JobSpec(name="run", phases=phases)


def _env(stubs: Path, **extra: str) -> dict[str, str]:
    return {
        "PATH": f"{stubs}/bin:/usr/bin:/bin",
        "HOME": str(stubs),
        "STUBS": str(stubs),
        "ENGINE_MODE": "ready",
        "COORDINATOR_MODE": "done",
        "PHASE": "1",
        **extra,
    }


def test_run_job_script_runs_the_four_phases_with_stub_engines(stubs: Path, tmp_path: Path) -> None:
    """The driver runs the rendered script in the pod: the engines' phases in order, the engine stopped
    before the next phase's starts, the engines' URLs per phase, and no engine left running at the end."""
    script = SlurmRunner().render([_job()])["run"]
    out = tmp_path / "out"
    status, log = run_job_script(script, out=out, env=_env(stubs))
    assert status == 0, log.read_text(encoding="utf-8")
    assert (out / "job.sh").read_text(encoding="utf-8") == script
    order = (stubs / "order").read_text(encoding="utf-8").splitlines()
    assert "engine-1 start" in order and "engine-2 saw phase 1 stopped" in order and "engine-3 start" in order
    coordinators = [json.loads(line.removeprefix("coord ")) for line in order if line.startswith("coord ")]
    assert [sorted(row) for row in coordinators] == [["encoder"], ["reranker"], ["judge"], []]
    assert coordinators[0]["encoder"]["urls"] == ["http://127.0.0.1:8100/v1"]
    assert coordinators[2]["judge"]["wait_on_outage_s"] == 900  # ServeConfig's outage_timeout_s default
    for phase in ("1", "2", "3"):
        pid = int((stubs / f"engine-{phase}.pid").read_text(encoding="utf-8"))
        with pytest.raises(OSError):
            os.kill(pid, 0)  # the srun shim's session stopped with the phase


def test_the_srun_shim_runs_the_step_in_its_own_session_and_stops_it(tmp_path: Path) -> None:
    """The pod's stand-in for SLURM's ``srun`` (the run's engines run under it): a stop stops the step's
    session, and the step's flags are its scheduling only."""
    shim = write_srun_shim(tmp_path)
    step_pid = tmp_path / "step.pid"
    command = f'"{shim}/srun" --overlap --gres=gpu:2 bash -c "echo $$ > {step_pid}; exec sleep 60"'
    process = subprocess.Popen(["bash", "-c", command])
    deadline = time.monotonic() + 10
    while not step_pid.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert step_pid.exists()
    process.terminate()  # the step's session stops with the shim
    process.wait(timeout=10)
    with pytest.raises(OSError):
        os.kill(int(step_pid.read_text(encoding="utf-8")), 0)


def test_the_probe_records_the_coordinators_prefix_and_versions(tmp_path: Path) -> None:
    """Node-runtime item 10: the probe records the coordinator's ``sys.prefix`` and versions, and the
    boundary check passes for an outside interpreter and fails for the engine's own."""
    site, probe_jsonl = write_probe(tmp_path)
    fake = tmp_path / "rcp-ndcg"
    fake.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    env = {
        "PATH": f"{tmp_path}:/usr/bin:/bin",
        "PYTHONPATH": str(site),
        "RCP_E2E_PROBE_JSONL": str(probe_jsonl),
    }
    assert subprocess.run([sys.executable, str(fake), "run", "resume", "--run", "x"], env=env).returncode == 0
    records = read_probe(probe_jsonl)
    (coordinator,) = records
    assert tuple(coordinator["argv"][1:3]) == ("run", "resume")
    assert coordinator["sys_prefix"]
    engine = {"sys_prefix": coordinator["sys_prefix"], "executable": coordinator["executable"]}
    assert not check_process_boundary(records, engine_python=engine, version="0.0.1")["ok"]
    assert check_process_boundary([], engine_python=engine_python_facts(), version=None)["ok"] is False
    other = {"sys_prefix": "/engine/python", "executable": "/engine/python3"}
    ok = check_process_boundary(records, engine_python=other, version=None)
    assert ok["ok"], ok["detail"]


class JudgeStub(BaseHTTPRequestHandler):
    """One judge route: ``/v1/models`` and one chat completion, as the T0 smoke asks."""

    def log_message(self, *args: object) -> None:  # noqa: A001, ARG002 - stdlib signature
        return

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        body = json.dumps({"data": [{"id": "judge", "max_model_len": 131072, "dtype": "auto"}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        body = json.dumps(
            {
                "object": "chat.completion",
                "model": "judge",
                "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def judge_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _judge_server(port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), JudgeStub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _managed(port: int, tmp_path: Path) -> ManagedEngine:
    """A ManagedEngine whose command is a stub engine announcing its port (the wave runner's rule)."""
    stub = tmp_path / "managed-engine.sh"
    stub.write_text(f'#!/usr/bin/env bash\necho "up" >> {tmp_path}/state.log\nexec sleep 60\n', encoding="utf-8")
    stub.chmod(0o755)
    return ManagedEngine([str(stub)], env={"TMPDIR": str(tmp_path / "tmp")}, log_path=tmp_path / "engine.log")


def test_the_t0_smoke_verifies_a_judge_and_reports_what_the_engine_says(judge_port: int, tmp_path: Path) -> None:
    """The T0 smoke's verdict (the Flash-Next NVFP4 vs FP8 decision): boot, ``/v1/models``, one request,
    and the report names the engine's ``max_model_len`` and dtype."""
    server = _judge_server(judge_port)
    try:
        report = t0_smoke(
            _managed(judge_port, tmp_path),
            url=f"http://127.0.0.1:{judge_port}/v1",
            served_name="judge",
            completion_body={"model": "judge", "messages": [{"role": "user", "content": "ok"}], "max_tokens": 4},
            timeout_s=10,
        )
    finally:
        server.shutdown()
    assert report["state"] == "verified", report
    assert report["models"] == ["judge"] and report["max_model_len"] == 131072 and report["dtype"] == "auto"


def test_the_t0_smoke_fails_a_judge_that_never_answers(judge_port: int, tmp_path: Path) -> None:
    """The fallback path's trigger: an engine that serves nothing fails the smoke, with its reason."""
    report = t0_smoke(
        _managed(judge_port, tmp_path),
        url=f"http://127.0.0.1:{judge_port}/v1",
        served_name="judge",
        completion_body={"model": "judge"},
        timeout_s=2,
    )
    assert report["state"] == "failed" and "no" in report["error"]


def test_the_wait_for_an_engine_ends_when_the_engine_exits(tmp_path: Path, judge_port: int) -> None:
    """A dead engine answers nothing: the wait for it ends when the process exits, not after the whole
    ``startup_timeout_s``.

    The T4's second run was cancelled on a driver that looked hung: an engine that died while it was
    waited for left the driver polling for the full 3600 s of the scenario's timeout, silently.  The
    engine's death is the failure; it must be reported in seconds, with the engine's own log beside it.
    """
    stub = tmp_path / "dies.sh"
    stub.write_text("#!/usr/bin/env bash\nsleep 0.5\nexit 7\n", encoding="utf-8")
    stub.chmod(0o755)
    engine = ManagedEngine([str(stub)], env={"TMPDIR": str(tmp_path / "tmp")}, log_path=tmp_path / "engine.log")
    engine.start()
    started = time.monotonic()
    assert engine.wait_ready(f"http://127.0.0.1:{judge_port}/v1/models", 30) is False
    assert time.monotonic() - started < 10  # not the 30 s the caller asked for: the engine had exited


def _staged_run(tmp_path: Path) -> Path:
    """A staged run directory of a synthetic four-phase scenario (CPU: the configs load, nothing runs).

    What :func:`~rcp_ndcg_test.e2e.stage_run_dir` leaves behind is what a resume reopens: the product's
    ``run.yaml`` and the manifest in status ``submitted``.  The dataset is a local ``jsonl:`` file: the
    resume's rendering needs a staged run directory, not the Hub -- a shipped scenario's ``suite:`` dataset
    would make this test depend on a warm Hub cache (staging records the dataset's name).
    """
    from rcp_ndcg_test.e2e import Scenario, build_run_config, stage_run_dir
    from rcp_ndcg_vllm.recipe import default_recipes_root

    from rcp_ndcg.runs.run import prepare

    rows = tmp_path / "rows.jsonl"
    rows.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query": "what is the capital of france",
                "doc_ids": ["d1", "d2"],
                "docs": ["paris", "lyon"],
                "qrels": {"d1": 1, "d2": 0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scenario = Scenario.model_validate(
        {
            "id": "resume-fixture",
            "description": "the resume test's staged run (a local dataset, the shipped recipes)",
            "dataset": {"uri": f"jsonl:{rows}"},
            "steps": ["retrieve", "rerank", "tournament", "rubric", "calibrate", "evaluate"],
            "encoder_recipe": "qwen3-embedding-0.6b",
            "rerank_recipe": "qwen3-reranker-0.6b",
            "judge": {
                "recipe": "gpt-oss-120b",
                "config": {"wait_on_outage_s": 300},
                "slot": {"port": 8120, "vllm_port": 9120, "cuda_visible_devices": "0"},
            },
            "slots": {
                "encoder": {"port": 8100, "vllm_port": 9100, "cuda_visible_devices": "0"},
                "reranker": {"port": 8110, "vllm_port": 9110, "cuda_visible_devices": "0"},
            },
        }
    )
    config = build_run_config(scenario, recipes_root=[default_recipes_root()])
    pipeline = prepare(config, runs_dir=str(tmp_path / "runs"), label=scenario.id)
    stage_run_dir(pipeline)
    return Path(pipeline.layout.root)


def _stub_engines(stubs: Path) -> None:
    """The stubs' engine announces the port its command binds (the recipes render ``--port``, not ``$PORT``)."""
    (stubs / "bin" / "vllm").write_text(
        "#!/usr/bin/env bash\n"
        'args=("$@")\n'
        'for ((i=0; i<${#args[@]}; i++)); do [[ "${args[i]}" == --port ]] && port="${args[i+1]}"; done\n'
        'echo $$ > "$STUBS/ready-${port:-8000}"\n'
        "exec sleep 60\n",
        encoding="utf-8",
    )


def test_the_resume_is_the_runs_phased_script_with_its_engines(
    stubs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume is what a submitted job would run: the run's own phased script, each phase's engines
    started and their URLs handed to the coordinator (``reopen`` -> the product's renderer).

    The direct ``rcp-ndcg run resume`` this replaced carried no engines overlay, so a resume that had to
    re-run a served step refused it with "the served encoder has no base_url" -- the outage scenario's
    resume, whose earlier phases the killed judge's device had starved, ended that way in the T4's second
    run instead of finishing the run.
    """
    from rcp_ndcg_test.e2e import resume_run

    run_dir = _staged_run(tmp_path)
    _stub_engines(stubs)
    monkeypatch.setenv("PATH", f"{stubs}/bin:/usr/bin:/bin")
    monkeypatch.setenv("STUBS", str(stubs))
    monkeypatch.setenv("HOME", str(stubs))
    monkeypatch.setenv("COORDINATOR_MODE", "done")

    out = tmp_path / "out"
    status = resume_run(run_dir, wheelhouse="/stage/wheelhouse", constraints=None, version="0.0.1", out=out)
    log = (out / "job.log").read_text(encoding="utf-8")
    assert status == 0, log
    order = (stubs / "order").read_text(encoding="utf-8").splitlines()
    coordinators = [json.loads(line.removeprefix("coord ")) for line in order if line.startswith("coord ")]
    assert [sorted(row) for row in coordinators] == [["encoder"], ["reranker"], ["judge"], []]
    assert coordinators[0]["encoder"]["urls"] == ["http://127.0.0.1:8100/v1"]  # the resume reaches its engine
    assert coordinators[2]["judge"]["urls"] == ["http://127.0.0.1:8120/v1"]
    script = (out / "job.sh").read_text(encoding="utf-8")
    assert f"rcp-ndcg run resume --run {run_dir}" in script  # the run's own directory, the run's own phases
    assert "--find-links /stage/wheelhouse" in script and "--no-index" in script
    assert "PYTHONPATH=" in script and "client-probe.jsonl" in script  # the probe rides the phases


def test_the_recovers_outage_fits_the_judge_window_or_refuses_the_scenario() -> None:
    """The recovers sub-run's outage is sized against the whole down span, not the sleep alone.

    The judge is down for the outage sleep **and** for the restarted engine's boot; the span must fit the
    run's ``wait_on_outage_s`` with margin, or the sub-run is not a recovery but a second expiry.  The
    rc0 outage run (waves-rc0-202610141005) scheduled 15 s plus a ~77 s restart boot against a 90 s
    window: the tournament failed with ``BackendUnavailableError`` about 2 s before the judge answered,
    and the scenario's own "parks and recovers" checks failed on a run the product had handled correctly.
    A scenario whose numbers cannot recover is refused at the start, with the fix in the message.
    """
    with pytest.raises(HarnessError, match="wait_on_outage_s"):
        recovers_outage_s(90.0, 77.0)  # the rc0 numbers: 15 s sleep + 77 s boot + margin > 90 s
    assert recovers_outage_s(300.0, 185.0) == 20.0  # the shipped scenario, even with a cold restart boot
    assert recovers_outage_s(300.0, 77.0) == 20.0  # ... and with the warm restart the node measured
    assert recovers_outage_s(60.0, 0.0) == 10.0  # the /6 sizing, floored at 5 (an unmeasured boot gates at 0)


def test_a_managed_engine_records_the_boot_it_was_waited_on(judge_port: int, tmp_path: Path) -> None:
    """:class:`ManagedEngine` measures each start-to-ready span (``boot_s``), and resets it per start.

    The outage driver plans the recovers outage around the judge's measured boot
    (:func:`recovers_outage_s`): without the measurement the scenario's numbers are guesswork, and the
    rc0 run's guess was 2 s short of the window.
    """
    server = _judge_server(judge_port)
    engine = _managed(judge_port, tmp_path)
    try:
        engine.start()
        assert engine.boot_s is None  # nothing waited on yet
        assert engine.wait_ready(f"http://127.0.0.1:{judge_port}/v1/models", 10) is True
        assert engine.boot_s is not None and engine.boot_s >= 0
        engine.kill()  # the restart's boot is measured from its own start
        engine.start()
        assert engine.boot_s is None
        assert engine.wait_ready(f"http://127.0.0.1:{judge_port}/v1/models", 10) is True
        assert engine.boot_s is not None
    finally:
        server.shutdown()
        engine.stop()


def test_a_managed_engine_is_killed_and_restarted(tmp_path: Path) -> None:
    """The outage scenario's run-scoped judge (the product's model: replicas that live elsewhere are
    "restarted by whatever runs them"): ``kill`` takes the process group, ``restart`` brings a new pid."""
    engine = _managed(0, tmp_path)
    engine.start()
    first = engine.process
    assert first is not None and first.poll() is None
    engine.kill()
    assert first.poll() is not None  # killed
    engine.start()
    second = engine.process
    assert second is not None and second.pid != first.pid and second.poll() is None
    engine.stop()
    assert second.poll() is not None


def test_a_managed_engine_never_sees_the_probe_on_pythonpath(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe rides the phase workers' ``PYTHONPATH``, never an engine's: the driver's own
    ``PYTHONPATH`` is kept out of :class:`ManagedEngine`'s processes exactly as out of the job's
    engines (node-runtime item 10: engines stay untouched)."""
    monkeypatch.setenv("PYTHONPATH", "/leak/from/driver-env")
    engine_stub = tmp_path / "env-engine.sh"
    engine_stub.write_text(
        '#!/usr/bin/env bash\necho "PYTHONPATH=[${PYTHONPATH:-}]" > "$STATE_ENV"\nexec sleep 60\n', encoding="utf-8"
    )
    engine_stub.chmod(0o755)
    engine = ManagedEngine(
        [str(engine_stub)],
        env={"TMPDIR": str(tmp_path / "tmp"), "STATE_ENV": str(tmp_path / "state.env")},
        log_path=tmp_path / "engine.log",
    )
    engine.start()
    state = tmp_path / "state.env"
    deadline = time.monotonic() + 10
    while not state.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    engine.stop()
    assert state.exists()
    assert state.read_text(encoding="utf-8") == "PYTHONPATH=[]\n"


def test_the_resume_keeps_the_drivers_pythonpath_out_and_rides_the_probe(
    stubs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driven resume gets the same hygiene as every other driven job: the driver's ``PYTHONPATH``
    out (a checkout's sources must not shadow the staged wheels) and the probe in, on the phases'
    workers (the resumed coordinators record like the script's ones)."""
    from rcp_ndcg_test.e2e import resume_run

    run_dir = _staged_run(tmp_path)
    _stub_engines(stubs)
    monkeypatch.setenv("PYTHONPATH", "/leak/from/driver-env")
    monkeypatch.setenv("PATH", f"{stubs}/bin:/usr/bin:/bin")
    monkeypatch.setenv("STUBS", str(stubs))
    monkeypatch.setenv("HOME", str(stubs))
    monkeypatch.setenv("COORDINATOR_MODE", "done")

    out = tmp_path / "out"
    assert resume_run(run_dir, wheelhouse="/stage/wheelhouse", constraints=None, version="0.0.1", out=out) == 0
    script = (out / "job.sh").read_text(encoding="utf-8")
    assert "/leak/from/driver-env" not in script  # the driver's entry is not the phases' PYTHONPATH
    assert f"export PYTHONPATH={out}/probe-site" in script  # the probe replaces it
    assert f"export RCP_E2E_PROBE_JSONL={out}/client-probe.jsonl" in script
    assert (out / "probe-site" / "sitecustomize.py").is_file()  # written before the phases start


def test_the_outage_launcher_puts_the_srun_shim_on_the_path(stubs: Path, tmp_path: Path) -> None:
    """The outage sub-runs launch the scripted job the way the driver runs it: with the ``srun`` shim on
    the path and the probe kept out.  The stubs' ``bin`` carries no ``srun`` (a pod ships no SLURM
    client): what makes an engine step start is the launcher's own shim."""
    from rcp_ndcg_test.e2e import launch_job_script

    process = launch_job_script(
        'srun --overlap --gres=gpu:1 bash -c "touch $STUBS/outage-engine-started"',
        out=tmp_path / "out",
        env=_env(stubs),
    )
    assert process.wait(timeout=10) == 0
    assert (stubs / "outage-engine-started").exists()
