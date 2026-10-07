"""The T4 driver on CPU: the rendered script's supervision with stub engines (the root suite's
``tests/runners/test_supervise.py`` stubs, in the same shape), the process-boundary probe, the ``srun``
shim and the run-scoped judge the outage scenario kills and restarts.

These are GPU-VALIDATION.md T4's "CPU tests of the driver with stub engines": the phases' engines are
bash stubs standing in for ``vllm serve`` processes (today) and for the verified fake engines'
``rcp_ndcg.testing.engines`` emulators (when lane ``fake-engines`` lands -- the engine commands in these
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
from rcp_ndcg_vllm.e2e import (
    ManagedEngine,
    check_process_boundary,
    engine_python_facts,
    read_probe,
    run_job_script,
    t0_smoke,
    write_probe,
    write_srun_shim,
)

from rcp_ndcg.runners import JobSpec, Resources, ServeConfig
from rcp_ndcg.runners.base import JobPhase
from rcp_ndcg.runners.slurm import SlurmRunner

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the job scripts target Linux nodes")

#: The root suite's supervision stubs (tests/runners/test_supervise.py), in the same shape: one engine stub
#: per phase, a coordinator stub, and the readiness probe as a marker-file check on the URL's port.  The
#: engine commands here are the seam where the verified fake engines (rcp_ndcg.testing.engines) plug in.
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
  ready) touch "$STUBS/ready-${PORT:-8000}"; exec sleep 60 ;;
  dies) touch "$STUBS/ready-${PORT:-8000}"; sleep 0.3; exit 7 ;;
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
PYTHON3 = '#!/usr/bin/env bash\nurl="${@: -1}"; port="${url##*:}"; port="${port%%/*}"\n[ -f "$STUBS/ready-$port" ]\n'


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
