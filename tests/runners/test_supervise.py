"""A job that owns its engines never outlives them: the supervision script, run with stub engines and coordinators.

The same script runs on SLURM (the sbatch body, with ``srun`` stubbed) and in a Kubernetes container (the
container's command, with ``uvx`` stubbed); every single-phase behaviour is checked on both renderings, and the
phase ordering on the SLURM script, which runs its phases one after another.
"""

from __future__ import annotations

import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rcp_ndcg.runners import JobSpec, KubernetesRunner, Resources, ServeConfig, SlurmRunner
from rcp_ndcg.runners.base import JobPhase

# The job scripts run on Linux cluster nodes: the stubs need bash >= 4.3, setsid and /proc, which macOS lacks.
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the job scripts target Linux nodes")

SERVE = ServeConfig(
    image="vllm/vllm-openai:v0.30.0",
    command="vllm serve org/model --served-model-name m --port 8000",
    resources=Resources(gpus=8),
    startup_timeout_s=2,
)
PHASE = JobPhase(engines={"judge": SERVE}, argv=("rcp-ndcg", "run", "resume", "--run", "/runs/x"))
JOB = JobSpec(name="run", phases=(PHASE,))

#: The engine stub: records its pid and that it started, then behaves as $ENGINE_MODE says ($ENGINE_MODE_OVERRIDE
#: wins, so a test can fail one phase only); $PHASE tells the phases apart, and phase 2 records whether phase 1's
#: engine was already stopped when it started.
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
case "${ENGINE_MODE_OVERRIDE:-$ENGINE_MODE}" in
  crash) echo "engine: CUDA out of memory" >&2; exit 3 ;;
  ready) touch "$STUBS/ready-${PORT:-8000}"; exec sleep 60 ;;
  dies) touch "$STUBS/ready-${PORT:-8000}"; sleep 0.3; exit 7 ;;
  dies_ready) touch "$STUBS/ready-${PORT:-8000}"; exit 7 ;;
  hang) exec sleep 60 ;;
esac
"""
#: The coordinator stub (``rcp-ndcg`` on the node, ``uvx`` in a pod): records the engines it sees, then as
#: $COORDINATOR_MODE says.
COORDINATOR = """#!/usr/bin/env bash
echo "coord $RCP_NDCG_ENGINES" >> "$STUBS/order"
echo "$RCP_NDCG_ENGINES" > "$STUBS/engines"
case "$COORDINATOR_MODE" in
  done) exit 0 ;;
  fails) exit 4 ;;
  runs) exec sleep 60 ;;
esac
"""
#: srun runs the step's tasks in a process group of their own, stops the group when it is stopped, and is gone
#: only once its tasks are (a real srun waits for them too).
SRUN = """#!/usr/bin/env bash
while [[ "$1" == --* ]]; do shift; done
setsid "$@" &
step=$!
trap 'kill -TERM -- -$step 2>/dev/null; wait "$step" 2>/dev/null; exit 143' TERM
wait "$step"
"""
#: The readiness probe answers once the engine stub of the probed port has marked itself ready (the URL's port
#: names the marker, so a phase's probe never passes on an earlier phase's engine).
PYTHON3 = '#!/usr/bin/env bash\nurl="${@: -1}"; port="${url##*:}"; port="${port%%/*}"\n[ -f "$STUBS/ready-$port" ]\n'


def _script(platform: str) -> list[str]:
    """The task script of a one-phase job: the SLURM batch body (engine on the node) or the pod's container."""
    if platform == "slurm":
        phases = (PHASE.model_copy(update={"engines": {"judge": SERVE.model_copy(update={"image": None})}}),)
        job = JOB.model_copy(update={"phases": phases})
        return ["bash", "-c", SlurmRunner().render([job])["run"]]
    command = KubernetesRunner().manifest(JOB)["spec"]["template"]["spec"]["containers"][0]["command"]
    assert command[:2] == ["bash", "-c"]
    return command


def _two_phase_text(phase_two_env: dict[str, str] | None = None) -> str:
    """The rendered sbatch body of a two-engine-phase job (told apart by $PHASE and by its port)."""
    phases = (
        PHASE.model_copy(
            update={
                "engines": {"judge": SERVE.model_copy(update={"image": None, "env": {"PHASE": "1", "PORT": "8000"}})},
                "argv": ("rcp-ndcg", "run", "resume", "--only", "tournament"),
            }
        ),
        PHASE.model_copy(
            update={
                "engines": {
                    "judge": SERVE.model_copy(
                        update={
                            "image": None,
                            "port": 8001,
                            "env": {"PHASE": "2", "PORT": "8001", **(phase_two_env or {})},
                        }
                    )
                },
                "argv": ("rcp-ndcg", "run", "resume", "--only", "rubric"),
            }
        ),
    )
    job = JobSpec(name="run", phases=phases)
    return SlurmRunner().render([job])["run"]


def _two_phase_script(phase_two_env: dict[str, str] | None = None) -> list[str]:
    return ["bash", "-c", _two_phase_text(phase_two_env)]


@pytest.fixture
def stubs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("rcp_ndcg.runners.script.PROBE_INTERVAL_S", 0.1)
    monkeypatch.setattr("rcp_ndcg.runners.script.STOP_GRACE_S", 3)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (
        ("vllm", ENGINE),
        ("rcp-ndcg", COORDINATOR),
        ("uvx", COORDINATOR),
        ("srun", SRUN),
        ("python3", PYTHON3),
    ):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    return tmp_path


def _env(stubs: Path, engine: str, coordinator: str) -> dict[str, str]:
    return {
        "PATH": f"{stubs / 'bin'}:/usr/bin:/bin",
        "HOME": str(stubs),
        "STUBS": str(stubs),
        "ENGINE_MODE": engine,
        "COORDINATOR_MODE": coordinator,
        "PHASE": "1",
    }


def _run(script: list[str], stubs: Path, engine: str, coordinator: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        script,
        cwd=stubs,
        env=_env(stubs, engine, coordinator),
        capture_output=True,
        text=True,
        timeout=30,
    )


def _run_file(script_text: str, stubs: Path, engine: str, coordinator: str) -> subprocess.CompletedProcess[str]:
    """Run a rendered script the way sbatch runs the batch body: as a file, so all its phases share one bash.

    ``bash -c`` (the Kubernetes invocation) differs: a child that ended before ``wait -n`` is invisible there, and
    a stopped-but-unreaped one is not re-reported either -- both once hid a phase boundary from the tests.
    """
    path = stubs / "job.sh"
    path.write_text(script_text)
    return subprocess.run(
        ["bash", str(path)],
        cwd=stubs,
        env=_env(stubs, engine, coordinator),
        capture_output=True,
        text=True,
        timeout=30,
    )


def _alive(pid_file: Path) -> bool:
    """Whether the process whose pid is in ``pid_file`` still runs (a zombie has ended)."""
    try:
        stat = Path(f"/proc/{int(pid_file.read_text())}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _gone(pid_file: Path, within_s: float = 5) -> bool:
    deadline = time.monotonic() + within_s
    while _alive(pid_file):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


PLATFORMS = pytest.mark.parametrize("platform", ["slurm", "kubernetes"])


@PLATFORMS
def test_an_engine_that_exits_while_starting_fails_the_job_at_once(platform: str, stubs: Path) -> None:
    started = time.monotonic()
    done = _run(_script(platform), stubs, engine="crash", coordinator="done")
    assert done.returncode == 1
    assert time.monotonic() - started < 10
    assert "engine: CUDA out of memory" in done.stderr  # the engine's own words
    assert "rcp-ndcg: the engine exited with status 3 before it answered /v1/models" in done.stderr
    assert not (stubs / "engines").exists()


@PLATFORMS
def test_an_engine_that_never_answers_fails_the_job_after_the_startup_timeout(platform: str, stubs: Path) -> None:
    started = time.monotonic()
    done = _run(_script(platform), stubs, engine="hang", coordinator="done")
    elapsed = time.monotonic() - started
    assert done.returncode == 1
    assert SERVE.startup_timeout_s - 1 <= elapsed < 15  # bash counts whole seconds
    assert "no engine replica answered /v1/models within 2 s (serve.startup_timeout_s)" in done.stderr
    assert not (stubs / "engines").exists()
    assert _gone(stubs / "engine-1.pid")


@PLATFORMS
def test_an_engine_that_dies_mid_run_stops_the_coordinator_and_fails_the_job(platform: str, stubs: Path) -> None:
    done = _run(_script(platform), stubs, engine="dies", coordinator="runs")
    assert done.returncode == 1
    assert "rcp-ndcg: the judge engine exited with status 7 while the run was going" in done.stderr
    assert "run resume" in done.stderr
    assert _gone(stubs / "engine-1.pid")


@PLATFORMS
@pytest.mark.parametrize(("coordinator", "status"), [("done", 0), ("fails", 4)])
def test_the_coordinators_status_is_the_jobs_and_the_engine_is_stopped(
    platform: str, stubs: Path, coordinator: str, status: int
) -> None:
    done = _run(_script(platform), stubs, engine="ready", coordinator=coordinator)
    assert done.returncode == status, done.stderr
    engines = (stubs / "engines").read_text()
    assert '"judge": {"urls": ["http://127.0.0.1:8000/v1"], "wait_on_outage_s": 900}' in engines
    assert _gone(stubs / "engine-1.pid")


@PLATFORMS
def test_a_scheduler_stopping_the_job_stops_the_engine_and_the_coordinator(platform: str, stubs: Path) -> None:
    job = subprocess.Popen(
        _script(platform),
        cwd=stubs,
        env=_env(stubs, "ready", "runs"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not (stubs / "engines").exists():
            assert time.monotonic() < deadline and job.poll() is None, job.stderr
            time.sleep(0.05)
        job.send_signal(signal.SIGTERM)
        assert job.wait(timeout=15) == 143
    finally:
        if job.poll() is None:
            job.kill()
    assert _gone(stubs / "engine-1.pid")


def test_the_engine_starts_once(stubs: Path) -> None:
    """No restart loop: an engine that exits is not started again."""
    (stubs / "bin" / "vllm").write_text(f"#!/usr/bin/env bash\necho started >> {stubs}/starts\nexit 3\n")
    done = _run(_script("slurm"), stubs, engine="crash", coordinator="done")
    assert done.returncode == 1
    time.sleep(0.2)
    assert (stubs / "starts").read_text() == "started\n"


def _only(stubs: Path, *tools: str) -> Path:
    """A bin directory holding bash and the named stubs, and nothing else from the host."""
    bin_dir = stubs / "only"
    bin_dir.mkdir()
    (bin_dir / "bash").symlink_to(shutil.which("bash"))
    for tool in tools:
        (bin_dir / tool).symlink_to(stubs / "bin" / tool)
    return bin_dir


@PLATFORMS
def test_a_job_without_python3_fails_before_the_engine_starts(platform: str, stubs: Path) -> None:
    """The readiness probe needs python3: without it the job would probe until the startup timeout."""
    env = {**_env(stubs, "ready", "done"), "PATH": str(_only(stubs, "vllm", "rcp-ndcg", "uvx", "srun"))}
    started = time.monotonic()
    done = subprocess.run(_script(platform), cwd=stubs, env=env, capture_output=True, text=True, timeout=30)
    assert done.returncode == 1
    assert time.monotonic() - started < 5
    assert "rcp-ndcg: this job needs python3 on PATH" in done.stderr
    assert not (stubs / "engine-1.pid").exists()


def test_a_pod_image_without_uv_or_pip_fails_before_the_engine_starts(stubs: Path) -> None:
    # The python3 stub fails `-m pip --version` (the engine is not ready), and no uvx is on PATH.
    env = {**_env(stubs, "ready", "done"), "PATH": str(_only(stubs, "vllm", "python3"))}
    done = subprocess.run(_script("kubernetes"), cwd=stubs, env=env, capture_output=True, text=True, timeout=30)
    assert done.returncode == 1
    assert "uvx, which is not on PATH, and python3 has no pip" in done.stderr
    assert not (stubs / "engine-1.pid").exists()
    # SLURM runs the coordinator on the node (or in the stock image, which has uv): it asks for neither.
    assert "has no pip" not in _script("slurm")[2]


@PLATFORMS
def test_the_bash_version_is_checked_before_the_engine_starts(platform: str) -> None:
    """``wait -n`` needs bash 4.3; an older bash would take its usage error for the engine's exit."""
    script = _script(platform)[2]
    assert script.index("BASH_VERSINFO[1] < 3") < script.index("RCP_NDCG_ENGINE_PID=$!")
    assert "needs bash 4.3 or later" in script


# --------------------------------------------------------------------------------------------------------------
# Two phases on one SLURM allocation
# --------------------------------------------------------------------------------------------------------------


def test_phase_two_starts_only_after_phase_one_exited_zero_and_its_engine_stopped(stubs: Path) -> None:
    done = _run(_two_phase_script(), stubs, engine="ready", coordinator="done")
    assert done.returncode == 0, done.stderr
    order = (stubs / "order").read_text().splitlines()
    assert order == [
        "engine-1 start",
        'coord {"judge": {"urls": ["http://127.0.0.1:8000/v1"], "wait_on_outage_s": 900}}',
        "engine-2 start",
        "engine-2 saw phase 1 stopped",
        'coord {"judge": {"urls": ["http://127.0.0.1:8001/v1"], "wait_on_outage_s": 900}}',
    ]
    assert _gone(stubs / "engine-2.pid")


def test_a_rendered_sbatch_runs_as_a_file_two_phases_in_order(stubs: Path) -> None:
    """sbatch execs the batch file: all phases share one bash, and a stopped step must be reaped at the boundary.

    Run as a file, the stopped phase-1 engine step is a terminated job of that bash; unreaped, the next phase's
    ``wait -n`` consumed its stale status and failed the job with a phantom engine exit (bash -c does not re-report
    it, which is why the ``bash -c`` tests above never saw it).
    """
    done = _run_file(_two_phase_text(), stubs, engine="ready", coordinator="done")
    assert done.returncode == 0, done.stderr
    order = (stubs / "order").read_text().splitlines()
    assert order == [
        "engine-1 start",
        'coord {"judge": {"urls": ["http://127.0.0.1:8000/v1"], "wait_on_outage_s": 900}}',
        "engine-2 start",
        "engine-2 saw phase 1 stopped",
        'coord {"judge": {"urls": ["http://127.0.0.1:8001/v1"], "wait_on_outage_s": 900}}',
    ]


def test_an_engine_that_dies_between_readiness_and_the_wait_fails_the_job(stubs: Path) -> None:
    """An engine that exits after answering but before the phase's wait must not be swallowed.

    The container command runs as ``bash -c``, where ``wait -n`` does not report a child that ended before it was
    invoked: the engine's failure reached the phase only through the started engines' reaped statuses.
    """
    done = _run(_script("kubernetes"), stubs, engine="dies_ready", coordinator="done")
    assert done.returncode != 0, done.stderr
    assert "exited with status 7" in done.stderr


def test_a_failing_phase_one_engine_ends_the_job_and_never_starts_phase_two(stubs: Path) -> None:
    done = _run(_two_phase_script(), stubs, engine="crash", coordinator="done")
    assert done.returncode == 1
    assert "rcp-ndcg: the engine exited with status 3 before it answered /v1/models" in done.stderr
    assert "127.0.0.1:8001" not in (stubs / "order").read_text()
    assert not (stubs / "engine-2.pid").exists()


def test_a_failing_phase_one_coordinator_never_starts_phase_two(stubs: Path) -> None:
    done = _run(_two_phase_script(), stubs, engine="ready", coordinator="fails")
    assert done.returncode == 4, done.stderr
    order = (stubs / "order").read_text().splitlines()
    assert order == [
        "engine-1 start",
        'coord {"judge": {"urls": ["http://127.0.0.1:8000/v1"], "wait_on_outage_s": 900}}',
    ]
    assert not (stubs / "engine-2.pid").exists()


def test_a_failing_phase_two_engine_ends_the_job(stubs: Path) -> None:
    """Phase 2's engine crashes while it loads: the job ends with ENGINE_FAILED and its coordinator never runs."""
    done = _run(_two_phase_script({"ENGINE_MODE_OVERRIDE": "crash"}), stubs, engine="ready", coordinator="done")
    assert done.returncode == 1
    assert "rcp-ndcg: the engine exited with status 3 before it answered /v1/models" in done.stderr
    order = (stubs / "order").read_text().splitlines()
    assert order == [
        "engine-1 start",
        'coord {"judge": {"urls": ["http://127.0.0.1:8000/v1"], "wait_on_outage_s": 900}}',
        "engine-2 start",
        "engine-2 saw phase 1 stopped",
    ]
