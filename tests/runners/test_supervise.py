"""A job that owns its engine never outlives it: the supervision script, run with stub engines and coordinators.

The same script runs on SLURM (the sbatch body, with ``srun`` stubbed) and in a Kubernetes pod of one replica (the
container's command, with ``uvx`` stubbed); every behaviour is checked on both renderings.
"""

from __future__ import annotations

import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from rcp_ndcg.runners import JobSpec, KubernetesRunner, Resources, ServeConfig, SlurmRunner

SERVE = ServeConfig(
    image="vllm/vllm-openai:v0.30.0",
    command="vllm serve org/model --served-model-name m --port 8000",
    resources=Resources(gpus=8),
    startup_timeout_s=2,
)
JOB = JobSpec(name="run", argv=("rcp-ndcg", "run", "resume", "--run", "/runs/x"), serve=SERVE)

#: The engine stub: records its pid, then behaves as $ENGINE_MODE says; ready means the readiness probe answers.
ENGINE = """#!/usr/bin/env bash
echo $$ > "$STUBS/engine.pid"
case "$ENGINE_MODE" in
  crash) echo "engine: CUDA out of memory" >&2; exit 3 ;;
  hang) exec sleep 60 ;;
  ready) touch "$STUBS/ready"; exec sleep 60 ;;
  dies) touch "$STUBS/ready"; sleep 0.3; exit 7 ;;
esac
"""
#: The coordinator stub (``rcp-ndcg`` on the node, ``uvx`` in a pod): records its pid, then as $COORDINATOR_MODE says.
COORDINATOR = """#!/usr/bin/env bash
echo $$ > "$STUBS/coordinator.pid"
echo "$RCP_NDCG_JUDGE_URLS" > "$STUBS/urls"
case "$COORDINATOR_MODE" in
  done) exit 0 ;;
  fails) exit 4 ;;
  runs) exec sleep 60 ;;
esac
"""
#: srun runs the step's tasks in a process group of their own and stops the group when it is stopped.
SRUN = """#!/usr/bin/env bash
while [[ "$1" == --* ]]; do shift; done
setsid "$@" &
step=$!
trap 'kill -TERM -- -$step 2>/dev/null; exit 143' TERM
wait "$step"
"""
#: The readiness probe answers once the engine stub has marked itself ready.
PYTHON3 = '#!/usr/bin/env bash\n[ -f "$STUBS/ready" ]\n'


def _script(platform: str) -> list[str]:
    if platform == "slurm":
        return ["bash", "-c", SlurmRunner().render([JOB])["run"]]
    command = KubernetesRunner().manifest(JOB)["spec"]["template"]["spec"]["containers"][0]["command"]
    assert command[:2] == ["bash", "-c"]
    return command


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
    }


def _run(platform: str, stubs: Path, engine: str, coordinator: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _script(platform),
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
    done = _run(platform, stubs, engine="crash", coordinator="done")
    assert done.returncode == 1
    assert time.monotonic() - started < 10
    assert "engine: CUDA out of memory" in done.stderr  # the engine's own words
    assert "rcp-ndcg: the engine exited with status 3 before it answered /v1/models" in done.stderr
    assert not (stubs / "coordinator.pid").exists()


@PLATFORMS
def test_an_engine_that_never_answers_fails_the_job_after_the_startup_timeout(platform: str, stubs: Path) -> None:
    started = time.monotonic()
    done = _run(platform, stubs, engine="hang", coordinator="done")
    elapsed = time.monotonic() - started
    assert done.returncode == 1
    assert SERVE.startup_timeout_s - 1 <= elapsed < 15  # bash counts whole seconds
    assert "no engine replica answered /v1/models within 2 s (serve.startup_timeout_s)" in done.stderr
    assert not (stubs / "coordinator.pid").exists()
    assert _gone(stubs / "engine.pid")


@PLATFORMS
def test_an_engine_that_dies_mid_run_stops_the_coordinator_and_fails_the_job(platform: str, stubs: Path) -> None:
    done = _run(platform, stubs, engine="dies", coordinator="runs")
    assert done.returncode == 1
    assert "rcp-ndcg: the engine exited with status 7 while the run was going" in done.stderr
    assert "run resume" in done.stderr
    assert _gone(stubs / "coordinator.pid")


@PLATFORMS
@pytest.mark.parametrize(("coordinator", "status"), [("done", 0), ("fails", 4)])
def test_the_coordinators_status_is_the_jobs_and_the_engine_is_stopped(
    platform: str, stubs: Path, coordinator: str, status: int
) -> None:
    done = _run(platform, stubs, engine="ready", coordinator=coordinator)
    assert done.returncode == status, done.stderr
    assert (stubs / "urls").read_text() == "http://127.0.0.1:8000/v1\n"
    assert _gone(stubs / "engine.pid")


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
        while not (stubs / "coordinator.pid").exists():
            assert time.monotonic() < deadline and job.poll() is None, job.stderr
            time.sleep(0.05)
        job.send_signal(signal.SIGTERM)
        assert job.wait(timeout=15) == 143
    finally:
        if job.poll() is None:
            job.kill()
    assert _gone(stubs / "engine.pid") and _gone(stubs / "coordinator.pid")


def test_the_engine_starts_once(stubs: Path) -> None:
    """No restart loop: an engine that exits is not started again."""
    (stubs / "bin" / "vllm").write_text(f"#!/usr/bin/env bash\necho started >> {stubs}/starts\nexit 3\n")
    done = _run("slurm", stubs, engine="crash", coordinator="done")
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
    assert not (stubs / "engine.pid").exists()


def test_a_pod_image_without_uv_or_pip_fails_before_the_engine_starts(stubs: Path) -> None:
    # The python3 stub fails `-m pip --version` (the engine is not ready), and no uvx is on PATH.
    env = {**_env(stubs, "ready", "done"), "PATH": str(_only(stubs, "vllm", "python3"))}
    done = subprocess.run(_script("kubernetes"), cwd=stubs, env=env, capture_output=True, text=True, timeout=30)
    assert done.returncode == 1
    assert "uvx, which is not on PATH, and python3 has no pip" in done.stderr
    assert not (stubs / "engine.pid").exists()
    # SLURM runs the coordinator on the node (or in the stock image, which has uv): it asks for neither.
    assert "has no pip" not in _script("slurm")[2]


@PLATFORMS
def test_the_bash_version_is_checked_before_the_engine_starts(platform: str) -> None:
    """``wait -n`` needs bash 4.3; an older bash would take its usage error for the engine's exit."""
    script = _script(platform)[2]
    assert script.index("BASH_VERSINFO[1] < 3") < script.index("RCP_NDCG_ENGINE_PID=$!")
    assert "needs bash 4.3 or later" in script
