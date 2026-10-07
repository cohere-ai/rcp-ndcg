"""The T4 phased supervision re-run, offline (GPU-VALIDATION.md T4's offline counterpart).

The end-to-end scenarios run the run's phased job script in the pod, engines and coordinators under one
supervision block per phase (``rcp_ndcg.runners.script.supervise``).  This test re-runs that supervision on
CPU with **the verified fake engines as its engines**: the seam is :func:`fake_engines`, whose stubs here
are the root suite's supervision stubs of ``tests/runners/test_supervise.py``.  When lane ``fake-engines``
lands its emulators (``rcp_ndcg.testing.engines``, replaying the recorded corpus), :func:`fake_engines`
returns their serve commands and this re-run drives the emulators instead -- the swap GPU-VALIDATION.md
("The phased job script's supervision runs with emulators as its engines") asks for.  Until then this pins
the four-phase supervision end to end: the phases in order, one engine per phase, the engine stopped before
the next phase starts, the engines' URLs in each coordinator's ``RCP_NDCG_ENGINES``, and no engine left.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rcp_ndcg.runners import JobSpec, Resources, ServeConfig
from rcp_ndcg.runners.base import JobPhase
from rcp_ndcg.runners.script import install_argv
from rcp_ndcg.runners.slurm import SlurmRunner
from rcp_ndcg.runs.execution import run_argv

from .test_supervise import COORDINATOR, ENGINE, PYTHON3, SRUN

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the job scripts target Linux nodes")

RUN_DIR = "/runs/rcp-t4"


def fake_engines(role: str, port: int, phase: int) -> ServeConfig:
    """One role's engine replicas of the supervision re-run -- the verified-fake-engines seam.

    Today: the root suite's supervision stub (a bash engine that marks itself ready and sleeps).  When
    lane ``fake-engines`` lands, this returns the emulator's serve command instead
    (``python -m rcp_ndcg.testing.engines serve vllm-0.31.0/<recipe> --port <port>``), and the
    coordinators' role configs point at ``fake://vllm-0.31.0/<recipe>``.  One home for the substitution:
    nothing else in this test names an engine.
    """
    return ServeConfig(
        command=("vllm", "serve", role),
        env={"PHASE": str(phase), "PORT": str(port)},
        resources=Resources(gpus=1),
        port=port,
        startup_timeout_s=5,
    )


def _wrap(argv: tuple[str, ...]) -> tuple[str, ...]:
    """The job's coordinator through the client mechanism (node-runtime items 1-2), as the T4 driver
    renders it."""
    return install_argv(argv, version="0.0.1", wheelhouse="/stage/wheelhouse")


def four_phase_job() -> JobSpec:
    """The T4 scenario-1 shape: encoder, reranker, judge, then the engine-free phase."""
    phases = (
        JobPhase(
            engines={"encoder": fake_engines("encoder", 8100, 1)},
            argv=_wrap(run_argv(RUN_DIR, None, ["retrieve"])),
        ),
        JobPhase(
            engines={"reranker": fake_engines("reranker", 8110, 2)},
            argv=_wrap(run_argv(RUN_DIR, None, ["rerank"])),
        ),
        JobPhase(
            engines={"judge": fake_engines("judge", 8120, 3)},
            argv=_wrap(run_argv(RUN_DIR, None, ["tournament", "rubric"])),
        ),
        JobPhase(engines={}, argv=_wrap(run_argv(RUN_DIR, None, ["calibrate", "evaluate"]))),
    )
    return JobSpec(name="t4", phases=phases)


@pytest.fixture
def stubs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("rcp_ndcg.runners.script.PROBE_INTERVAL_S", 0.1)
    monkeypatch.setattr("rcp_ndcg.runners.script.STOP_GRACE_S", 3)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs_and_bodies = (
        ("vllm", ENGINE),
        ("rcp-ndcg", COORDINATOR),
        ("uvx", COORDINATOR),
        ("srun", SRUN),
        ("python3", PYTHON3),
    )
    for name, body in stubs_and_bodies:
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    return tmp_path


def test_the_four_phase_supervision_runs_with_the_fake_engines(stubs: Path) -> None:
    """The T4 supervision re-run: four phases in order, one engine each, each coordinator handed its
    phase's engines (or ``{}``), and nothing left running at the job's end."""
    script = SlurmRunner().render([four_phase_job()])["t4"]
    (stubs / "job.sh").write_text(script, encoding="utf-8")
    env = {
        "PATH": f"{stubs}/bin:/usr/bin:/bin",
        "HOME": str(stubs),
        "STUBS": str(stubs),
        "ENGINE_MODE": "ready",
        "COORDINATOR_MODE": "done",
        "PHASE": "1",
    }
    run = subprocess.run(
        ["bash", str(stubs / "job.sh")], cwd=stubs, env=env, capture_output=True, text=True, timeout=60
    )
    assert run.returncode == 0, run.stdout + run.stderr
    order = (stubs / "order").read_text(encoding="utf-8").splitlines()
    assert "engine-1 start" in order and "engine-3 start" in order
    assert "engine-2 saw phase 1 stopped" in order  # one engine at a time
    coordinators = [json.loads(line.removeprefix("coord ")) for line in order if line.startswith("coord ")]
    assert [sorted(row) for row in coordinators] == [["encoder"], ["reranker"], ["judge"], []]
    for phase in ("1", "2", "3"):
        pid = int((stubs / f"engine-{phase}.pid").read_text(encoding="utf-8"))
        with pytest.raises(OSError):
            os.kill(pid, 0)  # the supervision stopped every engine it started
