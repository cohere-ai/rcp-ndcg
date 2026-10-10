"""The T4 phased supervision re-run, offline (GPU-VALIDATION.md T4's offline counterpart).

The end-to-end scenarios run the run's phased job script in the pod, engines and coordinators under one
supervision block per phase (``rcp_ndcg.runners.script.supervise``).  This test re-runs that supervision on
CPU with **the verified fake engines as its engines** ("The phased job script's supervision runs with
emulators as its engines"): :func:`fake_engines` serves the encoder and the reranker phases with the
``rcp_ndcg_test.engines`` emulators of recipes with a current corpus (``tests/_emulator_server.py``,
over HTTP), and each phase's coordinator sends its engine a recorded request, which must be answered
``replayed``.  No judge corpus is recorded, so the judge phase's engine stays the root suite's supervision
stub (``tests/runners/test_supervise.py``).  It pins the four-phase supervision end to end: the phases in
order, one engine per phase, the engine stopped before the next phase starts, the engines' URLs in each
coordinator's ``RCP_NDCG_ENGINES``, the emulators' answers, and no engine left.
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
from tests._engines import ROOT, corpus_of, load_recipe

ENGINE = """#!/usr/bin/env bash
phase="${PHASE:-1}"
echo $$ > "$STUBS/engine-$phase.pid"
echo "engine-$phase start" >> "$STUBS/order"
if [[ "$phase" == 2 ]]; then
  if kill -0 "$(cat "$STUBS/engine-1.pid" 2>/dev/null)"; then
    echo "engine-2 saw phase 1 running" >> "$STUBS/order"
  else
    echo "engine-2 saw phase 1 stopped" >> "$STUBS/order"
  fi
fi
case "${ENGINE_MODE_OVERRIDE:-$ENGINE_MODE}" in
  crash) echo "engine: CUDA out of memory" >&2; exit 3 ;;
  ready) echo $$ > "$STUBS/ready-${PORT:-8000}"; exec sleep 60 ;;
  dies) echo $$ > "$STUBS/ready-${PORT:-8000}"; sleep 0.3; exit 7 ;;
  dies_ready) echo $$ > "$STUBS/ready-${PORT:-8000}"; exit 7 ;;
  hang) exec sleep 60 ;;
esac
"""

SRUN = """#!/usr/bin/env bash
while [[ "$1" == --* ]]; do shift; done
setsid "$@" &
step=$!
trap 'kill -TERM -- -$step 2>/dev/null; wait "$step" 2>/dev/null; exit 143' TERM
wait "$step"
"""

PYTHON3 = (
    '#!/usr/bin/env bash\nurl="${@: -1}"; port="${url##*:}"; port="${port%%/*}"\n'
    'pid_file="$STUBS/ready-$port"\n[ -f "$pid_file" ] || exit 1\n'
    'pid="$(cat "$pid_file")"\nkill -0 "$pid" 2>/dev/null\n'
)

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the job scripts target Linux nodes")

RUN_DIR = "/runs/rcp-t4"


#: The recipes whose verified fake engines serve a phase: one current corpus each (``tests/conformance``).
EMULATED = {"encoder": "qwen3-embedding-0.6b", "reranker": "qwen3-reranker-8b"}

#: The coordinator stub: records the engines it sees, then sends each emulated engine its recorded request
#: (``ask.py``, the test's real Python) and records the answer's status and emulator source.
COORDINATOR = """#!/usr/bin/env bash
echo "coord $RCP_NDCG_ENGINES" >> "$STUBS/order"
"$PY" "$STUBS/ask.py" >> "$STUBS/order" || exit 5
"""

ASK = """import json, os, pathlib, urllib.request
stubs = pathlib.Path(os.environ["STUBS"])
for role, engine in sorted(json.loads(os.environ["RCP_NDCG_ENGINES"]).items()):
    request = stubs / f"request-{role}.json"
    if not request.is_file():
        continue
    sent = json.loads(request.read_text())
    root = engine["urls"][0].removesuffix("/v1")
    http = urllib.request.Request(root + sent["path"], data=sent["body"].encode(), method="POST",
                                  headers={"content-type": "application/json"})
    with urllib.request.urlopen(http, timeout=30) as reply:
        print(f"answer {role} {reply.status} {reply.headers['x-rcp-ndcg-emulator-source']}")
"""


def fake_engines(role: str, port: int, phase: int) -> ServeConfig:
    """One role's engine of the supervision re-run -- the verified-fake-engines seam (one home: nothing else in
    this test names an engine).

    The encoder and the reranker are their recipes' emulators served over HTTP; the judge (no recorded judge
    corpus) is the root suite's supervision stub.
    """
    env = {"PHASE": str(phase), "PORT": str(port)}
    if role in EMULATED:
        command: tuple[str, ...] = (sys.executable, str(ROOT / "tests" / "_emulator_server.py"), EMULATED[role])
        env["PYTHONPATH"] = str(ROOT)
    else:
        command = ("vllm", "serve", role)
    return ServeConfig(command=command, env=env, resources=Resources(gpus=1), port=port, startup_timeout_s=60)


def _recorded_request(recipe_id: str) -> dict[str, str]:
    """The recipe corpus's first answered role request, as sent: its route and its body."""
    from rcp_ndcg_test.engines import exchanges_of

    exchange = next(
        item
        for item in exchanges_of(corpus_of(load_recipe(recipe_id)))
        if item.status == 200
        and item.path.endswith(("/embeddings", "/rerank"))
        and "unknown_field" not in (item.request_body or {})
    )
    body = exchange.request_raw.decode("utf-8") if exchange.request_raw else json.dumps(exchange.request_body)
    return {"path": exchange.path, "body": body}


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
    (tmp_path / "ask.py").write_text(ASK, encoding="utf-8")
    for role, recipe_id in EMULATED.items():
        (tmp_path / f"request-{role}.json").write_text(json.dumps(_recorded_request(recipe_id)), encoding="utf-8")
    return tmp_path


def test_the_four_phase_supervision_runs_with_the_verified_fake_engines(stubs: Path) -> None:
    """The T4 supervision re-run: four phases in order, one engine each, each coordinator handed its
    phase's engines (or ``{}``) and answered by its phase's emulator, and nothing left running at the job's
    end."""
    script = SlurmRunner().render([four_phase_job()])["t4"]
    (stubs / "job.sh").write_text(script, encoding="utf-8")
    env = {
        "PATH": f"{stubs}/bin:/usr/bin:/bin",
        "HOME": str(stubs),
        "STUBS": str(stubs),
        "ENGINE_MODE": "ready",
        "COORDINATOR_MODE": "done",
        "PHASE": "1",
        "PY": sys.executable,
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
    # The verified fake engines served the encoder and reranker phases: each coordinator's recorded request
    # was answered over HTTP by its phase's emulator, replayed from the recorded corpus.
    answers = [line.removeprefix("answer ") for line in order if line.startswith("answer ")]
    assert answers == ["encoder 200 replayed", "reranker 200 replayed"], order
    for phase in ("1", "2", "3"):
        pid = int((stubs / f"engine-{phase}.pid").read_text(encoding="utf-8"))
        with pytest.raises(OSError):
            os.kill(pid, 0)  # the supervision stopped every engine it started
