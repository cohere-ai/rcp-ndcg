"""SlurmRunner: golden sbatch scripts, and the scheduler calls with the SLURM tools mocked."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.runners import (
    COORDINATOR_IMAGE,
    JobSpec,
    JobStatus,
    Resources,
    RunnerError,
    ServeConfig,
    SlurmRunner,
    get_runner,
)
from rcp_ndcg.runners.base import JobPhase
from rcp_ndcg.runners.script import EngineStep, engines_env_value, supervise
from rcp_ndcg.runners.slurm import slurm_time
from tests.runners.shell import assert_shellcheck_clean

JUDGE = JobSpec(
    name="exp-judge",
    argv=("rcp-ndcg", "run", "start", "run.yaml"),
    resources=Resources(gpus=8, cpus=16, memory_gb=512, time_limit_s=90061),
    env={"HF_HOME": "/cache/hf"},
)


def test_host_script_golden() -> None:
    runner = SlurmRunner(partition="gpu", account="proj", setup=["source .venv/bin/activate"], workdir="/work")
    assert runner.render([JUDGE])["exp-judge"] == (
        "#!/usr/bin/env bash\n"
        "#SBATCH --job-name=exp-judge\n"
        "#SBATCH --output=logs/slurm/%x-%j.out\n"
        "#SBATCH --ntasks=1\n"
        "#SBATCH --partition=gpu\n"
        "#SBATCH --account=proj\n"
        "#SBATCH --gres=gpu:8\n"
        "#SBATCH --cpus-per-task=16\n"
        "#SBATCH --mem=524288M\n"
        "#SBATCH --time=1-01:01:01\n"
        "set -euo pipefail\n"
        "source .venv/bin/activate\n"
        "read -r -d '' WORKER <<'RCP_NDCG_WORKER' || true\n"
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd /work\n"
        "export HF_HOME=/cache/hf\n"
        "exec rcp-ndcg run start run.yaml\n"
        "RCP_NDCG_WORKER\n"
        'bash -c "$WORKER"\n'
    )


@pytest.mark.parametrize(
    ("runtime", "image", "launcher"),
    [
        ("apptainer", "img.sif", 'apptainer exec --nv --bind /data:/data img.sif bash -c "$WORKER"'),
        ("apptainer", "org/img:1", 'apptainer exec --nv --bind /data:/data docker://org/img:1 bash -c "$WORKER"'),
        ("pyxis", "img.sif", 'srun --container-image=img.sif --container-mounts=/data:/data bash -c "$WORKER"'),
        (
            "pyxis",
            "ghcr.io/org/img:1",
            "srun --container-image='ghcr.io#org/img:1' --container-mounts=/data:/data bash -c \"$WORKER\"",
        ),
    ],
)
def test_container_runtimes(runtime: str, image: str, launcher: str) -> None:
    runner = SlurmRunner(container_runtime=runtime, image=image, container_mounts=["/data:/data"])
    script = runner.render([JobSpec(name="j", argv=("rcp-ndcg", "--help"))])["j"]
    assert script.splitlines()[-1] == launcher
    assert "exec uvx --from" in script and script.count("rcp-ndcg --help") == 1


def test_a_container_runtime_defaults_to_the_stock_coordinator_image() -> None:
    script = SlurmRunner(container_runtime="apptainer").render([JobSpec(name="j", argv=("true",))])["j"]
    assert script.splitlines()[-1] == f'apptainer exec --nv docker://{COORDINATOR_IMAGE} bash -c "$WORKER"'
    with pytest.raises(ConfigError, match="container_runtime"):
        SlurmRunner(container_runtime="docker")


def test_the_rendered_script_is_valid_bash() -> None:
    script = SlurmRunner().render([JUDGE])["exp-judge"]
    assert subprocess.run(["bash", "-n", "-c", script], capture_output=True).returncode == 0


def test_slurm_time() -> None:
    assert slurm_time(59) == "0-00:00:59"
    assert slurm_time(86400 + 3600 + 60 + 1) == "1-01:01:01"


class _FakeCli:
    """Records scheduler calls and answers them like SLURM would."""

    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers
        self.calls: list[tuple[list[str], str | None]] = []

    def __call__(self, argv, *, input_text=None):
        self.calls.append((list(argv), input_text))
        return self.answers.get(argv[0], "")


def test_submit_sends_each_script_to_sbatch_in_order(monkeypatch, tmp_path: Path) -> None:
    ids = iter(["101", "102;cluster"])
    fake = _FakeCli({})
    monkeypatch.setattr(
        "rcp_ndcg.runners.slurm.run_cli",
        lambda argv, input_text=None: (fake(argv, input_text=input_text), next(ids))[1],
    )
    runner = SlurmRunner(log_dir=str(tmp_path / "logs"))
    assert runner.submit([JobSpec(name="first", argv=("true",)), JobSpec(name="second", argv=("true",))]) == [
        "101",
        "102",
    ]
    (argv0, script0), (argv1, script1) = fake.calls
    assert argv0 == argv1 == ["sbatch", "--parsable"]
    assert "--job-name=first" in script0 and "--job-name=second" in script1
    assert (tmp_path / "logs").is_dir()


@pytest.mark.parametrize(
    ("squeue", "sacct", "expected"),
    [
        ("101 RUNNING\n", "", JobStatus.RUNNING),
        ("101 PENDING\n", "", JobStatus.PENDING),
        ("", "101|COMPLETED\n", JobStatus.SUCCEEDED),
        ("", "101|FAILED\n", JobStatus.FAILED),
        ("", "101|CANCELLED by 1000\n", JobStatus.CANCELLED),
        ("", "101|TIMEOUT\n", JobStatus.FAILED),
        ("", "", JobStatus.UNKNOWN),
    ],
)
def test_status_maps_scheduler_states(monkeypatch, squeue: str, sacct: str, expected: JobStatus) -> None:
    fake = _FakeCli({"squeue": squeue, "sacct": sacct})
    monkeypatch.setattr("rcp_ndcg.runners.slurm.run_cli", fake)
    assert SlurmRunner().status("101") is expected


def test_logs_and_cancel(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "exp-judge-101.out").write_text("zero\none\n")
    runner = SlurmRunner(log_dir=str(tmp_path))
    assert runner.logs("101") == "==> exp-judge-101.out <==\nzero\none\n"
    assert runner.logs("101", tail=1) == "one\n"
    with pytest.raises(RunnerError, match="no output"):
        runner.logs("999")
    fake = _FakeCli({})
    monkeypatch.setattr("rcp_ndcg.runners.slurm.run_cli", fake)
    runner.cancel("101")
    assert fake.calls == [(["scancel", "101"], None)]


def test_it_is_registered_by_name() -> None:
    assert isinstance(get_runner("slurm", partition="gpu"), SlurmRunner)


SERVE = ServeConfig(
    image="vllm/vllm-openai:v0.30.0",
    command="vllm serve org/model --served-model-name m --host 0.0.0.0 --port 8000",
    env={"HF_HOME": "/shared/hf"},
    resources=Resources(gpus=8, cpus=16),
)
ENCODER = ServeConfig(
    image="org/encoder:v2",
    command=["python3", "-m", "encoder", "--host", "0.0.0.0", "--port", "8001"],
    port=8001,
)
RERANKER = ServeConfig(
    image="org/reranker:v1",
    command=["python3", "-m", "reranker", "--host", "0.0.0.0", "--port", "8002"],
    port=8002,
    replicas=2,
)


def _on_node(phases: tuple[JobPhase, ...]) -> tuple[JobPhase, ...]:
    """The same phases with the engines run on the node (``container_runtime: none`` takes no image)."""
    return tuple(
        phase.model_copy(
            update={
                "engines": {role: engine.model_copy(update={"image": None}) for role, engine in phase.engines.items()}
            }
        )
        for phase in phases
    )


class TestPhases:
    def test_one_phase_golden(self) -> None:
        phase = JobPhase(
            engines={"judge": SERVE.model_copy(update={"image": None})},
            argv=("rcp-ndcg", "run", "resume", "--run", "/shared/runs/x"),
        )
        job = JobSpec(
            name="run",
            argv=("rcp-ndcg", "run", "resume", "--run", "/shared/runs/x"),
            resources=Resources(cpus=8, time_limit_s=86400),
            phases=(phase,),
        )
        script = SlurmRunner(partition="gpu", setup=["source .venv/bin/activate"]).render([job])["run"]
        engine = (
            "srun --overlap --nodes=1 --ntasks-per-node=1 --kill-on-bad-exit=1 --wait=10 --gres=gpu:8 "
            'bash -c "$ENGINE_JUDGE"'
        )
        supervision = supervise(
            [EngineStep(serve=SERVE, role="judge", start=engine, hosts="127.0.0.1")],
            coordinator='bash -c "$WORKER_1"',
            engines_env=f"'{engines_env_value({'judge': SERVE}, {'judge': ['http://127.0.0.1:8000/v1']})}'",
        )
        assert script == (
            "#!/usr/bin/env bash\n"
            "#SBATCH --job-name=run\n"
            "#SBATCH --output=logs/slurm/%x-%j.out\n"
            "#SBATCH --nodes=1\n"
            "#SBATCH --ntasks-per-node=1\n"
            "#SBATCH --partition=gpu\n"
            "#SBATCH --gres=gpu:8\n"
            "#SBATCH --cpus-per-task=24\n"
            "#SBATCH --mem=0\n"
            "#SBATCH --time=1-00:00:00\n"
            "set -euo pipefail\n"
            "source .venv/bin/activate\n"
            "read -r -d '' WORKER_1 <<'RCP_NDCG_WORKER_1' || true\n"
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "exec rcp-ndcg run resume --run /shared/runs/x\n"
            "RCP_NDCG_WORKER_1\n"
            "read -r -d '' ENGINE_JUDGE <<'RCP_NDCG_ENGINE_JUDGE' || true\n"
            "export HF_HOME=/shared/hf\n"
            "exec vllm serve org/model --served-model-name m --host 0.0.0.0 --port 8000\n"
            "RCP_NDCG_ENGINE_JUDGE\n" + "\n".join(supervision) + "\n"
        )
        # The phase's engine starts once in the background (no restart loop), a bounded readiness wait, the
        # coordinator with the phase's engines in RCP_NDCG_ENGINES, and the phase ends with the first of the two.
        assert "while true; do\n  vllm" not in script and "restarting" not in script
        assert f"{engine} &\nRCP_NDCG_ENGINE_PID=$!\n" in script
        assert 'bash -c "$WORKER_1" &\nRCP_NDCG_COORDINATOR_PID=$!\nstatus=0\nwait -n || status=$?\n' in script
        assert "local deadline=$((SECONDS + timeout)) status" in script

    def test_four_phases_ask_for_the_maximum_over_phases(self) -> None:
        """The paper run: retrieve (encoder), rerank (2 rerankers), judge, then calibrate+evaluate (none)."""
        phases = (
            JobPhase(engines={"encoder": ENCODER}, argv=("a",)),
            JobPhase(engines={"reranker": RERANKER}, argv=("b",)),
            JobPhase(engines={"judge": SERVE}, argv=("c",)),
            JobPhase(argv=("d",)),
        )
        script = SlurmRunner(container_runtime="pyxis", container_mounts=["/shared:/shared"]).render(
            [JobSpec(name="paper", argv=("x",), phases=phases)]
        )["paper"]
        assert "#SBATCH --nodes=2\n#SBATCH --ntasks-per-node=1\n" in script  # the largest phase's replicas
        assert "#SBATCH --gres=gpu:8\n" in script  # the largest single-replica request over the phases
        # Each role's replicas are pinned to their slice of the allocation's nodes.
        assert 'mapfile -t RCP_NDCG_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")\n' in script
        assert 'HOSTS_RERANKER=("${RCP_NDCG_HOSTS[@]:0:2}")\n' in script
        assert '--nodelist="$(IFS=,; echo "${HOSTS_RERANKER[*]}")"' in script
        # The RCP_NDCG_ENGINES of a multi-node phase is built when the job starts, from the pinned nodes.
        assert "RCP_NDCG_ENGINES_SPEC" in script
        assert (
            'RCP_NDCG_ENGINES="$(python3 -c "$RCP_NDCG_ENGINES_SPEC" '
            '"reranker:8002:900:$( IFS=,; echo "${HOSTS_RERANKER[*]}" )")"' in script
        )
        # The last phase has no engines: its command runs directly, with an empty RCP_NDCG_ENGINES.
        assert "export RCP_NDCG_ENGINES='{}'" in script
        (last,) = [line for line in script.splitlines() if line.endswith('bash -c "$WORKER_4"')]
        assert last.startswith("srun --container-image=")

    def test_engine_free_phases_run_directly_between_engine_phases(self) -> None:
        phases = (
            JobPhase(engines={"judge": SERVE.model_copy(update={"image": None})}, argv=("a",)),
            JobPhase(argv=("b",)),
            JobPhase(engines={"judge": SERVE.model_copy(update={"image": None})}, argv=("c",)),
        )
        script = SlurmRunner().render([JobSpec(name="j", argv=("x",), phases=phases)])["j"]
        blocks = [
            line
            for line in script.splitlines()
            if line in ('bash -c "$WORKER_2"', 'bash -c "$WORKER_1" &', 'bash -c "$WORKER_3" &')
        ]
        assert blocks == ['bash -c "$WORKER_1" &', 'bash -c "$WORKER_2"', 'bash -c "$WORKER_3" &']

    @pytest.mark.parametrize("runtime", ["none", "apptainer", "pyxis"])
    @pytest.mark.parametrize(
        "phases",
        [
            (JobPhase(engines={"judge": SERVE}, argv=("a",)),),
            (
                JobPhase(engines={"encoder": ENCODER}, argv=("a",)),
                JobPhase(engines={"reranker": RERANKER}, argv=("b",)),
                JobPhase(argv=("c",)),
            ),
            (
                JobPhase(engines={"encoder": ENCODER}, argv=("a",)),
                JobPhase(engines={"reranker": RERANKER}, argv=("b",)),
                JobPhase(engines={"judge": SERVE}, argv=("c",)),
                JobPhase(argv=("d",)),
            ),
        ],
        ids=["one", "three", "four"],
    )
    def test_the_rendered_script_is_valid_bash(self, runtime: str, phases: tuple[JobPhase, ...]) -> None:
        rendered = _on_node(phases) if runtime == "none" else phases
        script = SlurmRunner(container_runtime=runtime).render([JobSpec(name="j", argv=("x",), phases=rendered)])["j"]
        assert subprocess.run(["bash", "-n", "-c", script], capture_output=True).returncode == 0
        assert_shellcheck_clean(script)

    def test_the_engine_image_is_refused_on_the_node_and_required_in_a_container(self) -> None:
        """With container_runtime none the image was silently ignored: the command ran on the node."""
        phases = (JobPhase(engines={"judge": SERVE}, argv=("a",)),)
        with pytest.raises(ConfigError, match="image would be ignored") as refused:
            SlurmRunner().render([JobSpec(name="j", argv=("x",), phases=phases)])
        assert "container_runtime: apptainer | pyxis" in (refused.value.hint or "")
        with pytest.raises(ConfigError, match="judge engine.*names no image"):
            SlurmRunner(container_runtime="apptainer").render([JobSpec(name="j", argv=("x",), phases=_on_node(phases))])


def test_the_runners_resources_and_env_are_every_jobs_defaults() -> None:
    """They were accepted as options and then ignored: no --gres, --mem or export."""
    runner = get_runner("slurm", resources={"gpus": 2, "memory_gb": 64}, env={"A": "1", "HF_HOME": "/default"})
    plain = runner.render([JobSpec(name="j", argv=("true",))])["j"]
    assert "#SBATCH --gres=gpu:2" in plain and "#SBATCH --mem=65536M" in plain and "export A=1" in plain
    own = runner.render([JUDGE])["exp-judge"]  # the job's own resources and env win
    assert "#SBATCH --gres=gpu:8" in own and "export HF_HOME=/cache/hf" in own and "export A=1" in own
    assert "--gres=gpu:2" not in own and "/default" not in own
