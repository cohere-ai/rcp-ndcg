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
from rcp_ndcg.runners.script import supervise
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
SERVED = JobSpec(
    name="run",
    argv=("rcp-ndcg", "run", "resume", "--run", "/shared/runs/x"),
    resources=Resources(cpus=8, time_limit_s=86400),
    serve=SERVE,
)
#: The same job with its engine run on the node (``container_runtime: none`` takes no image).
ON_NODE = SERVED.model_copy(update={"serve": SERVE.model_copy(update={"image": None})})


class TestServe:
    def test_one_node_golden(self) -> None:
        script = SlurmRunner(partition="gpu", setup=["source .venv/bin/activate"]).render([ON_NODE])["run"]
        engine = (
            'srun --overlap --nodes=1 --ntasks-per-node=1 --kill-on-bad-exit=1 --wait=10 --gres=gpu:8 bash -c "$ENGINE"'
        )
        supervision = supervise(SERVE, engine=engine, coordinator='bash -c "$WORKER"', hosts='"${HOSTS[@]}"')
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
            "read -r -d '' WORKER <<'RCP_NDCG_WORKER' || true\n"
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "exec rcp-ndcg run resume --run /shared/runs/x\n"
            "RCP_NDCG_WORKER\n"
            "read -r -d '' ENGINE <<'RCP_NDCG_ENGINE' || true\n"
            "export HF_HOME=/shared/hf\n"
            "exec vllm serve org/model --served-model-name m --host 0.0.0.0 --port 8000\n"
            "RCP_NDCG_ENGINE\n"
            "HOSTS=(127.0.0.1)\n"
            'URLS=(); for host in "${HOSTS[@]}"; do URLS+=("http://$host:8000/v1"); done\n'
            'RCP_NDCG_JUDGE_URLS="$(IFS=,; echo "${URLS[*]}")"\n'
            "export RCP_NDCG_JUDGE_URLS\n" + "\n".join(supervision) + "\n"
        )
        # The supervision block: the engine once in the background (no restart loop), a bounded readiness wait,
        # the coordinator in the background, and the job ends with the first of the two.
        assert "while true; do\n  vllm" not in script and "restarting" not in script
        assert f"{engine} &\nRCP_NDCG_ENGINE_PID=$!\n" in script
        assert 'bash -c "$WORKER" &\nRCP_NDCG_COORDINATOR_PID=$!\nstatus=0\nwait -n || status=$?\n' in script
        assert "local deadline=$((SECONDS + 1800)) status" in script

    def test_several_nodes_run_one_engine_each_and_read_the_urls_from_the_node_list(self) -> None:
        served = SERVED.model_copy(update={"serve": SERVE.model_copy(update={"replicas": 3})})
        runner = SlurmRunner(container_runtime="pyxis", container_mounts=["/shared:/shared"])
        script = runner.render([served])["run"]
        assert "#SBATCH --nodes=3\n#SBATCH --ntasks-per-node=1\n" in script
        # One replica that exits ends the whole engine step (at once on a failure, 10 s later on status 0).
        assert (
            "srun --overlap --nodes=3 --ntasks-per-node=1 --kill-on-bad-exit=1 --wait=10 --gres=gpu:8 "
            '--container-image=vllm/vllm-openai:v0.30.0 --container-mounts=/shared:/shared bash -c "$ENGINE" &\n'
        ) in script
        assert 'mapfile -t HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")\n' in script
        # The coordinator is one task on the first node, in the stock image, beside the engine step.
        (coordinator,) = [line for line in script.splitlines() if line.endswith('bash -c "$WORKER" &')]
        assert coordinator.startswith("srun --overlap --nodes=1 --ntasks=1 --container-image=")
        assert "exec uvx --from" in script

    @pytest.mark.parametrize("runtime", ["none", "apptainer", "pyxis"])
    @pytest.mark.parametrize("replicas", [1, 2])
    def test_the_rendered_script_is_valid_bash(self, runtime: str, replicas: int) -> None:
        job = ON_NODE if runtime == "none" else SERVED
        served = job.model_copy(update={"serve": job.serve.model_copy(update={"replicas": replicas})})
        script = SlurmRunner(container_runtime=runtime).render([served])["run"]
        assert subprocess.run(["bash", "-n", "-c", script], capture_output=True).returncode == 0
        assert_shellcheck_clean(script)

    def test_a_replica_over_several_nodes_is_refused_when_the_config_is_read(self) -> None:
        """It was accepted here and failed as NotImplementedError (exit 1, 'this is a bug') when rendered."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="one replica per node"):
            ServeConfig(image="i", command="serve", nodes_per_replica=2)

    def test_an_image_is_refused_on_the_node_and_required_in_a_container(self) -> None:
        """With container_runtime none the image was silently ignored: the command ran on the node."""
        with pytest.raises(ConfigError, match="image would be ignored") as refused:
            SlurmRunner().render([SERVED])
        assert "container_runtime: apptainer | pyxis" in (refused.value.hint or "")
        with pytest.raises(ConfigError, match="names no image"):
            SlurmRunner(container_runtime="apptainer").render([ON_NODE])


def test_the_runners_resources_and_env_are_every_jobs_defaults() -> None:
    """They were accepted as options and then ignored: no --gres, --mem or export."""
    runner = get_runner("slurm", resources={"gpus": 2, "memory_gb": 64}, env={"A": "1", "HF_HOME": "/default"})
    plain = runner.render([JobSpec(name="j", argv=("true",))])["j"]
    assert "#SBATCH --gres=gpu:2" in plain and "#SBATCH --mem=65536M" in plain and "export A=1" in plain
    own = runner.render([JUDGE])["exp-judge"]  # the job's own resources and env win
    assert "#SBATCH --gres=gpu:8" in own and "export HF_HOME=/cache/hf" in own and "export A=1" in own
    assert "--gres=gpu:2" not in own and "/default" not in own
