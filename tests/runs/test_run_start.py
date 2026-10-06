"""``rcp-ndcg run start`` against the offline judge: estimate, plan, run, resume, no-resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from rcp_ndcg.cli.main import cli
from tests.conftest import SESSION_TOKENIZER
from tests.runs.conftest import tiny_config

_SERVED_BUDGET = {"tokenizer": str(SESSION_TOKENIZER), "max_tokens": 8192}


def _command(name: str, *args: str) -> dict:
    result = CliRunner().invoke(cli, ["run", name, *args, "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["data"]


def _start(*args: str) -> dict:
    return _command("start", *args)


def test_estimate_plan_run_and_resume(data: Path, tmp_path: Path) -> None:
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(tiny_config(data).resolved()), encoding="utf-8")
    runs = ("--runs-dir", str(tmp_path / "runs"))

    estimate = _start(str(config), *runs, "--estimate")
    assert estimate["mode"] == "estimate"
    assert estimate["estimate"]["calls"] > 0 and "usd" not in estimate["estimate"]
    plan = _start(str(config), *runs, "--dry-run", "--only", "tournament")
    assert plan["plan"] == [{"step": "tournament", "status": "would run"}]
    assert not (tmp_path / "runs").exists()

    started = _start(str(config), *runs, "--set", "limit=1", "--label", "one")
    assert started["mode"] == "ran" and started["state"]["status"] == "completed"
    assert "-one" in started["run_id"]
    assert json.loads(Path(started["run_dir"], "manifest.json").read_text())["config"]["limit"] == 1

    resumed = _command("resume", "--run", started["run_dir"])
    assert {step["status"] for step in resumed["state"]["steps"]} == {"completed"}, "a no-op resume keeps them"
    assert resumed["state"]["done"] is True
    redone = _command("resume", "--run", started["run_dir"], "--no-resume", "--only", "calibrate")
    assert [(step["name"], step["status"]) for step in redone["state"]["steps"]][-2:] == [
        ("calibrate", "completed"),
        ("evaluate", "completed"),
    ]


class _FakeRunner:
    """A job runner that records what it was handed and runs nothing."""

    submitted: list[tuple[str, ...]] = []

    def __init__(self, **options) -> None:
        self.options = options

    def submit(self, jobs) -> list[str]:
        _FakeRunner.submitted.extend(job.argv for job in jobs)
        return [f"h{len(_FakeRunner.submitted)}"]

    def status(self, handle: str) -> str:
        return "pending"

    def logs(self, handle: str, *, tail: int | None = None) -> str:
        return ""

    def cancel(self, handle: str) -> None:
        pass


@pytest.fixture
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    _FakeRunner.submitted = []
    monkeypatch.setattr("rcp_ndcg.runs.execution.get_runner", lambda name, **options: _FakeRunner(**options))
    return _FakeRunner.submitted


def test_a_run_on_a_runner_is_submitted_once_and_its_job_runs_it_here(
    data: Path, tmp_path: Path, fake_runner: list
) -> None:
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(tiny_config(data).resolved()), encoding="utf-8")

    started = _start(str(config), "--runs-dir", str(tmp_path / "runs"), "--set", "runner.name=fake")
    assert started["mode"] == "submitted" and len(fake_runner) == 1
    job = fake_runner[0]
    assert job[1:4] == ("run", "resume", "--run")

    # What the submitted job executes: it must run the pipeline, not hand the run to the runner again.
    inside = _command("resume", *job[3:])
    assert inside["mode"] == "ran" and inside["state"]["status"] == "completed"
    assert len(fake_runner) == 1


def test_the_library_entry_point_honours_the_configured_runner(data: Path, tmp_path: Path, fake_runner: list) -> None:
    import rcp_ndcg

    config = tiny_config(data, runner={"name": "fake"})
    run = rcp_ndcg.run(config, runs_dir=str(tmp_path / "runs"))
    assert len(fake_runner) == 1 and run.manifest.status.value == "submitted"
    # Resuming the run directory (what the job does) runs it here.
    assert rcp_ndcg.run(run.dir).manifest.status.value == "completed"
    assert len(fake_runner) == 1


def test_the_runs_directory_variable_is_where_runs_start_and_are_listed(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved = tiny_config(data).resolved()
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump({**resolved, "steps": ["tournament"]}), encoding="utf-8")
    monkeypatch.setenv("RCP_NDCG_RUNS_DIR", str(tmp_path / "x"))
    monkeypatch.chdir(tmp_path)

    started = _start(str(config))
    assert Path(started["run_dir"]).parent == tmp_path / "x"
    assert [row["run_id"] for row in _command("list")["runs"]] == [started["run_id"]]


def test_a_dry_run_without_resume_plans_every_step(finished: Path) -> None:
    planned = _command("resume", "--run", str(finished), "--no-resume", "--dry-run")["plan"]
    assert {row["status"] for row in planned} == {"would run"}
    assert {row["status"] for row in _command("resume", "--run", str(finished), "--dry-run")["plan"]} == {"would skip"}


def test_a_cluster_run_requests_the_resources_image_and_env_of_its_runner_options(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rcp_ndcg
    from rcp_ndcg.errors import ConfigError
    from rcp_ndcg.runs.config import DatasetSource
    from rcp_ndcg.runs.run import Run

    submitted: list[str] = []

    def scheduler(argv, *, input_text=None) -> str:
        submitted.append(input_text or "")
        return '{"metadata": {"uid": "u"}}' if argv[0] == "kubectl" else "4242\n"

    monkeypatch.setattr("rcp_ndcg.runners.slurm.run_cli", scheduler)
    monkeypatch.setattr("rcp_ndcg.runners.kubernetes.run_cli", scheduler)
    job = {
        "resources": {"gpus": 2, "cpus": 8, "memory_gb": 64, "time_limit_s": 7200},
        "image": "registry.example.com/rcp-ndcg:vllm",
        "env": {"HF_HOME": "/scratch/hf"},
    }
    slurm = {"partition": "gpu", "log_dir": str(tmp_path / "slurm"), "container_runtime": "apptainer", **job}
    run = rcp_ndcg.run(tiny_config(data, runner={"name": "slurm", "options": slurm}), runs_dir=str(tmp_path / "runs"))
    (script,) = submitted
    for line in ("--gres=gpu:2", "--cpus-per-task=8", "--mem=65536M", "--time=0-02:00:00", "--partition=gpu"):
        assert f"#SBATCH {line}\n" in script
    assert "registry.example.com/rcp-ndcg:vllm" in script and "export HF_HOME=/scratch/hf" in script
    # The job fields are the job's, not the runner's: the recorded options rebuild the runner for run status.
    assert "resources" not in Run(run.dir).jobs()["options"]
    assert rcp_ndcg.runs.execution.status(run.dir).runner == "slurm"

    submitted.clear()
    kubernetes = {"namespace": "eval", **job}
    local = tiny_config(data, runner={"name": "kubernetes", "options": kubernetes}, mirror="memory://runs/k8s")
    with pytest.raises(ConfigError, match="do not see this host's files") as refused:  # the pod cannot read it
        rcp_ndcg.run(local, runs_dir=str(tmp_path / "refused"))
    assert refused.value.details["inputs"] == [f"dataset: {data}"] and not (tmp_path / "refused").exists()
    import fsspec

    fsspec.filesystem("memory").pipe("/data/rows.jsonl", data.read_bytes())
    config = local.model_copy(update={"dataset": DatasetSource(uri="jsonl:memory://data/rows.jsonl")})
    run = rcp_ndcg.run(config, runs_dir=str(tmp_path / "runs"))
    manifest = yaml.safe_load(submitted[0])
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "registry.example.com/rcp-ndcg:vllm"
    assert container["resources"]["limits"] == {"nvidia.com/gpu": 2, "cpu": 8, "memory": "64Gi"}
    assert manifest["spec"]["activeDeadlineSeconds"] == 7200
    assert "export HF_HOME=/scratch/hf" in container["command"][2]
    # The pod does not see this host's files: it restores the run into its scratch volume from the mirror,
    # which holds the prepared run directory before the job is submitted.
    assert f"--run /scratch/runs/{Path(run.dir).name} --mirror memory://runs/k8s" in container["command"][2]
    assert fsspec.filesystem("memory").exists("/runs/k8s/manifest.json")


def test_a_plan_or_an_estimate_of_an_existing_run_names_its_directory(finished: Path) -> None:
    assert _command("resume", "--run", str(finished), "--dry-run")["run_dir"] == str(finished)
    assert _command("resume", "--run", str(finished), "--estimate")["run_dir"] == str(finished)


def test_an_ad_hoc_judge_is_estimated_without_being_called(data: Path, tmp_path: Path) -> None:
    """The documented re-judge command: --judge-url with --judge-model, and --set for the judge's other fields."""
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump({**tiny_config(data).resolved(), "judge": "gpt_oss_120b"}), encoding="utf-8")
    ad_hoc = ("--judge-url", "http://127.0.0.1:9/v1", "--judge-model", "my-model", "--set", "judge.concurrency=4")

    estimate = _start(str(config), *ad_hoc, "--estimate")

    assert estimate["estimate"]["calls"] > 0 and estimate["estimate"]["input_tokens"] > 0


def _failed(name: str, *args: str) -> dict:
    result = CliRunner().invoke(cli, ["run", name, *args, "--json"])
    assert result.exit_code != 0, result.output
    return json.loads(result.stdout)["error"]


SERVE = {
    "judge": {
        "image": "vllm/vllm-openai:v0.30.0",
        "command": "vllm serve org/model --served-model-name m --host 0.0.0.0 --port 8000",
        "resources": {"gpus": 8},
    }
}
SERVED_JUDGE = {"base_url": "http://unused/v1", "model": "m"}
#: SLURM with a container runtime, so an engine would run in the engine's image.
SLURM_PYXIS = {"name": "slurm", "options": {"container_runtime": "pyxis"}}


class TestRunnersAndServe:
    def test_invalid_job_options_leave_no_run_directory(self, data: Path, tmp_path: Path, fake_runner: list) -> None:
        config = tmp_path / "run.yaml"
        bad = tiny_config(data, runner={"name": "fake", "options": {"resources": {"gpu": 1}}}).resolved()
        config.write_text(yaml.safe_dump(bad), encoding="utf-8")
        error = _failed("start", str(config), "--runs-dir", str(tmp_path / "runs"))
        assert error["exit_code"] == 3 and "gpu" in error["message"]
        assert not (tmp_path / "runs").exists() and fake_runner == []

    def test_a_runner_whose_jobs_do_not_see_this_host_needs_a_mirror_before_anything_is_written(
        self, data: Path, tmp_path: Path
    ) -> None:
        config = tmp_path / "run.yaml"
        config.write_text(yaml.safe_dump(tiny_config(data, runner={"name": "kubernetes"}).resolved()))
        error = _failed("start", str(config), "--runs-dir", str(tmp_path / "runs"))
        assert "mirror" in error["message"] and "--mirror" in error["hint"]
        assert not (tmp_path / "runs").exists()

    def test_the_install_source_option_is_taken_from_the_config_and_the_cli(self, data: Path, tmp_path: Path) -> None:
        """`runner.options.wheelhouse`/`constraints` (a pre-release or air-gapped install) reach the rendered
        coordinator scripts; the generic --set override sets them from the command line."""
        config = tmp_path / "run.yaml"
        config.write_text(
            yaml.safe_dump(
                tiny_config(
                    data,
                    runner={
                        "name": "slurm",
                        "options": {
                            "container_runtime": "pyxis",
                            "wheelhouse": "/shared/wheels",
                            "constraints": "/shared/wheels/constraints.txt",
                        },
                    },
                ).resolved()
            ),
            encoding="utf-8",
        )
        (script,) = _start(str(config), "--runs-dir", str(tmp_path / "runs"), "--dry-run")["rendered"].values()
        assert "--constraints /shared/wheels/constraints.txt" in script
        assert "--find-links /shared/wheels --no-index" in script
        overriden = _start(
            str(config),
            "--runs-dir", str(tmp_path / "set-runs"),
            "--dry-run",
            "--set", "runner.options.wheelhouse=https://storage.example/wheels",
            "--set", "runner.options.constraints=https://storage.example/wheels/c.txt",
        )  # fmt: skip
        (script,) = overriden["rendered"].values()
        assert "--find-links https://storage.example/wheels --no-index" in script
        assert not (tmp_path / "runs").exists()

    def test_a_dry_run_on_a_runner_prints_what_it_would_submit(self, data: Path, tmp_path: Path) -> None:
        config = tmp_path / "run.yaml"
        config.write_text(yaml.safe_dump(tiny_config(data, runner=SLURM_PYXIS).resolved()), encoding="utf-8")
        plan = _start(str(config), "--runs-dir", str(tmp_path / "runs"), "--dry-run")
        (script,) = plan["rendered"].values()
        assert script.startswith("#!/usr/bin/env bash\n#SBATCH --job-name=rcp-")
        assert "rcp-ndcg run resume --run" in script
        assert not (tmp_path / "runs").exists()
        text = CliRunner().invoke(cli, ["run", "start", str(config), "--runs-dir", str(tmp_path), "--dry-run"])
        assert "what the runner would submit" in text.stdout and "#SBATCH --ntasks=1" in text.stdout

    def test_serve_is_refused_where_no_engine_is_started(self, data: Path, tmp_path: Path) -> None:
        config = tmp_path / "run.yaml"
        config.write_text(yaml.safe_dump(tiny_config(data, judge=SERVED_JUDGE, serve=SERVE).resolved()))
        runs = str(tmp_path / "runs")
        for extra in ((), ("--runner", "local"), ("--detach",)):
            error = _failed("start", str(config), "--runs-dir", runs, *extra)
            assert error["exit_code"] == 3 and "starts no engine" in error["message"], extra
        assert not (tmp_path / "runs").exists()

    def test_a_serving_run_is_refused_on_a_runner_that_does_not_render_phases(
        self, data: Path, tmp_path: Path, fake_runner: list
    ) -> None:
        """A serving run is refused on a runner that renders no phases (a plugin), never degraded."""
        config = tmp_path / "run.yaml"
        config.write_text(
            yaml.safe_dump(tiny_config(data, judge=SERVED_JUDGE, serve=SERVE, runner={"name": "fake"}).resolved())
        )
        error = _failed("start", str(config), "--runs-dir", str(tmp_path / "runs"), "--dry-run")
        assert error["exit_code"] == 3 and "does not start a phase's engines" in error["message"]
        assert "--engine" in error["hint"] and not (tmp_path / "runs").exists()

    def test_a_served_encoder_run_reaches_the_refusal_through_prepare(
        self, data: Path, tmp_path: Path, fake_runner: list
    ) -> None:
        """`run start` of a served-encoder run (no judge) reaches the runners' phase refusal — not the recorded
        config's re-validation of its own defaults."""
        config = tmp_path / "run.yaml"
        fields = {
            "candidates": {
                "from": "retrieval",
                "retrieval": {"kind": "dense", "encoder": {"api": "openai_embeddings", "model": "e", **_SERVED_BUDGET}},
            },
            "steps": ["retrieve"],
            "serve": {"encoder": {"command": ["vllm", "serve", "e", "--host", "0.0.0.0", "--port", "8000"]}},
        }
        config.write_text(
            yaml.safe_dump(tiny_config(data, **{"runner": {"name": "fake"}, **fields}).resolved()), encoding="utf-8"
        )
        error = _failed("start", str(config), "--runs-dir", str(tmp_path / "runs"), "--dry-run")
        assert error["exit_code"] == 3 and "does not start a phase's engines" in error["message"]
        assert not (tmp_path / "runs").exists()

    def test_a_served_multi_role_run_renders_phases_for_the_job_runners(self, data: Path, tmp_path: Path) -> None:
        """The path users take: a served encoder and judge run through `job_for` -> `JobSpec(phases)` and both
        job renderers - the SLURM script (shellcheck-clean) and the Kubernetes objects (schema-checked)."""
        import subprocess

        import fsspec

        from rcp_ndcg.runs.config import RunConfig
        from rcp_ndcg.runs.execution import job_for
        from rcp_ndcg.runs.run import prepare
        from tests.runners.k8s_schema import check_objects
        from tests.runners.shell import assert_shellcheck_clean

        fsspec.filesystem("memory").pipe("/data/rows.jsonl", data.read_bytes())
        fields = {
            "label": "multi-role",
            "dataset": "jsonl:memory://data/rows.jsonl",
            "judge": {"base_url": "http://unused/v1", "model": "m"},
            "candidates": {
                "from": "retrieval",
                "retrieval": {"kind": "dense", "encoder": {"api": "openai_embeddings", "model": "embedder"}},
            },
            "steps": ["retrieve", "tournament", "rubric", "calibrate", "evaluate"],
            "serve": {
                "encoder": {
                    "image": "org/encoder:v2",
                    "command": ["python3", "-m", "enc", "--host", "0.0.0.0", "--port", "8001"],
                    "resources": {"gpus": 1},
                },
                "judge": {
                    "image": "vllm/vllm-openai:v0.30.0",
                    "command": ["vllm", "serve", "org/model", "--host", "0.0.0.0", "--port", "8000"],
                    "resources": {"gpus": 8},
                },
            },
            "runner": {"name": "slurm", "options": {"container_runtime": "pyxis"}},
        }
        config = RunConfig.model_validate({**tiny_config(data).resolved(), **fields})
        pipeline = prepare(config, runs_dir=str(tmp_path / "runs"))
        backend, job, _ = job_for(pipeline, "slurm")
        # The phase plan: retrieve (encoder), the judging steps (judge), then calibrate+evaluate (no engine).
        assert [[*sorted(p.engines)] for p in job.phases] == [["encoder"], ["judge"], []]
        assert "--only retrieve" in " ".join(job.phases[0].argv)
        assert "--only tournament" in " ".join(job.phases[1].argv) and "--only rubric" in " ".join(job.phases[1].argv)
        (rendered,) = backend.render([job]).values()
        assert rendered.startswith("#!/usr/bin/env bash\n#SBATCH --job-name=rcp-")
        assert "ENGINE_ENCODER" in rendered and "ENGINE_JUDGE" in rendered
        assert "RCP_NDCG_ENGINES=" in rendered  # each phase's engine URLs, as the runtime overlay
        assert_shellcheck_clean(rendered)
        assert subprocess.run(["bash", "-n", "-c", rendered], capture_output=True).returncode == 0

        kubernetes = RunConfig.model_validate(
            {
                **config.resolved(),
                "runner": {"name": "kubernetes", "options": {"namespace": "eval"}},
                "mirror": "memory://runs/multi-role",
            }
        )
        pipeline = prepare(kubernetes, runs_dir=str(tmp_path / "runs"))
        backend, job, _ = job_for(pipeline, "kubernetes")
        (rendered,) = backend.render([job]).values()
        objects = list(yaml.safe_load_all(rendered))
        check_objects(objects)
        assert [c["name"] for c in objects[0]["spec"]["template"]["spec"]["initContainers"]] == ["phase-1", "phase-2"]
        assert [c["name"] for c in objects[0]["spec"]["template"]["spec"]["containers"]] == ["phase-3"]
        assert not (tmp_path / "runs").exists()

    def test_a_replica_over_several_nodes_is_a_config_error(self, data: Path, tmp_path: Path) -> None:
        config = tmp_path / "run.yaml"
        config.write_text(
            yaml.safe_dump(tiny_config(data, judge=SERVED_JUDGE, serve=SERVE, runner=SLURM_PYXIS).resolved())
        )
        error = _failed("start", str(config), "--set", "serve.judge.nodes_per_replica=2", "--dry-run")
        assert error["exit_code"] == 3 and error["details"]["errors"][0]["field"] == "serve.judge.nodes_per_replica"

    def test_serve_needs_a_served_judge(self, data: Path) -> None:
        from rcp_ndcg.errors import ConfigError

        with pytest.raises(ConfigError, match="no served judge"):
            tiny_config(data, serve=SERVE)  # judge: fake

    def test_an_old_single_engine_serve_shape_shows_the_new_one(self, data: Path) -> None:
        from rcp_ndcg.errors import ConfigError

        old = {"image": "vllm/vllm-openai:v0.30.0", "command": ["vllm", "serve", "m"]}
        with pytest.raises(ConfigError, match="old single-engine shape") as refused:
            tiny_config(data, judge=SERVED_JUDGE, serve=old)
        assert "serve: {judge:" in (refused.value.hint or "")
        with pytest.raises(ConfigError, match="unknown engine role"):
            tiny_config(data, judge=SERVED_JUDGE, serve={"gpu": {"command": ["x"]}})


class TestTheEngineFlag:
    """``run resume --engine role=url[,url]``: the runtime overlay's command-line spelling."""

    def test_the_flag_sets_the_engines_the_run_applies_at_runtime(
        self, finished: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The finished run is judged by the offline fake, so a judge URL is refused there — which shows the flag
        reached the run as an overlay (a fake judge takes no engine URL)."""
        code, error = _refusal("resume", "--run", str(finished), "--engine", "judge=http://node:8000/v1")
        assert code == 3 and "offline fake" in error["message"]
        monkeypatch.delenv("RCP_NDCG_ENGINES", raising=False)  # the flag, not a stray variable, did this

    def test_bad_role_and_empty_and_repeated_roles_are_usage_errors(self, finished: Path) -> None:
        for spec, match in (
            ("gpu=http://n:8000/v1", "not <role>=<url>"),
            ("judge", "not <role>=<url>"),
            ("judge=", "has no URL"),
            ("judge=,", "has no URL"),
        ):
            code, error = _refusal("resume", "--run", str(finished), "--engine", spec)
            assert code == 2 and match in error["message"], spec
        code, error = _refusal(
            "resume", "--run", str(finished), "--engine", "judge=http://a/v1", "--engine", "judge=http://b/v1"
        )
        assert code == 2 and "twice" in error["message"]

    def test_the_flag_does_not_change_a_resubmission(self, finished: Path) -> None:
        code, error = _refusal(
            "resume", "--run", str(finished), "--engine", "judge=http://node:8000/v1", "--runner", "local"
        )
        assert code == 2 and "--engine apply to a resume in this process" in error["message"]


def test_a_packaged_config_starts_by_name_from_any_directory(tmp_path: Path, monkeypatch) -> None:
    """Outside a checkout the example configs once did not exist: `run start tiny` resolves the packaged one."""
    monkeypatch.chdir(tmp_path)
    estimate = _start("tiny", "--estimate")
    assert estimate["mode"] == "estimate" and estimate["estimate"]["calls"] > 0
    missing = CliRunner().invoke(cli, ["run", "start", "tinny", "--estimate", "--json"])
    error = json.loads(missing.stdout)["error"]
    assert missing.exit_code == 4 and "tiny" in error["hint"]


def test_a_failed_resume_with_set_and_only_leaves_the_run_resumable(finished: Path, tmp_path: Path) -> None:
    """The usability study's case: a --set that fails the evaluate step must not stick to the run."""
    import shutil

    run_dir = Path(shutil.copytree(finished, tmp_path / finished.name))
    run_yaml = (run_dir / "run.yaml").read_text(encoding="utf-8")
    missing = tmp_path / "sytems.jsonl"  # a typo: the evaluate step fails on it

    failed = CliRunner().invoke(
        cli,
        ["run", "resume", "--run", str(run_dir), "--set", f"evaluation.systems.mine={missing}", "--only", "evaluate",
         "--json"],
    )  # fmt: skip
    assert failed.exit_code == 4, failed.output

    assert (run_dir / "run.yaml").read_text(encoding="utf-8") == run_yaml
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and manifest["config"]["steps"] == [
        "tournament", "rubric", "calibrate", "evaluate"
    ]  # fmt: skip
    resumed = _command("resume", "--run", str(run_dir))
    assert resumed["state"]["status"] == "completed"


def _refusal(*args: str) -> tuple[int, dict]:
    result = CliRunner().invoke(cli, ["run", *args, "--json"])
    return result.exit_code, json.loads(result.stdout)["error"]


@pytest.mark.parametrize("mode", ["--dry-run", "--estimate"])
def test_a_preflight_gives_the_refusal_of_a_resume_that_changes_the_judging_identity(
    finished: Path, tmp_path: Path, mode: str
) -> None:
    import shutil

    run_dir = Path(shutil.copytree(finished, tmp_path / finished.name))
    manifest = (run_dir / "manifest.json").read_text(encoding="utf-8")
    preflight = _refusal("resume", "--run", str(run_dir), "--set", "rubric.seed=7", mode)
    assert (run_dir / "manifest.json").read_text(encoding="utf-8") == manifest
    real = _refusal("resume", "--run", str(run_dir), "--set", "rubric.seed=7")

    assert real[0] == preflight[0] == 11
    assert preflight[1]["details"]["differences"] == real[1]["details"]["differences"]
    # The preflight's way out is the run's, not a judge command's (`run resume` has no --out or --force).
    assert preflight[1]["hint"] == real[1]["hint"] and "--force" not in preflight[1]["hint"]


def test_an_estimate_or_plan_of_a_new_run_names_no_run_id(data: Path, tmp_path: Path) -> None:
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(tiny_config(data).resolved()), encoding="utf-8")
    for mode in ("--estimate", "--dry-run"):
        assert _start(str(config), "--runs-dir", str(tmp_path / "runs"), mode)["run_id"] is None


def test_the_offline_judge_is_estimated_quick(data: Path, tmp_path: Path) -> None:
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(tiny_config(data).resolved()), encoding="utf-8")
    estimate = _start(str(config), "--runs-dir", str(tmp_path / "runs"), "--estimate")["estimate"]
    assert estimate["wall_s"] < 5
    assert any("offline" in line for line in estimate["assumptions"])


def test_an_estimate_of_a_resume_counts_only_the_judging_steps_it_would_run(finished: Path) -> None:
    """A completed run's resume was estimated at every call of every judging step, although it would ask none."""
    estimate = _command("resume", "--run", str(finished), "--estimate")["estimate"]
    assert estimate["calls"] == 0 and any("skipped" in line for line in estimate["assumptions"])
    redo = _command("resume", "--run", str(finished), "--estimate", "--no-resume")["estimate"]
    assert redo["calls"] > 0
    assert any("already holds" in line and "judged windows" in line for line in redo["assumptions"])


def test_the_judge_flags_say_what_they_take_and_refuse_what_they_would_ignore(data: Path, tmp_path: Path) -> None:
    """--judge took a shipped name its help did not list, and --judge-model without --judge-url was ignored."""
    from rcp_ndcg.llm.judges import judge_names

    help_text = " ".join(CliRunner().invoke(cli, ["run", "start", "--help"]).output.split())
    assert all(name in help_text for name in judge_names())
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump(tiny_config(data).resolved()), encoding="utf-8")
    error = _failed("start", str(config), "--judge", "fake", "--judge-model", "other", "--dry-run")
    assert error["exit_code"] == 2 and "--judge-url" in error["message"]


class TestEngineFlagEdges:
    """Round-2 polish: the flag's own refusals stay usage errors, and an inline fake judge is refused."""

    def test_a_duplicate_replica_is_a_usage_error_naming_the_flag(self, finished: Path) -> None:
        code, error = _refusal("resume", "--run", str(finished), "--engine", "judge=http://a/v1,http://a/v1")
        assert code == 2 and "twice" in error["message"] and "--engine" in error["hint"]

    def test_an_inline_fake_judge_is_refused_like_the_name(self, data: Path) -> None:
        from rcp_ndcg.errors import ConfigError

        with pytest.raises(ConfigError, match="no served judge"):
            tiny_config(
                data,
                judge={"base_url": "fake://seed/0", "model": "fake"},
                serve={"judge": {"command": ["vllm", "serve", "m"]}},
            )
