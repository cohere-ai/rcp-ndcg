"""KubernetesRunner: golden Job manifests, and kubectl calls mocked."""

from __future__ import annotations

import json
import shlex

import pytest
import yaml

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.runners import (
    COORDINATOR_IMAGE,
    JobSpec,
    JobStatus,
    KubernetesRunner,
    Resources,
    RunnerError,
    ServeConfig,
    get_runner,
    install_argv,
)
from rcp_ndcg.runners.kubernetes import JOB_UID, k8s_name
from rcp_ndcg.runners.script import supervise, worker_script
from tests.runners.k8s_schema import check_objects
from tests.runners.shell import assert_shellcheck_clean

JUDGE = JobSpec(
    name="exp-judge",
    argv=("rcp-ndcg", "run", "start", "run.yaml"),
    resources=Resources(gpus=8, cpus=16, memory_gb=512, time_limit_s=3600),
    env={"HF_HOME": "/cache/hf"},
)


def test_manifest_golden() -> None:
    runner = KubernetesRunner(image="ghcr.io/you/uv:py3.12", namespace="eval", secrets=["hf-token"])
    (manifest,) = yaml.safe_load_all(runner.render([JUDGE])["exp-judge"])
    labels = {"app.kubernetes.io/name": "rcp-ndcg", "rcp-ndcg/job": "exp-judge"}
    assert manifest == {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": "exp-judge", "namespace": "eval", "labels": labels},
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": 3600,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "coordinator",
                            "image": "ghcr.io/you/uv:py3.12",
                            "command": [
                                "bash",
                                "-c",
                                "#!/usr/bin/env bash\n"
                                "set -euo pipefail\n"
                                "export UV_CACHE_DIR=/scratch/uv-cache\n"
                                "export UV_LINK_MODE=copy\n"
                                "export HF_HOME=/cache/hf\n"
                                "if ! command -v uvx >/dev/null; then\n"
                                '  python3 -m pip install --quiet --target "${TMPDIR:-/tmp}/rcp-ndcg-uv" uv\n'
                                '  export PATH="${TMPDIR:-/tmp}/rcp-ndcg-uv/bin:$PATH"\n'
                                "fi\n"
                                f"exec {shlex.join(install_argv(JUDGE.argv))}\n",
                            ],
                            "volumeMounts": [{"name": "scratch", "mountPath": "/scratch"}],
                            "envFrom": [{"secretRef": {"name": "hf-token"}}],
                            "resources": {
                                "requests": {"cpu": 16, "memory": "512Gi"},
                                "limits": {"nvidia.com/gpu": 8, "cpu": 16, "memory": "512Gi"},
                            },
                        }
                    ],
                    "volumes": [{"name": "scratch", "emptyDir": {}}],
                },
            },
        },
    }
    check_objects([manifest])


def test_the_jobs_image_wins() -> None:
    spec = JobSpec(name="j", argv=("true",), image="own:1")
    manifest = KubernetesRunner(image="default:1").manifest(spec)
    assert manifest["spec"]["template"]["spec"]["containers"][0]["image"] == "own:1"
    assert "resources" not in manifest["spec"]["template"]["spec"]["containers"][0]


def test_the_coordinator_runs_in_the_stock_uv_image_by_default() -> None:
    manifest = KubernetesRunner().manifest(JobSpec(name="j", argv=("true",)))
    assert manifest["spec"]["template"]["spec"]["containers"][0]["image"] == COORDINATOR_IMAGE
    with pytest.raises(ConfigError, match="namespce"):
        KubernetesRunner(namespce="typo")


def test_long_names_are_shortened_deterministically() -> None:
    long = "a" * 70
    assert len(k8s_name(long)) <= 63
    assert k8s_name(long) == k8s_name(long) != k8s_name("b" * 70)
    assert k8s_name("short") == "short"


class _FakeKubectl:
    def __init__(self, jobs: dict[str, dict] | None = None) -> None:
        self.jobs = jobs or {}
        self.calls: list[tuple[list[str], str | None]] = []

    def __call__(self, argv, *, input_text=None):
        self.calls.append((list(argv), input_text))
        if argv[1:3] == ["get", "job"] or argv[3:5] == ["get", "job"]:
            name = argv[argv.index("job") + 1]
            if name not in self.jobs:
                raise RunnerError(f'Error from server (NotFound): jobs.batch "{name}" not found')
            return json.dumps({"status": self.jobs[name]})
        if "logs" in argv:
            return "pod-0 a\npod-1 b\n"
        if "apply" in argv and "json" in argv:
            applied = yaml.safe_load(input_text)
            return json.dumps(
                {**applied, "metadata": {**applied["metadata"], "uid": f"uid-{applied['metadata']['name']}"}}
            )
        return ""


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ({"active": 2}, JobStatus.RUNNING),
        ({}, JobStatus.PENDING),
        ({"conditions": [{"type": "Complete", "status": "True"}]}, JobStatus.SUCCEEDED),
        ({"conditions": [{"type": "Failed", "status": "True"}]}, JobStatus.FAILED),
    ],
)
def test_status_from_job_conditions(monkeypatch, status: dict, expected: JobStatus) -> None:
    monkeypatch.setattr("rcp_ndcg.runners.kubernetes.run_cli", _FakeKubectl({"j": status}))
    assert KubernetesRunner(image="i").status("ns/j") is expected


def test_status_of_a_deleted_job_is_unknown(monkeypatch) -> None:
    monkeypatch.setattr("rcp_ndcg.runners.kubernetes.run_cli", _FakeKubectl())
    assert KubernetesRunner(image="i").status("ns/gone") is JobStatus.UNKNOWN


def test_submit_applies_in_order(monkeypatch) -> None:
    fake = _FakeKubectl()
    monkeypatch.setattr("rcp_ndcg.runners.kubernetes.run_cli", fake)
    runner = KubernetesRunner(image="i", namespace="eval", context="ctx")
    jobs = [JobSpec(name="first", argv=("true",)), JobSpec(name="second", argv=("true",))]
    assert runner.submit(jobs) == ["eval/first", "eval/second"]
    applied = [yaml.safe_load(text)["metadata"]["name"] for argv, text in fake.calls if "apply" in argv]
    assert applied == ["first", "second"]
    assert all(argv[:3] == ["kubectl", "--context", "ctx"] for argv, _ in fake.calls)


def test_logs_and_cancel(monkeypatch) -> None:
    fake = _FakeKubectl()
    monkeypatch.setattr("rcp_ndcg.runners.kubernetes.run_cli", fake)
    runner = KubernetesRunner(image="i")
    assert runner.logs("eval/j", tail=1) == "pod-1 b\n"
    runner.cancel("eval/j")
    assert fake.calls[-1][0] == ["kubectl", "delete", "job", "j", "-n", "eval", "--ignore-not-found", "--wait=false"]
    assert fake.calls[0][0][:6] == ["kubectl", "logs", "-n", "eval", "-l", "job-name=j"]


def test_it_is_registered_by_name() -> None:
    assert isinstance(get_runner("kubernetes", image="i"), KubernetesRunner)


SERVE = ServeConfig(
    image="vllm/vllm-openai:v0.30.0",
    command=["vllm", "serve", "org/model", "--served-model-name", "m", "--host", "0.0.0.0", "--port", "8000"],
    env={"HF_HOME": "/models"},
    resources=Resources(gpus=8),
)
SERVED = JobSpec(name="run", argv=("rcp-ndcg", "run", "resume", "--run", "/scratch/runs/x"), serve=SERVE)


class TestServe:
    def test_one_replica_is_one_container_in_the_engines_image_running_the_supervision_script(self) -> None:
        """Any launcher that takes an image and a command runs it: no init container, no sidecar, no probes."""
        (job,) = yaml.safe_load_all(KubernetesRunner(namespace="eval", secrets=["hf-token"]).render([SERVED])["run"])
        pod = job["spec"]["template"]["spec"]
        assert "initContainers" not in pod and pod["restartPolicy"] == "Never"
        (container,) = pod["containers"]
        assert container["image"] == SERVE.image
        assert container["resources"] == {"limits": {"nvidia.com/gpu": 8}}  # the GPUs are on the one container
        assert container["envFrom"] == [{"secretRef": {"name": "hf-token"}}]
        assert container["volumeMounts"] == [
            {"name": "scratch", "mountPath": "/scratch"},
            {"name": "dshm", "mountPath": "/dev/shm"},
        ]
        assert pod["volumes"] == [
            {"name": "scratch", "emptyDir": {}},
            {"name": "dshm", "emptyDir": {"medium": "Memory"}},
        ]
        assert not {"startupProbe", "readinessProbe", "livenessProbe"} & set(container)
        bash, flag, script = container["command"]
        worker = worker_script(
            SERVED,
            install=True,
            workdir=None,
            env={
                "UV_CACHE_DIR": "/scratch/uv-cache",
                "UV_LINK_MODE": "copy",
                "RCP_NDCG_JUDGE_URLS": "http://127.0.0.1:8000/v1",
            },
        )
        supervision = supervise(
            SERVE, engine='bash -c "$ENGINE"', coordinator='bash -c "$WORKER"', hosts="127.0.0.1", uv=True
        )
        assert (bash, flag) == ("bash", "-c")
        assert script == (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "read -r -d '' WORKER <<'RCP_NDCG_WORKER' || true\n"
            f"{worker}"
            "RCP_NDCG_WORKER\n"
            "read -r -d '' ENGINE <<'RCP_NDCG_ENGINE' || true\n"
            "export HF_HOME=/models\n"
            "exec vllm serve org/model --served-model-name m --host 0.0.0.0 --port 8000\n"
            "RCP_NDCG_ENGINE\n" + "\n".join(supervision) + "\n"
        )
        assert "command -v uvx >/dev/null" in worker  # the engine's image gets uv from pip when it has none
        check_objects([job])

    def test_one_container_asks_for_what_the_engine_and_the_coordinator_need_together(self) -> None:
        engine = SERVE.model_copy(update={"resources": Resources(gpus=8, cpus=32, memory_gb=400)})
        served = SERVED.model_copy(update={"serve": engine, "resources": Resources(cpus=4, memory_gb=16)})
        container = KubernetesRunner().manifest(served)["spec"]["template"]["spec"]["containers"][0]
        limits = {"nvidia.com/gpu": 8, "cpu": 36, "memory": "416Gi"}
        assert container["resources"] == {"requests": {"cpu": 36, "memory": "416Gi"}, "limits": limits}
        # An engine of unstated CPUs or memory is not capped by the coordinator's share.
        served = SERVED.model_copy(update={"resources": Resources(cpus=4, memory_gb=16)})
        container = KubernetesRunner().manifest(served)["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"] == {"limits": {"nvidia.com/gpu": 8}}

    def test_several_replicas_are_a_statefulset_behind_a_headless_service_owned_by_the_job(self) -> None:
        served = SERVED.model_copy(update={"serve": SERVE.model_copy(update={"replicas": 3})})
        job, stateful_set, service = yaml.safe_load_all(KubernetesRunner(namespace="eval").render([served])["run"])
        assert "initContainers" not in job["spec"]["template"]["spec"]
        assert stateful_set["kind"] == "StatefulSet" and service["kind"] == "Service"
        assert stateful_set["spec"]["replicas"] == 3 and stateful_set["spec"]["podManagementPolicy"] == "Parallel"
        assert stateful_set["spec"]["serviceName"] == service["metadata"]["name"] == "run-engine"
        assert service["spec"]["clusterIP"] == "None"
        for owned in (stateful_set, service):
            (owner,) = owned["metadata"]["ownerReferences"]
            assert (owner["kind"], owner["name"], owner["uid"]) == ("Job", "run", JOB_UID)
        script = job["spec"]["template"]["spec"]["containers"][0]["command"][2]
        urls = ",".join(f"http://run-engine-{i}.run-engine.eval.svc:8000/v1" for i in range(3))
        assert f"export RCP_NDCG_JUDGE_URLS={urls}" in script
        # The coordinator waits for a replica at most startup_timeout_s; so does each engine's startup probe.
        assert "rcp_ndcg_wait_ready run-engine-0.run-engine.eval.svc" in script
        assert "local deadline=$((SECONDS + 1800)) status" in script
        (engine,) = stateful_set["spec"]["template"]["spec"]["containers"]
        assert engine["startupProbe"]["failureThreshold"] * engine["startupProbe"]["periodSeconds"] == 1800
        check_objects([job, stateful_set, service])

    def test_submit_owns_the_engines_by_the_applied_jobs_uid(self, monkeypatch) -> None:
        fake = _FakeKubectl()
        monkeypatch.setattr("rcp_ndcg.runners.kubernetes.run_cli", fake)
        served = SERVED.model_copy(update={"serve": SERVE.model_copy(update={"replicas": 2})})
        assert KubernetesRunner(namespace="eval").submit([served]) == ["eval/run"]
        (_, job_yaml), (_, engines_yaml) = [(argv, text) for argv, text in fake.calls if "apply" in argv]
        assert yaml.safe_load(job_yaml)["kind"] == "Job"
        owners = [obj["metadata"]["ownerReferences"][0]["uid"] for obj in yaml.safe_load_all(engines_yaml)]
        assert owners == ["uid-run", "uid-run"]

    @pytest.mark.parametrize("replicas", [1, 2])
    def test_an_engine_without_an_image_is_refused(self, replicas: int) -> None:
        serve = SERVE.model_copy(update={"image": None, "replicas": replicas})
        with pytest.raises(ConfigError, match="names no image"):
            KubernetesRunner().render([SERVED.model_copy(update={"serve": serve})])

    @pytest.mark.parametrize("replicas", [1, 2])
    def test_the_rendered_script_is_valid_bash(self, replicas: int) -> None:
        import subprocess

        served = SERVED.model_copy(update={"serve": SERVE.model_copy(update={"replicas": replicas})})
        (job, *_) = yaml.safe_load_all(KubernetesRunner().render([served])["run"])
        script = job["spec"]["template"]["spec"]["containers"][0]["command"][2]
        assert subprocess.run(["bash", "-n", "-c", script], capture_output=True).returncode == 0
        assert_shellcheck_clean(script)


def test_the_runners_resources_and_env_are_every_jobs_defaults() -> None:
    runner = get_runner("kubernetes", resources={"gpus": 2}, env={"A": "1"})
    (container,) = runner.manifest(JobSpec(name="j", argv=("true",)))["spec"]["template"]["spec"]["containers"]
    assert container["resources"]["limits"] == {"nvidia.com/gpu": 2}
    assert "export A=1" in container["command"][2]


def test_the_engine_statefulset_of_a_long_run_label_leaves_room_for_its_pod_names() -> None:
    """A 63-character StatefulSet name gave pod names and revision labels over 63: no engine pod was ever created."""
    from rcp_ndcg.runs.layout import new_run_id, slugify

    run_id = new_run_id("nano-nfcorpus-gpt-oss-120b-with-a-very-long-label")
    job = SERVED.model_copy(
        update={"name": slugify(f"rcp-{run_id}", max_length=60), "serve": SERVE.model_copy(update={"replicas": 3})}
    )
    runner = KubernetesRunner(namespace="eval")
    stateful_set, service = runner.engine_objects(job)
    name = stateful_set["metadata"]["name"]
    assert name == service["metadata"]["name"] == stateful_set["spec"]["serviceName"]
    assert len(name) <= 52 and len(f"{name}-2") <= 63 and len(f"{name}-0123456789") <= 63
    assert name == runner.engine_objects(job)[0]["metadata"]["name"]  # deterministic
    hosts = runner.manifest(job)["spec"]["template"]["spec"]["containers"][0]["command"][2]
    assert f"{name}-0.{name}.eval.svc" in hosts
