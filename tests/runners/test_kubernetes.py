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
from rcp_ndcg.runners.base import JobPhase
from rcp_ndcg.runners.kubernetes import JOB_UID, k8s_name
from rcp_ndcg.runners.script import EngineStep, engines_env_value, supervise, worker_script
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


def _phase(step: int, engines: dict[str, ServeConfig]) -> JobPhase:
    """One phase of the paper run's shape: its engines and the steps it runs."""
    steps = ("retrieve", "rerank", "tournament", "rubric", "calibrate", "evaluate")
    return JobPhase(
        engines=engines,
        argv=("rcp-ndcg", "run", "resume", "--run", "/scratch/runs/x", "--only", steps[step - 1]),
    )


class TestPhases:
    def test_one_phase_is_one_container_in_the_engines_image_running_the_supervision_script(self) -> None:
        """Any launcher that takes an image and a command runs it: no init container, no sidecar, no probes."""
        phase = JobPhase(engines={"judge": SERVE}, argv=("rcp-ndcg", "run", "resume", "--run", "/scratch/runs/x"))
        job = JobSpec(name="run", argv=("rcp-ndcg", "run", "resume"), phases=(phase,))
        (job_obj,) = yaml.safe_load_all(KubernetesRunner(namespace="eval", secrets=["hf-token"]).render([job])["run"])
        pod = job_obj["spec"]["template"]["spec"]
        assert "initContainers" not in pod and pod["restartPolicy"] == "Never"
        (container,) = pod["containers"]
        assert container["name"] == "phase-1"
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
            job.model_copy(update={"argv": phase.argv}),
            install=True,
            workdir=None,
            env={"UV_CACHE_DIR": "/scratch/uv-cache", "UV_LINK_MODE": "copy"},
        )
        supervision = supervise(
            [EngineStep(serve=SERVE, start='bash -c "$ENGINE_JUDGE"', hosts="127.0.0.1")],
            coordinator='bash -c "$WORKER_1"',
            engines_env=('\'{"judge": {"urls": ["http://127.0.0.1:8000/v1"], "wait_on_outage_s": 900}}\''),
            uv=True,
        )
        assert (bash, flag) == ("bash", "-c")
        assert script == (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "read -r -d '' WORKER_1 <<'RCP_NDCG_WORKER_1' || true\n"
            f"{worker}"
            "RCP_NDCG_WORKER_1\n"
            "read -r -d '' ENGINE_JUDGE <<'RCP_NDCG_ENGINE_JUDGE' || true\n"
            "export HF_HOME=/models\n"
            "exec vllm serve org/model --served-model-name m --host 0.0.0.0 --port 8000\n"
            "RCP_NDCG_ENGINE_JUDGE\n" + "\n".join(supervision) + "\n"
        )
        assert "command -v uvx >/dev/null" in worker  # the engine's image gets uv from pip when it has none
        check_objects([job_obj])

    def test_each_engine_phase_is_an_init_container_and_the_last_phase_the_main_container(self) -> None:
        """The paper run's three phases: encoder, judge, and an engine-free phase, the run directory on emptyDir."""
        phases = (
            JobPhase(engines={"encoder": ENCODER}, argv=("retrieve",)),
            JobPhase(engines={"judge": SERVE}, argv=("tournament",)),
            JobPhase(argv=("calibrate", "evaluate")),
        )
        job = JobSpec(name="paper", argv=("rcp-ndcg", "run", "resume"), phases=phases)
        (job_obj,) = yaml.safe_load_all(KubernetesRunner(namespace="eval").render([job])["paper"])
        pod = job_obj["spec"]["template"]["spec"]
        assert [c["name"] for c in pod["initContainers"]] == ["phase-1", "phase-2"]
        assert [c["name"] for c in pod["containers"]] == ["phase-3"]
        assert [c["image"] for c in pod["initContainers"] + pod["containers"]] == [
            ENCODER.image,
            SERVE.image,
            COORDINATOR_IMAGE,
        ]
        assert {c["name"]: c["volumeMounts"] for c in pod["initContainers"] + pod["containers"]} == {
            "phase-1": [{"name": "scratch", "mountPath": "/scratch"}, {"name": "dshm", "mountPath": "/dev/shm"}],
            "phase-2": [{"name": "scratch", "mountPath": "/scratch"}, {"name": "dshm", "mountPath": "/dev/shm"}],
            "phase-3": [{"name": "scratch", "mountPath": "/scratch"}],
        }
        # The engine-free phase runs its command directly, with an empty RCP_NDCG_ENGINES.
        main = pod["containers"][0]["command"][2]
        assert "export RCP_NDCG_ENGINES='{}'" in main
        assert "RCP_NDCG_ENGINE_PID" not in main and "exec c" in main
        # Each engine phase waits for its own engine and exports its URLs before its coordinator starts.
        assert "RCP_NDCG_ENGINE_PID=$!" in pod["initContainers"][0]["command"][2]
        assert (
            "RCP_NDCG_ENGINES='{" in pod["initContainers"][0]["command"][2]
            and '"encoder"' in pod["initContainers"][0]["command"][2]
        )
        check_objects([job_obj])

    def test_several_replicas_are_a_statefulset_owned_by_the_job_and_run_scoped(self) -> None:
        """A several-replica engine lives for the whole run: one StatefulSet, created at submit, for every phase."""
        phases = (
            JobPhase(engines={"reranker": RERANKER}, argv=("a",)),
            JobPhase(engines={"reranker": RERANKER}, argv=("b",)),
            JobPhase(argv=("c",)),
        )
        job = JobSpec(name="run", argv=("rcp-ndcg", "run", "resume"), phases=phases)
        job_obj, stateful_set, service = yaml.safe_load_all(KubernetesRunner(namespace="eval").render([job])["run"])
        assert stateful_set["spec"]["replicas"] == 2 and stateful_set["spec"]["podManagementPolicy"] == "Parallel"
        assert stateful_set["spec"]["serviceName"] == service["metadata"]["name"] == "run-engine-reranker"
        assert service["spec"]["clusterIP"] == "None"
        for owned in (stateful_set, service):
            (owner,) = owned["metadata"]["ownerReferences"]
            assert (owner["kind"], owner["name"], owner["uid"]) == ("Job", "run", JOB_UID)
        script = job_obj["spec"]["template"]["spec"]["initContainers"][0]["command"][2]
        hosts = [f"run-engine-reranker-{i}.run-engine-reranker.eval.svc" for i in range(2)]
        engines = engines_env_value({"reranker": RERANKER}, {"reranker": [f"http://{host}:8002/v1" for host in hosts]})
        assert f"RCP_NDCG_ENGINES='{engines}'" in script
        # The phase's coordinator waits for a replica at most startup_timeout_s; so does the startup probe.
        assert (
            "rcp_ndcg_wait_ready RCP_NDCG_ENGINE_PID_REMOTE 1800 8002 /v1/models "
            "run-engine-reranker-0.run-engine-reranker.eval.svc" in script
        )
        (engine,) = stateful_set["spec"]["template"]["spec"]["containers"]
        assert engine["startupProbe"]["failureThreshold"] * engine["startupProbe"]["periodSeconds"] == 1800
        check_objects([job_obj, stateful_set, service])

    def test_the_phases_container_asks_for_what_the_engines_and_the_coordinator_need_together(self) -> None:
        engine = SERVE.model_copy(update={"resources": Resources(gpus=8, cpus=32, memory_gb=400)})
        phases = (JobPhase(engines={"judge": engine}, argv=("a",)),)
        job = JobSpec(name="run", argv=("x",), resources=Resources(cpus=4, memory_gb=16), phases=phases)
        container = KubernetesRunner().manifest(job)["spec"]["template"]["spec"]["containers"][0]
        limits = {"nvidia.com/gpu": 8, "cpu": 36, "memory": "416Gi"}
        assert container["resources"] == {"requests": {"cpu": 36, "memory": "416Gi"}, "limits": limits}
        # An engine of unstated CPUs or memory is not capped by the coordinator's share.
        phases = (JobPhase(engines={"judge": SERVE}, argv=("a",)),)
        job = JobSpec(name="run", argv=("x",), resources=Resources(cpus=4, memory_gb=16), phases=phases)
        container = KubernetesRunner().manifest(job)["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"] == {"limits": {"nvidia.com/gpu": 8}}

    def test_a_phase_with_engines_of_several_images_is_refused(self) -> None:
        """One init container runs the phase's engines together: they cannot have two images."""
        phases = (JobPhase(engines={"judge": SERVE, "encoder": ENCODER}, argv=("a",)),)
        with pytest.raises(ConfigError, match="images differ"):
            KubernetesRunner().manifest(JobSpec(name="run", argv=("x",), phases=phases))

    def test_a_phase_with_several_engines_on_one_port_is_refused(self) -> None:
        """Two engines in one container can only listen on different ports."""
        phases = (JobPhase(engines={"judge": SERVE, "encoder": SERVE}, argv=("a",)),)
        with pytest.raises(ConfigError, match="one port"):
            KubernetesRunner().manifest(JobSpec(name="run", argv=("x",), phases=phases))

    def test_a_role_with_differing_engines_across_phases_is_refused(self) -> None:
        """One StatefulSet serves every phase that uses the role, so it cannot differ between them."""
        phases = (
            JobPhase(engines={"reranker": RERANKER}, argv=("a",)),
            JobPhase(engines={"reranker": RERANKER.model_copy(update={"port": 8009})}, argv=("b",)),
        )
        with pytest.raises(ConfigError, match="different configurations"):
            KubernetesRunner().manifest(JobSpec(name="run", argv=("x",), phases=phases))

    def test_submit_owns_the_engines_by_the_applied_jobs_uid(self, monkeypatch) -> None:
        fake = _FakeKubectl()
        monkeypatch.setattr("rcp_ndcg.runners.kubernetes.run_cli", fake)
        phases = (JobPhase(engines={"reranker": RERANKER}, argv=("a",)),)
        job = JobSpec(name="run", argv=("x",), phases=phases)
        assert KubernetesRunner(namespace="eval").submit([job]) == ["eval/run"]
        (_, job_yaml), (_, engines_yaml) = [(argv, text) for argv, text in fake.calls if "apply" in argv]
        assert yaml.safe_load(job_yaml)["kind"] == "Job"
        owners = [obj["metadata"]["ownerReferences"][0]["uid"] for obj in yaml.safe_load_all(engines_yaml)]
        assert owners == ["uid-run", "uid-run"]

    @pytest.mark.parametrize(
        "phases",
        [
            (JobPhase(engines={"judge": SERVE}, argv=("a",)),),
            (
                JobPhase(engines={"encoder": ENCODER}, argv=("a",)),
                JobPhase(engines={"judge": SERVE}, argv=("b",)),
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
    def test_the_rendered_script_is_valid_bash(self, phases: tuple[JobPhase, ...]) -> None:
        import subprocess

        (job_obj, *_) = yaml.safe_load_all(
            KubernetesRunner().render([JobSpec(name="j", argv=("x",), phases=phases)])["j"]
        )
        pod = job_obj["spec"]["template"]["spec"]
        for container in pod.get("initContainers", []) + pod["containers"]:
            script = container["command"][2]
            assert subprocess.run(["bash", "-n", "-c", script], capture_output=True).returncode == 0
            assert_shellcheck_clean(script)
        check_objects([job_obj, *KubernetesRunner().engine_objects(JobSpec(name="j", argv=("x",), phases=phases))])


def test_the_runners_resources_and_env_are_every_jobs_defaults() -> None:
    runner = get_runner("kubernetes", resources={"gpus": 2}, env={"A": "1"})
    (container,) = runner.manifest(JobSpec(name="j", argv=("true",)))["spec"]["template"]["spec"]["containers"]
    assert container["resources"]["limits"] == {"nvidia.com/gpu": 2}
    assert "export A=1" in container["command"][2]


def test_the_engine_statefulset_of_a_long_run_label_leaves_room_for_its_pod_names() -> None:
    """A 63-character StatefulSet name gave pod names and revision labels over 63: no engine pod was ever created."""
    from rcp_ndcg.runs.layout import new_run_id, slugify

    run_id = new_run_id("nano-nfcorpus-gpt-oss-120b-with-a-very-long-label")
    phases = (JobPhase(engines={"judge": SERVE.model_copy(update={"replicas": 3})}, argv=("tournament",)),)
    job = JobSpec(name=slugify(f"rcp-{run_id}", max_length=60), argv=("rcp-ndcg", "run", "resume"), phases=phases)
    runner = KubernetesRunner(namespace="eval")
    stateful_set, service = runner.engine_objects(job)
    name = stateful_set["metadata"]["name"]
    assert name == service["metadata"]["name"] == stateful_set["spec"]["serviceName"]
    assert len(name) <= 52 and len(f"{name}-2") <= 63 and len(f"{name}-0123456789") <= 63
    assert name == runner.engine_objects(job)[0]["metadata"]["name"]  # deterministic
    containers = runner.manifest(job)["spec"]["template"]["spec"]["containers"]
    assert f"{name}-0.{name}.eval.svc" in containers[0]["command"][2]
