"""KubernetesRunner -- one ``batch/v1`` Job per :class:`JobSpec`, through ``kubectl``.

The Job's coordinator container runs a stock image with uv and Python 3.12
(:data:`~rcp_ndcg.runners.script.COORDINATOR_IMAGE`), which installs this release
when the pod starts (:func:`~rcp_ndcg.runners.script.install_argv`). Nothing is
built or maintained by this package. The pod's disk is scratch: an ``emptyDir``
at ``/scratch`` holds uv's cache and the run directory (:attr:`KubernetesRunner.run_root`),
which the job restores from the run's mirror and mirrors back while it runs.

A job with engine replicas (``JobSpec.serve``) gets them as follows:

* **one replica** -- the Job's pod has one container, in the engine's image, whose
  command is the supervision script SLURM runs too
  (:func:`~rcp_ndcg.runners.script.supervise`): it installs uv if the image lacks
  it, starts the engine in the background, waits until it answers on localhost,
  runs the coordinator, and ends when either ends. An engine that dies or never
  answers fails the pod, and the Job. Any launcher that takes an image and a
  command runs it; the engine's image needs ``bash`` 4.3 or later and ``python3``
  (with ``pip`` when it has no uv): without bash the container does not start, and
  a pod whose image lacks one of the others stops at once, naming it;
* **several replicas** -- a StatefulSet of engine pods behind a headless Service,
  both owned by the Job (``ownerReferences``), so deleting the Job deletes them.
  Their URLs are the pods' stable names, known when the objects are rendered. The
  coordinator waits at most ``startup_timeout_s`` for one to answer; an engine that
  dies later is restarted by its StatefulSet, and a judge that meets no replica for
  ``outage_timeout_s`` fails the run.

A failed Job is retried ``backoff_limit`` times (0 by default); a retried pod resumes
the run from its mirror.

Everything goes through the ``kubectl`` on ``PATH`` and its current (or the
configured) context; nothing here needs cluster credentials of its own.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

import yaml
from pydantic import Field

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.runners._cli import run_cli
from rcp_ndcg.runners.base import JobHandle, JobOptions, JobSpec, JobStatus, RunnerError, tail_lines
from rcp_ndcg.runners.script import (
    COORDINATOR_IMAGE,
    engine_script,
    heredoc,
    supervise,
    wait_for_replicas,
    worker_script,
)
from rcp_ndcg.support.resources import Resources
from rcp_ndcg.support.serve import JUDGE_URLS_ENV, ServeConfig

_LABEL_VALUE = re.compile(r"[^A-Za-z0-9_.-]+")

#: The pod's scratch volume: uv's cache and the run directory.
SCRATCH = "/scratch"
#: Seconds between two startup probes of a StatefulSet's engine pod.
PROBE_PERIOD_S = 10
#: Stands for the Job's uid in a rendered engine StatefulSet or Service; :meth:`KubernetesRunner.submit` fills it in.
JOB_UID = "<the Job's uid, set at submit>"


def _label_value(value: str) -> str:
    """A valid label value: at most 63 of ``[A-Za-z0-9_.-]``, alphanumeric at both ends."""
    cleaned = _LABEL_VALUE.sub("-", value)[:63]
    return cleaned.strip("-_.") or "x"


#: The longest name of a StatefulSet (and of its Service): its pods are ``<name>-<ordinal>`` and carry the label
#: ``controller-revision-hash=<name>-<10-character hash>``, and both must fit Kubernetes' 63 characters.
STATEFUL_SET_NAME_MAX = 52


def k8s_name(name: str, max_length: int = 63) -> str:
    """``name`` if it has at most ``max_length`` characters (Kubernetes' 63 by default), else a prefix plus a stable
    8-hex hash of the whole name."""
    if len(name) <= max_length:
        return name
    digest = hashlib.sha256(name.encode()).hexdigest()[:8]
    return f"{name[: max_length - 9].rstrip('-')}-{digest}"


class KubernetesOptions(JobOptions):
    """The ``kubernetes`` runner's options.

    Attributes:
        image: The coordinator's image: a stock image with uv and Python 3.12 (default
            :data:`~rcp_ndcg.runners.script.COORDINATOR_IMAGE`); a job's own image wins.
        namespace: Where the Job goes.
        context: The kubectl context; default the current one.
        service_account: The pods' service account.
        secrets: Secret names exposed to every container as environment (``envFrom``), e.g. an HF token or the
            mirror's credentials.
        node_selector: The coordinator pod's node selector.
        engine_node_selector: The engine pods' node selector (a StatefulSet of several replicas).
        backoff_limit: Pod retries before the Job fails; a retried pod resumes the run from its mirror.
        ttl_seconds_after_finished: When a finished Job (and the engines it owns) is deleted.
    """

    image: str | None = None
    namespace: str = "default"
    context: str | None = None
    service_account: str | None = None
    secrets: list[str] = Field(default_factory=list)
    node_selector: dict[str, str] = Field(default_factory=dict)
    engine_node_selector: dict[str, str] = Field(default_factory=dict)
    backoff_limit: int = Field(default=0, ge=0)
    ttl_seconds_after_finished: int | None = Field(default=None, ge=0)


def _resources(res: Resources) -> dict[str, Any]:
    limits: dict[str, Any] = {}
    requests: dict[str, Any] = {}
    if res.gpus:
        limits["nvidia.com/gpu"] = res.gpus
    if res.cpus:
        requests["cpu"] = limits["cpu"] = res.cpus
    if res.memory_gb:
        requests["memory"] = limits["memory"] = f"{res.memory_gb:g}Gi"
    return {key: value for key, value in (("requests", requests), ("limits", limits)) if value}


def _engine_image(serve: ServeConfig) -> str:
    """The engine's image.

    Raises:
        ConfigError: ``serve`` names none; a Kubernetes engine runs in one.
    """
    if serve.image is None:
        raise ConfigError(
            "serve: names no image, and a Kubernetes engine runs in one",
            hint="set serve.image to the engine's image (pin the tag)",
        )
    return serve.image


def _together(job: Resources, engine: Resources) -> Resources:
    """What one container running both the coordinator and the engine asks for.

    The GPUs are the larger request; CPUs and memory add up, and one the engine leaves unstated stays unlimited
    (a coordinator's share must not cap the engine).
    """
    return Resources(
        gpus=max(job.gpus, engine.gpus),
        cpus=engine.cpus + (job.cpus or 0) if engine.cpus else None,
        memory_gb=engine.memory_gb + (job.memory_gb or 0) if engine.memory_gb else None,
    )


class KubernetesRunner:
    """Apply Jobs with ``kubectl``; query, read logs and delete them the same way.

    Options (``get_runner("kubernetes", **options)``): :class:`KubernetesOptions`; its ``resources`` and ``env``
    are every job's defaults.
    """

    name = "kubernetes"
    #: Where a run is restored inside the pod (the scratch volume), from the run's mirror.
    run_root = f"{SCRATCH}/runs"

    def __init__(self, **options: Any) -> None:
        self.options = KubernetesOptions.parse(self.name, options)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _env_from(self) -> list[dict[str, Any]]:
        return [{"secretRef": {"name": secret}} for secret in self.options.secrets]

    def _engine(self, serve: ServeConfig) -> dict[str, Any]:
        """A StatefulSet's engine container: the user's image and command, verbatim, probed on the readiness path."""
        probe = {"httpGet": {"path": serve.readiness_path, "port": serve.port}, "periodSeconds": PROBE_PERIOD_S}
        startup = max(1, -(-serve.startup_timeout_s // PROBE_PERIOD_S))  # the kubelet restarts it after that
        container: dict[str, Any] = {
            "name": "engine",
            "image": _engine_image(serve),
            "command": list(serve.command),
            "ports": [{"containerPort": serve.port}],
            "startupProbe": {**probe, "failureThreshold": startup},
            "readinessProbe": probe,
            "volumeMounts": [{"name": "dshm", "mountPath": "/dev/shm"}],
        }
        if serve.env:
            container["env"] = [{"name": key, "value": value} for key, value in serve.env.items()]
        if self.options.secrets:
            container["envFrom"] = self._env_from()
        resources = _resources(serve.resources)
        if resources:
            container["resources"] = resources
        return container

    def _engine_hosts(self, job: JobSpec) -> list[str]:
        """The replicas' host names: localhost for one replica, else the StatefulSet pods' stable names."""
        assert job.serve is not None
        if job.serve.replicas == 1:
            return ["127.0.0.1"]
        engines = self._engines_name(job)
        return [f"{engines}-{i}.{engines}.{self.options.namespace}.svc" for i in range(job.serve.replicas)]

    def _engines_name(self, job: JobSpec) -> str:
        """The engine StatefulSet's and Service's name: capped so the controller's pod names and labels fit."""
        return k8s_name(f"{job.name}-engine", STATEFUL_SET_NAME_MAX)

    def manifest(self, job: JobSpec) -> dict[str, Any]:
        """The Job object for ``job``, as a mapping (the runner's ``resources`` and ``env`` under the job's own)."""
        job = self.options.defaults_for(job)
        name = k8s_name(job.name)
        labels = {"app.kubernetes.io/name": "rcp-ndcg", "rcp-ndcg/job": _label_value(job.name)}
        env = {"UV_CACHE_DIR": f"{SCRATCH}/uv-cache", "UV_LINK_MODE": "copy"}
        serve = job.serve
        mounts = [{"name": "scratch", "mountPath": SCRATCH}]
        volumes: list[dict[str, Any]] = [{"name": "scratch", "emptyDir": {}}]
        if serve is None:
            container = self._coordinator(job, env, mounts)
        else:
            _engine_image(serve)
            hosts = self._engine_hosts(job)
            env[JUDGE_URLS_ENV] = ",".join(serve.url(host) for host in hosts)
            if serve.replicas == 1:
                container = self._engine_and_coordinator(job, env, mounts)
                volumes.append({"name": "dshm", "emptyDir": {"medium": "Memory"}})
            else:
                container = self._coordinator(job, env, mounts, wait_for_replicas(serve, " ".join(hosts)))
        pod: dict[str, Any] = {"restartPolicy": "Never", "containers": [container], "volumes": volumes}
        if self.options.service_account:
            pod["serviceAccountName"] = self.options.service_account
        if self.options.node_selector:
            pod["nodeSelector"] = dict(self.options.node_selector)
        spec: dict[str, Any] = {"backoffLimit": self.options.backoff_limit}
        if job.resources.time_limit_s:
            spec["activeDeadlineSeconds"] = job.resources.time_limit_s
        if self.options.ttl_seconds_after_finished is not None:
            spec["ttlSecondsAfterFinished"] = self.options.ttl_seconds_after_finished
        spec["template"] = {"metadata": {"labels": dict(labels)}, "spec": pod}
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": name, "namespace": self.options.namespace, "labels": labels},
            "spec": spec,
        }

    def _coordinator(
        self, job: JobSpec, env: dict[str, str], mounts: list[dict[str, str]], prologue: list[str] | None = None
    ) -> dict[str, Any]:
        """The coordinator's container: the stock image (or the job's), installing the release when it starts."""
        script = worker_script(job, install=True, workdir=None, env=env, prologue=prologue or ())
        return self._container(
            "coordinator", job.image or self.options.image or COORDINATOR_IMAGE, script, mounts, job.resources
        )

    def _engine_and_coordinator(
        self, job: JobSpec, env: dict[str, str], mounts: list[dict[str, str]]
    ) -> dict[str, Any]:
        """One replica's container: the engine's image, running the supervision script with the coordinator."""
        assert job.serve is not None
        serve = job.serve
        script = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            *heredoc("WORKER", worker_script(job, install=True, workdir=None, env=env)),
            *heredoc("ENGINE", engine_script(serve)),
            *supervise(serve, engine='bash -c "$ENGINE"', coordinator='bash -c "$WORKER"', hosts="127.0.0.1", uv=True),
        ]
        mounts = [*mounts, {"name": "dshm", "mountPath": "/dev/shm"}]
        return self._container(
            "run", _engine_image(serve), "\n".join(script) + "\n", mounts, _together(job.resources, serve.resources)
        )

    def _container(
        self, name: str, image: str, script: str, mounts: list[dict[str, str]], resources: Resources
    ) -> dict[str, Any]:
        container: dict[str, Any] = {
            "name": name,
            "image": image,
            "command": ["bash", "-c", script],
            "volumeMounts": mounts,
        }
        if self.options.secrets:
            container["envFrom"] = self._env_from()
        requests_limits = _resources(resources)
        if requests_limits:
            container["resources"] = requests_limits
        return container

    def engine_objects(self, job: JobSpec, job_uid: str = JOB_UID) -> list[dict[str, Any]]:
        """The StatefulSet and headless Service of a job's engine replicas, owned by its Job (none for 0 or 1)."""
        if job.serve is None or job.serve.replicas == 1:
            return []
        serve = job.serve
        engines = self._engines_name(job)
        labels = {"app.kubernetes.io/name": "rcp-ndcg-engine", "rcp-ndcg/job": _label_value(job.name)}
        owner = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "name": k8s_name(job.name),
            "uid": job_uid,
            "blockOwnerDeletion": True,
        }
        metadata = {"name": engines, "namespace": self.options.namespace, "labels": labels, "ownerReferences": [owner]}
        pod: dict[str, Any] = {
            "containers": [self._engine(serve)],
            "volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
        }
        if self.options.service_account:
            pod["serviceAccountName"] = self.options.service_account
        if self.options.engine_node_selector:
            pod["nodeSelector"] = dict(self.options.engine_node_selector)
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": metadata,
            "spec": {
                "clusterIP": "None",
                "publishNotReadyAddresses": True,
                "selector": dict(labels),
                "ports": [{"name": "http", "port": serve.port, "targetPort": serve.port}],
            },
        }
        stateful_set = {
            "apiVersion": "apps/v1",
            "kind": "StatefulSet",
            "metadata": metadata,
            "spec": {
                "serviceName": engines,
                "replicas": serve.replicas,
                "podManagementPolicy": "Parallel",
                "selector": {"matchLabels": dict(labels)},
                "template": {"metadata": {"labels": dict(labels)}, "spec": pod},
            },
        }
        return [stateful_set, service]

    def render(self, jobs: Sequence[JobSpec]) -> dict[str, str]:
        """The YAML each job is applied as: its Job, then any engine objects it owns (one multi-document stream)."""
        return {
            job.name: yaml.safe_dump_all([self.manifest(job), *self.engine_objects(job)], sort_keys=False, width=1000)
            for job in jobs
        }

    # ------------------------------------------------------------------
    # Scheduler calls
    # ------------------------------------------------------------------

    def _kubectl(self, *args: str, input_text: str | None = None) -> str:
        context = ["--context", self.options.context] if self.options.context else []
        return run_cli(["kubectl", *context, *args], input_text=input_text)

    def _split(self, handle: JobHandle) -> tuple[str, str]:
        namespace, _, name = handle.rpartition("/")
        return namespace or self.options.namespace, name

    def submit(self, jobs: Sequence[JobSpec]) -> list[JobHandle]:
        """``kubectl apply`` each job in order, then the engine objects it owns (with the Job's uid).

        Raises:
            RunnerError: ``kubectl`` fails.
        """
        handles: list[JobHandle] = []
        for job in jobs:
            applied = self._kubectl(
                "apply", "-o", "json", "-f", "-", input_text=yaml.safe_dump(self.manifest(job), sort_keys=False)
            )
            engines = self.engine_objects(job, job_uid=json.loads(applied)["metadata"]["uid"]) if job.serve else []
            if engines:
                self._kubectl("apply", "-f", "-", input_text=yaml.safe_dump_all(engines, sort_keys=False))
            handles.append(f"{self.options.namespace}/{k8s_name(job.name)}")
        return handles

    def status(self, handle: JobHandle) -> JobStatus:
        """From the Job's ``status``: a ``Complete``/``Failed`` condition, else active pods."""
        namespace, name = self._split(handle)
        try:
            job = json.loads(self._kubectl("get", "job", name, "-n", namespace, "-o", "json"))
        except RunnerError as exc:
            if "NotFound" in str(exc) or "not found" in str(exc):
                return JobStatus.UNKNOWN
            raise
        status = job.get("status") or {}
        conditions = {c.get("type"): c.get("status") for c in status.get("conditions") or []}
        if conditions.get("Failed") == "True":
            return JobStatus.FAILED
        if conditions.get("Complete") == "True":
            return JobStatus.SUCCEEDED
        if status.get("active"):
            return JobStatus.RUNNING
        return JobStatus.PENDING

    def logs(self, handle: JobHandle, *, tail: int | None = None) -> str:
        """Logs of every container of the Job's pods, each line prefixed with its pod and container."""
        namespace, name = self._split(handle)
        text = self._kubectl(
            "logs", "-n", namespace, "-l", f"job-name={name}", "--all-containers", "--prefix", "--tail=-1"
        )
        return tail_lines(text, tail)

    def cancel(self, handle: JobHandle) -> None:
        """Delete the Job; Kubernetes garbage-collects the engine objects it owns."""
        namespace, name = self._split(handle)
        self._kubectl("delete", "job", name, "-n", namespace, "--ignore-not-found", "--wait=false")


__all__ = [
    "JOB_UID",
    "PROBE_PERIOD_S",
    "SCRATCH",
    "STATEFUL_SET_NAME_MAX",
    "KubernetesOptions",
    "KubernetesRunner",
    "k8s_name",
]
