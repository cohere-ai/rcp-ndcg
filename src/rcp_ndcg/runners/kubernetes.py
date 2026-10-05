"""KubernetesRunner -- one ``batch/v1`` Job per :class:`JobSpec`, through ``kubectl``.

The Job's coordinator container runs a stock image with uv and Python 3.12
(:data:`~rcp_ndcg.runners.script.COORDINATOR_IMAGE`), which installs this release
when the pod starts (:func:`~rcp_ndcg.runners.script.install_argv`). Nothing is
built or maintained by this package. The pod's disk is scratch: an ``emptyDir``
at ``/scratch`` holds uv's cache and the run directory (:attr:`KubernetesRunner.run_root`),
which the job restores from the run's mirror and mirrors back while it runs.

A job with phases runs them in order in one pod: every phase but the last is an
init container, the last is the main container, and ``restartPolicy: Never``
fails the pod on the first phase that exits non-zero. Each container runs its
phase's supervision script (:func:`~rcp_ndcg.runners.script.supervise`), the
same one SLURM runs:

* **one replica per engine** -- the phase's container runs in the engine's image
  (all of a phase's engines share one image: they are processes of one
  container): it installs uv if the image lacks it, starts the engines in the
  background, waits until each role answers on localhost, exports their URLs in
  ``RCP_NDCG_ENGINES``, runs the phase's coordinator, and ends the phase when
  either ends. An engine that dies or never answers fails the phase, and the pod.
  Any launcher that takes an image and a command runs it; the engine's image
  needs ``bash`` 4.3 or later and ``python3`` (with ``pip`` when it has no uv):
  without bash the container does not start, and a pod whose image lacks one of
  the others stops at once, naming it;
* **several replicas of an engine** (RFC Q8) -- a StatefulSet of engine pods
  behind a headless Service, both owned by the Job (``ownerReferences``), created
  at submit and **run-scoped**: such engines live for the whole run, not one
  phase, and deleting the Job deletes them. Their URLs are the pods' stable
  names, known when the objects are rendered; a phase that uses them waits for a
  replica at most ``startup_timeout_s``. A phased pod on one node with the
  scheduler reserving the largest init container's GPUs matches the maximum over
  the phases; multi-replica engines add their own pods outside it.

A failed Job is retried ``backoff_limit`` times (0 by default); a retried pod resumes
the run from its mirror.

Everything goes through the ``kubectl`` on ``PATH`` and its current (or the
configured) context; nothing here needs cluster credentials of its own.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
from collections.abc import Sequence
from typing import Any

import yaml
from pydantic import Field

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.runners._cli import run_cli
from rcp_ndcg.runners.base import JobHandle, JobOptions, JobPhase, JobSpec, JobStatus, RunnerError, tail_lines
from rcp_ndcg.runners.script import (
    COORDINATOR_IMAGE,
    REMOTE_ENGINE_PID,
    EngineStep,
    engine_script,
    engines_env_value,
    heredoc,
    supervise,
    wait_for_replicas,
    worker_script,
)
from rcp_ndcg.support.resources import Resources
from rcp_ndcg.support.serve import ENGINES_ENV, EngineRole, ServeConfig

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


def _together(job: Resources, engines: Sequence[Resources]) -> Resources:
    """What one container running the coordinator and several engines asks for.

    The GPUs are the largest request; CPUs and memory add up, and one an engine leaves unstated stays unlimited
    (a coordinator's share must not cap the engine).
    """
    if not engines:
        return job
    cpus = [engine.cpus for engine in engines]
    memory = [engine.memory_gb for engine in engines]
    return Resources(
        gpus=max(job.gpus, max(engine.gpus for engine in engines)),
        cpus=(job.cpus or 0) + sum(value for value in cpus if value is not None)
        if all(value is not None for value in cpus)
        else None,
        memory_gb=(job.memory_gb or 0) + sum(value for value in memory if value is not None)
        if all(value is not None for value in memory)
        else None,
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

    def _engines_name(self, job: JobSpec, role: str) -> str:
        """The engine StatefulSet's and Service's name: capped so the controller's pod names and labels fit."""
        return k8s_name(f"{job.name}-engine-{role}", STATEFUL_SET_NAME_MAX)

    def _engine_hosts(self, serve: ServeConfig, role: str, job: JobSpec) -> list[str]:
        """The replicas' host names: localhost for one replica, else the StatefulSet pods' stable names."""
        if serve.replicas == 1:
            return ["127.0.0.1"]
        engines = self._engines_name(job, role)
        return [f"{engines}-{i}.{engines}.{self.options.namespace}.svc" for i in range(serve.replicas)]

    def _stateful_engines(self, job: JobSpec) -> dict[EngineRole, ServeConfig]:
        """The engines a job runs as StatefulSets: every role with several replicas, once, by role.

        Raises:
            ConfigError: a role appears in several phases with different configurations (one StatefulSet serves
                every phase that uses it, so it cannot differ); or names no image.
        """
        engines: dict[EngineRole, ServeConfig] = {}
        for phase in job.phases:
            for role, serve in sorted(phase.engines.items()):
                if serve.replicas == 1:
                    continue
                _engine_image(serve)
                if role in engines and engines[role] != serve:
                    raise ConfigError(
                        f"the {role} engine appears in several phases of job {job.name!r} with different "
                        "configurations, and one StatefulSet serves them all (several replicas are run-scoped: "
                        "the engine lives for the whole run)",
                        hint="give the engine the same configuration in every phase, or give each phase its own "
                        "role (its own served model) so each gets its own StatefulSet",
                    )
                engines[role] = serve
        return engines

    def manifest(self, job: JobSpec) -> dict[str, Any]:
        """The Job object for ``job``, as a mapping (the runner's ``resources`` and ``env`` under the job's own)."""
        job = self.options.defaults_for(job)
        name = k8s_name(job.name)
        labels = {"app.kubernetes.io/name": "rcp-ndcg", "rcp-ndcg/job": _label_value(job.name)}
        env = {"UV_CACHE_DIR": f"{SCRATCH}/uv-cache", "UV_LINK_MODE": "copy"}
        mounts = [{"name": "scratch", "mountPath": SCRATCH}]
        volumes: list[dict[str, Any]] = [{"name": "scratch", "emptyDir": {}}]
        self._stateful_engines(job)  # refused here, before anything is rendered
        if job.phases:
            containers = [
                self._phase_container(job, index, phase, env, mounts, volumes)
                for index, phase in enumerate(job.phases, 1)
            ]
            pod: dict[str, Any] = {"restartPolicy": "Never"}
            if len(containers) > 1:  # every phase but the last is an init container
                pod["initContainers"] = containers[:-1]
            pod["containers"] = [containers[-1]]
        else:
            pod = {"restartPolicy": "Never", "containers": [self._coordinator(job, env, mounts)]}
        pod["volumes"] = volumes
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

    def _phase_container(
        self,
        job: JobSpec,
        index: int,
        phase: JobPhase,
        env: dict[str, str],
        mounts: list[dict[str, str]],
        volumes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """One phase's container: an init container, unless it is the job's last phase (the main container).

        A phase with engines of one replica each runs them and its coordinator in one container of the (first)
        engine's image, under the supervision script. A phase whose engines are all StatefulSet replicas waits
        for them in the coordinator's image. A phase without engines runs its command directly, with an empty
        ``RCP_NDCG_ENGINES``, in the coordinator's image.
        """
        worker_var = f"WORKER_{index}"
        name = f"phase-{index}"
        local: dict[EngineRole, ServeConfig] = {
            role: serve for role, serve in sorted(phase.engines.items()) if serve.replicas == 1
        }
        remote: dict[EngineRole, ServeConfig] = {
            role: serve for role, serve in sorted(phase.engines.items()) if serve.replicas > 1
        }
        steps: list[EngineStep] = []
        urls: dict[EngineRole, list[str]] = {}
        for role, serve in remote.items():
            hosts = self._engine_hosts(serve, role, job)
            steps.append(EngineStep(serve=serve, start=None, hosts=" ".join(hosts)))
            urls[role] = [serve.url(host) for host in hosts]
        if local:
            images = {serve.image for serve in local.values()}
            if len(images) > 1:
                raise ConfigError(
                    f"phase {index} of job {job.name!r} starts {sorted(local)} in one container, and their images "
                    f"differ ({', '.join(sorted(str(image) for image in images))})",
                    hint="give the phase's engines one image (one container starts and stops them together), or "
                    "split the steps into phases that each start one engine",
                )
            ports = [serve.port for serve in local.values()]
            if len(set(ports)) != len(ports):
                raise ConfigError(
                    f"phase {index} of job {job.name!r} starts several engines on one port ({ports}) in one "
                    "container, where only one of them can listen",
                    hint="give the phase's engines distinct ports",
                )
            image = _engine_image(next(iter(local.values())))
            script = ["#!/usr/bin/env bash", "set -euo pipefail"]
            for role, serve in local.items():
                steps.append(EngineStep(serve=serve, start=f'bash -c "$ENGINE_{role.upper()}"', hosts="127.0.0.1"))
                urls[role] = [serve.url("127.0.0.1")]
            script += [
                *heredoc(
                    worker_var,
                    worker_script(job.model_copy(update={"argv": phase.argv}), install=True, workdir=None, env=env),
                ),
                *(
                    line
                    for role, serve in local.items()
                    for line in heredoc(f"ENGINE_{role.upper()}", engine_script(serve))
                ),
                *supervise(
                    steps,
                    coordinator=f'bash -c "${worker_var}"',
                    engines_env=shlex.quote(engines_env_value(phase.engines, urls)),
                    uv=True,
                ),
            ]
            mounts = [*mounts, {"name": "dshm", "mountPath": "/dev/shm"}]
            if "dshm" not in {volume["name"] for volume in volumes}:
                volumes.append({"name": "dshm", "emptyDir": {"medium": "Memory"}})
            resources = _together(job.resources, [serve.resources for serve in local.values()])
            return self._container(name, image, "\n".join(script) + "\n", mounts, resources)
        if remote:
            # No engine of this phase runs in this container: it is the coordinator, waiting for the replicas
            # (wait_for_replicas) and carrying their URLs in RCP_NDCG_ENGINES, exec'd so SIGTERM reaches it.
            script = worker_script(
                job.model_copy(update={"argv": phase.argv}),
                install=True,
                workdir=None,
                env={**env, ENGINES_ENV: engines_env_value(phase.engines, urls)},
                prologue=[
                    line
                    for role, serve in remote.items()
                    for line in wait_for_replicas(
                        serve, " ".join(self._engine_hosts(serve, role, job)), REMOTE_ENGINE_PID
                    )
                ],
            )
            return self._container(
                name, job.image or self.options.image or COORDINATOR_IMAGE, script, mounts, job.resources
            )
        # A phase without engines runs its command directly, with an empty RCP_NDCG_ENGINES, so no engine of an
        # earlier phase reaches it.
        script = worker_script(
            job.model_copy(update={"argv": phase.argv}), install=True, workdir=None, env={**env, ENGINES_ENV: "{}"}
        )
        return self._container(
            name, job.image or self.options.image or COORDINATOR_IMAGE, script, mounts, job.resources
        )

    def _coordinator(
        self, job: JobSpec, env: dict[str, str], mounts: list[dict[str, str]], prologue: list[str] | None = None
    ) -> dict[str, Any]:
        """The coordinator's container: the stock image (or the job's), installing the release when it starts."""
        script = worker_script(job, install=True, workdir=None, env=env, prologue=prologue or ())
        return self._container(
            "coordinator", job.image or self.options.image or COORDINATOR_IMAGE, script, mounts, job.resources
        )

    def engine_objects(self, job: JobSpec, job_uid: str = JOB_UID) -> list[dict[str, Any]]:
        """The StatefulSet and headless Service of each of a job's several-replica engines, owned by its Job.

        They are created at submit and **run-scoped** (owned by the Job): such engines live for the whole run,
        not one phase.

        Returns:
            One StatefulSet and Service per role with several replicas (none for 0 of them).
        """
        engines = self._stateful_engines(job)
        if not engines:
            return []
        objects: list[dict[str, Any]] = []
        for role in sorted(engines):
            serve = engines[role]
            engines_name = self._engines_name(job, role)
            labels = {
                "app.kubernetes.io/name": f"rcp-ndcg-engine-{role}",
                "rcp-ndcg/job": _label_value(job.name),
            }
            owner = {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "name": k8s_name(job.name),
                "uid": job_uid,
                "blockOwnerDeletion": True,
            }
            metadata = {
                "name": engines_name,
                "namespace": self.options.namespace,
                "labels": labels,
                "ownerReferences": [owner],
            }
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
                    "serviceName": engines_name,
                    "replicas": serve.replicas,
                    "podManagementPolicy": "Parallel",
                    "selector": {"matchLabels": dict(labels)},
                    "template": {"metadata": {"labels": dict(labels)}, "spec": pod},
                },
            }
            objects += [stateful_set, service]
        return objects

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
            engines = self.engine_objects(job, job_uid=json.loads(applied)["metadata"]["uid"])
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
