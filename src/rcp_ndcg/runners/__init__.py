"""Job runners: where jobs run.

A :class:`JobSpec` names a command (argv) with its image, resources and
environment, and optionally the engine replicas to start beside it; a
:class:`JobRunner` submits, watches, reads and cancels it. Three runners ship,
each with one options model:

* ``local`` (:class:`LocalRunner`, :class:`LocalOptions`) -- run each job on the calling host;
* ``slurm`` (:class:`SlurmRunner`, :class:`SlurmOptions`) -- one ``sbatch`` script per job,
  optionally in an Apptainer or enroot/pyxis container;
* ``kubernetes`` (:class:`KubernetesRunner`, :class:`KubernetesOptions`) -- one ``batch/v1``
  Job per job, applied with ``kubectl``.

A runner is selected by name (:func:`get_runner`). Any other scheduler plugs in as
a separate package through the ``rcp_ndcg.runners`` entry-point group, the one
lookup the built-ins use too (see :mod:`rcp_ndcg.runners.registry`).
"""

from rcp_ndcg.runners.base import (
    JobHandle,
    JobOptions,
    JobPhase,
    JobRunner,
    JobSpec,
    JobStatus,
    Resources,
    RunnerError,
)
from rcp_ndcg.runners.kubernetes import KubernetesOptions, KubernetesRunner
from rcp_ndcg.runners.local import LocalOptions, LocalRunner
from rcp_ndcg.runners.registry import ENTRY_POINT_GROUP, get_runner
from rcp_ndcg.runners.script import COORDINATOR_IMAGE, install_argv, worker_script
from rcp_ndcg.runners.slurm import SlurmOptions, SlurmRunner
from rcp_ndcg.support.serve import ServeConfig

__all__ = [
    "COORDINATOR_IMAGE",
    "ENTRY_POINT_GROUP",
    "JobHandle",
    "JobOptions",
    "JobPhase",
    "JobRunner",
    "JobSpec",
    "JobStatus",
    "KubernetesOptions",
    "KubernetesRunner",
    "LocalOptions",
    "LocalRunner",
    "Resources",
    "RunnerError",
    "ServeConfig",
    "SlurmOptions",
    "SlurmRunner",
    "get_runner",
    "install_argv",
    "worker_script",
]
