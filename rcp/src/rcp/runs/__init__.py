"""Runs: a config, a directory with a manifest, and the pipeline that fills it.

* :class:`RunConfig` -- the typed, ``extra="forbid"`` config of one run
  (:mod:`rcp.runs.config`), composed with ``extends:``;
* :class:`RunLayout` -- the fixed file names of a run directory, layout
  ``rcp-ndcg.run-layout.v1`` (:mod:`rcp.runs.layout`);
* :class:`RunManifest` -- per step: identity, inputs and outputs with hashes,
  usage and timing (:mod:`rcp.runs.manifest`);
* :class:`Pipeline` -- runs the steps, resuming what is current
  (:mod:`rcp.runs.pipeline`);
* :class:`Mirror` -- copies a run directory to any fsspec URI while it runs, and
  :func:`restore` rebuilds it from there (:mod:`rcp.runs.mirror`).
"""

from rcp.runs.config import STEPS, RunConfig, StepName
from rcp.runs.layout import LAYOUT_VERSION, RunLayout, discover_runs, new_run_id
from rcp.runs.manifest import MANIFEST_SCHEMA, DatasetRef, RunManifest, RunStatus, StepRecord, StepStatus
from rcp.runs.mirror import Mirror, MirrorState, mirrored, restore
from rcp.runs.pipeline import Pipeline

__all__ = [
    "LAYOUT_VERSION",
    "MANIFEST_SCHEMA",
    "STEPS",
    "DatasetRef",
    "Mirror",
    "MirrorState",
    "Pipeline",
    "RunConfig",
    "RunLayout",
    "RunManifest",
    "RunStatus",
    "StepName",
    "StepRecord",
    "StepStatus",
    "discover_runs",
    "mirrored",
    "new_run_id",
    "restore",
]
