"""The on-disk contract of a run: layout ``rcp-ndcg.run-layout.v1``.

Everything a run produces lands under one directory with fixed names::

    runs/<run_id>/
      run.yaml             the resolved config (after extends and --set)
      manifest.json        per step: state, identity, inputs and outputs with hashes, usage, timing, versions
      candidates.parquet   each query's candidate pool as rankings: query_id, doc_id, score (system "candidates")
      judgements/          the append-only judgement store: tournament.jsonl, rubric.jsonl, identity.json,
                           preprocessing.jsonl (every per-window text cut)
      calibration/         items.json, queries.parquet, thetas.parquet, coverage.json, diagnostics.json,
                           extensions.jsonl, identity.json
      metrics/             report.json (the evaluation report), comparison.json (the systems compared)
      logs/                run.log (the pipeline's log), jobs.json (the jobs a runner was handed),
                           mirror.json (the mirror's last upload)

Fixed names are the point: a resumed run, the CLI and an agent locate
artifacts the same way. A run directory is always a local (or shared POSIX)
path, so a preempted job never loses what an object store did not receive yet;
:mod:`rcp_ndcg.runs.mirror` copies it to an object store while it runs, and
restores it from there.
"""

from __future__ import annotations

import os
import re
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rcp_ndcg.storage import is_remote, local_dir

#: The layout's version, recorded in every manifest.
LAYOUT_VERSION = "rcp-ndcg.run-layout.v1"

MANIFEST_NAME = "manifest.json"
#: The run's scratch directory (indices, caches the steps can recompute); it is not mirrored.
WORK_DIR = "work"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(value: str, *, max_length: int = 32) -> str:
    """A filesystem-safe fragment of *value* for a run id."""
    return _SLUG_RE.sub("-", value.lower()).strip("-")[:max_length].strip("-")


def new_run_id(label: str | None = None) -> str:
    """A sortable, unique run id, optionally carrying a label: ``20260724-151233-nano-gpt5-a1b2c3``."""
    parts = [datetime.now(UTC).strftime("%Y%m%d-%H%M%S")]
    if label and (slug := slugify(label)):
        parts.append(slug)
    parts.append(secrets.token_hex(3))
    return "-".join(parts)


@dataclass(frozen=True)
class RunLayout:
    """The paths of one run directory (:meth:`at` an existing or planned root)."""

    root: str

    @classmethod
    def at(cls, root: str | Path) -> RunLayout:
        """The layout of the run directory ``root``.

        Raises:
            ConfigError: ``root`` is a remote URI; a run directory is local, and a mirror copies it to a bucket.
        """
        return cls(root=str(local_dir(root, "a run directory")).rstrip("/"))

    @property
    def run_id(self) -> str:
        return Path(self.root).name

    def path(self, *parts: str) -> str:
        """A path inside this run."""
        return str(Path(self.root).joinpath(*parts))

    @property
    def manifest(self) -> str:
        return self.path(MANIFEST_NAME)

    @property
    def config(self) -> str:
        return self.path("run.yaml")

    @property
    def candidates(self) -> str:
        return self.path("candidates.parquet")

    @property
    def judgements(self) -> str:
        """The judgement store directory."""
        return self.path("judgements")

    @property
    def calibration(self) -> str:
        """The calibration directory."""
        return self.path("calibration")

    @property
    def metrics_dir(self) -> str:
        return self.path("metrics")

    @property
    def metrics(self) -> str:
        return self.path("metrics", "report.json")

    @property
    def comparison(self) -> str:
        return self.path("metrics", "comparison.json")

    @property
    def logs_dir(self) -> str:
        return self.path("logs")

    @property
    def log(self) -> str:
        """The pipeline's log of this run."""
        return self.path("logs", "run.log")

    @property
    def jobs(self) -> str:
        """The jobs a runner was handed for this run (runner name, options and handles)."""
        return self.path("logs", "jobs.json")

    @property
    def mirror_state(self) -> str:
        """The mirror's record of its last upload (:mod:`rcp_ndcg.runs.mirror`)."""
        return self.path("logs", "mirror.json")

    @property
    def work(self) -> str:
        """Scratch space of the steps (indices, intermediate files)."""
        return self.path(WORK_DIR)

    def ensure(self) -> RunLayout:
        """Create the directory skeleton, owner-only (the run directory holds the config, the records and the
        judgements, and a cluster filesystem is shared with every other user). An existing directory is
        tightened too: a run created before this rule must not stay world-traversable."""
        for directory in (self.root, self.judgements, self.calibration, self.metrics_dir, self.logs_dir):
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(path, 0o700)
        return self

    def relative(self, uri: str) -> str:
        """``uri`` relative to this run (manifests record run-relative paths), or unchanged if outside it."""
        prefix = f"{self.root}/"
        return uri[len(prefix) :] if uri.startswith(prefix) else uri

    def resolve(self, path: str) -> str:
        """The inverse of :meth:`relative`."""
        return path if is_remote(path) or path.startswith("/") else self.path(path)


def discover_runs(runs_dir: str | Path) -> list[str]:
    """Run ids under ``runs_dir`` (directories holding a manifest), newest first."""
    root = local_dir(runs_dir, "the runs directory")
    if not root.is_dir():
        return []
    return sorted((entry.name for entry in root.iterdir() if (entry / MANIFEST_NAME).is_file()), reverse=True)


__all__ = ["LAYOUT_VERSION", "MANIFEST_NAME", "WORK_DIR", "RunLayout", "discover_runs", "new_run_id", "slugify"]
