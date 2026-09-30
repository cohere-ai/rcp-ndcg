"""The run manifest: what happened in a run, to what, at what cost.

One JSON file per run (schema ``rcp-ndcg.run-manifest.v1``), updated as each
step finishes. Per step it records the state, the step's identity (a digest of
everything that decides its output), the inputs and outputs with content
hashes, the judge's usage, the timing and, for a judging step, what the judge's
endpoints said they serve; for the run, the resolved config (the seed and the
preprocessing included), the dataset and its revisions, the judgement families
and the code version. A run is resumable exactly when every completed step's
identity and inputs are unchanged.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from rcp_ndcg_core.schemas import Family

from rcp_ndcg.errors import DataError
from rcp_ndcg.llm.client import EngineInfo, Usage
from rcp_ndcg.runs.layout import LAYOUT_VERSION, RunLayout
from rcp_ndcg.storage.artifacts import ArtifactRef, CodeVersion, code_version
from rcp_ndcg.support.identity import hash_payload

MANIFEST_SCHEMA = "rcp-ndcg.run-manifest.v1"


def _now() -> datetime:
    return datetime.now(UTC)


class StepStatus(StrEnum):
    """The state of one step."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    """Stopped by ``run cancel`` while it ran."""


class RunStatus(StrEnum):
    """The state of the run."""

    SUBMITTED = "submitted"
    """Handed to a job runner; the job has not started the run yet."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    """Ended with steps of its config not done (an invocation that ran only some of them): a resume runs the rest."""
    CANCELLED = "cancelled"


class StepRecord(BaseModel):
    """One step's execution: its identity, what it read and wrote, the judge calls it made, how long it took."""

    model_config = ConfigDict(extra="forbid")

    name: str
    status: StepStatus
    identity: dict[str, Any] = Field(default_factory=dict)
    identity_hash: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_s: float | None = None
    inputs: list[ArtifactRef] = Field(default_factory=list)
    outputs: list[ArtifactRef] = Field(default_factory=list)
    usage: Usage | None = None
    engines: list[EngineInfo] = Field(default_factory=list)
    """What the judge's endpoints said they serve (judging steps; runtime information, never in an identity)."""
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is StepStatus.COMPLETED


class DatasetRef(BaseModel):
    """What was evaluated: the dataset's name and the revision each source resolved to."""

    model_config = ConfigDict(extra="forbid")

    name: str
    revisions: dict[str, dict[str, Any]] | None = None


class RunManifest(BaseModel):
    """Everything known about one run."""

    model_config = ConfigDict(extra="forbid", validate_by_name=True, serialize_by_alias=True)

    schema_name: Literal["rcp-ndcg.run-manifest.v1"] = Field(default=MANIFEST_SCHEMA, alias="schema")
    layout: Literal["rcp-ndcg.run-layout.v1"] = LAYOUT_VERSION
    run_id: str
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    status: RunStatus = RunStatus.RUNNING
    code: CodeVersion
    config: dict[str, Any] = Field(default_factory=dict)
    dataset: DatasetRef | None = None
    families: dict[str, Family] = Field(default_factory=dict)
    """``{family_key: Family}`` of the judgements the run wrote."""
    steps: list[StepRecord] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    """``{"<system>/<metric>@<k>": value}`` of the evaluate step's summary, every system but the judge's own order
    (the full report, with confidence intervals: ``metrics/report.json``)."""
    usage: Usage = Field(default_factory=Usage)

    @classmethod
    def new(cls, run_id: str, *, config: dict[str, Any]) -> RunManifest:
        """A fresh manifest for ``run_id``."""
        return cls(run_id=run_id, code=code_version(), config=config)

    # -- steps ---------------------------------------------------------------

    def step(self, name: str) -> StepRecord | None:
        return next((record for record in self.steps if record.name == name), None)

    def start_step(self, name: str, *, identity: dict[str, Any]) -> StepRecord:
        """Mark ``name`` running (one record per step: a retry replaces it)."""
        record = self.step(name)
        if record is None:
            record = StepRecord(name=name, status=StepStatus.RUNNING)
            self.steps.append(record)
        record.status = StepStatus.RUNNING
        record.started_at = _now()
        record.ended_at = record.duration_s = record.error = None
        record.identity = identity
        record.identity_hash = hash_payload(identity)
        self.updated_at = _now()
        return record

    def finish_step(
        self,
        name: str,
        *,
        status: StepStatus = StepStatus.COMPLETED,
        inputs: list[ArtifactRef] | None = None,
        outputs: list[ArtifactRef] | None = None,
        usage: Usage | None = None,
        error: str | None = None,
    ) -> StepRecord:
        """Close ``name`` and add its usage to the run total."""
        record = self.step(name)
        if record is None:
            record = StepRecord(name=name, status=status)
            self.steps.append(record)
        record.status = status
        record.ended_at = _now()
        if record.started_at is not None:
            record.duration_s = (record.ended_at - record.started_at).total_seconds()
        if inputs is not None:
            record.inputs = inputs
        if outputs is not None:
            record.outputs = outputs
        if usage is not None:
            record.usage = usage
            self.usage = self.usage.merged_with(usage)
        record.error = error
        self.updated_at = _now()
        return record

    # -- disk ----------------------------------------------------------------

    def save(self, layout: RunLayout) -> str:
        """Write the manifest atomically (a temp file and a rename, locally)."""
        self.updated_at = _now()
        payload = self.model_dump_json(indent=2)
        target = layout.manifest
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
        return target

    @classmethod
    def load(cls, source: str | Path | RunLayout) -> RunManifest:
        """Read the manifest of a run directory (or the manifest file itself).

        Raises:
            DataError: the file is not a manifest of this schema.
        """
        path = source.manifest if isinstance(source, RunLayout) else str(source)
        if not path.endswith(".json"):
            path = RunLayout.at(path).manifest
        text = Path(path).read_text(encoding="utf-8")
        try:
            return cls.model_validate_json(text)
        except ValueError as exc:
            raise DataError(f"{path} is not a {MANIFEST_SCHEMA} manifest: {exc}") from exc


__all__ = ["MANIFEST_SCHEMA", "DatasetRef", "RunManifest", "RunStatus", "StepRecord", "StepStatus"]
