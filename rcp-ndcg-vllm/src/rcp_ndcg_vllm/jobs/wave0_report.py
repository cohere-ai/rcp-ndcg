"""The schema of wave 0's report (``rcp-ndcg.wave0-report.v1``), as pydantic models.

``wave0.sh`` assembles the report at run time with the stdlib-only ``report.py`` (the sections are the
probes' JSON fragments); this module is the one place the report's shape is declared, exported to
``schema/wave0-report.schema.json``, and validated against in the package's tests. A wave-0 report that
died early (fail fast) carries ``failed_step`` and ``error`` and may lack the later sections.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Wave0Embed",
    "Wave0Engines",
    "Wave0Evict",
    "Wave0Hub",
    "Wave0Gcs",
    "Wave0InputRow",
    "Wave0Report",
    "Wave0Stop",
    "wave0_report_schema",
]

_WAVE0_SCHEMA = "rcp-ndcg.wave0-report.v1"


def _no_extra() -> dict[str, Any]:
    """The common model config: frozen, unknown fields refused."""
    return {"extra": "forbid", "frozen": True}


class Wave0InputRow(BaseModel):
    """One of the embed check's inputs, with both sides' token counts (node-runtime item 6)."""

    model_config = ConfigDict(**_no_extra())

    id: str
    kind: Literal["short", "over_length"]
    fit_tokens: int = Field(ge=0)
    engine_tokens: int = Field(ge=0)
    ids_equal: bool
    cut: bool
    text_head: str


class Wave0Embed(BaseModel):
    """Step (e): the product's fit under an explicit budget, the product's client, the engine's /tokenize."""

    model_config = ConfigDict(**_no_extra())

    budget_tokens: int = Field(gt=0)
    n_inputs: int = Field(gt=0)
    n_over_length: int = Field(ge=0)
    cuts_recorded: int = Field(ge=0)
    overhead_tokens: int | None = None
    client_budget_wired: bool  # always true since the clients-final merge: the wired role client fits
    client: dict[str, Any]  # n_vectors, dim, finite
    tokenize_check: dict[str, Any]  # checked, passed, rows (the product's own /tokenize check shape)
    passed: bool


class Wave0Engines(BaseModel):
    """Step (d): two engines on two slots at the same time, no slot sharing anything."""

    model_config = ConfigDict(**_no_extra())

    slots: list[dict[str, Any]]
    concurrent: bool
    isolation: dict[str, dict[str, Any]]
    passed: bool


class Wave0Hub(BaseModel):
    """Step (c), the Hub half: the metadata call with the token secret, the sha against the pin."""

    model_config = ConfigDict(**_no_extra())

    model: str
    revision: str | None
    status: int | None
    hub_sha: str | None = None
    pinned_match: bool | None = None
    error: str | None = None
    passed: bool


class Wave0Gcs(BaseModel):
    """Step (c), the gs:// round-trip through the product's storage, from the client environment."""

    model_config = ConfigDict(**_no_extra())

    uri: str
    wrote_bytes: int = Field(ge=0)
    listed: bool
    read_equal: bool
    deleted: bool
    passed: bool


class Wave0Stop(BaseModel):
    """Step (g): everything stopped, and no engine process left on the node."""

    model_config = ConfigDict(**_no_extra())

    pids: list[int]
    leaked: list[str]
    scan_found: list[str]
    scan_all: list[str]
    all_stopped: bool
    free_disk_after_stop_bytes: int | None = None
    passed: bool


class Wave0Evict(BaseModel):
    """Step (f): the models out of the HF cache, the free disk before and after (item 8)."""

    model_config = ConfigDict(**_no_extra())

    models: list[dict[str, Any]]
    note: str | None = None
    passed: bool


class Wave0Report(BaseModel):
    """The whole wave-0 report: the envelope, then one section per step (absent until it ran)."""

    model_config = ConfigDict(**_no_extra(), populate_by_name=True)

    schema_name: Literal["rcp-ndcg.wave0-report.v1"] = Field(alias="schema")
    started: str
    passed: bool
    failed_step: str | None = None
    error: str | None = None
    host: dict[str, Any] | None = None
    bootstrap: dict[str, Any] | None = None
    reach: dict[str, Wave0Hub | Wave0Gcs] | None = None
    engines: Wave0Engines | None = None
    embed: Wave0Embed | None = None
    evict: Wave0Evict | None = None
    stop: Wave0Stop | None = None
    uploads: list[dict[str, Any]] | None = None
    finished: dict[str, Any] | None = None


def wave0_report_schema() -> dict[str, Any]:
    """The JSON Schema of :class:`Wave0Report`, exported to ``schema/wave0-report.schema.json``."""
    return Wave0Report.model_json_schema(by_alias=True)
