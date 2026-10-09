"""Which windows of each query a calibration could not read, and when that is worth a warning.

A window whose answer did not parse contributes nothing to the fit. A few spread
over a query are harmless; many in one query, or one among the tournament's
adaptive windows (the few windows that order the top of the ranking), leave that
query's abilities resting on less evidence than the schedule planned.
:func:`window_coverage` counts the invalid windows of every query by stage,
phase and category; :func:`coverage_flags` names the queries that cross
:data:`INVALID_WINDOW_SHARE` in a stage or have an invalid adaptive window.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from rcp_ndcg_core.schemas import Judgement, Stage

from rcp_ndcg.calibration._projection import namespace

#: A query is flagged when more than this share of its windows of one stage is invalid.
INVALID_WINDOW_SHARE = 0.05

#: The phase coverage reports for a planned window (asked outside the schedule's phases: ``judge(windows=...)``).
PLANNED_PHASE = "planned"

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class StageWindows(BaseModel):
    """One query's windows of one stage.

    Attributes:
        windows: Windows judged.
        invalid: Of which invalid.
        share: ``invalid / windows``.
        invalid_by_phase: Invalid windows per schedule phase (``planned`` for windows asked outside the phases).
        invalid_by_category: Invalid windows per failure category.
    """

    model_config = _FROZEN

    windows: int
    invalid: int
    share: float
    invalid_by_phase: dict[str, int] = Field(default_factory=dict)
    invalid_by_category: dict[str, int] = Field(default_factory=dict)


class CoverageFlag(BaseModel):
    """A query and stage whose invalid windows warrant a warning.

    Attributes:
        query: ``<dataset>||<query_id>``.
        stage: The stage.
        windows: Its windows of the stage.
        invalid: Of which invalid.
        share: ``invalid / windows``.
        invalid_adaptive: Invalid adaptive windows.
        reasons: ``"share"`` (more than :data:`INVALID_WINDOW_SHARE` invalid) and/or ``"adaptive"``.
    """

    model_config = _FROZEN

    query: str
    stage: Stage
    windows: int
    invalid: int
    share: float
    invalid_adaptive: int
    reasons: list[Literal["share", "adaptive"]]


class WindowCount(BaseModel):
    """Windows of one stage over every query, and how many were invalid."""

    model_config = _FROZEN

    windows: int = 0
    invalid: int = 0


class QueryCounts(BaseModel):
    """Queries with rubric judgements, with tournament judgements, and with calibrated abilities."""

    model_config = _FROZEN

    rubric: int
    tournament: int
    calibrated: int


class DegenerateCounts(BaseModel):
    """Documents that failed, or passed, every criterion in every placement."""

    model_config = _FROZEN

    all_fail: int
    all_pass: int


class CalibrationCoverage(BaseModel):
    """What a calibration covers, and which of its windows it could not read.

    Attributes:
        mode: The fit's mode.
        queries: Queries per stage, and those calibrated.
        uncalibrated_queries: Queries judged but without calibrated abilities.
        uncalibrated_documents: ``<query>/<doc_id>`` of the documents with rubric verdicts but no calibrated
            ability (tournament mode: documents the tournament did not judge, whose verdicts the fit cannot use).
        no_tournament_evidence_documents: ``<query>/<doc_id>`` of the documents whose tournament windows are all
            invalid: the Bradley-Terry fit gives them the query mean ability and the ridge's standard error, and
            they carry no tournament evidence (tournament mode).
        windows: Windows per stage over every query.
        invalid_windows: ``{query: {stage: StageWindows}}`` for the queries with an invalid window.
        invalid_window_share: The share above which a query is flagged.
        flagged_queries: The flagged queries and stages.
        documents: Documents with rubric verdicts.
        degenerate_documents: Documents that failed, or passed, every criterion.
        per_dataset: Calibrated queries per dataset.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_by_name=True, serialize_by_alias=True)

    schema_name: Literal["rcp-ndcg.calibration-coverage.v1"] = Field(
        default="rcp-ndcg.calibration-coverage.v1", alias="schema"
    )
    mode: Literal["tournament", "rubric_only"]
    queries: QueryCounts
    uncalibrated_queries: list[str]
    uncalibrated_documents: list[str] = Field(default_factory=list)
    no_tournament_evidence_documents: list[str] = Field(default_factory=list)
    windows: dict[str, WindowCount]
    invalid_windows: dict[str, dict[str, StageWindows]]
    invalid_window_share: float
    flagged_queries: list[CoverageFlag]
    documents: int
    degenerate_documents: DegenerateCounts
    per_dataset: dict[str, int]


def _query(judgement: Judgement) -> str:
    return namespace(judgement.dataset, judgement.query_id)


def window_coverage(judgements: Iterable[Judgement], stages: Iterable[Stage]) -> dict[str, dict[str, StageWindows]]:
    """Per query and stage: the windows, the invalid ones, their share, and the invalid ones by phase and category.

    Args:
        judgements: The judgement records.
        stages: The stages the fit reads; records of other stages are left out.

    Returns:
        ``{"<dataset>||<query_id>": {stage: StageWindows}}``, for every query with a window of those stages.
    """
    wanted = set(stages)
    counts: dict[str, dict[str, dict]] = {}
    for judgement in judgements:
        if judgement.stage not in wanted:
            continue
        entry = counts.setdefault(_query(judgement), {}).setdefault(
            judgement.stage, {"windows": 0, "invalid": 0, "invalid_by_phase": {}, "invalid_by_category": {}}
        )
        entry["windows"] += 1
        if judgement.valid:
            continue
        entry["invalid"] += 1
        phase = judgement.phase or PLANNED_PHASE
        category = judgement.invalid_category or "unknown"
        entry["invalid_by_phase"][phase] = entry["invalid_by_phase"].get(phase, 0) + 1
        entry["invalid_by_category"][category] = entry["invalid_by_category"].get(category, 0) + 1
    return {
        query: {
            stage: StageWindows(
                windows=entry["windows"],
                invalid=entry["invalid"],
                share=entry["invalid"] / entry["windows"],
                invalid_by_phase=dict(sorted(entry["invalid_by_phase"].items())),
                invalid_by_category=dict(sorted(entry["invalid_by_category"].items())),
            )
            for stage, entry in stages_of_query.items()
        }
        for query, stages_of_query in sorted(counts.items())
    }


def coverage_flags(coverage: Mapping[str, Mapping[str, StageWindows]]) -> list[CoverageFlag]:
    """The queries whose windows warrant a warning, one entry per query and stage.

    A query and stage is flagged when more than :data:`INVALID_WINDOW_SHARE` of its windows are invalid, or
    when any of its adaptive windows is.

    Args:
        coverage: :func:`window_coverage`'s result.

    Returns:
        The flags, sorted by query and stage.
    """
    flags = []
    for query, stages in sorted(coverage.items()):
        for stage, entry in sorted(stages.items()):
            adaptive = int(entry.invalid_by_phase.get("adaptive", 0))
            reasons = [
                reason
                for reason, hit in (("share", entry.share > INVALID_WINDOW_SHARE), ("adaptive", adaptive > 0))
                if hit
            ]
            if reasons:
                flags.append(
                    CoverageFlag(
                        query=query,
                        stage=stage,  # type: ignore[arg-type]
                        windows=entry.windows,
                        invalid=entry.invalid,
                        share=entry.share,
                        invalid_adaptive=adaptive,
                        reasons=reasons,  # type: ignore[arg-type]
                    )
                )
    return flags


def flags_message(flags: list[CoverageFlag]) -> str:
    """One sentence naming how many queries are flagged and why, with the first few."""
    queries = sorted({flag.query for flag in flags})
    share = sorted({flag.query for flag in flags if "share" in flag.reasons})
    adaptive = sorted({flag.query for flag in flags if "adaptive" in flag.reasons})
    parts = []
    if share:
        parts.append(f"{len(share)} with more than {INVALID_WINDOW_SHARE:.0%} invalid windows in a stage")
    if adaptive:
        parts.append(f"{len(adaptive)} with an invalid adaptive window")
    head = ", ".join(queries[:5]) + (", ..." if len(queries) > 5 else "")
    return (
        f"{len(queries)} queries rest on fewer windows than their schedule planned ({'; '.join(parts)}): {head}. "
        "The counts per query, phase and category are in the calibration's coverage"
    )


__all__ = [
    "INVALID_WINDOW_SHARE",
    "PLANNED_PHASE",
    "CalibrationCoverage",
    "CoverageFlag",
    "DegenerateCounts",
    "QueryCounts",
    "StageWindows",
    "WindowCount",
    "coverage_flags",
    "flags_message",
    "window_coverage",
]
