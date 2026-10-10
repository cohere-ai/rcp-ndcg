"""``rcp-ndcg calibration``: fit the 2PL calibration, extend it without refitting, and show it.

* ``fit`` -- :func:`rcp_ndcg.calibration.calibrate` over one or more judgement stores: with the tournament or
  without it (``--mode``), for one judge or several pooled (``--judges pooled``).
* ``score`` -- :func:`~rcp_ndcg.calibration.score_documents`: documents the calibration lacks, from their own rubric
  judgements, items frozen.
* ``insert`` -- :func:`~rcp_ndcg.calibration.insert_documents`: new documents into a tournament calibration, the
  anchor report included. ``insert --dry-run --query Q --doc D`` prints the opponent window to judge first
  (:func:`~rcp_ndcg.calibration.select_opponents`).
* ``show`` -- parameters, coverage and diagnostics of a calibration (or a run's).

``score`` and ``insert`` write the extended calibration to ``--out``; the input is never modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import click
from pydantic import BaseModel, Field
from rcp_ndcg_core.irt import DEFAULT_SE_TARGET, MIN_OPPONENTS
from rcp_ndcg_core.schemas import JudgementFamily

from rcp_ndcg.calibration.coverage import CalibrationCoverage
from rcp_ndcg.calibration.diagnostics import Diagnostics
from rcp_ndcg.calibration.extend import MAX_GAIN_SHIFT, Extension
from rcp_ndcg.cli.command import command
from rcp_ndcg.errors import MissingInputError, UsageError

_CALIBRATION_HELP = "A calibration directory (what `calibration fit` writes), or a run directory."


def _calibration_dir(path: str) -> Path:
    root = Path(path)
    if (root / "items.json").is_file():
        return root
    if (root / "calibration" / "items.json").is_file():
        return root / "calibration"
    raise MissingInputError(f"{path} holds no calibration", hint="run `rcp-ndcg calibration fit` first")


def _load(path: str) -> Any:
    from rcp_ndcg.calibration import Calibration

    return Calibration.load(_calibration_dir(path))


def _judgement_store(path: str) -> str:
    root = Path(path)
    if (root / "identity.json").is_file():
        return str(root)
    if (root / "judgements" / "identity.json").is_file():
        return str(root / "judgements")
    raise MissingInputError(f"{path} holds no judgement store", hint="run `rcp-ndcg judge tournament|rubric` first")


class ItemSummary(BaseModel):
    """The criterion parameters: discrimination ``gamma`` (dimensionless) and difficulty ``beta`` (logits)."""

    criteria: list[str]
    gamma: list[float]
    beta: list[float]


class CalibrationSummary(BaseModel):
    """A calibration at a glance: its mode, items, what it covers and how reliable it is."""

    calibration: str = Field(description="The calibration directory.")
    mode: Literal["tournament", "rubric_only"]
    fingerprint: str
    datasets: list[str]
    queries: int
    documents: int = Field(description="Documents with an ability, from the fit and from extensions.")
    sources: dict[str, int] = Field(description="Abilities per source: fit, scored, inserted.")
    items: ItemSummary
    families: dict[str, JudgementFamily] = Field(description="The judgement families fitted, by family key.")
    judge_severity: dict[str, float] = Field(description="Per-judge logit offsets of a pooled fit.")
    coverage: CalibrationCoverage
    diagnostics: Diagnostics
    warnings: list[dict[str, str]] = Field(
        default_factory=list, description="The fit's typed warnings (code, message), e.g. INVALID_WINDOWS."
    )


def summary(calibration: Any, where: str) -> CalibrationSummary:
    """The :class:`CalibrationSummary` of a :class:`~rcp_ndcg.calibration.Calibration`."""
    sources: dict[str, int] = {}
    for row in calibration.thetas:
        sources[row.source] = sources.get(row.source, 0) + 1
    return CalibrationSummary(
        calibration=where,
        mode=calibration.mode,
        fingerprint=calibration.fingerprint,
        datasets=calibration.datasets,
        queries=len({(row.dataset, row.query_id) for row in calibration.thetas}),
        documents=len(calibration.thetas),
        sources=sources,
        items=ItemSummary(
            criteria=list(calibration.items.criteria),
            gamma=list(calibration.items.gamma),
            beta=list(calibration.items.beta),
        ),
        families={key: family.model_dump(mode="json") for key, family in calibration.families.items()},
        judge_severity=dict(calibration.judge_severity),
        coverage=calibration.coverage.model_dump(mode="json"),
        diagnostics=calibration.diagnostics.model_dump(mode="json"),
        warnings=calibration.warnings,
    )


def _summary_text(result: CalibrationSummary) -> str:
    lines = [
        f"calibration {result.calibration} ({result.mode}, {result.fingerprint})",
        f"  {result.queries} queries of {', '.join(result.datasets)}, {result.documents} documents {result.sources}",
        "  criterion  gamma    beta",
    ]
    for name, gamma, beta in zip(result.items.criteria, result.items.gamma, result.items.beta, strict=True):
        lines.append(f"  {name:<9} {gamma:6.3f} {beta:+7.3f}")
    for judge, offset in result.judge_severity.items():
        lines.append(f"  severity {judge}: {offset:+.3f}")
    for warning in result.warnings:
        lines.append(f"  warning {warning['code']}: {warning['message']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------
# fit and show
# ----------------------------------------------------------------------------------------------------------------


class CalibrationFitRequest(BaseModel):
    judgements: list[str] = Field(
        min_length=1, description="Judgement stores to fit (repeat for several judges); a run directory works too."
    )
    mode: Literal["auto", "tournament", "rubric_only"] = Field(
        default="auto", description="tournament (theta = tau * theta_BT + alpha), rubric_only, or auto."
    )
    judges: Literal["single", "pooled"] = Field(
        default="single", description="single judge, or pooled judges sharing a rubric (one severity per judge)."
    )
    out: str = Field(description="The calibration directory to write.")
    strict: bool = Field(
        default=False,
        description="Refuse to fit when a query has more than 5% invalid windows in a stage or an invalid "
        "adaptive window (default: warn).",
    )


@command(
    "calibration fit", request=CalibrationFitRequest, result=CalibrationSummary, text=_summary_text, read_only=False
)
def calibration_fit(request: CalibrationFitRequest) -> CalibrationSummary:
    """Fit the 2PL calibration to judgements, with or without the tournament, for one judge or pooled judges."""
    from rcp_ndcg.calibration import calibrate, judged_bt_l2, read_judgements

    stores = [_judgement_store(path) for path in request.judgements]
    sets = [read_judgements(store) for store in stores]
    fitted = calibrate(
        sets[0] if len(sets) == 1 else sets,
        mode=request.mode,
        judges=request.judges,
        judged_bt_l2=judged_bt_l2(*stores),
        strict=request.strict,
    )
    fitted.save(request.out)
    return summary(fitted, request.out)


class CalibrationShowRequest(BaseModel):
    calibration: str = Field(description=_CALIBRATION_HELP)


@command("calibration show", request=CalibrationShowRequest, result=CalibrationSummary, text=_summary_text)
def calibration_show(request: CalibrationShowRequest) -> CalibrationSummary:
    """Show a calibration: items, coverage, per-judge severity, reliability diagnostics and provenance."""
    return summary(_load(request.calibration), str(_calibration_dir(request.calibration)))


# ----------------------------------------------------------------------------------------------------------------
# score and insert
# ----------------------------------------------------------------------------------------------------------------


class ExtensionResult(BaseModel):
    """What an extension added, and where the extended calibration is."""

    out: str | None = Field(description="The extended calibration directory.")
    extension: Extension


class CalibrationScoreRequest(BaseModel):
    calibration: str = Field(description=_CALIBRATION_HELP)
    judgements: str = Field(description="The rubric judgement store of the documents to score.")
    out: str = Field(description="The extended calibration directory to write.")
    se_target: float = Field(
        default=DEFAULT_SE_TARGET, gt=0, description="The standard error (logits) the evidence should reach."
    )


@command("calibration score", request=CalibrationScoreRequest, result=ExtensionResult, read_only=False)
def calibration_score(request: CalibrationScoreRequest) -> ExtensionResult:
    """Score documents a calibration lacks from their own rubric judgements, the items frozen."""
    from rcp_ndcg.calibration import read_judgements, score_documents

    calibration = _load(request.calibration)
    extension = score_documents(
        calibration,
        read_judgements(_judgement_store(request.judgements)),
        se_target=request.se_target,
    )
    calibration.extended(extension).save(request.out)
    return ExtensionResult(out=request.out, extension=extension)


class OpponentPlan(BaseModel):
    """The windows to judge a new document in, against opponents from the calibration."""

    query_id: str
    doc_id: str
    windows: list[list[str]] = Field(
        description="Each window: the new document first, then its opponents. Judge exactly these windows into "
        "the calibration's tournament store: `judge tournament --plan <this plan's file> --out <store>`."
    )
    calls: int | None = Field(
        default=None,
        description="Judge calls the windows take under the store's schedule (each window twice when it mirrors); "
        "null without --judgements.",
    )
    opponents: int = Field(description="Distinct opponents the windows hold.")
    capped: bool = Field(
        description="Whether --n asked for more opponents than the query has other documents; a larger --n then "
        "adds nothing."
    )


class InsertResult(BaseModel):
    """``insert --dry-run``: the opponent windows; otherwise the extension and the extended calibration."""

    plan: OpponentPlan | None = None
    out: str | None = None
    extension: Extension | None = None


class CalibrationInsertRequest(BaseModel):
    calibration: str = Field(description=_CALIBRATION_HELP)
    judgements: str | None = Field(
        default=None,
        description="The tournament store with the fitted windows and the new documents' windows. With --dry-run: "
        "the calibration's tournament store, whose schedule sizes the windows.",
    )
    out: str | None = Field(
        default=None,
        description="The extended calibration directory to write; with --dry-run, the plan file (JSON) to write "
        "for `judge tournament --plan`.",
    )
    dry_run: bool = Field(default=False, description="Plan the opponent windows for --query/--doc; insert nothing.")
    query: str | None = Field(default=None, description="With --dry-run: the query of the new document.")
    doc: str | None = Field(default=None, description="With --dry-run: the new document.")
    dataset_name: str | None = Field(
        default=None, description="With --dry-run: the dataset, when several are calibrated."
    )
    n: int = Field(default=9, ge=1, description="With --dry-run: opponents in all.")
    window: int | None = Field(
        default=None,
        ge=2,
        description="With --dry-run: documents per window, the new one included. Default: the schedule.window of "
        "the --judgements store, else one window.",
    )
    max_gain_shift: float = Field(default=MAX_GAIN_SHIFT, ge=0, description="The anchor tolerance, gain units.")
    se_target: float = Field(
        default=DEFAULT_SE_TARGET, gt=0, description="The standard error (logits) the evidence must reach."
    )
    min_opponents: int = Field(
        default=MIN_OPPONENTS, ge=1, description="Distinct opponents each new document must face."
    )


@command("calibration insert", request=CalibrationInsertRequest, result=InsertResult, read_only=False)
def calibration_insert(request: CalibrationInsertRequest) -> InsertResult:
    """Insert new documents into a tournament calibration (anchor report included); --dry-run picks opponents."""
    from rcp_ndcg.calibration import insert_documents, read_judgements, select_opponents

    calibration = _load(request.calibration)
    if request.dry_run:
        from rcp_ndcg.judging.store import JudgementStore

        if request.query is None or request.doc is None:
            raise UsageError(
                "--dry-run needs --query and --doc",
                hint="name the new document: --query q1 --doc doc1",
            )
        schedule = JudgementStore(_judgement_store(request.judgements)).schedule("tournament") if (
            request.judgements
        ) else None  # fmt: skip
        window = request.window or (schedule.window if schedule is not None else None)
        windows = select_opponents(
            calibration, request.query, request.doc, n=request.n, window=window, dataset=request.dataset_name
        )
        calls = None if schedule is None else len(windows) * (2 if getattr(schedule, "mirror", False) else 1)
        opponents = len({doc for window in windows for doc in window[1:]})
        planned = OpponentPlan(
            query_id=request.query,
            doc_id=request.doc,
            windows=windows,
            calls=calls,
            opponents=opponents,
            capped=opponents < request.n,
        )
        if request.out is not None:
            Path(request.out).write_text(planned.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return InsertResult(plan=planned, out=request.out)
    if request.judgements is None or request.out is None:
        raise UsageError(
            "inserting needs --judgements and --out (or --dry-run to choose opponents first)",
            hint="pass the tournament store and where the extended calibration goes: --judgements store --out extended",
        )
    extension = insert_documents(
        calibration,
        read_judgements(_judgement_store(request.judgements)),
        max_gain_shift=request.max_gain_shift,
        se_target=request.se_target,
        min_opponents=request.min_opponents,
    )
    calibration.extended(extension).save(request.out)
    return InsertResult(out=request.out, extension=extension)


@click.group(name="calibration", help="Fit the 2PL calibration, extend it without refitting, and show it.")
def calibration_group() -> None:
    """``rcp-ndcg calibration``."""


for _command in (calibration_fit, calibration_score, calibration_insert, calibration_show):
    calibration_group.add_command(_command)


__all__ = [
    "CalibrationFitRequest",
    "CalibrationInsertRequest",
    "CalibrationScoreRequest",
    "CalibrationShowRequest",
    "CalibrationSummary",
    "ExtensionResult",
    "InsertResult",
    "ItemSummary",
    "OpponentPlan",
    "calibration_group",
    "summary",
]
