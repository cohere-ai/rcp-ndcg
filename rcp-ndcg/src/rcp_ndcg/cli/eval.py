"""``rcp-ndcg eval``: score rankings, compare systems, explain one query (:mod:`rcp_ndcg.eval`).

* ``score`` -- RCP-nDCG and qrel-nDCG of a rankings file under the suite's scoring protocol (``plain`` for a
  dataset that is no suite), against a public suite (``--suite nanobeir``) or a dataset (``--dataset URI``); the
  gains are a calibration's (``--calibration``), else the dataset's released gains. Count-nDCG needs count gains:
  :func:`rcp_ndcg.eval.evaluate` takes them (``count_gains=``). ``--system NAME`` (repeatable) scores only those
  systems of the file, so one system whose rankings match nothing does not stop the others; ``--json`` prints the
  summary, the per-dataset means and the warnings; ``--per-query`` adds every per-query value, ``--fields`` picks
  top-level fields, and ``--out`` writes the full report (what ``compare --report`` and ``explain --report`` read).
* ``compare`` -- the paired t-test and a query-clustered bootstrap interval between the systems of a report
  (``--report``) or of a run (``--run``; without its reference systems ``candidates`` and ``judge`` unless
  ``--include-reference``).
* ``explain`` -- one query side by side: each system's top k with theta, gain and per-criterion probabilities, from
  a run (``--run``) or a report (``--report``; re-scored by default for the systems the report scored, and
  ``--system NAME`` re-scores only the named ones of the saved rankings); ``--include-text`` adds the query and
  document texts.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any, Literal

import click
from pydantic import BaseModel, Field, model_serializer
from rcp_ndcg_core.protocol import Protocol

from rcp_ndcg.cli.command import command
from rcp_ndcg.errors import DataError, MissingInputError, UsageError
from rcp_ndcg.eval import (
    Comparison,
    DatasetValue,
    EvalReport,
    QueryExplanation,
    QueryValue,
    ReportInputs,
    ReportWarning,
    SummaryValue,
)
from rcp_ndcg.runs.pipeline import REFERENCE_SYSTEMS

#: What ``eval score`` computes; Count-nDCG needs count gains, which only :func:`rcp_ndcg.eval.evaluate` takes.
ScoreMetric = Literal["rcp_ndcg", "qrel_ndcg"]
#: What ``eval compare`` compares: any metric of a report, including one the library wrote with Count-nDCG.
Metric = Literal["rcp_ndcg", "qrel_ndcg", "count_ndcg"]


class EvalScoreRequest(BaseModel):
    rankings: str = Field(description="The rankings to score: parquet, TREC run, JSONL or CSV (one or more systems).")
    suite: str | None = Field(default=None, description="A public suite (nanobeir, bright, vidore, trecdl).")
    dataset: str | None = Field(
        default=None, description="A dataset URI instead of a suite (hf://, beir:, jsonl:, ...)."
    )
    subset: str | None = Field(default=None, description="The subset of a hf:// dataset.")
    revision: str | None = Field(default=None, description="The Hub revision of the data.")
    calibration: str | None = Field(
        default=None, description="A calibration (or run) directory whose gains score RCP-nDCG."
    )
    protocol: str | None = Field(
        default=None, description="Override the suite's protocol (nanobeir, bright, vidore, trecdl, mteb, plain)."
    )
    k: list[int] = Field(default_factory=lambda: [10], description="Cutoffs (repeatable).")
    metrics: list[ScoreMetric] = Field(
        default_factory=lambda: ["rcp_ndcg", "qrel_ndcg"], description="Metrics to compute (repeatable)."
    )
    system: list[str] = Field(
        default_factory=list,
        description="Score only these systems (repeatable; default every system the rankings hold). Use it to "
        "score the healthy systems of a file one of whose systems matches nothing of the dataset.",
    )
    bootstrap: int = Field(default=1000, ge=0, description="Bootstrap resamples of the summary interval.")
    seed: int = Field(default=0, description="The bootstrap seed.")
    out: str | None = Field(
        default=None,
        description="Also write the full report (JSON, every per-query value) here, for `eval compare --report` "
        "and `eval explain --report`.",
    )
    per_query: bool = Field(
        default=False, description="Also print every per-query value (one row per system, query, metric and k)."
    )
    fields: list[str] = Field(
        default_factory=list,
        description="Print only these top-level fields of the result (repeatable), e.g. --fields summary.",
    )


class EvalScoreResult(BaseModel):
    """What ``eval score`` prints: the report's summary, per-dataset means and warnings, where the full report was
    written, and the per-query values when asked for (``--per-query``). ``--fields`` keeps only the named fields;
    a field that is left out is absent, not null."""

    protocol: Protocol | None = None
    gains_source: str | None = None
    metrics: list[str] | None = None
    k: list[int] | None = None
    summary: list[SummaryValue] | None = None
    per_dataset: list[DatasetValue] | None = None
    per_query: list[QueryValue] | None = Field(default=None, description="Present with --per-query.")
    warnings: list[ReportWarning] | None = None
    inputs: ReportInputs | None = None
    out: str | None = Field(default=None, description="Where the full report was written (--out).")

    @model_serializer(mode="wrap")
    def _present_fields_only(self, handler: Any) -> dict[str, Any]:
        return {key: value for key, value in handler(self).items() if value is not None}


_DEFAULT_FIELDS = ("protocol", "gains_source", "metrics", "k", "summary", "per_dataset", "warnings", "inputs", "out")


def _data(request: EvalScoreRequest | ReportInputs) -> dict[str, Any]:
    from rcp_ndcg.data import load_dataset

    if (request.suite is None) == (request.dataset is None):
        raise UsageError(
            "pass exactly one of --suite and --dataset",
            hint="score a public suite with --suite nanobeir, or a dataset URI with --dataset jsonl:rows.jsonl",
        )
    if request.suite is not None:
        return {"dataset": load_dataset(f"suite:{request.suite}", subset=request.subset, revision=request.revision)}
    assert request.dataset is not None
    return {"dataset": load_dataset(request.dataset, subset=request.subset, revision=request.revision)}


def _absolute(location: str | None) -> str | None:
    """A local path (or the path of a ``scheme:path`` dataset URI) made absolute; URIs of remote stores unchanged."""
    if location is None or "://" in location:
        return location
    scheme, sep, rest = location.partition(":")
    if sep and len(scheme) > 1 and scheme not in ("suite",):
        return f"{scheme}:{Path(rest).absolute()}" if Path(rest).exists() else location
    return str(Path(location).absolute()) if Path(location).exists() else location


def _inputs(request: EvalScoreRequest, *, dataset: Any) -> ReportInputs:
    """The report's inputs: the request's paths, and the provenance the loaded dataset records."""
    return ReportInputs(
        rankings=_absolute(request.rankings) or request.rankings,
        suite=request.suite,
        dataset=_absolute(request.dataset),
        subset=request.subset,
        revision=dataset.revision,
        split=dataset.split,
        task=dataset.task,
        calibration=_absolute(request.calibration),
    )


def _selected(fields: list[str], *, per_query: bool) -> set[str]:
    known = list(EvalScoreResult.model_fields)
    unknown = [name for name in fields if name not in known]
    if unknown:
        close = difflib.get_close_matches(unknown[0], known, n=1)
        raise UsageError(
            f"--fields {unknown[0]!r} is not a field of the result"
            + (f"; did you mean {close[0]!r}?" if close else ""),
            hint=f"fields: {', '.join(known)}",
            details={"unknown": unknown, "fields": known, "did_you_mean": close[0] if close else None},
        )
    if fields:
        return set(fields)
    return {*_DEFAULT_FIELDS, *(("per_query",) if per_query else ())}


def _systems_are_known(held: list[str], requested: list[str], *, what: str) -> None:
    """Refuse an unknown ``--system``/``--baseline`` value as the command-line mistake it is (exit 2).

    The library keeps its own ``ConfigError`` for its Python callers (``systems=`` is a config value there);
    on the command line an unknown name is a usage error, like an unknown ``--fields`` or ``--metrics`` one.
    The message and the systems list are the library's.
    """
    unknown = sorted(set(requested) - set(held))
    if unknown:
        if what == "systems":
            message = f"systems {unknown} are not in the rankings; systems: {held}"
            hint = "score one of the systems the rankings hold (--system, repeatable)"
        else:
            message = f"baseline {unknown[0]!r} is not a compared system; systems: {held}"
            hint = "pass one of the compared systems (--baseline), or drop --baseline to compare every pair"
        raise UsageError(message, hint=hint, details={"unknown": unknown, "systems": held})


def _gains(calibration: str | None) -> Any:
    if calibration is None:
        return None
    from rcp_ndcg.cli.calibration import _load

    return _load(calibration)


def _score_text(report: EvalScoreResult) -> str:
    lines = []
    if report.protocol is not None:
        lines += [f"protocol {report.protocol.name}, gains from {report.gains_source}", ""]
    lines.append(f"  {'system':<28} {'metric':<11} {'k':>3}  {'value':>7}  interval")
    for row in report.summary or []:
        value = f"{row.value:.4f}" if row.value is not None else "   -   "
        interval = (
            f"[{row.ci_low:.4f}, {row.ci_high:.4f}]" if row.ci_low is not None and row.ci_high is not None else ""
        )
        lines.append(f"  {row.system:<28} {row.metric:<11} {row.k:>3}  {value:>7}  {interval}")
    if report.per_query:
        lines += ["", "  per query"]
        for row in report.per_query:
            value = f"{row.value:.4f}" if row.value is not None else "   -   "
            lines.append(f"  {row.system:<28} {row.metric:<11} {row.k:>3}  {value:>7}  {row.dataset} {row.query_id}")
    for warning in report.warnings or []:
        lines.append(f"  warning [{warning.code}]: {warning.message}")
    if report.out is not None:
        lines.append(f"  full report: {report.out}")
    return "\n".join(lines)


@command(
    "eval score",
    request=EvalScoreRequest,
    result=EvalScoreResult,
    output_schema="rcp-ndcg.eval-score.v1",
    text=_score_text,
)
def eval_score(request: EvalScoreRequest) -> EvalScoreResult:
    """Score rankings with RCP-nDCG and qrel-nDCG under the suite's scoring protocol (plain for a non-suite dataset).

    The result is the summary, the per-dataset means and the warnings; --per-query adds every per-query value,
    and --out writes the full report.
    """
    from rcp_ndcg.data import load_rankings
    from rcp_ndcg.eval import evaluate

    selected = _selected(request.fields, per_query=request.per_query)
    data = _data(request)
    rankings = load_rankings(request.rankings)
    if request.system:
        _systems_are_known(rankings.systems, request.system, what="systems")
    report = evaluate(
        rankings,
        **data,
        gains=_gains(request.calibration),
        protocol=request.protocol,
        k=request.k,
        metrics=request.metrics,
        systems=request.system or None,
        bootstrap=request.bootstrap,
        seed=request.seed,
    )
    report = report.model_copy(update={"inputs": _inputs(request, dataset=data["dataset"])})
    if request.out is not None:
        from rcp_ndcg import storage

        storage.write_text(request.out, report.to_json(indent=2))
    values = {name: getattr(report, name) for name in selected if name in EvalReport.model_fields}
    return EvalScoreResult(**values, out=request.out if "out" in selected else None)


class EvalCompareRequest(BaseModel):
    report: str | None = Field(default=None, description="A report file from `eval score --out`.")
    run: str | None = Field(default=None, description="A run directory (its systems, scored with its calibration).")
    baseline: str | None = Field(default=None, description="Compare every system against this one.")
    metric: Metric = Field(default="rcp_ndcg", description="The metric compared.")
    k: int | None = Field(default=None, ge=1, description="The cutoff (may be omitted when the report has one).")
    alpha: float = Field(default=0.05, gt=0, lt=1, description="Significance level; the interval covers 1 - alpha.")
    bootstrap: int = Field(default=10_000, ge=0, description="Bootstrap resamples of the interval.")
    seed: int = Field(default=0, description="The bootstrap seed.")
    include_reference: bool = Field(
        default=False,
        description="With --run, also compare the run's reference systems: candidates (the pool order) and judge "
        "(the judge's own abilities, RCP-nDCG 1 by construction).",
    )


def _load_report(path: str) -> EvalReport:
    report_path = Path(path)
    if not report_path.is_file():
        raise MissingInputError(f"no report at {report_path}", hint="write one with `rcp-ndcg eval score --out`")
    return EvalReport.model_validate(json.loads(report_path.read_text(encoding="utf-8")))


def _compare_text(comparison: Comparison) -> str:
    lines = [f"{comparison.metric}@{comparison.k}, paired t-test, alpha {comparison.alpha}", ""]
    for pair in comparison.pairs:
        span = (
            f" [{pair.ci_low:+.4f}, {pair.ci_high:+.4f}]"
            if pair.ci_low is not None and pair.ci_high is not None
            else ""
        )
        p = f"  p={pair.p_value:.4g}" if pair.p_value is not None else ""
        lines.append(f"  {pair.system_b} - {pair.system_a}: {pair.delta:+.4f}{span}{p}")
    return "\n".join(lines)


@command(
    "eval compare",
    request=EvalCompareRequest,
    result=Comparison,
    output_schema="rcp-ndcg.comparison.v1",
    text=_compare_text,
)
def eval_compare(request: EvalCompareRequest) -> Comparison:
    """Compare systems: delta (B minus A), paired t-test, query-clustered bootstrap interval, sign flips."""
    from rcp_ndcg.eval import compare

    if (request.report is None) == (request.run is None):
        raise UsageError(
            "pass exactly one of --report and --run",
            hint="compare a report written by `eval score --out` with --report, or a run directory with --run",
        )
    systems = None
    if request.run is not None:
        from rcp_ndcg.runs.inspect import evaluation_report

        report = evaluation_report(request.run)
        if not request.include_reference:
            systems = [s for s in report.systems if s not in REFERENCE_SYSTEMS or s == request.baseline]
            if len(systems) < 2:
                raise DataError(
                    f"the run scores {len(systems)} system(s) of its own besides the reference systems "
                    f"{list(REFERENCE_SYSTEMS)}; a comparison needs two",
                    hint="add systems to the run (evaluation.systems), or pass --include-reference",
                    details={"systems": report.systems},
                )
    else:
        assert request.report is not None
        report = _load_report(request.report)
    if request.baseline is not None:
        compared = systems if systems is not None else report.systems
        _systems_are_known(list(compared), [request.baseline], what="baseline")
    return compare(
        report,
        baseline=request.baseline,
        metric=request.metric,
        k=request.k,
        alpha=request.alpha,
        bootstrap=request.bootstrap,
        seed=request.seed,
        systems=systems,
    )


class EvalExplainRequest(BaseModel):
    run: str | None = Field(default=None, description="A run directory with a calibration.")
    report: str | None = Field(
        default=None,
        description="A report from `eval score --out`: its rankings, data and calibration are read again.",
    )
    query_id: str = Field(description="The query to explain.")
    subset: str | None = Field(default=None, description="The query's dataset, when a suite's report has it twice.")
    system: list[str] = Field(
        default_factory=list,
        description="With --report, re-score only these systems of the saved rankings (repeatable; default the "
        "systems the report scored).",
    )
    k: int = Field(default=10, ge=1, description="Documents shown per system (and the cutoff of the deltas).")
    include_text: bool = Field(default=False, description="Add the query and document texts.")


class ExplainedQuery(QueryExplanation):
    """A :class:`~rcp_ndcg.eval.QueryExplanation`, with the texts it is about when asked for (--include-text)."""

    query: str | None = Field(default=None, description="The query text (with --include-text).")
    texts: dict[str, str | None] = Field(
        default_factory=dict, description="{doc_id: text} of every shown document (with --include-text)."
    )


def _explain_text(result: ExplainedQuery) -> str:
    lines = [f"query {result.query_id}: {result.query or ''}".rstrip()]
    for system in result.systems:
        lines.append(f"  {system.system}  " + "  ".join(f"{key}={value:.4f}" for key, value in system.values.items()
                                                     if value is not None))  # fmt: skip
        for doc in system.top:
            theta = f"{doc.theta:+.3f}" if doc.theta is not None else "unjudged"
            gain = f"{doc.gain:.3f}" if doc.gain is not None else "-"
            lines.append(f"    {doc.rank:>2}. {doc.doc_id:<24} theta {theta:>9}  gain {gain}")
            if text := result.texts.get(doc.doc_id):
                lines.append(f"        {text[:200]}")
    for delta in result.deltas:
        lines.append(
            f"  {delta.system_b} - {delta.system_a} @{result.k}: {delta.total:+.4f} "
            f"(selection {delta.selection:+.4f}, ordering {delta.ordering:+.4f})"
        )
    return "\n".join(lines)


def _explain_report(request: EvalExplainRequest) -> tuple[QueryExplanation, Any]:
    """Explain a query of a saved report: score its recorded inputs again (no bootstrap), then explain."""
    from rcp_ndcg.data import load_rankings
    from rcp_ndcg.eval import evaluate, explain

    assert request.report is not None
    saved = _load_report(request.report)
    if saved.inputs is None:
        raise UsageError(
            f"{request.report} records no inputs to explain from",
            hint="write the report with `rcp-ndcg eval score --out` (which records its rankings, data and calibration)",
        )
    inputs = saved.inputs
    rankings = load_rankings(inputs.rankings)
    if request.system:
        _systems_are_known(rankings.systems, request.system, what="systems")
    report = evaluate(
        rankings,
        **_data(inputs),
        gains=_gains(inputs.calibration),
        protocol=saved.protocol,
        k=saved.k,
        metrics=saved.metrics,
        systems=request.system or saved.systems or None,  # the report's own systems, unless narrowed or none
        bootstrap=0,
    )
    calibration = _calibration(inputs.calibration)
    explained = explain(report, request.query_id, calibration=calibration, k=request.k, dataset=request.subset)
    data = report._inputs["dataset"]
    part = next((p for p in data.parts if p.name == explained.dataset), data)
    return explained, part


def _calibration(location: str | None) -> Any:
    if location is None:
        return None
    from rcp_ndcg.cli.calibration import _load

    return _load(location)


@command(
    "eval explain",
    request=EvalExplainRequest,
    result=ExplainedQuery,
    output_schema="rcp-ndcg.query-explanation.v1",
    text=_explain_text,
)
def eval_explain(request: EvalExplainRequest) -> ExplainedQuery:
    """Explain one query of a run or a saved report: each system's top k with theta, gain and per-criterion
    probabilities, and the gaps between systems split into selection and ordering."""
    if (request.report is None) == (request.run is None):
        raise UsageError(
            "pass exactly one of --run and --report",
            hint="explain a query of a run directory with --run, or of a saved report with --report",
        )
    if request.run is not None:
        if request.system:
            raise UsageError(
                "--system re-scores the rankings of a saved report and has no effect with --run",
                hint="drop --system, or explain a report written by `eval score --out` (--report)",
            )
        if request.subset:
            raise UsageError(
                "--subset names the dataset of a saved report's query and has no effect with --run",
                hint="drop --subset, or explain a report written by `eval score --out` (--report)",
            )
        from rcp_ndcg.runs.inspect import explain_query

        explained, dataset = explain_query(request.run, request.query_id, k=request.k)
    else:
        explained, dataset = _explain_report(request)
    payload = explained.model_dump()
    if not request.include_text:
        return ExplainedQuery.model_validate(payload)
    query = dataset.queries.get(request.query_id)
    corpus = dataset.corpus
    texts = {
        doc.doc_id: (corpus[doc.doc_id].text if doc.doc_id in corpus else None)
        for system in explained.systems
        for doc in system.top
    }
    return ExplainedQuery.model_validate(
        {**payload, "query": query.text if query is not None else None, "texts": texts}
    )


@click.group(name="eval", help="Score rankings, compare systems and explain queries.")
def eval_group() -> None:
    """``rcp-ndcg eval``."""


for _command in (eval_score, eval_compare, eval_explain):
    eval_group.add_command(_command)


__all__ = [
    "REFERENCE_SYSTEMS",
    "EvalCompareRequest",
    "EvalExplainRequest",
    "EvalScoreRequest",
    "EvalScoreResult",
    "ExplainedQuery",
    "eval_group",
]
