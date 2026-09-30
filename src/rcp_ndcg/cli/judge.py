"""``rcp-ndcg judge``: the two judging stages over a dataset's candidate pools, and re-parsing a store.

``judge tournament`` (Stage A) and ``judge rubric`` (Stage B, criteria C1 to C5) run :func:`rcp_ndcg.llm.judge`
into an append-only judgement store (``--out``): a rerun asks only the windows that are missing, and ``--docs``
re-judges a subset. ``--estimate`` counts the pass's calls and tokens (of ``--docs`` alone when given) without
calling the judge.
``judge reparse`` reads a store's stored answers again with the current parser into a new store
(:func:`rcp_ndcg.llm.reparse`), without calling the judge. Serving the model is the user's: any
OpenAI-compatible URL judges (see ``docs/concepts/serving.md``).

The judge is ``--judge fake`` (the offline judge), ``--judge <config.yaml>``, ``--judge <name>`` (a shipped
config, :mod:`rcp_ndcg.llm.judges`), or an ad-hoc endpoint ``--judge-url URL --judge-model ID``. ``--set``
overrides one field of ``judge.*``, ``schedule.*`` or ``preprocessing.*`` (e.g. ``--set judge.concurrency=8
--set schedule.window=5``).
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, ClassVar, Literal

import click
from pydantic import BaseModel, Field

from rcp_ndcg.cli._args import DatasetInput
from rcp_ndcg.cli.command import command
from rcp_ndcg.errors import MissingInputError, UsageError
from rcp_ndcg.llm.client import JudgeConfig, Usage
from rcp_ndcg.llm.cost import CostEstimate
from rcp_ndcg.runs.mirror import mirrored, restore
from rcp_ndcg.storage import local_dir

_JUDGE_HELP = "fake | a judge config YAML | a shipped judge config's name (e.g. gpt_oss_120b)."
_SET_HELP = "Override judge.<field>, schedule.<field> or preprocessing.<field>: KEY=VALUE, repeatable."


class JudgeSource(BaseModel):
    """Which judge: a config, the offline judge, or an ad-hoc OpenAI-compatible endpoint."""

    #: The ``--set`` sections this command reads.
    SECTIONS: ClassVar[tuple[str, ...]] = ("judge", "schedule", "preprocessing")

    judge: str | None = Field(default=None, description=_JUDGE_HELP)
    judge_url: str | None = Field(default=None, description="An OpenAI-compatible base URL (.../v1), ad hoc.")
    judge_model: str | None = Field(
        default=None, description="The served model name (with --judge-url, or to override)."
    )
    set: list[str] = Field(default_factory=list, description=_SET_HELP)
    seed: int | None = Field(
        default=None,
        description="Seed of the window schedule and of the offline judge (default: the schedule's own, 42).",
    )

    def _fake(self) -> dict[str, Any]:
        from rcp_ndcg.llm import TournamentSchedule

        seed = self.seed if self.seed is not None else TournamentSchedule.model_fields["seed"].default
        return JudgeConfig.fake(seed).model_dump(mode="json")

    def sections(self) -> dict[str, Any]:
        """One mapping per section of :attr:`SECTIONS`, with ``--set`` applied."""
        from rcp_ndcg.llm.judges import judge_config_path
        from rcp_ndcg.support.config import apply_overrides, load_config

        if self.judge_url is not None:
            if self.judge is not None:
                raise UsageError("pass --judge or --judge-url, not both")
            if self.judge_model is None:
                raise UsageError("--judge-url needs --judge-model (the served model name)")
            judge: dict[str, Any] = {"base_url": self.judge_url, "model": self.judge_model}
        elif self.judge is None:
            raise UsageError("name the judge: --judge fake | <config.yaml> | <name>, or --judge-url and --judge-model")
        elif self.judge == "fake":
            judge = self._fake()
            if self.judge_model is not None:  # two offline judges pool only under two names
                judge["model"] = self.judge_model
        else:
            judge = load_config(judge_config_path(self.judge))
            if self.judge_model is not None:
                judge["model"] = self.judge_model
        unknown = [o for o in self.set if o.split("=", 1)[0].split(".", 1)[0] not in self.SECTIONS]
        if unknown:
            keys = ", ".join(f"{section}." for section in self.SECTIONS)
            raise UsageError(f"--set {unknown[0]!r}: the key starts with one of {keys}")
        return apply_overrides({"judge": judge, **{section: {} for section in self.SECTIONS[1:]}}, self.set)

    def judge_config(self, sections: dict[str, Any] | None = None) -> JudgeConfig:
        """The :class:`~rcp_ndcg.llm.JudgeConfig` (``--set judge.*`` applied)."""
        from pydantic import ValidationError

        try:
            return JudgeConfig.model_validate((sections or self.sections())["judge"])
        except ValidationError as exc:
            from rcp_ndcg.support.config import config_error

            raise config_error(
                exc, model=JudgeConfig, source="the judge config", overrides=self.set, prefix="judge"
            ) from exc


# ----------------------------------------------------------------------------------------------------------------
# tournament and rubric
# ----------------------------------------------------------------------------------------------------------------


class JudgeRequest(JudgeSource, DatasetInput):
    candidates: str | None = Field(
        default=None, description="A rankings file of the pools to judge (default: the dataset's own pools)."
    )
    system: str | None = Field(default=None, description="The system of a rankings file that holds several.")
    depth: int = Field(default=150, ge=1, description="Candidates judged per query (the top of each pool).")
    limit: int | None = Field(default=None, ge=1, description="Only the first N queries (a smoke run).")
    docs: list[str] = Field(
        default_factory=list,
        description="Judge only these documents: QUERY_ID:DOC_ID, or DOC_ID in every pool holding it (repeatable).",
    )
    out: str = Field(description="The judgement store directory (append-only; a rerun asks only what is missing).")
    estimate: bool = Field(
        default=False,
        description="Print calls, tokens and wall time of the pass into an empty store; judge nothing. "
        "Refuses what the pass would refuse (another identity in --out).",
    )
    force: bool = Field(default=False, description="Supersede judgements of another identity in --out.")
    mirror: str | None = Field(
        default=None,
        description="Mirror the store to any fsspec URI (e.g. s3://bucket/store) while judging, restoring what "
        "--out lacks from it first.",
    )


class TournamentRequest(JudgeRequest):
    plan: list[str] = Field(
        default_factory=list,
        description="Ask exactly the windows of these plan files (`calibration insert --plan --out FILE`), with the "
        "schedule of the --out store (repeatable; no --docs, --set schedule.* or --seed).",
    )


class JudgeReport(BaseModel):
    """What a judging command did (``judged``) or would take (``estimate``)."""

    stage: Literal["tournament", "rubric"]
    mode: Literal["judged", "estimate"]
    out: str
    dataset: str
    queries: int = Field(description="Queries judged (or to be judged).")
    documents: int = Field(description="Documents in the judged pools.")
    stored: int = Field(description="Windows of this stage in the store before the command ran.")
    judgements: int | None = Field(default=None, description="Judgements of the judged queries now in the store.")
    valid: int | None = Field(default=None, description="Of which parsed into placements.")
    families: list[str] = Field(default_factory=list, description="The judgement family keys.")
    usage: Usage | None = Field(default=None, description="Calls and tokens of this command.")
    estimate: CostEstimate | None = None


def _pools(request: JudgeRequest, dataset: Any) -> dict[str, list[str]]:
    from rcp_ndcg.data import load_rankings

    if request.candidates is not None:
        rankings = load_rankings(request.candidates)
        system = request.system or (rankings.systems[0] if len(rankings.systems) == 1 else None)
        if system is None:
            raise UsageError(f"{request.candidates} holds systems {rankings.systems}; pass --system")
        queries = rankings.queries(system=system, dataset=dataset.name)
        pools = {q: sorted(s, key=lambda d: (s[d], d), reverse=True) for q, s in queries.items()}
    elif dataset.candidates is not None:
        pools = {q: list(pool) for q, pool in dataset.candidates.items()}
    else:
        pools = {q: list(docs) for q, docs in dataset.qrels.items() if docs}
    if request.limit is not None:
        pools = dict(list(pools.items())[: request.limit])
    return {q: pool[: request.depth] for q, pool in pools.items()}


def _docs(request: JudgeRequest, pools: dict[str, list[str]]) -> dict[str, list[str]] | None:
    if not request.docs:
        return None
    wanted: dict[str, list[str]] = {}
    for item in request.docs:
        query, sep, doc = item.partition(":")
        if sep:
            if doc not in pools.get(query, ()):
                raise UsageError(f"--docs {item}: {doc!r} is not among the candidates of query {query!r}")
            wanted.setdefault(query, []).append(doc)
            continue
        holders = [q for q, pool in pools.items() if item in pool]
        if not holders:
            raise UsageError(f"--docs {item}: no candidate pool holds {item!r}")
        for q in holders:
            wanted.setdefault(q, []).append(item)
    return wanted


def _planned(request: TournamentRequest, sections: dict[str, Any]) -> tuple[dict[str, list], Any]:
    """``(windows, schedule)`` of ``--plan``: the plans' windows by query, and the ``--out`` store's schedule."""
    from rcp_ndcg.cli.calibration import OpponentPlan
    from rcp_ndcg.llm.store import JudgementStore

    if request.docs or sections["schedule"] or request.seed is not None:
        raise UsageError(
            "with --plan the windows are the plan's and the schedule is the --out store's: drop --docs, "
            "--set schedule.* and --seed"
        )
    schedule = JudgementStore(request.out).schedule("tournament")
    if schedule is None:
        raise MissingInputError(
            f"{request.out} holds no tournament judgements, whose schedule --plan judges with",
            hint="pass the calibration's tournament store as --out",
        )
    windows: dict[str, list] = {}
    for path in request.plan:
        planned = OpponentPlan.model_validate_json(Path(path).read_text(encoding="utf-8"))
        windows.setdefault(planned.query_id, []).extend(planned.windows)
    return windows, schedule


def _run_stage(stage: Literal["tournament", "rubric"], request: JudgeRequest) -> JudgeReport:
    from pydantic import ValidationError

    from rcp_ndcg.data.preprocess import Preprocessing
    from rcp_ndcg.llm import JudgeClient, RubricSchedule, TournamentSchedule, estimate, judge
    from rcp_ndcg.llm.judging import preflight
    from rcp_ndcg.llm.store import JudgementStore

    sections = request.sections()
    config = request.judge_config(sections)
    local_dir(request.out, "a judgement store (--out)")
    if request.mirror and not request.estimate:
        restore(request.out, request.mirror)
    schedule_type = TournamentSchedule if stage == "tournament" else RubricSchedule
    try:
        seed = {} if request.seed is None else {"seed": request.seed}
        schedule = (
            schedule_type.model_validate({**sections["schedule"], **seed}) if sections["schedule"] or seed else None
        )
        preprocessing = Preprocessing.model_validate(sections["preprocessing"]) if sections["preprocessing"] else None
    except ValidationError as exc:
        from rcp_ndcg.support.config import config_error

        section = "preprocessing" if exc.title == Preprocessing.__name__ else "schedule"
        model = Preprocessing if section == "preprocessing" else schedule_type
        raise config_error(
            exc, model=model, source=f"the {stage} settings", overrides=request.set, prefix=section
        ) from exc
    windows = None
    if isinstance(request, TournamentRequest) and request.plan:
        windows, schedule = _planned(request, sections)
    dataset = request.load()
    if windows is not None:
        # The planned documents are the pools: a new document need not be in the dataset's own pools.
        pools = {q: list(dict.fromkeys(doc for window in rows for doc in window)) for q, rows in windows.items()}
    else:
        pools = _pools(request, dataset)
    docs = _docs(request, pools)
    if docs is not None:
        pools = {q: pool for q, pool in pools.items() if q in docs}
    stored = len(JudgementStore(request.out).records(stage)) if Path(request.out).is_dir() else 0
    base = {
        "stage": stage,
        "out": request.out,
        "dataset": dataset.name,
        "queries": len(pools),
        "documents": sum(len(pool) for pool in pools.values()),
        "stored": stored,
    }
    if request.estimate:
        # The real command's refusals first (another identity in --out); writes nothing.
        preflight(
            dataset, pools, config, stage=stage, out=request.out, docs=docs, schedule=schedule,
            preprocessing=preprocessing, force=request.force, windows=windows,
        )  # fmt: skip
        projected = estimate(
            dataset, pools, config, stages=(stage,), schedules={stage: schedule} if schedule else None,
            preprocessing=preprocessing, docs=docs, windows=windows,
        )  # fmt: skip
        return JudgeReport(**base, mode="estimate", estimate=projected)
    client = JudgeClient.from_config(config)
    with mirrored(request.out, request.mirror) if request.mirror else nullcontext():
        judged = judge(
            dataset,
            pools,
            client,
            stage=stage,
            out=request.out,
            docs=docs,
            schedule=schedule,
            preprocessing=preprocessing,
            force=request.force,
            windows=windows,
        )
    return JudgeReport(
        **base,
        mode="judged",
        judgements=len(judged),
        valid=sum(1 for record in judged.judgements if record.valid),
        families=sorted(judged.families),
        usage=client.usage,
    )


def _text(report: JudgeReport) -> str:
    lines = [f"{report.stage} on {report.dataset}: {report.queries} queries, {report.documents} documents"]
    if report.estimate is not None:
        e = report.estimate
        lines.append(f"  {e.calls:,} calls, ~{e.input_tokens:,} input + ~{e.output_tokens:,} output tokens")
        lines.append(f"  {e.assumptions[0]}")
        lines.append(f"  about {e.wall_s:,.0f} s at the judge's concurrency")
    lines.append(f"  store {report.out}: {report.stored} windows of this stage before")
    if report.mode == "judged":
        lines.append(f"  {report.judgements} judgements ({report.valid} valid) for these queries now")
        if report.usage is not None:
            usage = report.usage
            lines.append(
                f"  {usage.requests} requests, {usage.input_tokens:,} input + {usage.output_tokens:,} output tokens"
            )
    return "\n".join(lines)


_JUDGE_OPTIONS = {"result": JudgeReport, "text": _text, "read_only": False}


@command("judge tournament", request=TournamentRequest, **_JUDGE_OPTIONS)
def judge_tournament(request: TournamentRequest) -> JudgeReport:
    """Stage A: judge listwise tournament windows of each query's pool into the judgement store."""
    return _run_stage("tournament", request)


@command("judge rubric", request=JudgeRequest, **_JUDGE_OPTIONS)
def judge_rubric(request: JudgeRequest) -> JudgeReport:
    """Stage B: judge the rubric criteria C1-C5 in windows of each query's pool into the judgement store."""
    return _run_stage("rubric", request)


# ----------------------------------------------------------------------------------------------------------------
# reparse
# ----------------------------------------------------------------------------------------------------------------


class JudgeReparseRequest(BaseModel):
    judgements: str = Field(description="The judgement store whose stored answers to parse again (not modified).")
    out: str = Field(description="The new judgement store to write (a new or empty directory).")


class ReparseStage(BaseModel):
    """What parsing one stage's stored answers again changed."""

    stage: Literal["tournament", "rubric"]
    records: int = Field(description="Window records read and written.")
    recovered: int = Field(description="Invalid before, valid now.")
    still_invalid: int = Field(description="Invalid before and now.")
    unchanged: int = Field(description="Valid before and now, with the same observation.")
    changed: int = Field(description="Valid before, and now invalid or with another observation.")
    invalid_categories: dict[str, int] = Field(description="The still-invalid records by category.")
    family_key: str = Field(description="The new store's family key of this stage.")


class ReparseReport(BaseModel):
    """What ``judge reparse`` wrote: the new store, its parse version and the per-stage counts."""

    source: str
    out: str
    parse_version: int = Field(description="The parser version the new store's records were read with.")
    stages: list[ReparseStage]


def _observation(record: Any) -> tuple:
    return (
        record.ranking,
        tuple((p.unit_id, p.score, tuple(sorted((p.criteria or {}).items()))) for p in record.placements),
    )


def _reparse_stage(stage: Literal["tournament", "rubric"], before: Any, after: Any) -> ReparseStage:
    def key(record: Any) -> tuple:
        return (record.dataset, record.query_id, record.window_seq, tuple(p.unit_id for p in record.placements))

    old = {key(record): record for record in before.judgements if record.stage == stage}
    new = [record for record in after.judgements if record.stage == stage]
    counts = {"recovered": 0, "still_invalid": 0, "unchanged": 0, "changed": 0}
    categories: dict[str, int] = {}
    for record in new:
        previous = old[key(record)]
        if not previous.valid:
            counts["recovered" if record.valid else "still_invalid"] += 1
        elif record.valid and _observation(record) == _observation(previous):
            counts["unchanged"] += 1
        else:
            counts["changed"] += 1
        if not record.valid:
            categories[record.invalid_category] = categories.get(record.invalid_category, 0) + 1
    (family_key,) = {record.family_key for record in new} or {""}
    return ReparseStage(
        stage=stage, records=len(new), invalid_categories=dict(sorted(categories.items())), family_key=family_key,
        **counts,
    )  # fmt: skip


def _reparse_text(report: ReparseReport) -> str:
    lines = [f"reparsed {report.source} -> {report.out} (parse version {report.parse_version})"]
    for row in report.stages:
        lines.append(
            f"  {row.stage}: {row.records} records, {row.recovered} recovered, {row.still_invalid} still invalid "
            f"{row.invalid_categories or ''}, {row.unchanged} unchanged, {row.changed} changed"
        )
    return "\n".join(lines)


@command("judge reparse", request=JudgeReparseRequest, result=ReparseReport, text=_reparse_text, read_only=False)
def judge_reparse(request: JudgeReparseRequest) -> ReparseReport:
    """Parse a store's stored answers again with the current parser into a new store; the judge is not called."""
    from rcp_ndcg.llm import JudgementStore, reparse
    from rcp_ndcg.llm._parsing.common import PARSE_VERSION

    before = JudgementStore(request.judgements).read()
    local_dir(request.out, "a judgement store (--out)")
    after = reparse(request.judgements, request.out)
    stages = [stage for stage in ("tournament", "rubric") if any(r.stage == stage for r in after.judgements)]
    return ReparseReport(
        source=request.judgements,
        out=request.out,
        parse_version=PARSE_VERSION,
        stages=[_reparse_stage(stage, before, after) for stage in stages],  # type: ignore[arg-type]
    )


@click.group(name="judge", help="Judge candidate pools with an LLM: the tournament and the rubric; re-parse a store.")
def judge_group() -> None:
    """``rcp-ndcg judge``."""


for _command in (judge_tournament, judge_rubric, judge_reparse):
    judge_group.add_command(_command)


__all__ = [
    "JudgeReparseRequest",
    "JudgeReport",
    "JudgeRequest",
    "JudgeSource",
    "ReparseReport",
    "ReparseStage",
    "TournamentRequest",
    "judge_group",
]
