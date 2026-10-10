"""The public records of RCP-nDCG: item and query parameters, judgements, families.

* :class:`ItemParams` -- the 2PL criterion parameters ``gamma`` (discrimination,
  dimensionless, > 0) and ``beta`` (difficulty, logits), one entry per rubric
  criterion ``C1..CK``;
* :class:`QueryParams` -- a query's affine map from the tournament's
  Bradley-Terry scale onto the calibrated scale, ``theta = tau * theta_BT + alpha``
  (``tau`` dimensionless and positive, ``alpha`` in logits);
* :class:`Judgement` -- one judge call's parsed observation: an ordered window of
  :class:`Placement` s, with window scores (tournament) or criterion verdicts
  (rubric), and when it was answered;
* :class:`Family` -- the poolability token: two judgements may enter one fit only
  when their families' :attr:`Family.key` match (same stage, judge, prompt,
  criteria, parse version, decoding and preprocessing);
* :class:`JudgementSet` -- judgements plus the families they reference.

Every record is an immutable pydantic model; JSON uses the field names shown here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rcp_ndcg_core._hashing import hash_payload, short
from rcp_ndcg_core.records import DocumentTitle

#: Schema id carried by every judgement record (``"schema"`` in JSON).
JUDGEMENT_SCHEMA = "rcp-ndcg.judgement.v1"

#: The two judging stages: Stage A (listwise tournament) and Stage B (rubric criteria C1..C5).
Stage = Literal["tournament", "rubric"]

#: The phase of a query's schedule a window belongs to: balanced random windows, windows stratified by a
#: preliminary ability, or the tournament's adaptive windows at the top of the ranking.
Phase = Literal["random", "stratified", "adaptive"]

#: Why a judgement is invalid, from a closed set:
#:
#: * ``truncated`` -- the answer stopped at the token limit (``finish_reason`` ``length``) and does not parse;
#: * ``no_json`` -- the answer holds no JSON object;
#: * ``invalid_json`` -- the first JSON object in the answer does not decode;
#: * ``schema`` -- the object lacks a required key, has a value of the wrong type, names an unknown or duplicate
#:   document, or repeats a key;
#: * ``incomplete`` -- the object is well formed but leaves documents of the window out;
#: * ``refused`` -- the endpoint rejected the request, so there is no answer;
#: * ``superseded`` -- a resumed pass refitted under an answer the window's first fit was missing, and this
#:   record's window (a later-phase window of the first fit) is no longer part of the query's generation. A
#:   tombstone: it is appended, never replaces a record in place, and it wins over the older record it names.
InvalidCategory = Literal["truncated", "no_json", "invalid_json", "schema", "incomplete", "refused", "superseded"]

#: How the judge was asked to answer: constrained to the stage's JSON schema (``response_format`` ``json_schema``),
#: or free text that the parser reads.
Decoding = Literal["json_schema", "free"]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


def criterion_labels(num_criteria: int) -> tuple[str, ...]:
    """The criterion labels ``("C1", ..., "C<num_criteria>")``."""
    return tuple(f"C{k + 1}" for k in range(num_criteria))


class ItemParams(BaseModel):
    """2PL criterion parameters: ``P(C_k = 1 | theta) = sigmoid(gamma_k * (theta - beta_k))``.

    Attributes:
        gamma: Discrimination per criterion (dimensionless, > 0).
        beta: Difficulty per criterion (logits).
    """

    model_config = _FROZEN

    gamma: tuple[float, ...]
    beta: tuple[float, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(self.gamma) != len(self.beta) or not self.gamma:
            raise ValueError(
                f"gamma and beta must be equal-length and non-empty, got {len(self.gamma)} and {len(self.beta)}"
            )
        if not all(math.isfinite(g) and g > 0 for g in self.gamma):
            raise ValueError(f"every discrimination gamma_k must be finite and positive, got {list(self.gamma)}")
        if not all(math.isfinite(b) for b in self.beta):
            raise ValueError(f"every difficulty beta_k must be finite, got {list(self.beta)}")
        return self

    @property
    def num_criteria(self) -> int:
        """The number of criteria ``K`` (``len(gamma)``)."""
        return len(self.gamma)

    @property
    def criteria(self) -> tuple[str, ...]:
        """The criterion labels ``C1..CK``."""
        return criterion_labels(self.num_criteria)


class QueryParams(BaseModel):
    """A query's map onto the calibrated scale: ``theta = tau * theta_BT + alpha``.

    Attributes:
        tau: Scale (dimensionless, > 0).
        alpha: Offset (logits).
    """

    model_config = _FROZEN

    tau: float = Field(gt=0)
    alpha: float

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not (math.isfinite(self.tau) and math.isfinite(self.alpha)):
            raise ValueError(
                f"tau and alpha must be finite numbers (tau > 0), got tau={self.tau}, alpha={self.alpha}: "
                "a non-finite one calibrates every ability to NaN"
            )
        return self

    def calibrated(self, theta_bt: float) -> float:
        """A Bradley-Terry ability of this query on the calibrated scale (logits)."""
        return self.tau * theta_bt + self.alpha

    def to_tournament(self, theta: float) -> float:
        """Invert :meth:`calibrated`: a calibrated ability back on the query's Bradley-Terry scale (logits)."""
        return (theta - self.alpha) / self.tau

    def calibrated_se(self, se_bt: float) -> float:
        """A Bradley-Terry standard error on the calibrated scale: ``tau * se_bt`` (logits)."""
        return self.tau * se_bt


class EstimateFlags(BaseModel):
    """What to read before comparing a document's estimated ability with others'.

    Attributes:
        degenerate: ``"all_fail"`` or ``"all_pass"`` when the document failed (passed) every
            criterion in every placement: the likelihood has no interior maximum and the
            estimate rests on the prior. ``None`` otherwise.
        low_information: The document's own evidence falls short of the standard-error target.
    """

    model_config = _FROZEN

    degenerate: Literal["all_fail", "all_pass"] | None = None
    low_information: bool = False


class DocumentEstimate(BaseModel):
    """One document's estimated ability.

    Attributes:
        theta: The ability (logits) on the scale of the items or abilities it was estimated against.
        se: Its standard error (logits).
        information: The Fisher information of the document's own evidence at ``theta`` (logits^-2).
        flags: What to read before comparing it with others.
    """

    model_config = _FROZEN

    theta: float
    se: float
    information: float
    flags: EstimateFlags = EstimateFlags()

    @model_validator(mode="after")
    def _check(self) -> Self:
        bad = [name for name in ("theta", "se", "information") if not math.isfinite(getattr(self, name))]
        if bad:
            raise ValueError(f"estimate {bad[0]} must be a finite number, got {getattr(self, bad[0])!r}")
        return self


class Placement(BaseModel):
    """One document shown in one judged window.

    Attributes:
        position: The 1-based prompt slot the document was shown in.
        doc_id: The document.
        chunk_id: The chunk shown, when the document was judged in chunks (``doc_id#k``).
        score: The judge's window score for this placement (tournament; the prompt's
            scale, -5 to 5).
        criteria: The judge's verdicts ``{"C1": 0|1, ..., "CK": 0|1}`` (rubric).
    """

    model_config = _FROZEN

    position: int = Field(ge=1)
    doc_id: str = Field(min_length=1)
    chunk_id: str | None = None
    score: float | None = None
    criteria: dict[str, int] | None = None

    @field_validator("score")
    @classmethod
    def _finite_score(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError(
                f"score must be a finite number (the judge's stated scale, -5 to 5), got {value!r}: "
                "a NaN score would flow into the fits as a valid window and NaN them"
            )
        return value

    @property
    def unit_id(self) -> str:
        """The id the judge saw: the chunk id when chunked, else the document id."""
        return self.chunk_id or self.doc_id


class Family(BaseModel):
    """The poolability token of a set of judgements.

    Two judgements measure the same thing -- and may enter one fit -- only when
    every field matches: the stage, the judge (model, checkpoint and tokenizer), the prompt,
    the criteria, the parser, the decoding and the preprocessing that shaped
    what the judge saw. :attr:`key` digests them; :attr:`rubric_key` leaves out the judge, which
    is the precondition for pooling several judges with a severity term.

    Attributes:
        stage: ``"tournament"`` or ``"rubric"``.
        judge_model: The served model name (``"fake"`` for the offline fake judge).
        judge_revision: The judge config's ``revision`` as configured (not resolved against the Hub), or ``None``.
        prompt_hash: SHA-256 of the prompt template text.
        criteria: The rubric's criterion labels (``C1..C5``); empty for the tournament.
        parse_version: Version of the parser that reads the answers.
        decoding: ``"json_schema"`` when the endpoint was made to answer in the stage's JSON
            schema, ``"free"`` when it answered in free text (:data:`Decoding`).
        preprocessing: Digest of the effective preprocessing policy (text caps,
            chunking, image and video policies), or ``None`` when nothing was declared.
        tokenizer: SHA-256 of the judge's ``tokenizer.json``, in whose tokens the text limits and the window
            budget were counted, or ``None`` when the judge names no tokenizer. It is the judge's, so
            :attr:`rubric_key` leaves it out with the model.
        temperature: The judge's sampling temperature, when it declared one (``None``: the server's default
            applies, and the field stays out of the digest).
        max_output_tokens: The judge's completion-token cap per request, when declared.
        context_tokens: The judge's prompt-plus-completion budget (the window text budget's basis), when
            declared.
        extra_body: The judge's extra request fields, when any are declared.
        api: The judge's wire adapter name, when the pass was given one other than its default wire (the
            judging pass normalizes the default wire's own name away, so families judged on the default keep
            their key whatever its spelling).
        title: How the documents' titles reached the judge (``separate``: the title as its own leading text
            part); ``None`` is the default join rule (``(title + " " + body).strip()``), so a config that
            declares nothing keeps the key it had. The rule decides the strings the judge reads, so a pass
            that reads title-joined documents never pools with one that reads body-only.
        text_formatting: The text-formatting rule's version
            (:data:`~rcp_ndcg_core.records.TEXT_FORMATTING_VERSION`), when the pass declared it: a changed
            rule shapes the strings the judge reads, so a resume across the version re-asks instead of
            reusing judgements built from the old strings.
        fake_seed: The offline fake judge's draw seed, when the pass ran one: two seeds answer differently,
            so they are different instruments.
    """

    model_config = _FROZEN

    stage: Stage
    judge_model: str
    judge_revision: str | None = None
    prompt_hash: str
    criteria: tuple[str, ...] = ()
    parse_version: int
    decoding: Decoding = "free"
    preprocessing: str | None = None
    tokenizer: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    context_tokens: int | None = None
    extra_body: dict[str, Any] | None = None
    api: str | None = None
    title: DocumentTitle | None = None
    text_formatting: str | None = None
    fake_seed: int | None = None

    @property
    def num_criteria(self) -> int:
        """The number of rubric criteria (0 for the tournament)."""
        return len(self.criteria)

    @property
    def key(self) -> str:
        """16-hex digest of every field: judgements sharing it may be fitted together.

        ``tokenizer`` and the judge's optional settings (``temperature``, ``max_output_tokens``,
        ``context_tokens``, ``extra_body``, ``api``) enter the digest only when they are set: a family judged
        under the defaults digests exactly as one that predates the fields, and a family judged under a declared
        value never pools with it. The document-reading fields (``title``, ``text_formatting``) and the fake
        judge's ``fake_seed`` follow the same rule.
        """
        unset = {
            name
            for name, value in (
                ("tokenizer", self.tokenizer),
                ("temperature", self.temperature),
                ("max_output_tokens", self.max_output_tokens),
                ("context_tokens", self.context_tokens),
                ("extra_body", self.extra_body),
                ("api", self.api),
                ("title", self.title),
                ("text_formatting", self.text_formatting),
                ("fake_seed", self.fake_seed),
            )
            if value is None or (name == "extra_body" and not value)
        }
        return short(hash_payload(self.model_dump(mode="json", exclude=unset)), 16)

    @property
    def rubric_key(self) -> str:
        """16-hex digest of the family without the judge: the instrument several judges can share.

        The document-reading fields (``title``, ``text_formatting``) stay in: they shape the strings the judge
        reads, so two rubrics built from different strings are not one instrument. The fake judge's seed leaves
        with the judge, and an unset optional field stays out of the digest exactly as :attr:`key` leaves it.
        """
        judge = {
            "judge_model",
            "judge_revision",
            "tokenizer",
            "temperature",
            "max_output_tokens",
            "context_tokens",
            "extra_body",
            "api",
            "fake_seed",
        }
        unset = {
            name for name, value in (("title", self.title), ("text_formatting", self.text_formatting)) if value is None
        }
        return short(hash_payload(self.model_dump(mode="json", exclude=judge | unset)), 16)


def judgement_record_id(
    family_key: str,
    query_id: str,
    stage: Stage,
    window_seq: int | None,
    placement_ids: Sequence[str],
    *,
    dataset: str,
    schedule_key: str | None = None,
) -> str:
    """The append-only store's key of one window.

    A window of the schedule is keyed ``H(family_key, query_id, stage, dataset, window_seq, placement ids)``.
    A planned window (``window_seq`` ``None``: asked outside the schedule's phases, e.g. an insertion plan) is
    keyed by its content alone, ``H(family_key, query_id, stage, dataset, schedule_key, placement ids)``, so the
    same window maps to one record however a command groups it.

    Args:
        family_key: :attr:`Family.key`.
        query_id: The query.
        stage: The stage.
        window_seq: The window's index in the query's schedule, or ``None`` for a planned window.
        placement_ids: The ids the judge saw, in prompt order (chunk ids when chunked).
        dataset: The dataset's identity key: a digest naming the dataset and, when it has one, its revision
            (a store identity's ``dataset`` entry), so two corpora that share query and document ids never
            share a record id.
        schedule_key: The digest of the schedule a planned window was asked under; required when ``window_seq``
            is ``None``.

    Raises:
        ValueError: a planned window without its ``schedule_key``.
    """
    if window_seq is None:
        if schedule_key is None:
            raise ValueError("a planned window (window_seq None) is keyed by its schedule_key, and none was given")
        payload: dict[str, object] = {
            "family_key": family_key,
            "query_id": query_id,
            "stage": stage,
            "dataset": dataset,
            "planned": schedule_key,
            "placements": list(placement_ids),
        }
    else:
        payload = {
            "family_key": family_key,
            "query_id": query_id,
            "stage": stage,
            "dataset": dataset,
            "window_seq": window_seq,
            "placements": list(placement_ids),
        }
    return short(hash_payload(payload), 32)


class Judgement(BaseModel):
    """One judge call's parsed observation (schema ``rcp-ndcg.judgement.v1``).

    A valid rubric judgement carries every family criterion, 0 or 1, on every
    placement; a valid tournament judgement carries a score on every placement
    (and the judge's stated ``ranking``). An answer that could not be parsed is recorded with
    ``valid=False``, the reason and its category, never dropped and never read as a vote.

    Attributes:
        record_id: :func:`judgement_record_id` of the window.
        dataset: The dataset the query belongs to.
        query_id: The query (as in the dataset, not namespaced).
        stage: ``"tournament"`` or ``"rubric"``.
        family_key: :attr:`Family.key` of the instrument that produced it.
        window_seq: The window's index in the query's schedule (0-based); ``None`` for a planned window (asked
            outside the schedule's phases, e.g. an insertion plan: ``judge(windows=...)``).
        phase: The schedule phase the window belongs to (:data:`Phase`); ``None`` for a planned window.
        placements: The documents shown, in prompt order.
        ranking: The judge's stated order as 1-based positions (tournament).
        response: The judge's raw answer text.
        finish_reason: Why the endpoint stopped generating (``"stop"``, ``"length"``, ...), when it said.
        valid: Whether the answer parsed into a complete observation.
        invalid_reason: Why it did not, when ``valid`` is false.
        invalid_category: The class of that reason (:data:`InvalidCategory`), when ``valid`` is false.
        input_tokens: Prompt tokens the call used, when the endpoint reported them.
        output_tokens: Completion tokens the call used, when the endpoint reported them.
        recorded_at: When the judge answered (timezone-aware). Who answered, with which prompt and parser,
            is the record's :class:`Family` (``JudgementSet.families[judgement.family_key]``).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_by_name=True, serialize_by_alias=True)

    schema_name: Literal["rcp-ndcg.judgement.v1"] = Field(default=JUDGEMENT_SCHEMA, alias="schema")
    record_id: str
    dataset: str
    query_id: str
    stage: Stage
    family_key: str
    window_seq: int | None = Field(ge=0)
    phase: Phase | None = None
    placements: tuple[Placement, ...]
    ranking: tuple[int, ...] | None = None
    response: str | None = None
    finish_reason: str | None = None
    valid: bool = True
    invalid_reason: str | None = None
    invalid_category: InvalidCategory | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    recorded_at: datetime

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.recorded_at.tzinfo is None or self.recorded_at.tzinfo.utcoffset(self.recorded_at) is None:
            raise ValueError(
                "recorded_at must be timezone-aware (the store and supersedes order windows by it "
                "across hosts; a naive datetime cannot be compared with an aware one)"
            )
        if not self.valid:
            if not self.invalid_reason or self.invalid_category is None:
                raise ValueError("an invalid judgement must say why (invalid_reason and invalid_category)")
            return self
        if self.invalid_reason is not None or self.invalid_category is not None:
            raise ValueError("a valid judgement carries no invalid_reason or invalid_category")
        positions = [placement.position for placement in self.placements]
        if len(set(positions)) != len(positions):
            raise ValueError(f"placement positions must be distinct, got {positions}")
        if self.stage == "rubric":
            missing = [p.position for p in self.placements if not p.criteria]
            if missing or not self.placements:
                raise ValueError(f"a valid rubric judgement needs criteria on every placement (missing at {missing})")
            for placement in self.placements:
                if placement.score is not None:
                    raise ValueError(
                        f"placement {placement.position} of a rubric judgement carries a score {placement.score!r}: "
                        "a rubric placement's verdicts are its criteria; a parser emitting both shapes is a bug"
                    )
                bad = {c: v for c, v in (placement.criteria or {}).items() if v not in (0, 1)}
                if bad:
                    raise ValueError(f"criterion verdicts must be 0 or 1, got {bad}")
        elif not self.placements or any(p.score is None for p in self.placements):
            raise ValueError("a valid tournament judgement needs a score on every placement")
        elif any(p.criteria is not None for p in self.placements):
            raise ValueError(
                "a tournament judgement's placements carry scores, not rubric criteria: a parser emitting "
                "both shapes is a bug"
            )
        if self.ranking is not None and sorted(self.ranking) != sorted(positions):
            raise ValueError(f"ranking {list(self.ranking)} is not a permutation of the positions {positions}")
        return self


class JudgementSet(BaseModel):
    """Judgements together with the families they reference.

    Attributes:
        judgements: The records, in store order.
        families: ``{family_key: Family}`` for every key a judgement uses.
    """

    model_config = _FROZEN

    judgements: tuple[Judgement, ...] = ()
    families: dict[str, Family] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        for judgement in self.judgements:
            family = self.families.get(judgement.family_key)
            if family is None:
                raise ValueError(f"judgement {judgement.record_id} references unknown family {judgement.family_key}")
            if family.stage != judgement.stage:
                raise ValueError(
                    f"judgement {judgement.record_id} is a {judgement.stage} judgement of a {family.stage} family"
                )
        for key, family in self.families.items():
            if family.key != key:
                raise ValueError(f"family registered under {key} hashes to {family.key}")
        return self

    def __len__(self) -> int:
        return len(self.judgements)

    def of_stage(self, stage: Stage) -> JudgementSet:
        """The judgements of one stage (and their families)."""
        return self.select(lambda judgement: judgement.stage == stage)

    def select(self, keep: Callable[[Judgement], bool]) -> JudgementSet:
        """The judgements for which ``keep(judgement)`` is true."""
        kept = tuple(judgement for judgement in self.judgements if keep(judgement))
        used = {judgement.family_key for judgement in kept}
        return JudgementSet(judgements=kept, families={k: f for k, f in self.families.items() if k in used})

    def query_ids(self, stage: Stage | None = None) -> list[tuple[str, str]]:
        """Sorted ``(dataset, query_id)`` pairs with at least one judgement (of ``stage``, if given)."""
        return sorted({(j.dataset, j.query_id) for j in self.judgements if stage is None or j.stage == stage})

    @classmethod
    def merge(cls, sets: Iterable[JudgementSet]) -> JudgementSet:
        """One set holding each window of ``sets`` once: per ``record_id``, its latest valid judgement.

        A window judged again (the same ``record_id``: the same family, query, position and documents) is one
        observation, not two. Its latest valid judgement wins (by ``recorded_at``; a later set wins a tie); an
        invalid judgement wins only when the window has no valid one. Windows keep the order they first appear in.
        """
        chosen: dict[str, Judgement] = {}
        families: dict[str, Family] = {}
        for judgement_set in sets:
            families.update(judgement_set.families)
            for judgement in judgement_set.judgements:
                present = chosen.get(judgement.record_id)
                if present is None or supersedes(judgement, present):
                    chosen[judgement.record_id] = judgement
        return cls(judgements=tuple(chosen.values()), families=families)


def supersedes(new: Judgement, old: Judgement) -> bool:
    """Whether ``new`` replaces ``old`` as the judgement of their window (the same ``record_id``).

    The one rule for a window judged more than once, which :meth:`JudgementSet.merge` and the judgement store
    apply: a valid judgement beats an invalid one, then the later ``recorded_at`` wins (``new`` on a tie). A
    ``superseded`` tombstone is the exception: it is an appended marker whose whole purpose is to retire an
    older record, so time decides against any record one side of which is a tombstone (a tombstone beats an
    older valid record, and a later valid record beats an older tombstone).
    """
    if new.invalid_category == "superseded" or old.invalid_category == "superseded":
        return new.recorded_at >= old.recorded_at
    if new.valid != old.valid:
        return new.valid
    return new.recorded_at >= old.recorded_at


def items_from_mapping(params: Mapping[str, Sequence[float]]) -> ItemParams:
    """:class:`ItemParams` from a ``{"gamma": [...], "beta": [...]}`` mapping (extra keys are ignored)."""
    return ItemParams(gamma=tuple(float(g) for g in params["gamma"]), beta=tuple(float(b) for b in params["beta"]))


__all__ = [
    "JUDGEMENT_SCHEMA",
    "Decoding",
    "DocumentEstimate",
    "EstimateFlags",
    "InvalidCategory",
    "Phase",
    "Family",
    "ItemParams",
    "Judgement",
    "JudgementSet",
    "Placement",
    "QueryParams",
    "Stage",
    "criterion_labels",
    "items_from_mapping",
    "judgement_record_id",
    "supersedes",
]
