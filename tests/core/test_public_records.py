"""The public records of ``rcp_ndcg_core.schemas``: validation, keys and round trips."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from rcp_ndcg_core.records import RankingExample
from rcp_ndcg_core.schemas import (
    DocumentEstimate,
    Family,
    ItemParams,
    Judgement,
    JudgementSet,
    Placement,
    QueryParams,
    judgement_record_id,
)

RECORDED_AT = datetime(2026, 1, 1, tzinfo=UTC)
RUBRIC = Family(stage="rubric", judge_model="m", prompt_hash="p" * 64, criteria=("C1", "C2"), parse_version=1)


def _rubric(window_seq: int = 0, **overrides) -> Judgement:
    placements = (
        Placement(position=1, doc_id="a", criteria={"C1": 1, "C2": 0}),
        Placement(position=2, doc_id="b", criteria={"C1": 0, "C2": 0}),
    )
    fields = {
        "record_id": judgement_record_id(RUBRIC.key, "q", "rubric", window_seq, ["a", "b"], dataset="d"),
        "dataset": "d",
        "query_id": "q",
        "stage": "rubric",
        "family_key": RUBRIC.key,
        "window_seq": window_seq,
        "placements": placements,
        "recorded_at": RECORDED_AT,
    }
    return Judgement(**(fields | overrides))


def test_item_params_derive_the_criterion_count_and_refuse_bad_values() -> None:
    items = ItemParams(gamma=(1.0, 2.0, 0.5), beta=(-1.0, 0.0, 2.0))
    assert items.num_criteria == 3
    assert items.criteria == ("C1", "C2", "C3")
    with pytest.raises(ValidationError, match="equal-length"):
        ItemParams(gamma=(1.0,), beta=(0.0, 1.0))
    with pytest.raises(ValidationError, match="positive"):
        ItemParams(gamma=(1.0, -1.0), beta=(0.0, 1.0))


def test_query_params_map_a_bradley_terry_theta_onto_the_calibrated_scale() -> None:
    params = QueryParams(tau=1.7, alpha=-3.2)
    assert params.calibrated(1.5) == pytest.approx(1.7 * 1.5 - 3.2)
    assert params.to_tournament(params.calibrated(0.4)) == pytest.approx(0.4)
    assert params.calibrated_se(0.25) == pytest.approx(0.425)
    with pytest.raises(ValidationError):
        QueryParams(tau=0.0, alpha=0.0)


def test_family_key_covers_every_field_and_rubric_key_leaves_out_the_judge() -> None:
    other_judge = RUBRIC.model_copy(update={"judge_model": "n"})
    other_prompt = RUBRIC.model_copy(update={"prompt_hash": "q" * 64})
    constrained = RUBRIC.model_copy(update={"decoding": "json_schema"})
    assert RUBRIC.key != other_judge.key
    assert RUBRIC.key != constrained.key and RUBRIC.rubric_key != constrained.rubric_key
    assert RUBRIC.rubric_key == other_judge.rubric_key
    assert RUBRIC.rubric_key != other_prompt.rubric_key
    assert len(RUBRIC.key) == 16
    assert RUBRIC.num_criteria == 2


def test_family_key_gains_a_judge_field_only_when_it_differs_from_the_default() -> None:
    """The declared-CONTENT judge settings the Family carries (temperature, output and context budgets,
    extra body, wire adapter) enter the digest only when set: a family judged under a default keeps its key,
    one judged under a declared value never pools with the default's (cross-store there is no gate)."""
    for field, value in (
        ("temperature", 0.7),
        ("max_output_tokens", 8192),
        ("context_tokens", 131072),
        ("extra_body", {"reasoning_effort": "low"}),
        ("api", "cohere"),
    ):
        declared = RUBRIC.model_copy(update={field: value})
        assert declared.key != RUBRIC.key, field
    # A family that does not carry them digests as before, and the tokenizer pattern holds.
    bare = Family(stage="rubric", judge_model="m", prompt_hash="p" * 64, criteria=("C1",), parse_version=1)
    defaulted = Family(
        stage="rubric",
        judge_model="m",
        prompt_hash="p" * 64,
        criteria=("C1",),
        parse_version=1,
        temperature=None,
        max_output_tokens=None,
        context_tokens=None,
        extra_body={},
        api=None,
    )
    assert bare.key == defaulted.key
    # rubric_key leaves the judge out, the new fields with it.
    assert RUBRIC.model_copy(update={"temperature": 0.7}).rubric_key == RUBRIC.rubric_key


def test_the_family_and_record_digests_are_pinned() -> None:
    """Stored judgements are keyed by these digests: a change orphans every store written before it."""
    assert RUBRIC.key == "d8a72042c3706de8"
    assert RUBRIC.rubric_key == "5f8379aca7521add"
    # The record id gained the dataset (the store identity's dataset entry, digested): judgements of two corpora
    # that share query and document ids no longer share a record id. Deliberate (0.0.1: the old ids conflated
    # corpora); it orphans stores written before the change, as the docstring says.
    assert judgement_record_id(RUBRIC.key, "q", "rubric", 0, ["a", "b"], dataset="d") == (
        "37cedfc94161a2e813916ea5006df468"
    )


def test_query_params_refuse_non_finite_values() -> None:
    """``Field(gt=0)`` admits ``inf``; an alpha of NaN calibrated every theta to NaN."""
    with pytest.raises(ValidationError, match="finite"):
        QueryParams(tau=1.0, alpha=float("nan"))
    with pytest.raises(ValidationError, match="finite"):
        QueryParams(tau=float("inf"), alpha=0.0)


def test_a_document_estimate_refuses_non_finite_values() -> None:
    for field in ("theta", "se", "information"):
        values: dict[str, float] = {"theta": 0.0, "se": 1.0, "information": 1.0}
        values[field] = float("inf") if field == "se" else float("nan")
        with pytest.raises(ValidationError, match="finite"):
            DocumentEstimate(**values)


def test_a_placement_score_must_be_finite() -> None:
    """A NaN score made a ``valid=True`` tournament judgement whose NaN NaNs the BT/2PL fits."""
    for score in (float("nan"), float("inf")):
        with pytest.raises(ValidationError, match="finite"):
            Placement(position=1, doc_id="a", score=score)


def test_recorded_at_must_be_timezone_aware() -> None:
    """The store and :func:`supersedes` order windows by ``recorded_at``; a naive datetime
    recorded on one host crashed every comparison with an aware one with a bare TypeError."""
    with pytest.raises(ValidationError, match="timezone-aware"):
        _rubric(recorded_at=datetime(2026, 1, 1))


def test_a_ranking_example_refuses_non_finite_scores() -> None:
    """A NaN comparison is always False, so the descending sort silently degenerated to the
    input order -- for pools, often the relevance order the tie rules warn about."""
    with pytest.raises(ValueError, match="finite"):
        RankingExample(query_id="q1", doc_ids=["a", "b"], scores=[1.0, float("nan")], docs=["A", "B"])


def test_a_valid_rubric_judgement_needs_every_verdict_binary() -> None:
    assert _rubric().valid
    with pytest.raises(ValidationError, match="criteria on every placement"):
        _rubric(placements=(Placement(position=1, doc_id="a"),))
    with pytest.raises(ValidationError, match="0 or 1"):
        _rubric(placements=(Placement(position=1, doc_id="a", criteria={"C1": 2, "C2": 0}),))
    unscored = (Placement(position=1, doc_id="a"),)
    invalid = _rubric(placements=unscored, valid=False, invalid_reason="no JSON", invalid_category="no_json")
    assert not invalid.valid
    with pytest.raises(ValidationError, match="invalid_category"):
        _rubric(valid=False, invalid_reason="no JSON")
    with pytest.raises(ValidationError, match="invalid_category"):
        _rubric(invalid_category="schema")


def test_a_placement_carries_one_shape_not_both() -> None:
    """A parser bug emitting both shapes must not be recorded as a valid observation: a rubric
    placement's verdicts are its criteria, a tournament placement's vote is its score."""
    with pytest.raises(ValidationError, match="score"):
        _rubric(placements=(Placement(position=1, doc_id="a", criteria={"C1": 1, "C2": 0}, score=999.0),))
    family = Family(stage="tournament", judge_model="m", prompt_hash="p" * 64, parse_version=1)
    with pytest.raises(ValidationError, match="criteria"):
        Judgement(
            record_id=judgement_record_id(family.key, "q", "tournament", 0, ["a"], dataset="d"),
            dataset="d",
            query_id="q",
            stage="tournament",
            family_key=family.key,
            window_seq=0,
            placements=(Placement(position=1, doc_id="a", score=1.0, criteria={"C1": 1}),),
            recorded_at=RECORDED_AT,
        )


def test_judgement_json_round_trip_carries_the_schema_id() -> None:
    judgement = _rubric()
    payload = judgement.model_dump_json()
    assert '"schema":"rcp-ndcg.judgement.v1"' in payload
    assert Judgement.model_validate_json(payload) == judgement


def test_record_id_depends_on_window_placements_and_dataset() -> None:
    base = judgement_record_id("f", "q", "rubric", 0, ["a", "b"], dataset="d")
    assert base != judgement_record_id("f", "q", "rubric", 1, ["a", "b"], dataset="d")
    assert base != judgement_record_id("f", "q", "rubric", 0, ["b", "a"], dataset="d")
    assert base == judgement_record_id("f", "q", "rubric", 0, ["a", "b"], dataset="d")


def test_the_record_id_names_the_dataset() -> None:
    """Two corpora that share query ids and document ids never share a record id (cross-store, and in a merge)."""
    base = judgement_record_id("f", "q", "rubric", 0, ["a", "b"], dataset="d")
    assert base != judgement_record_id("f", "q", "rubric", 0, ["a", "b"], dataset="other")


def test_a_planned_record_id_names_the_dataset_too() -> None:
    planned = judgement_record_id("f", "q", "rubric", None, ["a"], dataset="d", schedule_key="s")
    assert planned != judgement_record_id("f", "q", "rubric", None, ["a"], dataset="other", schedule_key="s")
    with pytest.raises(ValueError, match="schedule_key"):
        judgement_record_id("f", "q", "rubric", None, ["a"], dataset="d")


def test_judgement_set_refuses_unknown_families_and_merges_by_record_id() -> None:
    first, second = _rubric(0), _rubric(1)
    with pytest.raises(ValidationError, match="unknown family"):
        JudgementSet(judgements=(first,))
    one = JudgementSet(judgements=(first, second), families={RUBRIC.key: RUBRIC})
    merged = JudgementSet.merge([one, JudgementSet(judgements=(second,), families={RUBRIC.key: RUBRIC})])
    assert len(merged) == 2
    assert merged.query_ids("rubric") == [("d", "q")]
    assert len(merged.of_stage("tournament")) == 0
