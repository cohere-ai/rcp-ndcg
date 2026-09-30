"""The judgement store's reading rules: a corrupt record, a torn last line, and several copies of one window."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from rcp_ndcg_core.schemas import Judgement, Placement

from rcp_ndcg.errors import DataError
from rcp_ndcg.llm import JudgementStore

RECORDED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _window(record_id: str, *, valid: bool = True, response: str | None = "answer") -> Judgement:
    return Judgement(
        record_id=record_id,
        dataset="d",
        query_id="q",
        stage="tournament",
        family_key="f",
        window_seq=0,
        placements=(Placement(position=1, doc_id="a", score=1.0), Placement(position=2, doc_id="b", score=0.0)),
        response=response,
        valid=valid,
        invalid_reason=None if valid else "refused",
        invalid_category=None if valid else "refused",
        recorded_at=RECORDED_AT,
    )


def test_a_corrupt_record_inside_the_file_is_refused_by_line(tmp_path: Path) -> None:
    store = JudgementStore(tmp_path)
    store.append(_window("r1"))
    with store.path("tournament").open("a") as handle:
        handle.write("{not a record}\n")
    store.append(_window("r2"))
    with pytest.raises(DataError, match=r"tournament.jsonl:2: not a judgement record"):
        store.records("tournament")


def test_a_torn_last_line_is_ignored_and_cut_before_the_next_append(tmp_path: Path) -> None:
    store = JudgementStore(tmp_path)
    store.append(_window("r1"))
    with store.path("tournament").open("a") as handle:
        handle.write(_window("r2").model_dump_json()[:40])
    assert list(store.records("tournament")) == ["r1"]
    JudgementStore(tmp_path).append(_window("r3"))
    assert list(JudgementStore(tmp_path).records("tournament")) == ["r1", "r3"]


def test_the_latest_valid_copy_of_a_window_wins_and_an_invalid_one_never_replaces_it(tmp_path: Path) -> None:
    store = JudgementStore(tmp_path)
    store.append(_window("r1", valid=False, response=None))
    store.append(_window("r1", response="first answer"))
    store.append(_window("r1", response="second answer"))
    store.append(_window("r1", valid=False, response=None))
    (record,) = store.records("tournament").values()
    assert record.valid and record.response == "second answer"


def test_the_store_and_merge_keep_the_same_copy_of_a_window(tmp_path: Path) -> None:
    """One rule: a later recorded_at wins among valid copies, whatever order the file holds them in."""
    from datetime import timedelta

    from rcp_ndcg_core.schemas import Family, JudgementSet

    store = JudgementStore(tmp_path)
    later = _window("r1", response="later answer").model_copy(update={"recorded_at": RECORDED_AT + timedelta(hours=1)})
    earlier = _window("r1", response="earlier answer")
    store.append(later)
    store.append(earlier)  # written after, recorded before
    (kept,) = store.records("tournament").values()
    family = Family(stage="tournament", judge_model="m", prompt_hash="p", parse_version=1)
    copies = [
        earlier.model_copy(update={"family_key": family.key}),
        later.model_copy(update={"family_key": family.key}),
    ]
    merged = JudgementSet.merge([JudgementSet(judgements=(copy,), families={family.key: family}) for copy in copies])
    assert kept.response == merged.judgements[0].response == "later answer"
