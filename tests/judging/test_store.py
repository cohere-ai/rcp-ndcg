"""The judgement store's reading rules: a corrupt record, a torn last line, and several copies of one window."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from rcp_ndcg_core.schemas import Family, Judgement, Placement

from rcp_ndcg.errors import DataError
from rcp_ndcg.judging import JudgementStore

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


def test_the_store_identity_is_replaced_never_rewritten(tmp_path: Path) -> None:
    """`run status` reads a running job's store progress while the job claims its stages: the identity file is
    written through a temp file and renamed (as the run manifest is), so a concurrent reader sees the old or
    the new file, never the truncated moment."""
    store = JudgementStore(tmp_path)
    family = Family(stage="tournament", judge_model="m", prompt_hash="p", parse_version=1)
    store.claim("tournament", {"a": 1}, family)
    first = store.identity_path.stat().st_ino
    store.claim("tournament", {"a": 2}, family, force=True)
    assert store.identity_path.stat().st_ino != first, "a rewrite in place is caught half-written by a reader"


def test_a_claim_racing_a_progress_reader_never_serves_a_partial_identity(tmp_path: Path) -> None:
    """The race the MCP e2e poll lost: `_judge_progress` reads the store's identity while the job's pass
    rewrites it; with a big payload the truncated-in-place window is wide, and the reader must never see it."""
    import threading

    store = JudgementStore(tmp_path)
    family = Family(stage="tournament", judge_model="m", prompt_hash="p", parse_version=1)
    sources = {f"{i}": "x" * 200 for i in range(800)}  # a payload big enough to make the window visible
    store.claim("tournament", {"a": 0}, family, sources=sources)
    stop = threading.Event()
    errors: list[BaseException] = []

    def read() -> None:
        while not stop.is_set():
            try:
                store.identities()
            except Exception as exc:  # noqa: BLE001 -- what run_status would surface to its caller
                errors.append(exc)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    try:
        for i in range(1, 12):
            store.claim("tournament", {"a": i}, family, force=True, sources=sources)
    finally:
        stop.set()
        reader.join()
    assert errors == []


def test_a_corrupt_record_inside_the_file_is_refused_by_line(tmp_path: Path) -> None:
    store = JudgementStore(tmp_path)
    store.append(_window("r1"))
    with store.path("tournament").open("a") as handle:
        handle.write("{not a record}\n")
    store.append(_window("r2"))
    with pytest.raises(DataError, match=r"tournament.jsonl:2: not a judgement record"):
        store.records("tournament")


def test_a_tombstone_retires_a_record_whose_clock_ran_ahead(tmp_path: Path) -> None:
    """The store resolves a tombstone pair by file order, not ``recorded_at``: a host whose clock ran ahead
    cannot leave a retired record valid, and a later valid re-ask of the window wins again."""
    from datetime import timedelta

    from rcp_ndcg.judging.store import records_stored

    store = JudgementStore(tmp_path)
    ahead = _window("r1").model_copy(update={"recorded_at": datetime.now(UTC) + timedelta(days=1)})
    store.append(ahead)
    store.supersede_records("tournament", {"r1": ahead}, reason="a resumed pass refitted")
    (record,) = store.records("tournament").values()
    assert record.invalid_category == "superseded"
    assert records_stored(store.path("tournament")) == 0  # the id has no live record

    store.append(_window("r1", response="new answer"))
    (record,) = store.records("tournament").values()
    assert record.valid and record.response == "new answer"
    assert records_stored(store.path("tournament")) == 1


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


def test_an_empty_stage_file_appends_cleanly(tmp_path: Path) -> None:
    """A zero-byte stage file (a killed writer's repair truncated a lone fragment to nothing) appends: the
    tail probe guards the empty file instead of seeking to -1."""
    store = JudgementStore(tmp_path)
    (tmp_path / "tournament.jsonl").write_text("", encoding="utf-8")
    store.append(_window("r1"))
    assert len(store.records("tournament")) == 1


def test_an_unterminated_but_parseable_last_record_is_torn_not_counted(tmp_path: Path) -> None:
    """A kill that lands after the record's JSON bytes but before its newline leaves a line `records()` can
    parse; the cutters' definition of unfinished governs -- the reader skips it (the window is asked again),
    so the next append can never silently delete a record the pass counted as reused."""
    store = JudgementStore(tmp_path)
    with (tmp_path / "tournament.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(_window("r1").model_dump_json() + "\n")
        handle.write(_window("r2").model_dump_json())  # complete JSON, no trailing newline: a killed write
    records = store.records("tournament")
    assert set(records) == {"r1"}, "the unterminated record is a torn write: absent, asked again"
    store.append(_window("r3"))
    records = store.records("tournament")
    assert set(records) == {"r1", "r3"}, "the cut did not resurrect the lost record, and r3 landed"
    assert (tmp_path / "tournament.jsonl").read_text(encoding="utf-8").endswith("\n")


def test_a_zero_byte_identity_file_is_absent_not_a_crash(tmp_path: Path) -> None:
    """A zero-byte identity.json (the store's own torn write: a supersede copyfile killed mid-write) is
    the torn tail the store already governs: the state is absent, so the claim that reads it rewrites the
    file instead of failing the parse."""
    family = Family(stage="tournament", judge_model="m", prompt_hash="p", parse_version=1)
    store = JudgementStore(tmp_path)
    store.claim("tournament", {"a": 1}, family)
    (tmp_path / "identity.json").write_bytes(b"")
    assert store.identities() == {}, "the empty file is treated as the store's torn tail"
    store.claim("tournament", {"a": 2}, family)
    assert store.identities()["tournament"]["identity"] == {"a": 2}
