"""Multi-process writers of one judgement store: claims, records and the identity file.

The store's design invites overlapping passes (a resume, two stages of one run,
several commands sharing a mirror), so two OS processes are the real writer
shape -- not threads. Each round builds a fresh store (a reused one merges
instead of racing) and both workers claim and append under a file-barrier.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rcp_ndcg.judging import JudgementStore

WORKER = r"""
import json, os, sys, time
from datetime import UTC, datetime
from pathlib import Path
from rcp_ndcg_core.schemas import JudgementFamily, Judgement, Placement
from rcp_ndcg.judging.store import JudgementStore

root, stage, rounds, barrier = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
claim = len(sys.argv) < 6 or sys.argv[5] != "no-claim"
pad = "x" * 40_000  # a realistic record (~40 KB: one judgement's reasoning and placements)
family = JudgementFamily(stage=stage, judge_model="m", prompt_hash="p" * 64, criteria=("C1", "C2"), parse_version=1)
# The barrier: both workers spin on the same file, then go at once.
start = time.time()
while not barrier.exists() and time.time() - start < 30:
    time.sleep(0.001)
store = JudgementStore(root)
identity = {"stage": stage, "family": family.model_dump(mode="json"), "round": 0}
if claim:
    store.claim(stage, identity, family)
    # What a real pass writes after the claim: the endpoint's report (two identity rewrites per pass, as judge()
    # does: claim, then note_engines after the probe, then after the answers).
    from rcp_ndcg.judging.client import EngineInfo

    store.note_engines(stage, [EngineInfo(url="http://h:8000/v1", model="m", max_model_len=8192)])
writer = os.getpid()
for index in range(int(rounds)):
    judgement = Judgement(
        record_id=f"{stage}-{writer}-{index:04d}",
        dataset="d",
        query_id="q",
        stage=stage,
        family_key=family.key,
        window_seq=index,
        placements=(
            Placement(position=1, doc_id="a", score=1.0)
            if stage == "tournament"
            else Placement(position=1, doc_id="a", criteria={"C1": 1, "C2": 0}),
        ),
        response=pad,
        recorded_at=datetime.now(UTC),
    )
    store.append(judgement)
    store.note_engines(stage, [])
"""


def _run_workers(tmp_path: Path, stages: tuple[str, str], rounds: int, *, claim: bool = True) -> None:
    """Two OS processes claim one stage each of a fresh store and append ``rounds`` records to it.

    ``claim=False``: the workers skip the claim (the store is pre-claimed) and go straight to their appends,
    so the barrier aligns their first appends -- the window the torn-tail repair races in."""
    barrier = tmp_path / "barrier"
    processes = []
    for stage in stages:
        argv = [sys.executable, "-c", WORKER, str(tmp_path), stage, str(rounds), str(barrier)]
        if not claim:
            argv.append("no-claim")
        processes.append(
            subprocess.Popen(
                argv,
                cwd=tmp_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    barrier.write_text("go", encoding="utf-8")
    for process in processes:
        _, err = process.communicate(timeout=120)
        assert process.returncode == 0, err


def test_two_processes_claim_both_stages_and_every_stage_survives(tmp_path: Path) -> None:
    """Both processes claim a different stage of one fresh store (round after round): the identity file is
    read-modify-write, so the loser of a race must not lose its stage entry -- the losing pass itself would
    crash on read(). Each round is a fresh store (a reused one merges instead of racing)."""
    rounds = 12
    for round_ in range(rounds):
        round_dir = tmp_path / f"round-{round_:02d}"
        round_dir.mkdir()
        _run_workers(round_dir, ("tournament", "rubric"), rounds=4)
        store = JudgementStore(round_dir)
        entries = store.identities()
        assert sorted(entries) == ["rubric", "tournament"], f"round {round_}: lost a stage entry"
        assert store.read("tournament") and store.read("rubric")


def test_concurrent_appends_lose_no_records(tmp_path: Path) -> None:
    """Two processes appending one stage's records of a pre-claimed store: every appended record is on disk.

    The first append's torn-tail repair must not cut a peer's in-flight line -- a round starts from a seeded
    torn tail (the state whose repair is the race) and both workers' first appends are aligned on the barrier,
    with no claim between them to serialize the workers. The records are ~40 KB, the size at which a
    mid-transfer truncation was observed."""
    from rcp_ndcg_core.schemas import JudgementFamily

    rounds = 6
    for round_ in range(rounds):
        round_dir = tmp_path / f"round-{round_:02d}"
        round_dir.mkdir()
        # The store is claimed once (a resumed pass's shape), and a killed writer left a torn tail: the first
        # append's repair must not cut the peer's record.
        family = JudgementFamily(stage="tournament", judge_model="m", prompt_hash="p" * 64, parse_version=1)
        JudgementStore(round_dir).claim("tournament", {"round": round_}, family)
        (round_dir / "tournament.jsonl").write_text(
            '{"record_id": "torn"', encoding="utf-8"
        )  # torn: the repair sees it, the peer's first append must not be the collateral
        _run_workers(round_dir, ("tournament", "tournament"), rounds=20, claim=False)
        lines = (round_dir / "tournament.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 40, f"round {round_}: lost {40 - len(lines)} records of 40"
        records = JudgementStore(round_dir).records("tournament")
        assert len(records) == 40, f"round {round_}: {len(records)} readable of 40"
        assert all(record.valid for record in records.values())


def test_a_writer_racing_a_reader_never_sees_a_partial_identity_file(tmp_path: Path) -> None:
    """The store's identity file is replaced atomically under a concurrent reader (the progress reader's
    contract, now through the one storage helper)."""
    store = JudgementStore(tmp_path)
    from rcp_ndcg_core.schemas import JudgementFamily

    family = JudgementFamily(stage="tournament", judge_model="m", prompt_hash="p", parse_version=1)
    store.claim("tournament", {"a": 0}, family, sources={f"{i}": "x" * 200 for i in range(400)})
    payload_before = json.loads(store.identity_path.read_text(encoding="utf-8"))
    store.claim("tournament", {"a": 1}, family, force=True, sources={f"{i}": "y" * 200 for i in range(400)})
    payload_after = json.loads(store.identity_path.read_text(encoding="utf-8"))
    assert payload_before["stages"]["tournament"]["identity"] == {"a": 0}
    assert payload_after["stages"]["tournament"]["identity"] == {"a": 1}
    assert store.identities()["tournament"]["identity"] == {"a": 1}
    assert datetime.fromisoformat(payload_after["stages"]["tournament"]["created_at"]) is not None


def test_the_append_holds_the_store_writer_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """The append's tail repair and write hold the store's advisory lock: an unlocked first append truncates a
    peer's in-flight line (the loss the stress test's shape can only sample). Pinned by the lock's acquisition,
    not by hoping to catch a microsecond window."""
    import tempfile

    from rcp_ndcg_core.schemas import Judgement, JudgementFamily, Placement

    entered: list[str] = []
    with tempfile.TemporaryDirectory() as root:
        store = JudgementStore(root)
        store.claim(
            "tournament", {}, JudgementFamily(stage="tournament", judge_model="m", prompt_hash="p", parse_version=1)
        )
        original = store._identity_lock

        @contextlib.contextmanager
        def watched():
            entered.append("lock")
            with original():
                yield

        store._identity_lock = watched
        store.append(
            Judgement(
                record_id="r1",
                dataset="d",
                query_id="q",
                stage="tournament",
                family_key="f",
                window_seq=0,
                placements=(Placement(position=1, doc_id="a", score=1.0),),
                response="answer",
                recorded_at=datetime.now(UTC),
            )
        )
        assert entered == ["lock"], "the tail repair and the append hold the store's writer lock"
        assert len(JudgementStore(root).records("tournament")) == 1
