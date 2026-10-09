"""The refit is order-canonical: the same windows in any order give bit-identical numbers.

A planned window (``window_seq=None``: an insertion plan or a re-judge) carries no schedule position, so the
refit's own order must not inherit the order the stores happened to hand the windows in. Two stores holding the
same planned windows, read in either order, must give bit-identical Bradley-Terry abilities and standard errors,
item parameters, calibrated abilities and fingerprint.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from rcp_ndcg_core.schemas import Family, Judgement, JudgementSet, Placement

from rcp_ndcg.calibration import Calibration, calibrate, read_judgements
from rcp_ndcg.calibration._projection import bradley_terry
from rcp_ndcg.judging.store import JudgementStore

from .conftest import RUBRIC_FAMILY, rubric_set

TOURNAMENT_FAMILY = Family(stage="tournament", judge_model="hand", prompt_hash="0" * 64, parse_version=2)
RECORDED_AT = datetime(2026, 1, 1, tzinfo=UTC)
DOCS = ("a", "b", "c", "d")

#: One scheduled window (position 0) and two planned ones (no position): the review's repro.
SCHEDULED = (("a", "b", "c", "d"), (2.0, 1.0, 0.0, -1.0))
PLANNED = (
    (("b", "d", "a", "c"), (0.9, -0.8, 1.7, 0.1)),
    (("c", "a", "d", "b"), (0.3, 2.2, -1.2, 1.1)),
)


def _window(record_id: str, seq: int | None, docs: tuple[str, ...], scores: tuple[float, ...]) -> Judgement:
    return Judgement(
        record_id=record_id,
        dataset="d",
        query_id="q",
        stage="tournament",
        family_key=TOURNAMENT_FAMILY.key,
        window_seq=seq,
        placements=tuple(
            Placement(position=i, doc_id=doc, score=score)
            for i, (doc, score) in enumerate(zip(docs, scores, strict=True), start=1)
        ),
        recorded_at=RECORDED_AT,
    )


def _windows() -> tuple[Judgement, Judgement, Judgement]:
    return (
        _window("scheduled", 0, *SCHEDULED),
        _window("planned-a", None, *PLANNED[0]),
        _window("planned-b", None, *PLANNED[1]),
    )


def _store(root: Path, name: str, family: Family, judgements: Sequence[Judgement]) -> Path:
    """A minimal judgement store holding ``judgements``: claimed through the product path, as ``judge`` does."""
    store = JudgementStore(root / name)
    store.root.mkdir(parents=True, exist_ok=True)
    store.claim(family.stage, {}, family)
    for judgement in judgements:
        store.append(judgement)
    return store.root


def _rubric_windows() -> JudgementSet:
    verdicts = ((1, 1, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0), (1, 1, 1, 1, 0))
    return rubric_set(
        {("d", "q"): [[(doc, None, list(v)) for doc, v in zip(DOCS, verdicts, strict=True)] for _ in range(3)]}
    )


def test_two_planned_windows_in_either_store_order_give_bit_identical_numbers(tmp_path: Path) -> None:
    scheduled, planned_a, planned_b = _windows()
    first = _store(tmp_path, "first", TOURNAMENT_FAMILY, [scheduled, planned_a])
    second = _store(tmp_path, "second", TOURNAMENT_FAMILY, [planned_b])
    rubric = _store(tmp_path, "rubric", RUBRIC_FAMILY, _rubric_windows().judgements)

    forward = calibrate(read_judgements(first, second, rubric))
    backward = calibrate(read_judgements(second, first, rubric))

    # read_judgements(second, first) hands the planned windows in the other order; every fitted number, and the
    # identity extensions anchor to, must still match bit for bit.
    assert forward.items == backward.items
    assert forward.thetas == backward.thetas
    assert forward.fingerprint == backward.fingerprint
    assert forward.theta_map() == backward.theta_map()
    assert forward.identity.judgements == backward.identity.judgements


def test_the_projection_alone_is_order_canonical() -> None:
    scheduled, planned_a, planned_b = _windows()
    families = {TOURNAMENT_FAMILY.key: TOURNAMENT_FAMILY}
    forward = JudgementSet(judgements=(scheduled, planned_a, planned_b), families=families)
    backward = JudgementSet(judgements=(scheduled, planned_b, planned_a), families=families)

    forward_thetas, forward_ses = bradley_terry(forward, l2=1e-4)
    backward_thetas, backward_ses = bradley_terry(backward, l2=1e-4)

    assert forward_thetas == backward_thetas
    assert forward_ses == backward_ses


def test_a_scheduled_window_stays_where_its_position_puts_it() -> None:
    """The canonical order sorts planned windows by record id *after* the scheduled ones: the scheduled
    windows' relative order is still their positions, so no anchor moves."""
    scheduled, planned_a, planned_b = _windows()
    later = _window("scheduled-2", 1, *PLANNED[0])
    families = {TOURNAMENT_FAMILY.key: TOURNAMENT_FAMILY}
    set_a = JudgementSet(judgements=(scheduled, later, planned_a), families=families)
    set_b = JudgementSet(judgements=(later, scheduled, planned_a), families=families)
    assert bradley_terry(set_a, l2=1e-4) == bradley_terry(set_b, l2=1e-4)


def test_the_calibration_survives_a_save_and_load_with_the_fingerprint_intact(tmp_path: Path) -> None:
    """The repro end to end, through the layout: the fingerprint an extension anchors to is the file's."""
    scheduled, planned_a, planned_b = _windows()
    first = _store(tmp_path, "first", TOURNAMENT_FAMILY, [scheduled, planned_a])
    second = _store(tmp_path, "second", TOURNAMENT_FAMILY, [planned_b])
    rubric = _store(tmp_path, "rubric", RUBRIC_FAMILY, _rubric_windows().judgements)
    calibration = calibrate(read_judgements(first, second, rubric))
    loaded = Calibration.load(calibration.save(tmp_path / "cal"))
    assert loaded == calibration
    assert json.loads((tmp_path / "cal" / "items.json").read_text())["fingerprint"] == calibration.fingerprint
