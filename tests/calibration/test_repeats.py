"""Repeated judgements: one window counts once (its latest valid judgement), and re-judged documents are refit."""

from __future__ import annotations

from datetime import timedelta

import pytest
from rcp_ndcg_core.schemas import Judgement, JudgementSet, judgement_record_id

from rcp_ndcg.calibration import Calibration, calibrate
from rcp_ndcg.errors import RcpNdcgWarning


def _later(judgement: Judgement, **update) -> Judgement:
    return judgement.model_copy(update={"recorded_at": judgement.recorded_at + timedelta(hours=1), **update})


def _flipped(judgement: Judgement) -> Judgement:
    """The window answered again, every rubric verdict of its first placement flipped."""
    first = judgement.placements[0]
    flipped = first.model_copy(update={"criteria": {c: 1 - v for c, v in (first.criteria or {}).items()}})
    return _later(judgement, placements=(flipped, *judgement.placements[1:]))


def test_the_latest_valid_judgement_of_a_window_wins_whatever_the_order(judgements: JudgementSet) -> None:
    old = next(j for j in judgements.judgements if j.stage == "rubric" and j.valid)
    new = _flipped(old)
    refused = _later(old, valid=False, invalid_reason="no answer", invalid_category="refused")
    for sets in ([[old], [new]], [[new], [old]], [[old], [new], [refused]]):
        merged = JudgementSet.merge(JudgementSet(judgements=tuple(s), families=judgements.families) for s in sets)
        assert merged.judgements == (new,)


def test_a_window_judged_twice_is_counted_once(judgements: JudgementSet) -> None:
    rubric = [j for j in judgements.judgements if j.stage == "rubric"]
    repeated = JudgementSet(judgements=(*judgements.judgements, *rubric[:5]), families=judgements.families)
    with pytest.warns(RcpNdcgWarning, match="rubric verdicts but no tournament ability"):
        once, twice = calibrate(judgements), calibrate(repeated)  # the store's rubric-only documents
    assert twice.coverage.model_dump() == once.coverage.model_dump()
    assert twice.thetas == once.thetas and twice.items == once.items


def test_a_refit_pools_the_new_windows_of_a_re_judged_document(judgements: JudgementSet) -> None:
    # Re-judging a document asks new windows (new record ids): a refit counts them with its earlier ones, so its
    # theta moves. (calibration score, in contrast, leaves a document that has a theta alone.)
    rubric = judgements.of_stage("rubric")
    target = next(j for j in rubric.judgements if j.valid).placements[0]
    passed = {c: 1 for c in target.criteria or {}}
    again = []
    for judgement in rubric.judgements:
        if not judgement.valid or target.doc_id not in {p.doc_id for p in judgement.placements}:
            continue
        seq = judgement.window_seq + 1000
        ids = [p.unit_id for p in judgement.placements]
        placements = tuple(p.model_copy(update={"criteria": passed}) if p.doc_id == target.doc_id else p for p in
                           judgement.placements)  # fmt: skip
        again.append(
            _later(
                judgement,
                window_seq=seq,
                placements=placements,
                record_id=judgement_record_id(
                    judgement.family_key, judgement.query_id, "rubric", seq, ids, dataset=judgement.dataset
                ),
            )
        )
    before = calibrate(rubric, mode="rubric_only")
    after = calibrate(
        JudgementSet(judgements=(*rubric.judgements, *again), families=rubric.families), mode="rubric_only"
    )

    def theta(calibration: Calibration) -> float:
        return next(row.theta for row in calibration.thetas if row.doc_id == target.doc_id)

    assert theta(after) > theta(before)
    assert after.coverage.windows["rubric"].windows > before.coverage.windows["rubric"].windows
