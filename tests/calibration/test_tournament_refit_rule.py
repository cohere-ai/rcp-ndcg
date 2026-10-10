"""The calibration's Bradley-Terry refit is the fit of every valid window's soft pairs at the window's weight.

A valid tournament window scores every document it showed; each pair of it enters the fit with the sigmoid of the
score gap as its soft label, at the weight ``2 / w`` of a window of ``w`` documents. An invalid window enters with
nothing. The refit must give exactly the fit of those observations.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from rcp_ndcg_core.irt import fit_bradley_terry
from rcp_ndcg_core.schemas import Judgement, JudgementFamily, JudgementSet, Placement

from rcp_ndcg.calibration._projection import bradley_terry, namespace

L2 = 1e-4
FAMILY = JudgementFamily(stage="tournament", judge_model="m", prompt_hash="0" * 64, parse_version=2)
RECORDED_AT = datetime(2026, 1, 1, tzinfo=UTC)
DOCS = ("a", "b", "c", "d")

#: Fully scored windows: (documents in prompt order, their scores in logits).
SCORED = [
    (("a", "b", "c", "d"), (2.0, 1.0, 0.0, -1.0)),
    (("d", "c", "b", "a"), (-0.5, 0.2, 0.8, 1.5)),
    (("b", "d", "a", "c"), (0.9, -0.8, 1.7, 0.1)),
    (("c", "a", "d", "b"), (0.3, 2.2, -1.2, 1.1)),
    (("c", "a", "b"), (0.4, 1.9, -0.3)),
]


def _record(seq: int, docs, scores, *, valid: bool = True) -> Judgement:
    return Judgement(
        record_id=f"r{seq}",
        dataset="ds",
        query_id="q",
        stage="tournament",
        family_key=FAMILY.key,
        window_seq=seq,
        placements=tuple(
            Placement(position=i, doc_id=doc, score=score)
            for i, (doc, score) in enumerate(zip(docs, scores, strict=True), 1)
        ),
        valid=valid,
        invalid_reason=None if valid else "garbled",
        invalid_category=None if valid else "invalid_json",
        recorded_at=RECORDED_AT,
    )


def _records() -> list[Judgement]:
    records = [_record(seq, docs, scores) for seq, (docs, scores) in enumerate(SCORED)]
    records.append(_record(len(records), ("d", "a"), (9.0, -9.0), valid=False))
    return records


def _soft(docs, scores, weight: float) -> list[tuple[str, str, float, float]]:
    """All pairs: winner first, sigmoid of the score gap clipped to [0.01, 0.99]."""
    import math

    scored = list(zip(docs, scores, strict=True))
    pairs = []
    for i, (x, sx) in enumerate(scored):
        for y, sy in scored[i + 1 :]:
            winner, loser, gap = (x, y, sx - sy) if sx >= sy else (y, x, sy - sx)
            pairs.append((winner, loser, weight, min(0.99, max(0.01, 1 / (1 + math.exp(-gap))))))
    return pairs


def _refit(records: list[Judgement]) -> dict[str, float]:
    thetas, _ = bradley_terry(JudgementSet(judgements=tuple(records), families={FAMILY.key: FAMILY}), l2=L2)
    return thetas[namespace("ds", "q")]


def test_the_refit_is_the_fit_of_every_valid_windows_pairs_at_the_windows_weight() -> None:
    observations = [pair for docs, scores in SCORED for pair in _soft(docs, scores, 2 / len(docs))]
    expected = fit_bradley_terry(observations, l2=L2, doc_ids=DOCS)
    assert _refit(_records()) == pytest.approx(expected, abs=1e-4)
