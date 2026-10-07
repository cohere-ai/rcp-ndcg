"""Shared fixtures: the tiny judged world (built once) and hand-made rubric judgements."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest
from rcp_ndcg_core.schemas import (
    Family,
    Judgement,
    JudgementSet,
    Placement,
    criterion_labels,
    judgement_record_id,
)

from rcp_ndcg.calibration import Calibration, read_judgements
from rcp_ndcg.testing import TinyWorld, build_tiny_world

RUBRIC_FAMILY = Family(
    stage="rubric", judge_model="hand", prompt_hash="0" * 64, criteria=criterion_labels(5), parse_version=1
)


def rubric_set(
    windows: dict[tuple[str, str], Sequence[Sequence[tuple[str, str | None, Sequence[int]]]]],
    family: Family = RUBRIC_FAMILY,
) -> JudgementSet:
    """Rubric judgements from ``{(dataset, query_id): [window, ...]}``.

    A window is ``[(doc_id, chunk_id, verdicts), ...]`` in prompt order.
    """
    recorded_at = datetime.now(UTC)
    judgements = []
    for (dataset, query_id), rows in windows.items():
        for seq, window in enumerate(rows):
            placements = tuple(
                Placement(
                    position=i + 1,
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    criteria=dict(zip(family.criteria, map(int, verdicts), strict=True)),
                )
                for i, (doc_id, chunk_id, verdicts) in enumerate(window)
            )
            ids = [p.unit_id for p in placements]
            judgements.append(
                Judgement(
                    record_id=judgement_record_id(family.key, query_id, "rubric", seq, ids, dataset=dataset),
                    dataset=dataset,
                    query_id=query_id,
                    stage="rubric",
                    family_key=family.key,
                    window_seq=seq,
                    placements=placements,
                    recorded_at=recorded_at,
                )
            )
    return JudgementSet(judgements=tuple(judgements), families={family.key: family})


@pytest.fixture(scope="session")
def world(tmp_path_factory: pytest.TempPathFactory) -> TinyWorld:
    return build_tiny_world(tmp_path_factory.mktemp("tiny-world"))


@pytest.fixture(scope="session")
def fitted(world: TinyWorld) -> Calibration:
    """The tiny world's saved calibration (tournament mode, one judge)."""
    return Calibration.load(world.calibration)


@pytest.fixture(scope="session")
def judgements(world: TinyWorld) -> JudgementSet:
    """Both stages of the tiny world's store, the re-judged subset included."""
    return read_judgements(world.judgements)
