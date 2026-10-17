"""The chunked tournament refit's SE mapping: a document takes its best chunk's standard error."""

from __future__ import annotations

from datetime import UTC, datetime

from rcp_ndcg_core.schemas import Judgement, JudgementFamily, JudgementSet, Placement

from rcp_ndcg.calibration._projection import bradley_terry

FAMILY = JudgementFamily(stage="tournament", judge_model="m", prompt_hash="0" * 64, parse_version=2)
RECORDED_AT = datetime(2026, 1, 1, tzinfo=UTC)


def _window(seq: int, units: tuple[tuple[str, str], ...], scores: tuple[float, ...]) -> Judgement:
    """One window of ``(unit_id, doc_id)`` placements in prompt order."""
    return Judgement(
        record_id=f"r{seq}",
        dataset="d",
        query_id="q",
        stage="tournament",
        family_key=FAMILY.key,
        window_seq=seq,
        placements=tuple(
            Placement(position=position, doc_id=doc_id, chunk_id=chunk or None, score=score)
            for position, ((chunk, doc_id), score) in enumerate(zip(units, scores, strict=True), start=1)
        ),
        ranking=None,
        response="[]",
        recorded_at=RECORDED_AT,
    )


def test_a_window_without_a_comparison_gives_no_ability_and_no_se() -> None:
    """A valid one-placement window carries no pair, so the estimator forms no information matrix: the unit has no
    comparison, so it gets no ability and no standard error -- never the prior with an SE of 0.0 ("certain")."""
    family = FAMILY
    judgements = JudgementSet(judgements=(_window(0, (("a", "a"),), (1.0,)),), families={family.key: family})

    thetas, ses = bradley_terry(judgements, l2=1.0)

    assert thetas.get("d||q", {}) == {}
    assert ses.get("d||q", {}) == {}, "no observation means no standard error, not an SE of zero"


def _invalid(seq: int, units: tuple[tuple[str, str], ...]) -> Judgement:
    """A window whose answer did not parse: its placements carry no score."""
    return Judgement(
        record_id=f"r{seq}",
        dataset="d",
        query_id="q",
        stage="tournament",
        family_key=FAMILY.key,
        window_seq=seq,
        placements=tuple(
            Placement(position=position, doc_id=doc_id, chunk_id=chunk or None)
            for position, (chunk, doc_id) in enumerate(units, start=1)
        ),
        response="{",
        valid=False,
        invalid_reason="the JSON object does not decode",
        invalid_category="invalid_json",
        recorded_at=RECORDED_AT,
    )


def test_a_unit_no_valid_window_compared_gets_no_ability() -> None:
    """A document shown only in windows that did not parse has no evidence: the refit would return its prior (0
    logits, a mid-scale gain once calibrated), so it gets no ability and the calibration leaves it uncalibrated,
    rather than filling it in. A chunk only invalid windows showed does not enter its document's max-pool either."""
    judgements = JudgementSet(
        judgements=(
            _window(0, (("", "a"), ("", "b")), (3.0, -1.0)),
            _invalid(1, (("", "a"), ("", "c"))),
            _window(2, (("d#0", "d"), ("", "b")), (-2.0, 1.0)),
            _invalid(3, (("d#1", "d"), ("", "a"))),
        ),
        families={FAMILY.key: FAMILY},
    )
    thetas, ses = bradley_terry(judgements, l2=1e-4)
    assert set(thetas["d||q"]) == set(ses["d||q"]) == {"a", "b", "d"}
    # The same windows with each chunk as its own document: d is its compared chunk's ability, never the prior of
    # the chunk only an invalid window showed (which the max-pool would pick: d#0 lost its only comparison).
    plain = JudgementSet(
        judgements=(
            _window(0, (("", "a"), ("", "b")), (3.0, -1.0)),
            _invalid(1, (("", "a"), ("", "c"))),
            _window(2, (("", "d0"), ("", "b")), (-2.0, 1.0)),
            _invalid(3, (("", "d1"), ("", "a"))),
        ),
        families={FAMILY.key: FAMILY},
    )
    units = bradley_terry(plain, l2=1e-4)[0]["d||q"]
    assert set(units) == {"a", "b", "d0"}
    assert thetas["d||q"]["d"] == units["d0"]


def test_a_chunked_tournament_refit_writes_the_best_chunks_se() -> None:
    """A document judged in chunks takes its best chunk's ability (max-pooled) and that chunk's standard
    error: the SE mapping is pinned, not just the score -- the document's SE is the winning chunk's."""
    # One chunk of "a" wins everything against "b"; its twin loses everything. The winning chunk's SE is what
    # a dominated unit gets in a two-item fit; the loser's SE (from a different draw) must not surface.
    family = FAMILY
    judgements = JudgementSet(
        judgements=(
            _window(0, (("a#0", "a"), ("b", "b")), (3.0, 1.0)),
            _window(1, (("a#1", "a"), ("b", "b")), (1.0, 3.0)),
            _window(2, (("b", "b"), ("a#0", "a")), (2.0, 4.0)),
        ),
        families={family.key: family},
    )
    thetas, ses = bradley_terry(judgements, l2=1.0)
    theta, se = thetas["d||q"], ses["d||q"]
    # The same windows with each chunk as its own document: the per-chunk SEs the mapping picks from.
    plain = JudgementSet(
        judgements=(
            _window(0, (("a0", "a0"), ("b", "b")), (3.0, 1.0)),
            _window(1, (("a1", "a1"), ("b", "b")), (1.0, 3.0)),
            _window(2, (("b", "b"), ("a0", "a0")), (2.0, 4.0)),
        ),
        families={FAMILY.key: FAMILY},
    )
    chunk_thetas, chunk_ses = bradley_terry(plain, l2=1.0)
    best = max(chunk_thetas["d||q"], key=lambda unit: chunk_thetas["d||q"][unit])
    loser = next(unit for unit in ("a0", "a1") if unit != best)
    assert theta["a"] == chunk_thetas["d||q"][best], "the document's ability is its best chunk's"
    assert se["a"] == chunk_ses["d||q"][best], "the document's SE is its best chunk's"
    assert se["a"] != chunk_ses["d||q"][loser], "the losing chunk's SE is not the document's"
