"""Unit tests for :mod:`rcp_ndcg.retrieval.fusion`."""

from __future__ import annotations

import math

import pytest
from rcp_ndcg_core.records import RankingExample

from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.retrieval.fusion import reciprocal_rank_fusion


def _ex(qid: str, doc_ids: list[str], qrels: dict[str, int] | None = None) -> RankingExample:
    return RankingExample(
        query_id=qid,
        query=f"query {qid}",
        doc_ids=doc_ids,
        scores=[1.0 - i * 0.01 for i in range(len(doc_ids))],
        qrels=qrels,
    )


def test_rrf_formula_two_rankers() -> None:
    """Fused score must equal the textbook RRF: sum of 1/(k+rank)."""
    a = [_ex("q", ["d1", "d2", "d3"])]
    b = [_ex("q", ["d2", "d1", "d4"])]
    fused = reciprocal_rank_fusion([a, b], top_k=4, rrf_k=60)
    assert len(fused) == 1

    expected = {
        "d1": 1 / 61 + 1 / 62,
        "d2": 1 / 62 + 1 / 61,
        "d3": 1 / 63,
        "d4": 1 / 63,
    }
    assert set(fused[0].doc_ids) == {"d1", "d2", "d3", "d4"}

    fused_scores = dict(zip(fused[0].doc_ids, fused[0].scores, strict=True))
    for doc, want in expected.items():
        assert math.isclose(fused_scores[doc], want, rel_tol=1e-9), f"score mismatch for {doc}"


def test_rrf_deterministic_tie_break_first_seen() -> None:
    """Ties (equal RRF scores) break by earliest appearance for determinism."""
    a = [_ex("q", ["d3", "d4"])]
    b = [_ex("q", ["d4", "d3"])]
    fused = reciprocal_rank_fusion([a, b], top_k=2, rrf_k=60)
    # Both docs accumulate 1/61 + 1/62.  d3 appears first in the first ranker
    # -> wins the tie-break.
    assert fused[0].doc_ids == ["d3", "d4"]


def test_rrf_top_k_truncates() -> None:
    a = [_ex("q", [f"d{i}" for i in range(10)])]
    b = [_ex("q", [f"d{i}" for i in reversed(range(10))])]
    fused = reciprocal_rank_fusion([a, b], top_k=3, rrf_k=60)
    assert len(fused[0].doc_ids) == 3
    # Highest fused score should be the docs that rank well in *both* lists,
    # i.e. the middle indices.  Sanity-check by verifying scores descend.
    assert fused[0].scores == sorted(fused[0].scores, reverse=True)


def test_rrf_of_one_ranker_keeps_its_order_with_rrf_scores() -> None:
    a = [_ex("q", [f"d{i}" for i in range(10)])]
    fused = reciprocal_rank_fusion([a], top_k=3, rrf_k=60)
    assert fused[0].doc_ids == ["d0", "d1", "d2"]
    assert fused[0].scores == [1 / 61, 1 / 62, 1 / 63]


def test_rrf_propagates_qrels_query_instruction() -> None:
    qrels = {"d1": 2, "d2": 0}
    a = [
        RankingExample(
            query_id="q",
            query="full text",
            instruction="Find a thing.",
            qrels=qrels,
            doc_ids=["d1"],
            scores=[1.0],
        )
    ]
    b = [
        RankingExample(
            query_id="q",
            query="full text",
            instruction="Find a thing.",
            qrels=qrels,
            doc_ids=["d2"],
            scores=[1.0],
        )
    ]
    fused = reciprocal_rank_fusion([a, b], top_k=2, rrf_k=60)
    assert fused[0].qrels == qrels
    assert fused[0].text == "full text"
    assert fused[0].instruction == "Find a thing."
    # docs is intentionally None to keep fused JSONLs small.
    assert fused[0].docs is None


def test_rrf_preserves_query_order_from_first_input() -> None:
    a = [_ex("q1", ["d1"]), _ex("q2", ["d2"]), _ex("q3", ["d3"])]
    # b has the same queries but in a different order.
    b = [_ex("q2", ["d2"]), _ex("q3", ["d3"]), _ex("q1", ["d1"])]
    fused = reciprocal_rank_fusion([a, b], top_k=1, rrf_k=60)
    assert [ex.id for ex in fused] == ["q1", "q2", "q3"]


def test_rrf_rejects_disagreeing_query_coverage() -> None:
    a = [_ex("q1", ["d1"]), _ex("q2", ["d2"])]
    b = [_ex("q1", ["d1"])]
    with pytest.raises(DataError, match="disagree"):
        reciprocal_rank_fusion([a, b], top_k=1, rrf_k=60)


def test_rrf_rejects_invalid_args() -> None:
    a = [_ex("q", ["d1"])]
    with pytest.raises(DataError, match="at least one"):
        reciprocal_rank_fusion([], top_k=1, rrf_k=60)
    with pytest.raises(ConfigError, match="top_k"):
        reciprocal_rank_fusion([a], top_k=0, rrf_k=60)
    with pytest.raises(ConfigError, match="rrf_k"):
        reciprocal_rank_fusion([a], top_k=1, rrf_k=0)
