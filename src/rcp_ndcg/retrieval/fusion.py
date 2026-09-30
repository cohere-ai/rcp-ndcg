"""Reciprocal Rank Fusion (RRF) as a pure post-processing step.

RRF combines multiple per-query rankings into a single ranking by summing
``1 / (k + rank)`` across rankers.  Reference:

    Cormack, Clarke & Buettcher.
    "Reciprocal Rank Fusion outperforms Condorcet and individual Rank
    Learning Methods", SIGIR 2009.

This module is intentionally side-effect free and operates exclusively on
:class:`rcp_ndcg_core._records.RankingExample` records -- the canonical
JSONL row shared by every other stage of the pipeline.  It does not know
about retrievers, indices, or HTTP clients; :func:`rcp_ndcg.retrieval.fuse` and
``rcp-ndcg retrieval fuse`` call it on rankings.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from rcp_ndcg_core._records import RankingExample


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RankingExample]],
    *,
    top_k: int,
    rrf_k: int = 60,
) -> list[RankingExample]:
    """Merge multiple per-query rankings using Reciprocal Rank Fusion.

    Args:
        rankings: One ranked list per ranker.  Each inner sequence must be
            aligned by ``query_id`` -- different rankers may interleave their
            output order, but every query must appear in every list.
        top_k: Number of documents to return per query after fusion.
        rrf_k: RRF smoothing constant (default 60, robust across datasets).

    Returns:
        One :class:`RankingExample` per query.  ``doc_ids`` are sorted by
        descending fused score; ``scores`` carries the RRF score (not the
        original retriever scores).  ``qrels``, ``query``, and ``instruction``
        are propagated from the first input ranking that contains the query.
        ``docs`` is left ``None`` -- the LLM rerank stage hydrates documents
        from the corpus, so we don't waste bytes shipping them through.

    Raises:
        ValueError: If the input is empty or the ranker outputs disagree on
            which queries are present.
    """
    if not rankings:
        raise ValueError("reciprocal_rank_fusion needs at least one ranking list.")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}.")
    if rrf_k <= 0:
        raise ValueError(f"rrf_k must be positive, got {rrf_k}.")

    if len(rankings) == 1:
        single = list(rankings[0])
        return [_truncate(ex, top_k) for ex in single]

    # Build per-ranker {query_id: RankingExample} indices and validate that
    # every ranker covers the same queries (set equality, order-agnostic).
    per_ranker: list[dict[str, RankingExample]] = []
    query_id_sets: list[set[str]] = []
    for r in rankings:
        idx = {ex.id: ex for ex in r}
        if len(idx) != len(r):
            duplicate_ids = [ex.id for ex in r if list(idx.values()).count(ex) > 1]
            raise ValueError(f"Duplicate query_ids in ranking input: {duplicate_ids[:5]}")
        per_ranker.append(idx)
        query_id_sets.append(set(idx.keys()))

    common = query_id_sets[0]
    for s in query_id_sets[1:]:
        if s != common:
            missing = (common - s) | (s - common)
            raise ValueError(f"Rankers disagree on query coverage; first 5 mismatched ids: {sorted(missing)[:5]}")

    # Preserve query ordering from the first ranking.
    query_id_order = [ex.id for ex in rankings[0]]

    fused: list[RankingExample] = []
    for qid in query_id_order:
        scores: dict[str, float] = defaultdict(float)
        # We also track the first time we see each doc so ties break
        # deterministically by earliest-appearance.
        first_seen: dict[str, int] = {}
        seq = 0
        for idx in per_ranker:
            ex = idx[qid]
            for rank_pos, doc_id in enumerate(ex.doc_ids, start=1):
                scores[doc_id] += 1.0 / (rrf_k + rank_pos)
                if doc_id not in first_seen:
                    first_seen[doc_id] = seq
                    seq += 1

        # Sort by (-score, first_seen) for determinism.
        sorted_docs = sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))[:top_k]

        head = per_ranker[0][qid]
        fused.append(
            RankingExample(
                query_id=qid,
                query=head.query,
                instruction=head.instruction,
                qrels=head.qrels,
                doc_ids=[d for d, _ in sorted_docs],
                scores=[s for _, s in sorted_docs],
                docs=None,
            )
        )
    return fused


def _truncate(ex: RankingExample, top_k: int) -> RankingExample:
    """Return a top-k copy of a single ranking (used in the 1-ranker fast path)."""
    if len(ex.doc_ids) <= top_k:
        return ex
    return RankingExample(
        query_id=ex.id,
        query=ex.query,
        instruction=ex.instruction,
        qrels=ex.qrels,
        doc_ids=list(ex.doc_ids[:top_k]),
        scores=list(ex.scores[:top_k]) if ex.scores is not None else None,
        docs=list(ex.docs[:top_k]) if ex.docs is not None else None,
    )


__all__ = ["reciprocal_rank_fusion"]
