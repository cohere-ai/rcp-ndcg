"""Exact late-interaction (MaxSim) scoring.

A single-vector model asks "how close are these two points"; a late-interaction
model asks, for each query token, "what is the best thing in this document to
match it against", and sums the answers:

.. math::

    s(Q, D) = \\sum_{i \\in Q} \\max_{j \\in D} q_i \\cdot d_j

That is not expressible as an inner product between two pooled vectors, which is
why it needs its own scorer rather than a different index. On page images the
difference is the point: a page has one region answering the query and a lot of
unrelated content, and pooling it into one vector averages the answer away.

Full scores are ``query_tokens x doc_tokens`` per pair, so both axes are
blocked: peak memory is bounded by the tile budget instead of by the corpus.

The scoring here is exact. An approximate late-interaction index (PLAID-style
centroid pruning) is a different thing with different recall, and would belong
beside this rather than inside it.
"""

from __future__ import annotations

import numpy as np

from rcp_ndcg.retrieval.encoder import Embeddings

#: Score-tile budget in bytes for one ``(query_tokens x doc_tokens)`` block.
#: 64 MiB: large enough that the per-block Python overhead vanishes, small
#: enough to stay well inside a CPU's last-level cache footprint per pass.
_TILE_BYTES = 64 << 20

#: Cap on document tokens per block, so a corpus of very short documents does not
#: end up with a block spanning millions of items and a huge reduceat.
_MAX_BLOCK_TOKENS = 1 << 20

#: Query tokens scored per pass.
_QUERY_BLOCK_TOKENS = 4096

#: Score for a document with no vectors at all: a max over nothing. Finite rather
#: than ``-inf`` so that it is strictly below every real score (which is bounded by
#: the query's token count) without making the running top-k's ``-inf``
#: placeholder its equal -- a tie there survives the partition step and leaves a
#: non-document index in the output.
_EMPTY_DOC_SCORE = -np.finfo(np.float32).max


def _spans(offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(starts, lengths)`` for every item described by *offsets*."""
    starts = offsets[:-1].astype(np.int64, copy=False)
    return starts, np.diff(offsets).astype(np.int64, copy=False)


def _grouped_max(scores: np.ndarray, starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Max over each item's columns: ``(rows, num_items)``.

    ``np.maximum.reduceat`` returns ``scores[:, start]`` verbatim for a
    zero-length group rather than the identity, so empty items are zeroed here
    and given their real score (:data:`_EMPTY_DOC_SCORE`) once, after the sum -- summing infinities
    per query token overflows instead.  Without this, a document with no vectors
    would score as whatever document happened to follow it.
    """
    reduced = np.maximum.reduceat(scores, starts, axis=1)
    empty = lengths == 0
    if empty.any():
        reduced[:, empty] = 0.0
    return reduced


def _grouped_sum(values: np.ndarray, starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Sum over each item's rows: ``(num_items, cols)``. Same reduceat caveat."""
    reduced = np.add.reduceat(values, starts, axis=0)
    empty = lengths == 0
    if empty.any():
        reduced[empty, :] = 0.0
    return reduced


def maxsim_topk(
    doc_embeddings: Embeddings,
    query_embeddings: Embeddings,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` documents per query by MaxSim.

    Args:
        doc_embeddings: Multi-vector document embeddings.
        query_embeddings: Multi-vector query embeddings.
        k: Documents per query, clamped to the corpus size.

    Returns:
        ``(scores, indices)``, both ``(num_queries, min(k, num_docs))``, sorted
        by score descending with ties broken toward the lower document index --
        so a run is reproducible across machines.
    """
    from rcp_ndcg.retrieval.topk import select_topk

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not doc_embeddings.is_multi_vector or not query_embeddings.is_multi_vector:
        raise ValueError(
            "maxsim_topk needs multi-vector embeddings on both sides "
            f"(docs multi_vector={doc_embeddings.is_multi_vector}, "
            f"queries multi_vector={query_embeddings.is_multi_vector}). "
            "Single-vector embeddings score with numpy_topk instead."
        )
    if doc_embeddings.num_items and query_embeddings.num_items and doc_embeddings.dim != query_embeddings.dim:
        raise ValueError(f"dimension mismatch: docs {doc_embeddings.dim}, queries {query_embeddings.dim}")

    num_docs = doc_embeddings.num_items
    num_queries = query_embeddings.num_items
    kk = min(k, num_docs)
    if num_queries == 0 or kk == 0:
        return np.zeros((num_queries, kk), dtype=np.float32), np.zeros((num_queries, kk), dtype=np.int64)

    assert doc_embeddings.offsets is not None and query_embeddings.offsets is not None
    doc_starts, doc_lengths = _spans(doc_embeddings.offsets)
    query_starts, query_lengths = _spans(query_embeddings.offsets)
    docs = np.ascontiguousarray(doc_embeddings.vectors, dtype=np.float32)
    queries = np.ascontiguousarray(query_embeddings.vectors, dtype=np.float32)

    # Block both axes by *token* count, then convert to item counts: items differ
    # wildly in vector count (a 3-token query, a 1030-patch page), so a budget
    # expressed in items would be meaningless.
    mean_query_tokens = max(1, int(np.ceil(queries.shape[0] / num_queries)))
    mean_doc_tokens = max(1, int(np.ceil(docs.shape[0] / num_docs)))
    query_block_items = max(1, min(num_queries, _QUERY_BLOCK_TOKENS // mean_query_tokens or 1))
    budget_tokens = min(_MAX_BLOCK_TOKENS, _TILE_BYTES // max(query_block_items * mean_query_tokens * 4, 1))
    doc_block_items = max(1, min(num_docs, budget_tokens // mean_doc_tokens or 1))

    best_scores = np.zeros((num_queries, kk), dtype=np.float32)
    best_indices = np.zeros((num_queries, kk), dtype=np.int64)

    for q_start in range(0, num_queries, query_block_items):
        q_stop = min(q_start + query_block_items, num_queries)
        q_slice = slice(int(query_embeddings.offsets[q_start]), int(query_embeddings.offsets[q_stop]))
        q_block = queries[q_slice]
        # Re-base the query offsets to this block's local token indexing.
        local_query_starts = query_starts[q_start:q_stop] - int(query_embeddings.offsets[q_start])
        local_query_lengths = query_lengths[q_start:q_stop]

        rows = q_stop - q_start
        block_best_scores = np.full((rows, kk), -np.inf, dtype=np.float32)
        block_best_indices = np.full((rows, kk), -1, dtype=np.int64)

        for d_start in range(0, num_docs, doc_block_items):
            d_stop = min(d_start + doc_block_items, num_docs)
            d_slice = slice(int(doc_embeddings.offsets[d_start]), int(doc_embeddings.offsets[d_stop]))
            if q_block.shape[0] == 0 or d_slice.start == d_slice.stop:
                continue

            token_scores = q_block @ docs[d_slice].T
            block_doc_lengths = doc_lengths[d_start:d_stop]
            local_doc_starts = doc_starts[d_start:d_stop] - int(doc_embeddings.offsets[d_start])
            per_token_best = _grouped_max(token_scores, local_doc_starts, block_doc_lengths)
            pair_scores = _grouped_sum(per_token_best, local_query_starts, local_query_lengths)
            # A max over no vectors matches nothing, so a document with none ranks
            # strictly below every real one instead of tying with a bad match.
            empty_docs = block_doc_lengths == 0
            if empty_docs.any():
                pair_scores[:, empty_docs] = _EMPTY_DOC_SCORE

            candidate_scores = np.concatenate([block_best_scores, pair_scores.astype(np.float32)], axis=1)
            candidate_indices = np.concatenate(
                [
                    block_best_indices,
                    np.broadcast_to(np.arange(d_start, d_stop, dtype=np.int64), (rows, d_stop - d_start)),
                ],
                axis=1,
            )
            block_best_scores, block_best_indices = select_topk(candidate_scores, candidate_indices, kk)

        best_scores[q_start:q_stop] = block_best_scores
        best_indices[q_start:q_stop] = block_best_indices

    return best_scores, best_indices


__all__ = ["maxsim_topk"]
