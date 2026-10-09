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
The vectors are accepted as float16 or float32 and stay in their stored dtype
between blocks; each query block and each document block is upcast to float32
as it is scored, so the dot products and the per-query sum accumulate in
float32 without ever materialising a float32 copy of the corpus. The peak
working set is one query block (budgeted at ``_QUERY_BLOCK_TOKENS`` rows by the
mean token counts), one document block (its float32 copy at most ``_TILE_BYTES``
-- 64 MiB), and one score tile bounded by ``_TILE_BYTES`` for near-uniform token
counts, larger in proportion to the skew within a block: a float16 token
vector costs 2 bytes stored and, transiently, 4 more per block; a float32 one,
4. Upcasting blockwise is exact (every float16 value is a
float32 value), so a float32 corpus is scored bit for bit as before and a
float16 corpus gains only what its storage precision costs.

The scoring here is exact. An approximate late-interaction index (PLAID-style
centroid pruning) is a different thing with different recall, and would belong
beside this rather than inside it.
"""

from __future__ import annotations

import numpy as np

from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.inference.types import Embeddings

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


def _f32_block(vectors: np.ndarray) -> np.ndarray:
    """One block's float32 view: a copy for float16 vectors (at most the tile budget of bytes), the same
    array for float32 ones. The only float32 materialisation the scorer does, so the peak working set stays
    bounded by the block, never by the corpus."""
    return np.ascontiguousarray(vectors, dtype=np.float32)


def _grouped_max(scores: np.ndarray, starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Max over each item's columns: ``(rows, num_items)``.

    ``np.maximum.reduceat`` returns ``scores[:, start]`` verbatim for a zero-length group rather than the
    identity, and raises when a trailing empty item puts its start one past the last column -- so the
    reduction runs over the non-empty items' starts only (an empty group between two real ones is bounded by
    the next real start, and the last real group runs to the array end, which is exactly its end), and the
    empty items' columns are zeroed here and given their real score (:data:`_EMPTY_DOC_SCORE`) once, after
    the sum -- summing infinities per query token overflows instead. Without this, a document with no vectors
    would score as whatever document happened to follow it.
    """
    if lengths.size == 0:
        return np.zeros((scores.shape[0], 0), dtype=scores.dtype)
    keep = np.flatnonzero(lengths)
    if keep.size == lengths.size:
        return np.maximum.reduceat(scores, starts, axis=1)
    out = np.zeros((scores.shape[0], lengths.size), dtype=scores.dtype)
    out[:, keep] = np.maximum.reduceat(scores, starts[keep], axis=1)
    return out


def _grouped_sum(values: np.ndarray, starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Sum over each item's rows: ``(num_items, cols)``. Same empty-group handling as :func:`_grouped_max`:
    the reduction runs over the non-empty items' starts, the empty items' rows come back zero."""
    if lengths.size == 0:
        return np.zeros((0, values.shape[1]), dtype=values.dtype)
    keep = np.flatnonzero(lengths)
    if keep.size == lengths.size:
        return np.add.reduceat(values, starts, axis=0)
    out = np.zeros((lengths.size, values.shape[1]), dtype=values.dtype)
    out[keep, :] = np.add.reduceat(values, starts[keep], axis=0)
    return out


def maxsim_topk(
    doc_embeddings: Embeddings,
    query_embeddings: Embeddings,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` documents per query by MaxSim.

    Accepts float16 or float32 vectors (any float dtype really: the scoring
    upcasts each block to float32, so the accumulation never happens in float16
    -- a float16 sum of two thousand terms loses most of its three decimal
    digits). Results for float32 inputs are the blocked scorer's as before; for
    float16 inputs they sit within float32 accumulation error of a float64
    reference over the same vectors.

    Args:
        doc_embeddings: Multi-vector document embeddings (float16 or float32).
        query_embeddings: Multi-vector query embeddings (float16 or float32).
        k: Documents per query, clamped to the corpus size.

    Returns:
        ``(scores, indices)``, both ``(num_queries, min(k, num_docs))``, scores float32, sorted
        by score descending with ties broken toward the lower document index --
        so a run is reproducible across machines. An empty document scores
        :data:`_EMPTY_DOC_SCORE` (it ranks below every real one), an empty query 0 against every
        real document; every returned index is a document index.

    Raises:
        ConfigError: ``k`` is not positive.
        DataError: Either side is not multi-vector, or the vector widths differ.
    """
    from rcp_ndcg.retrieval.topk import select_topk

    if k <= 0:
        raise ConfigError(f"k must be positive, got {k}", hint="pass the number of documents per query")
    doc_offsets, query_offsets = doc_embeddings.offsets, query_embeddings.offsets
    if doc_offsets is None or query_offsets is None:
        raise DataError(
            "maxsim_topk needs multi-vector embeddings on both sides "
            f"(docs multi_vector={doc_embeddings.is_multi_vector}, "
            f"queries multi_vector={query_embeddings.is_multi_vector}). "
            "Single-vector embeddings score with numpy_topk instead.",
            hint="score_topk dispatches on the layout, single-vector embeddings to numpy_topk",
        )
    if doc_embeddings.num_items and query_embeddings.num_items and doc_embeddings.dim != query_embeddings.dim:
        raise DataError(f"dimension mismatch: docs {doc_embeddings.dim}, queries {query_embeddings.dim}")

    num_docs = doc_embeddings.num_items
    num_queries = query_embeddings.num_items
    kk = min(k, num_docs)
    if num_queries == 0 or kk == 0:
        return np.zeros((num_queries, kk), dtype=np.float32), np.zeros((num_queries, kk), dtype=np.int64)

    doc_starts, doc_lengths = _spans(doc_offsets)
    query_starts, query_lengths = _spans(query_offsets)
    # Kept in the stored dtype (float16 or float32): the float32 working set is
    # what each block below materialises, never the whole corpus.
    docs = doc_embeddings.vectors
    queries = query_embeddings.vectors

    # Block both axes by *token* count, then convert to item counts: items differ
    # wildly in vector count (a 3-token query, a 1030-patch page), so a budget
    # expressed in items would be meaningless.
    mean_query_tokens = max(1, int(np.ceil(queries.shape[0] / num_queries)))
    mean_doc_tokens = max(1, int(np.ceil(docs.shape[0] / num_docs)))
    query_block_items = max(1, min(num_queries, _QUERY_BLOCK_TOKENS // mean_query_tokens or 1))
    budget_tokens = min(_MAX_BLOCK_TOKENS, _TILE_BYTES // max(query_block_items * mean_query_tokens * 4, 1))
    # The per-block float32 upcast of the document vectors is part of the same
    # budget, so a float16 corpus never pays more than the tile for it either.
    upcast_tokens = max(1, _TILE_BYTES // max(int(docs.shape[1]) * 4, 1))
    budget_tokens = min(budget_tokens, upcast_tokens)
    doc_block_items = max(1, min(num_docs, budget_tokens // mean_doc_tokens or 1))

    best_scores = np.zeros((num_queries, kk), dtype=np.float32)
    best_indices = np.zeros((num_queries, kk), dtype=np.int64)

    for q_start in range(0, num_queries, query_block_items):
        q_stop = min(q_start + query_block_items, num_queries)
        q_slice = slice(int(query_offsets[q_start]), int(query_offsets[q_stop]))
        # The float32 the scoring runs in, one query block at a time (budgeted
        # at _QUERY_BLOCK_TOKENS rows by the mean token counts).
        q_block = _f32_block(queries[q_slice])
        # Re-base the query offsets to this block's local token indexing.
        local_query_starts = query_starts[q_start:q_stop] - int(query_offsets[q_start])
        local_query_lengths = query_lengths[q_start:q_stop]

        rows = q_stop - q_start
        block_best_scores = np.full((rows, kk), -np.inf, dtype=np.float32)
        block_best_indices = np.full((rows, kk), -1, dtype=np.int64)

        for d_start in range(0, num_docs, doc_block_items):
            d_stop = min(d_start + doc_block_items, num_docs)
            d_slice = slice(int(doc_offsets[d_start]), int(doc_offsets[d_stop]))

            token_scores = q_block @ _f32_block(docs[d_slice]).T
            block_doc_lengths = doc_lengths[d_start:d_stop]
            local_doc_starts = doc_starts[d_start:d_stop] - int(doc_offsets[d_start])
            # A block with no tokens to multiply (every document in it empty, or every query) still
            # scores: the grouped reductions return zeros for the empty groups, so an empty query
            # scores 0 against every real document and an empty document's column is overwritten with
            # the sentinel below -- the block's items rank with real indices, never with the running
            # -inf/-1 placeholders.
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
