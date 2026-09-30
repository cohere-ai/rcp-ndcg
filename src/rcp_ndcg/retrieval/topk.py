"""Brute-force inner-product top-k on numpy: the one flat search of dense retrieval.

Exact flat search with no index structure and no optional dependency: the answer
an exact inner-product index gives, from a matrix multiply, on every platform.
Ties break by score descending, then by the lower row; :func:`rcp_ndcg.retrieval.search`
lays an index's rows out in document-id order, so a tie goes to the lower document id.

Documents are streamed in blocks so peak memory is bounded by the block size
rather than by ``num_queries x num_docs``: a 100k-query, 1M-document score
matrix is 400 GB in float32, and materialising it is the only thing that would
make this approach unusable at the sizes people actually run.

Single-vector only, by construction: late interaction is a different scoring
function, not a different array shape, and lives in
:mod:`rcp_ndcg.retrieval.maxsim`. :func:`score_topk` dispatches between the two
on the layout so callers do not have to.
"""

from __future__ import annotations

import numpy as np

from rcp_ndcg.retrieval.encoder import Embeddings

#: Score-tile budget in bytes.  16 MiB is small enough to stay in cache-friendly
#: territory and large enough that the per-block overhead disappears.
_TILE_BYTES = 16 << 20


def numpy_topk(
    doc_embs: np.ndarray,
    query_embs: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``k`` highest inner products per query.

    Args:
        doc_embs: ``(num_docs, dim)`` float array.
        query_embs: ``(num_queries, dim)`` float array.
        k: Neighbours per query.  Clamped to ``num_docs``.

    Returns:
        ``(scores, indices)``, both ``(num_queries, min(k, num_docs))``, sorted
        by score descending.  Ties break toward the lower document index, so
        results are stable across runs and platforms.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    docs = np.ascontiguousarray(doc_embs, dtype=np.float32)
    queries = np.ascontiguousarray(query_embs, dtype=np.float32)
    if docs.ndim != 2 or queries.ndim != 2 or docs.shape[1] != queries.shape[1]:
        raise ValueError(
            f"embeddings must be aligned 2D matrices, got {docs.shape} and {queries.shape}. "
            "Multi-vector (late-interaction) embeddings are scored by "
            "rcp_ndcg.retrieval.maxsim.maxsim_topk, or by score_topk which dispatches on layout."
        )

    num_docs = docs.shape[0]
    num_queries = queries.shape[0]
    kk = min(k, num_docs)
    if num_queries == 0 or kk == 0:
        return (
            np.zeros((num_queries, kk), dtype=np.float32),
            np.zeros((num_queries, kk), dtype=np.int64),
        )

    block_size = max(1, min(max(kk, _TILE_BYTES // max(num_queries * 4, 1)), num_docs))

    best_scores = np.full((num_queries, kk), -np.inf, dtype=np.float32)
    best_indices = np.full((num_queries, kk), -1, dtype=np.int64)

    for start in range(0, num_docs, block_size):
        stop = min(start + block_size, num_docs)
        block_scores = queries @ docs[start:stop].T
        block_indices = np.arange(start, stop, dtype=np.int64)

        candidate_scores = np.concatenate([best_scores, block_scores], axis=1)
        candidate_indices = np.concatenate(
            [best_indices, np.broadcast_to(block_indices, (num_queries, stop - start))],
            axis=1,
        )
        best_scores, best_indices = select_topk(candidate_scores, candidate_indices, kk)

    return best_scores, best_indices


def score_topk(
    doc_embeddings: Embeddings,
    query_embeddings: Embeddings,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` per query, using whichever scorer the embedding layout implies.

    The dispatch is on the layout rather than on a flag because the layout is the
    ground truth: a ragged buffer *is* a late-interaction index, and scoring it
    with an inner product would require pooling it first, which is a different
    retrieval method that should be requested explicitly.
    """
    if doc_embeddings.is_multi_vector or query_embeddings.is_multi_vector:
        from rcp_ndcg.retrieval.maxsim import maxsim_topk

        return maxsim_topk(doc_embeddings, query_embeddings, k)
    return numpy_topk(doc_embeddings.as_matrix(), query_embeddings.as_matrix(), k)


def select_topk(scores: np.ndarray, indices: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` of each row, ordered by descending score then ascending index.

    Shared with the MaxSim scorer: the tie-break is what makes a run reproducible
    across machines, and having two copies of it would eventually mean two
    different orders for the same scores.
    """
    if scores.shape[1] <= k:
        chosen = np.broadcast_to(np.arange(scores.shape[1]), scores.shape)
    else:
        chosen = np.argpartition(-scores, k - 1, axis=1)[:, :k]

    rows = np.arange(scores.shape[0])[:, None]
    part_scores = scores[rows, chosen]
    part_indices = indices[rows, chosen]

    # lexsort's last key is primary: score descending, index ascending as the
    # tie-break so two identical scores always come back in the same order.
    order = np.lexsort((part_indices, -part_scores), axis=1)
    return (
        np.take_along_axis(part_scores, order, axis=1).astype(np.float32),
        np.take_along_axis(part_indices, order, axis=1).astype(np.int64),
    )


__all__ = ["numpy_topk", "score_topk", "select_topk"]
