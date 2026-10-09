"""Brute-force inner-product top-k on numpy: the one flat search of dense retrieval.

Exact flat search with no index structure and no optional dependency: the answer
an exact inner-product index gives, on every platform. The scores are computed
twice: a float32 GEMM pre-selects each block's candidates (fast, but its rounding
depends on the tile position and the BLAS kernel), and every candidate is then
rescored exactly by a deterministic float64 reduction, so the selected set and
its order are a function of the inputs alone -- never of the host's thread count,
the tile size or how many queries travelled together.

Ties break by score descending, then by the lower row; :func:`rcp_ndcg.retrieval.search`
lays an index's rows out in document-id order, so a tie goes to the lower document id.
The same rule orders the rerank depth cut and BM25 (:mod:`rcp_ndcg.retrieval.sparse`),
and it is stated once in ``docs/concepts/retrieval.md``.

Documents are streamed in blocks so the working set stays bounded by the block
rather than by ``num_queries x num_docs``: a 100k-query, 1M-document score
matrix is 400 GB in float32, and materialising it is the only thing that would
make this approach unusable at the sizes people actually run. The bound is a
constant multiple of the block, not the block itself: a block's float32 tile,
the merged candidate arrays (float32 scores and int64 indices), the float64
rescoring buffer and the selected rows are live together, which at ``k=150``
and a 16 MiB tile is a few hundred MiB, not 16 MiB (:data:`_TILE_BYTES`).

Single-vector only, by construction: late interaction is a different scoring
function, not a different array shape, and lives in
:mod:`rcp_ndcg.retrieval.maxsim`. :func:`score_topk` dispatches between the two
on the layout so callers do not have to.
"""

from __future__ import annotations

import numpy as np

from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.inference.types import Embeddings

#: Score-tile budget in bytes.  16 MiB is small enough to stay in cache-friendly
#: territory and large enough that the per-block overhead disappears.  It budgets
#: the float32 score tile; the transient peak is a small multiple of it (see the
#: module docstring).
_TILE_BYTES = 16 << 20

#: The result-size ceiling: the returned scores (float32) and indices (int64) alone.
_MAX_OUTPUT_BYTES = 1 << 30

#: The pre-selection margin's factor.  The float32 GEMM's error on one pair is bounded by
#: ``dim * eps32 * ||q|| * ||d||`` (the standard dot-product bound, ``gamma_2dim ~= 2*dim*eps32``,
#: over ``sum |q_i d_i| <= ||q|| ||d||``); two BLAS kernels (tile positions, thread counts) differ
#: by at most twice that.  The factor below covers both with room for the blocked-accumulation
#: constants, so every document whose exact score belongs in the top ``k`` is inside the margin.
_MARGIN_FACTOR = 8.0 * float(np.finfo(np.float32).eps)


def _refuse_oversized_output(num_queries: int, k: int) -> None:
    """Refuse a top-k whose result alone exceeds :data:`_MAX_OUTPUT_BYTES`.

    The scorer's memory is bounded by its blocks, but the *result* is ``num_queries x k`` and grows without
    limit with ``depth``: at 100 000 queries and ``k=10 000`` it is 12 GiB of scores and indices, allocated
    on top of the block working set.  A typed refusal beats an out-of-memory kill.

    Args:
        num_queries: Queries in the call.
        k: The clamped neighbours per query.

    Raises:
        ConfigError: The result would exceed the ceiling, with the ``depth`` hint.
    """
    needed = num_queries * k * (np.dtype(np.float32).itemsize + np.dtype(np.int64).itemsize)
    if needed > _MAX_OUTPUT_BYTES:
        raise ConfigError(
            f"a top-{k} of {num_queries} queries returns {needed / 2**30:.1f} GiB of scores and indices, "
            f"over the {_MAX_OUTPUT_BYTES // 2**30} GiB ceiling",
            hint="lower depth (documents per query), or search the queries in batches",
        )


def numpy_topk(
    doc_embs: np.ndarray,
    query_embs: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``k`` highest inner products per query.

    A float32 GEMM pre-selects the candidates of every block (the top ``k`` plus every document whose
    score is within the margin that bounds the GEMM's own rounding error), and each candidate is
    rescored exactly in float64 by a deterministic reduction.  The selected set and its order are
    therefore a function of the inputs alone: not of the BLAS thread count, the tile position or the
    query-block width.

    Args:
        doc_embs: ``(num_docs, dim)`` float array.
        query_embs: ``(num_queries, dim)`` float array.
        k: Neighbours per query.  Clamped to ``num_docs``.

    Returns:
        ``(scores, indices)``, both ``(num_queries, min(k, num_docs))``, sorted
        by score descending.  Ties break toward the lower document index, so
        results are stable across runs and platforms.

    Raises:
        ConfigError: ``k`` is not positive, or the result alone exceeds the declared ceiling.
        DataError: The matrices are not aligned 2-D arrays (a 3-D, multi-vector array is
            :func:`rcp_ndcg.retrieval.maxsim.maxsim_topk`'s input).
    """
    if k <= 0:
        raise ConfigError(f"k must be positive, got {k}", hint="pass the number of documents per query")
    docs = np.ascontiguousarray(doc_embs, dtype=np.float32)
    queries = np.ascontiguousarray(query_embs, dtype=np.float32)
    if docs.ndim != 2 or queries.ndim != 2 or docs.shape[1] != queries.shape[1]:
        raise DataError(
            f"embeddings must be aligned 2D matrices, got {docs.shape} and {queries.shape}. "
            "Multi-vector (late-interaction) embeddings are scored by "
            "rcp_ndcg.retrieval.maxsim.maxsim_topk, or by score_topk which dispatches on layout.",
            hint="score_topk dispatches on the layout, multi-vector embeddings to maxsim_topk",
        )

    num_docs = docs.shape[0]
    num_queries = queries.shape[0]
    dim = docs.shape[1]
    kk = min(k, num_docs)
    if num_queries == 0 or kk == 0:
        return (
            np.zeros((num_queries, kk), dtype=np.float32),
            np.zeros((num_queries, kk), dtype=np.int64),
        )
    _refuse_oversized_output(num_queries, kk)

    block_size = max(1, min(max(kk, _TILE_BYTES // max(num_queries * 4, 1)), num_docs))
    query_norms = np.linalg.norm(queries, axis=1)

    # The running float32 top-k is the pre-selection's threshold only (its values, never its order or its
    # tie classes); the returned answer is the exact one below.
    running = np.full((num_queries, kk), -np.inf, dtype=np.float32)
    exact_scores = np.full((num_queries, kk), -np.inf, dtype=np.float64)
    exact_indices = np.full((num_queries, kk), -1, dtype=np.int64)

    for start in range(0, num_docs, block_size):
        stop = min(start + block_size, num_docs)
        block_scores = queries @ docs[start:stop].T  # float32 GEMM: pre-selection only
        merged = np.concatenate([running, block_scores], axis=1)
        running = np.partition(merged, merged.shape[1] - kk, axis=1)[:, merged.shape[1] - kk :]
        threshold = running.min(axis=1)
        doc_norm = float(np.linalg.norm(docs[start:stop], axis=1).max())
        margin = _MARGIN_FACTOR * dim * doc_norm * query_norms
        # Every document the pre-selection cannot exclude: the running threshold only rises, so a document
        # dropped here can never be within the final threshold's margin either.
        rows, cols = np.nonzero(block_scores >= threshold[:, None] - margin[:, None])
        if rows.size == 0:
            continue
        block_exact = np.full(block_scores.shape, -np.inf, dtype=np.float64)
        block_exact[rows, cols] = np.einsum(
            "ij,ij->i",  # one deterministic reduction per pair: no BLAS, no threads, no tile position
            docs[start + cols].astype(np.float64),
            queries[rows].astype(np.float64),
            optimize=False,
        )
        exact_scores, exact_indices = select_topk(
            np.concatenate([exact_scores, block_exact], axis=1),
            np.concatenate(
                [
                    exact_indices,
                    np.broadcast_to(np.arange(start, stop, dtype=np.int64), block_scores.shape),
                ],
                axis=1,
            ),
            kk,
        )

    return exact_scores.astype(np.float32), exact_indices


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
    different orders for the same scores. The tie rule decides the cut as well as
    the order: ``argpartition``'s pick among the candidates tied at the k-th score
    is implementation-defined, so a tie class straddling the cut is re-selected by
    index (:func:`_repair_ties_at_the_cut`) before the final ordering.

    The scores come back in the dtype they went in (float32 for a float32 caller, float64 for the exact
    rescoring of :func:`numpy_topk`), so the caller decides the precision of its own comparison.
    """
    if scores.shape[1] <= k:
        chosen = np.broadcast_to(np.arange(scores.shape[1]), scores.shape)
    else:
        # Both order statistics in one call: a tie class straddles the cut exactly where the k-th and the
        # (k+1)-th scores are equal, so the expensive per-row mask runs only for the rows that can have one
        # (the common tie-free row costs two comparisons here and nothing else).
        part = np.argpartition(-scores, [k - 1, k], axis=1)
        chosen = part[:, :k]
        kth, next_ = (np.take_along_axis(-scores, part[:, index : index + 1], axis=1)[:, 0] for index in (k - 1, k))
        suspects = np.flatnonzero(kth == next_)
        if suspects.size:
            _repair_ties_at_the_cut(scores, indices, chosen, k, suspects)

    rows = np.arange(scores.shape[0])[:, None]
    part_scores = scores[rows, chosen]
    part_indices = indices[rows, chosen]

    # lexsort's last key is primary: score descending, index ascending as the
    # tie-break so two identical scores always come back in the same order.
    order = np.lexsort((part_indices, -part_scores), axis=1)
    return (
        np.take_along_axis(part_scores, order, axis=1),
        np.take_along_axis(part_indices, order, axis=1).astype(np.int64),
    )


def _repair_ties_at_the_cut(
    scores: np.ndarray, indices: np.ndarray, chosen: np.ndarray, k: int, suspects: np.ndarray
) -> None:
    """Give the candidates tied at the k-th score the cut's remaining slots by ascending index, in place.

    ``argpartition`` selects an arbitrary subset of a tie class straddling the k-th score, so the documented
    rule (ties toward the lower index) is restored per affected row, in the selected-positions array itself.
    Only the rows ``suspects`` names are examined (the caller found the k-th and (k+1)-th scores equal there,
    the only rows a tie class can straddle); within them, a row whose threshold class is fully selected keeps
    the partition's answer.

    Args:
        scores: ``(rows, candidates)``, the candidates' scores.
        indices: The candidate indices, aligned with ``scores``.
        chosen: The selected positions per row, from ``argpartition`` (written in place).
        k: The cut; ``scores.shape[1] > k``.
        suspects: The rows whose k-th and (k+1)-th scores are equal (ascending).
    """
    subset = np.arange(suspects.size)[:, None]
    part_scores = scores[suspects[:, None], chosen[suspects]]
    threshold = part_scores.min(axis=1)
    at_threshold = scores[suspects] == threshold[:, None]
    selected_at_threshold = at_threshold[subset, chosen[suspects]].sum(axis=1)
    for row in np.flatnonzero(at_threshold.sum(axis=1) > selected_at_threshold):
        slots = int(selected_at_threshold[row])
        tied = np.flatnonzero(at_threshold[row])
        selected = chosen[suspects[row]]
        keep = tied[np.argsort(indices[suspects[row], tied])[:slots]]  # the tie class's lowest document indices
        picked = selected[at_threshold[row, selected]]
        drop = np.setdiff1d(picked, keep)
        fill = np.setdiff1d(keep, picked)
        positions = np.flatnonzero(np.isin(selected, drop))
        chosen[suspects[row], positions] = fill


__all__ = ["numpy_topk", "score_topk", "select_topk"]
