"""Tests for the numpy brute-force top-k.

This is the search that runs when no GPU is present, so it has to give exactly
what the GPU path gives, not approximately.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.retrieval import topk
from rcp_ndcg.retrieval.topk import numpy_topk, select_topk


def _random(rows: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(rows, dim)).astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


class TestCorrectness:
    def test_matches_an_exhaustive_argsort(self) -> None:
        docs, queries = _random(200, 16), _random(7, 16, seed=1)

        scores, indices = numpy_topk(docs, queries, k=5)

        expected = np.argsort(-(queries @ docs.T), axis=1, kind="stable")[:, :5]
        assert np.array_equal(indices, expected)
        assert np.allclose(scores, np.take_along_axis(queries @ docs.T, expected, axis=1), atol=1e-6)

    def test_scores_are_descending(self) -> None:
        scores, _ = numpy_topk(_random(50, 8), _random(3, 8, seed=2), k=10)

        assert np.all(np.diff(scores, axis=1) <= 1e-6)

    def test_blocking_does_not_change_the_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Streaming documents is a memory strategy, not a different algorithm."""
        docs, queries = _random(97, 12), _random(5, 12, seed=3)

        whole = numpy_topk(docs, queries, k=8)
        monkeypatch.setattr(topk, "_TILE_BYTES", 1)  # blocks of k documents
        blocked = numpy_topk(docs, queries, k=8)

        assert np.array_equal(whole[1], blocked[1])
        assert np.allclose(whole[0], blocked[0])

    def test_ties_break_toward_the_lower_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Identical documents must not reorder between runs or platforms, within a block or across blocks."""
        docs = np.ones((4, 3), dtype=np.float32)
        queries = np.ones((1, 3), dtype=np.float32)
        monkeypatch.setattr(topk, "_TILE_BYTES", 1)

        _, indices = numpy_topk(docs, queries, k=3)

        assert indices.tolist() == [[0, 1, 2]]

    @pytest.mark.parametrize("tied", [2, 3, 5, 8])
    def test_a_tie_class_straddling_the_cut_breaks_toward_the_lower_index(self, tied: int) -> None:
        """One winner plus a tie class competing for the remaining slots: ``argpartition``'s pick among the
        candidates tied at the k-th score is arbitrary (the sweep's repro dropped index 1 while keeping 2 and
        3), so the cut itself must resolve the tie by index, as the documented rule says."""
        docs = np.array([[9.0, 0.0]] + [[5.0, 0.0]] * tied, dtype=np.float32)
        queries = np.array([[1.0, 0.0]], dtype=np.float32)

        _, indices = numpy_topk(docs, queries, k=3)

        assert indices[0].tolist() == [0, 1, 2]

    def test_a_tie_class_across_blocks_breaks_toward_the_lower_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        docs = np.array([[9.0, 0.0], [8.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0], [5.0, 0.0]], dtype=np.float32)
        queries = np.array([[1.0, 0.0]], dtype=np.float32)
        monkeypatch.setattr(topk, "_TILE_BYTES", 1)  # one document per block

        _, indices = numpy_topk(docs, queries, k=5)

        assert indices[0].tolist() == [0, 1, 2, 3, 4]

    def test_select_topk_resolves_a_straddling_tie_by_index(self) -> None:
        scores = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
        indices = np.array([[0, 1, 2]], dtype=np.int64)

        kept_scores, kept_indices = select_topk(scores, indices, 2)

        assert kept_indices[0].tolist() == [0, 1], "three tied candidates for two slots keep the lowest indices"
        np.testing.assert_allclose(kept_scores[0], [5.0, 5.0])


_IDENTICAL_DOCS_SCRIPT = """
import json
import numpy as np
from rcp_ndcg.retrieval.topk import numpy_topk

rng = np.random.default_rng(0)
docs = np.repeat(rng.normal(size=(1, 768)).astype(np.float32), 15, axis=0)
query = rng.normal(size=(1, 768)).astype(np.float32)
scores, indices = numpy_topk(docs, query, 3)
print(json.dumps({"distinct": len(set(scores[0].tolist())), "indices": indices[0].tolist()}))
"""


class TestTheSelectedSetIsAFunctionOfTheInputsOnly:
    """V1/A1: the float32 GEMM's result for a column depends on its position in the tile, on the BLAS thread
    count and on the query-block width (``topk.py`` derives the block from ``num_queries``), so the selected
    *set* moved with the host. The answer must be a function of the inputs alone: the GEMM pre-selects with a
    margin, the candidates are rescored exactly, and the documented tie rule decides the order and the cut.
    """

    @pytest.mark.parametrize("threads", ["1", "8"])
    def test_identical_documents_get_one_score_and_the_tie_rule(self, threads: str) -> None:
        """Fifteen byte-identical documents, dim 768: one distinct score, and the three lowest indices."""
        result = subprocess.run(
            [sys.executable, "-c", _IDENTICAL_DOCS_SCRIPT],
            env={**os.environ, "OMP_NUM_THREADS": threads},
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        answer = json.loads(result.stdout)
        assert answer["distinct"] == 1, "identical documents must not receive different scores"
        assert answer["indices"] == [0, 1, 2], "ties break toward the lower index"

    def test_the_query_block_width_does_not_move_the_answer(self) -> None:
        """One query alone and the same query inside a 1000-query call score the same document block: the
        answer (set and order) must be identical, since the block is a memory strategy, not the algorithm."""
        rng = np.random.default_rng(3)
        base = rng.normal(size=(1, 128)).astype(np.float32)
        docs = base + rng.normal(scale=1e-7, size=(6000, 128)).astype(np.float32)
        query = rng.normal(size=(1, 128)).astype(np.float32)

        alone = numpy_topk(docs, query, 150)[1][0]
        within = numpy_topk(docs, np.repeat(query, 1000, axis=0), 150)[1][0]

        assert alone.tolist() == within.tolist(), "the query-block width changed the returned top-150"

    def test_the_tile_size_does_not_move_the_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A near-duplicate corpus scored in one block and in one-document blocks returns the same set."""
        rng = np.random.default_rng(11)
        base = rng.normal(size=(1, 64)).astype(np.float32)
        docs = base + rng.normal(scale=1e-7, size=(400, 64)).astype(np.float32)
        query = rng.normal(size=(1, 64)).astype(np.float32)

        whole = numpy_topk(docs, query, 20)
        monkeypatch.setattr(topk, "_TILE_BYTES", 1)
        blocked = numpy_topk(docs, query, 20)

        assert whole[1].tolist() == blocked[1].tolist()
        assert whole[0].tolist() == blocked[0].tolist()

    def test_a_huge_orthogonal_document_does_not_hide_the_true_winner(self) -> None:
        """The verifier's A1 attack: a document with a huge magnitude, orthogonal to the query, carries a
        float32 GEMM score far above the true top score, so it sets the running threshold. The margin must be
        computed with the largest document norm *seen* (the threshold's own setter), not the current block's,
        or a later block's true winner falls outside it. 100 000 queries force the block size that carries a
        threshold across blocks."""
        dim, magnitude = 16, float(2**30)
        rng = np.random.default_rng(8)
        q = rng.normal(size=dim).astype(np.float32)
        u = rng.normal(size=dim)
        u = u - (u @ q.astype(np.float64)) / (q.astype(np.float64) @ q.astype(np.float64)) * q.astype(np.float64)
        docs = np.zeros((82, dim), dtype=np.float32)
        docs[0] = (magnitude * u / np.linalg.norm(u)).astype(np.float32)
        docs[41:] = (q / np.linalg.norm(q)).astype(np.float32)
        exact = q.astype(np.float64) @ docs.astype(np.float64).T
        expected = int(np.argmax(exact))

        alone = numpy_topk(docs, q[None, :], 1)[1][0, 0]
        within = numpy_topk(docs, np.repeat(q[None, :], 100_000, axis=0), 1)[1][0, 0]

        assert alone == expected, "the single-query call is right"
        assert within == expected, "the wide call drops the true top-1"

    def test_an_overflowing_norm_or_score_does_not_drop_the_true_winner(self) -> None:
        """The round-2 attack: finite float32 inputs whose norm (and score) overflow. The margin is then
        infinite -- or the threshold is -- and the float32 comparison is unusable, so every document of the
        block must be exact-rescored. Before the fix the call returned the ``-1`` placeholder with a ``-inf``
        score while the exact float64 top-1 was a finite-scoring document."""
        for dim, magnitude in ((768, 3e18), (1024, 3e18), (32, 3e38)):
            half = dim // 2
            q = np.empty(dim, dtype=np.float32)
            q[:half] = magnitude
            q[half:] = -magnitude
            outlier = np.empty(dim, dtype=np.float32)
            outlier[:half] = magnitude
            outlier[half:] = magnitude
            winner = np.zeros(dim, dtype=np.float32)
            winner[0] = np.float32(100.0 / magnitude)
            docs = np.zeros((8, dim), dtype=np.float32)
            docs[3] = outlier
            docs[5] = winner
            exact = np.einsum("d,nd->n", q.astype(np.float64), docs.astype(np.float64), optimize=False)
            expected = int(np.argmax(exact))

            scores, indices = numpy_topk(docs, q[None, :], 1)

            assert indices[0, 0] == expected, (dim, magnitude)
            assert np.isfinite(scores[0, 0])

    def test_an_infinite_gemm_pair_does_not_drop_the_true_winner(self) -> None:
        """Round-3 attack: a finite corpus where one GEMM pair overflows to ``-inf`` (or NaN) while the
        threshold and cutoff stay finite. The ``-inf >= cutoff`` comparison silently dropped the document,
        although its exact float64 score is the true maximum -- and whether it is -inf or NaN depends on the
        accumulation order, i.e. on the BLAS kernel, which is exactly the host-dependence A1 removes."""
        for dim, neg_at in ((768, 7), (768, 383), (1024, 767)):
            q = np.full(dim, np.float32(3.4e38), dtype=np.float32)
            winner = np.full(dim, np.float32(0.002), dtype=np.float32)
            winner[neg_at] = np.float32(-1.1)
            docs = np.zeros((4, dim), dtype=np.float32)
            docs[2] = winner
            exact = np.einsum("d,nd->n", q.astype(np.float64), docs.astype(np.float64), optimize=False)
            expected = int(np.argmax(exact))
            assert expected == 2 and np.isfinite(exact[2])

            scores, indices = numpy_topk(docs, q[None, :], 1)

            assert indices[0, 0] == 2, (dim, neg_at)
            assert np.isfinite(scores[0, 0]), (dim, neg_at)

    def test_a_norm_that_underflows_to_zero_does_not_collapse_the_margin(self, monkeypatch) -> None:
        """The round-4 attack: a float32 norm can underflow to exactly 0.0 (finite, so the non-finite
        recompute never ran), collapsing the margin to 0 and leaving the raw float32 GEMM order -- wrong for a
        near-tie, and dependent on the tile size. The verifier's shape: dim 4096, a 1e38 query (whose own norm
        overflows), two documents of random +-1e-25 tuned to a ~1e7 gap on a ~6e13 score."""
        dim = 4096
        rng = np.random.default_rng(1)
        q = np.full(dim, np.float32(1e38), dtype=np.float32)
        a = (rng.choice([-1.0, 1.0], size=dim) * np.float32(1e-25)).astype(np.float32)
        b = (rng.choice([-1.0, 1.0], size=dim) * np.float32(1e-25)).astype(np.float32)

        def exact(row: np.ndarray) -> float:
            return float(np.einsum("j,j->", row.astype(np.float64), q.astype(np.float64), optimize=False))

        b[0] = np.float32(b[0] + np.float32((exact(a) - 1e7 - exact(b)) / float(q[0])))
        docs = np.stack([a, b])

        assert exact(docs[0]) > exact(docs[1]), "the exact top-1 is document 0"
        assert np.linalg.norm(docs, axis=1).tolist() == [0.0, 0.0], "both float32 norms underflow to 0"

        whole = numpy_topk(docs, q[None, :], 1)
        monkeypatch.setattr(topk, "_TILE_BYTES", 1)  # one document per block
        blocked = numpy_topk(docs, q[None, :], 1)

        assert whole[1][0, 0] == 0, "the exact order, not the GEMM's rounded one"
        assert blocked[1][0, 0] == 0, "the answer must not depend on the tile size"

    def test_the_answer_is_the_exact_float64_top_k(self) -> None:
        """The rescoring is the documented inner product, not the GEMM's rounded one: the returned set and
        order equal a float64 reference ranked by (score descending, index ascending)."""
        docs, queries = _random(300, 32), _random(5, 32, seed=7)

        scores, indices = numpy_topk(docs, queries, 10)

        exact = queries.astype(np.float64) @ docs.astype(np.float64).T
        expected = np.lexsort((np.broadcast_to(np.arange(docs.shape[0]), exact.shape), -exact), axis=1)[:, :10]
        assert indices.tolist() == expected.tolist()
        assert scores.tolist() == np.take_along_axis(exact, expected, axis=1).astype(np.float32).tolist()

    def test_an_oversized_result_is_refused_with_the_depth_hint(self) -> None:
        """The result matrix is the caller's ``num_queries x k``: over the ceiling it is refused, not
        allocated (a 100k-query, depth-10k search asks for 12 GiB of scores and indices)."""
        with pytest.raises(ConfigError, match="GiB of scores and indices") as caught:
            numpy_topk(_random(10_000, 2), _random(100_000, 2, seed=8), k=10_000)

        assert "depth" in (caught.value.hint or "")


class TestEdgeCases:
    def test_k_larger_than_the_corpus_is_clamped(self) -> None:
        scores, indices = numpy_topk(_random(3, 4), _random(2, 4, seed=4), k=10)

        assert scores.shape == (2, 3)
        assert indices.shape == (2, 3)

    def test_no_queries_returns_empty(self) -> None:
        scores, indices = numpy_topk(_random(5, 4), np.zeros((0, 4), dtype=np.float32), k=3)

        assert scores.shape == (0, 3)
        assert indices.shape == (0, 3)

    def test_empty_corpus_returns_empty(self) -> None:
        scores, _ = numpy_topk(np.zeros((0, 4), dtype=np.float32), _random(2, 4), k=3)

        assert scores.shape == (2, 0)

    def test_non_positive_k_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="k must be positive"):
            numpy_topk(_random(3, 4), _random(1, 4), k=0)

    def test_mismatched_dimensions_are_rejected(self) -> None:
        with pytest.raises(DataError, match="aligned 2D matrices"):
            numpy_topk(_random(3, 4), _random(1, 8), k=1)
