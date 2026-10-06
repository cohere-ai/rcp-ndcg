"""Tests for the numpy brute-force top-k.

This is the search that runs when no GPU is present, so it has to give exactly
what the GPU path gives, not approximately.
"""

from __future__ import annotations

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
