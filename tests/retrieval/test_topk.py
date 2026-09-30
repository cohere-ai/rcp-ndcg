"""Tests for the numpy brute-force top-k.

This is the search that runs when no GPU is present, so it has to give exactly
what the GPU path gives, not approximately.
"""

from __future__ import annotations

import numpy as np
import pytest

from rcp_ndcg.retrieval import topk
from rcp_ndcg.retrieval.topk import numpy_topk


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
        with pytest.raises(ValueError, match="k must be positive"):
            numpy_topk(_random(3, 4), _random(1, 4), k=0)

    def test_mismatched_dimensions_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="aligned 2D matrices"):
            numpy_topk(_random(3, 4), _random(1, 8), k=1)
