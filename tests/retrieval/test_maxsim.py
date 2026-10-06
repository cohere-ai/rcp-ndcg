"""Exact MaxSim scoring and token pooling.

The blocked implementation is checked against the two-line reference definition
on random data, at several block sizes, because the blocking is where this can go
subtly wrong: a reduceat boundary off by one gives a plausible score attributed to
the neighbouring document, which no shape assertion would catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from rcp_ndcg.inference.types import Embeddings
from rcp_ndcg.retrieval import maxsim
from rcp_ndcg.retrieval.maxsim import maxsim_topk


def _ragged(rng: np.random.Generator, lengths: list[int], dim: int) -> Embeddings:
    return Embeddings.ragged(
        [rng.standard_normal((length, dim)).astype(np.float32) for length in lengths]
    ).l2_normalized()


def _item(embeddings: Embeddings, index: int) -> np.ndarray:
    assert embeddings.offsets is not None
    return embeddings.vectors[int(embeddings.offsets[index]) : int(embeddings.offsets[index + 1])]


def _maxsim(query_vectors: np.ndarray, doc_vectors: np.ndarray) -> float:
    """The definition: for each query vector the best document vector, summed."""
    if len(query_vectors) == 0 or len(doc_vectors) == 0:
        return 0.0
    return float(np.max(query_vectors @ doc_vectors.T, axis=1).sum())


def _reference_scores(docs: Embeddings, queries: Embeddings) -> np.ndarray:
    return np.array(
        [
            [_maxsim(_item(queries, qi), _item(docs, di)) for di in range(docs.num_items)]
            for qi in range(queries.num_items)
        ],
        dtype=np.float64,
    )


class TestMaxSimTopk:
    def test_matches_the_reference_definition(self) -> None:
        rng = np.random.default_rng(0)
        docs = _ragged(rng, [3, 7, 1, 12, 5], dim=8)
        queries = _ragged(rng, [2, 4, 1], dim=8)

        scores, indices = maxsim_topk(docs, queries, k=5)
        expected = _reference_scores(docs, queries)

        for qi in range(queries.num_items):
            order = np.lexsort((np.arange(docs.num_items), -expected[qi]))
            np.testing.assert_array_equal(indices[qi], order[:5])
            np.testing.assert_allclose(scores[qi], expected[qi][order[:5]], rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(("tile_bytes", "query_tokens"), [(1, 1), (200, 1), (400, 4)])
    def test_blocking_does_not_change_the_answer(
        self, tile_bytes: int, query_tokens: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rng = np.random.default_rng(1)
        docs = _ragged(rng, [4, 2, 9, 1, 6, 3], dim=6)
        queries = _ragged(rng, [3, 5], dim=6)

        whole = maxsim_topk(docs, queries, k=4)
        monkeypatch.setattr(maxsim, "_TILE_BYTES", tile_bytes)
        monkeypatch.setattr(maxsim, "_QUERY_BLOCK_TOKENS", query_tokens)
        blocked = maxsim_topk(docs, queries, k=4)
        np.testing.assert_array_equal(blocked[1], whole[1])
        np.testing.assert_allclose(blocked[0], whole[0], rtol=1e-5, atol=1e-5)

    def test_a_document_of_one_vector_is_ordinary(self) -> None:
        """The degenerate case is not special: MaxSim over one vector is a dot product."""
        docs = Embeddings.ragged([np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])])
        queries = Embeddings.ragged([np.array([[1.0, 0.0]])])
        scores, indices = maxsim_topk(docs, queries, k=2)
        assert indices[0].tolist() == [0, 1]
        np.testing.assert_allclose(scores[0], [1.0, 0.0], atol=1e-6)

    def test_empty_document_ranks_last_and_stays_a_real_index(self) -> None:
        """A document with no vectors must rank last without poisoning the sum.

        Two failure modes are being excluded: summing ``-inf`` per query token
        overflows, and a ``-inf`` score ties with the running top-k's placeholder,
        which then survives and leaves ``-1`` in the output where a document id
        belongs.
        """
        docs = Embeddings.ragged([np.zeros((0, 2)), np.array([[1.0, 0.0]])])
        queries = Embeddings.ragged([np.array([[1.0, 0.0], [1.0, 0.0]])])
        scores, indices = maxsim_topk(docs, queries, k=2)
        assert indices[0].tolist() == [1, 0]
        assert scores[0][0] == pytest.approx(2.0)
        assert np.isfinite(scores[0][1]) and scores[0][1] < -1e30

    def test_a_trailing_empty_document_or_query_does_not_crash(self) -> None:
        """An empty item at the END of either side sits past the last column/row, where reduceat's index goes
        out of bounds; it is an ordinary empty item, not a crash."""
        docs = Embeddings.ragged([np.array([[1.0, 0.0]]), np.zeros((0, 2))])
        queries = Embeddings.ragged([np.array([[1.0, 0.0]]), np.zeros((0, 2))])

        scores, indices = maxsim_topk(docs, queries, k=2)

        assert indices[0].tolist() == [0, 1]
        np.testing.assert_allclose(scores[0][0], 1.0)
        np.testing.assert_allclose(scores[1][0], 0.0)  # an empty query scores 0, like the reference definition

    def test_ties_break_toward_the_lower_index(self) -> None:
        vector = np.array([[1.0, 0.0]])
        docs = Embeddings.ragged([vector, vector, vector])
        queries = Embeddings.ragged([vector])
        _, indices = maxsim_topk(docs, queries, k=3)
        assert indices[0].tolist() == [0, 1, 2]

    def test_k_larger_than_the_corpus_is_clamped(self) -> None:
        docs = Embeddings.ragged([np.array([[1.0, 0.0]])])
        queries = Embeddings.ragged([np.array([[1.0, 0.0]])])
        scores, indices = maxsim_topk(docs, queries, k=10)
        assert scores.shape == indices.shape == (1, 1)

    def test_no_queries_returns_an_empty_result(self) -> None:
        docs = Embeddings.ragged([np.array([[1.0, 0.0]])])
        scores, indices = maxsim_topk(docs, Embeddings.empty(2, multi_vector=True), k=3)
        assert scores.shape == (0, 1)

    def test_single_vector_input_is_refused(self) -> None:
        """Silently pooling would report late-interaction numbers that are not."""
        flat = Embeddings.single(np.ones((2, 3), dtype=np.float32))
        ragged = Embeddings.ragged([np.ones((1, 3))])
        with pytest.raises(ValueError, match="needs multi-vector embeddings"):
            maxsim_topk(flat, ragged, k=1)

    def test_dimension_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="dimension mismatch"):
            maxsim_topk(
                Embeddings.ragged([np.ones((1, 4))]),
                Embeddings.ragged([np.ones((1, 3))]),
                k=1,
            )


class TestTransferPrecision:
    """Float16 vectors in, float32 scoring out: the transfer dtype never reaches the accumulation."""

    @staticmethod
    def _float16_embeddings(
        rng: np.random.Generator, docs: int, doc_tokens: int, queries: int, dim: int
    ) -> tuple[Embeddings, Embeddings, list[np.ndarray], list[np.ndarray]]:
        """Random float16 documents and queries, with the float64 originals alongside."""
        doc_vectors = [rng.standard_normal((doc_tokens, dim)) for _ in range(docs)]
        query_vectors = [rng.standard_normal((12, dim)) for _ in range(queries)]
        # ragged(dtype=float16): the buffer keeps the transfer precision, as the pooling client's does
        doc_embeddings = Embeddings.ragged([v.astype(np.float16) for v in doc_vectors], dtype=np.float16)
        query_embeddings = Embeddings.ragged([v.astype(np.float16) for v in query_vectors], dtype=np.float16)
        return doc_embeddings, query_embeddings, doc_vectors, query_vectors

    @staticmethod
    def _reference_at(doc_embeddings: Embeddings, query_embeddings: Embeddings, indices: np.ndarray) -> np.ndarray:
        """The two-line float64 reference, at the indices the scorer returned."""

        def item(embeddings: Embeddings, index: int) -> np.ndarray:
            assert embeddings.offsets is not None
            span = embeddings.vectors[int(embeddings.offsets[index]) : int(embeddings.offsets[index + 1])]
            return span.astype(np.float64)

        return np.array(
            [
                [np.max(item(query_embeddings, qi) @ item(doc_embeddings, di).T, axis=1).sum() for di in row]
                for qi, row in enumerate(indices)
            ]
        )

    def test_float16_inputs_match_a_float64_reference(self) -> None:
        """Long documents (2,000 tokens, dim 128) scored in float32 sit within 1e-3 relative of the float64
        reference over the same float16 vectors -- and within 1e-3 of the float64 originals, so float16
        storage costs its quantisation and nothing more."""
        rng = np.random.default_rng(7)
        doc_embeddings, query_embeddings, doc_vectors, query_vectors = self._float16_embeddings(
            rng, docs=6, doc_tokens=2_000, queries=4, dim=128
        )

        scores, indices = maxsim_topk(doc_embeddings, query_embeddings, 4)

        assert scores.dtype == np.float32
        original = np.array(
            [
                [np.max(q @ doc_vectors[di].T, axis=1).sum() for di in row]
                for q, row in zip(query_vectors, indices, strict=True)
            ]
        )
        np.testing.assert_allclose(scores, original, rtol=1e-3, atol=0)

    def test_the_accumulation_is_float32_not_float16(self) -> None:
        """The pin that keeps the upcast honest: scoring the same float16 vectors in float32 sits ~1e-7 from
        the float64 reference; accumulating in float16 drifts ~1e-4, which this bound refuses."""
        rng = np.random.default_rng(7)
        doc_embeddings, query_embeddings, _, _ = self._float16_embeddings(
            rng, docs=6, doc_tokens=2_000, queries=4, dim=128
        )

        scores, indices = maxsim_topk(doc_embeddings, query_embeddings, 4)

        reference = self._reference_at(doc_embeddings, query_embeddings, indices)
        np.testing.assert_allclose(scores, reference, rtol=1e-5, atol=0)

    def test_the_upcast_is_blockwise_and_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The scorer never materialises a float32 copy of the corpus: with a small tile budget, each block
        copy is at most the budget and many blocks run -- a whole-corpus upcast would be one giant copy."""
        rng = np.random.default_rng(8)
        doc_embeddings = Embeddings.ragged(
            [rng.standard_normal((50, 128)).astype(np.float16) for _ in range(40)], dtype=np.float16
        )
        query_embeddings = Embeddings.ragged(
            [rng.standard_normal((5, 128)).astype(np.float16) for _ in range(8)], dtype=np.float16
        )

        copied: list[int] = []
        real = maxsim._f32_block

        def recording(vectors: np.ndarray) -> np.ndarray:
            block = real(vectors)
            copied.append(block.nbytes)
            return block

        monkeypatch.setattr(maxsim, "_TILE_BYTES", 64 << 10)  # 64 KiB: two documents per block
        monkeypatch.setattr(maxsim, "_f32_block", recording)
        maxsim_topk(doc_embeddings, query_embeddings, 4)

        assert len(copied) > 1  # the corpus was never upcast whole
        assert max(copied) <= 64 << 10
        assert max(copied) < doc_embeddings.vectors.nbytes


class TestScoreTopkDispatch:
    def test_flat_embeddings_use_the_inner_product_path(self) -> None:
        from rcp_ndcg.retrieval.topk import numpy_topk, score_topk

        rng = np.random.default_rng(2)
        docs = Embeddings.single(rng.standard_normal((10, 4)).astype(np.float32))
        queries = Embeddings.single(rng.standard_normal((3, 4)).astype(np.float32))
        np.testing.assert_array_equal(
            score_topk(docs, queries, 5)[1],
            numpy_topk(docs.vectors, queries.vectors, 5)[1],
        )

    def test_ragged_embeddings_use_maxsim(self) -> None:
        from rcp_ndcg.retrieval.topk import score_topk

        rng = np.random.default_rng(3)
        docs = _ragged(rng, [2, 3], dim=4)
        queries = _ragged(rng, [2], dim=4)
        np.testing.assert_allclose(
            score_topk(docs, queries, 2)[0],
            maxsim_topk(docs, queries, 2)[0],
        )

    def test_numpy_topk_points_at_maxsim_when_given_a_3d_array(self) -> None:
        from rcp_ndcg.retrieval.topk import numpy_topk

        with pytest.raises(ValueError, match="maxsim_topk"):
            numpy_topk(np.zeros((2, 3, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32), 1)
