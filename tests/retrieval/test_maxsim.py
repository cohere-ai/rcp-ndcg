"""Exact MaxSim scoring and token pooling.

The blocked implementation is checked against the two-line reference definition
on random data, at several block sizes, because the blocking is where this can go
subtly wrong: a reduceat boundary off by one gives a plausible score attributed to
the neighbouring document, which no shape assertion would catch.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from rcp_ndcg.errors import ConfigError, DataError
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


def _expected_scores(docs: Embeddings, queries: Embeddings) -> np.ndarray:
    """The full expected score matrix, including the empty-item policy: a document with no vectors scores the
    sentinel (strictly below every real score, so it ranks last), an empty query scores 0 like the definition."""
    out = np.zeros((queries.num_items, docs.num_items), dtype=np.float64)
    for qi in range(queries.num_items):
        for di in range(docs.num_items):
            query_vectors, doc_vectors = _item(queries, qi), _item(docs, di)
            if len(doc_vectors) == 0:
                out[qi, di] = maxsim._EMPTY_DOC_SCORE
            elif len(query_vectors) == 0:
                out[qi, di] = 0.0
            else:
                out[qi, di] = _maxsim(query_vectors, doc_vectors)
    return out


def _assert_matches_reference(docs: Embeddings, queries: Embeddings, k: int | None = None) -> None:
    """The scorer's output equals the two-line reference, ordered by score then lower index, including where
    empty items sit -- the reference encodes the empty-item policy, so a neighbour's stolen token shows up."""
    k = docs.num_items if k is None else k
    expected = _expected_scores(docs, queries)
    scores, indices = maxsim_topk(docs, queries, k)
    for qi in range(queries.num_items):
        order = np.lexsort((np.arange(docs.num_items), -expected[qi]))
        np.testing.assert_array_equal(indices[qi], order[:k])
        np.testing.assert_allclose(scores[qi], expected[qi][order[:k]], rtol=1e-5, atol=1e-5)


def _ragged_explicit(lengths: Sequence[int], dim: int) -> Embeddings:
    """A set whose every item is empty keeps its declared width: ``Embeddings.ragged`` gives one of width 0
    by design, and the scorer's width check (rightly) refuses a 0-width side against a real one."""
    if any(lengths):
        return Embeddings.ragged([np.zeros((length, dim), dtype=np.float32) for length in lengths])
    return Embeddings(vectors=np.zeros((0, dim)), offsets=np.zeros(len(lengths) + 1, dtype=np.int64))


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

    def test_a_trailing_empty_item_does_not_truncate_its_predecessor(self) -> None:
        """An empty item at the END of either side sits past the last column/row, where reduceat's index would
        go out of bounds. The old clamp for that (``np.minimum(start, size - 1)``) stole the predecessor's last
        token instead: a two-token item followed by an empty one scored over all but its last token. With exact
        float inputs, both axes must come out at the full item's score."""
        # Doc axis: doc0 = [[1, 0], [0, 1]] followed by an empty doc; query = [[0, 1]].
        # The clamp computed max over doc0's FIRST column only: 0.0 instead of 1.0.
        docs = Embeddings.ragged([np.array([[1.0, 0.0], [0.0, 1.0]]), np.zeros((0, 2))])
        queries = Embeddings.ragged([np.array([[0.0, 1.0]])])
        scores, indices = maxsim_topk(docs, queries, k=2)
        assert indices[0].tolist() == [0, 1]
        np.testing.assert_allclose(scores[0][0], 1.0)

        # Query axis: query0 = two tokens followed by an empty query; doc = [[0, 1]].
        # The clamp dropped query0's last token row: 0.0 instead of 0.0 + 1.0.
        queries = Embeddings.ragged([np.array([[1.0, 0.0], [0.0, 1.0]]), np.zeros((0, 2))])
        docs = Embeddings.ragged([np.array([[0.0, 1.0]])])
        scores, _ = maxsim_topk(docs, queries, k=1)
        np.testing.assert_allclose(scores[0][0], 1.0)

    @pytest.mark.parametrize(
        "doc_lengths",
        [(2, 0, 3), (0, 3, 2), (3, 0, 0), (0, 0), (4, 0, 1, 0, 2)],
    )
    def test_empty_documents_score_like_the_reference_at_any_position(
        self, monkeypatch: pytest.MonkeyPatch, doc_lengths: tuple[int, ...]
    ) -> None:
        """Empty documents first, between, and after real ones, with the blocking forced small: every item's
        score is the reference's, so neither the reduceat clamp nor a skipped block can attribute one item's
        score to its neighbour."""
        rng = np.random.default_rng(11)
        docs = _ragged_explicit(list(doc_lengths), 4)
        queries = Embeddings.ragged(
            [rng.standard_normal((2, 4)).astype(np.float32), rng.standard_normal((1, 4)).astype(np.float32)]
        )
        monkeypatch.setattr(maxsim, "_TILE_BYTES", 128)
        monkeypatch.setattr(maxsim, "_QUERY_BLOCK_TOKENS", 1)

        _assert_matches_reference(docs, queries)

    @pytest.mark.parametrize("query_lengths", [(2, 0), (0, 2), (3, 0, 0), (0, 0)])
    def test_empty_queries_score_like_the_reference_at_any_position(
        self, monkeypatch: pytest.MonkeyPatch, query_lengths: tuple[int, ...]
    ) -> None:
        rng = np.random.default_rng(12)
        docs = Embeddings.ragged([rng.standard_normal((3, 4)).astype(np.float32) for _ in range(4)])
        queries = _ragged_explicit(list(query_lengths), 4)
        monkeypatch.setattr(maxsim, "_TILE_BYTES", 128)
        monkeypatch.setattr(maxsim, "_QUERY_BLOCK_TOKENS", 1)

        _assert_matches_reference(docs, queries)

    def test_an_all_empty_query_set_scores_zero_with_real_indices(self) -> None:
        """Every query empty: the old block skip left the running ``-inf``/``-1`` placeholders in the output;
        the definition scores an empty query 0 against every real document, with real document indices."""
        docs = Embeddings.ragged([np.array([[1.0, 0.0]])])
        queries = Embeddings(vectors=np.zeros((0, 2)), offsets=np.zeros(3, dtype=np.int64))

        scores, indices = maxsim_topk(docs, queries, k=1)

        np.testing.assert_allclose(scores, [[0.0], [0.0]])
        np.testing.assert_array_equal(indices, [[0], [0]])

    def test_an_all_empty_corpus_ranks_its_documents_with_the_sentinel(self) -> None:
        """Every document empty: both slots carry the designed sentinel and a real index, never ``-inf``/``-1``."""
        docs = Embeddings(vectors=np.zeros((0, 2)), offsets=np.zeros(3, dtype=np.int64))
        queries = Embeddings.ragged([np.array([[1.0, 0.0]])])

        scores, indices = maxsim_topk(docs, queries, k=2)

        assert indices[0].tolist() == [0, 1], "real document indices, never the -1 placeholder"
        assert np.isfinite(scores[0]).all() and (scores[0] < -1e30).all(), "the sentinel, not -inf"

    def test_an_empty_item_at_a_block_boundary_keeps_a_real_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A block whose every document is empty was skipped whole, leaving its slots at the running
        placeholders whenever ``k`` reached them; it is scored with the sentinel instead."""
        monkeypatch.setattr(maxsim, "_TILE_BYTES", 8)  # one document per block
        # Item 0 is empty; items 1 and 2 are single-token documents.
        docs = Embeddings(vectors=np.array([[1.0, 0.0], [0.0, 1.0]]), offsets=np.array([0, 0, 1, 2], dtype=np.int64))
        queries = Embeddings.ragged([np.array([[1.0, 0.0]])])

        scores, indices = maxsim_topk(docs, queries, k=3)

        assert indices[0].tolist() == [1, 2, 0], "no -1 anywhere: the empty item ranks last with its own index"
        np.testing.assert_allclose(scores[0][0], 1.0)
        np.testing.assert_allclose(scores[0][1], 0.0)
        assert scores[0][2] < -1e30

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
        with pytest.raises(DataError, match="needs multi-vector embeddings"):
            maxsim_topk(flat, ragged, k=1)

    def test_dimension_mismatch_is_refused(self) -> None:
        with pytest.raises(DataError, match="dimension mismatch"):
            maxsim_topk(
                Embeddings.ragged([np.ones((1, 4))]),
                Embeddings.ragged([np.ones((1, 3))]),
                k=1,
            )

    def test_non_positive_k_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="k must be positive"):
            maxsim_topk(Embeddings.ragged([np.ones((1, 4))]), Embeddings.ragged([np.ones((1, 4))]), k=0)


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


class TestTiesAtTheCut:
    """The tie rule decides the cut as well as the order: ``argpartition``'s pick among the candidates tied at
    the k-th score is implementation-defined, so a tie class straddling the cut is re-selected by index."""

    def test_maxsim_ties_straddling_the_cut_break_toward_the_lower_index(self) -> None:
        docs = Embeddings.ragged([np.array([[9.0, 0.0]]), *([np.array([[5.0, 0.0]])] * 4)])
        queries = Embeddings.ragged([np.array([[1.0, 0.0]])])

        _, indices = maxsim_topk(docs, queries, k=3)

        assert indices[0].tolist() == [0, 1, 2], "the tied documents keep the cut's slots in index order"


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

        with pytest.raises(DataError, match="maxsim_topk"):
            numpy_topk(np.zeros((2, 3, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32), 1)
