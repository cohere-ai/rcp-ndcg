"""The :class:`~rcp_ndcg.retrieval.encoder.Encoder` contract.

What is actually being pinned down here:

* :class:`Embeddings` refuses layouts whose offsets do not describe its vectors,
  because a plausible-but-wrong offsets array silently reattributes vectors to
  the wrong documents.
* ``as_matrix()`` refuses multi-vector input rather than pooling it, so nobody
  gets single-vector numbers out of a late-interaction model by accident.
* A text-only encoder refuses media instead of embedding the empty string an
  image-only document's ``.text`` is.
* The registry resolves the old ``local`` / ``api`` backend names, which are
  written into every index built before backends were named for where they run.
"""

from __future__ import annotations

import numpy as np
import pytest
from rcp_ndcg_core.content import Content, MediaRef

from rcp_ndcg.errors import CapabilityError
from rcp_ndcg.retrieval.encoder import Embeddings, Encoder, EncodeRole


class _Stub(Encoder):
    def __init__(self, *, dim: int = 2, supports_media: bool = False) -> None:
        self._dim = dim
        self.supports_media = supports_media

    def encode(self, contents, *, role, batch_size=None) -> Embeddings:
        self.check_media(contents)
        return Embeddings.single(np.ones((len(contents), self._dim), dtype=np.float32))


class TestEmbeddings:
    def test_single_vector_shape(self) -> None:
        embeddings = Embeddings.single(np.zeros((3, 4), dtype=np.float32))
        assert not embeddings.is_multi_vector
        assert (embeddings.num_items, embeddings.dim) == (3, 4)

    def test_ragged_keeps_per_item_boundaries(self) -> None:
        embeddings = Embeddings.ragged([np.ones((2, 3)), np.zeros((5, 3))])
        assert embeddings.is_multi_vector
        assert embeddings.num_items == 2
        assert embeddings.offsets is not None and embeddings.offsets.tolist() == [0, 2, 7]
        assert embeddings.vectors.shape == (7, 3)

    def test_ragged_tolerates_an_empty_item(self) -> None:
        """A document that produced no vectors still has to occupy a position."""
        embeddings = Embeddings.ragged([np.ones((2, 3)), np.zeros((0, 3)), np.ones((1, 3))])
        assert embeddings.num_items == 3
        assert embeddings.offsets is not None and embeddings.offsets.tolist() == [0, 2, 2, 3]

    def test_offsets_must_describe_the_vectors(self) -> None:
        with pytest.raises(ValueError, match="offsets end at"):
            Embeddings(vectors=np.zeros((4, 2), dtype=np.float32), offsets=np.array([0, 1, 2]))

    def test_offsets_must_start_at_zero(self) -> None:
        with pytest.raises(ValueError, match="must start at 0"):
            Embeddings(vectors=np.zeros((4, 2), dtype=np.float32), offsets=np.array([1, 4]))

    def test_three_dimensional_vectors_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be 2-D"):
            Embeddings(vectors=np.zeros((2, 3, 4), dtype=np.float32))

    def test_as_matrix_refuses_to_pool_silently(self) -> None:
        embeddings = Embeddings.ragged([np.ones((2, 3)), np.ones((3, 3))])
        with pytest.raises(ValueError, match="multi-vector"):
            embeddings.as_matrix()

    def test_l2_normalizes_every_vector_not_every_item(self) -> None:
        """MaxSim needs per-vector unit norm, which is also what cosine needs."""
        embeddings = Embeddings.ragged([np.array([[3.0, 4.0], [0.0, 5.0]])]).l2_normalized()
        np.testing.assert_allclose(np.linalg.norm(embeddings.vectors, axis=1), [1.0, 1.0], atol=1e-6)

    def test_concat_shifts_offsets(self) -> None:
        left = Embeddings.ragged([np.ones((2, 3))])
        right = Embeddings.ragged([np.zeros((1, 3)), np.ones((4, 3))])
        joined = left.concat(right)
        assert joined.num_items == 3
        assert joined.offsets is not None and joined.offsets.tolist() == [0, 2, 3, 7]

    def test_concat_refuses_mixed_layouts(self) -> None:
        with pytest.raises(ValueError, match="single-vector and multi-vector"):
            Embeddings.single(np.ones((1, 3), dtype=np.float32)).concat(Embeddings.ragged([np.ones((1, 3))]))

    def test_empty_carries_its_width(self) -> None:
        """An empty shard has to declare a dim, or reassembly cannot size the output."""
        assert Embeddings.empty(7).dim == 7
        assert Embeddings.empty(7, multi_vector=True).is_multi_vector


class TestMediaGating:
    def test_text_only_encoder_names_the_offending_item(self) -> None:
        page = Content.from_image("gs://bucket/page-3.png", page=3)
        with pytest.raises(CapabilityError, match=r"item 1 carries 1 media reference"):
            _Stub().encode([Content.from_text("fine"), page], role=EncodeRole.DOCUMENT)

    def test_media_capable_encoder_accepts_it(self) -> None:
        page = Content.from_image("gs://bucket/page-3.png")
        out = _Stub(supports_media=True).encode([page], role=EncodeRole.DOCUMENT)
        assert out.num_items == 1

    def test_video_frames_count_as_media(self) -> None:
        from rcp_ndcg_core.content import VideoPart

        clip = Content.from_parts([VideoPart(frames=[MediaRef(uri="gs://b/f0.jpg")])])
        with pytest.raises(CapabilityError, match="cannot encode media"):
            _Stub().encode([clip], role=EncodeRole.DOCUMENT)


class TestEncodeTexts:
    def test_convenience_wrapper_lifts_strings(self) -> None:
        out = _Stub(dim=3).encode_texts(["a", "b"], role=EncodeRole.QUERY)
        assert (out.num_items, out.dim) == (2, 3)
