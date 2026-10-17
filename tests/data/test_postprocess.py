"""The postprocess home's guards: the skip keep-mask and L2 normalisation (the MRL head moved to
:mod:`rcp_ndcg.data.mrl`, and its tests with it)."""

from __future__ import annotations

import numpy as np
import pytest

from rcp_ndcg.data.postprocess import kept_vector_count, l2_normalize, skip_keep_mask


class TestKeptVectorCount:
    """The declared count a reply is checked against when the served plugin applies the rule engine-side."""

    def test_a_text_document_counts_the_ids_outside_the_skip_list(self) -> None:
        ids = [10, 11, 12, 13, 11]
        assert kept_vector_count(ids, [11, 13]) == 2
        assert kept_vector_count(ids, []) == 5

    def test_a_media_document_counts_the_kept_head_plus_the_media_block(self) -> None:
        # The head (the [D] prompt) is tokenised and masked; the media block (wrapper + patches) is kept
        # whole, so the count is the block's own tokens plus the head's kept ids.
        assert kept_vector_count([248078], [11], media_tokens=42) == 43
        assert kept_vector_count([11], [11], media_tokens=42) == 42
        assert kept_vector_count([], [], media_tokens=42) == 42

    def test_nothing_to_count_is_refused(self) -> None:
        from rcp_ndcg.errors import DataError

        with pytest.raises(DataError, match="count") as caught:
            kept_vector_count([], [])
        assert caught.value.hint, "the typed refusal names the next step"
        with pytest.raises(DataError, match="media_tokens"):
            kept_vector_count([1], [], media_tokens=-1)


class TestSkipKeepMask:
    """The late-interaction keep-mask for a text document's ids (the image-position rule is the pool
    client's per-item exemption; this mask is what the text documents use)."""

    def test_the_kept_positions_are_the_ids_outside_the_skip_list(self) -> None:
        ids = [10, 11, 12, 13, 11]
        # Every position whose id is listed drops, however often the id recurs.
        assert skip_keep_mask(ids, [11, 13]) == [0, 2]
        assert skip_keep_mask(ids, [12]) == [0, 1, 3, 4]

    def test_an_empty_skip_list_keeps_everything(self) -> None:
        assert skip_keep_mask([5, 6], []) == [0, 1]

    def test_empty_token_ids_are_refused(self) -> None:
        from rcp_ndcg.errors import DataError

        with pytest.raises(DataError, match="token ids") as caught:
            skip_keep_mask([], [1])
        assert caught.value.hint, "the typed refusal names the next step"


class TestL2Normalize:
    def test_rows_scale_to_unit_norm_and_zeros_stay_zero(self) -> None:
        normalized = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
        np.testing.assert_allclose(normalized[0], [0.6, 0.8], rtol=1e-6)
        np.testing.assert_array_equal(normalized[1], [0.0, 0.0])

    def test_a_float16_buffer_keeps_its_dtype(self) -> None:
        assert l2_normalize(np.array([[3.0, 4.0]], dtype=np.float16)).dtype == np.float16
