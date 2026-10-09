"""The postprocess home's guards: the skip keep-mask, the MRL cut and L2 normalisation."""

from __future__ import annotations

import numpy as np
import pytest

from rcp_ndcg.data.postprocess import l2_normalize, mrl_cut, skip_keep_mask


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


class TestMrlCut:
    def test_the_cut_slices_then_renormalises(self) -> None:
        vectors = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        cut = mrl_cut(vectors, 2)
        # The slice renormalised: [0.6, 0.8]; the zero row stays zero.
        np.testing.assert_allclose(cut[0], [0.6, 0.8], rtol=1e-6)
        np.testing.assert_array_equal(cut[1], [0.0, 0.0])
        assert cut.flags["C_CONTIGUOUS"]

    def test_a_float16_buffer_stays_float16(self) -> None:
        vectors = np.array([[3.0, 4.0, 0.0]], dtype=np.float16)
        assert mrl_cut(vectors, 2).dtype == np.float16


class TestL2Normalize:
    def test_rows_scale_to_unit_norm_and_zeros_stay_zero(self) -> None:
        normalized = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
        np.testing.assert_allclose(normalized[0], [0.6, 0.8], rtol=1e-6)
        np.testing.assert_array_equal(normalized[1], [0.0, 0.0])

    def test_a_float16_buffer_keeps_its_dtype(self) -> None:
        assert l2_normalize(np.array([[3.0, 4.0]], dtype=np.float16)).dtype == np.float16
