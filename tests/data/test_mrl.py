"""The MRL head: the truncation cut and the learned-projection head, one home.

The head applies a selected output dimension to full-width vectors: truncation slices and renormalises
(the card's order), projection applies the checkpoint's own learned matrices in float32 and renormalises.
The projection file is a safetensors file read through the product's own minimal reader, so the tests
build the bytes (``tests/_safetensors.py``) and hand-compute the expected matmul.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rcp_ndcg.data.mrl import MrlHead, MrlProjection, mrl_cut
from rcp_ndcg.errors import ConfigError, DataError
from tests._safetensors import write_safetensors


def _head(**overrides: object) -> MrlHead:
    settings: dict[str, object] = {"kind": "truncation", "dims": (2,)}
    settings.update(overrides)
    return MrlHead(**settings)  # type: ignore[arg-type]


class TestTruncation:
    def test_the_cut_is_slice_then_renormalise(self) -> None:
        """The card's order: [3, 4] (unit [0.6, 0.8]) cut to one dimension renormalises to [1.0]; a cut
        after the normalisation would ship [0.6] -- wrong."""
        cut = mrl_cut(np.asarray([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32), 1)
        np.testing.assert_allclose(cut, [[1.0], [1.0]], atol=1e-6)

    def test_the_cut_keeps_the_input_dtype(self) -> None:
        vectors = np.asarray([[3.0, 4.0]], dtype=np.float16)
        assert mrl_cut(vectors, 1).dtype == np.float16

    def test_a_zero_row_stays_zero_and_the_cut_is_contiguous(self) -> None:
        vectors = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        cut = mrl_cut(vectors, 2)
        np.testing.assert_allclose(cut[0], [0.6, 0.8], rtol=1e-6)
        np.testing.assert_array_equal(cut[1], [0.0, 0.0])
        assert cut.flags["C_CONTIGUOUS"]

    def test_the_head_applies_the_cut(self) -> None:
        head = _head(dims=(1, 2))
        vectors = np.asarray([[3.0, 4.0]], dtype=np.float32)
        np.testing.assert_allclose(head.apply(vectors, 1), [[1.0]], atol=1e-6)

    def test_a_cut_wider_than_the_vectors_is_refused(self) -> None:
        head = _head(dims=(4,))
        with pytest.raises(ConfigError, match="mrl_dims|mrl_dim") as caught:
            head.apply(np.ones((1, 2), dtype=np.float32), 4)
        assert "mrl_dims" in (caught.value.hint or "")

    def test_a_k_outside_the_declared_set_is_refused(self) -> None:
        head = _head(dims=(1, 2))
        with pytest.raises(ConfigError, match="mrl_dims") as caught:
            head.apply(np.ones((1, 2), dtype=np.float32), 3)
        assert "mrl_dims" in (caught.value.hint or "")

    def test_a_range_selects_every_k_inside_it(self) -> None:
        """A card whose prose gives a range ("from 32 to 1024") declares mrl_range, and every k in the
        closed interval is selectable: the head cuts and renormalises it."""
        head = _head(kind="truncation", dims=(), mrl_range=(2, 4))
        vectors = np.asarray([[3.0, 4.0, 5.0, 6.0]], dtype=np.float32)
        cut = head.apply(vectors, 3)
        expected = vectors[:, :3]
        np.testing.assert_allclose(cut, expected / np.linalg.norm(expected), atol=1e-6)

    def test_a_range_refuses_a_k_below_the_floor_and_above_the_ceiling(self) -> None:
        head = _head(kind="truncation", dims=(), mrl_range=(2, 4))
        vectors = np.ones((1, 8), dtype=np.float32)
        for k in (1, 5):
            with pytest.raises(ConfigError, match="mrl_range") as caught:
                head.apply(vectors, k)
            assert "mrl_range" in (caught.value.hint or "")

    def test_a_none_kind_applies_nothing(self) -> None:
        head = MrlHead(kind="none")
        with pytest.raises(ConfigError, match="mrl_kind"):
            head.apply(np.ones((1, 2), dtype=np.float32), 1)


class TestProjection:
    def test_the_projection_equals_a_hand_computed_matmul(self, tmp_path: Path) -> None:
        """A 4-wide vector through the checkpoint's own (4, 2) matrix, renormalised: the head's output is
        exactly ``normalise(v @ W)`` in float32."""
        matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        source = write_safetensors(tmp_path / "projections.safetensors", {"2": matrix})
        head = _head(
            kind="projection",
            dims=(2,),
            projection=MrlProjection(source=str(source)),
        )
        vectors = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

        projected = head.apply(vectors, 2)

        expected = vectors @ matrix
        expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(projected, expected, atol=1e-6)

    def test_the_chain_applies_the_widest_matrix_first(self, tmp_path: Path) -> None:
        """The zembed convention: the tensor names are the target widths, and the chain for ``k`` is every
        declared dimension at or above it, widest first -- 8 -> 4 -> 2."""
        wide = np.eye(8, 4, dtype=np.float32)
        narrow = np.eye(4, 2, dtype=np.float32)
        source = write_safetensors(tmp_path / "projections.safetensors", {"4": wide, "2": narrow})
        head = _head(
            kind="projection",
            dims=(2, 4),
            projection=MrlProjection(source=str(source)),
        )
        vectors = np.arange(8, dtype=np.float32).reshape(1, 8)

        projected = head.apply(vectors, 2)

        expected = vectors @ wide @ narrow
        expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(projected, expected, atol=1e-6)

    def test_an_explicit_chain_overrides_the_naming_convention(self, tmp_path: Path) -> None:
        matrix = np.eye(4, 2, dtype=np.float32)
        source = write_safetensors(tmp_path / "projections.safetensors", {"t0": matrix})
        projection = MrlProjection(source=str(source), chains={2: ("t0",)})
        head = _head(kind="projection", dims=(2,), projection=projection)
        vectors = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

        projected = head.apply(vectors, 2)

        expected = vectors @ matrix
        expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(projected, expected, atol=1e-6)

    def test_a_missing_matrix_for_the_selected_dim_is_refused(self, tmp_path: Path) -> None:
        source = write_safetensors(tmp_path / "projections.safetensors", {"4": np.eye(4, 2, dtype=np.float32)})
        projection = MrlProjection(source=str(source), chains={4: ("4",)})
        head = _head(kind="projection", dims=(2, 4), projection=projection)
        with pytest.raises(ConfigError, match="mrl_projection") as caught:
            head.apply(np.ones((1, 4), dtype=np.float32), 2)
        assert "mrl_projection" in (caught.value.hint or "")

    def test_a_matrix_that_does_not_match_the_vector_width_is_refused(self, tmp_path: Path) -> None:
        source = write_safetensors(tmp_path / "projections.safetensors", {"2": np.eye(4, 2, dtype=np.float32)})
        head = _head(kind="projection", dims=(2,), projection=MrlProjection(source=str(source)))
        with pytest.raises(DataError, match="width|shape") as caught:
            head.apply(np.ones((1, 8), dtype=np.float32), 2)
        assert "mrl_projection" in (caught.value.hint or "")

    def test_a_chain_that_does_not_connect_is_refused(self, tmp_path: Path) -> None:
        """A file with one matrix per k (each mapping the FULL width) is not a chain: the second matrix's
        input width does not equal the first's output width, and the head refuses instead of computing."""
        first = np.eye(8, 4, dtype=np.float32)
        second = np.eye(8, 2, dtype=np.float32)
        source = write_safetensors(tmp_path / "projections.safetensors", {"4": first, "2": second})
        head = _head(kind="projection", dims=(2, 4), projection=MrlProjection(source=str(source)))
        with pytest.raises(DataError, match="chain|width"):
            head.apply(np.ones((1, 8), dtype=np.float32), 2)

    def test_a_missing_projection_source_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="mrl_projection"):
            MrlHead(kind="projection", dims=(2,))

    def test_the_projection_spec_refuses_an_empty_chain(self) -> None:
        with pytest.raises(ValueError, match="chains"):
            MrlProjection(source="x.safetensors", chains={2: ()})

    def test_the_projection_spec_refuses_a_non_positive_k(self) -> None:
        with pytest.raises(ValueError, match="chains"):
            MrlProjection(source="x.safetensors", chains={0: ("t",)})
