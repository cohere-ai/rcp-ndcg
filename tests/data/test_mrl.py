"""The MRL head: the truncation cut and the learned-projection head, one home.

The head applies a selected output dimension to full-width vectors: truncation slices and renormalises
(the card's order), projection applies the checkpoint's own learned matrices in float32 and renormalises.
The projection file is a safetensors file read through the product's own minimal reader, so the tests
build the bytes (``tests/_safetensors.py``) and hand-compute the expected matmul.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from rcp_ndcg.data.mrl import (
    MrlHead,
    MrlProjection,
    _read_safetensors,
    clear_projection_cache,
    mrl_cut,
    projection_tensors,
)
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.support.identity import hash_payload, identity_payload
from tests._safetensors import write_safetensors


def _buffer(header: dict[str, object], payload: bytes = b"") -> bytes:
    """A safetensors buffer with a hand-built header (the reader's negative cases)."""
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw + payload


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

    def test_the_head_at_the_full_width_is_the_identity(self) -> None:
        """The identity selection (owner decision, 2026-10-09): a k equal to the observed width applies
        no head -- the vectors come back unchanged (no slice, no renormalisation, the same dtype)."""
        head = _head(dims=(1, 2))
        vectors = np.asarray([[3.0, 4.0]], dtype=np.float32)
        out = head.apply(vectors, 2)
        np.testing.assert_array_equal(out, vectors)
        assert out.dtype == vectors.dtype

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

    def test_a_missing_matrix_for_the_selected_dim_is_refused(self, tmp_path: Path) -> None:
        """The one convention: the chain is the tensors named by their target widths, so a file missing
        the k tensor is refused naming it, never silently returned."""
        source = write_safetensors(tmp_path / "projections.safetensors", {"4": np.eye(8, 4, dtype=np.float32)})
        head = _head(kind="projection", dims=(2, 4), projection=MrlProjection(source=str(source)))
        with pytest.raises(DataError, match="has no tensor") as caught:
            head.apply(np.ones((1, 8), dtype=np.float32), 2)
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

    def test_a_chain_that_does_not_end_at_k_is_refused(self, tmp_path: Path) -> None:
        """A tensor named for k whose output is not k-wide would mislabel every vector: refused, never
        silently returned as the k cut."""
        source = write_safetensors(tmp_path / "p.safetensors", {"2": np.eye(8, 5, dtype=np.float32)})
        head = _head(kind="projection", dims=(2,), projection=MrlProjection(source=str(source)))
        with pytest.raises(DataError, match="ends 5-wide|produce exactly k"):
            head.apply(np.ones((1, 8), dtype=np.float32), 2)

    def test_the_projection_spec_is_the_source_alone(self, tmp_path: Path) -> None:
        """One convention: the file's tensor names are the widths, and the declaration is just the source
        (the identity payload hashes it)."""
        source = write_safetensors(tmp_path / "p.safetensors", {"2": np.eye(4, 2, dtype=np.float32)})
        projection = MrlProjection(source=str(source))
        assert identity_payload(projection) == {"source": str(source)}
        assert len(hash_payload(identity_payload(projection))) == 64

    def test_the_chain_for_an_empty_declaration_is_refused(self) -> None:
        """A selection with no declared dimension at or above it has no chain: refused with a hint."""
        projection = MrlProjection(source="x.safetensors")
        with pytest.raises(ConfigError, match="mrl_dims") as caught:
            projection.chain_for(3, (2,))
        assert "mrl_dims" in (caught.value.hint or "")

    def test_the_projection_returns_float32_over_a_float16_store(self, tmp_path: Path) -> None:
        """The learned matrices are F32 and the chain accumulates in float32: a float16 input (a
        late-interaction store's transfer precision) still yields a float32 output."""
        source = write_safetensors(tmp_path / "p.safetensors", {"2": np.eye(4, 2, dtype=np.float32)})
        head = _head(kind="projection", dims=(2,), projection=MrlProjection(source=str(source)))
        vectors = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float16)
        projected = head.apply(vectors, 2)
        assert projected.dtype == np.float32
        expected = np.asarray([[1.0, 2.0]], dtype=np.float32)
        np.testing.assert_allclose(projected, expected / np.linalg.norm(expected), atol=1e-3)

    def test_the_truncation_cut_returns_float32_for_a_wider_input(self) -> None:
        """The cut keeps float16 (the store's transfer precision) and otherwise returns float32: the
        normalisation computes in float32, so a float64 input comes back float32."""
        head = _head(dims=(2,))
        cut = head.apply(np.asarray([[3.0, 4.0, 0.0]], dtype=np.float64), 2)
        assert cut.dtype == np.float32

    def test_the_reader_decodes_bfloat16(self, tmp_path: Path) -> None:
        """The projection files' BF16 tensors are the top 16 bits of a float32; the reader widens them."""
        matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        raw = (matrix.view(np.uint32) >> 16).astype("<u2")
        source = write_safetensors(tmp_path / "p.safetensors", {"2": raw})
        head = _head(kind="projection", dims=(2,), projection=MrlProjection(source=str(source)))
        vectors = np.asarray([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(head.apply(vectors, 2), [[0.6, 0.8]], atol=1e-6)

    @pytest.mark.parametrize(
        "payload",
        [
            b"",
            b"\x01\x02",
            b"\xff" * 8,
            struct.pack("<Q", 2) + b"!!",
        ],
        ids=["short", "truncated-header", "header-past-end", "not-json"],
    )
    def test_the_reader_refuses_a_broken_buffer(self, payload: bytes) -> None:
        with pytest.raises(DataError, match="safetensors|projection|JSON"):
            _read_safetensors(payload)

    def test_the_reader_refuses_an_unknown_dtype_and_out_of_range_offsets(self) -> None:
        with pytest.raises(DataError, match="dtype"):
            _read_safetensors(_buffer({"t": {"dtype": "F99", "shape": [1], "data_offsets": [0, 4]}}, b"\x00" * 4))
        with pytest.raises(DataError, match="offsets|bytes"):
            _read_safetensors(_buffer({"t": {"dtype": "F32", "shape": [1], "data_offsets": [0, 40]}}, b"\x00" * 4))

    def test_the_reader_refuses_a_header_that_is_not_an_object(self) -> None:
        raw = b"[]"
        payload = struct.pack("<Q", len(raw)) + raw
        with pytest.raises(DataError, match="JSON object"):
            _read_safetensors(payload)

    def test_the_reader_refuses_a_shape_that_does_not_match_its_span(self) -> None:
        """A span wider than the tensor's bytes silently reads the next tensor's bytes; refused."""
        header = {
            "t1": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]},
            "t2": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]},
        }
        with pytest.raises(DataError, match="bytes"):
            _read_safetensors(_buffer(header, b"\x00" * 8))
        with pytest.raises(DataError, match="shape"):
            _read_safetensors(_buffer({"t": {"dtype": "F32", "shape": ["1"], "data_offsets": [0, 4]}}, b"\x00" * 4))

    def test_the_projection_file_is_read_once_per_source(self, tmp_path: Path) -> None:
        source = write_safetensors(tmp_path / "p.safetensors", {"2": np.eye(2, 2, dtype=np.float32)})
        clear_projection_cache()
        _, first = projection_tensors(str(source))
        _, second = projection_tensors(str(source))
        assert first is second  # cached: the declared revision is immutable
        clear_projection_cache()
        _, third = projection_tensors(str(source))
        assert third is not first
