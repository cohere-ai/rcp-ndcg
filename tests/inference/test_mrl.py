"""The MRL config fields and the two clients' head application.

The declared kind and set are validated at load (a k outside the set, `dimensions` on a non-truncation
kind, `dimensions` beside `mrl_dim`, and a projection kind without its source are all refused naming the
field); the dense client cuts client-side exactly as the pooling client does, and every row the head
changed carries a ``mrl_cut`` :class:`~rcp_ndcg.data.text_budget.ProcessingRecord`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.inference.clients.embed import EmbeddingClient
from rcp_ndcg.inference.config import PoolingEndpoint
from rcp_ndcg.inference.types import Call, EncodeRole, Reply
from tests._safetensors import write_safetensors
from tests.inference import _budget
from tests.inference._embed import FakeSender, endpoint
from tests.inference._pooling import PoolingServer, server_sender

_SET: dict[str, Any] = {"mrl_kind": "truncation", "mrl_dims": (1, 2)}


def _vectors_handler(vectors: dict[str, list[float]]) -> Any:
    """A sender handler answering each text with the vector it maps to."""

    def one(call: Call) -> Reply:
        texts = call.json["input"]
        return Reply(
            200,
            {
                "data": [
                    {"index": index, "embedding": vectors.get(text, [1.0, 0.0])} for index, text in enumerate(texts)
                ]
            },
            {},
        )

    return one


class TestConfigValidation:
    def test_a_k_outside_the_declared_set_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="mrl_dim") as caught:
            endpoint(**_SET, mrl_dim=3)
        assert "mrl_dims" in (caught.value.hint or "")

    def test_dimensions_outside_the_declared_set_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="dimensions") as caught:
            endpoint(**_SET, dimensions=3)
        assert "mrl_dims" in (caught.value.hint or "")

    def test_dimensions_and_mrl_dim_together_are_refused(self) -> None:
        with pytest.raises(ConfigError, match="dimensions") as caught:
            endpoint(**_SET, dimensions=2, mrl_dim=1)
        assert "mrl_dim" in (caught.value.hint or "")

    def test_dimensions_needs_the_truncation_kind(self) -> None:
        with pytest.raises(ConfigError, match="mrl_kind") as caught:
            endpoint(dimensions=2)
        assert "truncation" in (caught.value.hint or "")

    def test_dimensions_is_refused_on_the_projection_kind(self, tmp_path: Path) -> None:
        source = write_safetensors(tmp_path / "p.safetensors", {"2": np.eye(4, 2, dtype=np.float32)})
        with pytest.raises(ConfigError, match="dimensions") as caught:
            endpoint(
                mrl_kind="projection",
                mrl_dims=(2,),
                mrl_projection={"source": str(source)},
                dimensions=2,
            )
        assert "projection" in (caught.value.hint or "") or "client-side" in (caught.value.hint or "")

    def test_a_declared_kind_needs_its_set(self) -> None:
        with pytest.raises(ConfigError, match="mrl_dims") as caught:
            endpoint(mrl_kind="truncation")
        assert "mrl_dims" in (caught.value.hint or "")

    def test_a_range_declares_the_selectable_interval(self) -> None:
        """A card whose prose gives a range ("from 32 to 1024") declares mrl_range instead of mrl_dims."""
        config = endpoint(mrl_kind="truncation", mrl_range=(2, 4), mrl_dim=3)
        assert config.mrl_dims is None and config.mrl_range == (2, 4)

    def test_a_k_outside_the_declared_range_is_refused(self) -> None:
        for k in (1, 5):
            with pytest.raises(ConfigError, match="mrl_range") as caught:
                endpoint(mrl_kind="truncation", mrl_range=(2, 4), mrl_dim=k)
            assert "mrl_range" in (caught.value.hint or "")

    def test_the_set_and_the_range_are_mutually_exclusive(self) -> None:
        with pytest.raises(ConfigError, match="mrl_dims") as caught:
            endpoint(mrl_kind="truncation", mrl_dims=(2,), mrl_range=(2, 4))
        assert "mrl_range" in (caught.value.hint or "")

    def test_a_range_needs_the_truncation_kind(self) -> None:
        with pytest.raises(ConfigError, match="mrl_range") as caught:
            endpoint(mrl_range=(2, 4))
        assert "mrl_kind" in (caught.value.hint or "")

    @pytest.mark.parametrize("mrl_range", [(0, 4), (4, 2)])
    def test_a_range_must_be_positive_and_ordered(self, mrl_range: tuple[int, int]) -> None:
        with pytest.raises(ConfigError, match="mrl_range"):
            endpoint(mrl_kind="truncation", mrl_range=mrl_range)

    def test_the_projection_kind_needs_the_discrete_set(self, tmp_path: Path) -> None:
        source = write_safetensors(tmp_path / "p.safetensors", {"2": np.eye(4, 2, dtype=np.float32)})
        with pytest.raises(ConfigError, match="mrl_dims") as caught:
            endpoint(mrl_kind="projection", mrl_range=(2, 4), mrl_projection={"source": str(source)})
        assert "mrl_range" in (caught.value.hint or "")

    def test_a_k_without_a_declared_kind_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="mrl_kind") as caught:
            endpoint(mrl_dim=2)
        assert "mrl_kind" in (caught.value.hint or "")

    def test_a_truncation_cut_without_a_declared_kind_is_refused(self) -> None:
        """M7 (pre/post-processing review): the legacy cut paths -- the dense and pooling `mrl_dim` and the
        engine-side `dimensions` -- cannot slice a projection-kind checkpoint as truncation: every cut now
        needs a declared `mrl_kind`, so a checkpoint whose card says "learned projections" can never be
        silently cut."""
        with pytest.raises(ConfigError, match="mrl_kind"):
            endpoint(mrl_dim=2)
        with pytest.raises(ConfigError, match="mrl_kind"):
            endpoint(dimensions=2)
        with pytest.raises(ConfigError, match="mrl_kind"):
            endpoint(mrl_kind="none", mrl_dim=2)
        with pytest.raises(ConfigError, match="mrl_kind"):
            PoolingEndpoint(
                base_url="http://a:8000/v1",
                model="colqwen",
                dim=8,
                mrl_dim=2,
                tokenizer=_budget.DEFAULT_TOKENIZER or "test/tokenizer",
                max_tokens=8192,
            )
        # The pooling `dimensions` field stays refused as inert (the route has no such field).
        with pytest.raises(ConfigError, match="dimensions"):
            PoolingEndpoint(
                base_url="http://a:8000/v1",
                model="colqwen",
                dim=8,
                dimensions=2,
                tokenizer=_budget.DEFAULT_TOKENIZER or "test/tokenizer",
                max_tokens=8192,
            )

    def test_the_projection_kind_needs_its_source(self) -> None:
        with pytest.raises(ConfigError, match="mrl_projection") as caught:
            endpoint(mrl_kind="projection", mrl_dims=(2,))
        assert "mrl_projection" in (caught.value.hint or "")

    def test_a_projection_source_needs_the_projection_kind(self, tmp_path: Path) -> None:
        source = write_safetensors(tmp_path / "p.safetensors", {"2": np.eye(4, 2, dtype=np.float32)})
        with pytest.raises(ConfigError, match="mrl_projection") as caught:
            endpoint(**_SET, mrl_projection={"source": str(source)})
        assert "mrl_kind" in (caught.value.hint or "")

    @pytest.mark.parametrize("dims", [(), (2, 2), (0,), (-1,)])
    def test_the_set_must_be_positive_and_unique(self, dims: tuple[int, ...]) -> None:
        with pytest.raises(ConfigError, match="mrl_dims"):
            endpoint(mrl_kind="truncation", mrl_dims=dims)

    def test_the_pooling_wire_still_refuses_dimensions(self) -> None:
        with pytest.raises(ConfigError, match="dimensions") as caught:
            PoolingEndpoint(
                base_url="http://a:8000/v1", model="colqwen", dim=128, dimensions=32, tokenizer="t", max_tokens=8192
            )
        assert "drop dimensions" in (caught.value.hint or "")

    def test_a_pooling_k_without_a_set_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="mrl_kind") as caught:
            PoolingEndpoint(
                base_url="http://a:8000/v1", model="colqwen", dim=4, mrl_dim=2, tokenizer="t", max_tokens=8192
            )
        assert "mrl_kind" in (caught.value.hint or "")


class TestDenseClient:
    def test_the_client_selects_inside_a_range(self) -> None:
        sender = FakeSender(_vectors_handler({"a": [1.0, 2.0, 3.0, 4.0]}))
        client = EmbeddingClient(endpoint(mrl_kind="truncation", mrl_range=(1, 3), mrl_dim=2), sender=sender)

        vectors = client.encode([Content.from_text("a")], EncodeRole.DOCUMENT)

        assert vectors.as_matrix().shape == (1, 2)
        assert client.processing[0].full_width == 4

    def test_the_client_cut_is_slice_then_renormalise(self) -> None:
        """[3, 4] cut to one dimension renormalises to [1.0]; a cut after normalisation would ship [0.6]."""
        sender = FakeSender(_vectors_handler({"a": [3.0, 4.0]}))
        client = EmbeddingClient(endpoint(**_SET, mrl_dim=1), sender=sender)

        vectors = client.encode([Content.from_text("a")], EncodeRole.DOCUMENT)

        np.testing.assert_allclose(vectors.as_matrix(), [[1.0]], atol=1e-6)

    def test_the_cut_renormalises_whatever_normalize_says(self) -> None:
        sender = FakeSender(_vectors_handler({"a": [3.0, 4.0]}))
        client = EmbeddingClient(endpoint(**_SET, mrl_dim=1, normalize=False), sender=sender)

        vectors = client.encode([Content.from_text("a")], EncodeRole.DOCUMENT)

        np.testing.assert_allclose(vectors.as_matrix(), [[1.0]], atol=1e-6)

    def test_the_engine_side_dimensions_path_is_untouched(self) -> None:
        """``dimensions`` still travels in the request and the client cuts nothing: the fake answers a
        two-wide vector for a four-wide endpoint and the reply is kept as is."""
        sender = FakeSender(_vectors_handler({"a": [3.0, 4.0]}))
        client = EmbeddingClient(endpoint(**_SET, dimensions=2), sender=sender)

        vectors = client.encode([Content.from_text("a")], EncodeRole.DOCUMENT)

        assert sender.calls[0].json["dimensions"] == 2
        assert vectors.as_matrix().shape == (1, 2)
        assert client.processing == []

    def test_the_cut_records_one_processing_record_per_row(self) -> None:
        sender = FakeSender(_vectors_handler({"a": [3.0, 4.0], "b": [0.0, 1.0]}))
        client = EmbeddingClient(endpoint(**_SET, mrl_dim=1), sender=sender)

        client.encode([Content.from_text("a"), Content.from_text("b")], EncodeRole.DOCUMENT)

        assert [(row.input_id, row.shape, row.mechanisms) for row in client.processing] == [
            ("0", "document", ("mrl_cut",)),
            ("1", "document", ("mrl_cut",)),
        ]
        first = client.processing[0]
        assert (first.mrl_kind, first.mrl_dim, first.full_width) == ("truncation", 1, 2)

    def test_no_cut_no_record(self) -> None:
        sender = FakeSender(_vectors_handler({"a": [3.0, 4.0]}))
        client = EmbeddingClient(endpoint(), sender=sender)

        client.encode([Content.from_text("a")], EncodeRole.DOCUMENT)

        assert client.processing == []


class TestPoolingClientRecord:
    def test_the_cut_records_one_processing_record_per_item(self) -> None:
        sender = server_sender(PoolingServer({}, default=np.asarray([[3.0, 4.0]], dtype=np.float16)))
        client = PoolingEndpoint(
            model="colbert",
            base_url="http://engine:8000/v1",
            dim=2,
            mrl_dim=1,
            mrl_kind="truncation",
            mrl_dims=(1,),
            normalize=True,
            tokenizer=_budget.DEFAULT_TOKENIZER or "test/tokenizer",
            max_tokens=8192,
        )
        from rcp_ndcg.inference.clients.pool import PoolingClient

        pooled = PoolingClient(client, sender=sender)
        pooled.encode([Content.from_text("a")], EncodeRole.DOCUMENT)

        assert [(row.input_id, row.shape, row.mechanisms) for row in pooled.processing] == [
            ("0", "document", ("mrl_cut",)),
        ]
        assert (pooled.processing[0].mrl_kind, pooled.processing[0].mrl_dim, pooled.processing[0].full_width) == (
            "truncation",
            1,
            2,
        )


class TestProjectionClient:
    def test_the_client_applies_the_checkpoint_matrix(self, tmp_path: Path) -> None:
        matrix = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32)
        source = write_safetensors(tmp_path / "projections.safetensors", {"2": matrix})
        sender = FakeSender(_vectors_handler({"a": [1.0, 2.0, 3.0, 4.0]}))
        client = EmbeddingClient(
            endpoint(
                mrl_kind="projection",
                mrl_dims=(2,),
                mrl_projection={"source": str(source)},
                mrl_dim=2,
            ),
            sender=sender,
        )

        vectors = client.encode([Content.from_text("a")], EncodeRole.DOCUMENT)

        expected = np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32) @ matrix
        expected = expected / np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(vectors.as_matrix(), expected, atol=1e-6)
        assert client.processing[0].mrl_kind == "projection"
