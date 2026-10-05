"""The vLLM pooling adapter: the wire shapes, the ragged decode and its error paths.

Tested against an in-test sender over ``httpx.MockTransport`` (real request objects, canned replies), so the
adapter's :class:`~rcp_ndcg.inference.types.Call` bodies are really serialised and its ``interpret`` reads
what a server actually sends: base64 frames of the declared dtype, nested float lists, and the bytes framing
with its ``metadata`` header.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import numpy as np
import pytest
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import CapabilityError, ConfigError, ProviderError, RequestRejectedError
from rcp_ndcg.inference.adapters.pooling import VllmPooling
from rcp_ndcg.inference.types import EncodeRole, PoolRequest, Reply, TokenCount
from tests.inference._pooling import PoolingServer, RecordingSender, b64, request, send


def _encoded(vectors: dict[str, np.ndarray], **server_kwargs: Any) -> tuple[PoolingServer, RecordingSender]:
    from tests.inference._pooling import server_sender

    server = PoolingServer(vectors, **server_kwargs)
    return server, server_sender(server)


class TestRequestShape:
    def test_a_text_batch_is_one_request_with_the_briefs_fields(self) -> None:
        server, sender = _encoded({"a": np.ones((2, 2), dtype=np.float16)})
        adapter = VllmPooling()
        pool_request = request([Content.from_text("a")], dim=2)

        adapter.interpret(pool_request, send(sender, adapter.calls(pool_request, model="colbert")))
        assert len(sender.requests) == 1
        assert sender.requests[0].url.path == "/pooling"
        assert json.loads(sender.requests[0].content) == {
            "model": "colbert",
            "input": ["a"],
            "task": "token_embed",
            "encoding_format": "base64",
            "embed_dtype": "float16",
            "endianness": "little",
        }

    @pytest.mark.parametrize(("dtype", "on_the_wire"), [("float16", "float16"), ("float32", "float32")])
    def test_embed_dtype_is_sent_as_configured(self, dtype: str, on_the_wire: str) -> None:
        vectors = {"a": np.array([[0.5, -0.5], [1.0, 0.0]], dtype=np.float32)}
        server, sender = _encoded(vectors, encoding="float")
        adapter = VllmPooling()
        pool_request = request([Content.from_text("a")], embed_dtype=dtype, dim=2)
        adapter.interpret(pool_request, send(sender, adapter.calls(pool_request, model="m")))
        assert json.loads(sender.requests[0].content)["embed_dtype"] == on_the_wire

    def test_the_default_transfer_dtype_is_float16(self) -> None:
        """The owner's Q11 decision: the transfer precision defaults to float16; float32 is opt-in."""
        assert PoolRequest(contents=(), role=EncodeRole.QUERY).embed_dtype == "float16"

    def test_dimensions_is_never_sent(self) -> None:
        """vLLM's ``/pooling`` refuses `dimensions`; the declared dim is decode-side only."""
        server, sender = _encoded({"a": np.ones((2, 2), dtype=np.float16)})
        adapter = VllmPooling()
        pool_request = request([Content.from_text("a")], dim=2)
        adapter.interpret(pool_request, send(sender, adapter.calls(pool_request, model="m")))
        assert "dimensions" not in json.loads(sender.requests[0].content)

    def test_media_items_go_one_request_each_as_messages(self, tmp_path: Any) -> None:
        """Only the messages shape gets the model's chat template applied to image placeholders."""
        image = tmp_path / "page.png"
        image.write_bytes(_png_bytes())
        sender = RecordingSender(
            lambda _request: httpx.Response(
                200,
                json={
                    "data": [{"index": 0, "data": b64(np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float16))}],
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                },
            )
        )
        adapter = VllmPooling()
        pool_request = request([Content.from_image(image.as_uri()), Content.from_text("plain")], dim=2)
        embeddings = adapter.interpret(pool_request, send(sender, adapter.calls(pool_request, model="m")))

        assert len(sender.requests) == 2
        first = json.loads(sender.requests[0].content)
        assert first["messages"][0]["content"][0]["type"] == "image_url"
        assert first["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert first["task"] == "token_embed" and first["encoding_format"] == "base64"
        second = json.loads(sender.requests[1].content)
        assert second["messages"][0]["content"][0] == {"type": "text", "text": "plain"}
        assert embeddings.is_multi_vector and embeddings.num_items == 2
        assert embeddings.offsets is not None and embeddings.offsets.tolist() == [0, 2, 4]
        np.testing.assert_allclose(
            np.asarray(embeddings.vectors), [[1.0, 0.0], [0.5, 0.5], [1.0, 0.0], [0.5, 0.5]], rtol=1e-6
        )


class TestDecoding:
    def test_base64_float16_decodes_to_the_declared_shape_and_dtype(self) -> None:
        vectors = {"a": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float16)}
        server, sender = _encoded(vectors)
        adapter = VllmPooling()
        pool_request = request([Content.from_text("a")], dim=2)

        embeddings = adapter.interpret(pool_request, send(sender, adapter.calls(pool_request, model="m")))

        assert embeddings.is_multi_vector
        assert embeddings.vectors.dtype == np.float16
        assert embeddings.dim == 2
        assert embeddings.num_items == 1
        np.testing.assert_array_equal(embeddings.vectors, vectors["a"])

    def test_ragged_offsets(self) -> None:
        vectors = {
            "a": np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float16),
            "b": np.array([[0.5, 0.5]], dtype=np.float16),
            "c": np.zeros((0, 2), dtype=np.float16),
        }
        server, sender = _encoded(vectors)
        adapter = VllmPooling()
        pool_request = request([Content.from_text("a"), Content.from_text("b"), Content.from_text("c")], dim=2)

        embeddings = adapter.interpret(pool_request, send(sender, adapter.calls(pool_request, model="m")))
        assert embeddings.offsets is not None and embeddings.offsets.tolist() == [0, 3, 4, 4]
        np.testing.assert_array_equal(embeddings.vectors[3], [0.5, 0.5])

    def test_base64_float32_decodes_exactly(self) -> None:
        vectors = {"a": np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)}
        server, sender = _encoded(vectors)
        adapter = VllmPooling()
        pool_request = request([Content.from_text("a")], embed_dtype="float32", dim=2)

        embeddings = adapter.interpret(pool_request, send(sender, adapter.calls(pool_request, model="m")))
        assert embeddings.vectors.dtype == np.float32
        np.testing.assert_array_equal(embeddings.vectors, vectors["a"])

    def test_float_lists_are_accepted_with_the_shape_as_sent(self) -> None:
        """The layout of the answer decides: 2-D lists are ragged, 1-D lists are one vector per item."""
        adapter = VllmPooling()

        ragged_reply = Reply(
            status=200,
            body={"data": [{"index": 0, "data": [[1.0, 0.0], [0.0, 1.0]]}, {"index": 1, "data": [[0.5, 0.5]]}]},
            headers={},
        )
        embeddings = adapter.interpret(request([Content.from_text("a"), Content.from_text("b")], dim=2), [ragged_reply])
        assert embeddings.is_multi_vector
        assert embeddings.offsets is not None and embeddings.offsets.tolist() == [0, 2, 3]
        assert embeddings.vectors.dtype == np.float16  # stored in the transfer dtype

        pooled_reply = Reply(status=200, body={"data": [{"index": 0, "data": [1.0, 0.0]}]}, headers={})
        single = adapter.interpret(request([Content.from_text("a")], dim=2), [pooled_reply])
        assert not single.is_multi_vector
        np.testing.assert_allclose(single.vectors, [[1.0, 0.0]])

    def test_float_frame_width_must_match_the_declared_dim(self) -> None:
        adapter = VllmPooling()
        reply = Reply(status=200, body={"data": [{"index": 0, "data": [[1.0, 0.0, 0.5]]}]}, headers={})
        with pytest.raises(ProviderError, match="does not match the declared dim 2"):
            adapter.interpret(request([Content.from_text("a")], dim=2), [reply])

    def test_base64_with_no_declared_dim_is_a_config_error(self) -> None:
        adapter = VllmPooling()
        frame = b64(np.ones((2, 3), dtype=np.float16))
        reply = Reply(status=200, body={"data": [{"index": 0, "data": frame}]}, headers={})
        with pytest.raises(ConfigError, match="set dim on the pooling endpoint"):
            adapter.interpret(request([Content.from_text("a")], dim=None), [reply])

    def test_base64_frame_must_hold_whole_vectors(self) -> None:
        adapter = VllmPooling()
        frame = base64.b64encode(np.ones(7, dtype="<f2").tobytes()).decode("ascii")
        reply = Reply(status=200, body={"data": [{"index": 0, "data": frame}]}, headers={})
        with pytest.raises(ProviderError, match="not a multiple of the declared dim 4"):
            adapter.interpret(request([Content.from_text("a")], dim=4), [reply])

    def test_a_mistyped_dim_is_caught_by_the_reply_s_own_usage(self) -> None:
        """A token_embed answer has one vector per prompt token: the wrong width decodes the wrong count.

        Dim-4 frames decoded as width 4 hold 5 vectors where the reply's own ``usage.prompt_tokens`` reports
        10 -- a loud error instead of a silently mis-shaped corpus.
        """
        vectors = {"a": np.ones((4, 2), dtype=np.float16), "b": np.ones((6, 2), dtype=np.float16)}
        server = PoolingServer(vectors)
        from tests.inference._pooling import server_sender

        adapter = VllmPooling()
        pool_request = request([Content.from_text("a"), Content.from_text("b")], dim=4)
        with pytest.raises(ProviderError, match="reports 10 prompt token.*decode to 5"):
            adapter.interpret(pool_request, send(server_sender(server), adapter.calls(pool_request, model="m")))

    def test_the_usage_check_passes_when_the_dim_is_right(self) -> None:
        vectors = {"a": np.ones((4, 2), dtype=np.float16)}
        server, sender = _encoded(vectors)
        adapter = VllmPooling()
        pool_request = request([Content.from_text("a")], dim=2)
        embeddings = adapter.interpret(pool_request, send(sender, adapter.calls(pool_request, model="m")))
        assert embeddings.num_items == 1 and len(embeddings.vectors) == 4

    def test_response_order_is_restored_from_index(self) -> None:
        vectors = {"a": np.array([[1.0, 1.0]], dtype=np.float16), "b": np.array([[2.0, 2.0]], dtype=np.float16)}
        server, sender = _encoded(vectors, reverse=True)
        adapter = VllmPooling()
        pool_request = request([Content.from_text("a"), Content.from_text("b")], dim=2)

        embeddings = adapter.interpret(pool_request, send(sender, adapter.calls(pool_request, model="m")))
        np.testing.assert_array_equal(embeddings.vectors[:, 0], [1.0, 2.0])

    def test_a_missing_data_key_is_an_error(self) -> None:
        adapter = VllmPooling()
        with pytest.raises(ProviderError, match="no 'data'"):
            adapter.interpret(request([Content.from_text("a")], dim=2), [Reply(200, {"object": "list"}, {})])

    def test_an_item_count_mismatch_is_refused(self) -> None:
        adapter = VllmPooling()
        reply = Reply(200, {"data": [{"index": 0, "data": [[1.0, 0.0]]}]}, {})
        with pytest.raises(RequestRejectedError, match="returned 1 item\\(s\\) for 2 input"):
            adapter.interpret(request([Content.from_text("a"), Content.from_text("b")], dim=2), [reply])

    def test_a_reply_count_mismatch_is_an_error(self) -> None:
        adapter = VllmPooling()
        reply = Reply(200, {"data": [{"index": 0, "data": [[1.0, 0.0]]}]}, {})
        with pytest.raises(ProviderError, match="answered 2 reply"):
            adapter.interpret(request([Content.from_text("a")], dim=2), [reply, reply])

    def test_usage_reports_the_prompt_tokens(self) -> None:
        adapter = VllmPooling()
        reply = Reply(200, {"data": [], "usage": {"prompt_tokens": 7}}, {})
        assert adapter.usage(reply) == TokenCount(input_tokens=7)
        assert adapter.usage(Reply(200, {"data": []}, {})) is None

    def test_an_empty_batch_is_the_zero_item_value(self) -> None:
        adapter = VllmPooling()
        embeddings = adapter.interpret(request([]), [])
        assert embeddings.num_items == 0 and embeddings.is_multi_vector


class TestRefusals:
    def test_an_over_length_input_is_a_capability_error(self) -> None:
        adapter = VllmPooling()
        reply = Reply(
            400,
            {"error": {"message": "The prompt (length 9000) is longer than the maximum model length of 8192."}},
            {},
        )
        with pytest.raises(CapabilityError, match="longer than its context"):
            adapter.interpret(request([Content.from_text("a")], dim=2), [reply])

    def test_any_other_client_error_is_a_request_rejection(self) -> None:
        adapter = VllmPooling()
        reply = Reply(400, {"error": {"message": "Unsupported task: 'embed'"}}, {})
        with pytest.raises(RequestRejectedError, match="Unsupported task"):
            adapter.interpret(request([Content.from_text("a")], dim=2), [reply])


class TestBytesFraming:
    @staticmethod
    def _bytes_reply(vectors: np.ndarray, *, dtype: str = "float16") -> Reply:
        frame = np.asarray(vectors, dtype="<f2" if dtype == "float16" else "<f4").tobytes()
        metadata = {
            "data": [
                {
                    "index": 0,
                    "embed_dtype": dtype,
                    "endianness": "little",
                    "start": 0,
                    "end": len(frame),
                    "shape": list(np.asarray(vectors).shape),
                }
            ],
            "usage": {"prompt_tokens": int(np.asarray(vectors).shape[0])},
        }
        return Reply(200, frame, {"metadata": json.dumps(metadata)})

    def test_a_bytes_reply_decodes_through_the_framing_metadata_without_a_declared_dim(self) -> None:
        vectors = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
        adapter = VllmPooling()
        embeddings = adapter.interpret(request([Content.from_text("a")]), [self._bytes_reply(vectors)])
        assert embeddings.is_multi_vector
        np.testing.assert_array_equal(embeddings.vectors, vectors)

    def test_a_bytes_float32_reply_decodes_by_its_own_metadata(self) -> None:
        vectors = np.array([[0.25, -0.5]], dtype=np.float32)
        adapter = VllmPooling()
        embeddings = adapter.interpret(request([Content.from_text("a")]), [self._bytes_reply(vectors, dtype="float32")])
        np.testing.assert_array_equal(embeddings.vectors, vectors)

    def test_bytes_only_has_no_framing_and_is_refused_naming_lane_l7(self) -> None:
        adapter = VllmPooling()
        reply = Reply(200, np.ones((2, 2), dtype="<f2").tobytes(), {})
        with pytest.raises(NotImplementedError, match="lane L7"):
            adapter.interpret(request([Content.from_text("a")]), [reply])


def _png_bytes() -> bytes:
    """A 1x1 PNG, so the resolver has something real to base64."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()
