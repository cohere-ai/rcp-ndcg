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
        """2-D lists are the token_embed answer: one slice per item, in the transfer dtype."""
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

    def test_a_pooled_answer_is_refused_for_the_token_embed_task(self) -> None:
        """A7: one vector per item is a *pooled* task, and a pooled answer must not become a
        late-interaction index (it scored as a dense inner product, silently, before)."""
        adapter = VllmPooling()
        pooled_reply = Reply(status=200, body={"data": [{"index": 0, "data": [1.0, 0.0]}]}, headers={})

        with pytest.raises(ProviderError, match="answered a pooled task") as caught:
            adapter.interpret(request([Content.from_text("a")], dim=2), [pooled_reply])

        assert "per_chunk" in (caught.value.hint or "")

    def test_a_per_chunk_model_may_answer_one_vector_per_item(self) -> None:
        """The declared opt-out (``outputs: per_chunk``) keeps the old layout rule."""
        adapter = VllmPooling()
        pooled_reply = Reply(status=200, body={"data": [{"index": 0, "data": [1.0, 0.0]}]}, headers={})

        single = adapter.interpret(request([Content.from_text("a")], dim=2, outputs="per_chunk"), [pooled_reply])

        assert not single.is_multi_vector
        np.testing.assert_allclose(single.vectors, [[1.0, 0.0]])

    def test_a_non_finite_float_frame_is_refused(self) -> None:
        """A7: the /embeddings wire refuses a non-finite vector; the /pooling wire must too -- a NaN document
        otherwise vanishes from every top-k with no error."""
        adapter = VllmPooling()
        reply = Reply(status=200, body={"data": [{"index": 0, "data": [[1.0, 0.0], [float("nan"), 1.0]]}]}, headers={})

        with pytest.raises(ProviderError, match="non-finite") as caught:
            adapter.interpret(request([Content.from_text("a")], dim=2), [reply])

        assert "embeddings" in (caught.value.hint or "")

    def test_a_non_finite_base64_frame_is_refused(self) -> None:
        adapter = VllmPooling()
        frame = b64(np.array([[1.0, 0.0], [np.inf, 1.0]], dtype=np.float16))
        reply = Reply(status=200, body={"data": [{"index": 0, "data": frame}]}, headers={})

        with pytest.raises(ProviderError, match="non-finite"):
            adapter.interpret(request([Content.from_text("a")], dim=2), [reply])

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

    def test_a_frame_whose_bytes_do_not_even_fill_one_element_is_a_typed_error(self) -> None:
        """Seven float16 bytes hold no whole element: frombuffer raises before the dim check, and the
        refusal is the adapter's typed ProviderError, never a raw ValueError."""
        adapter = VllmPooling()
        frame = base64.b64encode(np.ones(7, dtype=np.uint8).tobytes()).decode("ascii")
        reply = Reply(status=200, body={"data": [{"index": 0, "data": frame}]}, headers={})
        with pytest.raises(ProviderError, match="does not decode"):
            adapter.interpret(request([Content.from_text("a")], dim=4), [reply])

    def test_bytes_framing_that_disagrees_with_the_frame_is_a_typed_error(self) -> None:
        adapter = VllmPooling()
        frame = np.ones((2, 2), dtype="<f2").tobytes()
        metadata = {
            "data": [
                {
                    "index": 0,
                    "embed_dtype": "float16",
                    "endianness": "little",
                    "start": 0,
                    "end": len(frame),
                    "shape": [3, 2],
                }
            ]
        }
        reply = Reply(200, frame, {"metadata": json.dumps(metadata)})
        with pytest.raises(ProviderError, match="framing metadata does not fit"):
            adapter.interpret(request([Content.from_text("a")]), [reply])

    def test_a_bytes_body_shorter_than_its_framing_is_a_typed_error(self) -> None:
        """A body truncated against its own metadata is a typed refusal, not a reshape ValueError."""
        adapter = VllmPooling()
        metadata = {
            "data": [
                {"index": 0, "embed_dtype": "float16", "endianness": "little", "start": 0, "end": 8, "shape": [2, 2]}
            ]
        }
        reply = Reply(200, np.ones(2, dtype="<f2").tobytes(), {"metadata": json.dumps(metadata)})
        with pytest.raises(ProviderError, match="framing metadata does not fit"):
            adapter.interpret(request([Content.from_text("a")]), [reply])

    def test_one_vector_items_of_unequal_width_are_a_typed_error(self) -> None:
        """A pooled reply whose vectors disagree in width is a refusal, not an np.stack ValueError (the
        per-chunk opt-out is the one request that may answer one vector per item)."""
        adapter = VllmPooling()
        reply = Reply(200, {"data": [{"index": 0, "data": [1.0, 0.0]}, {"index": 1, "data": [1.0]}]}, {})
        with pytest.raises(ProviderError, match="mixes one-vector and per-token items|different widths"):
            adapter.interpret(
                request([Content.from_text("a"), Content.from_text("b")], dim=2, outputs="per_chunk"), [reply]
            )

    def test_an_inhomogeneous_float_list_is_a_typed_error(self) -> None:
        """A jagged float list cannot be an array: a typed refusal, never a raw ValueError."""
        adapter = VllmPooling()
        reply = Reply(200, {"data": [{"index": 0, "data": [[1.0, 0.0], [1.0]]}]}, {})
        with pytest.raises(ProviderError, match="does not decode"):
            adapter.interpret(request([Content.from_text("a")], dim=2), [reply])

    def test_data_items_that_are_not_dicts_are_a_typed_error(self) -> None:
        """A reply whose data entries are floats or strings, not items, is a typed refusal."""
        adapter = VllmPooling()
        with pytest.raises(ProviderError, match="entries that are not items"):
            adapter.interpret(request([Content.from_text("a")], dim=2), [Reply(200, {"data": [1.0, 2.0]}, {})])
        metadata = json.dumps({"data": [1.0]})
        with pytest.raises(ProviderError, match="framing metadata is incomplete"):
            adapter.interpret(
                request([Content.from_text("a")], dim=2),
                [Reply(200, b"\x00" * 8, {"metadata": metadata})],
            )

    @pytest.mark.parametrize("bad", ["7", {"a": 1}, None])
    def test_a_malformed_usage_is_refused_not_coerced(self, bad: object) -> None:
        """The usage cross-check is load-bearing for the declared dim: a usage the reply cannot honestly
        report is a malformed reply, never a default that skips the check."""
        adapter = VllmPooling()
        frame = b64(np.ones((2, 2), dtype=np.float16))
        reply = Reply(200, {"data": [{"index": 0, "data": frame}], "usage": {"prompt_tokens": bad}}, {})
        with pytest.raises(ProviderError, match="usage"):
            adapter.interpret(request([Content.from_text("a")], dim=2), [reply])

    @pytest.mark.parametrize(("start", "end", "shape"), [(8, 0, [-2, 2]), (0, 8, [-1]), (2, 2, [2, 2])])
    def test_framing_that_slices_nothing_is_a_typed_error(self, start: int, end: int, shape: list[int]) -> None:
        """A negative shape or an inverted/empty byte range would decode to an empty result silently; it is
        refused instead."""
        adapter = VllmPooling()
        metadata = {
            "data": [
                {
                    "index": 0,
                    "embed_dtype": "float16",
                    "endianness": "little",
                    "start": start,
                    "end": end,
                    "shape": shape,
                }
            ]
        }
        reply = Reply(200, np.ones(4, dtype="<f2").tobytes(), {"metadata": json.dumps(metadata)})
        with pytest.raises(ProviderError, match="framing metadata"):
            adapter.interpret(request([Content.from_text("a")]), [reply])

    def test_a_reply_mixing_one_vector_and_per_token_items_is_refused(self) -> None:
        """The mixing guard holds whichever item came first: 1-D first, 2-D second is a refusal, not an
        np.stack crash."""
        adapter = VllmPooling()
        reply = Reply(
            200,
            {"data": [{"index": 0, "data": [1.0, 0.0]}, {"index": 1, "data": [[1.0, 0.0]]}]},
            {},
        )
        with pytest.raises(ProviderError, match="mixes one-vector and per-token items"):
            adapter.interpret(request([Content.from_text("a"), Content.from_text("b")], dim=2), [reply])

    def test_usage_reads_a_bytes_reply_s_metadata(self) -> None:
        adapter = VllmPooling()
        frame = np.ones((2, 2), dtype="<f2").tobytes()
        metadata = {
            "data": [
                {
                    "index": 0,
                    "embed_dtype": "float16",
                    "endianness": "little",
                    "start": 0,
                    "end": len(frame),
                    "shape": [2, 2],
                }
            ],
            "usage": {"prompt_tokens": 2},
        }
        reply = Reply(200, frame, {"metadata": json.dumps(metadata)})
        assert adapter.usage(reply) == TokenCount(input_tokens=2)

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

    def test_a_refusal_message_keeps_300_characters_and_drops_the_301st(self) -> None:
        """The declared message cap: the server's words are quoted up to 300 characters -- the 301st
        character must never surface in the error."""
        adapter = VllmPooling()
        reply = Reply(500, {"error": {"message": "a" * 300 + "!" + "b" * 20}}, {})
        with pytest.raises(RequestRejectedError) as caught:
            adapter.interpret(request([Content.from_text("a")], dim=2), [reply])
        assert "a" * 300 in str(caught.value) and "!" not in str(caught.value)


class TestDecodedTokenCounting:
    """The usage cross-check counts one vector per token: a 1-D item is one pooled vector, a 2-D item
    one vector per row -- the count that decides whether the reply's own usage is believed."""

    @pytest.mark.parametrize(
        ("data", "prompt_tokens", "multi", "offsets"),
        [
            pytest.param([[1.0, 2.0], [3.0, 4.0]], 2, False, None, id="one_vector_items_count_one_each"),
            pytest.param([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]], 3, True, [0, 2, 3], id="per_token_count_per_row"),
        ],
    )
    def test_the_usage_cross_check_counts_one_vector_per_token(
        self, data: list[Any], prompt_tokens: int, multi: bool, offsets: list[int] | None
    ) -> None:
        adapter = VllmPooling()
        items = [{"index": index, "data": item} for index, item in enumerate(data)]
        reply = Reply(200, {"data": items, "usage": {"prompt_tokens": prompt_tokens}}, {})
        embeddings = adapter.interpret(
            request([Content.from_text("a"), Content.from_text("b")], dim=2, outputs="per_chunk"), [reply]
        )
        assert embeddings.num_items == 2 and embeddings.is_multi_vector is multi
        if offsets is None:
            np.testing.assert_array_equal(embeddings.vectors, data)
        else:
            assert embeddings.offsets.tolist() == offsets


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


class TestPoolReplyCorners:
    """The pooling reply's framing corners are refused by name, never silently mis-read (the sweep's gap
    hunt): duplicated indices, a non-object metadata header, a usage without prompt_tokens, a mislabeled
    endianness."""

    @staticmethod
    def _request(dim: int = 2) -> PoolRequest:
        return PoolRequest(contents=(Content.from_text("x"),), role=EncodeRole.DOCUMENT, dim=dim)

    def test_duplicated_indices_are_refused(self) -> None:
        replies = [Reply(200, {"data": [{"index": 0, "data": [[1.0, 1.0]]}, {"index": 0, "data": [[2.0, 2.0]]}]}, {})]
        with pytest.raises(RequestRejectedError, match="exactly one int"):
            VllmPooling().interpret(self._request(), replies)

    def test_duplicated_indices_in_the_bytes_framing_are_refused(self) -> None:
        """The bytes framing realigns by the same rule as the JSON data: two items framed as index 0 are
        refused, never sorted into a silent misalignment."""
        frame = np.ones((2, 2), dtype="<f2").tobytes()
        half = len(frame) // 2
        item = {"embed_dtype": "float16", "endianness": "little", "shape": [1, 2]}
        metadata = {
            "data": [
                {**item, "index": 0, "start": 0, "end": half},
                {**item, "index": 0, "start": half, "end": len(frame)},
            ]
        }
        request = PoolRequest(
            contents=(Content.from_text("x"), Content.from_text("y")), role=EncodeRole.DOCUMENT, dim=2
        )
        with pytest.raises(RequestRejectedError, match="exactly one int"):
            VllmPooling().interpret(request, [Reply(200, frame, {"metadata": json.dumps(metadata)})])

    def test_an_index_on_some_entries_only_is_refused(self) -> None:
        replies = [Reply(200, {"data": [{"index": 0, "data": [[1.0, 1.0]]}, {"data": [[2.0, 2.0]]}]}, {})]
        with pytest.raises(RequestRejectedError, match="only some"):
            VllmPooling().interpret(self._request(), replies)

    def test_a_non_object_metadata_header_is_a_provider_error(self) -> None:
        """A re-serialising proxy can answer a framing header that is JSON but not an object: a typed
        refusal naming it, never a bare AttributeError."""
        metadata = json.dumps(["not", "an", "object"])
        with pytest.raises(ProviderError, match="not an object"):
            VllmPooling()._decode_bytes_reply(Reply(200, b"\x00\x01\x02\x03", {"metadata": metadata}), expected_items=1)

    def test_a_usage_without_prompt_tokens_refuses_the_check(self) -> None:
        """The load-bearing cross-check never switches itself off: a usage dict without prompt_tokens is a
        malformed reply (a gateway reshaping usage would otherwise switch off the only dim guard)."""
        reply = Reply(200, {"data": [{"index": 0, "data": [[1.0, 1.0]]}], "usage": {"completion_tokens": 3}}, {})
        with pytest.raises(ProviderError, match="prompt_tokens"):
            VllmPooling().interpret(self._request(), [reply])

    def test_an_unknown_endianness_is_refused_by_name(self) -> None:
        with pytest.raises(ProviderError, match="endianness"):
            # a base64 path never reaches the framing dtype, so drive the framed bytes corner directly
            VllmPooling()._decode_bytes_reply(
                Reply(
                    200,
                    b"\x00\x00\x00\x00\x00\x00\x00\x00",
                    {
                        "metadata": json.dumps(
                            {
                                "data": [
                                    {
                                        "index": 0,
                                        "embed_dtype": "float32",
                                        "endianness": "middle",
                                        "start": 0,
                                        "end": 8,
                                        "shape": [2, 1],
                                    }
                                ]
                            }
                        )
                    },
                ),
                expected_items=1,
            )
