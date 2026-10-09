"""The pooling role client: prompts per role, batching, concurrency, normalisation and the transfer dtype.

The client is tested through its wire adapter against the in-test pooling server
(:mod:`tests.inference._pooling`), so what is asserted is what a run would see: the requests that go out and
the ragged vectors that come back, in the transfer dtype end to end.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np
import pytest
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import ConfigError, ProviderError, RequestRejectedError
from rcp_ndcg.inference.clients.pool import PoolingClient
from rcp_ndcg.inference.config import PoolingEndpoint
from rcp_ndcg.inference.types import EncodeRole
from tests.inference import _budget
from tests.inference._pooling import PoolingServer, RecordingSender, server_sender


class _GatedSender:
    """A sender that overlaps its sends, counts the peak in flight and sleeps a tick per request."""

    def __init__(self, server: PoolingServer) -> None:
        self._inner = server_sender(server)
        self.in_flight = 0
        self.peak = 0
        self.sent: list[list[dict[str, Any]]] = []

    async def send(self, calls: Any) -> list[Any]:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        self.sent.append([call.json for call in calls])
        await asyncio.sleep(0)
        try:
            return await self._inner.send(calls)
        finally:
            self.in_flight -= 1

    def run(self, coroutine: Any) -> Any:
        return asyncio.run(coroutine)


def _client(sender: Any, **config: Any) -> PoolingClient:
    settings: dict[str, Any] = {
        "model": "colbert",
        "base_url": "http://engine:8000/v1",
        "dim": 2,
        "normalize": False,
        "tokenizer": _budget.DEFAULT_TOKENIZER or "test/tokenizer",
        "max_tokens": 8192,
        "image_policy": {"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
        "max_images": 4,
    }
    settings.update(config)
    return PoolingClient(PoolingEndpoint(**settings), sender=sender)


class TestEncode:
    def test_batches_are_reassembled_in_input_order(self) -> None:
        vectors = {f"doc{i}": np.full((i + 1, 2), i, dtype=np.float16) for i in range(5)}
        sender = _GatedSender(PoolingServer(vectors))
        client = _client(sender, batch_size=2)

        embeddings = asyncio.run(client.aencode([Content.from_text(f"doc{i}") for i in range(5)], EncodeRole.DOCUMENT))

        assert embeddings.num_items == 5
        assert embeddings.offsets is not None and embeddings.offsets.tolist() == [0, 1, 3, 6, 10, 15]
        np.testing.assert_array_equal(embeddings.vectors[0], [0, 0])
        np.testing.assert_array_equal(embeddings.vectors[14], [4, 4])
        assert len(sender.sent) == 3

    def test_batch_size_argument_overrides_the_config(self) -> None:
        vectors = {f"d{i}": np.ones((1, 2), dtype=np.float16) for i in range(3)}
        sender = _GatedSender(PoolingServer(vectors))
        client = _client(sender, batch_size=3)

        asyncio.run(client.aencode([Content.from_text(f"d{i}") for i in range(3)], EncodeRole.DOCUMENT))
        assert len(sender.sent) == 1
        asyncio.run(client.aencode([Content.from_text(f"d{i}") for i in range(3)], EncodeRole.DOCUMENT, batch_size=1))
        assert len(sender.sent) == 4

    def test_a_served_wire_s_declared_cap_never_refuses_a_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The batch cap is a HOSTED profile's fact on every role (one rule, the embed role's): a served
        /pooling engine answers its own over-count refusal, so a cap declared on a served wire refuses
        nothing client-side -- the batch goes out whole."""
        from rcp_ndcg.inference.adapters.pooling import VllmPooling

        monkeypatch.setattr(VllmPooling, "MAX_BATCH", 2)
        assert VllmPooling.HOSTED is False
        vectors = {f"d{i}": np.ones((1, 2), dtype=np.float16) for i in range(4)}
        sender = _GatedSender(PoolingServer(vectors))
        client = _client(sender, batch_size=4)

        asyncio.run(client.aencode([Content.from_text(f"d{i}") for i in range(4)], EncodeRole.DOCUMENT))

        assert len(sender.sent) == 1

    def test_prompts_apply_per_role(self) -> None:
        sender = _GatedSender(PoolingServer({}, default=np.ones((1, 2), dtype=np.float16)))
        client = _client(sender, query_prompt="Query: ", doc_prompt="Document: ")

        asyncio.run(client.aencode([Content.from_text("q")], EncodeRole.QUERY))
        asyncio.run(client.aencode([Content.from_text("d")], EncodeRole.DOCUMENT))

        first, second = sender.sent[0][0], sender.sent[1][0]
        assert first["input"] == ["Query: q"]
        assert second["input"] == ["Document: d"]

    def test_media_items_travel_as_messages_through_the_client(self, tmp_path: Any) -> None:
        """The adapter decides the shapes; the client prepares the media (one call, tokens counted) and
        splits and reassembles."""
        image = tmp_path / "page.png"
        image.write_bytes(_png_bytes())
        sender = _GatedSender(
            PoolingServer({"plain": np.ones((1, 2), dtype=np.float16)}, media_vector=np.ones((1, 2), dtype=np.float16))
        )
        client = _client(sender, image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"})

        embeddings = asyncio.run(
            client.aencode([Content.from_image(image.as_uri()), Content.from_text("plain")], EncodeRole.DOCUMENT)
        )

        assert len(sender.sent[0]) == 2  # a media batch goes one request per item, in one send
        assert "messages" in sender.sent[0][0]
        assert embeddings.num_items == 2

    def test_normalize_unit_norms_every_token_and_keeps_the_transfer_dtype(self) -> None:
        sender = _GatedSender(PoolingServer({}, default=np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float16)))
        client = _client(sender, normalize=True)

        embeddings = asyncio.run(client.aencode([Content.from_text("a")], EncodeRole.DOCUMENT))

        assert embeddings.vectors.dtype == np.float16
        norms = np.linalg.norm(embeddings.vectors.astype(np.float32), axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-3)

    def test_normalize_false_keeps_the_vectors_as_decoded(self) -> None:
        sender = _GatedSender(PoolingServer({}, default=np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float16)))
        client = _client(sender, normalize=False)

        embeddings = asyncio.run(client.aencode([Content.from_text("a")], EncodeRole.DOCUMENT))
        np.testing.assert_allclose(np.asarray(embeddings.vectors), [[3.0, 4.0], [1.0, 0.0]], rtol=1e-3)

    def test_the_mrl_cut_is_slice_then_renormalise(self) -> None:
        """The card's order (2g, hand-computed): the model's vector [3, 4] (unit norm [0.6, 0.8]) cut to one
        dimension renormalises to [1.0]; a cut after the normalisation would ship [0.6] -- wrong."""
        sender = _GatedSender(PoolingServer({}, default=np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float16)))
        client = _client(sender, normalize=True, mrl_dim=1, mrl_kind="truncation", mrl_dims=(1, 2))

        embeddings = asyncio.run(client.aencode([Content.from_text("a")], EncodeRole.DOCUMENT))
        np.testing.assert_allclose(np.asarray(embeddings.vectors, dtype=np.float32), [[1.0], [1.0]], atol=1e-3)

    def test_the_mrl_cut_renormalises_whatever_normalize_says(self) -> None:
        """The cut destroys unit-ness, so it renormalises even when the config's ``normalize`` is false (the
        field governs the uncut vectors): [3, 4] cut to one dimension ships [1.0], not the un-normalised
        [3.0]."""
        sender = _GatedSender(PoolingServer({}, default=np.array([[3.0, 4.0]], dtype=np.float16)))
        client = _client(sender, normalize=False, mrl_dim=1, mrl_kind="truncation", mrl_dims=(1, 2))

        embeddings = asyncio.run(client.aencode([Content.from_text("a")], EncodeRole.DOCUMENT))
        np.testing.assert_allclose(np.asarray(embeddings.vectors, dtype=np.float32), [[1.0]], atol=1e-3)

    def test_an_mrl_dim_at_or_over_dim_is_refused_at_the_config(self) -> None:
        with pytest.raises(ConfigError, match="mrl_dim") as caught:
            PoolingEndpoint(
                base_url="http://engine:8000/v1",
                model="colqwen",
                dim=2,
                tokenizer=_budget.DEFAULT_TOKENIZER,
                max_tokens=8192,
                mrl_dim=2,
                mrl_kind="truncation",
                mrl_dims=(1, 2),
            )
        assert "mrl_dim" in (caught.value.hint or ""), "the refusal names the field to change"

    def test_the_declared_dim_shapes_the_decode(self) -> None:
        """The config's dim rebuilds (tokens, dim) from the flat frame: 8 values at dim 4 are two vectors."""
        vectors = {"a": np.arange(8, dtype=np.float16).reshape(2, 4)}
        server = PoolingServer(vectors, usage=False)
        sender = _GatedSender(server)
        client = PoolingClient(
            PoolingEndpoint(
                model="m",
                base_url="http://engine:8000/v1",
                dim=4,
                normalize=False,
                tokenizer=_budget.DEFAULT_TOKENIZER,
                max_tokens=8192,
            ),
            sender=sender,
        )

        embeddings = asyncio.run(client.aencode([Content.from_text("a")], EncodeRole.DOCUMENT))
        assert embeddings.vectors.shape == (2, 4)
        np.testing.assert_array_equal(embeddings.vectors, np.arange(8, dtype=np.float16).reshape(2, 4))

    def test_an_empty_batch_sends_nothing(self) -> None:
        sender = _GatedSender(PoolingServer({}))
        client = _client(sender)

        embeddings = asyncio.run(client.aencode([], EncodeRole.QUERY))

        assert sender.sent == []
        assert embeddings.num_items == 0 and embeddings.is_multi_vector

    def test_the_sync_encode_matches_the_async_one(self) -> None:
        vectors = {"a": np.ones((2, 2), dtype=np.float16), "b": np.zeros((1, 2), dtype=np.float16)}
        sender = _GatedSender(PoolingServer(vectors))
        client = _client(sender, batch_size=1)
        contents = [Content.from_text("a"), Content.from_text("b")]

        sync_embeddings = client.encode(contents, EncodeRole.DOCUMENT)
        async_embeddings = asyncio.run(client.aencode(contents, EncodeRole.DOCUMENT))
        np.testing.assert_array_equal(sync_embeddings.vectors, async_embeddings.vectors)
        np.testing.assert_array_equal(sync_embeddings.offsets, async_embeddings.offsets)


class TestConcurrency:
    def test_batch_requests_overlap_up_to_the_configured_concurrency(self) -> None:
        vectors = {f"d{i}": np.ones((1, 2), dtype=np.float16) for i in range(6)}
        sender = _GatedSender(PoolingServer(vectors))
        client = PoolingClient(
            PoolingEndpoint(
                model="m",
                base_url="http://engine:8000/v1",
                dim=2,
                batch_size=1,
                concurrency=2,
                tokenizer=_budget.DEFAULT_TOKENIZER,
                max_tokens=8192,
            ),
            sender=sender,
        )

        embeddings = asyncio.run(client.aencode([Content.from_text(f"d{i}") for i in range(6)], EncodeRole.DOCUMENT))

        assert 1 < sender.peak <= 2  # overlapping, and gated by the config's concurrency
        assert embeddings.num_items == 6
        assert embeddings.offsets is not None and embeddings.offsets.tolist() == [0, 1, 2, 3, 4, 5, 6]

    def test_concurrency_one_serialises(self) -> None:
        vectors = {f"d{i}": np.ones((1, 2), dtype=np.float16) for i in range(3)}
        sender = _GatedSender(PoolingServer(vectors))
        client = PoolingClient(
            PoolingEndpoint(
                model="m",
                base_url="http://engine:8000/v1",
                dim=2,
                batch_size=1,
                concurrency=1,
                tokenizer=_budget.DEFAULT_TOKENIZER,
                max_tokens=8192,
            ),
            sender=sender,
        )

        asyncio.run(client.aencode([Content.from_text(f"d{i}") for i in range(3)], EncodeRole.DOCUMENT))
        assert sender.peak == 1


class TestRefusals:
    def test_max_tokens_is_the_budget_the_client_fits_to(self) -> None:
        """The text-budget mechanism is wired: a declared budget is the item fit's (test_client_budget.py
        pins the cuts and the census); a client is built."""
        client = PoolingClient(
            PoolingEndpoint(
                model="colbert",
                base_url="http://engine:8000/v1",
                dim=2,
                tokenizer=_budget.DEFAULT_TOKENIZER,
                max_tokens=8192,
            ),
            sender=_GatedSender(PoolingServer({})),
        )
        assert client.config.max_tokens == 8192

    def test_an_unaligned_answer_is_refused(self) -> None:
        """One item for two inputs is a refusal from the adapter, never a short result."""
        from tests.inference._pooling import RecordingSender, b64

        sender = RecordingSender(
            lambda _request: httpx.Response(
                200,
                json={
                    "data": [{"index": 0, "data": b64(np.ones((1, 2), dtype=np.float16))}],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )
        )
        client = _client(sender)

        with pytest.raises(RequestRejectedError, match="returned 1 item\\(s\\) for 2 input"):
            asyncio.run(client.aencode([Content.from_text("a"), Content.from_text("ghost")], EncodeRole.DOCUMENT))


def _png_bytes() -> bytes:
    """A 1x1 PNG, so a media content resolves."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestDocumentSkipIds:
    """``document_skip_token_ids`` (2, the topk hand-off): the pooling client drops document vectors at the
    positions whose token id is listed (the reference drops punctuation/special vectors before MaxSim;
    queries keep all their vectors); a count mismatch between the sent ids and the returned vectors is a
    typed error, never a silent misalignment."""

    @staticmethod
    def _client(sender: Any, **config: Any) -> PoolingClient:
        settings: dict[str, Any] = {
            "base_url": "http://engine:8000/v1",
            "model": "colqwen",
            "dim": 2,
            "normalize": False,
            "tokenizer": _budget.DEFAULT_TOKENIZER,
            "max_tokens": 8192,
            "document_skip_token_ids": (2,),  # the word-level fixture's id of 'a'
        }
        settings.update(config)
        return PoolingClient(PoolingEndpoint(**settings), sender=sender)

    def test_document_vectors_are_dropped_at_the_skip_positions(self) -> None:
        from tests._tokenizers import word_tokenizer

        words = word_tokenizer()
        text = "the a of to"  # four word tokens; ids [1, 2, 3, 4]
        assert words.ids(text) == [1, 2, 3, 4]
        sender = _GatedSender(PoolingServer({}, default=np.ones((4, 2), dtype=np.float16)))
        client = self._client(sender)
        embeddings = asyncio.run(client.aencode([Content.from_text(text)], EncodeRole.DOCUMENT))
        assert embeddings.num_items == 1
        assert embeddings.offsets is not None and embeddings.offsets.tolist() == [0, 3]  # 'a' dropped
        np.testing.assert_array_equal(embeddings.vectors[0], [1.0, 1.0])  # position 0 ('the') kept
        np.testing.assert_array_equal(embeddings.vectors[2], [1.0, 1.0])  # position 2 ('of') kept

    def test_query_vectors_are_all_kept(self) -> None:
        from tests._tokenizers import word_tokenizer

        text = "the a of to"
        assert word_tokenizer().count(text) == 4
        sender = _GatedSender(PoolingServer({}, default=np.ones((4, 2), dtype=np.float16)))
        client = self._client(sender)
        embeddings = asyncio.run(client.aencode([Content.from_text(text)], EncodeRole.QUERY))
        assert embeddings.offsets is not None and embeddings.offsets.tolist() == [0, 4]

    def test_a_count_mismatch_between_sent_ids_and_vectors_is_a_typed_error(self) -> None:
        """The engine returned five token vectors where the client sent four ids: refusing beats a silent
        misalignment (the skip positions would drop the wrong tokens)."""
        from rcp_ndcg.errors import ProviderError

        sender = _GatedSender(PoolingServer({}, default=np.ones((5, 2), dtype=np.float16)))
        client = self._client(sender)
        with pytest.raises(ProviderError, match=r"5 token vector\(s\).*4 token id"):
            asyncio.run(client.aencode([Content.from_text("the a of to")], EncodeRole.DOCUMENT))

    def test_media_documents_are_sent_and_their_positions_are_never_skipped(self, tmp_path: Any) -> None:
        """The skip rule at image positions: a media document rides the messages route and the client
        keeps every returned vector (a media request's positions are the server's chat-template render,
        which the client cannot tokenise -- the image positions are exempt, never silently unskipped),
        and the deviation is on the row's record."""
        image = tmp_path / "page.png"
        image.write_bytes(_png_bytes())
        sender = _GatedSender(
            PoolingServer({}, default=np.ones((1, 2), dtype=np.float16), media_vector=np.ones((1, 2), dtype=np.float16))
        )
        client = self._client(
            sender,
            image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
            max_images=4,
        )
        embeddings = asyncio.run(client.aencode([Content.from_image(image.as_uri())], EncodeRole.DOCUMENT))
        assert sender.sent, "the media document is sent (the image positions are exempt from the skip)"
        assert embeddings.num_items == 1
        assert embeddings.offsets is not None and embeddings.offsets.tolist() == [0, 1]
        records = [record for record in client.processing if record.changed]
        assert [record.mechanisms for record in records] == [("skip_unapplied",)], "never silently unskipped"

    def test_a_batch_mixing_text_and_media_documents_is_refused_upfront(self, tmp_path: Any) -> None:
        """A media item routes the whole batch through the messages wire, whose positions are the engine's
        chat-template render: the text documents' skip positions cannot be found there. Refused before
        anything is sent (a media document alone, or a text-only batch, is the honest shape)."""
        from rcp_ndcg.errors import CapabilityError

        image = tmp_path / "page.png"
        image.write_bytes(_png_bytes())
        sender = _GatedSender(
            PoolingServer({}, default=np.ones((4, 2), dtype=np.float16), media_vector=np.ones((4, 2), dtype=np.float16))
        )
        client = self._client(
            sender,
            image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
            max_images=4,
        )
        with pytest.raises(CapabilityError, match="mixes text-only"):
            asyncio.run(
                client.aencode(
                    [Content.from_image(image.as_uri()), Content.from_text("the a of to")], EncodeRole.DOCUMENT
                )
            )
        assert sender.sent == [], "refused before anything is sent"

    def test_a_captioned_media_document_is_sent_under_skip_ids(self, tmp_path: Any) -> None:
        """A media document whose text part carries a caption rides alone through the messages route (one
        item per call): its vectors are kept whole and the deviation is on the row's record."""
        from rcp_ndcg_core.content import ImagePart, MediaRef, TextPart

        image = tmp_path / "page.png"
        image.write_bytes(_png_bytes())
        sender = _GatedSender(
            PoolingServer({}, default=np.ones((4, 2), dtype=np.float16), media_vector=np.ones((4, 2), dtype=np.float16))
        )
        client = self._client(
            sender,
            image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
            max_images=4,
        )
        captioned = Content.from_parts([TextPart(text="a caption"), ImagePart(ref=MediaRef(uri=image.as_uri()))])
        embeddings = asyncio.run(client.aencode([captioned], EncodeRole.DOCUMENT))
        assert sender.sent, "sent: a media-only batch has no text items to misalign"
        assert embeddings.num_items == 1
        records = [record for record in client.processing if record.changed]
        assert [record.mechanisms for record in records] == [("skip_unapplied",)]

    def test_token_ids_travel_as_the_input_and_skip_ids_still_apply(self) -> None:
        """3 (pplx): the pooling wire sends the ids the fit tokenised, and the document skip drops the same
        positions (the ids sent are the ids checked)."""
        from tests._tokenizers import word_tokenizer

        text = "the a of to"  # four word tokens; the skip drops 'a' (id 2)
        sender = _GatedSender(PoolingServer({}, default=np.ones((4, 2), dtype=np.float16)))
        client = self._client(sender, request_shape="token_ids")

        embeddings = asyncio.run(client.aencode([Content.from_text(text)], EncodeRole.DOCUMENT))

        sent = sender.sent[0][0]["input"]
        assert sent == [word_tokenizer().ids(text)]
        assert embeddings.offsets is not None and embeddings.offsets.tolist() == [0, 3]


class TestEngineSideSkip:
    """``document_skip_engine_side``: the served plugin drops the rule's positions before the reply, so the
    wire carries only the kept vectors and the client checks the declared count instead of slicing. A reply
    whose count disagrees is a typed error -- the engine's ``usage.prompt_tokens`` counts the *prompt*, so a
    reply that ignored the rule cannot hide behind it."""

    @staticmethod
    def _client(sender: Any, **config: Any) -> PoolingClient:
        settings: dict[str, Any] = {
            "base_url": "http://engine:8000/v1",
            "model": "pplx-late",
            "dim": 2,
            "normalize": False,
            "tokenizer": _budget.DEFAULT_TOKENIZER,
            "max_tokens": 8192,
            "document_skip_token_ids": (2,),  # the word-level fixture's id of 'a'
            "document_skip_engine_side": True,
            "image_policy": {"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
            "max_images": 4,
        }
        settings.update(config)
        return PoolingClient(PoolingEndpoint(**settings), sender=sender)

    def test_a_text_reply_of_the_kept_count_is_accepted(self) -> None:
        from tests._tokenizers import word_tokenizer

        text = "the a of to"  # four word tokens; the rule drops 'a' (id 2)
        assert word_tokenizer().ids(text) == [1, 2, 3, 4]
        # The engine applied the rule: three vectors, while its usage reports the prompt's four tokens.
        sender = RecordingSender(_pooling_reply(rows=3, usage=4))
        client = self._client(sender)
        embeddings = asyncio.run(client.aencode([Content.from_text(text)], EncodeRole.DOCUMENT))
        assert embeddings.num_items == 1
        assert embeddings.offsets is not None and embeddings.offsets.tolist() == [0, 3]

    def test_a_text_reply_that_ignored_the_rule_is_refused(self) -> None:
        """Four vectors with usage four: the engine returned every prompt token, so the declared kept count
        (three) was not applied -- refused, never silently sliced (the rule's declared home is the plugin)."""
        sender = RecordingSender(_pooling_reply(rows=4, usage=4))
        client = self._client(sender)
        with pytest.raises(ProviderError, match=r"4 token vector\(s\).*declared keep-rule leaves 3 kept"):
            asyncio.run(client.aencode([Content.from_text("the a of to")], EncodeRole.DOCUMENT))

    def test_the_query_side_keeps_every_vector(self) -> None:
        """The rule is document-side (the checkpoint's ``skiplist_tasks``), so the engine spares a query
        prompt: a query reply of its full count is the answer, not a violation."""
        sender = RecordingSender(_pooling_reply(rows=4, usage=4))
        client = self._client(sender)
        embeddings = asyncio.run(client.aencode([Content.from_text("the a of to")], EncodeRole.QUERY))
        assert embeddings.offsets is not None and embeddings.offsets.tolist() == [0, 4]

    def test_a_query_reply_with_dropped_positions_is_refused(self) -> None:
        """The engine-side rule must spare queries (the plugin's document role gate): a short query reply is
        not the checkpoint's own query output, and the usage check refuses it -- never silently accepted."""
        sender = RecordingSender(_pooling_reply(rows=3, usage=4))
        client = self._client(sender)
        with pytest.raises(ProviderError, match=r"reports 4 prompt token\(s\).*decode to 3"):
            asyncio.run(client.aencode([Content.from_text("the a of to")], EncodeRole.QUERY))

    def test_a_media_document_counts_the_head_and_the_media_block(self, tmp_path: Any) -> None:
        """A media document's declared count is the head's kept ids plus the prepared media block (the
        vision wrapper and the patch run); the reply is checked against it and no ``skip_unapplied`` record
        is written -- the engine applied the rule to the render."""
        from tests._tokenizers import save, spaced_special_tokenizer

        (tmp_path / "t").mkdir()
        tokenizer_file = save(spaced_special_tokenizer(), tmp_path / "t")
        image = tmp_path / "page.png"
        image.write_bytes(_png_bytes())
        expected = _media_kept_count(image)
        sender = RecordingSender(_pooling_reply(rows=expected, usage=expected))
        client = self._client(sender, tokenizer=str(tokenizer_file), media_head_as_system=True, template=_DOC_TEMPLATE)
        embeddings = asyncio.run(client.aencode([Content.from_image(image.as_uri())], EncodeRole.DOCUMENT))
        assert embeddings.offsets is not None and embeddings.offsets.tolist() == [0, expected]
        assert [record.mechanisms for record in client.processing if record.changed] == [], (
            "the engine applied the rule to the media render: no skip_unapplied"
        )

    def test_a_media_reply_off_the_declared_count_is_refused(self, tmp_path: Any) -> None:
        from tests._tokenizers import save, spaced_special_tokenizer

        (tmp_path / "t").mkdir()
        tokenizer_file = save(spaced_special_tokenizer(), tmp_path / "t")
        image = tmp_path / "page.png"
        image.write_bytes(_png_bytes())
        expected = _media_kept_count(image)
        sender = RecordingSender(_pooling_reply(rows=expected + 1, usage=expected + 1))
        client = self._client(sender, tokenizer=str(tokenizer_file), media_head_as_system=True, template=_DOC_TEMPLATE)
        with pytest.raises(ProviderError, match=rf"declared keep-rule leaves {expected} kept"):
            asyncio.run(client.aencode([Content.from_image(image.as_uri())], EncodeRole.DOCUMENT))

    def test_the_flag_needs_the_declared_rule(self, tokenizer_json: str) -> None:
        with pytest.raises(ConfigError, match="document_skip_engine_side"):
            PoolingEndpoint(
                base_url="http://engine:8000/v1",
                model="m",
                dim=2,
                tokenizer=tokenizer_json,
                max_tokens=8192,
                document_skip_engine_side=True,
            )

    def test_the_flag_is_refused_beside_a_per_chunk_model(self, tokenizer_json: str) -> None:
        """A per-chunk reply has several outputs per input: there is no per-token kept count to check the
        declared rule against, so the combination is refused rather than silently unverified."""
        with pytest.raises(ConfigError, match="per_chunk"):
            PoolingEndpoint(
                base_url="http://engine:8000/v1",
                model="m",
                dim=2,
                tokenizer=tokenizer_json,
                max_tokens=8192,
                document_skip_token_ids=(2,),
                document_skip_engine_side=True,
                outputs="per_chunk",
            )


def _pooling_reply(rows: int, *, usage: int) -> Callable[[httpx.Request], httpx.Response]:
    """A canned ``/pooling`` reply: one base64 item of ``rows`` token vectors and a ``usage`` report of
    ``usage`` prompt tokens (the engine's prompt count, which the engine-side rule no longer matches)."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = base64.b64encode(np.ones((rows, 2), dtype=np.float16).tobytes()).decode("ascii")
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": json.loads(request.content).get("model"),
                "data": [{"index": 0, "object": "pooling", "data": payload}],
                "usage": {"prompt_tokens": usage, "total_tokens": usage},
            },
        )

    return handler


#: The document template the engine-side media tests declare: the trained head as its leading fixed segment.
_DOC_TEMPLATE: dict[str, Any] = {"document": [{"fixed": "{special:[D] }"}, {"content": "document"}], "anchor": "mean"}


def _media_kept_count(image: Any) -> int:
    """The declared count the client checks a media reply against: the head's ids plus the prepared media
    block, built here from the product's own preparation, media count and tokenizer (the same inputs the
    client passes) rather than from the client's private helper."""
    from rcp_ndcg.data.prepare import prepare_request
    from rcp_ndcg.data.resolution import ImagePolicy
    from tests._tokenizers import spaced_special_tokenizer

    policy = ImagePolicy(min_px=3136, max_px=1003520, processor="qwen2_vl")
    tokenizer = spaced_special_tokenizer()
    prepared = prepare_request([Content.from_image(image.as_uri())], policy, None, tokenizer=tokenizer)
    head = tokenizer.ids("[D] ", add_special_tokens=True)
    return len(head) + prepared.tokens.tokens


class TestMediaHeadAsSystem:
    """``media_head_as_system``: a media document sends the shape's leading fixed template head as a
    leading ``system`` message -- the card's sentence-transformers render for a pass-through engine chat
    template (pplx-embed-v2-late), where the user turn keeps only the media."""

    @staticmethod
    def _client(sender: Any, tokenizer: str, **config: Any) -> PoolingClient:
        settings: dict[str, Any] = {
            "base_url": "http://engine:8000/v1",
            "model": "pplx-late",
            "dim": 2,
            "normalize": False,
            "tokenizer": tokenizer,
            "max_tokens": 8192,
            "template": {
                "document": [{"fixed": "{special:[D] }"}, {"content": "document"}],
                "anchor": "mean",
                "add_special_tokens": True,
            },
            "image_policy": {"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
            "max_images": 1,
            "media_head_as_system": True,
        }
        settings.update(config)
        return PoolingClient(PoolingEndpoint(**settings), sender=sender)

    def test_an_image_document_sends_the_head_as_a_system_message(self, tmp_path: Any) -> None:
        from tests._tokenizers import save, spaced_special_tokenizer

        (tmp_path / "t").mkdir()
        tokenizer_file = save(spaced_special_tokenizer(), tmp_path / "t")
        image = tmp_path / "page.png"
        image.write_bytes(_png_bytes())
        sender = _GatedSender(
            PoolingServer({}, default=np.ones((1, 2), dtype=np.float16), media_vector=np.ones((1, 2), dtype=np.float16))
        )
        client = self._client(sender, str(tokenizer_file))

        asyncio.run(client.aencode([Content.from_image(image.as_uri())], EncodeRole.DOCUMENT))

        messages = sender.sent[0][0]["messages"]
        assert [message["role"] for message in messages] == ["system", "user"]
        assert messages[0]["content"] == "[D] ", "the trained head arrives as a system message"
        assert all(part["type"] != "text" for part in messages[1]["content"]), "the user turn keeps only the media"

    def test_the_head_is_not_sent_without_the_declaration(self, tmp_path: Any) -> None:
        """The default: the user turn alone (the engine chat template's own frame, if any)."""
        from tests._tokenizers import save, spaced_special_tokenizer

        (tmp_path / "t").mkdir()
        tokenizer_file = save(spaced_special_tokenizer(), tmp_path / "t")
        image = tmp_path / "page.png"
        image.write_bytes(_png_bytes())
        sender = _GatedSender(
            PoolingServer({}, default=np.ones((1, 2), dtype=np.float16), media_vector=np.ones((1, 2), dtype=np.float16))
        )
        client = self._client(sender, str(tokenizer_file), media_head_as_system=False)

        asyncio.run(client.aencode([Content.from_image(image.as_uri())], EncodeRole.DOCUMENT))

        messages = sender.sent[0][0]["messages"]
        assert [message["role"] for message in messages] == ["user"]

    def test_a_mixed_batch_sends_the_head_only_on_the_media_item(self, tmp_path: Any) -> None:
        """A text item's user text already carries the head (it is sent as the fitted render); attaching the
        system head to it too would duplicate the trained prefix and make the prompt depend on the batch."""
        from tests._tokenizers import save, spaced_special_tokenizer

        (tmp_path / "t").mkdir()
        tokenizer_file = save(spaced_special_tokenizer(), tmp_path / "t")
        image = tmp_path / "page.png"
        image.write_bytes(_png_bytes())
        sender = _GatedSender(
            PoolingServer({}, default=np.ones((1, 2), dtype=np.float16), media_vector=np.ones((1, 2), dtype=np.float16))
        )
        client = self._client(sender, str(tokenizer_file))

        asyncio.run(
            client.aencode([Content.from_text("plain"), Content.from_image(image.as_uri())], EncodeRole.DOCUMENT)
        )

        text_call, media_call = sender.sent[0][0], sender.sent[0][1]
        assert [message["role"] for message in text_call["messages"]] == ["user"]
        assert text_call["messages"][0]["content"] == [{"type": "text", "text": "[D] plain"}]
        assert [message["role"] for message in media_call["messages"]] == ["system", "user"]
        assert media_call["messages"][0]["content"] == "[D] "

    def test_a_config_without_a_template_is_refused(self, tokenizer_json: str) -> None:
        with pytest.raises(ConfigError, match="no template"):
            PoolingEndpoint(
                base_url="http://engine:8000/v1",
                model="m",
                dim=2,
                tokenizer=tokenizer_json,
                max_tokens=8192,
                image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
                max_images=1,
                media_head_as_system=True,
            )

    def test_a_config_without_media_is_refused(self, tokenizer_json: str) -> None:
        with pytest.raises(ConfigError, match="inert"):
            PoolingEndpoint(
                base_url="http://engine:8000/v1",
                model="m",
                dim=2,
                tokenizer=tokenizer_json,
                max_tokens=8192,
                template={"document": [{"fixed": "[D] "}, {"content": "document"}], "anchor": "mean"},
                media_head_as_system=True,
            )

    def test_a_shape_that_opens_with_content_is_refused(self, tokenizer_json: str) -> None:
        with pytest.raises(ConfigError, match="opens with a content span"):
            PoolingEndpoint(
                base_url="http://engine:8000/v1",
                model="m",
                dim=2,
                tokenizer=tokenizer_json,
                max_tokens=8192,
                template={"document": [{"content": "document"}, {"fixed": "[D] "}], "anchor": "mean"},
                image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
                max_images=1,
                media_head_as_system=True,
            )


class TestVideoPolicyFamily:
    """The engine's fps rule is ported for the qwen3_vl family only: a config that declares it beside
    another processor family is refused (the count would describe frames the engine never samples)."""

    def test_the_fps_rule_needs_the_qwen3_vl_family(self, tokenizer_json: str) -> None:
        with pytest.raises(ConfigError, match="qwen3_vl"):
            PoolingEndpoint(
                base_url="http://engine:8000/v1",
                model="m",
                dim=2,
                tokenizer=tokenizer_json,
                max_tokens=8192,
                image_processor="qwen2_vl",
                image_policy={"min_px": 3136, "max_px": 1003520},
                max_images=1,
                video_policy={"fps": 2.0, "wire": "video_url", "engine_video_pinning": True},
                max_videos=1,
            )

    def test_the_fps_rule_loads_on_the_qwen3_vl_family(self, tokenizer_json: str) -> None:
        endpoint = PoolingEndpoint(
            base_url="http://engine:8000/v1",
            model="m",
            dim=2,
            tokenizer=tokenizer_json,
            max_tokens=8192,
            image_processor="qwen3_vl",
            image_policy={"min_px": 65536, "max_px": 16777216},
            max_images=1,
            video_policy={"fps": 2.0, "wire": "video_url", "engine_video_pinning": True},
            max_videos=1,
        )
        assert endpoint.video_policy is not None and endpoint.video_policy.fps == 2.0


def _media_chunk_client(sender: Any) -> PoolingClient:
    """A pooling client whose config declares chunk overflow (refused at construction: vectors do not pool);
    the media-fit refusal for chunk must be reachable before that."""
    return PoolingClient(
        PoolingEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="colqwen",
            dim=2,
            tokenizer=_budget.DEFAULT_TOKENIZER or "test/tokenizer",
            max_tokens=60,
            image_policy={"min_px": 3136, "max_px": 1003520, "processor": "qwen2_vl"},
            max_images=4,
            on_overflow="fail",
        ),
        sender=sender,
    )
