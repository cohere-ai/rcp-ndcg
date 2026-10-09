"""The pooling role client: prompts per role, batching, concurrency, normalisation and the transfer dtype.

The client is tested through its wire adapter against the in-test pooling server
(:mod:`tests.inference._pooling`), so what is asserted is what a run would see: the requests that go out and
the ragged vectors that come back, in the transfer dtype end to end.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import numpy as np
import pytest
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import ConfigError, RequestRejectedError
from rcp_ndcg.inference.clients.pool import PoolingClient
from rcp_ndcg.inference.config import PoolingEndpoint
from rcp_ndcg.inference.types import EncodeRole
from tests.inference import _budget
from tests.inference._pooling import PoolingServer, server_sender


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
