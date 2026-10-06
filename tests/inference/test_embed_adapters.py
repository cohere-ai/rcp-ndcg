"""The embedding wire adapters: what each profile sends, what it reads back, and what it refuses.

Every test is offline: adapters build and read
:class:`~rcp_ndcg.inference.types.Call` / :class:`~rcp_ndcg.inference.types.Reply` objects directly.
"""

from __future__ import annotations

import base64
from typing import Any, ClassVar

import numpy as np
import pytest
from rcp_ndcg_core.content import Content, ImagePart, MediaRef, TextPart, VideoPart

from rcp_ndcg.errors import CapabilityError, ProviderError, RequestRejectedError
from rcp_ndcg.inference import Adapter, EmbedRequest, EncodeRole, get_adapter, known_adapters
from rcp_ndcg.inference.adapters.embeddings import (
    CohereEmbeddings,
    GeminiEmbeddings,
    OpenAIEmbeddings,
    VoyageEmbeddings,
)
from rcp_ndcg.inference.types import Reply, TokenCount
from tests.inference._embed import embeddings_data, vendor_payload

#: Every shipped embedding adapter, under its registered name.
ADAPTERS: dict[str, type] = {
    "openai_embeddings": OpenAIEmbeddings,
    "cohere": CohereEmbeddings,
    "voyage": VoyageEmbeddings,
    "gemini": GeminiEmbeddings,
}


def request(
    role: EncodeRole = EncodeRole.DOCUMENT, texts: tuple[str, ...] = ("hello",), **overrides: Any
) -> EmbedRequest:
    """An embedding request over plain-text items."""
    return EmbedRequest(contents=tuple(Content.from_text(text) for text in texts), role=role, **overrides)


def reply(status: int, body: Any) -> Reply:
    """One reply with an empty header map."""
    return Reply(status, body, {})


def base64_f32(vector: list[float]) -> str:
    """One embedding as the base64 little-endian float32 string the OpenAI and vLLM framings use."""
    return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii")


def calls_with(name: str, content: Content) -> Any:
    """``calls`` of adapter ``name`` over a single non-text item."""
    return ADAPTERS[name]().calls(EmbedRequest(contents=(content,), role=EncodeRole.DOCUMENT), model="m")


class TestRegistry:
    def test_every_profile_is_registered_under_its_name_and_role(self) -> None:
        assert {"openai_embeddings", "cohere", "voyage", "gemini"} <= set(known_adapters("embed"))
        for name, cls in ADAPTERS.items():
            assert get_adapter(name, role="embed") is cls
            assert cls.role == "embed"

    def test_the_embed_names_resolve_inside_their_role_only(self) -> None:
        """The registry is scoped by role: ``cohere`` is shared with the rerank role, whose lookup resolves to
        the rerank wire, never to the embedding profile."""
        from rcp_ndcg.errors import ConfigError
        from rcp_ndcg.inference.adapters.rerank import CohereRerankAdapter

        assert get_adapter("cohere", role="rerank") is CohereRerankAdapter
        assert get_adapter("cohere", role="embed") is CohereEmbeddings
        with pytest.raises(ConfigError, match="unknown judge adapter 'gemini'"):
            get_adapter("gemini", role="judge")

    def test_the_adapters_satisfy_the_protocol(self) -> None:
        for cls in ADAPTERS.values():
            assert isinstance(cls(), Adapter)


class TestOpenAIShape:
    adapter = OpenAIEmbeddings()

    def test_the_body_is_model_input_and_encoding_format(self) -> None:
        call = self.adapter.calls(request(texts=("a", "b")), model="octen-embedding-8b")[0]
        assert call.method == "POST" and call.path == "/embeddings"
        assert call.json == {"model": "octen-embedding-8b", "input": ["a", "b"], "encoding_format": "float"}

    def test_dimensions_is_sent_only_when_set(self) -> None:
        assert self.adapter.calls(request(dimensions=1024), model="m")[0].json["dimensions"] == 1024
        assert "dimensions" not in self.adapter.calls(request(), model="m")[0].json

    def test_the_reply_is_read_in_index_order(self) -> None:
        body = embeddings_data([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], indices=[2, 0, 1])
        vectors = self.adapter.interpret(request(texts=("a", "b", "c")), [reply(200, body)])
        assert vectors.as_matrix()[:, 0].tolist() == [1.0, 0.0, 1.0]  # items 0, 1, 2 -- not reply order

    def test_a_reply_without_indices_is_read_in_reply_order(self) -> None:
        """A reply with no ``index`` at all (no engine ships one, but the shape has no field to trust) is read
        in reply order, which aligns with the request's items."""
        body = {"data": [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]}
        vectors = self.adapter.interpret(request(texts=("a", "b")), [reply(200, body)])
        assert vectors.as_matrix()[:, 0].tolist() == [1.0, 0.0]

    def test_a_base64_reply_decodes_as_float32(self) -> None:
        body = {"data": [{"index": 0, "embedding": base64_f32([0.25, -0.5, 0.75])}]}
        vectors = self.adapter.interpret(request(), [reply(200, body)])
        assert vectors.as_matrix()[0].tolist() == [0.25, -0.5, 0.75]
        assert vectors.vectors.dtype == np.float32

    def test_a_reply_without_a_data_list_is_rejected(self) -> None:
        with pytest.raises(RequestRejectedError, match="data"):
            self.adapter.interpret(request(), [reply(200, {"embeddings": [[1.0]]})])

    def test_usage_reads_prompt_tokens(self) -> None:
        assert self.adapter.usage(reply(200, {"data": [], "usage": {"prompt_tokens": 7}})) == TokenCount(7)
        assert self.adapter.usage(reply(200, {"data": []})) is None
        assert self.adapter.usage(reply(200, {"data": [], "usage": {"prompt_tokens": "abc"}})) is None
        assert self.adapter.usage(reply(200, {"data": [], "usage": {"prompt_tokens": [1]}})) is None


class TestCohereShape:
    adapter = CohereEmbeddings()

    def test_the_body_names_the_side_of_the_pair(self) -> None:
        call = self.adapter.calls(request(EncodeRole.QUERY), model="embed-v4.0")[0]
        assert call.path == "/embed"
        assert call.json == {
            "model": "embed-v4.0",
            "texts": ["hello"],
            "input_type": "search_query",
            "embedding_types": ["float"],
        }
        document = self.adapter.calls(request(texts=("d",)), model="embed-v4.0")[0].json
        assert document["input_type"] == "search_document"

    def test_the_reply_is_read_from_embeddings_float(self) -> None:
        vectors = self.adapter.interpret(request(), [reply(200, vendor_payload("cohere", [[3.0, 4.0]]))])
        assert vectors.as_matrix()[0].tolist() == [3.0, 4.0]

    def test_usage_reads_the_billed_units(self) -> None:
        body = {"embeddings": {"float": [[1.0]]}, "meta": {"billed_units": {"input_tokens": 11}}}
        assert self.adapter.usage(reply(200, body)) == TokenCount(11)
        assert self.adapter.usage(reply(200, {"embeddings": {"float": [[1.0]]}})) is None


class TestVoyageShape:
    adapter = VoyageEmbeddings()

    def test_the_body_names_the_input_type(self) -> None:
        call = self.adapter.calls(request(EncodeRole.QUERY), model="voyage-3")[0]
        assert call.path == "/embeddings"
        assert call.json == {"model": "voyage-3", "input": ["hello"], "input_type": "query"}
        document = self.adapter.calls(request(texts=("d",)), model="voyage-3")[0].json
        assert document["input_type"] == "document"

    def test_the_reply_is_the_openai_shape(self) -> None:
        vectors = self.adapter.interpret(request(), [reply(200, embeddings_data([[0.0, 1.0]]))])
        assert vectors.as_matrix()[0].tolist() == [0.0, 1.0]


class TestGeminiShape:
    adapter = GeminiEmbeddings()

    def test_the_body_batches_one_request_entry_per_text(self) -> None:
        call = self.adapter.calls(request(EncodeRole.QUERY, texts=("x", "y")), model="gemini-embed")[0]
        assert call.path == "/models/gemini-embed:batchEmbedContents"
        assert call.json["requests"] == [
            {"model": "models/gemini-embed", "content": {"parts": [{"text": "x"}]}, "taskType": "RETRIEVAL_QUERY"},
            {"model": "models/gemini-embed", "content": {"parts": [{"text": "y"}]}, "taskType": "RETRIEVAL_QUERY"},
        ]
        document = self.adapter.calls(request(texts=("d",)), model="gemini-embed")[0].json
        assert document["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"

    def test_the_reply_is_read_from_values(self) -> None:
        vectors = self.adapter.interpret(request(), [reply(200, vendor_payload("gemini", [[2.0, 0.0]]))])
        assert vectors.as_matrix()[0].tolist() == [2.0, 0.0]

    def test_a_reply_without_embeddings_is_rejected(self) -> None:
        with pytest.raises(RequestRejectedError, match="embeddings"):
            self.adapter.interpret(request(), [reply(200, {"responses": []})])


class TestProfiles:
    def test_every_profile_declares_its_published_batch_cap(self) -> None:
        assert {name: cls.MAX_BATCH for name, cls in ADAPTERS.items()} == {
            "openai_embeddings": 128,
            "cohere": 96,
            "voyage": 128,
            "gemini": 100,
        }

    def test_every_profile_names_its_public_base_url(self) -> None:
        assert {name: cls.DEFAULT_BASE_URL for name, cls in ADAPTERS.items()} == {
            "openai_embeddings": "https://api.openai.com/v1",
            "cohere": "https://api.cohere.com/v2",
            "voyage": "https://api.voyageai.com/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta",
        }

    def test_the_hosted_profiles_require_a_key_and_openai_does_not(self) -> None:
        assert OpenAIEmbeddings.KEY_REQUIRED is False
        assert CohereEmbeddings.KEY_REQUIRED is True
        assert VoyageEmbeddings.KEY_REQUIRED is True
        assert GeminiEmbeddings.KEY_REQUIRED is True

    def test_the_key_variables_and_the_gemini_header(self) -> None:
        assert OpenAIEmbeddings.API_KEY_ENV == ("OPENAI_API_KEY",)
        assert CohereEmbeddings.API_KEY_ENV == ("CO_API_KEY", "COHERE_API_KEY")
        assert VoyageEmbeddings.API_KEY_ENV == ("VOYAGE_API_KEY",)
        assert GeminiEmbeddings.API_KEY_ENV == ("GEMINI_API_KEY", "GOOGLE_API_KEY")
        assert OpenAIEmbeddings.AUTH_HEADER is None and GeminiEmbeddings.AUTH_HEADER == "x-goog-api-key"


class TestRefusals:
    adapter: ClassVar[OpenAIEmbeddings] = OpenAIEmbeddings()

    @pytest.mark.parametrize("name", ["cohere", "voyage", "gemini"])
    def test_a_hosted_profile_refuses_a_dimensions_request(self, name: str) -> None:
        """The hosted APIs take no dimensions parameter; sending one would be silently ignored, which is
        refused instead -- a Matryoshka cut that never reaches the wire would change the vectors."""
        with pytest.raises(CapabilityError, match="dimensions"):
            ADAPTERS[name]().calls(request(dimensions=256), model="m")

    def test_an_openai_shaped_reply_with_duplicate_indices_is_rejected(self) -> None:
        body = {
            "data": [
                {"index": 0, "embedding": [9.0]},
                {"index": 0, "embedding": [1.0]},
                {"index": 1, "embedding": [0.0]},
            ]
        }
        with pytest.raises(RequestRejectedError, match="index"):
            self.adapter.interpret(request(texts=("a", "b", "c")), [reply(200, body)])

    def test_a_reply_with_only_some_indices_is_rejected(self) -> None:
        body = {"data": [{"index": 0, "embedding": [1.0]}, {"embedding": [0.0]}]}
        with pytest.raises(RequestRejectedError, match="index"):
            self.adapter.interpret(request(texts=("a", "b")), [reply(200, body)])

    @pytest.mark.parametrize("raw", [[1.0, None], [], 3.0])
    def test_an_empty_non_finite_or_scalar_embedding_is_rejected(self, raw: Any) -> None:
        body = {"data": [{"index": 0, "embedding": raw}]}
        with pytest.raises(RequestRejectedError, match="embedding"):
            self.adapter.interpret(request(), [reply(200, body)])

    @pytest.mark.parametrize("raw", [["a"], {"x": 1.0}, [[1.0, 2.0], [3.0]]])
    def test_a_non_numeric_embedding_payload_is_rejected_typed(self, raw: Any) -> None:
        """A payload numpy cannot coerce cleanly is a typed refusal, never a raw ValueError/TypeError."""
        body = {"data": [{"index": 0, "embedding": raw}]}
        with pytest.raises(RequestRejectedError, match="embedding"):
            self.adapter.interpret(request(), [reply(200, body)])

    def test_a_short_base64_embedding_is_rejected(self) -> None:
        body = {"data": [{"index": 0, "embedding": base64.b64encode(b"abc").decode("ascii")}]}
        with pytest.raises(RequestRejectedError, match="embedding"):
            self.adapter.interpret(request(), [reply(200, body)])

    def test_non_integer_index_values_are_rejected(self) -> None:
        body = {"data": [{"index": "1", "embedding": [1.0]}, {"index": "0", "embedding": [0.0]}]}
        with pytest.raises(RequestRejectedError, match="index"):
            self.adapter.interpret(request(texts=("a", "b")), [reply(200, body)])

    def test_a_reply_wider_or_narrower_than_the_cut_is_rejected(self) -> None:
        body = embeddings_data([[1.0, 0.0]])
        with pytest.raises(RequestRejectedError, match="dimensions=8"):
            self.adapter.interpret(request(dimensions=8), [reply(200, body)])

    @pytest.mark.parametrize("name", sorted(ADAPTERS))
    def test_an_image_part_is_refused_with_its_media_type(self, name: str) -> None:
        content = Content(root=[ImagePart(ref=MediaRef(uri="gs://bucket/page_1.png")), TextPart(text="caption")])
        with pytest.raises(CapabilityError, match="image"):
            calls_with(name, content)

    @pytest.mark.parametrize("name", sorted(ADAPTERS))
    def test_a_video_part_is_refused_with_its_media_type(self, name: str) -> None:
        content = Content(root=[VideoPart(ref=MediaRef(uri="gs://bucket/clip.mp4"))])
        with pytest.raises(CapabilityError, match="video"):
            calls_with(name, content)

    def test_a_batch_over_the_server_cap_is_a_capability_error_naming_batch_size(self) -> None:
        with pytest.raises(CapabilityError, match="batch_size"):
            self.adapter.interpret(request(), [reply(413, {"error": "batch size 33 > 32"})])

    def test_an_over_length_400_is_a_capability_error_naming_max_tokens_and_batch_size(self) -> None:
        body = {"error": "This model's maximum context length is 8192 tokens. However, you requested 9000 tokens."}
        with pytest.raises(CapabilityError) as caught:
            self.adapter.interpret(request(), [reply(400, body)])
        assert "max_tokens" in (caught.value.hint or "")
        assert "batch_size" in (caught.value.hint or "")

    def test_another_400_is_a_request_rejection(self) -> None:
        with pytest.raises(RequestRejectedError, match="HTTP 400"):
            self.adapter.interpret(request(), [reply(400, {"error": "bad model"})])

    def test_a_500_is_a_provider_error(self) -> None:
        with pytest.raises(ProviderError, match="HTTP 500"):
            self.adapter.interpret(request(), [reply(500, {"error": "boom"})])

    @pytest.mark.parametrize(
        ("status", "body", "prefix"),
        [
            (400, {"error": "maximum context length " + "x" * 277 + "!" + "y" * 20}, "x" * 277),
            (400, {"error": "x" * 300 + "!" + "y" * 20}, "x" * 300),
            (500, {"error": "x" * 300 + "!" + "y" * 20}, "x" * 300),
        ],
    )
    def test_an_error_message_carries_at_most_300_characters(
        self, status: int, body: dict[str, str], prefix: str
    ) -> None:
        """The declared message cap: a refusal quotes at most 300 characters of the body -- the 301st
        character never surfaces (here the '!' sitting exactly 300 characters in)."""
        with pytest.raises((CapabilityError, RequestRejectedError, ProviderError)) as caught:
            self.adapter.interpret(request(), [reply(status, body)])
        message = str(caught.value)
        assert prefix in message and "!" not in message

    def test_vectors_must_align_to_the_request(self) -> None:
        body = embeddings_data([[1.0, 0.0], [0.0, 1.0]])
        with pytest.raises(RequestRejectedError, match=r"2 vector\(s\) for 1 item"):
            self.adapter.interpret(request(texts=("a",)), [reply(200, body)])

    def test_ragged_vectors_are_rejected(self) -> None:
        body = {"data": [{"index": 0, "embedding": [1.0, 0.0]}, {"index": 1, "embedding": [1.0, 0.0, 0.0]}]}
        with pytest.raises(RequestRejectedError, match="differing dimension"):
            self.adapter.interpret(request(texts=("a", "b")), [reply(200, body)])


class TestNullEmbeddingRefused:
    """A NULL embedding (a mutation that survived the sweep's suite) is refused: one entry without usable
    data is a refused answer, never a silent zero row the corpus would index."""

    def test_a_null_embedding_is_refused(self) -> None:
        from rcp_ndcg.errors import RequestRejectedError

        reply = Reply(200, {"data": [{"index": 0, "embedding": None}]}, {})
        with pytest.raises(RequestRejectedError, match="without"):
            OpenAIEmbeddings().interpret(
                EmbedRequest(contents=(Content.from_text("x"),), role=EncodeRole.DOCUMENT), [reply]
            )

    def test_a_missing_embedding_entry_is_refused(self) -> None:
        from rcp_ndcg.errors import RequestRejectedError

        reply = Reply(200, {"data": [{"index": 0}]}, {})
        with pytest.raises(RequestRejectedError, match="embedding"):
            OpenAIEmbeddings().interpret(
                EmbedRequest(contents=(Content.from_text("x"),), role=EncodeRole.DOCUMENT), [reply]
            )
