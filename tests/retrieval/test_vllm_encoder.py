"""The vLLM encoder backend.

No engine is loaded here. What is under test is everything around it: mode and
task validation, the ragged-vs-flat layout that follows from the pooling task,
role prefixes, and the HTTP path. Loading an engine is a GPU integration test.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from rcp_ndcg_core.content import Content

from rcp_ndcg.retrieval.encoder import Embeddings, EncodeRole
from rcp_ndcg.retrieval.encoders.vllm_encoder import VllmEncoder


class _FakePoolingClient:
    """Stands in for :class:`VllmPoolingClient`, returning unit vectors."""

    def __init__(self, *, multi_vector: bool = True) -> None:
        self.multi_vector = multi_vector
        self.seen: list[list[Content]] = []

    def pooling(self, contents: Sequence[Content]) -> Embeddings:
        self.seen.append(list(contents))
        if self.multi_vector:
            return Embeddings.ragged([np.ones((2, 3), dtype=np.float32) for _ in contents])
        return Embeddings.single(np.ones((len(contents), 3), dtype=np.float32))


def _http_encoder(client: Any, **kwargs: Any) -> VllmEncoder:
    kwargs.setdefault("pooling_task", "token_embed")
    encoder = VllmEncoder(model_name="stub", mode="http", api_base="http://host:8000", **kwargs)
    encoder._client = client
    return encoder


class TestConstruction:
    def test_construction_does_not_load_an_engine(self) -> None:
        """Building an encoder claims no GPU; the engine loads on the first offline encode."""
        encoder = VllmEncoder(model_name="stub", pooling_task="embed")
        assert encoder._engine is None


class TestConfigValidation:
    def test_unknown_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown vllm mode"):
            VllmEncoder(model_name="stub", pooling_task="embed", mode="grpc")

    def test_unknown_pooling_task_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown pooling_task"):
            VllmEncoder(model_name="stub", pooling_task="classify")

    def test_http_mode_needs_an_api_base(self) -> None:
        with pytest.raises(ValueError, match="needs api_base"):
            VllmEncoder(model_name="stub", pooling_task="embed", mode="http")

    def test_the_pooling_task_is_required(self) -> None:
        """It once defaulted to token_embed: a config without it silently built a multi-vector index."""
        with pytest.raises(TypeError, match="pooling_task"):
            VllmEncoder(model_name="stub")  # type: ignore[call-arg]

    def test_pooling_task_decides_the_layout(self) -> None:
        assert VllmEncoder(model_name="stub", pooling_task="token_embed").is_multi_vector
        assert not VllmEncoder(model_name="stub", pooling_task="embed").is_multi_vector


class TestHttpEncoding:
    def test_ragged_output_survives(self) -> None:
        client = _FakePoolingClient()
        encoder = _http_encoder(client)
        embeddings = encoder.encode(
            [Content.from_text("a"), Content.from_text("b")],
            role=EncodeRole.DOCUMENT,
        )
        assert embeddings.is_multi_vector
        assert embeddings.num_items == 2
        assert len(embeddings.vectors) == 4

    def test_role_prefixes_are_applied_per_side(self) -> None:
        """Asymmetric models need the side named; a flag both paths set would drift."""
        client = _FakePoolingClient()
        encoder = _http_encoder(client, query_prefix="Query: ", document_prefix="Passage: ")

        encoder.encode([Content.from_text("cats")], role=EncodeRole.QUERY)
        encoder.encode([Content.from_text("cats")], role=EncodeRole.DOCUMENT)
        assert client.seen[0][0].text == "Query: cats"
        assert client.seen[1][0].text == "Passage: cats"

    def test_prefix_does_not_drop_an_image(self) -> None:
        """Prefixing ``.text`` and rebuilding would lose the parts that matter."""
        client = _FakePoolingClient()
        encoder = _http_encoder(client, document_prefix="Passage: ")
        encoder.encode([Content.from_image("file:///tmp/page.png")], role=EncodeRole.DOCUMENT)

        sent = client.seen[0][0]
        assert sent.has_media
        assert sent.text == "Passage: "

    def test_normalisation_is_applied_per_vector(self) -> None:
        client = _FakePoolingClient()
        encoder = _http_encoder(client)
        embeddings = encoder.encode([Content.from_text("a")], role=EncodeRole.QUERY)
        np.testing.assert_allclose(np.linalg.norm(embeddings.vectors, axis=1), 1.0, atol=1e-6)

    def test_a_short_response_is_refused(self) -> None:
        """Misaligned vectors make an index that works and is wrong."""

        class _Dropping:
            def pooling(self, contents: Sequence[Content]) -> Embeddings:
                return Embeddings.ragged([np.ones((1, 3), dtype=np.float32)])

        encoder = _http_encoder(_Dropping())
        with pytest.raises(RuntimeError, match="refusing to return misaligned"):
            encoder.encode([Content.from_text("a"), Content.from_text("b")], role=EncodeRole.QUERY)

    def test_empty_input_needs_no_request(self) -> None:
        client = _FakePoolingClient()
        encoder = _http_encoder(client)
        assert encoder.encode([], role=EncodeRole.QUERY).num_items == 0
        assert client.seen == []

    def test_media_is_not_refused_before_the_engine_has_a_say(self) -> None:
        """vLLM's registry is the authority, and answering needs the engine loaded."""
        assert VllmEncoder(model_name="stub", pooling_task="token_embed").supports_media


class TestOfflinePrompt:
    def test_text_only_prompt_carries_no_multi_modal_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("vllm", reason="offline prompts need the vllm package")
        encoder = VllmEncoder(model_name="stub", pooling_task="token_embed")
        prompt = encoder._offline_prompt(Content.from_text("hello"))
        assert prompt["prompt"] == "hello"
        assert "multi_modal_data" not in prompt

    def test_image_prompt_uses_the_checkpoints_chat_template(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """Hand-written vision placeholders drift from the weights silently."""
        pytest.importorskip("vllm", reason="offline prompts need the vllm package")

        class _Processor:
            def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
                parts = messages[0]["content"]
                return "".join("<IMG>" if part["type"] == "image" else part["text"] for part in parts)

        image = tmp_path / "page.png"
        image.write_bytes(_png_bytes())

        encoder = VllmEncoder(model_name="stub", pooling_task="token_embed")
        encoder._processor = _Processor()
        content = Content.from_parts(
            [
                *Content.from_text("caption").parts,
                *Content.from_image(image.as_uri()).parts,
            ]
        )
        prompt = encoder._offline_prompt(content)

        assert prompt["prompt"] == "caption<IMG>"
        assert len(prompt["multi_modal_data"]["image"]) == 1


def _png_bytes() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (0, 128, 255)).save(buffer, format="PNG")
    return buffer.getvalue()
