"""The vLLM ``/pooling`` and ``/rerank`` client.

Tested against a stubbed ``httpx.post`` rather than a live engine: what belongs to
this module is the request shape, the response-to-order mapping, and the retry
policy, not the engine's numbers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import ProviderError
from rcp_ndcg.retrieval.vllm_http import VllmPoolingClient


class _Response:
    def __init__(self, status_code: int, body: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._body = body
        self.text = text
        self.headers: dict[str, str] = {}

    def json(self) -> Any:
        return self._body


class _Recorder:
    """Captures posts and replays a queue of responses."""

    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, *, headers: Any = None, json: Any = None, timeout: Any = None) -> _Response:
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.responses.pop(0) if self.responses else _Response(200, {"data": []})


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retry backoff without the wall-clock cost."""
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


def _install(monkeypatch: pytest.MonkeyPatch, responses: list[_Response]) -> _Recorder:
    recorder = _Recorder(responses)
    monkeypatch.setattr("httpx.post", recorder)
    return recorder


class TestApiBase:
    @pytest.mark.parametrize(
        "given",
        ["http://host:8000", "http://host:8000/", "http://host:8000/v1", "http://host:8000/v1/"],
    )
    def test_v1_suffix_and_slashes_are_normalised(self, given: str) -> None:
        """Configs in the wild carry either form; the pooling routes are at the root."""
        assert VllmPoolingClient(given).api_base == "http://host:8000"


class TestPooling:
    def test_text_batch_is_one_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _install(
            monkeypatch,
            [_Response(200, {"data": [{"index": 0, "data": [1.0, 0.0]}, {"index": 1, "data": [0.0, 1.0]}]})],
        )
        client = VllmPoolingClient("http://host:8000", model="m")
        embeddings = client.pooling([Content.from_text("a"), Content.from_text("b")])

        assert len(recorder.calls) == 1
        assert recorder.calls[0]["url"] == "http://host:8000/pooling"
        assert recorder.calls[0]["json"] == {"input": ["a", "b"], "model": "m"}
        assert not embeddings.is_multi_vector
        assert embeddings.num_items == 2

    def test_token_embed_response_becomes_ragged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The layout of the answer is what says which pooling task ran."""
        _install(
            monkeypatch,
            [
                _Response(
                    200,
                    {
                        "data": [
                            {"index": 0, "data": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]},
                            {"index": 1, "data": [[0.5, 0.5]]},
                        ]
                    },
                )
            ],
        )
        embeddings = VllmPoolingClient("http://host:8000").pooling([Content.from_text("a"), Content.from_text("b")])
        assert embeddings.is_multi_vector
        assert embeddings.offsets is not None and embeddings.offsets.tolist() == [0, 3, 4]

    def test_response_order_is_restored_from_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(
            monkeypatch,
            [_Response(200, {"data": [{"index": 1, "data": [9.0]}, {"index": 0, "data": [1.0]}]})],
        )
        embeddings = VllmPoolingClient("http://host:8000").pooling(
            [Content.from_text("first"), Content.from_text("second")]
        )
        np.testing.assert_allclose(embeddings.vectors[:, 0], [1.0, 9.0])

    def test_media_items_go_one_request_each_as_messages(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        """Only the ``messages`` shape gets the model's chat template applied."""
        image = tmp_path / "page.png"
        image.write_bytes(_png_bytes())
        recorder = _install(
            monkeypatch,
            [
                _Response(200, {"data": [{"index": 0, "data": [[1.0, 0.0]]}]}),
                _Response(200, {"data": [{"index": 0, "data": [[0.0, 1.0]]}]}),
            ],
        )
        contents = [Content.from_image(image.as_uri()), Content.from_text("plain")]
        embeddings = VllmPoolingClient("http://host:8000").pooling(contents)

        assert len(recorder.calls) == 2
        first = recorder.calls[0]["json"]["messages"][0]["content"]
        assert first[0]["type"] == "image_url"
        assert first[0]["image_url"]["url"].startswith("data:image/png;base64,")
        assert embeddings.num_items == 2

    def test_missing_data_key_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, [_Response(200, {"object": "list"})])
        with pytest.raises(ProviderError, match="no 'data'"):
            VllmPoolingClient("http://host:8000").pooling([Content.from_text("a")])

    def test_no_inputs_is_no_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _install(monkeypatch, [])
        assert VllmPoolingClient("http://host:8000").pooling([]).num_items == 0
        assert recorder.calls == []


class TestRerank:
    def test_scores_are_returned_in_input_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The response is ranked; callers align positionally with their doc_ids."""
        _install(
            monkeypatch,
            [
                _Response(
                    200,
                    {
                        "results": [
                            {"index": 2, "relevance_score": 0.9, "document": {"text": "c"}},
                            {"index": 0, "relevance_score": 0.5, "document": {"text": "a"}},
                            {"index": 1, "relevance_score": 0.1, "document": {"text": "b"}},
                        ]
                    },
                )
            ],
        )
        scores = VllmPoolingClient("http://host:8000").rerank("q", ["a", "b", "c"])
        assert scores == [0.5, 0.1, 0.9]

    def test_top_n_defaults_to_every_document(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A partial response would leave documents unscored with nothing to fill in."""
        recorder = _install(
            monkeypatch,
            [_Response(200, {"results": [{"index": i, "relevance_score": 0.0, "document": {}} for i in range(3)]})],
        )
        VllmPoolingClient("http://host:8000").rerank("q", ["a", "b", "c"])
        assert recorder.calls[0]["json"]["top_n"] == 3

    def test_truncation_fields_are_only_sent_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _install(
            monkeypatch,
            [_Response(200, {"results": [{"index": 0, "relevance_score": 1.0, "document": {}}]})],
        )
        VllmPoolingClient("http://host:8000").rerank("q", ["a"], max_tokens_per_query=64)
        payload = recorder.calls[0]["json"]
        assert payload["max_tokens_per_query"] == 64
        assert "max_tokens_per_doc" not in payload
        assert "truncate_prompt_tokens" not in payload

    def test_an_unscored_document_is_an_error_not_a_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Filling in a score would silently change the ranking it feeds."""
        _install(
            monkeypatch,
            [_Response(200, {"results": [{"index": 0, "relevance_score": 1.0, "document": {}}]})],
        )
        with pytest.raises(ProviderError, match="scored 1 of 2"):
            VllmPoolingClient("http://host:8000").rerank("q", ["a", "b"])

    def test_an_out_of_range_index_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(
            monkeypatch,
            [_Response(200, {"results": [{"index": 7, "relevance_score": 1.0, "document": {}}]})],
        )
        with pytest.raises(ProviderError, match="index 7"):
            VllmPoolingClient("http://host:8000").rerank("q", ["a"])

    def test_media_documents_travel_as_content_parts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        image = tmp_path / "page.png"
        image.write_bytes(_png_bytes())
        recorder = _install(
            monkeypatch,
            [_Response(200, {"results": [{"index": 0, "relevance_score": 1.0, "document": {}}]})],
        )
        VllmPoolingClient("http://host:8000").rerank("q", [Content.from_image(image.as_uri())])

        document = recorder.calls[0]["json"]["documents"][0]
        assert set(document) == {"content"}
        assert document["content"][0]["type"] == "image_url"
        # A text query stays a bare string, so a text-only server sees the request
        # it always saw.
        assert recorder.calls[0]["json"]["query"] == "q"

    def test_text_only_content_is_sent_as_a_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _install(
            monkeypatch,
            [_Response(200, {"results": [{"index": 0, "relevance_score": 1.0, "document": {}}]})],
        )
        VllmPoolingClient("http://host:8000").rerank(Content.from_text("q"), [Content.from_text("d")])
        assert recorder.calls[0]["json"]["documents"] == ["d"]

    def test_no_documents_is_no_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _install(monkeypatch, [])
        assert VllmPoolingClient("http://host:8000").rerank("q", []) == []
        assert recorder.calls == []


class TestRetries:
    def test_retryable_status_is_retried(self, monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> None:
        """A busy engine queue and a restarting pod both look like this."""
        recorder = _install(
            monkeypatch,
            [
                _Response(503, text="loading"),
                _Response(200, {"results": [{"index": 0, "relevance_score": 1.0, "document": {}}]}),
            ],
        )
        assert VllmPoolingClient("http://host:8000").rerank("q", ["a"]) == [1.0]
        assert len(recorder.calls) == 2

    def test_a_client_error_is_not_retried(self, monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> None:
        """A 400 means the request is wrong; sending it again cannot help."""
        recorder = _install(monkeypatch, [_Response(400, text="bad request")])
        with pytest.raises(ProviderError, match="400"):
            VllmPoolingClient("http://host:8000").rerank("q", ["a"])
        assert len(recorder.calls) == 1

    def test_retries_are_bounded(self, monkeypatch: pytest.MonkeyPatch, no_sleep: None) -> None:
        recorder = _install(monkeypatch, [_Response(503, text="x") for _ in range(10)])
        with pytest.raises(ProviderError):
            VllmPoolingClient("http://host:8000", max_retries=2).rerank("q", ["a"])
        assert len(recorder.calls) == 3

    def test_bearer_token_is_sent_when_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _install(
            monkeypatch,
            [_Response(200, {"results": [{"index": 0, "relevance_score": 1.0, "document": {}}]})],
        )
        VllmPoolingClient("http://host:8000", api_key="secret").rerank("q", ["a"])
        assert recorder.calls[0]["headers"]["Authorization"] == "Bearer secret"


def _png_bytes() -> bytes:
    """A 1x1 PNG, so the resolver has something real to base64."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()
