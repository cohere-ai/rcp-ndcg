"""Tests for hosted-embedding retrieval.

Every vendor call is stubbed at the HTTP layer: what is under test is our
request shapes, retry policy and search, not the vendors' servers.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from rcp_ndcg.errors import CredentialsError, ProviderError
from rcp_ndcg.retrieval.api_dense import VENDORS, EmbeddingAPIClient

CORPUS = ["a document about foxes", "a document about circuits", "a document about dogs"]


class _Response:
    def __init__(self, status_code: int, payload: Any = None, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.text = json.dumps(self._payload)

    def json(self) -> Any:
        return self._payload


def _vendor_payload(vendor: str, vectors: list[list[float]]) -> dict:
    if vendor in ("openai", "voyage"):
        return {"data": [{"embedding": v} for v in vectors]}
    if vendor == "cohere":
        return {"embeddings": {"float": vectors}}
    return {"embeddings": [{"values": v} for v in vectors]}


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch):
    """Capture outbound requests and replay scripted responses."""
    import httpx

    calls: list[dict] = []
    queue: list[_Response] = []

    def post(url: str, *, headers: dict, json: dict, timeout: float) -> _Response:
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return queue.pop(0) if queue else _Response(200, _vendor_payload("openai", [[1.0, 0.0]]))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr("rcp_ndcg.retrieval._http.time.sleep", lambda _: None)
    return type("Http", (), {"calls": calls, "queue": queue})()


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch):
    for name in ("OPENAI_API_KEY", "CO_API_KEY", "VOYAGE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(name, f"test-{name.lower()}")


class TestVendorRequests:
    @pytest.mark.parametrize("vendor", sorted(VENDORS))
    def test_every_vendor_round_trips(self, vendor: str, http) -> None:
        http.queue.append(_Response(200, _vendor_payload(vendor, [[3.0, 4.0]])))
        client = EmbeddingAPIClient(vendor, model="m")

        vectors = client.embed(["hello"], "document")

        assert vectors.shape == (1, 2)
        assert vectors[0].tolist() == pytest.approx([0.6, 0.8]), "embeddings are L2-normalised"

    @pytest.mark.parametrize(
        "input_type, expected",
        [("query", "search_query"), ("document", "search_document")],
    )
    def test_cohere_distinguishes_queries_from_documents(self, http, input_type: str, expected: str) -> None:
        """Asymmetric embedders return different vectors per side; sending the
        wrong one silently degrades recall."""
        http.queue.append(_Response(200, _vendor_payload("cohere", [[1.0, 0.0]])))

        EmbeddingAPIClient("cohere", model="m").embed(["x"], input_type)  # type: ignore[arg-type]

        assert http.calls[0]["json"]["input_type"] == expected

    @pytest.mark.parametrize(
        "input_type, expected",
        [("query", "RETRIEVAL_QUERY"), ("document", "RETRIEVAL_DOCUMENT")],
    )
    def test_gemini_sets_the_task_type(self, http, input_type: str, expected: str) -> None:
        http.queue.append(_Response(200, _vendor_payload("gemini", [[1.0, 0.0]])))

        EmbeddingAPIClient("gemini", model="m").embed(["x"], input_type)  # type: ignore[arg-type]

        assert http.calls[0]["json"]["requests"][0]["taskType"] == expected
        assert "x-goog-api-key" in http.calls[0]["headers"]

    def test_requests_are_split_at_the_vendor_batch_limit(self, http, monkeypatch) -> None:
        monkeypatch.setitem(
            VENDORS, "openai", VENDORS["openai"].__class__(**{**VENDORS["openai"].__dict__, "max_batch": 2})
        )
        for _ in range(3):
            http.queue.append(_Response(200, _vendor_payload("openai", [[1.0, 0.0], [0.0, 1.0]])))

        EmbeddingAPIClient("openai", model="m").embed(["a", "b", "c", "d", "e"])

        assert [len(call["json"]["input"]) for call in http.calls] == [2, 2, 1]

    def test_the_configured_batch_size_is_the_request_size_up_to_the_vendor_limit(self, http) -> None:
        """An encoder config's batch_size reaches the requests, and cannot exceed what the vendor accepts."""
        from rcp_ndcg.retrieval import OpenAICompatibleEncoder
        from rcp_ndcg.retrieval._api import _encoder

        for _ in range(3):
            http.queue.append(_Response(200, _vendor_payload("openai", [[1.0, 0.0], [0.0, 1.0]])))
        config = OpenAICompatibleEncoder(model="m", base_url="https://api.openai.com/v1", batch_size=2)

        _encoder(config).client.embed(["a", "b", "c", "d", "e"])

        assert [len(call["json"]["input"]) for call in http.calls] == [2, 2, 1]
        assert EmbeddingAPIClient("cohere", model="m", batch_size=1000).batch_size == VENDORS["cohere"].max_batch

    def test_unknown_vendor_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown embedding vendor"):
            EmbeddingAPIClient("notavendor", model="m")

    def test_missing_key_names_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

        with pytest.raises(CredentialsError) as caught:
            EmbeddingAPIClient("voyage", model="m").embed(["x"])
        assert "VOYAGE_API_KEY" in (caught.value.hint or "")

    def test_empty_input_makes_no_request(self, http) -> None:
        assert EmbeddingAPIClient("openai", model="m").embed([]).shape[0] == 0
        assert http.calls == []


class TestRetries:
    def test_rate_limit_is_retried(self, http) -> None:
        http.queue.extend(
            [
                _Response(429, {}, {"retry-after": "1"}),
                _Response(200, _vendor_payload("openai", [[1.0, 0.0]])),
            ]
        )

        assert EmbeddingAPIClient("openai", model="m").embed(["x"]).shape == (1, 2)
        assert len(http.calls) == 2

    def test_server_errors_are_retried(self, http) -> None:
        http.queue.extend([_Response(503), _Response(200, _vendor_payload("openai", [[1.0, 0.0]]))])

        EmbeddingAPIClient("openai", model="m").embed(["x"])

        assert len(http.calls) == 2

    def test_client_errors_fail_immediately(self, http) -> None:
        """A 400 will fail identically on every attempt."""
        http.queue.append(_Response(400, {"error": "bad model"}))

        with pytest.raises(ProviderError, match="400"):
            EmbeddingAPIClient("openai", model="m").embed(["x"])

        assert len(http.calls) == 1

    def test_retries_are_bounded(self, http, monkeypatch: pytest.MonkeyPatch) -> None:
        http.queue.extend([_Response(503)] * 10)

        with pytest.raises(ProviderError):
            EmbeddingAPIClient("openai", model="m", max_retries=2).embed(["x"])

        assert len(http.calls) == 3
