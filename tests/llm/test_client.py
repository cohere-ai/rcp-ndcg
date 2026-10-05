"""The judge client: config, the request it sends, the endpoint's checks, probe and accounting.

The transport below it (replicas, retries, parking, the status map, the pool) has its own tests in
tests/inference/test_transport.py, and the ``openai_chat`` wire its own in tests/inference/test_chat_adapter.py;
these test the judge client above them, against an in-process endpoint.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from rcp_ndcg_core.content import Content, MediaRef, VideoPart

from rcp_ndcg.errors import CapabilityError, ConfigError, CredentialsError, ProviderError
from rcp_ndcg.llm import JudgeClient, JudgeConfig, Usage
from rcp_ndcg.llm.client import REASONING_WATCH, CompletionInput, RequestRejectedError


def _answer(content: str = "ok") -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1_000, "completion_tokens": 100, "total_tokens": 1_100},
    }


class Endpoint:
    """An OpenAI-compatible endpoint that answers from a script of statuses and records every request."""

    def __init__(self, *statuses: int) -> None:
        self.statuses = list(statuses)
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        status = self.statuses.pop(0) if self.statuses else 200
        return httpx.Response(status, json=_answer() if status == 200 else {"error": {"message": "down"}})


def _client(endpoint: Endpoint, **config) -> JudgeClient:
    settings = {"base_url": "http://judge.test/v1", "model": "m", "max_retries": 0, **config}
    return JudgeClient(JudgeConfig(**settings), httpx_transport=httpx.MockTransport(endpoint))


def _ask(client: JudgeClient, request: CompletionInput | None = None):
    return asyncio.run(client.complete(request or CompletionInput(user_prompt="judge this")))


class TestConfig:
    def test_load_reads_a_config_and_refuses_unknown_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "judge.yaml"
        path.write_text("base_url: http://h/v1/\nmodel: m\nconcurrency: 2\n")
        config = JudgeConfig.load(path)
        assert (config.base_url, config.concurrency) == ("http://h/v1", 2)
        path.write_text("base_url: http://h/v1\nmodel: m\nnum_criteria: 5\n")
        with pytest.raises(ValueError, match="num_criteria"):
            JudgeConfig.load(path)

    def test_the_hosted_judges_windows_fit_its_input_limit(self) -> None:
        """GPT-5 takes at most 272,000 input tokens (its 400,000-token window includes up to 128,000 of output)."""
        from rcp_ndcg.llm.judging import window_tokens

        config = JudgeConfig.load("gpt5_hosted")
        per_doc = window_tokens(config, 10, overhead_tokens=5_000)
        assert per_doc is not None and 5_000 + 10 * per_doc + (config.max_output_tokens or 0) <= 272_000

    def test_identity_is_the_content_fields_only(self) -> None:
        config = JudgeConfig(base_url="http://a/v1", model="m", revision="abc")
        moved = config.model_copy(update={"base_url": "http://b/v1", "concurrency": 2})
        assert config.identity() == moved.identity()
        assert config.identity() != config.model_copy(update={"temperature": 0.7}).identity()
        assert "api" not in config.identity(), "the unset wire adapter stays out of the payload"

    def test_api_defaults_to_the_chat_wire_and_keeps_the_identity_unchanged(self) -> None:
        """The unset ``api`` resolves to ``openai_chat`` without entering the identity payload."""
        client = _client(Endpoint())
        assert client._wire()[0].name == "openai_chat"  # noqa: SLF001
        assert (
            JudgeConfig(base_url="http://a/v1", model="m").identity() == {"model": "m"}
            or "api" not in JudgeConfig(base_url="http://a/v1", model="m").identity()
        )

    def test_a_missing_key_names_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RCP_NDCG_TEST_JUDGE_KEY", raising=False)
        config = JudgeConfig(base_url="http://a/v1", model="m", api_key_env="RCP_NDCG_TEST_JUDGE_KEY")
        with pytest.raises(CredentialsError, match="RCP_NDCG_TEST_JUDGE_KEY") as caught:
            config.api_key()
        assert caught.value.details == {"variable": "RCP_NDCG_TEST_JUDGE_KEY"}
        monkeypatch.setenv("RCP_NDCG_TEST_JUDGE_KEY", "k")
        assert config.api_key() == "k"
        assert JudgeConfig(base_url="http://a/v1", model="m").api_key() == "EMPTY"

    def test_base_url_is_one_url_or_a_list_of_replicas_and_neither_is_identity(self) -> None:
        one = JudgeConfig(base_url="http://a/v1/", model="m")
        replicas = JudgeConfig(base_url=["http://a/v1/", "http://b/v1"], model="m")
        assert one.urls == ("http://a/v1",) and replicas.urls == ("http://a/v1", "http://b/v1")
        assert one.identity() == replicas.identity()
        for bad in ([], ["http://a/v1", "http://a/v1/"], ["fake://seed/0", "http://a/v1"]):
            with pytest.raises(ValidationError):
                JudgeConfig(base_url=bad, model="m")

    def test_a_floating_alias_on_the_openai_api_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="floating alias"):
            JudgeConfig(base_url="https://api.openai.com/v1", model="gpt-5")
        assert JudgeConfig(base_url="https://api.openai.com/v1", model="gpt-5-2025-08-07")
        assert JudgeConfig(base_url="https://api.openai.com/v1", model="gpt-5", allow_floating_model=True)
        assert JudgeConfig(base_url="http://localhost:8000/v1", model="gpt-oss-120b")

    def test_an_api_of_another_role_is_refused_by_the_registry(self) -> None:
        from rcp_ndcg.inference.adapters import known_adapters

        client = _client(Endpoint(), api="openai_embeddings")
        with pytest.raises(ConfigError) as caught:
            _ask(client)
        message = str(caught.value)
        assert "openai_embeddings" in message and "judge" in message
        assert "openai_chat" in (caught.value.hint or ""), "the hint lists the judge role's own adapters"
        assert set(known_adapters("judge")) >= {"openai_chat"}

    def test_a_replaced_config_is_built_from_at_the_next_call(self) -> None:
        """The offline fakes' tests replace the config after construction (a model_copy); the wire follows."""
        endpoint = Endpoint()
        client = _client(endpoint)
        client.config = client.config.model_copy(update={"temperature": 0.3})
        _ask(client)
        assert endpoint.requests[0]["temperature"] == 0.3

    def test_a_replaced_config_keeps_the_failures_of_the_wire_it_replaces(self) -> None:
        client = JudgeClient(
            JudgeConfig(base_url="http://judge.test/v1", model="m", max_retries=0, api_key_env=None),
            httpx_transport=httpx.MockTransport(lambda request: httpx.Response(401, json={"error": {"message": "x"}})),
        )
        with pytest.raises(CredentialsError):
            _ask(client)
        assert client.usage.failed_requests == 1
        client.config = client.config.model_copy(update={"temperature": 0.3})
        assert client.usage.failed_requests == 1, "the replaced wire's failures stay counted"

    def test_a_replaced_config_closes_the_transport_it_replaces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import rcp_ndcg.inference.transport as transport_module

        closed: list[str] = []
        monkeypatch.setattr(transport_module.Transport, "aclose", lambda self: closed.append(self.endpoint.model))
        client = _client(Endpoint())
        _ask(client)  # builds the wire
        client.config = client.config.model_copy(update={"temperature": 0.3})
        _ask(client)  # builds the replacement
        client.config = client.config.model_copy(update={"temperature": 0.5})
        assert closed == ["m", "m"], "each replaced config closes the wire it replaces"


class TestRequests:
    def test_the_request_carries_the_sampling_settings(self) -> None:
        endpoint = Endpoint()
        client = _client(endpoint, max_output_tokens=64, extra_body={"reasoning_effort": "low", "top_k": 1})
        assert _ask(client).response == "ok"
        (body,) = endpoint.requests
        # max_tokens is refused by the OpenAI API for reasoning models; vLLM and SGLang read either.
        assert body["model"] == "m" and body["max_completion_tokens"] == 64 and "max_tokens" not in body
        assert (body["reasoning_effort"], body["top_k"]) == ("low", 1)
        assert body["messages"] == [{"role": "user", "content": "judge this"}]

    def test_no_temperature_is_sent_by_default(self) -> None:
        """The server's default sampling applies unless the config sets a temperature (owner decision)."""
        endpoint = Endpoint()
        _ask(_client(endpoint))
        _ask(_client(endpoint, temperature=0.3))
        default, explicit = endpoint.requests
        assert "temperature" not in default and explicit["temperature"] == 0.3

    def test_usage_counts_calls_and_tokens_and_carries_no_usd(self) -> None:
        client = _client(Endpoint())
        _ask(client)
        _ask(client)
        assert (client.usage.requests, client.usage.input_tokens) == (2, 2_000)
        assert not any("usd" in name or "cost" in name for name in Usage.model_fields)

    def test_an_endpoint_refusing_the_answer_schema_is_a_capability_error(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "response_format json_schema is not supported"}})

        config = JudgeConfig(base_url="http://judge.test/v1", model="m", max_retries=0, decoding="json_schema")
        client = JudgeClient(config, httpx_transport=httpx.MockTransport(refuse))
        schema = {"type": "json_schema", "json_schema": {"name": "a", "schema": {"type": "object"}}}
        with pytest.raises(CapabilityError, match="refused the answer schema") as caught:
            _ask(client, CompletionInput(user_prompt="judge this", response_format=schema))
        assert "decoding: free" in (caught.value.hint or "")
        with pytest.raises(RequestRejectedError):  # the same words without a schema sent: an ordinary refusal
            _ask(client)

    def test_a_refused_request_is_raised_not_parked(self) -> None:
        endpoint = Endpoint(400)
        client = _client(endpoint)
        with pytest.raises(RequestRejectedError, match="HTTP 400"):
            _ask(client)
        assert len(endpoint.requests) == 1 and client.usage.failed_requests == 1

    @pytest.mark.parametrize(
        ("status", "error"), [(401, CredentialsError), (403, CredentialsError), (404, ProviderError)]
    )
    def test_an_endpoint_that_cannot_serve_this_client_is_not_a_refused_request(self, status: int, error) -> None:
        with pytest.raises(error) as caught:
            _ask(_client(Endpoint(status)))
        assert not isinstance(caught.value, RequestRejectedError)


class TestGates:
    PAGE = Content.from_image("file:///page.png")

    def test_images_to_a_text_judge_are_refused_before_sending(self) -> None:
        endpoint = Endpoint()
        request = CompletionInput(user_prompt="judge", user_content=self.PAGE)
        with pytest.raises(CapabilityError, match="images"):
            _ask(_client(endpoint), request)
        assert endpoint.requests == []

    def test_video_to_a_judge_not_declared_to_read_it_is_refused_before_sending(self) -> None:
        endpoint = Endpoint()
        clip = Content.from_parts([VideoPart(ref=MediaRef(uri="file:///clip.mp4", mime="video/mp4"))])
        with pytest.raises(CapabilityError, match="video"):
            _ask(
                _client(endpoint, max_images=4),
                CompletionInput(user_prompt="j", user_content=clip),
            )
        assert endpoint.requests == []

    def test_a_window_over_the_video_limit_is_refused(self) -> None:
        clip = VideoPart(ref=MediaRef(uri="file:///clip.mp4", mime="video/mp4"))
        client = _client(Endpoint(), max_videos=1)
        with pytest.raises(CapabilityError, match="max_videos"):
            _ask(client, CompletionInput(user_prompt="judge", user_content=Content.from_parts([clip, clip])))

    def test_a_window_over_the_image_limit_is_refused(self) -> None:
        pages = Content.from_parts([*self.PAGE.parts, *self.PAGE.parts])
        client = _client(Endpoint(), max_images=1)
        with pytest.raises(CapabilityError, match="max_images"):
            _ask(client, CompletionInput(user_prompt="judge", user_content=pages))


class TestServerChecks:
    @pytest.mark.parametrize(
        "message",
        [
            "At most 4 image(s) may be provided in one prompt. Set `--limit-mm-per-prompt` to increase this limit.",
            "Image count 12 exceeds limit 10 per request.",
            "Too many videos in the request",
        ],
    )
    def test_a_refused_media_count_is_a_capability_error_naming_the_servers_limit(self, message: str) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": message}})

        config = JudgeConfig(base_url="http://judge.test/v1", model="m", max_retries=0)
        client = JudgeClient(config, httpx_transport=httpx.MockTransport(refuse))
        with pytest.raises(CapabilityError, match="refused the number of") as caught:
            _ask(client)
        assert "per-request media limit" in (caught.value.hint or "") and "serving.md" in caught.value.hint

    def test_a_pixel_refusal_is_not_a_media_count_refusal(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "Image dimensions 9000x9000 exceed the limit"}})

        config = JudgeConfig(base_url="http://judge.test/v1", model="m", max_retries=0)
        client = JudgeClient(config, httpx_transport=httpx.MockTransport(refuse))
        with pytest.raises(RequestRejectedError):
            _ask(client)

    @staticmethod
    def _reasoning_client(reasoning: str | None) -> JudgeClient:
        def answer(request: httpx.Request) -> httpx.Response:
            body = _answer('{"ranking": [1]}')
            if reasoning is not None:
                body["choices"][0]["message"]["reasoning_content"] = reasoning
            return httpx.Response(200, json=body)

        config = JudgeConfig(base_url="http://judge.test/v1", model="m", max_retries=0, decoding="json_schema")
        return JudgeClient(config, httpx_transport=httpx.MockTransport(answer))

    SCHEMA = {"type": "json_schema", "json_schema": {"name": "a", "schema": {"type": "object"}}}

    @pytest.mark.parametrize(("reasoning", "warned"), [(None, 1), ("", 1), ("I compare the documents.", 0)])
    def test_answers_without_reasoning_under_a_schema_warn_once(
        self, reasoning: str | None, warned: int, caplog: pytest.LogCaptureFixture
    ) -> None:
        client = self._reasoning_client(reasoning)
        with caplog.at_level("WARNING", logger="rcp_ndcg"):
            for _ in range(3 * REASONING_WATCH):
                _ask(client, CompletionInput(user_prompt="judge", response_format=self.SCHEMA))
        assert sum("reasoning parser" in record.getMessage() for record in caplog.records) == warned

    def test_free_decoding_is_not_watched_for_reasoning(self, caplog: pytest.LogCaptureFixture) -> None:
        client = self._reasoning_client(None)
        with caplog.at_level("WARNING", logger="rcp_ndcg"):
            for _ in range(2 * REASONING_WATCH):
                _ask(client)
        assert not any("reasoning parser" in record.getMessage() for record in caplog.records)


class TestProbe:
    def test_the_probe_records_what_each_replica_reports(self) -> None:
        def serve(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                models = [{"id": "other", "owned_by": "x"}, {"id": "m", "owned_by": "vllm", "max_model_len": 131072}]
                return httpx.Response(200, json={"object": "list", "data": models}, headers={"server": "uvicorn"})
            return httpx.Response(200, json={**_answer(), "system_fingerprint": "vllm-0.30.0-tp4-abcdef12"})

        config = JudgeConfig(base_url="http://judge.test/v1", model="m", max_retries=0)
        client = JudgeClient(config, httpx_transport=httpx.MockTransport(serve))
        (engine,) = asyncio.run(client.probe())
        assert (engine.model, engine.owned_by, engine.max_model_len) == ("m", "vllm", 131072)
        assert engine.headers == {"server": "uvicorn"} and engine.error is None
        _ask(client)
        assert client.engines[0].system_fingerprint == "vllm-0.30.0-tp4-abcdef12"

    def test_a_completion_records_its_fingerprint_even_where_the_probe_found_nothing(self) -> None:
        def serve(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(500)
            return httpx.Response(200, json={**_answer(), "system_fingerprint": "vllm-0.30.0-tp4-abcdef12"})

        config = JudgeConfig(base_url="http://judge.test/v1", model="m", max_retries=0)
        client = JudgeClient(config, httpx_transport=httpx.MockTransport(serve))
        (probed,) = asyncio.run(client.probe())
        assert probed.error is not None
        _ask(client)
        assert client.engines[0].system_fingerprint == "vllm-0.30.0-tp4-abcdef12"

    def test_an_endpoint_that_cannot_be_read_is_recorded_not_raised(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        config = JudgeConfig(base_url=["http://a/v1", "http://b/v1"], model="m")
        client = JudgeClient(config, httpx_transport=httpx.MockTransport(refuse))
        engines = asyncio.run(client.probe())
        assert [engine.url for engine in engines] == ["http://a/v1", "http://b/v1"]
        assert all(engine.error and "ConnectError" in engine.error for engine in engines)

    def test_a_server_that_does_not_serve_the_judges_model_is_named(self, caplog: pytest.LogCaptureFixture) -> None:
        def serve(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"object": "list", "data": [{"id": "org/weights"}]})

        client = JudgeClient(
            JudgeConfig(base_url="http://judge.test/v1", model="m"),
            httpx_transport=httpx.MockTransport(serve),
        )
        with caplog.at_level("WARNING", logger="rcp_ndcg"):
            (engine,) = asyncio.run(client.probe())
        assert engine.model == "org/weights"
        assert any("--served-model-name m" in record.getMessage() for record in caplog.records)

    def test_the_fake_judge_has_no_engine(self) -> None:
        assert asyncio.run(JudgeClient.from_config(JudgeConfig.fake(0)).probe()) == []
