"""The judge client against an in-process endpoint: requests, retries, outages, gates and accounting."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError
from rcp_ndcg_core.content import Content, MediaRef, VideoPart

from rcp_ndcg.errors import CapabilityError, CredentialsError, ProviderError
from rcp_ndcg.llm import JudgeClient, JudgeConfig, Usage
from rcp_ndcg.llm.client import (
    REASONING_WATCH,
    BackendUnavailableError,
    Completion,
    CompletionInput,
    RequestRejectedError,
)
from rcp_ndcg.testing import FakeJudge


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
    transport = httpx.MockTransport(endpoint)
    return JudgeClient(JudgeConfig(**settings), http_client=httpx.AsyncClient(transport=transport))


def _ask(client: JudgeClient, request: CompletionInput | None = None):
    return asyncio.run(client.complete(request or CompletionInput(user_prompt="judge this")))


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(JudgeClient, "BACKOFF_S", 0.001)
    monkeypatch.setattr(JudgeClient, "MAX_BACKOFF_S", 0.002)


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

    def test_the_fake_config_builds_the_fake_judge(self) -> None:
        client = JudgeClient.from_config(JudgeConfig.fake(seed=3))
        assert isinstance(client, FakeJudge) and client.seed == 3 and client.model == "fake"

    def test_a_floating_alias_on_the_openai_api_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="floating alias"):
            JudgeConfig(base_url="https://api.openai.com/v1", model="gpt-5")
        assert JudgeConfig(base_url="https://api.openai.com/v1", model="gpt-5-2025-08-07")
        assert JudgeConfig(base_url="https://api.openai.com/v1", model="gpt-5", allow_floating_model=True)
        assert JudgeConfig(base_url="http://localhost:8000/v1", model="gpt-oss-120b")


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
        client = JudgeClient(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)))
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


class TestOutages:
    def test_the_sdk_retries_a_transient_failure(self) -> None:
        endpoint = Endpoint(503)
        client = _client(endpoint, max_retries=1)
        assert _ask(client).response == "ok" and len(endpoint.requests) == 2

    def test_a_down_endpoint_parks_the_request_until_it_answers(self) -> None:
        endpoint = Endpoint(503, 429, 502, 503)
        assert _ask(_client(endpoint)).response == "ok"
        assert len(endpoint.requests) == 5

    def test_parking_ends_after_wait_on_outage(self) -> None:
        endpoint = Endpoint(*[503] * 1000)
        with pytest.raises(BackendUnavailableError, match="wait_on_outage_s"):
            _ask(_client(endpoint, wait_on_outage_s=0.02))


class _Queued(JudgeClient):
    """Answers take ``ANSWER_S``; the sends named in ``down`` find the endpoint unreachable."""

    ANSWER_S = 0.15

    def __init__(self, config: JudgeConfig, down: set[int]) -> None:
        super().__init__(config)
        self.down, self.sends = down, 0

    async def _send(self, request: CompletionInput, replica=None) -> Completion:
        self.sends += 1
        if self.sends in self.down:
            raise httpx.ConnectError("connection refused")
        await asyncio.sleep(self.ANSWER_S)
        return Completion(response="{}")


def _queue(client: JudgeClient, requests: int) -> list:
    async def main() -> list:
        prompts = (CompletionInput(user_prompt=str(i)) for i in range(requests))
        return await asyncio.gather(*(client.complete(p) for p in prompts), return_exceptions=True)

    return asyncio.run(main())


class TestOutageClock:
    """``wait_on_outage_s`` counts from a request's first unavailable failure, never its time in the queue."""

    CONFIG = JudgeConfig(base_url="http://127.0.0.1:9/v1", model="m", concurrency=1, wait_on_outage_s=0.2)

    def test_a_blip_after_a_long_queue_is_waited_out(self) -> None:
        # The fourth request queued 0.45 s (three answers of 0.15 s) behind the others, longer than
        # wait_on_outage_s; its one failed send is a short outage, which it waits out.
        client = _Queued(self.CONFIG, down={4})
        results = _queue(client, 5)
        assert all(isinstance(result, Completion) for result in results), results
        assert client.sends == 6

    def test_the_message_states_how_long_the_endpoint_was_unavailable(self) -> None:
        client = _Queued(self.CONFIG, down=set(range(3, 10_000)))
        results = _queue(client, 3)
        assert isinstance(results[0], Completion) and isinstance(results[1], Completion)
        error = results[2]
        assert isinstance(error, BackendUnavailableError), error
        seconds = float(str(error).split("was unavailable for ")[1].split("s ")[0])
        assert 0.2 <= seconds < 0.3  # the outage it saw, not the 0.3 s it also spent queued


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


@pytest.mark.parametrize("concurrency", [8, 256])
def test_the_connection_pool_holds_as_many_requests_as_the_concurrency(concurrency: int) -> None:
    """httpx's default pool holds 100 connections, so a concurrency of 256 once ran at most 100 requests at once."""
    client = JudgeClient(JudgeConfig(base_url="http://judge.test/v1", model="m", concurrency=concurrency))
    pool = client._transport()._transport._pool
    assert pool._max_connections >= concurrency and pool._max_keepalive_connections >= concurrency


class Replicas:
    """Several OpenAI-compatible replicas behind one mock transport: each host answers from its own status script,
    after ``delay`` seconds, and counts its requests and the most it held in flight at once."""

    def __init__(self, scripts: dict[str, list[int]], *, delay: float = 0.0) -> None:
        self.scripts = {host: list(statuses) for host, statuses in scripts.items()}
        self.delay = delay
        self.requests = dict.fromkeys(scripts, 0)
        self.in_flight = dict.fromkeys(scripts, 0)
        self.peak = dict.fromkeys(scripts, 0)

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        self.requests[host] += 1
        self.in_flight[host] += 1
        self.peak[host] = max(self.peak[host], self.in_flight[host])
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight[host] -= 1
        script = self.scripts[host]
        status = script.pop(0) if script else 200
        return httpx.Response(status, json=_answer() if status == 200 else {"error": {"message": "down"}})

    def client(self, **config) -> JudgeClient:
        urls = [f"http://{host}/v1" for host in self.scripts]
        settings = {"base_url": urls, "model": "m", "max_retries": 0, **config}
        return JudgeClient(JudgeConfig(**settings), http_client=httpx.AsyncClient(transport=httpx.MockTransport(self)))


def _ask_many(client: JudgeClient, count: int) -> list:
    async def many() -> list:
        return await asyncio.gather(*(client.complete(CompletionInput(user_prompt=f"q{i}")) for i in range(count)))

    return asyncio.run(many())


class TestReplicas:
    def test_requests_spread_over_the_replicas_by_requests_in_flight(self) -> None:
        replicas = Replicas({"a": [], "b": [], "c": []}, delay=0.01)
        answers = _ask_many(replicas.client(concurrency=6), 30)
        assert len(answers) == 30
        assert replicas.requests == {"a": 10, "b": 10, "c": 10}
        # The concurrency is shared: six in flight over three replicas is two on each, never more.
        assert replicas.peak == {"a": 2, "b": 2, "c": 2}

    def test_a_down_replica_is_set_aside_and_its_request_moves_to_a_live_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(JudgeClient, "BACKOFF_S", 60.0)
        replicas = Replicas({"a": [503] * 100, "b": []})
        client = replicas.client()
        answers = [_ask(client) for _ in range(5)]
        assert [answer.response for answer in answers] == ["ok"] * 5
        assert replicas.requests == {"a": 1, "b": 5}  # a failed once, then b carried every request

    def test_a_replica_that_answers_again_is_used_again(self) -> None:
        replicas = Replicas({"a": [503], "b": []})
        client = replicas.client()
        for _ in range(6):
            _ask(client)
            asyncio.run(asyncio.sleep(0.005))  # past the backoff
        assert replicas.requests["a"] > 1

    def test_with_every_replica_down_the_request_parks_until_one_answers(self) -> None:
        replicas = Replicas({"a": [503, 503, 503], "b": [502, 502, 502]})
        assert _ask(replicas.client()).response == "ok"
        assert replicas.requests["a"] + replicas.requests["b"] == 7

    def test_parking_over_every_replica_ends_after_wait_on_outage(self) -> None:
        replicas = Replicas({"a": [503] * 1000, "b": [503] * 1000})
        with pytest.raises(BackendUnavailableError, match="http://a/v1, http://b/v1"):
            _ask(replicas.client(wait_on_outage_s=0.02))

    def test_a_request_that_keeps_failing_on_a_replica_that_serves_others_is_refused(self) -> None:
        def serve(request: httpx.Request) -> httpx.Response:
            stuck = "stuck" in request.content.decode() or request.url.host == "b"
            return httpx.Response(503 if stuck else 200, json={"error": {"message": "x"}} if stuck else _answer())

        config = JudgeConfig(base_url=["http://a/v1", "http://b/v1"], model="m", max_retries=0)
        client = JudgeClient(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)))

        async def both() -> list:
            prompts = [CompletionInput(user_prompt="stuck"), CompletionInput(user_prompt="fine")]
            return await asyncio.gather(*(client.complete(p) for p in prompts), return_exceptions=True)

        stuck, fine = asyncio.run(both())
        assert isinstance(stuck, RequestRejectedError) and fine.response == "ok"


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
        client = JudgeClient(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)))
        with pytest.raises(CapabilityError, match="refused the number of") as caught:
            _ask(client)
        assert "per-request media limit" in (caught.value.hint or "") and "serving.md" in caught.value.hint

    def test_a_pixel_refusal_is_not_a_media_count_refusal(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "Image dimensions 9000x9000 exceed the limit"}})

        config = JudgeConfig(base_url="http://judge.test/v1", model="m", max_retries=0)
        client = JudgeClient(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)))
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
        return JudgeClient(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(answer)))

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
        client = JudgeClient(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)))
        (engine,) = asyncio.run(client.probe())
        assert (engine.model, engine.owned_by, engine.max_model_len) == ("m", "vllm", 131072)
        assert engine.headers == {"server": "uvicorn"} and engine.error is None
        _ask(client)
        assert client.engines[0].system_fingerprint == "vllm-0.30.0-tp4-abcdef12"

    def test_an_endpoint_that_cannot_be_read_is_recorded_not_raised(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        config = JudgeConfig(base_url=["http://a/v1", "http://b/v1"], model="m")
        client = JudgeClient(config, http_client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)))
        engines = asyncio.run(client.probe())
        assert [engine.url for engine in engines] == ["http://a/v1", "http://b/v1"]
        assert all(engine.error and "ConnectError" in engine.error for engine in engines)

    def test_a_server_that_does_not_serve_the_judges_model_is_named(self, caplog: pytest.LogCaptureFixture) -> None:
        def serve(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"object": "list", "data": [{"id": "org/weights"}]})

        client = JudgeClient(
            JudgeConfig(base_url="http://judge.test/v1", model="m"),
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
        )
        with caplog.at_level("WARNING", logger="rcp_ndcg"):
            (engine,) = asyncio.run(client.probe())
        assert engine.model == "org/weights"
        assert any("--served-model-name m" in record.getMessage() for record in caplog.records)

    def test_the_fake_judge_has_no_engine(self) -> None:
        assert asyncio.run(JudgeClient.from_config(JudgeConfig.fake(0)).probe()) == []


class _PlainHttpx(JudgeClient):
    """A client whose transport is httpx itself, as an extension that replaces ``_send`` would have."""

    def __init__(self, handler, **config) -> None:
        settings = {"base_url": "http://judge.test/v1", "model": "m", **config}
        super().__init__(JudgeConfig(**settings))
        self.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def _send(self, request: CompletionInput, replica=None):
        response = await self.http.post(f"{replica.url}/chat/completions", json={"model": self.model})
        response.raise_for_status()
        return Completion(response=response.json()["choices"][0]["message"]["content"])


class TestTransportErrors:
    """httpx's own failures map like the SDK's: an outage parks, a refusal is typed."""

    @pytest.mark.parametrize("error", [httpx.ConnectError("refused"), httpx.ReadTimeout("slow")])
    def test_a_transport_failure_parks_then_answers(self, error: Exception) -> None:
        calls = []

        def flaky(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) < 3:
                raise error
            return httpx.Response(200, json=_answer())

        assert _ask(_PlainHttpx(flaky)).response == "ok" and len(calls) == 3

    def test_a_lasting_transport_failure_ends_as_backend_unavailable(self) -> None:
        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        with pytest.raises(BackendUnavailableError, match="ConnectError"):
            _ask(_PlainHttpx(down, wait_on_outage_s=0.02))

    @pytest.mark.parametrize(("status", "error"), [(503, None), (400, RequestRejectedError), (401, CredentialsError)])
    def test_an_httpx_status_error_maps_like_the_sdks(self, status: int, error) -> None:
        answered = []

        def respond(request: httpx.Request) -> httpx.Response:
            if not answered and status == 503:
                answered.append(1)
                return httpx.Response(503, json={"error": {"message": "busy"}})
            return httpx.Response(status if status != 503 else 200, json=_answer())

        if error is None:
            assert _ask(_PlainHttpx(respond)).response == "ok"
        else:
            with pytest.raises(error):
                _ask(_PlainHttpx(respond))


def test_a_probe_in_a_later_event_loop_reads_the_endpoint_again() -> None:
    """Each judging pass may run in its own event loop; the probe must not reuse connections of a closed one."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Models(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep-alive: a connection outlives the first event loop

        def do_GET(self) -> None:
            body = json.dumps({"object": "list", "data": [{"id": "m", "owned_by": "engine"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Models)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = JudgeClient(JudgeConfig(base_url=f"http://127.0.0.1:{server.server_port}/v1", model="m"))
        for _ in range(2):
            (engine,) = asyncio.run(client.probe())
            assert engine.error is None and engine.owned_by == "engine", engine.error
    finally:
        server.shutdown()
