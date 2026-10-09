"""The client base (``clients/_base.py``): the one ground every role client shares.

The base owns what R5/R14/R15 asked one home for: the adapter lookup within the client's role, the
hosted profile's default base URL, the transport (built from the config unless a ``Sender`` is given),
the sync bridge (one rule: a non-transport sender must provide ``run``), and the lifecycle
(``close()`` sync, ``async aclose()`` true async, context managers).
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import pytest
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import ConfigError, CredentialsError, RequestRejectedError
from rcp_ndcg.inference import EncodeRole
from rcp_ndcg.inference.clients import EmbeddingClient, PoolingClient, RerankClient
from rcp_ndcg.inference.clients._base import RoleClient
from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.inference.types import Call, Reply, Usage


class RecordingSender:
    """A sender with the sync bridge (``run``), recording its lifecycle calls."""

    def __init__(self) -> None:
        self.calls: list[Call] = []
        self.closed = 0

    async def send(self, calls: Any) -> list[Reply]:
        self.calls.extend(calls)
        return []

    async def probe(self) -> list[Any]:
        return []

    @property
    def usage(self) -> Usage:
        return Usage()

    def run(self, coroutine: Any) -> Any:
        return asyncio.run(coroutine)

    def close(self) -> None:
        self.closed += 1


class Bridgeless(RecordingSender):
    """A sender with no ``run``: only send/probe/usage."""

    run = None  # type: ignore[assignment]


def test_every_role_client_derives_from_the_base() -> None:
    """One class, three subclasses: the shared ground is written once."""
    for client in (EmbeddingClient, RerankClient, PoolingClient):
        assert issubclass(client, RoleClient)


class TestSyncBridge:
    def test_a_sender_without_run_is_refused_at_construction(self, tokenizer_json: str) -> None:
        """The one sync-bridge rule: a Transport's ``run``, or the sender's own. Neither, no client."""
        with pytest.raises(ConfigError, match="run"):
            EmbeddingClient(
                EmbeddingEndpoint(api="openai_embeddings", model="m", tokenizer=tokenizer_json, max_tokens=8192),
                sender=Bridgeless(),
            )

    def test_a_transport_sender_is_accepted(self, tokenizer_json: str) -> None:
        from rcp_ndcg.inference.transport import Transport

        config = EmbeddingEndpoint(
            base_url="http://127.0.0.1:9000/v1", model="m", tokenizer=tokenizer_json, max_tokens=8192
        )
        client = EmbeddingClient(config, sender=Transport(config))
        client.close()


class TestLifecycle:
    def test_close_and_aclose_and_the_context_managers(self, tokenizer_json: str) -> None:
        sender = RecordingSender()
        client = EmbeddingClient(
            EmbeddingEndpoint(
                base_url="http://127.0.0.1:9000/v1", model="m", tokenizer=tokenizer_json, max_tokens=8192
            ),
            sender=sender,
        )
        with client:
            pass
        assert sender.closed == 1
        client.close()  # safe twice
        assert sender.closed == 2

        async def aclose() -> None:
            await client.aclose()

        asyncio.run(aclose())
        assert sender.closed == 3

    def test_close_closes_the_transport_the_client_built(self, tokenizer_json: str) -> None:
        from rcp_ndcg.inference.transport import Transport

        config = EmbeddingEndpoint(
            base_url="http://127.0.0.1:9000/v1", model="m", tokenizer=tokenizer_json, max_tokens=8192
        )
        client = EmbeddingClient(config)
        assert isinstance(client._sender, Transport)
        pool = client._sender._pool
        client.close()
        assert client._sender._pool is None and pool is None


class TestAdapterLookup:
    def test_the_adapter_is_looked_up_within_the_client_s_role(self, tokenizer_json: str) -> None:
        from rcp_ndcg.inference.adapters import get_adapter

        client = EmbeddingClient(
            EmbeddingEndpoint(
                base_url="http://127.0.0.1:9000/v1", model="m", tokenizer=tokenizer_json, max_tokens=8192
            ),
            sender=RecordingSender(),
        )
        assert client._adapter_cls is get_adapter("openai_embeddings", role="embed")

    def test_an_adapter_of_another_role_is_refused_by_the_base(self, tokenizer_json: str) -> None:
        """The role check is the base's, not each client's: a rerank api on an embed config is refused."""
        with pytest.raises(ConfigError, match="unknown embed adapter 'rerank'"):
            EmbeddingClient(
                EmbeddingEndpoint(
                    api="rerank",
                    base_url="http://127.0.0.1:9000/v1",
                    model="m",
                    tokenizer=tokenizer_json,
                    max_tokens=8192,
                ),
                sender=RecordingSender(),
            )


class TestHostedBaseUrl:
    def test_the_profile_url_fills_a_hosted_config(self) -> None:
        client = EmbeddingClient(EmbeddingEndpoint(api="cohere", model="m", max_tokens=1024), sender=RecordingSender())
        assert client.endpoint.base_url == "https://api.cohere.com/v2"
        assert client.config.base_url is None  # the config as given is untouched

    def test_a_profile_without_a_default_url_refuses_a_base_url_less_config(self, tokenizer_json: str) -> None:
        """A served engine has no public root: an adapter that declares none refuses a config without one."""
        from rcp_ndcg.inference import adapters as registry
        from rcp_ndcg.inference.adapters.base import AdapterRole, register_adapter

        class _NoDefaultEmbed:
            name = "probe_nodefault"
            role: ClassVar[AdapterRole] = "embed"
            HOSTED = False
            API_KEY_ENV = ()
            KEY_REQUIRED = False
            AUTH_HEADER = None
            DEFAULT_BASE_URL = None

            def calls(self, request: Any, *, model: str) -> list[Call]:
                return []

            def interpret(self, request: Any, replies: list[Reply]) -> Any:
                return None

            def usage(self, reply: Reply) -> None:
                return None

        saved = dict(registry.base._BUILTINS)
        register_adapter(_NoDefaultEmbed)  # type: ignore[arg-type]
        try:
            with pytest.raises(ConfigError, match="base_url"):
                EmbeddingClient(
                    EmbeddingEndpoint(api="probe_nodefault", model="m", tokenizer=tokenizer_json, max_tokens=8192),
                    sender=RecordingSender(),
                )
        finally:
            registry.base._BUILTINS.clear()
            registry.base._BUILTINS.update(saved)


class _FailingFanOut:
    """A sender that answers the first two of ten sends, fails the third, and parks the rest.

    The parked sends never finish: if the fan-out leaves a sibling running, the test sees it."""

    def __init__(self, *, fail: int = 2) -> None:
        self.fail = fail
        self.started = 0
        self.finished = 0
        self.hold = asyncio.Event()

    async def send(self, calls: Any) -> list[Reply]:
        index = self.started
        self.started += 1
        if index < self.fail:
            self.finished += 1
            return [self._answer(call) for call in calls]
        if index == self.fail:
            raise RequestRejectedError("the third request was refused")
        await self.hold.wait()  # never set: only cancellation ends these
        raise AssertionError("unreachable: a parked sibling was never cancelled")

    @staticmethod
    def _answer(call: Any) -> Reply:
        """One 2xx body per wire path, so the first two requests interpret cleanly."""
        body = call.json or {}
        if call.path == "/rerank":
            return Reply(
                200, {"results": [{"index": i, "relevance_score": 0.5} for i in range(len(body["documents"]))]}, {}
            )
        if call.path == "/pooling":
            count = len(body["input"])
            return Reply(
                200,
                {"data": [{"index": i, "data": [[1.0, 1.0]]} for i in range(count)], "usage": {"prompt_tokens": count}},
                {},
            )
        if call.path == "/embeddings":
            return Reply(200, {"data": [{"index": i, "embedding": [1.0, 1.0]} for i in range(len(body["input"]))]}, {})
        return Reply(200, {}, {})

    async def probe(self) -> list[Any]:
        return []

    @property
    def usage(self) -> Usage:
        return Usage()

    def run(self, coroutine: Any) -> Any:
        return asyncio.run(coroutine)


async def _no_pending_tasks() -> list[Any]:
    """The tasks still running in this loop, minus the caller's own."""
    return [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]


class TestNoSiblingLeftRunning:
    """A failing request cancels its siblings (R7): TaskGroup semantics in every client's fan-out."""

    def _embed(self, sender: Any, tokenizer_json: str) -> EmbeddingClient:
        from rcp_ndcg_core.content import Content

        return EmbeddingClient(
            EmbeddingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                batch_size=1,
                concurrency=10,
                tokenizer=tokenizer_json,
                max_tokens=8192,
            ),
            sender=sender,
        ), [Content.from_text(f"doc {index}") for index in range(10)]

    def test_embed_cancels_its_sibling_requests(self, tokenizer_json: str) -> None:
        from rcp_ndcg.inference import EncodeRole

        sender = _FailingFanOut()
        client, contents = self._embed(sender, tokenizer_json)

        async def run() -> None:
            with pytest.raises(RequestRejectedError, match="third request"):
                await client.aencode(contents, EncodeRole.DOCUMENT)
            assert sender.finished == 2, "only the two completed requests; the siblings were cancelled"
            assert await _no_pending_tasks() == []

        asyncio.run(run())

    def test_rerank_many_cancels_its_sibling_queries_and_checkpoints_nothing_after_the_failure(
        self, tokenizer_json: str
    ) -> None:
        from rcp_ndcg_core.records import RankingExample

        sender = _FailingFanOut()
        client = RerankClient(
            RerankEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                concurrency=10,
                tokenizer=tokenizer_json,
                max_tokens=8192,
                use_activation=False,
            ),
            sender=sender,
        )
        examples = [
            RankingExample(query=f"q{index}", id=f"q{index}", doc_ids=["d1"], docs=["doc"]) for index in range(10)
        ]
        seen: list[str] = []

        async def run() -> None:
            with pytest.raises(RequestRejectedError, match="third request"):
                await client.arerank_many(examples, checkpoint=lambda query_id, scores: seen.append(query_id))
            assert seen == ["q0", "q1"], "only the completed queries checkpointed; none after the failure"
            assert await _no_pending_tasks() == []

        asyncio.run(run())

    def test_pool_cancels_its_sibling_requests(self, tokenizer_json: str) -> None:
        from rcp_ndcg_core.content import Content

        sender = _FailingFanOut()
        client = PoolingClient(
            PoolingEndpoint(
                base_url="http://127.0.0.1:9000/v1",
                model="m",
                dim=2,
                batch_size=1,
                concurrency=10,
                tokenizer=tokenizer_json,
                max_tokens=8192,
            ),
            sender=sender,
        )
        contents = [Content.from_text(f"doc {index}") for index in range(10)]

        async def run() -> None:
            with pytest.raises(RequestRejectedError, match="third request"):
                await client.aencode(contents, EncodeRole.DOCUMENT)
            assert sender.finished == 2, "only the two completed requests; the siblings were cancelled"
            assert await _no_pending_tasks() == []

        asyncio.run(run())


class TestEngineAdapterRoles:
    """F7 (integration review): the two role vocabularies meet in one written mapping, and a config whose
    ``api`` selects an adapter of a different engine role is refused where the runners resolve engines."""

    def test_the_mapping_is_written_once(self) -> None:
        from rcp_ndcg.inference.adapters.base import ENGINE_ADAPTER_ROLES

        assert set(ENGINE_ADAPTER_ROLES) == {"judge", "encoder", "reranker"}
        assert ENGINE_ADAPTER_ROLES["judge"] == frozenset({"judge"})
        assert ENGINE_ADAPTER_ROLES["encoder"] == frozenset({"embed", "multi_vector"})
        assert ENGINE_ADAPTER_ROLES["reranker"] == frozenset({"rerank"})

    def test_adapter_roles_of_returns_the_mapping_s_values(self) -> None:
        from rcp_ndcg.inference.adapters.base import adapter_roles_of

        assert adapter_roles_of("encoder") == frozenset({"embed", "multi_vector"})

    def test_an_api_of_another_role_is_refused_by_the_engine_check(self) -> None:
        from rcp_ndcg.inference.adapters.base import check_engine_api

        check_engine_api("openai_embeddings", engine_role="encoder", where="serve.encoder")
        check_engine_api("vllm_pooling", engine_role="encoder", where="serve.encoder")
        check_engine_api("rerank", engine_role="reranker", where="serve.reranker")
        with pytest.raises(ConfigError, match="encoder"):
            check_engine_api("rerank", engine_role="encoder", where="serve.encoder")
        with pytest.raises(ConfigError, match="judge"):
            check_engine_api("rerank", engine_role="judge", where="serve.judge")

    def test_a_config_without_an_api_is_not_checked(self) -> None:
        from rcp_ndcg.inference.adapters.base import check_engine_api

        check_engine_api(None, engine_role="judge", where="serve.judge")

    def test_an_unknown_api_names_where_the_name_does_live(self) -> None:
        from rcp_ndcg.inference.adapters.base import check_engine_api

        with pytest.raises(ConfigError) as caught:
            check_engine_api("vllm_pooling", engine_role="reranker", where="serve.reranker")
        assert "multi_vector" in (caught.value.hint or "")


class TestANamedKeyStaysOnTheConfigsUrls:
    """An explicitly named ``api_key_env`` belongs to the config that names it: it travels to the config's own
    URLs (its replicas, or the profile's home when it names none) and never to an injected transport aimed
    elsewhere -- fail closed."""

    @staticmethod
    def _run(config: Any, transport_url: str, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
        import httpx

        from rcp_ndcg.inference.transport import Transport

        monkeypatch.setenv("RCP_NDCG_NAMED_KEY", "fake-secret-named")
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.url.path.endswith("/embed"):
                return httpx.Response(200, json={"embeddings": {"float": [[0.0, 0.0]]}})
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0, 0.0]}]})

        injected = config.model_copy(update={"base_url": transport_url, "api_key_env": None})
        client = EmbeddingClient(config, sender=Transport(injected, httpx_transport=httpx.MockTransport(handler)))
        client.encode([Content.from_text("x")], EncodeRole.DOCUMENT)
        client.close()
        return seen

    def test_a_named_key_never_reaches_an_injected_transport_s_other_url(
        self, tokenizer_json: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = EmbeddingEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=8192,
            api_key_env="RCP_NDCG_NAMED_KEY",
        )
        seen = self._run(config, "https://evil.example/v1", monkeypatch)
        assert seen and all("Authorization" not in request.headers for request in seen)

    def test_a_named_key_reaches_an_injected_transport_on_the_config_s_own_url(
        self, tokenizer_json: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = EmbeddingEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=8192,
            api_key_env="RCP_NDCG_NAMED_KEY",
        )
        seen = self._run(config, "http://127.0.0.1:9000/v1", monkeypatch)
        assert seen and all(request.headers["Authorization"] == "Bearer fake-secret-named" for request in seen)

    def test_a_hosted_config_s_named_key_goes_to_the_profile_s_home_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        config = EmbeddingEndpoint(api="cohere", model="m", max_tokens=1024, api_key_env="RCP_NDCG_NAMED_KEY")
        at_home = self._run(config, "https://api.cohere.com/v2", monkeypatch)
        assert at_home and at_home[0].headers["Authorization"] == "Bearer fake-secret-named"
        elsewhere = self._run(config, "https://gateway.example/v2", monkeypatch)
        assert elsewhere and "Authorization" not in elsewhere[0].headers


class TestInjectedTransportCredentials:
    """The config's named variable reaches an injected transport (the verifier's case C): the profile the
    client points the transport at carries the config's ``api_key_env`` first."""

    def test_a_config_named_variable_is_resolved_through_a_foreign_transport(
        self, tokenizer_json: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rcp_ndcg.inference.transport import Transport

        monkeypatch.delenv("RCP_NDCG_CLIENT_KEY", raising=False)
        config = EmbeddingEndpoint(
            api="openai_embeddings",
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=8192,
            api_key_env="RCP_NDCG_CLIENT_KEY",
        )
        client = EmbeddingClient(config, sender=Transport(config, httpx_transport=_null_transport()))
        assert client._sender._auth.variables == ("RCP_NDCG_CLIENT_KEY",)
        assert client._sender._auth.required is True

        with pytest.raises(CredentialsError, match="RCP_NDCG_CLIENT_KEY"):
            client.encode([Content.from_text("x")], EncodeRole.DOCUMENT)

    def test_an_injected_transport_without_the_variable_named_keeps_the_key_off_a_foreign_host(
        self, tokenizer_json: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cohere client over a transport built on a bare config, aimed at a gateway URL: the profile's
        variables stay home (they name the vendor's own host), so a key reaches this request only through
        the config's own ``api_key_env`` -- never by falling back to the vendor's variables."""
        from rcp_ndcg.inference.transport import Transport

        monkeypatch.setenv("CO_API_KEY", "co-secret-key")
        monkeypatch.setenv("COHERE_API_KEY", "cohere-secret-key")
        config = EmbeddingEndpoint(api="cohere", base_url="http://127.0.0.1:9000/v1", model="m", max_tokens=1024)
        foreign = EmbeddingEndpoint(
            base_url="http://127.0.0.1:9000/v1", model="m", tokenizer=tokenizer_json, max_tokens=8192
        )
        client = EmbeddingClient(config, sender=Transport(foreign, httpx_transport=_null_transport()))
        assert not client._sender._auth.applies_to("http://127.0.0.1:9000/v1")
        assert client._sender._base_headers("http://127.0.0.1:9000/v1") == {}


def _capturing_transport() -> tuple[Any, list[Any]]:
    """A mock transport that records every request it is handed (the wire's headers included)."""
    import httpx

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/embed"):  # the Cohere v2 shape; /embeddings takes the OpenAI one
            return httpx.Response(200, json={"embeddings": {"float": [[0.0, 0.0]]}})
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0, 0.0]}]})

    return httpx.MockTransport(handler), seen


@pytest.mark.parametrize("role", ["embed", "pool"])
def test_every_role_builds_its_adapter_with_the_role_config(
    role: str, tokenizer_json: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The constructor convention is uniform: a role client instantiates its adapter with the config it
    serves (the resolved endpoint), so an adapter whose requests depend on a config field can read it -- an
    adapter built bare would raise at construction, or silently never see the config."""
    from rcp_ndcg.inference.adapters.embeddings import OpenAIEmbeddings
    from rcp_ndcg.inference.adapters.pooling import VllmPooling

    adapter = OpenAIEmbeddings if role == "embed" else VllmPooling
    seen: list[Any] = []

    def needs_its_config(self: Any, config: Any) -> None:
        self.config = config
        seen.append(config)

    monkeypatch.setattr(adapter, "__init__", needs_its_config)
    fields: dict[str, Any] = {"base_url": "http://127.0.0.1:9000/v1", "model": "m", "tokenizer": tokenizer_json}
    if role == "embed":
        client: Any = EmbeddingClient(EmbeddingEndpoint(max_tokens=8192, **fields), sender=RecordingSender())
    else:
        client = PoolingClient(PoolingEndpoint(max_tokens=8192, dim=2, **fields), sender=RecordingSender())
    assert seen == [client.endpoint]


class TestKeysStayOnTheProfileHost:
    """A profile's default key variables travel only to the profile's own default host.

    ``OPENAI_API_KEY`` exists to pay OpenAI: a request to any other ``base_url`` (a self-hosted engine,
    a third party) carries a key only when the config names one with ``api_key_env``. A variable set for
    one vendor must never authenticate a request somewhere else.
    """

    def test_a_served_engines_profile_key_never_reaches_a_custom_base_url(
        self, tokenizer_json: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``openai_embeddings`` engine at a local URL is sent no OpenAI key, however the variable is set."""
        from rcp_ndcg.inference.transport import Transport

        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-openai-key")
        config = EmbeddingEndpoint(
            api="openai_embeddings",
            base_url="http://127.0.0.1:8000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=8192,
        )
        mock, seen = _capturing_transport()
        client = EmbeddingClient(config, sender=Transport(config, httpx_transport=mock))
        assert not client._sender._auth.applies_to("http://127.0.0.1:8000/v1")
        client.encode([Content.from_text("x")], EncodeRole.DOCUMENT)
        client.close()
        assert seen, "the request never went out"
        assert "Authorization" not in seen[0].headers

    def test_a_hosted_profile_behind_a_custom_base_url_sends_no_key_without_an_explicit_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cohere's variables behind a gateway URL: the gateway injects its own credential, so the vendor's
        key stays home; an explicitly named variable is the one exception."""
        from rcp_ndcg.inference.transport import Transport

        monkeypatch.setenv("CO_API_KEY", "co-secret-key")
        config = EmbeddingEndpoint(
            api="cohere", base_url="https://gateway.example.com/cohere", model="m", max_tokens=1024
        )
        mock, seen = _capturing_transport()
        client = EmbeddingClient(config, sender=Transport(config, httpx_transport=mock))
        assert not client._sender._auth.applies_to("https://gateway.example.com/cohere")
        client.encode([Content.from_text("x")], EncodeRole.DOCUMENT)
        client.close()
        assert seen and "Authorization" not in seen[0].headers

    def test_a_named_api_key_env_travels_to_any_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The one exception is explicit: ``api_key_env`` names the variable, wherever the request goes."""
        from rcp_ndcg.inference.transport import Transport

        monkeypatch.setenv("GATEWAY_COHERE_KEY", "gw-secret-key")
        config = EmbeddingEndpoint(
            api="cohere",
            base_url="https://gateway.example.com/cohere",
            model="m",
            max_tokens=1024,
            api_key_env="GATEWAY_COHERE_KEY",
        )
        mock, seen = _capturing_transport()
        client = EmbeddingClient(config, sender=Transport(config, httpx_transport=mock))
        assert client._sender._auth.variables == ("GATEWAY_COHERE_KEY",)
        client.encode([Content.from_text("x")], EncodeRole.DOCUMENT)
        client.close()
        assert seen and seen[0].headers["Authorization"] == "Bearer gw-secret-key"

    def test_the_profile_keys_apply_on_the_profile_s_own_default_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """At the vendor's own root the profile's variables resolve as before -- that is what they are for."""
        from rcp_ndcg.inference.transport import Transport

        monkeypatch.setenv("CO_API_KEY", "co-secret-key")
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        config = EmbeddingEndpoint(api="cohere", model="m", max_tokens=1024)
        resolved = config.model_copy(update={"base_url": "https://api.cohere.com/v2"})
        mock, seen = _capturing_transport()
        client = EmbeddingClient(config, sender=Transport(resolved, httpx_transport=mock))
        assert client._sender._auth.variables == ("CO_API_KEY", "COHERE_API_KEY")
        assert client._sender._auth.required is True
        client.encode([Content.from_text("x")], EncodeRole.DOCUMENT)
        client.close()
        assert seen and seen[0].headers["Authorization"] == "Bearer co-secret-key"


@pytest.mark.parametrize(
    ("api", "variable", "host"),
    [
        ("cohere", "CO_API_KEY", "https://api.cohere.com/v2"),
        ("voyage", "VOYAGE_API_KEY", "https://api.voyageai.com/v1"),
    ],
)
def test_a_hosted_rerank_profile_s_key_stays_on_its_own_host(
    api: str, variable: str, host: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The host rule is the client base's, so the rerank role keeps it too: the vendor's variable reaches the
    vendor's own root and nothing else -- behind a gateway URL the request goes out with no key."""
    import httpx

    from rcp_ndcg.inference.adapters.base import get_adapter
    from rcp_ndcg.inference.transport import Transport

    monkeypatch.setattr(get_adapter(api, role="rerank"), "PAUSE_S", 0.0)  # Voyage's rate-limit pause, not under test
    monkeypatch.setenv(variable, "vendor-secret-key")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})

    for base_url, header in ((host, "Bearer vendor-secret-key"), ("https://gateway.example.com/rerank", None)):
        config = RerankEndpoint(api=api, model="m", base_url=base_url)
        client = RerankClient(config, sender=Transport(config, httpx_transport=httpx.MockTransport(handler)))
        client.rerank("q", ["d"])
        client.close()
        assert seen[-1].url.host == httpx.URL(base_url).host
        assert seen[-1].headers.get("Authorization") == header, base_url


class TestKeysFollowTheReplicaNotTheConfig:
    """The key-host rule is decided where the request goes -- per replica, in the transport -- never from
    the client config's ``base_url``: a transport injected on another URL, a config swap, or a replica
    list mixing the vendor's root with a stranger never carries a profile's key away from its home."""

    VENDOR_KEYS: ClassVar[dict[str, str]] = {
        "OPENAI_API_KEY": "fake-secret-openai",
        "CO_API_KEY": "fake-secret-co",
        "COHERE_API_KEY": "fake-secret-cohere",
        "VOYAGE_API_KEY": "fake-secret-voyage",
        "GEMINI_API_KEY": "fake-secret-gemini",
        "GOOGLE_API_KEY": "fake-secret-google",
    }

    @staticmethod
    def _capture() -> tuple[Any, list[Any]]:
        import httpx

        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            path = request.url.path
            if path.endswith("/embed"):
                return httpx.Response(200, json={"embeddings": {"float": [[0.0, 0.0]]}})
            if "batchEmbedContents" in path:
                return httpx.Response(200, json={"embeddings": [{"values": [0.0, 0.0]}]})
            if path.endswith("/embeddings"):
                return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0, 0.0]}]})
            if path.endswith("/pooling"):
                return httpx.Response(200, json={"data": [{"index": 0, "data": [[0.0, 0.0]]}]})
            if path.endswith("/chat/completions"):
                return httpx.Response(
                    200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}}
                )
            return httpx.Response(
                200,
                json={
                    "results": [{"index": 0, "relevance_score": 0.5}],
                    "data": [{"index": 0, "relevance_score": 0.5}],
                },
            )

        return httpx.MockTransport(handler), seen

    @staticmethod
    def _secrets_in(requests: list[Any]) -> list[str]:
        return [
            f"{request.url.host}: {name}"
            for request in requests
            for name, value in request.headers.items()
            if "fake-secret" in value
        ]

    @pytest.mark.parametrize("api", ["openai_embeddings", "cohere", "voyage", "gemini"])
    def test_an_embed_transport_injected_on_another_url_gets_no_vendor_key(
        self, api: str, tokenizer_json: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rcp_ndcg.inference.transport import Transport

        for name, value in self.VENDOR_KEYS.items():
            monkeypatch.setenv(name, value)
        fields: dict[str, Any] = {"api": api, "model": "m", "max_tokens": 1024}
        if api == "openai_embeddings":
            fields["tokenizer"] = tokenizer_json
        config = EmbeddingEndpoint(**fields)  # the profile's own default host
        foreign = EmbeddingEndpoint(base_url="https://evil.example/v1", **fields)
        mock, seen = self._capture()
        client = EmbeddingClient(config, sender=Transport(foreign, httpx_transport=mock))
        client.encode([Content.from_text("x")], EncodeRole.DOCUMENT)
        client.close()
        assert seen and {request.url.host for request in seen} == {"evil.example"}
        assert self._secrets_in(seen) == []

    @pytest.mark.parametrize("api", ["cohere", "voyage"])
    def test_a_rerank_transport_injected_on_another_url_gets_no_vendor_key(
        self, api: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rcp_ndcg.inference.adapters.base import get_adapter
        from rcp_ndcg.inference.transport import Transport

        for name, value in self.VENDOR_KEYS.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setattr(get_adapter(api, role="rerank"), "PAUSE_S", 0.0)
        config = RerankEndpoint(api=api, model="m", max_tokens=1000)
        foreign = RerankEndpoint(api=api, base_url="https://evil.example/v1", model="m", max_tokens=1000)
        mock, seen = self._capture()
        client = RerankClient(config, sender=Transport(foreign, httpx_transport=mock))
        client.rerank("q", ["d"])
        client.close()
        assert seen and {request.url.host for request in seen} == {"evil.example"}
        assert self._secrets_in(seen) == []

    def test_a_pool_transport_injected_on_another_url_gets_no_vendor_key(
        self, tokenizer_json: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rcp_ndcg.inference.transport import Transport

        for name, value in self.VENDOR_KEYS.items():
            monkeypatch.setenv(name, value)
        fields: dict[str, Any] = {"model": "m", "dim": 2, "tokenizer": tokenizer_json, "max_tokens": 1024}
        config = PoolingEndpoint(base_url="http://127.0.0.1:9000/v1", **fields)
        foreign = PoolingEndpoint(base_url="https://evil.example/v1", **fields)
        mock, seen = self._capture()
        client = PoolingClient(config, sender=Transport(foreign, httpx_transport=mock))
        client.encode([Content.from_text("x")], EncodeRole.DOCUMENT)
        client.close()
        assert seen and {request.url.host for request in seen} == {"evil.example"}
        assert self._secrets_in(seen) == []

    def test_a_judge_config_swapped_to_another_url_gets_no_vendor_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio as _asyncio

        from rcp_ndcg.inference.types import CompletionInput
        from rcp_ndcg.judging import JudgeClient, JudgeConfig

        for name, value in self.VENDOR_KEYS.items():
            monkeypatch.setenv(name, value)
        mock, seen = self._capture()
        client = JudgeClient(
            JudgeConfig(base_url="http://judge.test/v1", model="m", max_retries=0), httpx_transport=mock
        )
        client.config = client.config.model_copy(update={"base_url": "https://evil.example/v1"})
        _asyncio.run(client.complete(CompletionInput(user_prompt="judge this")))
        client.close()
        assert seen and {request.url.host for request in seen} == {"evil.example"}
        assert self._secrets_in(seen) == []

    def test_a_replica_list_sends_the_key_to_the_home_replica_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One endpoint, two replicas -- the vendor's root and a stranger: the transport decides per replica."""
        import asyncio as _asyncio

        from rcp_ndcg.inference.transport import AuthProfile, Transport

        monkeypatch.setenv("CO_API_KEY", "fake-secret-co")
        mock, seen = self._capture()
        endpoint = EmbeddingEndpoint(
            api="cohere",
            model="m",
            max_tokens=1024,
            base_url=["https://api.cohere.com/v2", "https://evil.example/v2"],
            concurrency=1,
        )
        transport = Transport(
            endpoint,
            auth=AuthProfile(variables=("CO_API_KEY",), required=True, homes=("https://api.cohere.com/v2",)),
            httpx_transport=mock,
        )
        for _ in range(4):
            _asyncio.run(transport.send([Call("POST", "/embed", {})]))
        transport.close()
        hosts = {request.url.host: request.headers.get("Authorization") for request in seen}
        assert hosts == {"api.cohere.com": "Bearer fake-secret-co", "evil.example": None}

    HOME = "https://api.openai.com/v1"

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.openai.com/v1.evil.example",
            "https://api.openai.com/v1@evil.example",
            "https://api.openai.com/v1@evil",
            "https://api.openai.com.evil.example/v1",
            "https://api.openai.com/v1?x=1",
            "https://api.openai.com/v1#frag",
            "https://api.openai.com/v1/../v1",
            "https://u:p@api.openai.com/v1",
        ],
    )
    def test_a_url_that_only_starts_like_the_home_gets_no_key(self, url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Home is the exact URL (a trailing slash aside), never a prefix: a host or a path that merely begins
        with the vendor's root, a query, a fragment or userinfo on it, is somewhere else."""
        import asyncio as _asyncio

        from rcp_ndcg.inference.transport import AuthProfile, Transport

        monkeypatch.setenv("OPENAI_API_KEY", "fake-secret-openai")
        profile = AuthProfile(variables=("OPENAI_API_KEY",), required=True, homes=(self.HOME,))
        assert not profile.applies_to(url)
        mock, seen = self._capture()
        transport = Transport(Endpoint(base_url=url, model="m", max_retries=0), auth=profile, httpx_transport=mock)
        _asyncio.run(transport.send([Call("POST", "/embeddings", {})]))
        transport.close()
        assert seen and self._secrets_in(seen) == []

    @pytest.mark.parametrize("url", ["https://api.openai.com/v1", "https://api.openai.com/v1/"])
    def test_the_home_itself_gets_the_key(self, url: str) -> None:
        from rcp_ndcg.inference.transport import AuthProfile

        assert AuthProfile(variables=("OPENAI_API_KEY",), homes=(self.HOME,)).applies_to(url)

    def test_a_profile_without_a_home_resolves_no_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fail closed: profile variables with no declared home go nowhere (only an explicit, config-named
        variable travels to any host)."""
        import asyncio as _asyncio

        from rcp_ndcg.inference.transport import AuthProfile, Transport

        monkeypatch.setenv("CO_API_KEY", "fake-secret-co")
        mock, seen = self._capture()
        endpoint = EmbeddingEndpoint(api="cohere", model="m", max_tokens=1024, base_url="https://gw.example/v2")
        transport = Transport(endpoint, auth=AuthProfile(variables=("CO_API_KEY",)), httpx_transport=mock)
        _asyncio.run(transport.send([Call("POST", "/embed", {})]))
        transport.close()
        assert self._secrets_in(seen) == []


def _null_transport() -> Any:
    """A mock endpoint that answers nothing useful; credential errors fail before anything is queued."""
    import httpx

    return httpx.MockTransport(lambda request: httpx.Response(500, json={}))


class TestUseActivationOnTheAdapter:
    """F10 keyed on the resolved adapter's HOSTED flag (the operator's decision): a served third-party
    rerank wire also needs an explicit ``use_activation``; hosted profiles keep ``None``."""

    def test_a_served_third_party_rerank_adapter_refuses_an_unset_use_activation(self) -> None:
        from rcp_ndcg.inference import adapters as registry
        from rcp_ndcg.inference.adapters.base import AdapterRole, register_adapter
        from tests.conftest import SESSION_TOKENIZER

        class _ServedThirdParty:
            name = "served_third_party_rerank"
            role: ClassVar[AdapterRole] = "rerank"
            HOSTED = False
            DEFAULT_BASE_URL = None
            API_KEY_ENV = ()
            KEY_REQUIRED = False
            AUTH_HEADER = None

            def calls(self, request: Any, *, model: str) -> list[Call]:
                return []

            def interpret(self, request: Any, replies: list[Reply]) -> Any:
                return None

            def usage(self, reply: Reply) -> None:
                return None

        saved = dict(registry.base._BUILTINS)
        register_adapter(_ServedThirdParty)  # type: ignore[arg-type]
        try:
            with pytest.raises(ConfigError, match="use_activation"):
                RerankClient(
                    RerankEndpoint(
                        api="served_third_party_rerank",
                        base_url="http://h:8000/v1",
                        model="m",
                        tokenizer=str(SESSION_TOKENIZER),
                        max_tokens=8192,
                    ),
                    sender=RecordingSender(),
                )
        finally:
            registry.base._BUILTINS.clear()
            registry.base._BUILTINS.update(saved)

    def test_a_hosted_third_party_rerank_adapter_keeps_none(self) -> None:
        from rcp_ndcg.inference import adapters as registry
        from rcp_ndcg.inference.adapters.base import AdapterRole, register_adapter

        class _HostedThirdParty:
            name = "hosted_third_party_rerank"
            role: ClassVar[AdapterRole] = "rerank"
            HOSTED = True
            DEFAULT_BASE_URL = "https://vendor.example/v1"
            API_KEY_ENV = ("VENDOR_API_KEY",)
            KEY_REQUIRED = True
            AUTH_HEADER = None

            def __init__(self, config: Any) -> None:
                self.config = config

            def calls(self, request: Any, *, model: str) -> list[Call]:
                return []

            def interpret(self, request: Any, replies: list[Reply]) -> Any:
                return None

            def usage(self, reply: Reply) -> None:
                return None

        saved = dict(registry.base._BUILTINS)
        register_adapter(_HostedThirdParty)  # type: ignore[arg-type]
        try:
            client = RerankClient(RerankEndpoint(api="hosted_third_party_rerank", model="m"), sender=RecordingSender())
            client.close()
        finally:
            registry.base._BUILTINS.clear()
            registry.base._BUILTINS.update(saved)


class TestUsageAccounting:
    """Retrieval-role usage is forwarded and recorded (the judge's per-reply rule): each client folds its
    replies' token reports into the transport's usage, and its ``usage`` property reads the same
    accounting -- an embed/pool/rerank run records the tokens its replies reported, never zeros."""

    @staticmethod
    def _usage_mock(tokens: int) -> tuple[Any, list[Any]]:
        import httpx

        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.url.path.endswith("/rerank"):
                return httpx.Response(
                    200, json={"results": [{"index": 0, "relevance_score": 0.5}], "usage": {"prompt_tokens": tokens}}
                )
            if request.url.path.endswith("/pooling"):
                # a token_embed answer has one vector per prompt token: the report must match the frame
                return httpx.Response(
                    200, json={"data": [{"index": 0, "data": [[1.0, 1.0]]}], "usage": {"prompt_tokens": 1}}
                )
            return httpx.Response(
                200, json={"data": [{"index": 0, "embedding": [0.0, 0.0]}], "usage": {"prompt_tokens": tokens}}
            )

        return httpx.MockTransport(handler), seen

    def test_the_embed_client_folds_reply_tokens_into_the_transport(
        self, tokenizer_json: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rcp_ndcg.inference.transport import Transport

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = EmbeddingEndpoint(
            api="openai_embeddings",
            base_url="http://127.0.0.1:8000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=8192,
        )
        mock, _seen = TestUsageAccounting._usage_mock(7)
        client = EmbeddingClient(config, sender=Transport(config, httpx_transport=mock))
        client.encode([Content.from_text("x")], EncodeRole.DOCUMENT)
        client.close()

        assert client.usage.requests == 1
        assert client.usage.input_tokens == 7

    def test_the_rerank_client_folds_reply_tokens_into_the_transport(self, tokenizer_json: str) -> None:
        from rcp_ndcg.inference.transport import Transport

        config = RerankEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="m",
            tokenizer=tokenizer_json,
            max_tokens=8192,
            use_activation=False,
        )
        mock, _seen = TestUsageAccounting._usage_mock(11)
        client = RerankClient(config, sender=Transport(config, httpx_transport=mock))
        client.rerank("q", ["d"])
        client.close()

        assert client.usage.requests == 1
        assert client.usage.input_tokens == 11

    def test_a_paused_profile_s_replies_land_in_the_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Voyage's calls go one at a time with a pause between them (a separate send path): its replies'
        token reports are folded in too."""
        from rcp_ndcg.inference.adapters.rerank import VoyageRerankAdapter
        from rcp_ndcg.inference.transport import Transport

        monkeypatch.setenv("VOYAGE_API_KEY", "fake-secret-voyage")
        monkeypatch.setattr(VoyageRerankAdapter, "PAUSE_S", 0.001)  # > 0: the paused path, kept short
        config = RerankEndpoint(api="voyage", model="m")
        resolved = config.model_copy(update={"base_url": "https://api.voyageai.com/v1"})
        mock, _seen = TestUsageAccounting._usage_mock(11)
        client = RerankClient(config, sender=Transport(resolved, httpx_transport=mock))
        client.rerank("q", ["d"])
        client.close()

        assert client.usage.input_tokens == 11

    def test_the_pool_client_folds_reply_tokens_into_the_transport(self, tokenizer_json: str) -> None:
        from rcp_ndcg.inference.transport import Transport

        config = PoolingEndpoint(
            base_url="http://127.0.0.1:9000/v1",
            model="colqwen",
            dim=2,
            tokenizer=tokenizer_json,
            max_tokens=8192,
        )
        mock, _seen = TestUsageAccounting._usage_mock(13)
        client = PoolingClient(config, sender=Transport(config, httpx_transport=mock))
        asyncio.run(client.aencode([Content.from_text("x")], EncodeRole.DOCUMENT))
        client.close()

        assert client.usage.requests == 1
        assert client.usage.input_tokens == 1  # the /pooling cross-check pins the report to the frame

    def test_the_embed_client_probes_the_transport(self, tokenizer_json: str) -> None:
        """The embed role's startup probe is the transport's replica probe (the role sends no media probe
        request, so no engine media check runs)."""
        client = EmbeddingClient(
            EmbeddingEndpoint(api="openai_embeddings", model="m", tokenizer=tokenizer_json, max_tokens=8192),
            sender=RecordingSender(),
        )
        assert asyncio.run(client.probe()) == []
