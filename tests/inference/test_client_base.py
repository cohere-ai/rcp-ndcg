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
        from rcp_ndcg_core._records import RankingExample

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
        assert client._sender._auth.variables == ()
        assert client._sender._auth.required is False


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
        assert client._sender._auth.variables == ()
        assert client._sender._auth.required is False
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
        assert client._sender._auth.variables == ()
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
