"""The inference layer's frozen surface: the wire types, the role configs, the adapter seam, the transport.

Every type here is a contract the later lanes build against, so each one gets its construction, its frozenness
and its refusal of a wrong value.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, ClassVar

import numpy as np
import pytest
from pydantic import ValidationError
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.inference import (
    ADAPTER_ENTRY_POINTS,
    Adapter,
    AdapterRole,
    Call,
    Completion,
    CompletionInput,
    Embeddings,
    EmbedRequest,
    EncodeRole,
    EngineInfo,
    PoolingEndpoint,
    PoolRequest,
    Reply,
    RerankEndpoint,
    RerankRequest,
    RerankResult,
    TokenCount,
    Transport,
    Usage,
    get_adapter,
    known_adapters,
    l2_normalize,
    register_adapter,
)
from rcp_ndcg.inference import adapters as _adapters
from rcp_ndcg.inference.config import EmbeddingEndpoint
from rcp_ndcg.inference.endpoint import Endpoint
from rcp_ndcg.llm import JudgeConfig
from rcp_ndcg.support.identity import FieldRole, check_declarations, identity_payload
from rcp_ndcg.support.serve import (
    ENGINES_ENV,
    EngineURLs,
    Phase,
    ServeByRole,
    parse_engines_env,
    plan_phases,
)

# ---------------------------------------------------------------------------------------------------------------
# The wire types (rfc-0001 section 4.2)
# ---------------------------------------------------------------------------------------------------------------


class TestCallAndReply:
    def test_a_call_carries_its_method_path_and_optional_body_and_headers(self) -> None:
        call = Call("POST", "/chat/completions", {"model": "m"}, {"X-Test": "1"})
        assert call.method == "POST" and call.path == "/chat/completions"
        assert call.json == {"model": "m"} and call.headers == {"X-Test": "1"}
        assert Call("GET", "/models").json is None
        assert Call("GET", "/models").headers == {}

    def test_a_call_refuses_a_method_that_is_not_get_or_post(self) -> None:
        with pytest.raises(ValueError, match="GET.*POST"):
            Call("PUT", "/chat/completions")  # type: ignore[arg-type]

    def test_calls_and_replies_are_frozen(self) -> None:
        call = Call("GET", "/models")
        reply = Reply(200, {"data": []}, {"server": "x"})
        with pytest.raises(AttributeError):
            call.path = "/other"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            reply.status = 500  # type: ignore[misc]

    def test_a_reply_holds_its_status_body_and_headers(self) -> None:
        reply = Reply(200, {"data": []}, {"server": "vllm"})
        assert reply.status == 200 and reply.body == {"data": []} and reply.headers == {"server": "vllm"}


class TestUsage:
    def test_usage_adds_element_wise(self) -> None:
        total = Usage(calls=1, failed_calls=0, input_tokens=10, output_tokens=2) + Usage(
            calls=2, failed_calls=1, input_tokens=5, output_tokens=3
        )
        assert total == Usage(calls=3, failed_calls=1, input_tokens=15, output_tokens=5)

    def test_usage_refuses_another_type(self) -> None:
        with pytest.raises(TypeError):
            Usage() + 1  # type: ignore[operator]

    def test_usage_is_frozen(self) -> None:
        with pytest.raises(AttributeError):
            Usage().calls = 3  # type: ignore[misc]


class TestTokenCount:
    def test_a_token_count_holds_what_the_api_reported(self) -> None:
        assert TokenCount(3, 7) == TokenCount(input_tokens=3, output_tokens=7)
        assert TokenCount().input_tokens is None and TokenCount().output_tokens is None


class TestEngineInfo:
    def test_an_engine_info_records_the_replica(self) -> None:
        info = EngineInfo(url="http://a:8000/v1", model="m", max_model_len=8192)
        assert info.url == "http://a:8000/v1" and info.max_model_len == 8192 and info.error is None
        with pytest.raises(ValidationError):
            EngineInfo(url="http://a:8000/v1", surprise=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------------------------------------------
# The role request and result types
# ---------------------------------------------------------------------------------------------------------------


class TestCompletionTypes:
    def test_a_completion_input_knows_whether_it_carries_media(self) -> None:
        text = CompletionInput(user_prompt="hi")
        assert text.has_media is False
        media = CompletionInput(user_prompt="", user_content=Content.from_image("file:///a.png"))
        assert media.has_media is True

    def test_a_completion_carries_its_tokens(self) -> None:
        done = Completion(response="yes", reasoning="why", input_tokens=11, output_tokens=2)
        assert done.finish_reason == "stop" and done.input_tokens == 11 and done.output_tokens == 2

    def test_a_completion_input_is_frozen(self) -> None:
        with pytest.raises(ValidationError):
            CompletionInput(user_prompt="hi").user_prompt = "no"  # type: ignore[misc]


class TestEmbedTypes:
    def test_an_embed_request_holds_its_side_and_optional_cut(self) -> None:
        request = EmbedRequest((Content.from_text("a"), Content.from_text("b")), EncodeRole.DOCUMENT)
        assert request.role is EncodeRole.DOCUMENT and request.dimensions is None
        cut = EmbedRequest((Content.from_text("a"),), EncodeRole.QUERY, dimensions=256)
        assert cut.dimensions == 256

    def test_a_pool_request_is_its_own_type(self) -> None:
        request = PoolRequest((Content.from_text("a"),), EncodeRole.QUERY)
        assert request.contents[0].text == "a" and request.role is EncodeRole.QUERY

    def test_embeddings_are_frozen_and_validate_their_layout(self) -> None:
        vectors = Embeddings.single(np.eye(2, dtype=np.float32))
        assert vectors.num_items == 2 and vectors.dim == 2 and not vectors.is_multi_vector
        ragged = Embeddings.ragged([np.ones((2, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32)])
        assert ragged.is_multi_vector and ragged.num_items == 2
        with pytest.raises(ValueError, match="2-D"):
            Embeddings(vectors=np.zeros(3, dtype=np.float32))

    def test_l2_normalize_scales_rows_to_unit_norm(self) -> None:
        normalized = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
        assert np.allclose(normalized[0], [0.6, 0.8]) and np.allclose(normalized[1], [0.0, 0.0])


class TestRerankTypes:
    def test_a_rerank_request_holds_query_documents_and_optional_instruction(self) -> None:
        request = RerankRequest(Content.from_text("q"), (Content.from_text("a"), Content.from_text("b")))
        assert request.instruction is None and len(request.documents) == 2

    def test_a_rerank_result_is_aligned_to_its_documents(self) -> None:
        request = RerankRequest(Content.from_text("q"), (Content.from_text("a"), Content.from_text("b")))
        result = RerankResult.aligned(request, [0.9, 0.1])
        assert result.scores == (0.9, 0.1)

    def test_a_rerank_result_refuses_a_mismatched_score_count(self) -> None:
        request = RerankRequest(Content.from_text("q"), (Content.from_text("a"), Content.from_text("b")))
        with pytest.raises(ValueError, match="3 score\\(s\\).*2 document"):
            RerankResult.aligned(request, [0.9, 0.1, 0.5])


# ---------------------------------------------------------------------------------------------------------------
# The endpoint and the role configs
# ---------------------------------------------------------------------------------------------------------------


class TestEndpoint:
    def test_an_endpoint_keys_on_its_model_checkpoint_and_adapter(self) -> None:
        one = Endpoint(api="openai_chat", base_url="http://a:8000/v1/", model="m", revision="abc", concurrency=4)
        other = Endpoint(api="openai_chat", base_url="http://b:9000/v1", model="m", revision="abc", timeout_s=30)
        assert one.base_url == "http://a:8000/v1"
        assert identity_payload(one) == identity_payload(other)
        assert identity_payload(one) == {"api": "openai_chat", "model": "m", "revision": "abc"}

    def test_headers_env_values_must_be_environment_variable_names(self) -> None:
        ok = Endpoint(model="m", headers_env={"X-Gateway-Key": "GATEWAY_KEY"})
        assert ok.headers_env == {"X-Gateway-Key": "GATEWAY_KEY"}
        with pytest.raises(ValidationError, match="not an environment variable name"):
            Endpoint(model="m", headers_env={"X-Gateway-Key": "not a name"})
        with pytest.raises(ValidationError, match="not an environment variable name"):
            Endpoint(model="m", headers_env={"X-Gateway-Key": "9LEADING_DIGIT"})

    def test_wait_on_outage_s_moved_up_from_the_judge(self) -> None:
        assert Endpoint(model="m").wait_on_outage_s is None
        assert Endpoint(model="m", wait_on_outage_s=900).wait_on_outage_s == 900
        with pytest.raises(ValidationError):
            Endpoint(model="m", wait_on_outage_s=-1)
        # The judge keeps working through inheritance, and the new fields never reach its identity payload:
        # api is unset (None, omitted), headers_env and wait_on_outage_s are runtime.
        judge = JudgeConfig(base_url="http://a:8000/v1", model="m", wait_on_outage_s=900)
        assert judge.wait_on_outage_s == 900
        bare = identity_payload(JudgeConfig(base_url="http://a:8000/v1", model="m"))
        assert "api" not in bare and "headers_env" not in bare and "wait_on_outage_s" not in bare
        assert identity_payload(judge) == bare

    def test_an_endpoint_is_frozen_and_refuses_unknown_fields(self) -> None:
        endpoint = Endpoint(model="m")
        with pytest.raises(ValidationError):
            endpoint.model = "other"  # type: ignore[misc]
        with pytest.raises(ValidationError):
            Endpoint(model="m", surprise=1)  # type: ignore[call-arg]

    def test_api_is_undeclared_nothing_and_defaults_to_none(self) -> None:
        assert Endpoint(model="m").api is None


class TestRoleConfigs:
    @pytest.mark.parametrize(
        "config_cls",
        [EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint],
        ids=lambda cls: cls.__name__,
    )
    def test_every_field_of_every_role_config_is_classified(self, config_cls: type[Endpoint]) -> None:
        check_declarations(config_cls)

    @pytest.mark.parametrize(
        ("config_cls", "api"),
        [(EmbeddingEndpoint, "openai_embeddings"), (PoolingEndpoint, "vllm_pooling"), (RerankEndpoint, "rerank")],
    )
    def test_each_role_defaults_its_wire_adapter(self, config_cls: type[Endpoint], api: str) -> None:
        assert config_cls(model="m").api == api

    def test_an_embedding_config_declares_its_content_and_runtime_knobs(self) -> None:
        config = EmbeddingEndpoint(
            base_url="http://a:8000/v1",
            model="octen-embedding-8b",
            recipe="last-token-l2",
            tokenizer="Octen/Octen-Embedding-8B@abc",
            max_tokens=8192,
            doc_prompt="- ",
            normalize=True,
            dimensions=1024,
            batch_size=96,
        )
        assert config.recipe == "last-token-l2" and config.query_prompt == ""
        # What changes the vectors is content; how fast it is asked is runtime.
        assert identity_payload(config)["recipe"] == "last-token-l2"
        assert "batch_size" not in identity_payload(config)
        assert "tokenizer" not in identity_payload(config)  # the name is runtime; its SHA-256 is lane L3a's

    def test_a_pooling_config_defaults_to_float16(self) -> None:
        config = PoolingEndpoint(base_url="http://a:8000/v1", model="colpali")
        assert config.embed_dtype == "float16"
        assert identity_payload(config)["embed_dtype"] == "float16"
        opted = PoolingEndpoint(base_url="http://a:8000/v1", model="colpali", embed_dtype="float32")
        assert opted.embed_dtype == "float32"

    def test_a_rerank_config_folds_its_instruction_by_default(self) -> None:
        config = RerankEndpoint(base_url="http://a:8000/v1", model="qwen3-reranker-8b")
        assert config.instruction == "fold" and config.use_activation is None and config.listwise is False
        assert identity_payload(config)["instruction"] == "fold"

    def test_a_listwise_reranker_refuses_a_batch_size(self) -> None:
        with pytest.raises(ValidationError, match="listwise"):
            RerankEndpoint(base_url="http://a:8000/v1", model="jina-reranker-v3", listwise=True, batch_size=8)
        assert RerankEndpoint(base_url="http://a:8000/v1", model="qwen3-reranker-8b", batch_size=8).batch_size == 8

    def test_a_rerank_config_can_split_the_pair_budget(self) -> None:
        config = RerankEndpoint(
            base_url="http://a:8000/v1", model="qwen3-reranker-8b", max_tokens=8192, query_max_tokens=256
        )
        assert config.query_max_tokens == 256
        # Content: the split changes what the model reads, so it keys; unset, it is the absence of a split.
        assert identity_payload(config)["query_max_tokens"] == 256
        assert "query_max_tokens" not in identity_payload(
            RerankEndpoint(base_url="http://a:8000/v1", model="qwen3-reranker-8b")
        )

    def test_role_configs_are_frozen_and_refuse_unknown_fields(self) -> None:
        config = RerankEndpoint(base_url="http://a:8000/v1", model="qwen3-reranker-8b")
        with pytest.raises(ValidationError):
            config.listwise = True  # type: ignore[misc]
        with pytest.raises(ValidationError):
            RerankEndpoint(base_url="http://a:8000/v1", model="m", surprise=1)  # type: ignore[call-arg]


class TestIdentityExtra:
    """One tokenizer-identity rule for every role config (RFC-0001 section 7.4).

    ``identity_extra()`` carries the SHA-256 of the config's ``tokenizer.json`` under the one key
    ``tokenizer_sha256`` -- never the tokenizer's name (RUNTIME) -- and the judge's own identity payload
    never moves: its family carries the digest under its own keys, which stay as they are.
    """

    @pytest.mark.parametrize(
        "config_cls",
        [EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint, JudgeConfig],
        ids=lambda cls: cls.__name__,
    )
    def test_identity_extra_is_the_tokenizers_sha256_under_one_key(
        self, config_cls: type[Endpoint], tmp_path: Any
    ) -> None:
        import hashlib

        from tests._tokenizers import save, word_tokenizer

        directory = tmp_path / "tok"
        directory.mkdir()
        file = save(word_tokenizer(), directory)
        named = config_cls(base_url="http://a:8000/v1", model="m", tokenizer=str(file))
        assert named.identity_extra() == {"tokenizer_sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
        assert set(named.identity_extra()) == {"tokenizer_sha256"}
        assert config_cls(base_url="http://a:8000/v1", model="m").identity_extra() == {}

    def test_the_judge_identity_payload_does_not_move(self, tmp_path: Any) -> None:
        """The judgement family's keys must not move: ``identity_extra()`` is not part of ``identity()``."""
        from tests._tokenizers import save, word_tokenizer

        directory = tmp_path / "tok"
        directory.mkdir()
        file = save(word_tokenizer(), directory)
        config = JudgeConfig(base_url="http://a:8000/v1", model="m", tokenizer=str(file))
        assert config.identity_extra() != {}
        assert "tokenizer_sha256" not in config.identity()
        assert "tokenizer" not in identity_payload(config)


# ---------------------------------------------------------------------------------------------------------------
# The adapter seam (C2)
# ---------------------------------------------------------------------------------------------------------------


class _ProbeAdapter:
    """A minimal adapter, registered per test."""

    name = "probe_adapter"
    role: ClassVar[AdapterRole] = "judge"

    def calls(self, request: Any, *, model: str) -> list[Call]:
        return [Call("POST", "/chat/completions", {"model": model})]

    def interpret(self, request: Any, replies: list[Reply]) -> str:
        return "ok"

    def usage(self, reply: Reply) -> TokenCount | None:
        return None


class _EmbedProbe(_ProbeAdapter):
    """The same adapter name in the embed role: the registry keys on ``(role, name)``."""

    role: ClassVar[AdapterRole] = "embed"


class _EmbedProbeAlias(_EmbedProbe):
    """A second class under the same registered name, for the duplicate-plugin refusal test."""


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Run each registry test against an empty registry, restoring whatever was there."""
    saved_builtins, saved_plugins = dict(_adapters.base._BUILTINS), _adapters.base._PLUGINS
    _adapters.base._BUILTINS.clear()
    yield
    _adapters.base._BUILTINS.clear()
    _adapters.base._BUILTINS.update(saved_builtins)
    _adapters.base._PLUGINS = saved_plugins


class TestAdapterRegistry:
    def test_an_adapter_is_registered_under_its_name_and_returned_by_it(self) -> None:
        register_adapter(_ProbeAdapter)
        assert get_adapter("probe_adapter", role="judge") is _ProbeAdapter
        assert "probe_adapter" in known_adapters("judge")

    def test_the_registry_is_scoped_by_role(self) -> None:
        """One name in two roles resolves per role: ``api: cohere`` names a different adapter for embed and
        for rerank, and every lookup and hint stays inside its role's namespace."""
        register_adapter(_ProbeAdapter)
        register_adapter(_EmbedProbe)
        assert get_adapter("probe_adapter", role="judge") is _ProbeAdapter
        assert get_adapter("probe_adapter", role="embed") is _EmbedProbe
        assert known_adapters("judge") == ("probe_adapter",)
        assert known_adapters("embed") == ("probe_adapter",)

    def test_known_adapters_without_a_role_lists_every_name_once(self) -> None:
        register_adapter(_ProbeAdapter)
        register_adapter(_EmbedProbe)
        assert known_adapters() == ("probe_adapter",)

    def test_register_adapter_returns_its_class_so_it_composes(self) -> None:
        assert register_adapter(_ProbeAdapter) is _ProbeAdapter

    def test_a_duplicate_name_in_one_role_is_refused(self) -> None:
        register_adapter(_ProbeAdapter)
        with pytest.raises(ConfigError, match="already registered"):
            register_adapter(_ProbeAdapter)

    def test_the_same_name_in_two_roles_is_no_duplicate(self) -> None:
        register_adapter(_ProbeAdapter)
        register_adapter(_EmbedProbe)  # must not raise: the roles' namespaces are separate

    def test_an_adapter_without_a_name_or_a_role_is_refused(self) -> None:
        class _Nameless(_ProbeAdapter):
            name = ""

        with pytest.raises(ConfigError, match="name"):
            register_adapter(_Nameless)

        class _Roleless(_ProbeAdapter):
            role = "embezzle"  # type: ignore[assignment]

        with pytest.raises(ConfigError, match="role"):
            register_adapter(_Roleless)

    def test_a_wrong_role_lookup_fails_with_that_role_s_names(self) -> None:
        """A name registered in another role is still unknown here, and the hint names where it lives."""
        register_adapter(_ProbeAdapter)
        register_adapter(_EmbedProbe)
        with pytest.raises(ConfigError) as caught:
            get_adapter("probe_adapter", role="rerank")
        assert caught.value.details["known"] == []
        assert "rerank" in str(caught.value)
        assert "judge" in (caught.value.hint or "") and "embed" in (caught.value.hint or "")

    def test_an_unknown_adapter_names_the_known_ones_of_its_role(self) -> None:
        register_adapter(_ProbeAdapter)
        with pytest.raises(ConfigError) as caught:
            get_adapter("nope", role="judge")
        assert caught.value.details["known"] == ["probe_adapter"]
        assert "probe_adapter" in (caught.value.hint or "")

    def test_the_role_is_a_required_keyword(self) -> None:
        register_adapter(_ProbeAdapter)
        with pytest.raises(TypeError):
            get_adapter("probe_adapter")  # type: ignore[call-arg]

    def test_an_unknown_role_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="role"):
            get_adapter("probe_adapter", role="embezzle")  # type: ignore[arg-type]

    def test_known_adapters_validates_the_role(self) -> None:
        """A typo'd role is refused, never silently reported as an empty namespace."""
        register_adapter(_ProbeAdapter)
        with pytest.raises(ConfigError, match="role"):
            known_adapters("embezzle")  # type: ignore[arg-type]

    def test_the_entry_point_group_name_is_the_charter_s(self) -> None:
        assert ADAPTER_ENTRY_POINTS == "rcp_ndcg.adapters"

    def test_the_protocol_is_satisfied_by_an_ordinary_class(self) -> None:
        register_adapter(_ProbeAdapter)
        adapter = get_adapter("probe_adapter", role="judge")()
        assert isinstance(adapter, Adapter)


class TestAdapterEntryPoints:
    """The ``rcp_ndcg.adapters`` group names its entries ``<role>.<name>``; a disagreeing prefix is refused."""

    @staticmethod
    def _entry(name: str, value: str) -> Any:
        from importlib.metadata import EntryPoint

        return EntryPoint(name=name, value=value, group=ADAPTER_ENTRY_POINTS)

    def _install(self, monkeypatch: pytest.MonkeyPatch, *entries: Any) -> None:
        monkeypatch.setattr(_adapters.base, "entry_points", lambda *, group: list(entries))
        monkeypatch.setattr(_adapters.base, "_PLUGINS", None)

    def test_an_entry_point_registers_under_its_class_role_and_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install(monkeypatch, self._entry("embed.probe_adapter", "tests.inference.test_types:_EmbedProbe"))
        assert get_adapter("probe_adapter", role="embed") is _EmbedProbe
        assert known_adapters("embed") == ("probe_adapter",)

    def test_an_entry_point_whose_role_disagrees_with_its_prefix_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install(monkeypatch, self._entry("judge.probe_adapter", "tests.inference.test_types:_EmbedProbe"))
        with pytest.raises(ConfigError, match="judge.*embed"):
            get_adapter("probe_adapter", role="embed")

    def test_an_entry_point_without_a_role_prefix_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install(monkeypatch, self._entry("probe_adapter", "tests.inference.test_types:_EmbedProbe"))
        with pytest.raises(ConfigError, match="<role>.<name>"):
            known_adapters()

    def test_an_entry_point_shadowing_a_builtin_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A plugin that would take a shipped adapter's (role, name) is an error, never a silent skip: the
        built-in would otherwise win the lookup and the plugin would never run."""
        register_adapter(_EmbedProbe)
        self._install(monkeypatch, self._entry("embed.probe_adapter", "tests.inference.test_types:_EmbedProbe"))
        with pytest.raises(ConfigError, match="already registered"):
            known_adapters()

    def test_an_entry_point_whose_name_suffix_disagrees_with_the_class_name_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The entry names its role *and* the adapter's registered name: a class registered under a name its
        entry point does not spell is a typo, not a convention."""
        self._install(monkeypatch, self._entry("embed.not_the_name", "tests.inference.test_types:_EmbedProbe"))
        with pytest.raises(ConfigError, match="not_the_name"):
            known_adapters()

    def test_two_entry_points_registering_one_name_are_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install(
            monkeypatch,
            self._entry("embed.probe_adapter", "tests.inference.test_types:_EmbedProbe"),
            self._entry("embed.probe_adapter", "tests.inference.test_types:_EmbedProbeAlias"),
        )
        with pytest.raises(ConfigError, match="a second time"):
            known_adapters()

    def test_a_broken_entry_point_import_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install(monkeypatch, self._entry("embed.probe_adapter", "tests.inference.test_types:_Missing"))
        with pytest.raises(ConfigError, match="failed to import"):
            known_adapters()


# ---------------------------------------------------------------------------------------------------------------
# The transport's interface (behaviour is lane L1's)
# ---------------------------------------------------------------------------------------------------------------


class TestTransport:
    def test_the_transport_is_not_built_yet(self) -> None:
        with pytest.raises(NotImplementedError, match="lane L1"):
            Transport(Endpoint(base_url="http://a:8000/v1", model="m"))


# ---------------------------------------------------------------------------------------------------------------
# The serve-by-role types (rfc-0001 section 5)
# ---------------------------------------------------------------------------------------------------------------


class TestServeByRole:
    def test_roles_are_optional_and_default_to_none(self) -> None:
        serve = ServeByRole()
        assert serve.judge is None and serve.encoder is None and serve.reranker is None

    def test_a_role_holds_todays_engine_config(self) -> None:
        from rcp_ndcg.support.serve import ServeConfig

        serve = ServeByRole(judge=ServeConfig(command=["vllm", "serve", "m"]))
        assert serve.judge is not None and serve.judge.port == 8000
        with pytest.raises(ValidationError):
            ServeByRole(surprise=1)  # type: ignore[call-arg]


class TestPhases:
    def test_a_phase_holds_its_engines_and_steps(self) -> None:
        phase = Phase(frozenset({"encoder"}), ("retrieve",))
        assert phase.engines == frozenset({"encoder"}) and phase.steps == ("retrieve",)
        with pytest.raises(AttributeError):
            phase.steps = ()  # type: ignore[misc]


class TestEnginesEnv:
    def test_a_phase_variable_parses_into_its_roles(self) -> None:
        parsed = parse_engines_env('{"encoder": {"urls": ["http://a:8000/v1"], "wait_on_outage_s": 900}}')
        assert set(parsed) == {"encoder"}
        assert parsed["encoder"].urls == ("http://a:8000/v1",) and parsed["encoder"].wait_on_outage_s == 900
        assert parse_engines_env("{}") == {}

    def test_bad_json_is_a_config_error(self) -> None:
        with pytest.raises(ConfigError, match="not valid JSON"):
            parse_engines_env("{oops")

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="JSON object"):
            parse_engines_env('["encoder"]')

    def test_an_unknown_role_names_the_known_ones(self) -> None:
        with pytest.raises(ConfigError) as caught:
            parse_engines_env('{"db": {"urls": ["http://a:8000/v1"]}}')
        assert caught.value.details["known"] == ["judge", "encoder", "reranker"]

    def test_urls_must_be_a_non_empty_tuple_of_non_empty_strings(self) -> None:
        with pytest.raises(ValidationError):
            EngineURLs(urls=())
        with pytest.raises(ValidationError):
            EngineURLs(urls=["http://a:8000/v1", ""])

    def test_the_variable_name_is_the_charter_s(self) -> None:
        assert ENGINES_ENV == "RCP_NDCG_ENGINES"

    def test_plan_phases_is_lane_l4a_s(self) -> None:
        with pytest.raises(NotImplementedError, match="lane L4a"):
            plan_phases(["retrieve"], ServeByRole(), {"retrieve": frozenset({"encoder"})})


# ---------------------------------------------------------------------------------------------------------------
# Identity declarations of the moved types
# ---------------------------------------------------------------------------------------------------------------


class TestDeclarations:
    def test_the_endpoint_declares_every_field(self) -> None:
        check_declarations(Endpoint)
        assert set(Endpoint.IDENTITY_ROLES) == set(Endpoint.model_fields)

    def test_the_new_fields_have_their_charter_roles(self) -> None:
        assert Endpoint.IDENTITY_ROLES["api"] is FieldRole.CONTENT
        assert Endpoint.IDENTITY_ROLES["headers_env"] is FieldRole.RUNTIME
        assert Endpoint.IDENTITY_ROLES["wait_on_outage_s"] is FieldRole.RUNTIME
