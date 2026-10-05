"""The embedding role client: prompts per side, batching, concurrency and reassembly, normalisation, and what it
refuses.

Offline: the client is exercised against a fake :class:`~rcp_ndcg.inference.transport.Sender`
(:class:`tests.inference._embed.FakeSender`) that records every call and replies with the profiles' shapes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np
import pytest
from rcp_ndcg_core.content import Content, ImagePart, MediaRef

from rcp_ndcg.errors import CapabilityError, ConfigError, CredentialsError, RequestRejectedError
from rcp_ndcg.inference import EmbeddingClient, EncodeRole
from rcp_ndcg.inference.types import Call, Embeddings, Reply
from tests._tokenizers import byte_bpe_tokenizer, save, word_tokenizer
from tests.inference._embed import FakeSender, openai_data, vendor_payload

#: Every profile under its registered api name.
APIS = ("openai_embeddings", "cohere", "voyage", "gemini")

KEY_ENV_NAMES = (
    "OPENAI_API_KEY",
    "CO_API_KEY",
    "COHERE_API_KEY",
    "VOYAGE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


@pytest.fixture(autouse=True)
def _vendor_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key for every profile's default variables (tests that need one absent delete it)."""
    for name in KEY_ENV_NAMES:
        monkeypatch.setenv(name, f"test-{name.lower()}")


def input_texts(api: str, call: Call) -> list[str]:
    """The texts one call carries, per profile."""
    body = call.json
    if api in ("openai_embeddings", "voyage"):
        return list(body["input"])
    if api == "cohere":
        return list(body["texts"])
    return [entry["content"]["parts"][0]["text"] for entry in body["requests"]]


def handler(api: str, values: dict[str, float], *, shuffle: bool = False) -> Callable[[Call], Reply]:
    """A handler that answers each text with the vector ``[value, 1.0]`` it maps to.

    With ``shuffle``, the OpenAI-shaped replies name their entries' ``index`` fields in reverse order, so a
    reassembly that trusts reply order produces a wrong matrix.
    """

    def one(call: Call) -> Reply:
        batch = input_texts(api, call)
        vectors = [[values.get(text, 1.0), 1.0] for text in batch]
        if shuffle and api in ("openai_embeddings", "voyage"):
            body = openai_data(vectors, indices=list(reversed(range(len(vectors)))))
        else:
            body = vendor_payload(api, vectors)
        return Reply(200, body, {})

    return one


def endpoint(api: str = "openai_embeddings", **overrides: Any) -> Any:
    """An embedding endpoint for tests: model ``m``, any profile, any overrides."""
    from tests.inference._embed import endpoint as make

    return make(api, **overrides)


def texts(*items: str) -> list[Content]:
    """Plain-text contents."""
    return [Content.from_text(item) for item in items]


class TestProfiles:
    @pytest.mark.parametrize("api", APIS)
    def test_every_profile_round_trips_l2_normalised(self, api: str) -> None:
        sender = FakeSender(handler(api, {"a document about foxes": 3.0}))
        client = EmbeddingClient(endpoint(api), sender=sender)

        vectors = client.encode(texts("a document about foxes"), EncodeRole.DOCUMENT)

        assert vectors.as_matrix().shape == (1, 2)
        expected = np.asarray([3.0, 1.0]) / np.linalg.norm([3.0, 1.0])
        assert vectors.as_matrix()[0].tolist() == pytest.approx(expected.tolist())
        assert vectors.vectors.dtype == np.float32

    @pytest.mark.parametrize("api", ("openai_embeddings", "voyage"))
    def test_a_shuffled_reply_is_reassembled_by_index(self, api: str) -> None:
        """The mutation probe: drop the adapters' reorder by ``index`` and this goes red."""
        sender = FakeSender(handler(api, {"one": 1.0, "two": 2.0}, shuffle=True))

        vectors = EmbeddingClient(endpoint(api, batch_size=2, normalize=False), sender=sender).encode(
            texts("one", "two"), EncodeRole.DOCUMENT
        )

        assert vectors.as_matrix()[:, 0].tolist() == [1.0, 2.0]


class TestContentDecisions:
    def test_prompts_are_applied_per_side(self) -> None:
        config = endpoint(query_prompt="Q: ", doc_prompt="D: ")
        sender = FakeSender(handler("openai_embeddings", {}))
        client = EmbeddingClient(config, sender=sender)

        client.encode(texts("what is nDCG"), EncodeRole.QUERY)
        client.encode(texts("a passage"), EncodeRole.DOCUMENT)

        assert input_texts("openai_embeddings", sender.calls[0]) == ["Q: what is nDCG"]
        assert input_texts("openai_embeddings", sender.calls[1]) == ["D: a passage"]

    def test_without_prompts_the_text_is_sent_as_given(self) -> None:
        sender = FakeSender(handler("openai_embeddings", {}))
        client = EmbeddingClient(endpoint(), sender=sender)

        client.encode(texts("as written"), EncodeRole.QUERY)

        assert input_texts("openai_embeddings", sender.calls[0]) == ["as written"]

    def test_dimensions_is_sent_only_when_the_config_sets_one(self) -> None:
        with_dimensions = FakeSender(handler("openai_embeddings", {}))
        EmbeddingClient(endpoint(dimensions=2), sender=with_dimensions).encode(texts("x"), EncodeRole.QUERY)
        without = FakeSender(handler("openai_embeddings", {}))
        EmbeddingClient(endpoint(), sender=without).encode(texts("x"), EncodeRole.QUERY)

        assert input_texts("openai_embeddings", with_dimensions.calls[0]) == ["x"]
        assert with_dimensions.calls[0].json["dimensions"] == 2
        assert "dimensions" not in without.calls[0].json

    def test_empty_input_makes_no_request(self) -> None:
        sender = FakeSender(handler("openai_embeddings", {}))
        vectors = EmbeddingClient(endpoint(), sender=sender).encode([], EncodeRole.DOCUMENT)

        assert vectors.num_items == 0
        assert sender.calls == []

    def test_empty_input_needs_no_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty call makes no request, so a missing key must not fail it (the hosted path's behaviour)."""
        monkeypatch.delenv("CO_API_KEY", raising=False)
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        client = EmbeddingClient(endpoint("cohere"), sender=FakeSender(handler("cohere", {})))

        assert client.encode([], EncodeRole.DOCUMENT).num_items == 0

    def test_media_is_refused_with_its_media_type_and_nothing_is_sent(self) -> None:
        sender = FakeSender(handler("openai_embeddings", {}))
        client = EmbeddingClient(endpoint(), sender=sender)
        page = Content(root=[ImagePart(ref=MediaRef(uri="gs://bucket/page_1.png"))])

        with pytest.raises(CapabilityError, match="image"):
            client.encode([page], EncodeRole.DOCUMENT)
        assert sender.calls == []


class TestBatching:
    def test_items_are_split_at_the_batch_size(self) -> None:
        sender = FakeSender(handler("openai_embeddings", {name: 1.0 for name in "abcde"}))
        client = EmbeddingClient(endpoint(batch_size=2), sender=sender)

        client.encode(texts(*"abcde"), EncodeRole.DOCUMENT)

        assert [len(call.json["input"]) for call in sender.calls] == [2, 2, 1]

    def test_a_batch_size_above_the_profile_cap_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="at most 96 texts per request"):
            EmbeddingClient(endpoint("cohere", batch_size=1000), sender=FakeSender(handler("cohere", {})))

    def test_a_per_call_batch_size_overrides_the_config_up_to_the_cap(self) -> None:
        sender = FakeSender(handler("openai_embeddings", {name: 1.0 for name in "abcd"}))
        client = EmbeddingClient(endpoint(batch_size=4), sender=sender)

        client.encode(texts(*"abcd"), EncodeRole.DOCUMENT, batch_size=2)
        assert [len(call.json["input"]) for call in sender.calls] == [2, 2]

        with pytest.raises(ConfigError, match="at most 128"):
            client.encode(texts(*"abcd"), EncodeRole.DOCUMENT, batch_size=129)
        with pytest.raises(ConfigError, match="at least 1"):
            client.encode(texts(*"abcd"), EncodeRole.DOCUMENT, batch_size=0)


class TestConcurrency:
    def test_at_most_concurrency_requests_are_in_flight(self) -> None:
        sender = FakeSender(handler("openai_embeddings", {name: 1.0 for name in "abcdef"}), delay_fn=lambda call: 0.01)
        client = EmbeddingClient(endpoint(batch_size=1, concurrency=2), sender=sender)

        client.encode(texts(*"abcdef"), EncodeRole.DOCUMENT)

        assert sender.peak == 2

    def test_batches_overlap_and_reassemble_in_input_order(self) -> None:
        items = list("abcd")
        values = {name: float(index + 1) for index, name in enumerate(items)}

        def delays(call: Call) -> float:
            return 0.02 * (len(items) - items.index(call.json["input"][0]))

        sender = FakeSender(handler("openai_embeddings", values), delay_fn=delays)
        client = EmbeddingClient(endpoint(batch_size=1, normalize=False), sender=sender)

        vectors = client.encode(texts(*items), EncodeRole.DOCUMENT)

        assert sender.peak > 1, "the batches did not overlap"
        assert vectors.as_matrix()[:, 0].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_batches_of_differing_dimension_are_refused(self) -> None:
        """Two well-formed batches of different width (replicas serving different models) cannot share a
        matrix: the reassembly refuses instead of crashing with a raw numpy error."""
        values = {"a": 1.0, "b": 2.0}

        def varying(call: Call) -> Reply:
            width = 2 if input_texts("openai_embeddings", call) == ["a"] else 3
            vectors = [[values[text], 1.0] + [0.0] * (width - 2) for text in input_texts("openai_embeddings", call)]
            return Reply(200, vendor_payload("openai_embeddings", vectors), {})

        sender = FakeSender(varying)
        client = EmbeddingClient(endpoint(batch_size=1, normalize=False), sender=sender)

        with pytest.raises(RequestRejectedError, match="differing dimension"):
            client.encode(texts("a", "b"), EncodeRole.DOCUMENT)

    def test_one_batch_sends_one_request(self) -> None:
        sender = FakeSender(handler("openai_embeddings", {"only": 1.0}))
        client = EmbeddingClient(endpoint(concurrency=1), sender=sender)

        vectors = client.encode(texts("only"), EncodeRole.DOCUMENT)

        assert sender.peak == 1
        assert len(sender.calls) == 1
        assert vectors.num_items == 1


class TestNormalisation:
    def test_vectors_are_l2_normalised_by_default(self) -> None:
        sender = FakeSender(handler("openai_embeddings", {"x": 3.0, "y": 0.0}))
        vectors = EmbeddingClient(endpoint(), sender=sender).encode(texts("x", "y"), EncodeRole.DOCUMENT)

        assert np.linalg.norm(vectors.as_matrix(), axis=1).tolist() == pytest.approx([1.0, 1.0])

    def test_normalize_false_keeps_the_raw_vectors(self) -> None:
        sender = FakeSender(handler("openai_embeddings", {"x": 3.0, "y": 0.0}))
        vectors = EmbeddingClient(endpoint(normalize=False), sender=sender).encode(texts("x", "y"), EncodeRole.DOCUMENT)

        assert vectors.as_matrix()[:, 0].tolist() == [3.0, 0.0]


class TestSyncBridge:
    def test_encode_runs_on_the_sender_bridge_and_matches_aencode(self) -> None:
        sender = FakeSender(handler("openai_embeddings", {"a": 1.0, "b": 2.0}))
        client = EmbeddingClient(endpoint(batch_size=1), sender=sender)

        sync = client.encode(texts("a", "b"), EncodeRole.DOCUMENT)
        async_ = asyncio.run(client.aencode(texts("a", "b"), EncodeRole.DOCUMENT))

        assert isinstance(sync, Embeddings)
        assert sync.as_matrix().tolist() == async_.as_matrix().tolist()
        assert len(sender.calls) == 4  # two batches, run twice


class TestConstruction:
    def test_max_tokens_is_refused_until_the_text_budget_mechanism_is_wired(self) -> None:
        with pytest.raises(ConfigError, match=r"max_tokens needs the text-budget mechanism, which is not wired yet"):
            EmbeddingClient(endpoint(max_tokens=8192), sender=FakeSender(handler("openai_embeddings", {})))

    @pytest.mark.parametrize("api", ("cohere", "voyage", "gemini"))
    def test_a_hosted_config_refuses_a_dimensions_cut(self, api: str) -> None:
        """The hosted profiles have no dimensions parameter; a cut that silently never reaches the wire would
        change the vectors without an error."""
        with pytest.raises(ConfigError, match="dimensions"):
            EmbeddingClient(endpoint(api, dimensions=256), sender=FakeSender(handler(api, {})))

    def test_an_explicit_api_key_env_that_is_unset_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_KEY_ENV", raising=False)
        client = EmbeddingClient(
            endpoint(base_url="http://127.0.0.1:8000/v1", api_key_env="MISSING_KEY_ENV"),
            sender=FakeSender(handler("openai_embeddings", {})),
        )

        with pytest.raises(CredentialsError, match="MISSING_KEY_ENV"):
            client.encode(texts("x"), EncodeRole.DOCUMENT)

    def test_an_unknown_api_names_the_known_adapters(self) -> None:
        with pytest.raises(ConfigError) as caught:
            EmbeddingClient(endpoint(api="not_an_adapter"), sender=FakeSender(handler("openai_embeddings", {})))

        assert "openai_embeddings" in (caught.value.hint or "")
        assert "embed" in (caught.value.hint or "")  # the hint stays inside the role's namespace

    def test_an_adapter_of_another_role_is_refused(self) -> None:
        """A registered adapter of the wrong role (a judge's, say, selected by typo) is unknown in the embed
        registry, and the error names where the name does live."""
        from rcp_ndcg.inference import adapters as registry
        from rcp_ndcg.inference.adapters.base import AdapterRole, register_adapter

        class _JudgeAdapter:
            """A minimal adapter of another role, registered for this test."""

            name = "probe_judge"
            role: ClassVar[AdapterRole] = "judge"

            def calls(self, request: Any, *, model: str) -> list[Call]:
                return []

            def interpret(self, request: Any, replies: list[Reply]) -> str:
                return ""

            def usage(self, reply: Reply) -> None:
                return None

        saved = dict(registry.base._BUILTINS)
        register_adapter(_JudgeAdapter)  # type: ignore[arg-type]
        try:
            with pytest.raises(ConfigError, match="unknown embed adapter 'probe_judge'"):
                EmbeddingClient(endpoint(api="probe_judge"), sender=FakeSender(handler("openai_embeddings", {})))
        finally:
            registry.base._BUILTINS.clear()
            registry.base._BUILTINS.update(saved)


class TestEndpoints:
    @pytest.mark.parametrize(
        ("api", "url"),
        [
            ("openai_embeddings", "https://api.openai.com/v1"),
            ("cohere", "https://api.cohere.com/v2"),
            ("voyage", "https://api.voyageai.com/v1"),
            ("gemini", "https://generativelanguage.googleapis.com/v1beta"),
        ],
    )
    def test_a_hosted_config_defaults_to_the_profile_url(self, api: str, url: str) -> None:
        client = EmbeddingClient(endpoint(api), sender=FakeSender(handler(api, {})))

        assert client.endpoint.base_url == url
        assert client.config.base_url is None  # the config itself is untouched

    def test_an_explicit_base_url_is_kept(self) -> None:
        client = EmbeddingClient(
            endpoint(base_url="http://127.0.0.1:8000/v1"), sender=FakeSender(handler("openai_embeddings", {}))
        )

        assert client.endpoint.base_url == "http://127.0.0.1:8000/v1"

    def test_the_transport_sees_no_api_key_env(self) -> None:
        client = EmbeddingClient(
            endpoint(api_key_env="MY_KEY_ENV"), sender=FakeSender(handler("openai_embeddings", {}))
        )

        assert client.config.api_key_env == "MY_KEY_ENV"
        assert client.endpoint.api_key_env is None


class TestCredentials:
    def test_an_empty_api_key_env_is_a_config_error(self) -> None:
        """An empty variable name would silently send no header; the endpoint config refuses it."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="api_key_env"):
            endpoint(api_key_env="")

    def test_a_hosted_profile_without_a_key_names_its_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CO_API_KEY", raising=False)
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        client = EmbeddingClient(endpoint("cohere"), sender=FakeSender(handler("cohere", {})))

        with pytest.raises(CredentialsError) as caught:
            client.encode(texts("x"), EncodeRole.DOCUMENT)

        assert "CO_API_KEY" in (caught.value.hint or "")
        assert "COHERE_API_KEY" in (caught.value.hint or "")

    def test_a_local_engine_sends_no_authorization_without_a_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        sender = FakeSender(handler("openai_embeddings", {}))
        client = EmbeddingClient(endpoint(base_url="http://127.0.0.1:8000/v1"), sender=sender)

        client.encode(texts("x"), EncodeRole.DOCUMENT)

        assert "Authorization" not in sender.calls[0].headers

    @pytest.mark.parametrize(
        ("api", "header", "value"),
        [
            ("openai_embeddings", "Authorization", "Bearer test-openai_api_key"),
            ("voyage", "Authorization", "Bearer test-voyage_api_key"),
            ("gemini", "x-goog-api-key", "test-gemini_api_key"),
        ],
    )
    def test_the_key_goes_in_the_profile_header(self, api: str, header: str, value: str) -> None:
        sender = FakeSender(handler(api, {}))

        EmbeddingClient(endpoint(api), sender=sender).encode(texts("x"), EncodeRole.DOCUMENT)

        assert sender.calls[0].headers[header] == value

    def test_the_config_api_key_env_overrides_the_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY_ENV", "key-from-config")
        sender = FakeSender(handler("cohere", {}))
        client = EmbeddingClient(endpoint("cohere", api_key_env="MY_KEY_ENV"), sender=sender)

        client.encode(texts("x"), EncodeRole.DOCUMENT)

        assert sender.calls[0].headers["Authorization"] == "Bearer key-from-config"


class TestTokenizerIdentity:
    def test_identity_extra_carries_the_tokenizer_sha256(self, tmp_path: Any) -> None:
        (tmp_path / "tokenizer").mkdir()
        directory = save(word_tokenizer(), tmp_path / "tokenizer")
        config = endpoint(tokenizer=str(directory))

        assert config.identity_extra() == {"tokenizer_sha256": word_tokenizer().sha256}

    def test_the_sha_is_of_the_bytes_not_the_name(self, tmp_path: Any) -> None:
        for name in ("one", "two", "three"):
            (tmp_path / name).mkdir()
        first = save(word_tokenizer(), tmp_path / "one")
        second = save(word_tokenizer(), tmp_path / "two")
        other = save(byte_bpe_tokenizer(), tmp_path / "three")

        same = endpoint(tokenizer=str(first)).identity_extra()
        assert same == endpoint(tokenizer=str(second)).identity_extra()  # same bytes, different path
        assert same != endpoint(tokenizer=str(other)).identity_extra()  # different bytes

    def test_no_tokenizer_no_extra_identity(self) -> None:
        assert endpoint().identity_extra() == {}
