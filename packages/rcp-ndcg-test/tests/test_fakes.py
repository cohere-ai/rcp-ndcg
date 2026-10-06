"""The fake-engine seam: the protocol, the registry and the shipped embed fake."""

from __future__ import annotations

from pathlib import Path

import pytest
from rcp_ndcg_test.errors import ConformanceError
from rcp_ndcg_test.fakes import (
    FakeEmbedEngine,
    FakeEngine,
    FakeReply,
    fake_engine_for,
    fake_http_transport,
    fixture_path,
    package_tokenizer_path,
    register_fake_engine,
    registered_fake_engines,
    unregister_fake_engine,
)

from rcp_ndcg.data.tokenizer import load_tokenizer


class _MinimalFake:
    """The smallest object that satisfies the protocol (the shape a test or a lane writes)."""

    recipe_id = "minimal-fake"
    name = "minimal"

    def handle(self, method: str, path: str, body: object) -> FakeReply:
        return FakeReply(status=200, json_body={"ok": True})


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Every test starts and ends with just the shipped fake registered."""
    import rcp_ndcg_test.fakes as fakes_module

    fakes_module.unregister_fake_engine("minimal-fake")
    yield
    fakes_module.unregister_fake_engine("minimal-fake")
    if "fake-embed" not in fakes_module.registered_fake_engines():
        fakes_module.register_fake_engine(fakes_module.FakeEmbedEngine())


def test_the_registry_round_trips_by_recipe_id() -> None:
    engine = _MinimalFake()
    register_fake_engine(engine)
    assert fake_engine_for("minimal-fake") is engine
    assert registered_fake_engines() == ("fake-embed", "minimal-fake")
    unregister_fake_engine("minimal-fake")
    assert registered_fake_engines() == ("fake-embed",)


def test_a_missing_fake_names_the_hint() -> None:
    with pytest.raises(ConformanceError, match="register_fake_engine"):
        fake_engine_for("no-such-recipe")


def test_a_second_fake_for_one_recipe_is_refused() -> None:
    register_fake_engine(_MinimalFake())
    with pytest.raises(ConformanceError, match="already registered"):
        register_fake_engine(_MinimalFake())


def test_an_object_without_the_protocol_is_refused() -> None:
    with pytest.raises(ConformanceError, match="not a FakeEngine"):
        register_fake_engine(object())  # type: ignore[arg-type]


def test_the_shipped_fake_answers_the_embed_route_deterministically() -> None:
    engine = FakeEmbedEngine()
    first = engine.handle("POST", "/v1/embeddings", {"input": "a short text"})
    again = engine.handle("POST", "/v1/embeddings", {"input": "a short text"})
    other = engine.handle("POST", "/v1/embeddings", {"input": "another text"})
    assert first == again
    assert first.json_body is not None and other.json_body is not None
    assert first.json_body["data"][0]["embedding"] != other.json_body["data"][0]["embedding"]
    unit = first.json_body["data"][0]["embedding"]
    assert abs(sum(value * value for value in unit) - 1.0) < 1e-6


def test_the_shipped_fake_refuses_non_string_inputs() -> None:
    reply = FakeEmbedEngine().handle("POST", "/v1/embeddings", {"input": [1, 2, 3]})
    assert reply.status == 400 and "string inputs" in (reply.json_body or {}).get("error", {}).get("message", "")


def test_the_shipped_fake_answers_models_and_404s_unknown_routes() -> None:
    engine = FakeEmbedEngine()
    assert engine.handle("GET", "/v1/models", None).status == 200
    unknown = engine.handle("POST", "/v1/whisper", {})
    assert unknown.status == 404 and "no route" in (unknown.json_body or {}).get("error", {}).get("message", "")


def test_the_shipped_fakes_tokenize_matches_the_fixture_tokenizer() -> None:
    engine = FakeEmbedEngine()
    prompt = "doc: the anchors survive the cut [END]"
    reply = engine.handle("POST", "/tokenize", {"model": "fake-embed", "prompt": prompt, "add_special_tokens": False})
    assert reply.status == 200 and reply.json_body is not None
    tokenizer = load_tokenizer(str(package_tokenizer_path()))
    assert reply.json_body["tokens"] == tokenizer.ids(prompt, add_special_tokens=False)
    assert reply.json_body["count"] == len(reply.json_body["tokens"])
    special = engine.handle("POST", "/v1/tokenize", {"prompt": prompt, "add_special_tokens": True})
    assert special.json_body is not None
    assert len(special.json_body["tokens"]) > len(reply.json_body["tokens"])


def test_the_shipped_fake_is_a_runtime_protocol_instance() -> None:
    assert isinstance(FakeEmbedEngine(), FakeEngine)
    assert not isinstance(object(), FakeEngine)


def test_the_packaged_fixture_files_are_on_disk() -> None:
    assert package_tokenizer_path().is_file()
    assert (fixture_path("recipes", "fake-embed", "recipe.yaml")).is_file()
    assert fixture_path("cases", "fake-embed", "short-single.yaml").is_file()
    assert (Path(fixture_path()) / "tokenizer.json").is_file()


def test_the_fake_transport_parses_the_body_for_the_engine() -> None:
    import asyncio

    class Recording:
        recipe_id = "rec"
        name = "rec"

        def __init__(self) -> None:
            self.seen: object = None

        def handle(self, method: str, path: str, body: object) -> FakeReply:
            self.seen = body
            return FakeReply(status=200, json_body={"echo": body})

    import httpx

    engine = Recording()
    transport = fake_http_transport(engine)
    request = httpx.Request("POST", "http://fake/v1/embeddings", json={"input": "hello"})
    response = asyncio.run(transport.handle_async_request(request))
    assert response.status_code == 200 and engine.seen == {"input": "hello"}
    broken = asyncio.run(
        transport.handle_async_request(
            httpx.Request(
                "POST",
                "http://fake/v1/embeddings",
                content=b"{not json",
                headers={"content-type": "application/json"},
            )
        )
    )
    assert broken.status_code == 400
