"""The offline fakes: each role's wire, deterministic, under the real transport.

The fakes sit below the transport, so routing, retries, parking and usage run in every offline test; the
reranker scores each document by the same hidden ability the fake judge reads, so a tiny run's rerank and
judge agree.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import time

import httpx
import numpy as np
import pytest

import rcp_ndcg.inference.fake as fake_module
from rcp_ndcg.errors import ConfigError
from rcp_ndcg.inference import Call, Endpoint, Transport, register_fake_route
from rcp_ndcg.inference.fake import FakeEndpoint, _fake_endpoint, fake_uniform, hidden_ability

PINNED_UNIT_VECTOR = (-0.41370025636660385, -0.5207374821709475, -0.5862891756258611, -0.4625468355619963)
"""The fake's unit vector for (seed 0, "embedding", "hello") at 4 dimensions."""

PINNED_DRAW_SHA256 = "00f0b4362192cbfb1b4e0a445b9a11d584891f7baf5d721d338e712615652faf"
"""SHA-256 of the little-endian float64 bytes of the fake's unit vectors (seed 0, "token", "some text", i) for
i in 0..255 at 2048 dimensions: the draw's exact bits."""

CHAT = ("POST", "/custom/route")
"""A third-party route for the registry tests; `/chat/completions` is the judge's shipped fake."""


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Transport, "BACKOFF_S", 0.001)
    monkeypatch.setattr(Transport, "MAX_BACKOFF_S", 0.002)
    monkeypatch.setattr(Transport, "RETRY_BACKOFF_S", 0.001)
    monkeypatch.setattr(Transport, "RETRY_MAX_BACKOFF_S", 0.002)


@pytest.fixture()
def _custom_route():
    """A third-party fake route, registered and removed (the shipped routes are never torn down)."""

    def chat(request: httpx.Request, endpoint: FakeEndpoint) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "model": endpoint.model,
                "choices": [{"message": {"role": "assistant", "content": f"echo:{body['prompt']}"}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
            },
        )

    register_fake_route(*CHAT, chat)
    yield chat
    with fake_module._ROUTES_LOCK:
        fake_module._ROUTES.pop(CHAT, None)


def _transport(url: str, model: str, **config: object) -> Transport:
    """A transport for a ``fake://`` endpoint: the fake rides below the real transport."""
    return Transport(Endpoint(base_url=url, model=model, **config))  # type: ignore[arg-type]


def test_the_seed_is_the_urls_numeric_tail_and_dim_its_query() -> None:
    assert _fake_endpoint("fake://seed/3", model="m").seed == 3
    assert _fake_endpoint("fake://embed", model="m").seed == 0
    assert _fake_endpoint("fake://seed/-2", model="m").seed == -2
    assert _fake_endpoint("fake://seed/--5", model="m").seed == 0  # not a number: the documented default
    assert _fake_endpoint("fake://seed/7?dim=16", model="m").dim == 16
    assert _fake_endpoint("fake://embed", model="m").dim == 64  # DEFAULT_DIM


def test_a_route_must_name_its_path() -> None:
    with pytest.raises(ConfigError, match="route"):
        register_fake_route("POST", "", lambda request, endpoint: httpx.Response(200))
    with pytest.raises(ConfigError, match="route"):
        register_fake_route("GET", "s", lambda request, endpoint: httpx.Response(200))  # a fragment would shadow


class TestEmbeddings:
    def test_vectors_are_unit_and_of_the_urls_dimension(self) -> None:
        transport = _transport("fake://embed?dim=8", "enc")
        replies = transport.run(transport.send([Call("POST", "/embeddings", {"model": "enc", "input": "hello"})]))
        data = replies[0].body["data"]
        assert len(data) == 1 and len(data[0]["embedding"]) == 8
        assert math.isclose(sum(value * value for value in data[0]["embedding"]), 1.0, rel_tol=1e-9)

    def test_one_vector_per_input_item_in_order(self) -> None:
        transport = _transport("fake://embed?dim=4", "enc")
        replies = transport.run(transport.send([Call("POST", "/embeddings", {"input": ["a", "b"]})]))
        assert [entry["index"] for entry in replies[0].body["data"]] == [0, 1]
        assert replies[0].body["data"][0]["embedding"] != replies[0].body["data"][1]["embedding"]

    def test_the_same_text_answers_the_same_vector_everywhere(self) -> None:
        one = _transport("fake://seed/7?dim=8", "enc")
        two = _transport("fake://seed/7?dim=8", "enc")
        request = {"input": ["hello world"]}
        first = one.run(one.send([Call("POST", "/embeddings", request)]))
        second = two.run(two.send([Call("POST", "/embeddings", request)]))
        assert first[0].body["data"][0]["embedding"] == second[0].body["data"][0]["embedding"]
        another_seed = _transport("fake://seed/8?dim=8", "enc")
        other = another_seed.run(another_seed.send([Call("POST", "/embeddings", request)]))
        assert other[0].body["data"][0]["embedding"] != first[0].body["data"][0]["embedding"]

    def test_the_embeddings_usage_counts_the_items_tokens_like_the_pooling_does(self) -> None:
        transport = _transport("fake://embed?dim=4", "enc")
        replies = transport.run(transport.send([Call("POST", "/embeddings", {"input": ["", "a b"]})]))
        assert replies[0].body["usage"]["prompt_tokens"] == 3  # an empty item still holds its one token

    def test_a_chat_style_embeddings_input_is_answered_one_vector_per_message(self) -> None:
        """2e: the chat-style embeddings input (``messages`` with content parts) is answered one vector per
        message, keyed by its parts -- the vision-language embedders' input over the fake engine."""
        transport = _transport("fake://embed?dim=4", "enc")
        body = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "a caption"}, {"type": "image_url"}]},
                {"role": "user", "content": [{"type": "text", "text": "another"}]},
            ]
        }
        replies = transport.run(transport.send([Call("POST", "/embeddings", body)]))
        assert len(replies[0].body["data"]) == 2
        assert [entry["index"] for entry in replies[0].body["data"]] == [0, 1]

    def test_the_matryoshka_dimensions_cut_the_vector(self) -> None:
        transport = _transport("fake://embed?dim=8", "enc")
        request = {"input": ["hello"], "dimensions": 4}
        replies = transport.run(transport.send([Call("POST", "/embeddings", request)]))
        cut = replies[0].body["data"][0]["embedding"]
        assert len(cut) == 4
        assert math.isclose(sum(value * value for value in cut), 1.0, rel_tol=1e-9)
        again = _transport("fake://embed?dim=8", "enc")
        assert again.run(again.send([Call("POST", "/embeddings", request)]))[0].body["data"][0]["embedding"] == cut


class TestPooling:
    def test_ragged_per_token_vectors(self) -> None:
        transport = _transport("fake://seed/2?dim=4", "mv")
        replies = transport.run(
            transport.send([Call("POST", "/pooling", {"input": ["one two", "a b c d"], "task": "token_embed"})])
        )
        data = replies[0].body["data"]
        assert [len(entry["data"]) for entry in data] == [2, 4]  # ragged: 2 tokens, then 4
        assert all(len(row) == 4 for entry in data for row in entry["data"])
        assert [len(entry["prompt_token_ids"]) for entry in data] == [2, 4]

    def test_base64_honours_embed_dtype(self) -> None:
        transport = _transport("fake://seed/2?dim=4", "mv")
        request = {"input": ["one two"], "encoding_format": "base64", "embed_dtype": "float16"}
        replies = transport.run(transport.send([Call("POST", "/pooling", request)]))
        raw = base64.b64decode(replies[0].body["data"][0]["data"])
        matrix = np.frombuffer(raw, dtype=np.float16).reshape(2, 4)
        assert abs(float(np.linalg.norm(matrix[0])) - 1.0) < 0.01  # float16 rounding keeps the unit norm
        request = {"input": ["one two"], "encoding_format": "base64", "embed_dtype": "float32"}
        replies = transport.run(transport.send([Call("POST", "/pooling", request)]))
        raw = base64.b64decode(replies[0].body["data"][0]["data"])
        assert np.frombuffer(raw, dtype=np.float32).reshape(2, 4).shape == (2, 4)

    def test_float_vectors_match_the_base64_ones(self) -> None:
        transport = _transport("fake://seed/2?dim=4", "mv")
        request = {"input": ["one two"]}
        float_replies = transport.run(transport.send([Call("POST", "/pooling", request)]))
        request = {"input": ["one two"], "encoding_format": "base64", "embed_dtype": "float32"}
        b64_replies = transport.run(transport.send([Call("POST", "/pooling", request)]))
        raw = base64.b64decode(b64_replies[0].body["data"][0]["data"])
        matrix = np.frombuffer(raw, dtype=np.float32).reshape(2, 4)
        assert np.allclose(np.asarray(float_replies[0].body["data"][0]["data"]), matrix)

    def test_a_long_wide_text_answers_in_seconds(self) -> None:
        """One seeded draw per vector, never one hash per scalar: a 16k-token text at 2048 dimensions (a
        late-interaction recipe's long document) answers in seconds -- the per-scalar hashing took minutes."""
        count, dim = 16_384, 2048
        transport = fake_module.fake_transport(f"fake://seed/1?dim={dim}", model="mv")
        text = " ".join(f"w{index}" for index in range(count))
        body = {"input": [text], "task": "token_embed", "encoding_format": "base64", "embed_dtype": "float16"}
        start = time.perf_counter()
        response = transport.handle_request(httpx.Request("POST", "fake://seed/1/pooling", json=body))
        elapsed = time.perf_counter() - start
        frame = base64.b64decode(response.json()["data"][0]["data"])
        matrix = np.frombuffer(frame, dtype=np.float16).reshape(count, dim).astype(np.float32)
        assert elapsed < 30.0, f"{elapsed:.1f} s for {count} x {dim}"
        assert np.allclose(np.linalg.norm(matrix[[0, count // 2, -1]], axis=1), 1.0, atol=0.01)
        assert not np.array_equal(matrix[0], matrix[1])  # every token its own draw

    def test_a_vector_is_one_pinned_draw(self) -> None:
        """The vectors are the same on every machine: one SHAKE-256 stream per vector, pinned here (a change of
        the draw moves these values, deliberately and with a CHANGELOG entry)."""
        vector = fake_module._unit_vector(0, "embedding", "hello", dim=4)
        assert np.allclose(vector, PINNED_UNIT_VECTOR, rtol=0.0, atol=1e-12), np.asarray(vector).tolist()
        wider = fake_module._unit_vector(0, "embedding", "hello", dim=8)
        assert np.allclose(wider[:4] / np.linalg.norm(wider[:4]), vector)  # a wider draw extends the stream

    def test_the_draw_is_bit_identical_on_every_machine(self) -> None:
        """The same BITS on every machine, not only the same values to a tolerance: the norm is exactly
        rounded (``math.fsum`` of the squares), never a BLAS reduction whose last bit depends on the CPU's
        kernel (``np.linalg.norm`` differed under OpenBLAS's Prescott, Sandybridge and Haswell kernels)."""
        digest = hashlib.sha256()
        for index in range(256):
            vector = fake_module._unit_vector(0, "token", "some text", index, dim=2048)
            digest.update(np.asarray(vector, dtype="<f8").tobytes())
        assert digest.hexdigest() == PINNED_DRAW_SHA256


class TestRerank:
    def test_scores_are_the_documents_hidden_ability_best_first(self) -> None:
        from rcp_ndcg.inference.fake import hidden_ability

        docs = ["first document", "second document", "third document"]
        transport = _transport("fake://seed/5", "rr")
        request = {"model": "rr", "query": "q", "documents": docs}
        replies = transport.run(transport.send([Call("POST", "/rerank", request)]))
        results = replies[0].body["results"]
        ability = hidden_ability(5)
        assert {entry["index"]: entry["relevance_score"] for entry in results} == {
            index: ability(doc) for index, doc in enumerate(docs)
        }
        scores = [entry["relevance_score"] for entry in results]
        assert scores == sorted(scores, reverse=True)

    def test_the_ability_is_the_fake_judges(self) -> None:
        """The rerank and the judge read one hidden truth, so a tiny run's rerank and judge agree."""
        from rcp_ndcg.inference.fake import hidden_ability
        from rcp_ndcg.llm._fake import _hidden_ability

        assert _hidden_ability(3)("some document") == hidden_ability(3)("some document")

    def test_top_n_keeps_the_best_entries(self) -> None:
        docs = ["first document", "second document", "third document"]
        transport = _transport("fake://seed/5", "rr")
        request = {"model": "rr", "query": "q", "documents": docs, "top_n": 2}
        replies = transport.run(transport.send([Call("POST", "/rerank", request)]))
        results = replies[0].body["results"]
        assert len(results) == 2
        best = sorted(range(3), key=lambda index: -hidden_ability(5)(docs[index]))
        assert [entry["index"] for entry in results] == best[:2]


class TestModelsAndRoutes:
    def test_the_models_route_names_the_endpoint_model(self) -> None:
        transport = _transport("fake://embed", "enc")
        (engine,) = transport.run(transport.probe())
        assert (engine.model, engine.owned_by, engine.error) == ("enc", "fake", None)

    def test_the_fake_urls_query_does_not_break_the_probe(self) -> None:
        transport = _transport("fake://embed?dim=8", "enc")
        (engine,) = transport.run(transport.probe())
        assert (engine.model, engine.error) == ("enc", None)

    def test_a_registered_route_answers(self, _custom_route: object) -> None:
        transport = _transport("fake://seed/0", "fake")
        replies = transport.run(transport.send([Call("POST", "/custom/route", {"prompt": "hi"})]))
        assert replies[0].body["choices"][0]["message"]["content"] == "echo:hi"

    def test_a_registered_route_refuses_a_conflicting_second(self, _custom_route: object) -> None:
        with pytest.raises(ConfigError, match="already registered"):
            register_fake_route(*CHAT, lambda request, endpoint: httpx.Response(200))
        register_fake_route(*CHAT, _custom_route)  # type: ignore[arg-type]  # the same handler again: fine

    def test_an_unknown_route_is_a_404(self) -> None:
        from rcp_ndcg.errors import ProviderError

        transport = _transport("fake://embed", "enc")
        with pytest.raises(ProviderError, match=r"fake://embed/nope.*no route for POST /nope"):
            transport.run(transport.send([Call("POST", "/nope", {})]))

    def test_the_fakes_run_under_the_real_transport(self) -> None:
        """A fake replica that fails is retried and set aside like a real one: the transport above it is real."""
        calls = {"n": 0}

        def flaky(request: httpx.Request, endpoint: FakeEndpoint) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503, json={"error": {"message": "down"}})
            return httpx.Response(200, json={"ok": True})

        register_fake_route("POST", "/flaky", flaky)
        try:
            transport = _transport("fake://seed/0", "fake", max_retries=1)
            replies = transport.run(transport.send([Call("POST", "/flaky", {})]))
            assert replies[0].status == 200 and calls["n"] == 2
        finally:
            with fake_module._ROUTES_LOCK:
                fake_module._ROUTES.pop(("POST", "/flaky"), None)

    def test_the_uniform_draw_is_deterministic(self) -> None:
        assert fake_uniform("a", 1) == fake_uniform("a", 1)
        assert fake_uniform("a", 1) != fake_uniform("a", 2)
        assert 0.0 <= fake_uniform("anything") < 1.0
