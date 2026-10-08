"""Framing (M3): the conformance check compares the status, the headers that matter and the raw bytes
as the transport reads them, refuses an undecodable body loudly, and the emulators declare every route
no recording covers.

The unobserved routes follow vLLM v0.31.0's source (``PoolingResponse``,
``vllm/entrypoints/pooling/utils.py::build_pooling_bytes_streaming_response``,
``vllm/entrypoints/serve/tokenize/protocol.py``) and say so in every reply.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest
from rcp_ndcg_test.engines import Exchange, compare_exchange
from tests._engines import emulator_for

from rcp_ndcg.errors import DataError

JSON = {"content-type": "application/json", "server": "uvicorn"}


def _reply(status: int, content: bytes, headers: dict[str, str]) -> httpx.Response:
    return httpx.Response(status, content=content, headers=headers)


def _recorded(body: dict, raw: bytes | None = None) -> Exchange:
    return Exchange(0, "POST", "/rerank", {"query": "q"}, 200, JSON, body, response_raw=raw)


BODY = {"id": "score-1", "model": "m", "usage": {"prompt_tokens": 3, "total_tokens": 3}, "results": []}


def _engine_bytes(body: dict) -> bytes:
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def test_a_wrong_content_type_fails_the_conformance_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mutation: the emulator answers ``text/plain`` with the right body. The suite's own replay path
    goes red naming the header (before, it compared parsed bodies only)."""
    from rcp_ndcg_test import engines
    from tests.conformance.test_conformance import corpus_dirs, replay_problems

    directory = next(path for path in corpus_dirs() if "qwen3-reranker-8b" in str(path))
    assert replay_problems(directory)[1] == []
    original = engines._json

    def plain(status: int, body: object) -> httpx.Response:
        response = original(status, body)
        response.headers["content-type"] = "text/plain"
        return response

    monkeypatch.setattr(engines, "_json", plain)
    problems = replay_problems(directory)[1]
    assert problems and all("header content-type" in problem for problem in problems), problems


def test_headers_and_raw_bytes_are_compared() -> None:
    same = _reply(200, _engine_bytes({**BODY, "id": "score-2"}), JSON)
    assert compare_exchange(_recorded(BODY, _engine_bytes(BODY)), same, None) == []  # the id is volatile
    spaced = _reply(200, json.dumps({**BODY, "id": "score-2"}).encode(), JSON)  # ", " and ": " separators
    problems = compare_exchange(_recorded(BODY, _engine_bytes(BODY)), spaced, None)
    assert problems and "raw bytes" in problems[0], problems
    server = _reply(200, _engine_bytes(BODY), {**JSON, "server": "hypercorn"})
    assert any("header server" in problem for problem in compare_exchange(_recorded(BODY), server, None))


def test_an_undecodable_replayed_body_is_a_named_problem() -> None:
    for content in (b"\xff\xfe not utf-8", b'{"truncated": '):
        problems = compare_exchange(_recorded(BODY), _reply(200, content, JSON), None)
        assert problems and "undecodable" in problems[0], problems


def test_binary_framing_is_compared_byte_for_byte() -> None:
    frame = b"\x00\x3c\x00\x3c"
    metadata = {"id": "pool-1", "created": 1, "model": "m", "data": [], "usage": {}}
    recorded = Exchange(
        0,
        "POST",
        "/pooling",
        {"input": ["x"], "encoding_format": "bytes"},
        200,
        {"content-type": "application/octet-stream", "metadata": json.dumps(metadata)},
        None,
        response_raw=frame,
    )
    headers = {"content-type": "application/octet-stream", "metadata": json.dumps({**metadata, "id": "pool-2"})}
    assert compare_exchange(recorded, _reply(200, frame, headers), None) == []  # metadata's id is volatile
    problems = compare_exchange(recorded, _reply(200, b"\x00\x3c\x00\x3d", headers), None)
    assert problems and "bytes" in problems[0], problems


def test_an_undecodable_recorded_body_is_refused_loudly(tmp_path: Path) -> None:
    """A recorded 2xx body whose outputs cannot be derived (a base64 /pooling item: vLLM sends no
    shape) is refused naming the record -- never an empty or guessed replay table."""
    from rcp_ndcg_test.engines import EngineFacts, StringsPrompts, VllmEmulator
    from tests._engines import observation_corpus

    from tests._tokenizers import word_tokenizer

    payload = base64.b64encode(b"\x00\x3c" * 6).decode()
    reply = {"object": "list", "data": [{"index": 0, "object": "pooling", "data": payload}]}
    exchange = Exchange(3, "POST", "/pooling", {"input": ["the a of"], "encoding_format": "base64"}, 200, JSON, reply)
    facts = EngineFacts("vllm", "0.31.0", "tiny", "fixtures/Tiny", 8)
    corpus = observation_corpus(tmp_path, [exchange])
    with pytest.raises(DataError) as error:
        VllmEmulator.from_corpus(corpus, StringsPrompts("token_vector"), word_tokenizer(), facts)
    assert "#3" in str(error.value) and "shape" in str(error.value)


def test_every_unobserved_route_is_declared_in_the_reply_and_the_record() -> None:
    emulator = emulator_for("qwen3-reranker-8b")
    assert "POST /rerank" in emulator.observed_routes
    assert {"POST /pooling", "POST /v1/embeddings", "POST /tokenize"} <= set(emulator.unobserved_routes)
    rerank = emulator.answer("/rerank", "POST", {"query": "q", "documents": ["d"]})
    tokenize = emulator.answer("/tokenize", "POST", {"prompt": "q"})
    assert rerank.headers["x-rcp-ndcg-emulator-route"] == "observed"
    assert tokenize.headers["x-rcp-ndcg-emulator-route"] == "unobserved"


def test_the_unobserved_routes_follow_the_engine_source() -> None:
    emulator = emulator_for("qwen3-embedding-0.6b")
    pooled = emulator.answer("/pooling", "POST", {"input": ["What is the capital of France?"]}).json()
    assert list(pooled) == ["id", "object", "created", "model", "data", "usage"]
    assert pooled["id"].startswith("pool-") and set(pooled["data"][0]) == {"index", "object", "data"}
    framed = emulator.answer("/pooling", "POST", {"input": ["a", "b"], "encoding_format": "bytes"})
    metadata = json.loads(framed.headers["metadata"])
    assert list(metadata) == ["id", "created", "model", "data", "usage"]
    assert [frame["index"] for frame in metadata["data"]] == [0, 1]
    tokenized = emulator.answer("/tokenize", "POST", {"prompt": "What is", "add_special_tokens": False}).json()
    assert list(tokenized) == ["count", "max_model_len", "tokens", "token_strs"]
    assert tokenized["count"] == len(tokenized["tokens"]) and tokenized["token_strs"] is None
