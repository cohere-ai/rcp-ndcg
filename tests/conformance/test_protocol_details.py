"""Protocol details the review found (each test failed before its fix)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.testing.engines import Exchange, behaviour_diff
from tests._engines import emulator_for

QUERY = {"query": "What is the capital of France?"}
DOCUMENTS = {"documents": ["Paris is the capital of France.", "Berlin is the capital of Germany."]}


def test_top_n_zero_returns_the_whole_ranked_list() -> None:
    """vLLM v0.31.0: ``top_n`` defaults to 0, which means every document
    (``vllm/entrypoints/pooling/scoring/serving.py``: ``top_n if top_n > 0 else len(...)``)."""
    emulator = emulator_for("qwen3-reranker-8b")
    answer = emulator.answer("/rerank", "POST", {**QUERY, **DOCUMENTS, "top_n": 0, "use_activation": True})
    assert len(answer.json()["results"]) == 2
    one = emulator.answer("/rerank", "POST", {**QUERY, **DOCUMENTS, "top_n": 1, "use_activation": True})
    assert len(one.json()["results"]) == 1


def test_a_surrogate_vector_has_the_observed_width() -> None:
    """An unseen input in a batch with an observed one must not answer a ragged batch: the surrogate
    takes the width the corpus observed for the model."""
    emulator = emulator_for("qwen3-embedding-0.6b")
    body = emulator.answer("/v1/embeddings", "POST", {"input": ["What is the capital of France?", "unseen text"]})
    assert body.headers["x-rcp-ndcg-emulator-source"] == "mixed"
    widths = {len(item["embedding"]) for item in body.json()["data"]}
    assert widths == {1024}, widths


def _corpus(
    tmp_path: Path, response: dict | None, fingerprint: str, headers: dict | None = None, raw: bytes | None = None
):
    from tests._engines import observation_corpus

    exchange = Exchange(1, "POST", "/rerank", {**QUERY, **DOCUMENTS}, 200, headers or {}, response, response_raw=raw)
    return observation_corpus(tmp_path / fingerprint, [exchange], recipe_id="r", fingerprint=fingerprint)


def test_the_behaviour_diff_ignores_volatile_ids_and_names_protocol_changes(tmp_path: Path) -> None:
    reply = {"id": "score-aaaa", "model": "m", "usage": {"prompt_tokens": 3}, "results": []}
    same = behaviour_diff(_corpus(tmp_path, reply, "a"), _corpus(tmp_path, {**reply, "id": "score-bbbb"}, "b"))
    assert same["summary"]["changed"] == 0, same["inputs"]
    renamed = behaviour_diff(_corpus(tmp_path, reply, "a"), _corpus(tmp_path, {**reply, "model": "other"}, "b"))
    assert renamed["summary"]["changed"] == 1 and renamed["inputs"][0]["protocol_changed"]
    header = behaviour_diff(
        _corpus(tmp_path, reply, "a", {"content-type": "application/json"}),
        _corpus(tmp_path, reply, "b", {"content-type": "text/plain"}),
    )
    assert header["summary"]["changed"] == 1, header["inputs"]


def test_the_behaviour_diff_compares_binary_bodies(tmp_path: Path) -> None:
    before = _corpus(tmp_path, None, "a", raw=b"\x00\x3c")
    moved = behaviour_diff(before, _corpus(tmp_path, None, "b", raw=b"\x00\x3d"))
    assert moved["summary"]["changed"] == 1, moved["inputs"]
    same = behaviour_diff(before, _corpus(tmp_path, None, "c", raw=b"\x00\x3c"))
    assert same["summary"]["changed"] == 0, same["inputs"]


def test_the_registry_names_only_the_requested_engine_versions_fingerprints() -> None:
    from dataclasses import replace

    from rcp_ndcg.testing.engines import registry

    emulator = emulator_for("qwen3-reranker-8b")
    assert emulator.verified is not None
    other = replace(
        emulator, verified=replace(emulator.verified, engine_version="0.32.0", behaviour_fingerprint="9" * 64)
    )
    registry.register(other)
    try:
        assert registry.fingerprints("vllm", "0.31.0", "qwen3-reranker-8b") == [emulator.verified.behaviour_fingerprint]
        with pytest.raises(ConfigError) as error:
            registry.resolve("vllm", "0.31.0", "f" * 64, "qwen3-reranker-8b")
        assert "9999" not in str(error.value), error.value
    finally:
        registry.clear()


@pytest.mark.parametrize("url", ["fake://vllm-1/qwen3-reranker-8b", "fake://vllm-0.31/qwen3-reranker-8b"])
def test_the_engine_host_is_one_pattern(url: str) -> None:
    """``rcp_ndcg.inference.fake`` routes a URL to the emulators exactly when the emulators read its
    host as an engine and version (one pattern, never two)."""
    from rcp_ndcg.inference.fake import RE_ENGINE_URL
    from rcp_ndcg.testing.engines import split_engine_host

    host = url.removeprefix("fake://").split("/", 1)[0]
    try:
        split_engine_host(host)
        parsed = True
    except ConfigError:
        parsed = False
    assert parsed == bool(RE_ENGINE_URL.match(url)), url
