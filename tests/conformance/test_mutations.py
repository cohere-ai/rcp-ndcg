"""The mutations (GPU-VALIDATION.md item 5): every gate is shown able to fail.

Change one emulated behaviour -- the over-length threshold by one token, the result ordering -- and
conformance goes red naming the record; edit a recipe's template and the staleness check fails naming
the template. The golden replay's own mutation (a perturbed rerank score) lives beside it in
``tests/e2e/test_golden_replay.py``.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import httpx

from rcp_ndcg.testing.engines import (
    EngineFacts,
    Exchange,
    StringsPrompts,
    VllmEmulator,
    compare_exchange,
)
from tests._engines import RECIPES_ROOT, corpus_of, harness, load_recipe

SHORT = "the a of to in is it"  # 7 tokens in the fixture word tokenizer
EDGE = "the a of to in is it evidence"  # 8 tokens: exactly at the tiny cap
LONG = "the a of to in is it evidence query"  # 9 tokens


def _fake_corpus(exchanges=()):
    """An in-memory corpus stand-in (``from_corpus`` reads the manifest and the raw records)."""

    class _M:
        manifest = {
            "recipe": {"id": "tiny", "revision": "f" * 40, "behaviour_fingerprint": "e" * 64},
            "engine": {"name": "vllm", "version": "0.31.0", "image": "vllm/vllm-openai:v0.31.0"},
        }
        engine = manifest["engine"]

    _M.exchanges = tuple(exchanges)
    return _M()


def _tiny(cap: int = 8) -> tuple[VllmEmulator, object]:
    """A tiny verified emulator with an exact token boundary at ``cap`` (the product's fixture word
    tokenizer: one word is one token)."""
    from tests._tokenizers import word_tokenizer

    tokenizer = word_tokenizer()
    facts = EngineFacts("vllm", "0.31.0", "tiny", "fixtures/Tiny", cap)
    emulator = VllmEmulator.from_corpus(_fake_corpus(), StringsPrompts(), tokenizer, facts, dim=4)
    return emulator, tokenizer


def _body(text: str) -> dict:
    return {"model": "tiny", "input": [text], "encoding_format": "float"}


def test_the_over_length_threshold_moving_by_one_token_makes_conformance_red() -> None:
    """Mutation: the refusal threshold one token earlier. Every recorded exchange that fit now
    refuses -- conformance goes red with the status change named."""
    emulator, tokenizer = _tiny(cap=8)
    assert tokenizer.count(SHORT, add_special_tokens=True) == 7
    # recorded behaviour: 7 and 8 tokens fit (200), 9 tokens is refused exactly as the engine refuses it
    ok = emulator.answer("/v1/embeddings", "POST", _body(SHORT))
    at_edge = emulator.answer("/v1/embeddings", "POST", _body(EDGE))
    refused = emulator.answer("/v1/embeddings", "POST", _body(LONG))
    assert ok.status_code == 200 and at_edge.status_code == 200 and refused.status_code == 400
    assert "at least 9 input tokens" in refused.json()["error"]["message"]

    recorded_ok = Exchange(0, "POST", "/v1/embeddings", _body(EDGE), 200, {}, at_edge.json())
    recorded_refused = Exchange(1, "POST", "/v1/embeddings", _body(LONG), 400, {}, refused.json())
    assert compare_exchange(recorded_ok, "POST", "/v1/embeddings", 200, at_edge.json(), (0.0, 0.0)) == []
    assert compare_exchange(recorded_refused, "POST", "/v1/embeddings", 400, refused.json(), (0.0, 0.0)) == []

    # the mutation: the over-length threshold moved by one token (the cap reads one less)
    mutant = replace(emulator, facts=replace(emulator.facts, max_model_len=7))
    answer = mutant.answer("/v1/embeddings", "POST", _body(EDGE))
    problems = compare_exchange(
        recorded_ok, "POST", "/v1/embeddings", answer.status_code, answer.json(), (0.0, 0.0)
    )
    assert problems and problems[0].startswith("status 400"), problems
    # and the boundary is exactly one token wide: the 8-token edge now refuses where 7 still fits
    assert mutant.answer("/v1/embeddings", "POST", _body(SHORT)).status_code == 200
    assert mutant.answer("/v1/embeddings", "POST", _body(EDGE)).status_code == 400


def test_the_result_ordering_making_conformance_red() -> None:
    """Mutation: the rerank results come back worst-first. The recorded order is best-first (the
    engine's ranked replies), so conformance goes red naming the permuted entries."""
    from rcp_ndcg.testing.engines import EnginePrompts

    emulator, tokenizer = _tiny(cap=128)
    emulator.strategy = EnginePrompts(builder=lambda query, documents: f"{query}|{'|'.join(documents)}")
    emulator.slot = "score_list"
    body = {"model": "tiny", "query": "query", "documents": ["page", "answer"]}
    original = emulator.answer("/rerank", "POST", body)
    scores = [entry["relevance_score"] for entry in original.json()["results"]]
    assert scores == sorted(scores, reverse=True)  # the engine replies ranked, best first

    # the mutation: the results come back worst-first
    flipped = original.json()
    flipped["results"] = list(reversed(flipped["results"]))
    problem = compare_exchange(
        Exchange(0, "POST", "/rerank", body, 200, {}, original.json()),
        "POST",
        "/rerank",
        200,
        flipped,
        (0.0, 0.0),
    )
    assert problem and any(".results[0].index" in item for item in problem), problem


def test_editing_a_recipes_template_fails_staleness_naming_the_template(tmp_path: Path) -> None:
    """Mutation: one byte in the recipe's template file. The staleness check names the template input."""
    from rcp_ndcg_vllm.fingerprint import fingerprint_changes, fingerprint_inputs

    harness()
    original = load_recipe("qwen3-vl-reranker-2b")
    corpus = corpus_of(original)
    recorded = dict(corpus.manifest["recipe"]["fingerprint_inputs"])

    copy = tmp_path / "qwen3-vl-reranker-2b"
    shutil.copytree(RECIPES_ROOT / "qwen3-vl-reranker-2b", copy)
    template = copy / "template.jinja"
    template.write_bytes(template.read_bytes() + b"\n{# edited #}\n")
    edited = load_recipe(copy)

    changed = fingerprint_changes(recorded, fingerprint_inputs(edited))
    assert changed == ["template_file"], changed
    assert fingerprint_changes(recorded, fingerprint_inputs(original)) == []


def test_the_surrogate_marking_is_honest(tmp_path: Path) -> None:
    """An unseen input answers a declared marked surrogate; an observed input answers the replay (the
    guard the golden replay asserts against)."""
    emulator, _ = _tiny()
    first = emulator.answer("/v1/embeddings", "POST", _body(SHORT))
    assert first.headers["x-rcp-ndcg-emulator-source"] == "surrogate"  # nothing recorded yet: all unseen
    observed = Exchange(0, "POST", "/v1/embeddings", _body(SHORT), 200, {}, first.json())
    replayed = VllmEmulator.from_corpus(
        _fake_corpus((observed,)),
        StringsPrompts(),
        emulator.tokenizer,
        emulator.facts,
        dim=4,
    )
    again = replayed.answer("/v1/embeddings", "POST", _body(SHORT))
    unseen = replayed.answer("/v1/embeddings", "POST", _body("evidence page"))  # unobserved, in budget
    refused = replayed.answer("/v1/embeddings", "POST", _body(LONG))  # over the cap: no model output
    assert again.headers["x-rcp-ndcg-emulator-source"] == "replayed"
    assert unseen.headers["x-rcp-ndcg-emulator-source"] == "surrogate"
    assert "x-rcp-ndcg-emulator-source" not in refused.headers  # an error carries no model output
    assert replayed.answer_log[-2:] == ["replayed", "surrogate"]
    _ = httpx  # the transport shape is exercised through fake:// elsewhere