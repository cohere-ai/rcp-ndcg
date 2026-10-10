"""The mutations (GPU-VALIDATION.md item 5): every gate is shown able to fail.

Change one emulated behaviour -- the over-length threshold by one token, the result ordering -- and
conformance goes red naming the record; edit a recipe's template and the staleness check fails naming
the template. The golden replay's own mutation (a perturbed rerank score) lives beside it in
``tests/e2e/test_golden_replay.py``.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import replace
from pathlib import Path

import httpx
from rcp_ndcg_test.engines import (
    EngineFacts,
    Exchange,
    StringsPrompts,
    VllmEmulator,
    compare_exchange,
)
from tests._engines import RECIPES_ROOT, corpus_of, harness, load_recipe, observation_corpus

SHORT = "the a of to in is it"  # 7 tokens in the fixture word tokenizer
EDGE = "the a of to in is it evidence"  # 8 tokens: exactly at the tiny cap
LONG = "the a of to in is it evidence query"  # 9 tokens


def _tiny(directory: Path, cap: int = 8, exchanges=()) -> tuple[VllmEmulator, object]:
    """A tiny verified emulator with an exact token boundary at ``cap`` (the product's fixture word
    tokenizer: one word is one token), over a corpus of ``exchanges``."""
    from tests._tokenizers import word_tokenizer

    tokenizer = word_tokenizer()
    facts = EngineFacts("vllm", "0.31.0", "tiny", "fixtures/Tiny", cap)
    corpus = observation_corpus(directory, exchanges)
    emulator = VllmEmulator.from_corpus(corpus, StringsPrompts(), tokenizer, facts, dim=4)
    return emulator, tokenizer


def _body(text: str) -> dict:
    return {"model": "tiny", "input": [text], "encoding_format": "float"}


def test_the_over_length_threshold_moving_by_one_token_makes_conformance_red(tmp_path: Path) -> None:
    """Mutation: the refusal threshold one token earlier. Every recorded exchange that fit now
    refuses -- conformance goes red with the status change named."""
    emulator, tokenizer = _tiny(tmp_path, cap=8)
    assert tokenizer.count(SHORT, add_special_tokens=True) == 7
    # recorded behaviour: 7 and 8 tokens fit (200), 9 tokens is refused exactly as the engine refuses it
    ok = emulator.answer("/v1/embeddings", "POST", _body(SHORT))
    at_edge = emulator.answer("/v1/embeddings", "POST", _body(EDGE))
    refused = emulator.answer("/v1/embeddings", "POST", _body(LONG))
    assert ok.status_code == 200 and at_edge.status_code == 200 and refused.status_code == 400
    assert "at least 9 input tokens" in refused.json()["error"]["message"]

    recorded_ok = Exchange(0, "POST", "/v1/embeddings", _body(EDGE), 200, {}, at_edge.json())
    recorded_refused = Exchange(1, "POST", "/v1/embeddings", _body(LONG), 400, {}, refused.json())
    assert compare_exchange(recorded_ok, at_edge, None) == []
    assert compare_exchange(recorded_refused, refused, None) == []

    # the mutation: the over-length threshold moved by one token (the cap reads one less)
    mutant = replace(emulator, facts=replace(emulator.facts, max_model_len=7))
    answer = mutant.answer("/v1/embeddings", "POST", _body(EDGE))
    problems = compare_exchange(recorded_ok, answer, None)
    assert problems and problems[0].startswith("status 400"), problems
    # and the boundary is exactly one token wide: the 8-token edge now refuses where 7 still fits
    assert mutant.answer("/v1/embeddings", "POST", _body(SHORT)).status_code == 200
    assert mutant.answer("/v1/embeddings", "POST", _body(EDGE)).status_code == 400


def test_the_result_ordering_making_conformance_red(tmp_path: Path) -> None:
    """Mutation: the rerank results come back worst-first. The recorded order is best-first (the
    engine's ranked replies), so conformance goes red naming the permuted entries."""
    from rcp_ndcg_test.engines import EnginePrompts

    emulator, tokenizer = _tiny(tmp_path, cap=128)
    emulator.strategy = EnginePrompts(builder=lambda query, documents: f"{query}|{'|'.join(documents)}")
    body = {"model": "tiny", "query": "query", "documents": ["page", "answer"]}
    original = emulator.answer("/rerank", "POST", body)
    scores = [entry["relevance_score"] for entry in original.json()["results"]]
    assert scores == sorted(scores, reverse=True)  # the engine replies ranked, best first

    # the mutation: the results come back worst-first
    flipped = original.json()
    flipped["results"] = list(reversed(flipped["results"]))
    problem = compare_exchange(
        Exchange(0, "POST", "/rerank", body, 200, {}, original.json()),
        httpx.Response(200, json=flipped),
        None,
    )
    assert problem and any(".results[0].index" in item for item in problem), problem


def test_editing_a_recipes_template_fails_staleness_naming_the_template(tmp_path: Path) -> None:
    """Mutation: one byte in the recipe's template file. The staleness gate fails and NAMES the template
    (through the corpus scan, so an edited recipe can never crash the check with a missing directory)."""
    from rcp_ndcg_test.fingerprint import fingerprint_changes, fingerprint_inputs

    harness()
    original = load_recipe("zerank-2-reranker")
    corpus = corpus_of(original)
    recorded = dict(corpus.manifest["recipe"]["fingerprint_inputs"])

    copy = tmp_path / "zerank"  # the variant's family directory (decision 34)
    shutil.copytree(RECIPES_ROOT / "zerank", copy)
    template = copy / "template.jinja"
    template.write_bytes(template.read_bytes() + b"\n{# edited #}\n")
    edited = load_recipe("zerank-2-reranker", root=tmp_path)

    changed = fingerprint_changes(recorded, fingerprint_inputs(edited))
    assert changed == ["template_file"], changed
    # the gate's own lookup path (corpus_of scans manifests) fails for the edited recipe, naming the template
    import pytest
    from rcp_ndcg_test.changes import StaleCorpusError

    with pytest.raises(StaleCorpusError) as error:
        corpus_of(edited)
    assert "['template_file']" in str(error.value), error.value
    assert fingerprint_changes(recorded, fingerprint_inputs(original)) == []


def test_the_surrogate_marking_is_honest(tmp_path: Path) -> None:
    """An unseen input answers a declared marked surrogate; an observed input answers the replay (the
    guard the golden replay asserts against)."""
    emulator, _ = _tiny(tmp_path / "empty")
    first = emulator.answer("/v1/embeddings", "POST", _body(SHORT))
    assert first.headers["x-rcp-ndcg-emulator-source"] == "surrogate"  # nothing recorded yet: all unseen
    observed = Exchange(0, "POST", "/v1/embeddings", _body(SHORT), 200, {}, first.json())
    replayed, _ = _tiny(tmp_path / "observed", exchanges=(observed,))
    again = replayed.answer("/v1/embeddings", "POST", _body(SHORT))
    unseen = replayed.answer("/v1/embeddings", "POST", _body("evidence page"))  # unobserved, in budget
    refused = replayed.answer("/v1/embeddings", "POST", _body(LONG))  # over the cap: no model output
    assert again.headers["x-rcp-ndcg-emulator-source"] == "replayed"
    assert unseen.headers["x-rcp-ndcg-emulator-source"] == "surrogate"
    assert "x-rcp-ndcg-emulator-source" not in refused.headers  # an error carries no model output
    assert replayed.answer_log[-2:] == ["replayed", "surrogate"]


def test_an_edited_recipe_fails_the_gate_naming_the_changed_input(tmp_path: Path) -> None:
    """B2: the gate resolves corpora by scanning manifests and compares the recorded fingerprint inputs
    with the recomputed ones. An edited recipe never silently resolves another fingerprint's corpus,
    and never dies on a missing directory: it fails naming what moved."""
    import pytest
    from rcp_ndcg_test.changes import StaleCorpusError

    copy = tmp_path / "zerank"  # the variant's family directory (decision 34)
    shutil.copytree(RECIPES_ROOT / "zerank", copy)
    recipe = copy / "family.yaml"
    text = recipe.read_text(encoding="utf-8")
    budget = re.search(r"^(  max_tokens: )(\d+)(.*)$", text, flags=re.MULTILINE)
    assert budget is not None
    recipe.write_text(
        text.replace(budget.group(0), f"{budget.group(1)}{int(budget.group(2)) - 1}{budget.group(3)}"), encoding="utf-8"
    )
    edited = load_recipe("zerank-2-reranker", root=tmp_path)
    with pytest.raises(StaleCorpusError) as error:
        corpus_of(edited)
    assert "client.max_tokens" in str(error.value), error.value
    assert "no manifest" not in str(error.value)


def test_the_surrogate_is_one_draw_per_vector_and_pins_no_values(monkeypatch) -> None:
    """The declared surrogate is deterministic, unit-norm and keyed by its parts (the emulator seeds the
    embedding draw with the model input, so a requested cut is a slice of the same full-width vector); it
    draws once per vector (the offline fake's own per-vector draw), so a long surrogate matrix stays fast.
    No test pins its values: they follow the offline fake's draw, which may change (it changed from one hash
    per component to one stream per vector)."""
    import math
    import time

    from rcp_ndcg_test.engines import surrogate_matrix, surrogate_vector

    first = surrogate_vector(0, "embedding", "key-a", dim=64)
    assert first == surrogate_vector(0, "embedding", "key-a", dim=64)
    assert first != surrogate_vector(0, "embedding", "key-b", dim=64)
    assert len(first) == 64 and math.isclose(math.fsum(v * v for v in first), 1.0, rel_tol=1e-9)
    from rcp_ndcg.inference import fake

    calls = []
    scalar_draw = fake.fake_uniform
    monkeypatch.setattr(fake, "fake_uniform", lambda *parts: calls.append(parts) or scalar_draw(*parts))
    surrogate_matrix(0, "pooling", "key-c", tokens=8, dim=64)
    assert calls == [], f"{len(calls)} scalar draws for 8 vectors"
    started = time.perf_counter()
    matrix = surrogate_matrix(0, "pooling", "key-a", tokens=2048, dim=1024)
    assert len(matrix) == 2048 and len(matrix[0]) == 1024
    assert time.perf_counter() - started < 10.0, "one draw per vector, never one hash per component"
