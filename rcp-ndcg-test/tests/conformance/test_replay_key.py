"""The model layer's replay key (B1): an output is replayed only for the exact behaviour-shaping
context it was observed under -- the engine prompt(s) plus every request field that changes what the
model returns. An unobserved context answers the declared surrogate, a field the emulator does not
model is refused with a marked 400, and a corpus whose one key holds different outputs is refused.

The field classes follow vLLM v0.31.0's request models (``vllm/entrypoints/pooling/*/protocol.py``):
a field the route's model does not declare is ignored by the engine
(``vllm/entrypoints/serve/engine/protocol.py``, "fields were present in the request but ignored").
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests._engines import corpus_of, emulator_for, load_recipe

from rcp_ndcg.errors import DataError

SOURCE = "x-rcp-ndcg-emulator-source"


def _recorded(recipe_id: str, sequence: int) -> dict:
    corpus = corpus_of(load_recipe(recipe_id))
    from rcp_ndcg_test.engines import exchanges_of

    (exchange,) = [item for item in exchanges_of(corpus) if item.sequence == sequence]
    return dict(exchange.request_body)


def test_an_observed_request_replays() -> None:
    emulator = emulator_for("qwen3-embedding-0.6b")
    answer = emulator.answer("/v1/embeddings", "POST", _recorded("qwen3-embedding-0.6b", 1))
    assert answer.status_code == 200 and answer.headers[SOURCE] == "replayed"


@pytest.mark.parametrize(
    ("recipe_id", "sequence", "field", "value"),
    [
        ("qwen3-embedding-0.6b", 1, "add_special_tokens", False),  # another tokenization of the prompt
        ("qwen3-embedding-0.6b", 1, "use_activation", False),
        ("zerank-2-reranker", 1, "use_activation", False),  # raw logit instead of the probability
        ("zerank-1-small-reranker", 1, "use_activation", False),
    ],
)
def test_an_unobserved_context_answers_the_marked_surrogate(
    recipe_id: str, sequence: int, field: str, value: object
) -> None:
    body = _recorded(recipe_id, sequence)
    assert body.get(field) != value
    body[field] = value
    path = "/v1/embeddings" if "embedding" in recipe_id else "/rerank"
    answer = emulator_for(recipe_id).answer(path, "POST", body)
    assert answer.status_code == 200
    assert answer.headers[SOURCE] == "surrogate", f"{field}={value!r} was answered from another context"


def test_an_unobserved_matryoshka_cut_is_refused_without_the_engine_gate() -> None:
    """A ``dimensions`` the engine's config cannot serve is the engine's 400, not a surrogate: this recipe
    declares no ``is_matryoshka``/``matryoshka_dimensions`` in ``serve.hf_overrides`` (no MRL head at all),
    so vLLM refuses any cut (``pooling_params.py`` gate 1) -- the emulator mirrors the refusal (the
    declared-set cases live in ``tests/conformance/test_mrl.py``)."""
    body = _recorded("octen-embedding-8b", 1)
    body["dimensions"] = 64
    answer = emulator_for("octen-embedding-8b").answer("/v1/embeddings", "POST", body)
    assert answer.status_code == 400
    assert "does not support Matryoshka embeddings" in answer.json()["error"]["message"]


def test_the_add_special_tokens_field_changes_the_counted_prompt() -> None:
    body = _recorded("qwen3-embedding-0.6b", 1)
    emulator = emulator_for("qwen3-embedding-0.6b")
    with_specials = emulator.answer("/v1/embeddings", "POST", body).json()["usage"]["prompt_tokens"]
    without = emulator.answer("/v1/embeddings", "POST", {**body, "add_special_tokens": False}).json()
    assert without["usage"]["prompt_tokens"] < with_specials


@pytest.mark.parametrize(
    ("recipe_id", "field", "value"),
    [
        ("zerank-2-reranker", "instruction", "Judge the passage."),  # folded into the template by the engine
        ("zerank-2-reranker", "truncate_prompt_tokens", 16),  # the engine cuts instead of refusing
        ("zerank-2-reranker", "max_tokens_per_doc", 8),
        ("qwen3-embedding-0.6b", "truncate_prompt_tokens", 16),
        ("qwen3-embedding-0.6b", "mm_processor_kwargs", {"max_pixels": 1024}),
    ],
)
def test_a_field_the_emulator_does_not_model_is_refused_marked(recipe_id: str, field: str, value: object) -> None:
    body = _recorded(recipe_id, 1)
    body[field] = value
    path = "/v1/embeddings" if "embedding" in recipe_id else "/rerank"
    answer = emulator_for(recipe_id).answer(path, "POST", body)
    assert answer.status_code == 400
    assert answer.headers[SOURCE] == "refused-unmodelled"
    assert field in answer.json()["error"]["message"]


def test_an_unknown_field_is_ignored_as_the_engine_ignores_it() -> None:
    """The recorded ``unknown_field`` exchanges answer 200 with the outputs of the same prompt."""
    body = _recorded("octen-embedding-8b", 1)
    answer = emulator_for("octen-embedding-8b").answer("/v1/embeddings", "POST", {**body, "not_a_field": 1})
    assert answer.status_code == 200 and answer.headers[SOURCE] == "replayed"


def test_a_corpus_whose_key_holds_different_outputs_is_refused(tmp_path: Path) -> None:
    """Two observations under one key that disagree mean the key misses a behaviour-shaping field (or the
    engine is non-deterministic beyond the measured tolerance): never pick one silently."""
    from rcp_ndcg_test.engines import EngineFacts, Exchange, StringsPrompts, VllmEmulator
    from tests._engines import observation_corpus

    from tests._tokenizers import word_tokenizer

    def exchange(sequence: int, body: dict, value: float) -> Exchange:
        reply = {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [value, 0.0]}]}
        return Exchange(sequence, "POST", "/v1/embeddings", body, 200, {}, reply)

    body = {"model": "tiny", "input": ["the a"]}
    corpus = observation_corpus(tmp_path, [exchange(0, body, 1.0), exchange(1, {**body, "ignored": 1}, 0.5)])
    facts = EngineFacts("vllm", "0.31.0", "tiny", "fixtures/Tiny", 8)
    with pytest.raises(DataError) as error:
        VllmEmulator.from_corpus(corpus, StringsPrompts(), word_tokenizer(), facts, dim=2)
    assert "#0" in str(error.value) and "#1" in str(error.value)


def test_every_declared_request_field_is_classified() -> None:
    """A field a route declares is either a prompt, protocol, an output (keyed) or refused as
    unmodelled -- never silently dropped from the key."""
    from rcp_ndcg_test.engines import FIELD_CLASSES, ROUTE_FIELDS

    for route, fields in ROUTE_FIELDS.items():
        assert fields <= set(FIELD_CLASSES), (route, sorted(fields - set(FIELD_CLASSES)))
    assert set(FIELD_CLASSES.values()) == {"prompt", "protocol", "output", "unmodelled"}
