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
from typing import Any

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


#: Candidate values per OUTPUT field (the only fields that enter the replay key beside the prompts, see
#: ``FIELD_CLASSES``): the test picks the first value no exchange of the corpus carries, so a recording
#: that happens to cover a value moves the test to the next candidate instead of breaking its premise.
_CANDIDATE_VALUES: dict[str, tuple[object, ...]] = {
    "use_activation": (False, True),
    "add_special_tokens": (False, True),
    "dimensions": (64, 128, 256, 512),
}

_SENTINEL = "rcp-ndcg-conformance-unobserved-prompt"


def _output_candidates(route: str) -> tuple[tuple[str, tuple[object, ...]], ...]:
    """The route's own output fields, in a stable order: a field the route does not declare is ignored by
    the engine, so probing it would replay the same context rather than answer the surrogate."""
    from rcp_ndcg_test.engines import FIELD_CLASSES, ROUTE_FIELDS

    declared = ROUTE_FIELDS[route]
    return tuple(
        (field, _CANDIDATE_VALUES[field])
        for field in sorted(_CANDIDATE_VALUES)
        if field in declared and FIELD_CLASSES.get(field) == "output"
    )


def _unobserved_context(recipe_id: str, body: dict[str, Any], route: str) -> tuple[str, object]:
    """A field/value pair no exchange of this corpus carries -- or, when the corpus observes every value
    of every output field the route declares (the rerank route has one, ``use_activation``, and a release
    recording carries both), a prompt the corpus never recorded: the key's prompt half is the other half
    of the context, so an unobserved prompt is an unobserved context by construction."""
    from rcp_ndcg_test.engines import exchanges_of

    candidates = _output_candidates(route)
    corpus = corpus_of(load_recipe(recipe_id))
    observed: dict[str, set[Any]] = {field: set() for field, _ in candidates}
    for exchange in exchanges_of(corpus):
        other = dict(exchange.request_body or {})
        for field, _ in candidates:
            observed[field].add(other.get(field))
    for field, values in candidates:
        for value in values:
            if value not in observed[field]:
                return field, value
    if route == "rerank":
        return "query", f"{body['query']} {_SENTINEL}"
    item = body["input"]
    if isinstance(item, list):
        return "input", [*item, _SENTINEL]
    return "input", f"{item} {_SENTINEL}"


@pytest.mark.parametrize(
    ("recipe_id", "sequence"),
    [
        ("qwen3-embedding-0.6b", 1),
        ("zerank-2-reranker", 1),
        ("zerank-1-small-reranker", 1),
    ],
)
def test_an_unobserved_context_answers_the_marked_surrogate(recipe_id: str, sequence: int) -> None:
    body = _recorded(recipe_id, sequence)
    path = "/v1/embeddings" if "embedding" in recipe_id else "/rerank"
    route = "embeddings" if "embedding" in recipe_id else "rerank"
    field, value = _unobserved_context(recipe_id, body, route)
    assert body.get(field) != value
    body[field] = value
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
        ("zerank-2-reranker", "chat_template_kwargs", {"instruction": "Judge the passage."}),  # template kwargs
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


def test_a_top_n_truncated_reply_is_the_ranked_prefix_of_the_full_observation(tmp_path: Path) -> None:
    """``top_n`` frames the same model output shorter (protocol, never a key field): the model layer
    stores the fullest observation for the key and compares a truncated reply as its first ranked
    entries, within the measured tolerance. A pointwise truncation leaves an unreturned position
    ABSENT, never a ``None`` score that would read as an observation."""
    from rcp_ndcg_test.engines import EngineFacts, EnginePrompts, Exchange, PairPrompts, VllmEmulator
    from tests._engines import observation_corpus

    from tests._tokenizers import word_tokenizer

    tokenizer = word_tokenizer()
    facts = EngineFacts("vllm", "0.31.0", "tiny", "fixtures/Tiny", 128)
    body = {"model": "tiny", "query": "q", "documents": ["a", "b", "c"]}
    full = {
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.4},
            {"index": 1, "relevance_score": 0.1},
        ]
    }
    top_one = {"results": [{"index": 0, "relevance_score": 0.9}]}

    class _Template:
        @staticmethod
        def render(shape: str, tokenizer: Any, *, query: str, document: str) -> str:
            return f"pair {query} <> {document}"

        @staticmethod
        def adds_special_tokens(shape: str) -> bool:
            return False

    def exchanges() -> list[Exchange]:
        return [
            Exchange(0, "POST", "/rerank", body, 200, {}, full),
            Exchange(1, "POST", "/rerank", {**body, "top_n": 1}, 200, {}, top_one),
        ]

    # listwise: one set-level observation per request; the replay keeps the full ranking
    listwise = VllmEmulator.from_corpus(
        observation_corpus(tmp_path / "listwise", exchanges(), tolerance=(1e-6, 1e-6)),
        EnginePrompts(builder=lambda query, documents: f"{query}|{'|'.join(documents)}"),
        tokenizer,
        facts,
        dim=2,
    )
    assert [entry["index"] for entry in listwise.answer("/rerank", "POST", body).json()["results"]] == [0, 2, 1]
    assert [entry["index"] for entry in listwise.answer("/rerank", "POST", {**body, "top_n": 1}).json()["results"]] == [
        0
    ]

    # pointwise: the truncated record observes only the position it returned
    pointwise = VllmEmulator.from_corpus(
        observation_corpus(tmp_path / "pointwise", exchanges(), tolerance=(1e-6, 1e-6)),
        PairPrompts(template=_Template(), tokenizer=tokenizer),
        tokenizer,
        facts,
        dim=2,
    )
    observed = [observation for history in pointwise.observations.values() for observation in history]
    assert observed and all(observation.score is not None for observation in observed), (
        "an unreturned position is absent, never a None score"
    )
    assert pointwise.answer("/rerank", "POST", body).json()["results"]


def test_a_top_n_truncated_reply_that_is_not_the_stored_prefix_is_refused(tmp_path: Path) -> None:
    """A truncated reply is accepted only as the stored ranking's own first entries: a top_n reply that
    returned another document (or a score beyond the measured tolerance) is still the inconsistency the
    build refuses."""
    from rcp_ndcg_test.engines import EngineFacts, EnginePrompts, Exchange, VllmEmulator
    from tests._engines import observation_corpus

    from tests._tokenizers import word_tokenizer

    body = {"model": "tiny", "query": "q", "documents": ["a", "b", "c"]}
    full = {"results": [{"index": 0, "relevance_score": 0.9}, {"index": 2, "relevance_score": 0.4}]}
    moved = {"results": [{"index": 2, "relevance_score": 0.4}]}
    corpus = observation_corpus(
        tmp_path / "corpus",
        [
            Exchange(0, "POST", "/rerank", body, 200, {}, full),
            Exchange(1, "POST", "/rerank", {**body, "top_n": 1}, 200, {}, moved),
        ],
        tolerance=(1e-6, 1e-6),
    )
    with pytest.raises(DataError) as error:
        VllmEmulator.from_corpus(
            corpus,
            EnginePrompts(builder=lambda query, documents: f"{query}|{'|'.join(documents)}"),
            word_tokenizer(),
            EngineFacts("vllm", "0.31.0", "tiny", "fixtures/Tiny", 128),
            dim=2,
        )
    assert "#0" in str(error.value) and "#1" in str(error.value)


@pytest.mark.parametrize(
    "recipe_id",
    ["jina-reranker-v3", "qwen3-reranker-0.6b", "qwen3-reranker-4b", "qwen3-reranker-8b", "qwen3-vl-reranker-2b"],
)
def test_an_instruction_bearing_request_replays_the_engines_own_render(recipe_id: str) -> None:
    """The engine folds a request ``instruction`` into the rendered prompt where its template reads one
    (the listwise reference builder, the served template's default override) and ignores it where it
    reads none (the qwen3-reranker pair template): either way the recorded instruction-bearing exchange
    replays, and the replayed usage is the engine's own count of that render."""
    from rcp_ndcg_test.engines import compare_exchange, corpus_tolerance, exchanges_of
    from tests._engines import corpus_of, emulator_for

    corpus = corpus_of(load_recipe(recipe_id))
    emulator = emulator_for(recipe_id)
    exchange = next(
        item
        for item in exchanges_of(corpus)
        if isinstance(item.request_body, dict)
        and item.request_body.get("instruction")
        and item.path.endswith("/rerank")
        and item.status == 200
    )
    answer = emulator.answer(exchange.path, exchange.method, exchange.request_body)
    assert compare_exchange(exchange, answer, corpus_tolerance(corpus)) == []
    recorded = (exchange.response_json or {}).get("usage", {}).get("prompt_tokens")
    assert answer.json()["usage"]["prompt_tokens"] == recorded, f"{recipe_id}: the instruction render moved"


def test_every_declared_request_field_is_classified() -> None:
    """A field a route declares is either a prompt, protocol, an output (keyed) or refused as
    unmodelled -- never silently dropped from the key."""
    from rcp_ndcg_test.engines import FIELD_CLASSES, ROUTE_FIELDS

    for route, fields in ROUTE_FIELDS.items():
        assert fields <= set(FIELD_CLASSES), (route, sorted(fields - set(FIELD_CLASSES)))
    assert set(FIELD_CLASSES.values()) == {"prompt", "protocol", "output", "unmodelled"}
