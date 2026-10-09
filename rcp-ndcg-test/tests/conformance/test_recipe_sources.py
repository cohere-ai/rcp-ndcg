"""Every recipe the compact corpus names must load through the product's own validation.

The verified fake engines replay one emulator per (engine, version, recipe, behaviour fingerprint), and
the staleness check recomputes the fingerprint from the repository -- both need the recipe to load. A
recipe that does not load is a corpus without a key.
"""

from __future__ import annotations

import json
from pathlib import Path

from rcp_ndcg_vllm.recipe import default_recipes_root

ENGINES = Path(__file__).resolve().parents[2] / "corpora"
RECIPES = default_recipes_root()

#: The 12 recipes the shake1c waves recorded (FINDINGS.md): the provisional corpus's coverage.
SHAKE1C = (
    "jina-reranker-v3",
    "qwen3-embedding-0.6b",
    "qwen3-reranker-0.6b",
    "qwen3-reranker-4b",
    "qwen3-reranker-8b",
    "qwen3-vl-embedding-2b",
    "qwen3-vl-reranker-2b",
    "zembed-1-embedding",
    "jina-embeddings-v5-text-small",
    "octen-embedding-8b",
    "zerank-1-small-reranker",
    "zerank-2-reranker",
)


def test_every_corpus_recipe_loads() -> None:
    from tests._engines import harness

    harness()

    from rcp_ndcg_vllm.recipe import iter_recipes

    harness()
    known = {recipe.id: recipe for recipe in iter_recipes(RECIPES)}
    failures = {}
    for recipe_id in sorted(SHAKE1C):
        try:
            known[recipe_id].client.get("tokenizer")  # resolved and validated at iter_recipes
        except Exception as error:  # noqa: BLE001 - one failure line per broken recipe in the message
            failures[recipe_id] = str(error).splitlines()[0]
    missing = sorted(set(SHAKE1C) - set(known))
    assert not failures and not missing, (
        f"recipes the corpus names do not load: {json.dumps({'failures': failures, 'missing': missing}, indent=2)}"
    )


def test_every_corpus_manifest_names_a_loaded_recipe() -> None:
    """Each committed corpus names a shipped recipe (its id is a family's variant id, decision 34)."""
    from rcp_ndcg_vllm.recipe import iter_recipes

    known = {recipe.id for recipe in iter_recipes(RECIPES)}
    for manifest in sorted(ENGINES.glob("*/*/*/manifest.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        recipe_id = data["recipe"]["id"]
        assert recipe_id in known, f"{manifest} names unknown recipe {recipe_id!r}"


def test_every_recorded_embedding_request_renders_from_the_current_recipe() -> None:
    """The provenance the manifests state: the recorded role request (the recorder's snippet through the
    product's role client) is byte-identical to what the current recipe renders for that snippet, so a
    recipe edit since the shakedown changed no recorded prompt. (Rerank pair prompts are rendered by
    the emulator itself and checked by every conformance replay.)"""
    from rcp_ndcg_test.corpus import load_corpus
    from rcp_ndcg_test.engines import exchanges_of
    from rcp_ndcg_test.fingerprint import load_recipe_tokenizer
    from tests._engines import current_corpora, load_recipe

    from rcp_ndcg.data.templates import TemplateSpec

    checked = expected = 0
    for directory in current_corpora():  # a declared-stale corpus fails the staleness gate instead
        corpus = load_corpus(directory)
        recipe = load_recipe(corpus.manifest["recipe"]["id"])
        if recipe.role != "embed":
            continue
        expected += 1
        role_request = next(
            exchange.request_body
            for exchange in exchanges_of(corpus)
            if exchange.path.endswith("/embeddings")
            and exchange.status == 200
            and "unknown_field" not in exchange.request_body
        )
        template = TemplateSpec.model_validate(recipe.client.get("template"))
        shape = "query" if "query" in template.shapes() else "document"
        text = "What is the capital of France?" if shape == "query" else "Paris is the capital of France."
        rendered = template.render(shape, load_recipe_tokenizer(recipe), query=text, document=text)
        assert role_request["input"] == [rendered], f"{recipe.id}: the recorded prompt no longer renders"
        checked += 1
    assert checked == expected >= 1


def test_every_corpus_manifest_states_it_is_provisional() -> None:
    """The shakedown corpus is valid to build and test the emulators, never release evidence: every
    manifest says so, and why."""
    for manifest in sorted(ENGINES.glob("*/*/*/manifest.json")):
        statement = json.loads(manifest.read_text(encoding="utf-8")).get("provisional") or {}
        assert statement.get("status") == "provisional", manifest
        assert "building and testing" in statement.get("valid_for", ""), manifest
        assert statement.get("not_valid_for", "").startswith("release evidence"), manifest
        assert "shakedown-only patches" in statement.get("why", ""), manifest
