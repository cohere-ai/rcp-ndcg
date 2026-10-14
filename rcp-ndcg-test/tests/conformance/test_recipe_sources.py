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
    """The provenance the manifests state: a recorded role request is byte-identical to what the current
    recipe renders for the input it names, so a recipe edit since the recording changed no recorded
    prompt. A release recording names its plan row (``pairs:N``): the row comes from the committed pairs
    file. A shakedown-era corpus records the old recorder's fixed snippet instead. (Rerank pair prompts
    are rendered by the emulator itself and checked by every conformance replay.)"""
    import re

    from rcp_ndcg_test.corpus import load_corpus
    from rcp_ndcg_test.fingerprint import load_recipe_tokenizer
    from tests._engines import current_corpora, load_recipe

    from rcp_ndcg.data.templates import TemplateSpec

    checked = 0
    skipped: list[str] = []
    for directory in current_corpora():  # a declared-stale corpus fails the staleness gate instead
        corpus = load_corpus(directory)
        recipe = load_recipe(corpus.manifest["recipe"]["id"])
        if recipe.role != "embed":
            continue
        records = [
            record
            for record in corpus.records
            if record["request"]["path"].endswith("/embeddings")
            and record["response"]["status"] == 200
            and "input" in (record["request"]["body_parsed"] or {})
        ]
        record = next(
            (
                record
                for record in records
                if re.fullmatch(r"pairs:\d+", str((record.get("inputs") or {}).get("request_id") or ""))
            ),
            None,
        )
        if record is None and "plan" in corpus.manifest:
            # A release plan's VL rows ride the chat route (``messages``), not a text ``input``: no
            # text-wire plan row is recorded, and the conformance replay checks those chat prompts.
            skipped.append(recipe.id)
            continue
        record = record or records[0]
        inputs = record["inputs"]
        template = TemplateSpec.model_validate(recipe.client.get("template"))
        row_match = re.fullmatch(r"pairs:(\d+)", str(inputs.get("request_id") or ""))
        if row_match:
            row = json.loads(
                (ENGINES.parent / "pairs" / f"{recipe.id}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[int(row_match.group(1))]
            )
            query = str(row["query"])
            if str(inputs.get("side") or "query") == "query":
                shape, text = "query", query
            else:
                shape = "document"
                text = str(row["documents"][int(str(inputs["item"]).split(":")[1])])
        else:  # the shakedown plan's fixed snippet (the old recorder's request set)
            shape = "query" if "query" in template.shapes() else "document"
            query = text = "What is the capital of France?" if shape == "query" else "Paris is the capital of France."
        rendered = template.render(shape, load_recipe_tokenizer(recipe), query=query, document=text)
        assert record["request"]["body_parsed"]["input"] == [rendered], (
            f"{recipe.id}: the recorded prompt no longer renders"
        )
        checked += 1
    assert checked >= 1
    assert skipped in ([], ["qwen3-vl-embedding-2b"]), f"unexpected skip: {skipped}"


def test_every_corpus_manifest_declares_its_provenance() -> None:
    """Every manifest states what its recording is: a shakedown corpus is valid to build and test the
    emulators, never release evidence, and says so in its ``provisional`` block (and why); a release
    recording (the eight re-recorded before the tag) carries the request plan instead and makes no
    shakedown-only claim. A manifest that declares neither is a corpus with no stated provenance and
    fails here; both kinds must be present, so neither branch passes vacuously."""
    provisional = release = 0
    for manifest in sorted(ENGINES.glob("*/*/*/manifest.json")):
        document = json.loads(manifest.read_text(encoding="utf-8"))
        statement = document.get("provisional")
        if statement is not None:
            provisional += 1
            assert statement.get("status") == "provisional", manifest
            assert "building and testing" in statement.get("valid_for", ""), manifest
            assert statement.get("not_valid_for", "").startswith("release evidence"), manifest
            assert "shakedown-only patches" in statement.get("why", ""), manifest
        else:
            release += 1
            assert (document.get("plan") or {}).get("request_ids"), (
                f"{manifest}: a release recording carries its request plan"
            )
    assert provisional and release, f"expected shakedown and release recordings, found {provisional} and {release}"
