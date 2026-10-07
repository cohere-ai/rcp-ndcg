"""Every recipe the compact corpus names must load through the product's own validation.

The verified fake engines replay one emulator per (engine, version, recipe, behaviour fingerprint), and
the staleness check recomputes the fingerprint from the repository -- both need the recipe to load. A
recipe that does not load is a corpus without a key.
"""

from __future__ import annotations

import json
from pathlib import Path

ENGINES = Path(__file__).resolve().parents[1] / "contract" / "engines"
RECIPES = Path(__file__).resolve().parents[2] / "packages" / "rcp-ndcg-vllm" / "recipes"

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
    from rcp_ndcg_vllm.recipe import load_recipe

    directories = {path.name: path for path in RECIPES.iterdir() if (path / "recipe.yaml").is_file()}
    failures = {}
    for recipe_id in sorted(SHAKE1C):
        try:
            load_recipe(directories[recipe_id])
        except Exception as error:  # noqa: BLE001 - one failure line per broken recipe in the message
            failures[recipe_id] = str(error).splitlines()[0]
    assert not failures, f"recipes the corpus names do not load: {json.dumps(failures, indent=2)}"


def test_every_corpus_manifest_names_a_loaded_recipe() -> None:
    """Each committed corpus names a recipe of the recipe directory (its id is the directory name)."""
    for manifest in sorted(ENGINES.glob("*/*/*/manifest.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        recipe_id = data["recipe"]["id"]
        assert (RECIPES / recipe_id / "recipe.yaml").is_file(), f"{manifest} names unknown recipe {recipe_id!r}"
