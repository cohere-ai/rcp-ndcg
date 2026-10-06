"""The recipe schema: the client block IS the product's endpoint config, validated at load.

Product rules are the product's tests (refusals for template shape rules, budgets, empty_doc pairing and the
rest happen in the product's own suite); here only the recipe-level rules and the round-trip through the
product's config loader are tested.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rcp_ndcg_vllm import (
    RecipeError,
    client_config,
    iter_recipes,
    load_recipe,
    recipe_json_schema,
)

from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint

_SCHEMA = Path(__file__).resolve().parents[1] / "schema" / "recipe.schema.json"
REV = "0123456789abcdef0123456789abcdef01234567"


def recipe_dirs_path() -> Path:
    from tests.fixtures import FIXTURES

    return FIXTURES / "recipes"


def test_every_fixture_recipe_loads_against_the_product_endpoints() -> None:
    """Every fixture recipe loads, with the client block constructing the product's endpoint model."""
    recipes = iter_recipes(recipe_dirs_path())
    types = {recipe.id: type(recipe.client).__name__ for recipe in recipes}
    assert types == {
        "fixture-embed": "EmbeddingEndpoint",
        "fixture-embed-cls": "EmbeddingEndpoint",
        "fixture-embed-edge": "EmbeddingEndpoint",
        "fixture-embed-marker": "EmbeddingEndpoint",
        "fixture-multi-vector": "PoolingEndpoint",
        "fixture-rerank-pointwise": "RerankEndpoint",
        "fixture-rerank-noisy": "RerankEndpoint",
        "fixture-rerank-listwise": "RerankEndpoint",
    }


def test_client_config_round_trips_through_the_product_loader() -> None:
    """client_config() is the product's config: every fixture's dump constructs the product model unchanged."""
    classes = {"embed": EmbeddingEndpoint, "multi_vector": PoolingEndpoint, "rerank": RerankEndpoint}
    for recipe in iter_recipes(recipe_dirs_path()):
        config = client_config(recipe, base_url="http://127.0.0.1:8100/v1")
        assert config["model"] == recipe.id
        assert config["recipe"] == recipe.id
        endpoint = classes[recipe.role](**config)
        assert str(endpoint.base_url) == "http://127.0.0.1:8100/v1"
        # and through the product's loader (the endpoint config's own model_validate)
        again = type(endpoint).model_validate(config)
        assert again.model == recipe.id


def test_client_model_and_revision_are_injected(tmp_path: Path) -> None:
    recipe = load_recipe(recipe_dirs_path() / "fixture-embed")
    assert recipe.client.model == "fixture-embed"
    assert recipe.client.revision == recipe.revision


def test_client_block_cannot_declare_the_injected_fields(tmp_path: Path) -> None:
    """client.model and client.revision are the recipe's own ids: declared in the YAML, they are refused."""
    import shutil

    import yaml

    copied = tmp_path / "fixture-embed"
    copied.mkdir()
    for name in ("recipe.yaml", "reference.py"):
        shutil.copy(recipe_dirs_path() / "fixture-embed" / name, copied / name)
    data = yaml.safe_load((copied / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["model"] = "other-name"
    (copied / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(RecipeError, match="drop the field"):
        load_recipe(copied)


def test_recipe_rules_refuse_the_budget_over_the_engine_context(tmp_path: Path) -> None:
    """client.max_tokens must fit the engine's max_model_len: the engine would 400 the rendered prompt."""
    import shutil

    import yaml

    copied = tmp_path / "fixture-embed"
    copied.mkdir()
    for name in ("recipe.yaml", "reference.py"):
        shutil.copy(recipe_dirs_path() / "fixture-embed" / name, copied / name)
    data = yaml.safe_load((copied / "recipe.yaml").read_text(encoding="utf-8"))
    data["client"]["max_tokens"] = 4096
    data["serve"]["max_model_len"] = 512
    (copied / "recipe.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(RecipeError, match="max_model_len"):
        load_recipe(copied)


def test_json_schema_export_is_current() -> None:
    assert json.loads(_SCHEMA.read_text(encoding="utf-8")) == recipe_json_schema()


def test_serve_argv_rerank_pointwise_is_golden() -> None:
    recipe = load_recipe(recipe_dirs_path() / "fixture-rerank-pointwise")
    from rcp_ndcg_vllm import serve_argv

    argv = serve_argv(recipe, port=8100, served_model_name="fixture-rerank-pointwise")
    assert argv[:3] == ["vllm", "serve", "fixtures/PointwiseReranker"]
    assert argv[argv.index("--revision") + 1] == REV
    assert argv[argv.index("--pooler-config") + 1] == '{"use_activation": true}'
    assert "--chat-template" in argv


def test_serve_argv_needs_the_recipe_directory_for_a_template() -> None:
    from rcp_ndcg_vllm import serve_argv

    data = load_recipe(recipe_dirs_path() / "fixture-rerank-pointwise")
    bare = data.model_copy()
    bare.__dict__["_dir"] = None
    with pytest.raises(RecipeError, match="load_recipe"):
        serve_argv(bare, port=1, served_model_name="x")
