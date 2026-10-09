"""The recipe schema: the client block is plain data that the product validates when it reads it.

Product rules are the product's tests (refusals for template shape rules, budgets, empty_doc pairing and the
rest happen in the product's own suite); here only the recipe-level rules and the round-trip through the
product's config loader are tested.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rcp_ndcg_vllm import RecipeError, iter_recipes, load_family, load_recipe
from rcp_ndcg_vllm.recipe import client_config, default_recipes_root, recipe_json_schema

from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint

_SCHEMA = Path(__file__).resolve().parents[2] / "rcp-ndcg-vllm" / "schema" / "recipe.schema.json"
REV = "0123456789abcdef0123456789abcdef01234567"

_ENDPOINTS = {"embed": EmbeddingEndpoint, "multi_vector": PoolingEndpoint, "rerank": RerankEndpoint}

_SCHEMA = Path(__file__).resolve().parents[2] / "rcp-ndcg-vllm" / "schema" / "recipe.schema.json"
REV = "0123456789abcdef0123456789abcdef01234567"


def recipe_dirs_path() -> Path:
    from tests.fixtures import FIXTURES

    return FIXTURES / "recipes"


def test_every_fixture_recipe_loads_against_the_product_endpoints() -> None:
    """Every fixture recipe loads, with the client block constructing the product's endpoint model."""
    recipes = iter_recipes(recipe_dirs_path())
    assert {recipe.id for recipe in recipes} == {
        "fixture-embed",
        "fixture-embed-cls",
        "fixture-embed-edge",
        "fixture-embed-marker",
        "fixture-multi-vector",
        "fixture-rerank-pointwise",
        "fixture-rerank-noisy",
        "fixture-rerank-listwise",
        "fixture-vl-embed",
        "fixture-vl-video",
        # the fakes' fixture recipes of this package (same schema, fake:// engines)
        "fake-pool",
        "fake-rerank",
    }
    # and the product's endpoint model accepts every plain client block (it validates when it reads it)
    for recipe in recipes:
        _ENDPOINTS[recipe.role].model_validate(client_config(recipe, base_url=None))


def test_client_config_round_trips_through_the_product_loader() -> None:
    """client_config() is the product's config: every fixture's block constructs the product model unchanged."""
    for recipe in iter_recipes(recipe_dirs_path()):
        config = client_config(recipe, base_url="http://127.0.0.1:8100/v1")
        assert config["model"] == recipe.id
        assert config["recipe"] == recipe.id
        endpoint = _ENDPOINTS[recipe.role](**config)
        assert str(endpoint.base_url) == "http://127.0.0.1:8100/v1"
        # and through the product's loader (the endpoint config's own model_validate)
        again = type(endpoint).model_validate(config)
        assert again.model == recipe.id


def test_client_model_and_revision_are_injected(tmp_path: Path) -> None:
    recipe = load_recipe(recipe_dirs_path() / "fixture-embed")
    assert recipe.client.get("model") == "fixture-embed"
    assert recipe.client.get("revision") == recipe.revision


def test_client_block_cannot_declare_the_injected_fields(tmp_path: Path) -> None:
    """client.model and client.revision are the recipe's own ids: declared in the YAML, they are refused."""
    import shutil

    import yaml

    copied = tmp_path / "fixture-embed"
    copied.mkdir()
    for name in ("family.yaml", "reference.py"):
        shutil.copy(recipe_dirs_path() / "fixture-embed" / name, copied / name)
    data = yaml.safe_load((copied / "family.yaml").read_text(encoding="utf-8"))
    data["client"]["model"] = "other-name"
    (copied / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(RecipeError, match="drop the field"):
        load_recipe(copied)


def test_the_family_client_block_cannot_declare_the_variant_tokenizer(tmp_path: Path) -> None:
    """client.tokenizer is the variant's own (model@revision) or the family's shared one; a variant-level
    override of it is refused (the tokenizer is never a per-size fact)."""
    import shutil

    import yaml

    copied = tmp_path / "fixture-embed"
    shutil.copytree(recipe_dirs_path() / "fixture-embed", copied)
    family = yaml.safe_load((copied / "family.yaml").read_text(encoding="utf-8"))
    family["variants"][0]["overrides"] = {
        "client": {"tokenizer": "other/tokenizer@0123456789abcdef0123456789abcdef01234567"}
    }
    (copied / "family.yaml").write_text(yaml.safe_dump(family, sort_keys=False), encoding="utf-8")
    with pytest.raises(RecipeError, match="not declared per-size fields"):
        load_recipe(copied)


def test_a_variant_id_not_unique_within_the_family_is_refused(tmp_path: Path) -> None:
    """Two variants of one family with the same id would resolve to one id twice: refused."""
    import shutil

    import yaml

    copied = tmp_path / "fixture-embed"
    shutil.copytree(recipe_dirs_path() / "fixture-embed", copied)
    family = yaml.safe_load((copied / "family.yaml").read_text(encoding="utf-8"))
    second = {**family["variants"][0], "model": "fixtures/OtherEmbedder"}
    family["variants"].append(second)
    (copied / "family.yaml").write_text(yaml.safe_dump(family, sort_keys=False), encoding="utf-8")
    with pytest.raises(RecipeError, match="unique"):
        load_family(copied)


def test_recipe_rules_refuse_the_budget_over_the_engine_context(tmp_path: Path) -> None:
    """client.max_tokens must fit the engine's max_model_len: the engine would 400 the rendered prompt."""
    import shutil

    import yaml

    copied = tmp_path / "fixture-embed"
    copied.mkdir()
    for name in ("family.yaml", "reference.py"):
        shutil.copy(recipe_dirs_path() / "fixture-embed" / name, copied / name)
    data = yaml.safe_load((copied / "family.yaml").read_text(encoding="utf-8"))
    data["client"]["max_tokens"] = 4096
    data["serve"]["max_model_len"] = 512
    (copied / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(RecipeError, match="max_model_len"):
        load_recipe(copied)


def test_json_schema_export_is_current() -> None:
    assert json.loads(_SCHEMA.read_text(encoding="utf-8")) == recipe_json_schema()


def test_an_engine_spec_accepts_a_dev_version_floor() -> None:
    """A digest-pinned nightly carries a setuptools-scm dev version (``0.31.1.devN``); the floor a recipe
    declares for it is the dev series, and the digest in ``image`` is the real pin (decision 38)."""
    from rcp_ndcg_vllm.recipe import EngineSpec

    digest = "sha256:" + "0" * 64
    spec = EngineSpec(
        name="vllm",
        image=f"vllm/vllm-openai:nightly-8cbd5d03006c33185f402249ff2b448efd594986@{digest}",
        min_version="0.31.1.dev0",
    )
    assert spec.min_version == "0.31.1.dev0"
    with pytest.raises(ValueError):
        EngineSpec(name="vllm", image="x", min_version="0.31.1.dev")
    with pytest.raises(ValueError):
        EngineSpec(name="vllm", image="x", min_version="nightly")


def test_family_json_schema_export_is_current() -> None:
    """The family file format has its own exported schema (decision 34; the spec requires it)."""
    from rcp_ndcg_vllm.recipe import family_json_schema

    family_schema = _SCHEMA.parent / "family.schema.json"
    assert json.loads(family_schema.read_text(encoding="utf-8")) == family_json_schema()


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


def _media_recipe(tmp_path: Path, *, client_policy: dict, serve_kwargs: dict) -> Path:
    """``fixture-embed`` copied with an image modality: the client's image policy and the engine's
    ``serve.mm_processor_kwargs`` as given (the Qwen3-VL processor family)."""
    import shutil

    import yaml

    copied = tmp_path / "fixture-embed"
    copied.mkdir()
    for name in ("family.yaml", "reference.py"):
        shutil.copy(recipe_dirs_path() / "fixture-embed" / name, copied / name)
    data = yaml.safe_load((copied / "family.yaml").read_text(encoding="utf-8"))
    data["input"] = ["text", "image"]
    data["client"].update(max_images=1, image_processor="qwen3_vl", image_policy=client_policy or None)
    data["serve"]["mm_processor_kwargs"] = serve_kwargs
    (copied / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return copied


#: The Qwen3-VL-Embedding card's pixel budget, below the qwen3_vl processor's stock floor (65536 px).
_CARD = {"min_px": 4096, "max_px": 1843200}
_CARD_PIN = {"images_kwargs": {"min_pixels": 4096, "max_pixels": 1843200}}


def test_a_pinned_pixel_budget_below_the_stock_floor_loads_when_serve_pins_the_same_numbers(tmp_path: Path) -> None:
    """H4: the client declares the budget pinned, serve pins the engine to the same numbers (the nested
    ``images_kwargs`` shape): the recipe loads, and the client counts images under that budget."""
    recipe = load_recipe(
        _media_recipe(tmp_path, client_policy={**_CARD, "engine_pixel_pinning": True}, serve_kwargs=_CARD_PIN)
    )
    policy = recipe.client.get("image_policy")
    assert policy is not None and policy["engine_pixel_pinning"]


@pytest.mark.parametrize(
    ("client_policy", "serve_kwargs", "match"),
    [
        # Pinned on the client, not on serve: the stock engine would scale the prepared image up again.
        ({**_CARD, "engine_pixel_pinning": True}, {}, "engine_pixel_pinning"),
        # Pinned on both sides, to different numbers.
        (
            {**_CARD, "engine_pixel_pinning": True},
            {"images_kwargs": {"min_pixels": 4096, "max_pixels": 1310720}},
            "max_pixels",
        ),
        # A flat key the engine applies to images too, disagreeing with the client.
        ({**_CARD, "engine_pixel_pinning": True}, {**_CARD_PIN, "min_pixels": 65536}, "min_pixels"),
        # Serve pins a budget, the client (unpinned, inside the stock range) counts under another one.
        ({"min_px": 65536, "max_px": 1003520}, _CARD_PIN, "image_policy"),
        # Serve pins a budget and the client declares none to count under.
        ({}, _CARD_PIN, "image_policy"),
        # The HF processor's own size keys pin the budget too.
        ({**_CARD, "engine_pixel_pinning": True}, {**_CARD_PIN, "size": {"shortest_edge": 65536}}, "shortest_edge"),
        (
            {**_CARD, "engine_pixel_pinning": True},
            {"images_kwargs": {**_CARD_PIN["images_kwargs"], "size": {"longest_edge": 16777216}}},
            "longest_edge",
        ),
    ],
)
def test_the_client_and_the_engine_pixel_budgets_must_agree(
    tmp_path: Path, client_policy: dict, serve_kwargs: dict, match: str
) -> None:
    with pytest.raises(RecipeError, match=match):
        load_recipe(_media_recipe(tmp_path, client_policy=client_policy, serve_kwargs=serve_kwargs))


def test_a_duplicate_yaml_key_is_refused(tmp_path: Path) -> None:
    """YAML keeps the last of two equal keys silently: a recipe declaring a field twice would serve whichever
    came last. The loader refuses it with a typed error naming the key and its line, and no shipped or fixture
    recipe declares a key twice."""
    import shutil

    from rcp_ndcg_vllm import RecipeError

    fixtures = recipe_dirs_path()
    directory = tmp_path / "fixture-embed"
    shutil.copytree(fixtures / "fixture-embed", directory)
    path = directory / "family.yaml"
    tokenizer = str((fixtures / ".." / "tokenizer.json").resolve())
    text = path.read_text(encoding="utf-8").replace("../../tokenizer.json", tokenizer)
    path.write_text(text.replace("  on_overflow: cut\n", "  on_overflow: cut\n  on_overflow: fail\n"), encoding="utf-8")
    with pytest.raises(RecipeError, match="duplicate key 'on_overflow'"):
        load_recipe(directory)
    # and no shipped or fixture recipe declares a key twice (the strict loader would refuse)
    iter_recipes(fixtures)
    iter_recipes(default_recipes_root())


_ENGINE_SPECIFIC_FIELDS: list[tuple[str, object]] = [
    ("runner", "pooling"),  # the engine's --runner: a serve field
    ("max_model_len", 4096),  # the engine's context: a serve field
    ("pooler_config", {"use_activation": True}),  # the engine-side activation pin
    ("dtype", "bfloat16"),  # the engine's weights dtype
    ("image", "vllm/vllm-openai:v0.31.0"),  # the engine image: an engine field
    ("gpus", 1),  # the tensor-parallel count: a resources field
]
"""Keys that name engine-side knobs (owner decision 19): none of them belongs in the engine-neutral client
block. The serve and engine blocks are already closed models (``extra=forbid``); this pins the plain-data
client side, which has no schema of its own in the lean package."""


@pytest.mark.parametrize(("key", "value"), _ENGINE_SPECIFIC_FIELDS, ids=[key for key, _ in _ENGINE_SPECIFIC_FIELDS])
def test_an_engine_specific_field_in_the_client_block_is_refused(tmp_path: Path, key: str, value: object) -> None:
    """Decision 19: the engine-neutral client block stays strictly apart from serve/engine/resources."""
    import shutil

    import yaml

    copied = tmp_path / "fixture-embed"
    copied.mkdir()
    for name in ("family.yaml", "reference.py"):
        shutil.copy(recipe_dirs_path() / "fixture-embed" / name, copied / name)
    data = yaml.safe_load((copied / "family.yaml").read_text(encoding="utf-8"))
    data["client"].update({key: value})
    (copied / "family.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(RecipeError, match="engine-specific"):
        load_recipe(copied)
