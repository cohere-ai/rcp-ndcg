"""The recipe schema: valid fixtures per role, one failing case per validator, and the golden argv."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rcp_ndcg_vllm import (
    Recipe,
    RecipeError,
    client_config,
    iter_recipes,
    load_recipe,
    recipe_json_schema,
    serve_argv,
)
from rcp_ndcg_vllm.errors import HarnessError

from tests.fixtures import OTHER_REV, REV, recipe_data

_SCHEMA = Path(__file__).resolve().parents[1] / "schema" / "recipe.schema.json"

# (field path, the change that must be refused, the message part a reader needs)
# Note for recipe lanes: YAML reads a bare ``no`` as false - always quote string tokens like "no" and "yes".
INVALID_CASES = [
    ("id", {"id": "Uppercase"}, [], "id"),
    ("revision", {"revision": "0123456789abcdef"}, [], "40"),
    ("role", {"role": "classify"}, [], "role"),
    ("input", {"input": []}, [], "input"),
    ("scoring-on-embed", {"scoring": "pointwise"}, [], "scoring", "fixture-embed"),
    ("scoring-missing", {}, ["scoring"], "scoring"),
    ("instruction-on-embed", {"client.instruction": "fold"}, [], "instruction", "fixture-embed"),
    ("instruction-missing", {}, ["client", "instruction"], "instruction"),
    ("default-instruction-missing", {"client.default_instruction": None}, [], "default_instruction"),
    ("embed-dtype-on-embed", {"client.embed_dtype": "float32"}, [], "embed_dtype"),
    ("normalize-on-rerank", {"client.normalize": True}, [], "normalize"),
    ("normalize-missing", {}, ["client", "normalize"], "normalize"),
    ("api-mismatch", {"client.api": "vllm_pooling"}, [], "client.api"),
    ("listwise-batch-size", {"client.batch_size": 4, "scoring": "listwise"}, [], "batch size"),
    ("unknown-field", {"totally_unknown": 1}, [], "totally_unknown"),
    ("tokenizer-format", {"client.tokenizer": "Qwen/Qwen3-Reranker-0.6B"}, [], "tokenizer"),
    ("chat-template-missing", {"serve.chat_template": "nope.jinja"}, [], "chat_template"),
    ("reference-entry-missing", {"reference.entry": "missing_reference.py"}, [], "reference.entry"),
    ("bad-min-version", {"engine.min_version": "0.31"}, [], "min_version"),
    ("zero-gpus", {"resources.gpus": 0}, [], "gpus"),
    ("zero-max-tokens", {"client.max_tokens": 0}, [], "max_tokens"),
    ("empty-sources", {"sources": ["   "]}, [], "sources"),
]


def test_every_fixture_recipe_loads() -> None:
    recipes = iter_recipes(recipe_dirs_path())
    assert {recipe.id for recipe in recipes} >= {
        "fixture-embed",
        "fixture-multi-vector",
        "fixture-rerank-pointwise",
        "fixture-rerank-listwise",
    }


def recipe_dirs_path() -> Path:
    from tests.fixtures import FIXTURES

    return FIXTURES / "recipes"


def _load_with(
    tmp_path: Path,
    changes: dict[str, object] | None = None,
    drop: list[str] | None = None,
    base: str = "fixture-rerank-pointwise",
) -> Recipe:
    """A fixture recipe with dotted-path fields overridden (or dropped), written to tmp and loaded back."""
    data = recipe_data(base)
    for path, value in (changes or {}).items():
        target = data
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    for path in drop or []:
        target = data
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target.pop(parts[-1], None)
    recipe_dir = recipe_dirs_path() / base
    directory = tmp_path / recipe_dir.name
    directory.mkdir(parents=True)
    (directory / "recipe.yaml").write_text(_to_yaml(data), encoding="utf-8")
    for name in ("template.jinja", "reference.py"):
        if (recipe_dir / name).is_file():
            (directory / name).write_text((recipe_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
    return load_recipe(directory)


def _to_yaml(data: dict) -> str:
    import yaml

    return yaml.safe_dump(data, sort_keys=False)


def test_load_recipe_from_a_bare_yaml_file(tmp_path: Path) -> None:
    data = recipe_data("fixture-embed")
    path = tmp_path / "recipe.yaml"
    path.write_text(_to_yaml(data), encoding="utf-8")
    (tmp_path / "reference.py").write_text(
        (recipe_dirs_path() / "fixture-embed" / "reference.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    recipe = load_recipe(path)
    assert recipe.id == "fixture-embed"
    assert recipe.client.api == "openai_embeddings"


def test_load_recipe_rejects_a_mismatched_directory_name(tmp_path: Path) -> None:
    data = recipe_data("fixture-embed")
    directory = tmp_path / "wrong-name"
    directory.mkdir()
    (directory / "recipe.yaml").write_text(_to_yaml(data), encoding="utf-8")
    with pytest.raises(RecipeError, match="must equal the directory name"):
        load_recipe(directory)


def test_load_recipe_rejects_a_missing_recipe_yaml(tmp_path: Path) -> None:
    with pytest.raises(RecipeError, match="no recipe at"):
        load_recipe(tmp_path / "missing")


def test_iter_recipes_names_the_broken_directory(tmp_path: Path) -> None:
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "recipe.yaml").write_text("id: broken\n", encoding="utf-8")
    with pytest.raises(RecipeError, match="broken"):
        iter_recipes(tmp_path)


@pytest.mark.parametrize(
    ("case", "changes", "drop", "message", "base"),
    [case if len(case) == 5 else (*case, "fixture-rerank-pointwise") for case in INVALID_CASES],
    ids=[case[0] for case in INVALID_CASES],
)
def test_invalid_recipes_fail_with_a_clear_message(
    tmp_path: Path, case: str, changes: dict[str, object], drop: list[str], message: str, base: str
) -> None:
    with pytest.raises(RecipeError, match=message) as excinfo:
        _load_with(tmp_path / "case" / case.replace("/", "-"), changes=changes, drop=drop, base=base)
    assert "recipe.yaml" in str(excinfo.value)


def test_stored_scores_needs_no_reference_file(tmp_path: Path) -> None:
    recipe = _load_with(tmp_path / "stored", changes={"reference.kind": "stored_scores"}, drop=["reference.entry"])
    assert recipe.reference.kind == "stored_scores"


def test_serve_argv_rerank_pointwise_is_golden() -> None:
    recipe = load_recipe(recipe_dirs_path() / "fixture-rerank-pointwise")
    argv = serve_argv(recipe, port=8100, served_model_name="fixture-rerank-pointwise")
    assert argv == [
        "vllm",
        "serve",
        "fixtures/PointwiseReranker",
        "--revision",
        REV,
        "--served-model-name",
        "fixture-rerank-pointwise",
        "--host",
        "0.0.0.0",
        "--port",
        "8100",
        "--tensor-parallel-size",
        "1",
        "--runner",
        "pooling",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "512",
        "--hf-overrides",
        '{"architectures": ["FixtureForSequenceClassification"], "classifier_from_token": ["no", "yes"],'
        ' "is_original_fixture_reranker": true}',  # fmt: skip
        "--chat-template",
        str(recipe_dirs_path() / "fixture-rerank-pointwise" / "template.jinja"),
        "--pooler-config",
        '{"use_activation": true}',
    ]


def test_serve_argv_embed_role_and_extra_args() -> None:
    recipe = load_recipe(recipe_dirs_path() / "fixture-rerank-listwise")
    argv = serve_argv(recipe, port=8101, served_model_name="fixture-rerank-listwise")
    assert argv[argv.index("--convert") + 1] == "classify"
    assert argv[argv.index("--dtype") + 1] == "float16"
    assert "--trust-remote-code" in argv
    assert argv[argv.index("--pooler-config") + 1] == "{}"
    assert argv[-1] == "--enable-prefix-caching"


def test_serve_argv_json_fields_are_sorted_and_deterministic() -> None:
    recipe = load_recipe(recipe_dirs_path() / "fixture-rerank-pointwise")
    first = serve_argv(recipe, port=1, served_model_name="x")
    second = serve_argv(recipe, port=1, served_model_name="x")
    assert first == second


def test_serve_argv_needs_the_recipe_directory_for_a_template() -> None:
    data = recipe_data("fixture-rerank-pointwise")
    recipe = Recipe.model_validate(data)
    with pytest.raises(RecipeError, match="load_recipe"):
        serve_argv(recipe, port=1, served_model_name="x")


def test_client_config_rerank() -> None:
    recipe = load_recipe(recipe_dirs_path() / "fixture-rerank-pointwise")
    config = client_config(recipe, base_url="http://127.0.0.1:8100/v1")
    assert config == {
        "api": "rerank",
        "base_url": "http://127.0.0.1:8100/v1",
        "model": "fixture-rerank-pointwise",
        "revision": REV,
        "recipe": "fixture-rerank-pointwise",
        "instruction": "fold",
        "default_instruction": "Follow the task.",
        "tokenizer": f"fixtures/PointwiseReranker@{REV}",
        "max_tokens": 512,
        "template": {
            "anchor": "last",
            "anchor_markers": [],
            "pair": [
                {"text": "SYSTEM: Judge whether the Document answers the Query. Answer yes or no.\nUSER:\nQuery: "},
                {"content": "query"},
                {"text": "\nDocument: "},
                {"content": "document"},
                {"text": "\nASSISTANT:"},
            ],
            "query_max_tokens": 256,
        },
    }


def test_client_config_listwise_has_no_batch_size() -> None:
    recipe = load_recipe(recipe_dirs_path() / "fixture-rerank-listwise")
    config = client_config(recipe, base_url="http://127.0.0.1:8100/v1")
    assert "batch_size" not in config
    assert config["instruction"] == "field"
    assert config["use_activation"] is False


def test_client_config_embed_carries_prompts_and_normalize() -> None:
    recipe = load_recipe(recipe_dirs_path() / "fixture-embed")
    config = client_config(recipe, base_url="http://127.0.0.1:8100/v1")
    assert config["kind"] == "dense"
    encoder = config["encoder"]
    assert encoder["api"] == "openai_embeddings"
    assert "doc_prompt" not in encoder  # the template block replaced the prompt prefixes
    assert encoder["normalize"] is True
    assert encoder["batch_size"] == 4
    assert encoder["template"]["anchor"] == "last"
    assert encoder["template"]["document"][-1] == {"text": " [END]"}


def test_client_config_multi_vector_carries_embed_dtype() -> None:
    recipe = load_recipe(recipe_dirs_path() / "fixture-multi-vector")
    encoder = client_config(recipe, base_url="http://127.0.0.1:8100/v1")["encoder"]
    assert encoder["api"] == "vllm_pooling"
    assert encoder["embed_dtype"] == "float16"


def test_effective_embed_dtype_defaults_to_float16() -> None:
    from rcp_ndcg_vllm.recipe import effective_embed_dtype

    recipe = load_recipe(recipe_dirs_path() / "fixture-multi-vector")
    assert effective_embed_dtype(recipe) == "float16"


def test_json_schema_export_is_current() -> None:
    assert json.loads(_SCHEMA.read_text(encoding="utf-8")) == recipe_json_schema()


def test_revision_pattern_is_40_hex() -> None:
    data = recipe_data("fixture-embed")
    data["revision"] = OTHER_REV
    assert Recipe.model_validate(data).revision == OTHER_REV


def test_startup_timeout_defaults_and_overrides() -> None:
    recipe = load_recipe(recipe_dirs_path() / "fixture-embed")
    assert recipe.engine.startup_timeout_s == 60
    from rcp_ndcg_vllm import EngineSpec

    engine = EngineSpec(name="vllm", image="vllm/vllm-openai:v0.31.0", min_version="0.31.0")
    assert engine.startup_timeout_s == 1800


def test_load_recipe_wraps_validation_errors_with_the_file_path(tmp_path: Path) -> None:
    directory = tmp_path / "broken"
    directory.mkdir()
    (directory / "recipe.yaml").write_text("id: [1, 2\n", encoding="utf-8")
    with pytest.raises(RecipeError, match="not valid YAML"):
        load_recipe(directory)


def test_stored_scores_stage2_is_refused_clearly() -> None:
    from rcp_ndcg_vllm.equivalence import stage2_scores

    recipe = load_recipe(recipe_dirs_path() / "fixture-embed")
    broken = recipe.model_copy(update={"reference": recipe.reference.model_copy(update={"kind": "stored_scores"})})
    with pytest.raises(HarnessError, match="stored_scores"):
        stage2_scores(broken, "http://127.0.0.1:1", [], None)


def test_serve_argv_embed_and_multi_vector_are_golden() -> None:
    """The embedding roles render their flags in the same fixed order, with an empty pooler object."""
    embed = load_recipe(recipe_dirs_path() / "fixture-embed")
    argv = serve_argv(embed, port=8102, served_model_name="fixture-embed")
    assert argv[:9] == ["vllm", "serve", "fixtures/DenseEmbedder", "--revision", REV, "--served-model-name",
                        "fixture-embed", "--host", "0.0.0.0"]  # fmt: skip
    assert argv[argv.index("--pooler-config") + 1] == '{"seq_pooling_type": "LAST"}'
    assert "--chat-template" not in argv
    multi = load_recipe(recipe_dirs_path() / "fixture-multi-vector")
    argv = serve_argv(multi, port=8103, served_model_name="fixture-multi-vector")
    assert argv[argv.index("--pooler-config") + 1] == '{"task": "token_embed"}'
    assert "--trust-remote-code" not in argv


def test_use_activation_is_rerank_only(tmp_path_factory: pytest.TempPathFactory) -> None:
    with pytest.raises(RecipeError, match="use_activation"):
        _load_with(tmp_path_factory.mktemp("useact"), {"client.use_activation": False}, base="fixture-embed")


def test_default_instruction_with_instruction_none_is_refused(tmp_path_factory: pytest.TempPathFactory) -> None:
    with pytest.raises(RecipeError, match="default_instruction"):
        _load_with(
            tmp_path_factory.mktemp("definstr"),
            {"client.instruction": "none", "client.default_instruction": "fold me"},
        )


def test_startup_timeout_must_be_positive(tmp_path_factory: pytest.TempPathFactory) -> None:
    with pytest.raises(RecipeError, match="startup_timeout_s"):
        _load_with(tmp_path_factory.mktemp("timeout"), {"engine.startup_timeout_s": 0})


def test_gates_reject_out_of_range_values(tmp_path_factory: pytest.TempPathFactory) -> None:
    with pytest.raises(RecipeError, match="tau_min"):
        _load_with(tmp_path_factory.mktemp("gates"), {"gates": {"tau_min": 2.0}})
    with pytest.raises(RecipeError, match="vec_min_cosine"):
        _load_with(tmp_path_factory.mktemp("gates2"), {"gates": {"vec_min_cosine": 2.0}})


def test_pooler_task_must_be_token_embed_for_multi_vector(tmp_path_factory: pytest.TempPathFactory) -> None:
    with pytest.raises(RecipeError, match="token_embed"):
        _load_with(
            tmp_path_factory.mktemp("pooler"), {"serve.pooler_config": {"task": "embed"}}, base="fixture-multi-vector"
        )


def test_cli_reports_a_recipe_error_without_a_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A bad recipe reaches the operator as one error line and exit 2, not a traceback."""
    from rcp_ndcg_vllm.equivalence import main

    directory = tmp_path / "wrong-name"
    directory.mkdir()
    (directory / "recipe.yaml").write_text(_to_yaml(recipe_data("fixture-embed")), encoding="utf-8")
    (tmp_path / "pairs.jsonl").write_text('{"query": "q", "documents": ["d"]}\n', encoding="utf-8")
    argv = ["--recipe", str(directory), "--pairs", str(tmp_path / "pairs.jsonl"), "--out", str(tmp_path / "o")]
    assert main(argv) == 2
    assert "must equal the directory name" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# The anchor/budget validators (research-lane follow-up).
# ---------------------------------------------------------------------------


def test_template_shape_without_content_span_is_refused(tmp_path_factory: pytest.TempPathFactory) -> None:
    with pytest.raises(RecipeError, match="content span"):
        _load_with(
            tmp_path_factory.mktemp("shape"),
            {
                "client.template": {
                    "pair": [{"text": "fixed head "}, {"text": "fixed tail"}],
                    "anchor": "last",
                    "query_max_tokens": 64,
                }
            },  # fmt: skip
        )


def test_anchor_last_shape_must_end_fixed(tmp_path_factory: pytest.TempPathFactory) -> None:
    with pytest.raises(RecipeError, match="anchor=last"):
        _load_with(
            tmp_path_factory.mktemp("tail"),
            {
                "client.template": {
                    "pair": [{"text": "head "}, {"content": "query"}, {"content": "document"}],
                    "anchor": "last",
                    "query_max_tokens": 64,
                }
            },  # fmt: skip
        )


def test_anchor_marker_needs_markers(tmp_path_factory: pytest.TempPathFactory) -> None:
    with pytest.raises(RecipeError, match="anchor_markers"):
        _load_with(
            tmp_path_factory.mktemp("markers"),
            {"client.template": {"document": [{"text": "doc: "}, {"content": "document"}], "anchor": "marker"}},
        )


def test_anchor_first_shape_must_start_fixed(tmp_path_factory: pytest.TempPathFactory) -> None:
    with pytest.raises(RecipeError, match="anchor=first"):
        _load_with(
            tmp_path_factory.mktemp("head"),
            {"client.template": {"document": [{"content": "document"}, {"text": " tail"}], "anchor": "first"}},
        )


def test_empty_default_instruction_is_refused(tmp_path_factory: pytest.TempPathFactory) -> None:
    with pytest.raises(RecipeError, match="empty string"):
        _load_with(tmp_path_factory.mktemp("empty"), {"client.default_instruction": ""})


def test_blocking_is_listwise_only(tmp_path_factory: pytest.TempPathFactory) -> None:
    with pytest.raises(RecipeError, match="listwise"):
        _load_with(
            tmp_path_factory.mktemp("blocking"),
            {"client.blocking": {"block_size": 8, "capacity_formula": "max_tokens - query", "weighting": "max"}},
        )


def test_pooler_config_unknown_key_is_refused(tmp_path_factory: pytest.TempPathFactory) -> None:
    with pytest.raises(RecipeError, match="not PoolerConfig fields"):
        _load_with(tmp_path_factory.mktemp("pooler"), {"serve.pooler_config": {"normalize": True}})


def test_truncate_prompt_tokens_field_does_not_exist(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Engine-side truncation is not a recipe field: the client owns every cut."""
    with pytest.raises(RecipeError, match="truncate_prompt_tokens"):
        _load_with(tmp_path_factory.mktemp("trunc"), {"client.truncate_prompt_tokens": 4096})


def test_aggregation_requires_chunking(tmp_path_factory: pytest.TempPathFactory) -> None:
    with pytest.raises(RecipeError, match="on_overflow: chunk"):
        _load_with(tmp_path_factory.mktemp("aggr"), {"client.aggregation": "max"})


def test_client_budget_must_fit_the_engine_context(tmp_path_factory: pytest.TempPathFactory) -> None:
    with pytest.raises(RecipeError, match="max_model_len"):
        _load_with(tmp_path_factory.mktemp("budget"), {"client.max_tokens": 4096})
