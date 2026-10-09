"""``recipe: <id>`` (docs-firstcontact Q1): the mapping form and the CLI shorthand.

The refusal tests first: an explicit CONTENT field that disagrees with the recipe is refused naming both
values, an unknown id names the shipped ids, and an absent rcp-ndcg-vllm is a typed error with the install
line (docs-site OQ-6).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.inference.config import EmbeddingEndpoint, PoolingEndpoint, RerankEndpoint
from rcp_ndcg.inference.recipes import (
    available_recipe_ids,
    expand_role_recipe,
    recipe_client_data,
    recipe_role,
    shorthand_config,
)

CLASSES = {"embed": EmbeddingEndpoint, "multi_vector": PoolingEndpoint, "rerank": RerankEndpoint}
RECIPE_ID = "qwen3-reranker-0.6b"


def test_the_mapping_form_takes_the_client_block_from_the_recipe() -> None:
    data = expand_role_recipe({"recipe": RECIPE_ID, "base_url": "http://127.0.0.1:8000/v1"}, classes=CLASSES)
    client = recipe_client_data(RECIPE_ID)
    assert data["tokenizer"] == client["tokenizer"]
    assert data["max_tokens"] == client["max_tokens"]
    assert data["model"] == RECIPE_ID
    assert data["revision"] == client["revision"]
    assert data["base_url"] == "http://127.0.0.1:8000/v1"  # RUNTIME stays on the config
    assert data["recipe"] == RECIPE_ID  # the pointer: the config's own spelling


def test_an_explicit_content_field_that_equals_the_recipes_is_accepted() -> None:
    client = recipe_client_data(RECIPE_ID)
    data = expand_role_recipe({"recipe": RECIPE_ID, "max_tokens": client["max_tokens"]}, classes=CLASSES)
    assert data["max_tokens"] == client["max_tokens"]


def test_an_explicit_content_field_that_disagrees_is_refused_naming_both_values() -> None:
    with pytest.raises(ConfigError) as excinfo:
        expand_role_recipe({"recipe": RECIPE_ID, "max_tokens": 7}, classes=CLASSES)
    message = str(excinfo.value)
    assert "max_tokens: 7" in message and "8192" in message, message


def test_the_recipe_itself_is_the_pointer_and_never_conflicts() -> None:
    data = expand_role_recipe({"recipe": "qwen3-reranker-0.6b"}, classes=CLASSES)
    assert data["recipe"] == "qwen3-reranker-0.6b"


def test_a_field_the_recipe_leaves_undeclared_accepts_the_configs_value() -> None:
    # context-specific fields the recipe declares nothing about: the config's explicit value is obeyed.
    assert "query_prompt" not in recipe_client_data(RECIPE_ID)
    data = expand_role_recipe({"recipe": RECIPE_ID, "query_prompt": "Ask: "}, classes=CLASSES)
    assert data["query_prompt"] == "Ask: "
    # ...while a declared one is the recipe's: qwen3-reranker-0.6b declares instruction: none.
    assert recipe_client_data(RECIPE_ID)["instruction"] == "none"
    assert expand_role_recipe({"recipe": RECIPE_ID}, classes=CLASSES)["instruction"] == "none"


def test_an_unknown_recipe_id_names_the_shipped_ids() -> None:
    with pytest.raises(ConfigError, match="no shipped recipe") as excinfo:
        expand_role_recipe({"recipe": "nonexistent-recipe"}, classes=CLASSES)
    assert RECIPE_ID in excinfo.value.hint


def test_a_missing_rcp_ndcg_vllm_is_a_typed_error_with_the_install_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "rcp_ndcg_vllm", None)  # make the lazy import raise ModuleNotFoundError
    with pytest.raises(ConfigError, match="needs the serving recipes") as excinfo:
        available_recipe_ids()
    assert "pip install rcp-ndcg-vllm" in excinfo.value.hint


def test_the_cli_shorthand_is_the_mapping_in_one_string() -> None:
    assert shorthand_config(f"recipe:{RECIPE_ID}") == {"recipe": RECIPE_ID}
    assert expand_role_recipe(shorthand_config(f"recipe:{RECIPE_ID}"), classes=CLASSES)["model"] == RECIPE_ID
    with pytest.raises(ConfigError, match="names no recipe"):
        shorthand_config("recipe:")


def test_a_declared_mrl_selection_is_not_a_content_disagreement() -> None:
    """A recipe declares its MRL kind and set; the run selects ``k`` from it (owner decision 39).

    The recipes ship the full width (``dimensions: null``, no ``mrl_dim``), so a selection would read as a
    CONTENT disagreement with the recipe's declared ``null``; a ``k`` inside the declared
    ``mrl_dims``/``mrl_range`` is a selection, not a disagreement, and one outside is refused naming the
    set. The selection is the same rule on the engine-side ``dimensions`` and the client-side ``mrl_dim``.
    """
    data = expand_role_recipe({"recipe": "qwen3-embedding-0.6b", "dimensions": 512}, classes=CLASSES)
    assert data["dimensions"] == 512  # the recipe declares null; the run selects from mrl_range
    config = EmbeddingEndpoint(**data)
    assert config.dimensions == 512
    assert config.mrl_kind == "truncation" and config.mrl_range == (32, 1024)

    with pytest.raises(ConfigError) as excinfo:
        expand_role_recipe({"recipe": "qwen3-embedding-0.6b", "dimensions": 4096}, classes=CLASSES)
    assert "mrl_range [32, 1024]" in str(excinfo.value), excinfo.value

    data = expand_role_recipe({"recipe": "qwen3-embedding-0.6b", "mrl_dim": 128}, classes=CLASSES)
    assert EmbeddingEndpoint(**data).mrl_dim == 128
    with pytest.raises(ConfigError, match=r"mrl_range \[32, 1024\]"):
        expand_role_recipe({"recipe": "qwen3-embedding-0.6b", "mrl_dim": 4096}, classes=CLASSES)

    # a discrete card set is the same rule (jina's table, declared once in the recipe)
    data = expand_role_recipe({"recipe": "jina-embeddings-v5-text-small", "mrl_dim": 512}, classes=CLASSES)
    assert EmbeddingEndpoint(**data).mrl_dim == 512
    with pytest.raises(ConfigError) as excinfo:
        expand_role_recipe({"recipe": "jina-embeddings-v5-text-small", "mrl_dim": 1000}, classes=CLASSES)
    assert "mrl_dims" in str(excinfo.value) and "1024" in str(excinfo.value), excinfo.value


def test_a_selection_on_a_kind_none_recipe_is_refused_naming_the_kind() -> None:
    """A recipe that declares ``mrl_kind: none`` has no set to select from: the refusal names the kind."""
    with pytest.raises(ConfigError, match="mrl_kind 'none'"):
        expand_role_recipe({"recipe": "octen-embedding-8b", "mrl_dim": 512}, classes=CLASSES)
    with pytest.raises(ConfigError, match="mrl_kind 'none'"):
        expand_role_recipe({"recipe": "octen-embedding-8b", "dimensions": 512}, classes=CLASSES)


def test_a_recipe_that_declares_its_own_selection_keeps_it(tmp_path: Path) -> None:
    """A recipe whose client block pins a selection is CONTENT: an in-set config selection may not
    silently replace it. Only a recipe that ships the full width (an undeclared selection) hands the
    selection to the run; a pinned one is served as declared or refused naming both values."""
    import shutil

    import yaml as yaml_module
    from rcp_ndcg_vllm.recipe import default_recipes_root

    target = tmp_path / "qwen3-embedding-pinned"
    shutil.copytree(default_recipes_root() / "qwen3-embedding", target)
    yaml_path = target / "family.yaml"
    data = yaml_module.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["id"] = "qwen3-embedding-pinned"
    data["client"]["dimensions"] = 128
    yaml_path.write_text(yaml_module.safe_dump(data, sort_keys=False), encoding="utf-8")

    # the recipe's own selection is served as declared, and the same value is accepted explicitly
    assert expand_role_recipe({"recipe": str(target)}, classes=CLASSES)["dimensions"] == 128
    assert expand_role_recipe({"recipe": str(target), "dimensions": 128}, classes=CLASSES)["dimensions"] == 128
    # a different in-set value is a CONTENT disagreement, refused naming both values
    with pytest.raises(ConfigError) as excinfo:
        expand_role_recipe({"recipe": str(target), "dimensions": 256}, classes=CLASSES)
    message = str(excinfo.value)
    assert "256" in message and "128" in message, message


def test_the_recipe_roles_drive_the_retriever_kind() -> None:
    assert recipe_role("octen-embedding-8b") == "embed"  # role data reads without the product resolution
    assert recipe_role("pplx-embed-v2-context-9b-preview") == "multi_vector"
    assert recipe_role(RECIPE_ID) == "rerank"


def test_a_served_wire_takes_the_recipe_and_builds() -> None:
    config = RerankEndpoint(**expand_role_recipe({"recipe": RECIPE_ID}, classes=CLASSES))
    assert config.model == RECIPE_ID
    assert config.api == "rerank"


def test_the_shorthand_flows_through_the_retrieval_unions() -> None:
    """The union's mapping form and the CLI shorthand land on the same validated config."""
    from rcp_ndcg.retrieval import validate_reranker, validate_retriever

    reranker = validate_reranker(shorthand_config(f"recipe:{RECIPE_ID}")).model_dump()
    assert reranker["model"] == RECIPE_ID and reranker["api"] == "rerank"

    retriever = validate_retriever(
        {"kind": "dense", "encoder": shorthand_config("recipe:jina-embeddings-v5-text-small")}
    )
    assert retriever.encoder.model == "jina-embeddings-v5-text-small"

    late = validate_retriever(
        {"kind": "late_interaction", "encoder": shorthand_config("recipe:pplx-embed-v2-context-9b-preview")}
    )
    assert late.encoder.api == "vllm_pooling"

    with pytest.raises(ConfigError, match="the recipe is a reranker"):
        from rcp_ndcg.cli.retrieval import _role_config

        _role_config(f"recipe:{RECIPE_ID}", [], which="retriever")


def _user_family(
    tmp_path: Path,
    *,
    family_id: str = "user-qwen3-reranker",
    variant_id: str = "user-qwen3-reranker-0.6b",
    extra_variants: tuple[str, ...] = (),
) -> Path:
    """A user recipe file: a copy of a shipped family, re-idd and reduced to one size (plus any extras)."""
    import shutil

    import yaml as yaml_module
    from rcp_ndcg_vllm.recipe import default_recipes_root

    target = tmp_path / family_id
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(default_recipes_root() / "qwen3-reranker", target)
    yaml_path = target / "family.yaml"
    data = yaml_module.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["id"] = family_id
    first = dict(data["variants"][0])
    first["id"] = variant_id
    data["variants"] = [first, *[{**first, "id": extra} for extra in extra_variants]]
    yaml_path.write_text(yaml_module.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def test_a_user_recipe_directory_resolves_through_the_path_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``recipe:./path`` (and ``recipe:/abs/path``): a file of the operator's own, loaded by the same
    schema, marked unshipped and identified by the content hash of its resolved form."""
    directory = _user_family(tmp_path)
    data = expand_role_recipe(shorthand_config(f"recipe:{directory}"), classes=CLASSES)
    assert data["model"] == "user-qwen3-reranker-0.6b"
    assert data["max_tokens"] == 8192  # the file's client block, exactly as a shipped recipe's would
    assert data["recipe"].startswith("unshipped:sha256:")  # the identity: never a shipped id

    monkeypatch.chdir(tmp_path)  # the relative spelling the docs name
    relative = expand_role_recipe(shorthand_config(f"recipe:./{directory.name}"), classes=CLASSES)
    assert relative["recipe"] == data["recipe"]


def test_a_user_recipes_identity_moves_with_the_file_and_not_otherwise(tmp_path: Path) -> None:
    directory = _user_family(tmp_path)
    identity = expand_role_recipe({"recipe": str(directory)}, classes=CLASSES)["recipe"]
    assert expand_role_recipe({"recipe": str(directory)}, classes=CLASSES)["recipe"] == identity
    copy = _user_family(tmp_path / "elsewhere")
    assert expand_role_recipe({"recipe": str(copy)}, classes=CLASSES)["recipe"] == identity

    yaml_path = directory / "family.yaml"
    yaml_path.write_text(
        yaml_path.read_text(encoding="utf-8").replace("max_tokens: 8192", "max_tokens: 4096"), encoding="utf-8"
    )
    assert expand_role_recipe({"recipe": str(directory)}, classes=CLASSES)["recipe"] != identity


def test_a_multi_variant_user_family_directory_names_its_variants(tmp_path: Path) -> None:
    """``recipe:`` resolves one recipe: a user family with several sizes needs a single-variant directory."""
    directory = _user_family(tmp_path, extra_variants=("user-qwen3-reranker-4b",))

    with pytest.raises(ConfigError) as excinfo:
        expand_role_recipe({"recipe": str(directory)}, classes=CLASSES)
    message = str(excinfo.value)
    assert "user-qwen3-reranker-0.6b" in message and "user-qwen3-reranker-4b" in message


def test_an_expanded_configs_identity_pointer_passes_through(tmp_path: Path) -> None:
    """A recorded config carries the recipe's expanded client block and the identity pointer: re-reading it --
    a run config's resume, a retrieval index's reload -- must not resolve the hash as an id or a path."""
    from rcp_ndcg.retrieval import validate_reranker

    directory = _user_family(tmp_path)
    expanded = expand_role_recipe({"recipe": str(directory), "base_url": None}, classes=CLASSES)
    assert expanded["recipe"].startswith("unshipped:sha256:")
    assert expand_role_recipe(expanded, classes=CLASSES) == expanded

    config = validate_reranker(expanded)
    again = validate_reranker(config.model_dump())
    assert again.recipe == expanded["recipe"]
    assert again.max_tokens == expanded["max_tokens"]


def test_a_tilde_path_is_not_claimed() -> None:
    """``~`` is not a path form here (nothing expands it), so it is refused like any unknown id; the docs
    name ``./``, ``../`` and ``/abs`` only."""
    with pytest.raises(ConfigError, match="no shipped recipe"):
        expand_role_recipe({"recipe": "~/my-family"}, classes=CLASSES)


def test_a_user_recipe_round_trips_through_a_run_config(tmp_path: Path) -> None:
    """The whole point of the identity: the recorded config holds the expanded block and the pointer, and
    re-validating it (``run status``, a resume, a reopen) accepts the pointer as it stands."""
    from rcp_ndcg.runs import RunConfig

    directory = _user_family(tmp_path)
    config = RunConfig.model_validate(
        {
            "dataset": "jsonl:rows.jsonl",
            "judge": "fake",
            "candidates": {
                "from": "retrieval",
                "retrieval": {"kind": "bm25"},
                "rerank": {"recipe": str(directory), "base_url": "http://127.0.0.1:8000/v1"},
            },
            "steps": ["retrieve", "rerank"],
        }
    )
    reranker = config.candidates.rerank
    assert reranker is not None and str(reranker.recipe).startswith("unshipped:sha256:")
    again = RunConfig.from_data(config.resolved())
    assert again.candidates.rerank is not None
    assert again.candidates.rerank.recipe == reranker.recipe
    assert again.candidates.rerank.max_tokens == reranker.max_tokens


def test_a_user_recipe_of_an_unreadable_schema_version_is_refused(tmp_path: Path) -> None:
    """Decision 18's version check applies to a user's file unchanged."""
    import yaml as yaml_module

    directory = _user_family(tmp_path)
    yaml_path = directory / "family.yaml"
    data = yaml_module.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["schema_version"] = "999"
    yaml_path.write_text(yaml_module.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="schema_version '999' is not one this rcp-ndcg reads"):
        expand_role_recipe({"recipe": str(directory)}, classes=CLASSES)


def test_a_recipe_of_an_unreadable_schema_version_is_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Decision 18: the recipe file format is the versioned contract; a recipe whose ``schema_version`` this
    rcp-ndcg does not read is refused naming the version and the ones it reads (a newer rcp-ndcg-vllm must
    fail here, not load with a schema surprise)."""
    import shutil

    import rcp_ndcg_vllm.recipe as vllm_recipe
    import yaml as yaml_module
    from rcp_ndcg_vllm.recipe import default_recipes_root

    family_id = RECIPE_ID.rsplit("-", 1)[0]  # the variant's family directory (decision 34)
    target = tmp_path / family_id
    shutil.copytree(default_recipes_root() / family_id, target)
    yaml_path = target / "family.yaml"
    data = yaml_module.safe_load(yaml_path.read_text(encoding="utf-8"))
    data["schema_version"] = "999"
    yaml_path.write_text(yaml_module.safe_dump(data, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(vllm_recipe, "default_recipes_root", lambda: tmp_path)

    with pytest.raises(ConfigError, match="schema_version '999' is not one this rcp-ndcg reads") as excinfo:
        expand_role_recipe({"recipe": RECIPE_ID, "base_url": None}, classes=CLASSES)
    assert "1" in excinfo.value.hint
