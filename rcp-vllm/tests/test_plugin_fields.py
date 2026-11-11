"""The plugin-provided fields: what a recipe may declare, and what the loader refuses.

The stock engine at the pinned release carries neither the embed chunker's three ``PoolerConfig`` knobs nor
the rerank route's ``relevant_spans`` wire field: all of it ships as the product's own plugin
(``rcp_vllm.recipe.PLUGIN_FIELDS`` is the one declaration of which plugin provides which field).  A recipe that
declares one therefore names the plugin in ``serve.plugin``, and the environment loading the recipe carries it
-- judged through the ``vllm.general_plugins`` entry-point group the packaging snapshot records, never by
importing the plugin's modules.

The gate, as tests:

- a recipe that declares the knobs renders an argv containing them
  (:func:`test_the_knobs_ride_inside_the_pooler_config`);
- against an engine without the plugin the loader refuses and the message names the plugin and its install
  line (:func:`test_against_an_engine_without_the_plugin_the_loader_refuses_and_names_the_install_line`);
- with the plugin installed the same recipe loads and its argv is unchanged
  (:func:`test_with_the_plugin_installed_the_recipe_loads_and_its_argv_is_unchanged`);
- removing the plugin removes the capability (:func:`test_removing_the_plugin_removes_the_capability`), so
  nobody mistakes the knobs for an engine feature.

``relevant_spans`` is the fourth case with a different rendering: it is a capability declaration, so nothing is
rendered at all, and :func:`test_no_flag_is_rendered_for_relevant_spans_against_the_pinned_flag_set` is the
positive control that keeps that choice from going stale against a later engine.

Every test is offline and writes only under ``tmp_path``; the plugin's presence is stubbed through
``rcp_vllm.recipe._installed_plugins``, the one function that reads installed metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import rcp_vllm.recipe as recipe_module
from rcp_vllm.errors import RecipeError
from rcp_vllm.recipe import (
    PINNED_POOLER_CONFIG_FIELDS,
    PLUGIN_FIELDS,
    Recipe,
    ServeConfig,
    load_recipe,
    serve_argv,
)

REVISION = "0000000000000000000000000000000000000000"

KNOBS = {"chunk_length": 512, "chunk_weighting": "linear", "normalize_aggregated_embeddings": True}
"""The three chunk knobs the plugin's patch adds to ``PoolerConfig``, as a recipe sets
them: one JSON object under ``--pooler-config``."""

_FAMILY = """\
# A user recipe file (never shipped): the gate's declaring and non-declaring halves.
id: {family}
schema_version: "1"
role: {role}
input: [text]
{scoring}{recipe_extra}licence: apache-2.0
engine:
  name: vllm
  image: "vllm/vllm-openai:v0.33.0"
  min_version: "0.33.0"
resources: {{gpus: 1}}
serve:
  runner: pooling
  dtype: bfloat16
  max_model_len: 10000
{serve_extra}client:
  api: {api}
  max_tokens: 8192
reference: {{kind: stored_scores, score_scale: {scale}}}
variants:
{variants}"""

_VARIANT = """\
  - id: {variant}
    model: example/{variant}
    revision: "{revision}"
"""


def _write_family(
    root: Path,
    *,
    role: str = "embed",
    family: str = "my-embedder",
    recipe_extra: str = "",
    serve_extra: str = "",
    variants: tuple[str, ...] | None = None,
) -> Path:
    """One user family directory carrying ``recipe_extra`` (a recipe-level line) and ``serve_extra`` (serve keys)."""
    rerank = role == "rerank"
    directory = root / family
    directory.mkdir(parents=True)
    text = _FAMILY.format(
        family=family,
        role=role,
        scoring="scoring: pointwise\n" if rerank else "",
        recipe_extra=recipe_extra,
        api="rerank" if rerank else "openai_embeddings",
        scale="probability" if rerank else "cosine",
        serve_extra=serve_extra,
        variants="".join(_VARIANT.format(variant=variant, revision=REVISION) for variant in (variants or (family,))),
    )
    (directory / "family.yaml").write_text(text, encoding="utf-8")
    return directory


def _flag_value(argv: list[str], flag: str) -> str:
    """The value of ``flag`` in ``argv`` (``--flag value``); the argv renders each flag once."""
    return argv[argv.index(flag) + 1]


def _knobs_recipe(root: Path, *, plugin: str = "rcp-vllm") -> Path:
    """A recipe that declares the three chunk knobs, beside the plugin spec it names (if any)."""
    pooler = json.dumps(KNOBS)
    serve = f"  pooler_config: {pooler}\n" if not plugin else f"  plugin: {plugin}\n  pooler_config: {pooler}\n"
    return _write_family(root, serve_extra=serve)


# ----------------------------------------------------------------------------------------------------------
# the gate: declared, rendered, refused, honoured
# ----------------------------------------------------------------------------------------------------------


def test_the_knobs_ride_inside_the_pooler_config(tmp_path: Path) -> None:
    """Gate 1: a recipe that declares the knobs renders an argv containing them.

    They are ``PoolerConfig`` fields, so they ride inside the ``--pooler-config`` JSON the engine already
    parses -- no flag of their own, which is also why an engine that lacks them rejects the argv rather than
    ignoring a knob it never sees.
    """
    recipe = load_recipe(_knobs_recipe(tmp_path))
    argv = serve_argv(recipe, port=8000, served_model_name=recipe.id)

    assert argv.count("--pooler-config") == 1
    assert json.loads(_flag_value(argv, "--pooler-config")) == KNOBS
    assert set(KNOBS).isdisjoint(PINNED_POOLER_CONFIG_FIELDS), "the knobs are not the pinned engine's fields"


def test_against_an_engine_without_the_plugin_the_loader_refuses_and_names_the_install_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate 2: no plugin in the environment, no capability -- and the refusal says how to get it.

    The message names the plugin and its exact install line (the recipe's own spec, shell-quoted), so an
    operator can paste it; a recipe whose declared field the target cannot honour never loads.
    """
    directory = _knobs_recipe(tmp_path, plugin="rcp-vllm>=0.0.1")
    monkeypatch.setattr(recipe_module, "_installed_plugins", lambda: frozenset())

    with pytest.raises(RecipeError) as excinfo:
        load_recipe(directory)

    message = str(excinfo.value)
    assert "chunk_length" in message and "chunk_weighting" in message
    assert "plugin rcp-vllm" in message and "does not carry" in message
    assert "pip install --no-deps 'rcp-vllm>=0.0.1'" in message
    assert "vllm.general_plugins" in message


def test_with_the_plugin_installed_the_recipe_loads_and_its_argv_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate 3: the same recipe loads with the plugin present, and the argv does not move.

    Installing the plugin adds the field to the engine, not to the recipe: the rendered command is a function
    of the recipe alone, so the recorded argv of a wave is the argv of the tree.
    """
    directory = _knobs_recipe(tmp_path)
    assert "rcp-vllm" in recipe_module._installed_plugins(), "this environment carries the plugin's entry point"

    installed = load_recipe(directory)
    monkeypatch.setattr(recipe_module, "_installed_plugins", lambda: frozenset({"rcp-vllm"}))
    stubbed = load_recipe(directory)

    assert serve_argv(stubbed, port=8000, served_model_name=stubbed.id) == serve_argv(
        installed, port=8000, served_model_name=installed.id
    )
    pooler = _flag_value(serve_argv(stubbed, port=8000, served_model_name=stubbed.id), "--pooler-config")
    assert json.loads(pooler) == KNOBS


def test_removing_the_plugin_removes_the_capability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gate 4: the knobs are the plugin's, not the engine's.

    Without the plugin the declaring recipe is refused, while the same recipe without the declaration loads and
    renders an empty ``--pooler-config``: nothing of the capability survives the plugin's removal, so it can
    never be mistaken for an engine feature.
    """
    monkeypatch.setattr(recipe_module, "_installed_plugins", lambda: frozenset())

    with pytest.raises(RecipeError, match="chunk_length"):
        load_recipe(_knobs_recipe(tmp_path / "declaring"))

    plain = load_recipe(_write_family(tmp_path / "plain", serve_extra="  pooler_config: {}\n"))
    argv = serve_argv(plain, port=8000, served_model_name=plain.id)
    assert json.loads(_flag_value(argv, "--pooler-config")) == {}


def test_a_recipe_that_declares_a_knob_without_the_plugin_is_refused(tmp_path: Path) -> None:
    """A field the engine lacks is never silently dropped: without ``serve.plugin`` the load refuses and names
    the plugin that provides it and the install line, before any engine sees the argv."""
    with pytest.raises(RecipeError) as excinfo:
        load_recipe(_knobs_recipe(tmp_path, plugin=""))

    message = str(excinfo.value)
    assert "not a field the pinned engine" in message
    assert "Declare serve.plugin: rcp-vllm" in message
    assert "pip install --no-deps rcp-vllm" in message


def test_a_plugin_that_does_not_provide_the_field_is_refused(tmp_path: Path) -> None:
    """The declaration names which plugin honours the field: a recipe whose ``serve.plugin`` is some other
    distribution would render a knob that engine rejects, so the load names both sides."""
    directory = _write_family(
        tmp_path, serve_extra=f"  plugin: some-other-plugin\n  pooler_config: {json.dumps(KNOBS)}\n"
    )
    with pytest.raises(RecipeError) as excinfo:
        load_recipe(directory)

    message = str(excinfo.value)
    assert "provided by the plugin rcp-vllm" in message and "some-other-plugin" in message


def test_a_pooler_key_no_plugin_provides_is_still_refused(tmp_path: Path) -> None:
    """The table is closed: a key that is neither the pinned engine's nor a declared plugin field is refused
    by name, and the refusal lists both sets, so a typo cannot ride in on the plugin's allowance."""
    directory = _write_family(tmp_path, serve_extra="  plugin: rcp-vllm\n  pooler_config: {chunk_len: 8}\n")
    with pytest.raises(RecipeError) as excinfo:
        load_recipe(directory)

    message = str(excinfo.value)
    assert "chunk_len" in message and "plugin-provided: chunk_length" in message


def test_the_exported_schema_declares_the_knobs_and_the_loader_still_refuses_a_typo(tmp_path: Path) -> None:
    """The exported schemas carry the three knobs as properties, and the loader stays the key authority.

    A JSON-Schema-only consumer reads ``serve.pooler_config``'s property set, so the fields the plugin honours
    must be there rather than in a description.  ``extra="allow"`` keeps the object open -- every stock
    ``PoolerConfig`` field still passes -- which is also why the schema cannot be the authority on which keys
    exist: a typo is refused by the loader, by name, before any engine sees the argv.
    """
    schema_dir = Path(__file__).resolve().parents[1] / "schema"
    for name in ("recipe.schema.json", "family.schema.json"):
        schema = json.loads((schema_dir / name).read_text(encoding="utf-8"))
        pooler = schema["$defs"]["ServeConfig"]["properties"]["pooler_config"]
        declared = schema["$defs"][pooler["$ref"].rsplit("/", 1)[-1]]
        assert set(declared["properties"]) == set(KNOBS), name
        assert declared["additionalProperties"] is True, f"{name}: a stock PoolerConfig field still passes"

    directory = _write_family(tmp_path, serve_extra="  plugin: rcp-vllm\n  pooler_config: {chunk_len: 8}\n")
    with pytest.raises(RecipeError) as excinfo:
        load_recipe(directory)

    message = str(excinfo.value)
    assert "chunk_len" in message and "plugin-provided: chunk_length" in message


def test_extra_args_may_not_repeat_the_pooler_config_flag(tmp_path: Path) -> None:
    """``serve_argv`` renders ``--pooler-config`` from ``serve.pooler_config``, so ``extra_args`` may not carry
    it: the engine keeps the last occurrence, and a second copy would silently drop every declared knob."""
    directory = _write_family(
        tmp_path,
        serve_extra=(
            f'  plugin: rcp-vllm\n  pooler_config: {json.dumps(KNOBS)}\n  extra_args: ["--pooler-config", "{{}}"]\n'
        ),
    )
    with pytest.raises(RecipeError) as excinfo:
        load_recipe(directory)

    message = str(excinfo.value)
    assert "--pooler-config" in message and "serve.pooler_config" in message


# ----------------------------------------------------------------------------------------------------------
# relevant_spans: a capability declaration, rendered nowhere
# ----------------------------------------------------------------------------------------------------------


def test_relevant_spans_renders_nothing(tmp_path: Path) -> None:
    """The rerank route's spans wire is a request field the plugin provides, so the recipe declares that the
    deployment honours it and renders no flag: the argv is the same with the declaration and without it."""
    plain = load_recipe(
        _write_family(tmp_path / "plain", role="rerank", family="my-reranker", serve_extra="  plugin: rcp-vllm\n")
    )
    declaring = load_recipe(
        _write_family(
            tmp_path / "declaring",
            role="rerank",
            family="my-reranker",
            recipe_extra="relevant_spans: true\n",
            serve_extra="  plugin: rcp-vllm\n",
        )
    )

    assert declaring.relevant_spans is True and plain.relevant_spans is False
    assert serve_argv(declaring, port=8000, served_model_name=declaring.id) == serve_argv(
        plain, port=8000, served_model_name=plain.id
    )


def test_no_flag_is_rendered_for_relevant_spans_against_the_pinned_flag_set(tmp_path: Path) -> None:
    """The positive control for the choice: the pinned ``vllm serve`` parser registers no spans option, so the
    declaration renders none -- and the same check would reject one if a later engine or plugin added a flag.

    This is the test to revisit when an engine grows a real flag: the record below is what makes the "renders
    nothing" decision visible rather than stale.
    """
    from tests.test_serve_argv_flags import VLLM_033_SERVE_FLAGS, _unknown_flags

    recipe = load_recipe(
        _write_family(
            tmp_path, role="rerank", recipe_extra="relevant_spans: true\n", serve_extra="  plugin: rcp-vllm\n"
        )
    )
    argv = serve_argv(recipe, port=8000, served_model_name=recipe.id)

    assert "--relevant-spans" not in VLLM_033_SERVE_FLAGS
    assert not [token for token in argv if "relevant" in token.lower()]
    assert _unknown_flags(argv) == []
    assert _unknown_flags([*argv, "--relevant-spans", "{}"]) == ["--relevant-spans"]


def test_relevant_spans_reaches_every_variant_of_the_family(tmp_path: Path) -> None:
    """The capability is the family's shared block (``scoring``'s neighbour), so every size resolves with it:
    it describes the request shape the deployment accepts, not a per-size fact."""
    directory = _write_family(
        tmp_path,
        role="rerank",
        family="my-two-sizes",
        recipe_extra="relevant_spans: true\n",
        serve_extra="  plugin: rcp-vllm\n",
        variants=("my-two-sizes-small", "my-two-sizes-large"),
    )
    for variant in ("my-two-sizes-small", "my-two-sizes-large"):
        assert load_recipe(directory, variant=variant).relevant_spans is True


def test_relevant_spans_needs_a_rerank_recipe(tmp_path: Path) -> None:
    """The spans wire answers a rerank request: an embed recipe declaring it would describe a route it does not
    serve, so the load refuses it by role."""
    directory = _write_family(tmp_path, recipe_extra="relevant_spans: true\n", serve_extra="  plugin: rcp-vllm\n")
    with pytest.raises(RecipeError) as excinfo:
        load_recipe(directory)

    message = str(excinfo.value)
    assert "relevant_spans" in message and "role=embed" in message


def test_relevant_spans_without_the_plugin_is_refused_with_the_install_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The capability is the plugin's: without it the loader refuses with the same install line the knobs get,
    because a served engine without the plugin would never honour the field it declares."""
    directory = _write_family(
        tmp_path, role="rerank", recipe_extra="relevant_spans: true\n", serve_extra="  plugin: rcp-vllm\n"
    )
    monkeypatch.setattr(recipe_module, "_installed_plugins", lambda: frozenset())

    with pytest.raises(RecipeError) as excinfo:
        load_recipe(directory)

    message = str(excinfo.value)
    assert "relevant_spans" in message and "plugin rcp-vllm" in message
    assert "pip install --no-deps rcp-vllm" in message


# ----------------------------------------------------------------------------------------------------------
# the declaration's shape, and why relevant_spans is not a serve field
# ----------------------------------------------------------------------------------------------------------


def test_the_declaration_names_the_plugin_that_honours_each_field() -> None:
    """One table, one plugin per field: the chunk knobs are ``serve.pooler_config`` keys and the spans wire is a
    recipe-level capability, and every row names the distribution that provides it."""
    assert PLUGIN_FIELDS == {
        "serve.pooler_config.chunk_length": "rcp-vllm",
        "serve.pooler_config.chunk_weighting": "rcp-vllm",
        "serve.pooler_config.normalize_aggregated_embeddings": "rcp-vllm",
        "relevant_spans": "rcp-vllm",
    }


def test_relevant_spans_is_not_a_serve_field_so_no_fingerprint_input_moves() -> None:
    """The tripwire on the placement: ``rcp_test.fingerprint`` hashes one input per field of the ``serve``
    block, so a serve-block home for this capability would re-key all 44 shipped recipes and invalidate every
    recorded corpus.  It changes the request shape the deployment accepts, not how the engine computes, so it
    lives at the top of the recipe; a lane that moves it under ``serve`` must bump the fingerprint and
    re-record, and this test is where that cost is read.
    """
    assert "relevant_spans" not in ServeConfig.model_fields
    assert "relevant_spans" in Recipe.model_fields
