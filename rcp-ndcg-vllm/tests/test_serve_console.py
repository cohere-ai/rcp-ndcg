"""``rcp-ndcg-vllm serve``: the deployment overrides, the user recipe files and the console's refusals.

Every test here is offline (the recipe package alone: package data, pydantic and PyYAML) and runs in the
``vllm-pkg`` job.  The console is driven through :func:`~rcp_ndcg_vllm.serve.run_console` -- the same
``argparse`` tree ``tests/contract`` snapshots -- so a refusal is asserted where the operator sees it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rcp_ndcg_vllm.errors import RecipeError
from rcp_ndcg_vllm.recipe import (
    FIELD_ROLES,
    Recipe,
    RecipeFieldRole,
    deployment_fields,
    load_recipe,
    parse_deployment_overrides,
    recipe_digest,
    serve_argv,
)
from rcp_ndcg_vllm.serve import run_console

SHIPPED = "qwen3-reranker-0.6b"
REVISION = "0000000000000000000000000000000000000000"

USER_FAMILY = """\
# A user recipe file (never shipped): the console loads it by path and marks every record unshipped.
id: {family}
schema_version: "1"
role: rerank
input: [text]
scoring: pointwise
licence: apache-2.0
engine:
  name: vllm
  image: "vllm/vllm-openai:v0.31.0"
  min_version: "0.31.0"
resources: {{gpus: 1}}
serve:
  runner: pooling
  dtype: bfloat16
  max_model_len: 10000
client:
  api: rerank
  max_tokens: {max_tokens}
  use_activation: true
reference: {{kind: stored_scores, score_scale: probability}}
status: {status}
variants:
{variants}"""

VARIANT = """\
  - id: {variant}
    model: {model}
    revision: "{revision}"
"""


def _write_family(
    root: Path,
    *,
    family: str = "my-reranker",
    variants: list[tuple[str, str]] | None = None,
    max_tokens: int = 8192,
    status: str = "{state: unverified, image: null, date: null, report: null}",
    directory_name: str | None = None,
    template: str | None = None,
    serve_extra: str = "",
) -> Path:
    """One user family directory: the YAML the console and the loader read, and the files it references."""
    variants = variants or [(family, "example/My-Reranker")]
    directory = root / (directory_name or family)
    directory.mkdir(parents=True, exist_ok=True)
    yaml_text = USER_FAMILY.format(
        family=family,
        max_tokens=max_tokens,
        status=status,
        variants="".join(
            VARIANT.format(variant=variant, model=model, revision=REVISION) for variant, model in variants
        ),
    )
    if template is not None:
        (directory / "template.jinja").write_text(template, encoding="utf-8")
        yaml_text = yaml_text.replace("  runner: pooling", "  runner: pooling\n  chat_template: template.jinja")
    if serve_extra:
        yaml_text = yaml_text.replace("  runner: pooling", f"  runner: pooling\n{serve_extra}")
    (directory / "family.yaml").write_text(yaml_text, encoding="utf-8")
    return directory


def _flag_value(argv: list[str], flag: str) -> str:
    """The single value ``flag`` carries in ``argv`` (the flag must appear exactly once)."""
    assert argv.count(flag) == 1, f"{flag} appears {argv.count(flag)} times in {argv}"
    return argv[argv.index(flag) + 1]


# ----------------------------------------------------------------------------------------------------------
# the declaration
# ----------------------------------------------------------------------------------------------------------


def test_the_declaration_marks_the_deployment_fields_and_their_flags() -> None:
    """The schema declares the deployment surface once: a DEPLOYMENT role, the flag, the value kind."""
    assert set(deployment_fields()) == {
        "resources.gpus",
        "serve.gpu_memory_utilization",
        "serve.host",
        "serve.max_model_len",
        "serve.max_num_batched_tokens",
        "serve.max_num_seqs",
        "serve.port",
    }
    assert all(spec.role is RecipeFieldRole.DEPLOYMENT and spec.flag for spec in deployment_fields().values())
    assert FIELD_ROLES["engine.startup_timeout_s"].role is RecipeFieldRole.RUNTIME


def test_every_declared_path_is_a_recipe_field_or_a_declared_argv_knob() -> None:
    """A declared path either names a field of the recipe schema or is one of the engine's own argv
    knobs (which the recipe schema does not carry: the engine's defaults apply)."""
    from rcp_ndcg_vllm.recipe import EngineSpec, Resources, ServeConfig

    models = {"resources": Resources, "serve": ServeConfig, "engine": EngineSpec}
    for path in FIELD_ROLES:
        block, _, field = path.partition(".")
        assert block in models, path
        assert field, path
        if field not in models[block].model_fields:
            assert FIELD_ROLES[path].role is RecipeFieldRole.DEPLOYMENT, path
            assert FIELD_ROLES[path].flag is not None, path


# ----------------------------------------------------------------------------------------------------------
# the overrides
# ----------------------------------------------------------------------------------------------------------


def test_the_deployment_overrides_render_into_the_argv() -> None:
    """Every declared field reaches the argv: the resource, scheduling and address knobs of the engine."""
    recipe = load_recipe(SHIPPED)
    values = parse_deployment_overrides(
        [
            "resources.gpus=2",
            "serve.gpu_memory_utilization=0.75",
            "serve.max_num_seqs=64",
            "serve.max_num_batched_tokens=4096",
            "serve.host=127.0.0.1",
            "serve.port=9001",
            "serve.max_model_len=12000",
        ]
    )
    argv = serve_argv(recipe, port=8000, served_model_name=recipe.id, deployment=values)
    assert _flag_value(argv, "--tensor-parallel-size") == "2"
    assert _flag_value(argv, "--gpu-memory-utilization") == "0.75"
    assert _flag_value(argv, "--max-num-seqs") == "64"
    assert _flag_value(argv, "--max-num-batched-tokens") == "4096"
    assert _flag_value(argv, "--host") == "127.0.0.1"
    assert _flag_value(argv, "--port") == "9001"  # the override beats the run's --port
    assert _flag_value(argv, "--max-model-len") == "12000"


def test_a_recipe_without_overrides_renders_exactly_todays_argv() -> None:
    """The shipped recipes' argv is unchanged: the deployment defaults are the values it always carried."""
    recipe = load_recipe(SHIPPED)
    assert serve_argv(recipe, port=8100, served_model_name=recipe.id) == serve_argv(
        recipe, port=8100, served_model_name=recipe.id, deployment={}
    )
    argv = serve_argv(recipe, port=8100, served_model_name=recipe.id)
    assert _flag_value(argv, "--host") == "0.0.0.0"
    assert _flag_value(argv, "--port") == "8100"
    assert "--gpu-memory-utilization" not in argv
    assert "--max-num-seqs" not in argv


def test_a_content_override_is_refused_by_name_with_the_variant_hint() -> None:
    """A content field is refused by name: a different value is a different variant, never a flag."""
    with pytest.raises(RecipeError) as excinfo:
        parse_deployment_overrides(["serve.dtype=float16"])
    message = str(excinfo.value)
    assert "serve.dtype" in message
    assert "a different revision or content is a different variant: add a variant row" in message


def test_a_content_override_is_refused_for_every_shape_of_content_field() -> None:
    """The model, the revision, the pooler config, a template, the hf overrides and a patch are all content --
    a path below a field (``serve.hf_overrides.architectures``) included."""
    for pair in (
        "model=example/Other-Model",
        "revision=1111111111111111111111111111111111111111",
        "serve.pooler_config={}",
        "serve.hf_overrides={}",
        "serve.hf_overrides.architectures=[OtherModel]",
        "serve.chat_template=other.jinja",
        "client.template={}",
    ):
        with pytest.raises(RecipeError) as excinfo:
            parse_deployment_overrides([pair])
        assert "add a variant row" in str(excinfo.value), pair


def test_a_runtime_override_is_refused_naming_its_role() -> None:
    """``engine.startup_timeout_s`` is declared RUNTIME: the run owns it, not the serve command."""
    with pytest.raises(RecipeError) as excinfo:
        parse_deployment_overrides(["engine.startup_timeout_s=60"])
    message = str(excinfo.value)
    assert "engine.startup_timeout_s" in message and "RUNTIME" in message


def test_an_unknown_override_names_the_deployment_surface() -> None:
    for path in ("serve.plugin_modules", "resources.gpus.extra", "serve.max_model_len.extra"):
        with pytest.raises(RecipeError) as excinfo:
            parse_deployment_overrides([f"{path}=[1]"])
        message = str(excinfo.value)
        assert path in message and "serve.max_num_seqs" in message, path
        assert "add a variant row" not in message, path  # below a deployment field is not a content field


def test_a_malformed_or_out_of_range_override_is_refused() -> None:
    with pytest.raises(RecipeError, match="PATH=VALUE"):
        parse_deployment_overrides(["serve.max_num_seqs"])
    with pytest.raises(RecipeError, match="serve.max_num_seqs"):
        parse_deployment_overrides(["serve.max_num_seqs=lots"])
    with pytest.raises(RecipeError, match="serve.gpu_memory_utilization"):
        parse_deployment_overrides(["serve.gpu_memory_utilization=2"])
    with pytest.raises(RecipeError, match="resources.gpus"):
        parse_deployment_overrides(["resources.gpus=0"])


def test_a_non_finite_or_zero_gpu_memory_utilization_is_refused() -> None:
    """NaN slips past a range comparison, and the engine's flag is a fraction strictly above zero."""
    for raw in ("nan", "inf", "-inf", "0", "0.0"):
        with pytest.raises(RecipeError) as excinfo:
            parse_deployment_overrides([f"serve.gpu_memory_utilization={raw}"])
        assert "serve.gpu_memory_utilization" in str(excinfo.value), raw
    values = parse_deployment_overrides(["serve.gpu_memory_utilization=0.05"])
    assert values["serve.gpu_memory_utilization"] == 0.05


def test_the_port_argument_is_checked_like_the_declared_field() -> None:
    """``--port`` is the run's own spelling of the same value: it gets the same range, and the message names
    the flag the operator used."""
    recipe = load_recipe(SHIPPED)
    for bad in (-1, 65536):
        with pytest.raises(RecipeError) as excinfo:
            serve_argv(recipe, port=bad, served_model_name=recipe.id)
        assert "--port" in str(excinfo.value) and str(bad) in str(excinfo.value)
    # 0 is the run's ephemeral-port convention (the wave runner's stub engines announce it), 65535 the top
    assert _flag_value(serve_argv(recipe, port=0, served_model_name=recipe.id), "--port") == "0"
    assert _flag_value(serve_argv(recipe, port=65535, served_model_name=recipe.id), "--port") == "65535"


def test_the_budget_rule_allows_the_largest_budget_and_refuses_below_it() -> None:
    """``serve.max_model_len`` at the client's largest token budget passes; one token below is refused with
    both numbers named.  Raising it above the recipe's own value is allowed."""
    recipe = load_recipe(SHIPPED)  # client.max_tokens 8192 (the largest budget), query_max_tokens 4096
    at_the_edge = serve_argv(
        recipe,
        port=8000,
        served_model_name=recipe.id,
        deployment=parse_deployment_overrides(["serve.max_model_len=8192"]),
    )
    assert _flag_value(at_the_edge, "--max-model-len") == "8192"

    raised = serve_argv(
        recipe,
        port=8000,
        served_model_name=recipe.id,
        deployment=parse_deployment_overrides(["serve.max_model_len=32768"]),
    )
    assert _flag_value(raised, "--max-model-len") == "32768"

    with pytest.raises(RecipeError) as excinfo:
        serve_argv(
            recipe,
            port=8000,
            served_model_name=recipe.id,
            deployment=parse_deployment_overrides(["serve.max_model_len=8191"]),
        )
    message = str(excinfo.value)
    assert "8191" in message and "8192" in message


def test_the_largest_budget_is_the_largest_of_the_clients_declared_budgets(tmp_path: Path) -> None:
    """A client whose document budget is its largest one sets the floor (not just ``max_tokens``)."""
    directory = _write_family(tmp_path, max_tokens=4096)
    yaml_path = directory / "family.yaml"
    yaml_path.write_text(
        yaml_path.read_text(encoding="utf-8").replace(
            "  max_tokens: 4096", "  max_tokens: 4096\n  document_max_tokens: 9000"
        ),
        encoding="utf-8",
    )
    recipe = load_recipe(directory)
    with pytest.raises(RecipeError) as excinfo:
        serve_argv(
            recipe,
            port=8000,
            served_model_name=recipe.id,
            deployment=parse_deployment_overrides(["serve.max_model_len=8192"]),
        )
    message = str(excinfo.value)
    assert "8192" in message and "9000" in message and "document_max_tokens" in message


def test_the_dry_run_prints_the_argv_the_identity_and_the_applied_overrides(capsys: pytest.CaptureFixture[str]) -> None:
    code = run_console(["serve", SHIPPED, "--dry-run", "--set", "serve.max_num_seqs=64"])
    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("vllm serve ") and "--max-num-seqs 64" in lines[0]
    assert lines[1] == f"identity: {SHIPPED}"
    assert lines[2] == "deployment overrides: serve.max_num_seqs=64"


def test_the_dry_run_says_so_when_there_are_no_overrides(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_console(["serve", SHIPPED, "--dry-run"]) == 0
    assert capsys.readouterr().out.splitlines()[2] == "deployment overrides: none"


def test_the_console_refuses_a_content_override(capsys: pytest.CaptureFixture[str]) -> None:
    code = run_console(["serve", SHIPPED, "--dry-run", "--set", "serve.dtype=float16"])
    assert code == 1
    error = capsys.readouterr().err
    assert "serve.dtype" in error and "add a variant row" in error


def test_the_console_refuses_a_port_out_of_range(capsys: pytest.CaptureFixture[str]) -> None:
    for bad in ("-1", "70000"):
        assert run_console(["serve", SHIPPED, "--dry-run", "--port", bad]) == 1
        assert "--port" in capsys.readouterr().err


def test_a_real_serve_logs_the_identity_and_the_applied_overrides(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A real serve writes the two record lines ``--dry-run`` prints (the argv is the engine's, exec'd)."""
    import rcp_ndcg_vllm.serve as serve_module

    monkeypatch.setattr(serve_module.shutil, "which", lambda _name: "/usr/bin/vllm")

    def _exec(*_args: object) -> None:
        raise SystemExit(0)

    monkeypatch.setattr(serve_module.os, "execvp", _exec)
    with pytest.raises(SystemExit):
        serve_module.run_console(["serve", SHIPPED, "--set", "serve.max_num_seqs=64"])
    error = capsys.readouterr().err
    assert f"rcp-ndcg-vllm: serving {SHIPPED}" in error
    assert f"rcp-ndcg-vllm: identity: {SHIPPED}" in error
    assert "rcp-ndcg-vllm: deployment overrides: serve.max_num_seqs=64" in error


def test_a_second_serve_port_flag_is_still_the_console_tree(capsys: pytest.CaptureFixture[str]) -> None:
    """``--port`` keeps working: the deployment override is the only new spelling."""
    assert run_console(["serve", SHIPPED, "--dry-run", "--port", "8123"]) == 0
    assert "--port 8123" in capsys.readouterr().out.splitlines()[0]


# ----------------------------------------------------------------------------------------------------------
# the user recipe files
# ----------------------------------------------------------------------------------------------------------


def test_a_user_recipe_directory_is_served_by_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    directory = _write_family(tmp_path)
    assert run_console(["serve", str(directory), "--dry-run"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "example/My-Reranker" in lines[0]
    assert lines[1].startswith("identity: unshipped:sha256:")
    assert lines[2] == "deployment overrides: none"


def test_a_family_directory_needs_a_variant_and_serves_with_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _write_family(
        tmp_path,
        variants=[("my-reranker-small", "example/My-Reranker-Small"), ("my-reranker-big", "example/My-Reranker-Big")],
    )
    assert run_console(["serve", str(directory), "--dry-run"]) == 1
    error = capsys.readouterr().err
    assert "my-reranker-small" in error and "my-reranker-big" in error and "--variant" in error

    assert run_console(["serve", str(directory), "--variant", "my-reranker-big", "--dry-run"]) == 0
    assert "example/My-Reranker-Big" in capsys.readouterr().out.splitlines()[0]


def test_a_variant_flag_on_a_shipped_id_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_console(["serve", SHIPPED, "--variant", "qwen3-reranker-4b", "--dry-run"]) == 1
    error = capsys.readouterr().err
    assert "--variant" in error and SHIPPED in error


def test_an_unknown_variant_of_a_family_directory_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _write_family(tmp_path)
    assert run_console(["serve", str(directory), "--variant", "my-other", "--dry-run"]) == 1
    assert "my-reranker" in capsys.readouterr().err


def test_a_missing_template_file_is_refused_by_name(tmp_path: Path) -> None:
    """The referenced-file check speaks first: a family naming a template it does not ship is refused with the
    field's name (the identity's own read would only name the path)."""
    directory = _write_family(tmp_path, family="templated")
    yaml_path = directory / "family.yaml"
    yaml_path.write_text(
        yaml_path.read_text(encoding="utf-8").replace(
            "  runner: pooling", "  runner: pooling\n  chat_template: missing.jinja"
        ),
        encoding="utf-8",
    )
    with pytest.raises(RecipeError, match="serve.chat_template .* does not exist in"):
        load_recipe(directory)


def test_a_user_recipes_identity_is_the_content_hash_of_its_resolved_form(tmp_path: Path) -> None:
    """Two loads of one file agree; a changed file does not.  The identity is never a shipped id."""
    directory = _write_family(tmp_path)
    first = load_recipe(directory)
    assert first.shipped is False
    assert first.identity == f"unshipped:sha256:{recipe_digest(first)}"
    assert load_recipe(directory).identity == first.identity

    # the same content at another path is the same recipe (the hash is over the resolved form, not the path)
    copy = _write_family(tmp_path / "elsewhere", directory_name="my-reranker")
    assert load_recipe(copy).identity == first.identity

    # a changed file (here: the client budget) is a different recipe
    changed = _write_family(tmp_path, max_tokens=4096, directory_name="my-reranker")
    assert load_recipe(changed).identity != first.identity


def test_a_user_recipes_identity_covers_the_referenced_template(tmp_path: Path) -> None:
    """The template file's bytes change what the engine renders, so they are part of the resolved form's
    content hash: two files that differ only in template.jinja are two recipes."""
    directory = _write_family(tmp_path, family="templated", template="{{ query }}")
    first = load_recipe(directory)
    (directory / "template.jinja").write_text("{{ document }}", encoding="utf-8")
    second = load_recipe(directory)
    assert first.identity != second.identity
    # ...and the same template bytes at another path are the same recipe
    (directory / "template.jinja").write_text("{{ query }}", encoding="utf-8")
    assert load_recipe(directory).identity == first.identity


def test_a_shipped_recipes_identity_is_its_id() -> None:
    recipe = load_recipe(SHIPPED)
    assert recipe.shipped is True and recipe.identity == SHIPPED
    assert Recipe.model_validate(recipe.model_dump(mode="json")).shipped is True  # the dump carries no marker


def test_shippedness_follows_the_directory_not_the_spelling(tmp_path: Path) -> None:
    """A shipped recipe stays shipped however its root is spelled (the harness passes the package's own root
    explicitly), and a directory outside that root is unshipped even when it holds a shipped family's copy."""
    from rcp_ndcg_vllm.recipe import default_recipes_root, resolve_recipe

    spelled_out = resolve_recipe(SHIPPED, root=default_recipes_root())
    assert spelled_out.shipped is True and spelled_out.identity == SHIPPED
    assert spelled_out.status.state == load_recipe(SHIPPED).status.state

    import shutil

    copy = tmp_path / "qwen3-reranker"
    shutil.copytree(default_recipes_root() / "qwen3-reranker", copy)
    assert resolve_recipe(SHIPPED, root=tmp_path).shipped is False
    assert load_recipe(copy, variant=SHIPPED).shipped is False


def test_an_unshipped_recipes_status_is_unverified(tmp_path: Path) -> None:
    """A user file's own verification claim is not ours: every record of it says unverified."""
    directory = _write_family(
        tmp_path,
        status="{state: verified, image: 'vllm/vllm-openai:v0.31.0', date: '2026-01-01', report: 'https://example.invalid/report'}",
    )
    recipe = load_recipe(directory)
    assert recipe.status.state == "unverified"
    assert recipe.status.image is None and recipe.status.date is None and recipe.status.report is None


def test_a_shipped_recipes_status_is_kept(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The forcing is the unshipped path's alone: a family directory under the package's own root keeps its
    declared status, so a verified shipped recipe stays verified."""
    import rcp_ndcg_vllm.recipe as recipe_module

    directory = _write_family(
        tmp_path,
        family="verified-family",
        status="{state: verified, image: 'vllm/vllm-openai:v0.31.0', date: '2026-01-01', report: 'https://example.invalid/r'}",
    )
    monkeypatch.setattr(recipe_module, "default_recipes_root", lambda: tmp_path)
    recipe = recipe_module.load_recipe(directory)
    assert recipe.shipped is True
    assert recipe.status.state == "verified"
    assert recipe.status.image == "vllm/vllm-openai:v0.31.0"


def test_a_shipped_id_is_not_shadowed_by_a_directory_of_the_same_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An id-shaped source is the catalog's recipe first: a directory of the same name in the working
    directory does not shadow it (the file is named as a path: ``./name``)."""
    directory = _write_family(tmp_path, family=SHIPPED, variants=[(SHIPPED, "example/Shadow")])
    monkeypatch.chdir(tmp_path)

    assert run_console(["serve", SHIPPED, "--dry-run"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == f"identity: {SHIPPED}" and "example/Shadow" not in lines[0]

    assert run_console(["serve", f"./{directory.name}", "--dry-run"]) == 0
    assert "example/Shadow" in capsys.readouterr().out.splitlines()[0]


# ----------------------------------------------------------------------------------------------------------
# the plugin code declaration (rcp-fp/4)
# ----------------------------------------------------------------------------------------------------------


def test_the_plugin_declaration_is_mandatory_and_checked() -> None:
    """A plugin recipe names the architectures its engine registers; a patch names a shipped patch; both
    without the plugin that runs them are refused, so no declaration is silently inert."""
    from pydantic import ValidationError
    from rcp_ndcg_vllm.recipe import ServeConfig

    base = {"runner": "pooling", "dtype": "bfloat16", "max_model_len": 1024}
    with pytest.raises(ValidationError, match="plugin_architectures"):
        ServeConfig(**base, plugin="rcp-ndcg-vllm")
    with pytest.raises(ValidationError, match="plugin_architectures"):
        ServeConfig(**base, plugin_architectures=("PplxContextualModel",))
    with pytest.raises(ValidationError, match="PplxTypo"):
        ServeConfig(**base, plugin="rcp-ndcg-vllm", plugin_architectures=("PplxTypo",))
    with pytest.raises(ValidationError, match="not-a-patch"):
        ServeConfig(
            **base,
            plugin="rcp-ndcg-vllm",
            plugin_architectures=("PplxContextualModel",),
            patches=("not-a-patch",),
        )
    with pytest.raises(ValidationError, match="patch"):
        ServeConfig(**base, patches=("pooling-full-context",))
    declared = ServeConfig(
        **base,
        plugin="rcp-ndcg-vllm",
        plugin_architectures=("PplxContextualModel",),
        patches=("pooling-full-context",),
    )
    assert declared.plugin_architectures == ("PplxContextualModel",)
    assert declared.patches == ("pooling-full-context",)


def test_a_real_serve_renders_the_recipes_patches_into_the_engine_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recipe's declared patches are the engine's opt-in: ``serve`` sets ``RCP_NDCG_VLLM_PATCHES``
    from the recipe (overriding an inherited value), so the fingerprint's patch inputs cover what runs."""
    import os

    import rcp_ndcg_vllm.serve as serve_module
    from rcp_ndcg_vllm.patches import PATCHES_ENV

    directory = _write_family(
        tmp_path,
        family="patched",
        serve_extra=(
            "  plugin: rcp-ndcg-vllm\n  plugin_architectures: [PplxContextualModel]\n  patches: [pooling-full-context]"
        ),
    )
    monkeypatch.setattr(serve_module.shutil, "which", lambda _name: "/usr/bin/vllm")
    monkeypatch.setenv(PATCHES_ENV, "some-other-patch")

    def _exec(*_args: object) -> None:
        raise SystemExit(0)

    monkeypatch.setattr(serve_module.os, "execvp", _exec)
    with pytest.raises(SystemExit):
        serve_module.run_console(["serve", str(directory)])
    assert os.environ[PATCHES_ENV] == "pooling-full-context"


def test_a_serve_without_patches_clears_an_inherited_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recipe that opts into no patch runs none: an inherited opt-in is cleared, never silently applied
    outside the fingerprint."""
    import os

    import rcp_ndcg_vllm.serve as serve_module
    from rcp_ndcg_vllm.patches import PATCHES_ENV

    monkeypatch.setattr(serve_module.shutil, "which", lambda _name: "/usr/bin/vllm")
    monkeypatch.setenv(PATCHES_ENV, "pooling-full-context")

    def _exec(*_args: object) -> None:
        raise SystemExit(0)

    monkeypatch.setattr(serve_module.os, "execvp", _exec)
    with pytest.raises(SystemExit):
        serve_module.run_console(["serve", SHIPPED])
    assert os.environ[PATCHES_ENV] == ""
