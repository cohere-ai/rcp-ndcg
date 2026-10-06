"""The plugins collector: what a wave's recipes install into the engine environment (item 11)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from rcp_ndcg_vllm.jobs.plugins import collect

from tests.conftest import RECIPES

PY = sys.executable
WHEEL = "plugin_wheel-1.0.0-py3-none-any.whl"


def _recipes_root(tmp_path: Path, plugin_spec: str | None, recipe_ids: list[str]) -> Path:
    """The fixture recipes with a serve.plugin set on the first, staged wheel file beside it."""
    root = tmp_path / "recipes"
    shutil.copytree(RECIPES, root)
    recipe_dir = root / recipe_ids[0]
    recipe_yaml = recipe_dir / "recipe.yaml"
    text = recipe_yaml.read_text(encoding="utf-8")
    if plugin_spec is not None:
        assert "  plugin: null\n" in text
        text = text.replace("  plugin: null\n", f"  plugin: {plugin_spec}\n")
        recipe_yaml.write_text(text, encoding="utf-8")
    if plugin_spec is not None and plugin_spec.endswith(".whl"):
        (recipe_dir / plugin_spec).write_bytes(b"stub wheel bytes")
    return root


def test_collect_returns_recipe_relative_paths_for_staged_wheels(tmp_path: Path) -> None:
    """A wheel inside the recipe directory is reported as <recipe-id>/<file>, so the bootstrap finds it."""
    root = _recipes_root(tmp_path, "plugin_wheel-1.0.0-py3-none-any.whl", ["fixture-embed"])
    assert collect(root, ["fixture-embed"]) == ["fixture-embed/plugin_wheel-1.0.0-py3-none-any.whl"]


def test_collect_passes_a_name_through_unchanged(tmp_path: Path) -> None:
    """A spec that is not a staged file passes through as named (the item-9 index fallback)."""
    root = _recipes_root(tmp_path, "private-plugin==1.2.3", ["fixture-embed"])
    assert collect(root, ["fixture-embed"]) == ["private-plugin==1.2.3"]


def test_collect_dedupes_across_recipes(tmp_path: Path) -> None:
    """Two recipes declaring the same plugin collect it once."""
    root = _recipes_root(tmp_path, "private-plugin==1.2.3", ["fixture-embed"])
    shutil.copytree(root / "fixture-embed", root / "fixture-embed-cls", dirs_exist_ok=True)
    cls_yaml = root / "fixture-embed-cls" / "recipe.yaml"
    cls_yaml.write_text(
        cls_yaml.read_text(encoding="utf-8").replace("id: fixture-embed", "id: fixture-embed-cls"), encoding="utf-8"
    )
    for recipe in ("fixture-embed", "fixture-embed-cls"):
        assert "  plugin: private-plugin==1.2.3\n" in (root / recipe / "recipe.yaml").read_text(encoding="utf-8")
    assert collect(root, ["fixture-embed", "fixture-embed-cls"]) == ["private-plugin==1.2.3"]


def test_collect_skips_recipes_without_a_plugin(tmp_path: Path) -> None:
    """A recipe with serve.plugin: null collects nothing; io_processor_plugin is recorded, not installed."""
    root = _recipes_root(tmp_path, None, ["fixture-embed"])
    assert collect(root, ["fixture-embed"]) == []


def test_collect_cli_prints_one_spec_per_line(tmp_path: Path) -> None:
    """The CLI the bootstrap calls: `python -m rcp_ndcg_vllm.jobs.plugins collect`."""
    root = _recipes_root(tmp_path, "plugin_wheel-1.0.0-py3-none-any.whl", ["fixture-embed"])
    wave_list = tmp_path / "wave.txt"
    wave_list.write_text("# one per line\nfixture-embed\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "rcp_ndcg_vllm.jobs.plugins",
            "collect",
            "--recipes-root",
            str(root),
            "--recipes",
            f"@{wave_list}",
        ],  # fmt: skip
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "fixture-embed/plugin_wheel-1.0.0-py3-none-any.whl"
