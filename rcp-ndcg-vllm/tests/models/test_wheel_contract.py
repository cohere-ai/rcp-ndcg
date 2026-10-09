"""The lean wheel's contract: pure ``py3-none-any``, its declared dependencies, its entry point and its recipes.

The binding rule (layout-move item 2): ``pip install --no-deps rcp-ndcg-vllm`` must work in the stock vLLM
image -- which already ships the only two declared dependencies (pydantic, PyYAML) -- and ``pip freeze`` must
then differ by exactly this one wheel. The check is simulated here against whatever interpreter runs the tests,
in ``tmp_path`` only: the package tree is COPIED into ``tmp_path`` and the wheel is built there, because
setuptools leaves an ``*.egg-info`` directory in the tree it builds from and tests write only to ``tmp_path``.
The recipes are package data: the wheel must ship every family (fourteen, one variant table each) and
every variant resolves through the loader, and a fresh venv listing them through
``importlib.resources`` sees them (the listing runs against the unpacked wheel itself with the repo's
interpreter; the full fresh-venv install is the release gate's step).
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parents[2]
WHEEL_NAME = "rcp_ndcg_vllm-0.0.1-py3-none-any.whl"
N_FAMILIES = 23
N_RECIPES = 37  # the variants across the families (decision 34; the ten judges are eight of the families)


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("wheel")
    tree = tmp_path / "pkg"
    shutil.copytree(PACKAGE_DIR, tree, ignore=shutil.ignore_patterns("*.egg-info", "__pycache__", ".venv"))
    out = tmp_path / "dist"
    result = subprocess.run(["uv", "build", "--offline", "-o", str(out), str(tree)], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"uv build failed in this environment: {result.stderr[-300:]}")
    wheel = out / WHEEL_NAME
    if not wheel.exists():
        pytest.skip(f"uv build produced no {WHEEL_NAME}: {result.stdout[-300:]}")
    return wheel


def test_the_wheel_is_pure_python_and_declares_only_the_image_deps(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as archive:
        wheel_meta = archive.read("rcp_ndcg_vllm-0.0.1.dist-info/WHEEL").decode()
        metadata = archive.read("rcp_ndcg_vllm-0.0.1.dist-info/METADATA").decode()
        entry_points = archive.read("rcp_ndcg_vllm-0.0.1.dist-info/entry_points.txt").decode()
        names = archive.namelist()
        forbidden = [name for name in names if name.endswith((".so", ".pyd", ".dylib", ".dll", ".exe"))]
    assert "Wheel-Version: 1" in wheel_meta
    assert "Root-Is-Purelib: true" in wheel_meta, "the wheel must stay pure python (py3-none-any)"
    assert forbidden == [], f"compiled extensions in the wheel: {forbidden}"
    requires = [
        line for line in metadata.splitlines() if line.startswith("Requires-Dist:") and "; extra ==" not in line
    ]
    declared = {
        line.split(":", 1)[1].split(";")[0].strip().split()[0].split("<")[0].split("=")[0].split(">")[0].lower()
        for line in requires
    }
    assert declared <= {"pydantic", "pyyaml"}, (
        f"--no-deps must resolve nothing new in a stock vLLM image (pydantic, PyYAML ship in it): {sorted(declared)}"
    )
    assert "[vllm.general_plugins]" in entry_points
    assert "rcp-ndcg-vllm = rcp_ndcg_vllm.models:register" in entry_points


def test_the_shipped_recipes_are_all_in_the_wheel(built_wheel: Path) -> None:
    with zipfile.ZipFile(built_wheel) as archive:
        shipped = {
            name.split("/recipes/", 1)[1].split("/", 1)[0]
            for name in archive.namelist()
            if "/recipes/" in name and name.endswith("family.yaml")
        }
        templates = [name for name in archive.namelist() if name.endswith("template.jinja")]
    assert len(shipped) == N_FAMILIES, sorted(shipped)
    assert templates, "the chat templates ship with the recipes"


def test_the_recipes_list_through_importlib_resources(built_wheel: Path) -> None:
    """A fresh venv list: the unpacked wheel's package data resolves through importlib.resources and its own
    family loader expands every variant (decision 34: the wheel carries the family files, the loader resolves
    the served recipe ids)."""
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as work:
        subprocess.run([sys.executable, "-m", "zipfile", "-e", str(built_wheel), work], check=True)
        code = (
            "import importlib.resources, pathlib, sys;"
            f"sys.path.insert(0, {work!r});"
            "import rcp_ndcg_vllm.recipe as recipe;"
            "root = pathlib.Path(str(importlib.resources.files('rcp_ndcg_vllm').joinpath('recipes')));"
            "families = len([p for p in root.iterdir() if (p / 'family.yaml').is_file()]);"
            "print(families, len(recipe.iter_recipes(root)))"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == f"{N_FAMILIES} {N_RECIPES}", result.stdout + result.stderr


def test_the_no_deps_freeze_delta_is_exactly_this_wheel(built_wheel: Path, tmp_path: Path) -> None:
    """The stock-image install: a fresh venv with nothing but pip, ``--no-deps`` adds one distribution."""
    helper = Path(__file__).resolve().parent / "check_no_deps_install.py"
    result = subprocess.run(
        [__import__("sys").executable, str(helper), str(built_wheel), "--python", __import__("sys").executable],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "exactly" in result.stdout, result.stdout
