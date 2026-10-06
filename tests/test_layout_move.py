"""``tools/layout_move.py`` moves a repository to the four-distribution tree, and stays replayable.

The tests run the real script on a scratch ``git worktree`` of a small fixture repository (the
moves, the path rewrites and the pyproject split), a second time to prove it changes nothing on an
already-moved tree, and once on a broken tree to watch it refuse (exit 1, the conflict listed). The
fixture lives under ``tmp_path`` like every test's data; the checkout is only ever read.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REAL_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "layout_move.py"

PYPROJECT = """\
[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[project]
name = "rcp-ndcg"
version = "0.0.1"
dependencies = ["rcp-ndcg-core==0.0.1"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
rcp-ndcg = "rcp_ndcg.cli.main:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.uv.workspace]
members = ["packages/*"]
exclude = ["packages/rcp-ndcg-vllm"]

[tool.uv.sources]
rcp-ndcg = { workspace = true }
rcp-ndcg-core = { workspace = true }

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 120
extend-exclude = ["packages/rcp-ndcg-vllm/recipes/*/qwen3_vl_embedding.py"]

[tool.basedpyright]
include = ["src", "packages/rcp-ndcg-core/src", "packages/rcp-ndcg-vllm/src"]
"""

MANIFEST = """\
include LICENSE NOTICE *.md
recursive-include src/rcp_ndcg *.py
prune tests
"""

HELPERS = '''\
"""A tests/ helper reading the root files by their paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FULL = (ROOT / "README.md").read_text(encoding="utf-8")
MANIFEST_DATA = ROOT / "pyproject.toml"
CORE_README = ROOT / "packages" / "rcp-ndcg-core" / "README.md"
LAYERS_SRC = Path(__file__).resolve().parents[2] / "src" / "rcp_ndcg"
SOURCES = (ROOT / "src", ROOT / "packages" / "rcp-ndcg-core" / "src")
FIXTURE_MANIFEST = package / "pyproject.toml"  # a tmp fixture's own manifest, not this repository's
UNDER = Path("packages/rcp-ndcg-core/src")
'''

CI = """\
jobs:
  test:
    steps:
      - run: uv run pytest tests/
      - name: the pins live beside the code (packages/rcp-ndcg-core/src/rcp_ndcg_core)
        run: sed -n 's/^version = "(.*)"/\\1/p' pyproject.toml
"""

BUILD_SH = """\
#!/usr/bin/env bash
# Build from the root: see packages/rcp-ndcg-vllm for the serving dist.
version=$(grep -m1 '^version' pyproject.toml)
echo "$version"
"""

UVLOCK = """\
[[package]]
name = "rcp-ndcg"
source = { editable = "." }

[[package]]
name = "rcp-ndcg-core"
source = { editable = "packages/rcp-ndcg-core" }
"""

AGENTS = """\
Layout: rcp_ndcg_core -> src/rcp_ndcg (imports point inward), tests under tests/.
"""

FIXTURE: dict[str, str] = {
    "pyproject.toml": PYPROJECT,
    "MANIFEST.in": MANIFEST,
    "README.md": "# Fixture\n\nThe long README.\n",
    "src/rcp_ndcg/__init__.py": '__version__ = "0.0.1"\n',
    "packages/rcp-ndcg-core/pyproject.toml": '[project]\nname = "rcp-ndcg-core"\nversion = "0.0.1"\n',
    "packages/rcp-ndcg-core/src/rcp_ndcg_core/__init__.py": '"""The core."""\n',
    "packages/rcp-ndcg-core/README.md": "# rcp-ndcg-core\n",
    "packages/rcp-ndcg-vllm/pyproject.toml": '[project]\nname = "rcp-ndcg-vllm"\nversion = "0.0.1"\n',
    "packages/rcp-ndcg-vllm/src/rcp_ndcg_vllm/__init__.py": '"""The serving dist."""\n',
    "packages/rcp-ndcg-vllm/jobs/build.sh": BUILD_SH,
    ".github/workflows/ci.yml": CI,
    "tests/snap/helpers.py": HELPERS,
    "uv.lock": UVLOCK,
    "AGENTS.md": AGENTS,
    "CHANGELOG.md": "## Unreleased\n- The code under src/rcp_ndcg moved.\n",
}


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return proc.stdout


def make_fixture(tmp_path: Path) -> Path:
    """A committed fixture repository and, beside it, a scratch worktree of its tip."""
    repo = tmp_path / "fixture"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    for name, text in FIXTURE.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=fixture", "-c", "user.email=f@example.invalid", "commit", "-qm", "fixture")
    worktree = tmp_path / "wt"
    git(repo, "worktree", "add", "-q", str(worktree), "HEAD")
    tools = worktree / "tools"
    tools.mkdir()
    shutil.copy(REAL_SCRIPT, tools / "layout_move.py")
    return worktree


def digest(root: Path) -> dict[str, str]:
    """Every file under ``root``, by content hash: what an idempotent second run must not move."""
    out: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts):
        out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def run_layout(worktree: Path, *args: str) -> str:
    proc = subprocess.run(
        [sys.executable, "tools/layout_move.py", *args], cwd=worktree, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise AssertionError(f"layout_move {' '.join(args)} exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def test_the_move_performs_every_move_and_rewrite(tmp_path: Path) -> None:
    worktree = make_fixture(tmp_path)
    run_layout(worktree)

    # The four destinations exist and carry what moved: the root package's files under rcp-ndcg/.
    assert (worktree / "rcp-ndcg" / "pyproject.toml").is_file()
    assert (worktree / "rcp-ndcg" / "MANIFEST.in").is_file()
    assert (worktree / "rcp-ndcg" / "README.md").read_text(encoding="utf-8").startswith("# Fixture")
    assert (worktree / "rcp-ndcg" / "src" / "rcp_ndcg" / "__init__.py").is_file()
    assert (worktree / "rcp-ndcg-core" / "src" / "rcp_ndcg_core" / "__init__.py").is_file()
    assert (worktree / "rcp-ndcg-vllm" / "jobs" / "build.sh").is_file()
    assert not (worktree / "packages").exists()
    assert not (worktree / "src").exists()

    # The pyproject split: rcp-ndcg/ carries the package tables, the root the workspace and tooling.
    package = (worktree / "rcp-ndcg" / "pyproject.toml").read_text(encoding="utf-8")
    workspace = (worktree / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project]" in package and "[tool.setuptools.packages.find]" in package
    assert "[tool.uv.workspace]" not in package and "[tool.ruff]" not in package
    assert "[project]" not in workspace
    assert 'members = ["rcp-ndcg", "rcp-ndcg-core", "rcp-ndcg-vllm"]' in workspace
    assert "[tool.uv.sources]" in workspace and "rcp-ndcg-core = { workspace = true }" in workspace
    assert 'include = ["rcp-ndcg/src", "rcp-ndcg-core/src", "rcp-ndcg-vllm/src"]' in workspace
    assert 'extend-exclude = ["rcp-ndcg-vllm/recipes/*/qwen3_vl_embedding.py"]' in workspace
    assert 'where = ["src"]' in package  # package-relative: unchanged

    # The rewrites reach CI, workflows, scripts, helpers, the lock and prose.
    ci = (worktree / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "rcp-ndcg-core/src/rcp_ndcg_core" in ci and "packages/" not in ci
    assert "rcp-ndcg/pyproject.toml" in ci  # the bare manifest token in a workflow is the root package's
    build = (worktree / "rcp-ndcg-vllm" / "jobs" / "build.sh").read_text(encoding="utf-8")
    assert "rcp-ndcg/pyproject.toml" in build and "rcp-ndcg-vllm/dist" not in build
    helpers = (worktree / "tests" / "snap" / "helpers.py").read_text(encoding="utf-8")
    assert 'ROOT / "rcp-ndcg" / "README.md"' in helpers
    assert 'ROOT / "rcp-ndcg" / "pyproject.toml"' in helpers
    assert 'ROOT / "rcp-ndcg-core" / "README.md"' in helpers  # the split Path spelling moved too
    assert 'parents[2] / "rcp-ndcg/src/rcp_ndcg"' in helpers  # the split Path spelling of its source
    assert 'ROOT / "rcp-ndcg" / "src"' in helpers  # the source-folder helper points at the moved source
    assert 'package / "pyproject.toml"' in helpers  # a tmp fixture's own manifest: untouched
    lock = (worktree / "uv.lock").read_text(encoding="utf-8")
    assert 'editable = "rcp-ndcg"' in lock and 'editable = "rcp-ndcg-core"' in lock
    assert 'editable = "."' not in lock
    changelog = (worktree / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "rcp-ndcg/src/rcp_ndcg moved" in changelog

    # Its own MANIFEST.in keeps the package-relative source spelling.
    assert "recursive-include src/rcp_ndcg *.py" in (worktree / "rcp-ndcg" / "MANIFEST.in").read_text(encoding="utf-8")

    # HISTORY KEPT: committing the staged result and following a moved file reaches the fixture commit.
    git(worktree, "-c", "user.name=fixture", "-c", "user.email=f@example.invalid", "commit", "-qam", "the move")
    history = git(worktree, "log", "--follow", "--oneline", "--", "rcp-ndcg/src/rcp_ndcg/__init__.py")
    assert "fixture" in history


def test_the_move_is_idempotent(tmp_path: Path) -> None:
    worktree = make_fixture(tmp_path)
    run_layout(worktree)
    before = digest(worktree)
    stdout = run_layout(worktree)
    assert "already in the target layout; nothing to do" in stdout
    assert digest(worktree) == before
    assert "layout_move --check: the tree is in the target layout" in run_layout(worktree, "--check")


def test_check_says_what_is_unmoved(tmp_path: Path) -> None:
    worktree = make_fixture(tmp_path)
    proc = subprocess.run(
        [sys.executable, "tools/layout_move.py", "--check"], cwd=worktree, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 1
    assert "is unmoved" in proc.stderr
    assert "pyproject.toml is unmoved" in proc.stderr


def test_an_optional_source_is_skipped_and_a_conflict_is_refused(tmp_path: Path) -> None:
    worktree = make_fixture(tmp_path)
    # No packages/rcp-ndcg-test in the fixture: the optional move is skipped, never an error.
    assert "skipped packages/rcp-ndcg-test (not in this tree)" in run_layout(worktree)

    # A tip that already carries the destination beside the source is refused, the conflict listed.
    second = make_fixture(tmp_path / "second")
    (second / "rcp-ndcg-core").mkdir()
    (second / "rcp-ndcg-core" / "stale").write_text("x", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "tools/layout_move.py"], cwd=second, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 1
    assert "packages/rcp-ndcg-core: destination rcp-ndcg-core already exists" in proc.stderr


def test_a_stale_old_path_left_behind_is_refused(tmp_path: Path) -> None:
    worktree = make_fixture(tmp_path)
    # `src/rcp_ndcg` is package-relative exactly once: in the root package's own MANIFEST.in. Any
    # other MANIFEST.in (core's, say) that spells it names the path the move took away -- the sweep
    # leaves a MANIFEST's source spelling alone, and the stale scan refuses the one that cannot be
    # right.
    stale = worktree / "packages" / "rcp-ndcg-core" / "MANIFEST.in"
    stale.write_text("recursive-include src/rcp_ndcg *.py\n", encoding="utf-8")
    git(worktree, "add", "packages/rcp-ndcg-core/MANIFEST.in")
    proc = subprocess.run(
        [sys.executable, "tools/layout_move.py"], cwd=worktree, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 1
    assert "stale match" in proc.stderr
    assert "MANIFEST.in" in proc.stderr


@pytest.mark.parametrize("source,destination", [("pyproject.toml", "rcp-ndcg/pyproject.toml"), ("src", "rcp-ndcg/src")])
def test_the_move_table_names_the_root_package(source: str, destination: str) -> None:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("layout_move_under_test", REAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (source, destination) in module.MOVES
    assert module.OPTIONAL_SOURCES == frozenset({"packages/rcp-ndcg-test"})
    assert module.EXEMPT == frozenset({"tools/layout_move.py", "tests/test_layout_move.py"})
