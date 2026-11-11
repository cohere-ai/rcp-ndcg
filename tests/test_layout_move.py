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
from types import ModuleType

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

HARNESS: dict[str, str] = {
    "packages/rcp-ndcg-test/pyproject.toml": (
        '[project]\nname = "rcp-ndcg-test"\nversion = "0.0.1"\n\n'
        '[project.entry-points."rcp.fake_transports"]\n'
        'rcp-ndcg-test = "rcp_ndcg_test.engines:transport_for"\n'
    ),
    "packages/rcp-ndcg-test/src/rcp_ndcg_test/__init__.py": '"""The harness."""\n',
}


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True)
    return proc.stdout


def make_fixture(tmp_path: Path, *, harness: bool = False) -> Path:
    """A committed fixture repository and, beside it, a scratch worktree of its tip."""
    repo = tmp_path / "fixture"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    for name, text in {**FIXTURE, **(HARNESS if harness else {})}.items():
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

    # The four destinations exist and carry what moved: the root package's files under rcp/.
    assert (worktree / "rcp" / "pyproject.toml").is_file()
    assert (worktree / "rcp" / "MANIFEST.in").is_file()
    assert (worktree / "rcp" / "README.md").read_text(encoding="utf-8").startswith("# Fixture")
    assert (worktree / "rcp" / "src" / "rcp" / "__init__.py").is_file()
    assert (worktree / "rcp-core" / "src" / "rcp_core" / "__init__.py").is_file()
    assert (worktree / "rcp-vllm" / "jobs" / "build.sh").is_file()
    assert (worktree / "rcp-vllm" / "src" / "rcp_vllm" / "__init__.py").is_file()
    assert not (worktree / "packages").exists()
    assert not (worktree / "src").exists()

    # The pyproject split: rcp/ carries the package tables, the root the workspace and tooling.
    package = (worktree / "rcp" / "pyproject.toml").read_text(encoding="utf-8")
    workspace = (worktree / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project]" in package and "[tool.setuptools.packages.find]" in package
    assert "[tool.uv.workspace]" not in package and "[tool.ruff]" not in package
    assert "[project]" not in workspace
    assert 'members = ["rcp", "rcp-core", "rcp-vllm"]' in workspace
    assert "[tool.uv.sources]" in workspace and "rcp-core = { workspace = true }" in workspace
    assert 'include = ["rcp/src", "rcp-core/src", "rcp-vllm/src"]' in workspace
    assert 'extend-exclude = ["rcp-vllm/recipes/*/qwen3_vl_embedding.py"]' in workspace
    assert 'where = ["src"]' in package  # package-relative: unchanged

    # The rewrites reach CI, workflows, scripts, helpers, the lock and prose.
    ci = (worktree / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "rcp-core/src/rcp_core" in ci and "packages/" not in ci
    assert "rcp/pyproject.toml" in ci  # the bare manifest token in a workflow is the root package's
    build = (worktree / "rcp-vllm" / "jobs" / "build.sh").read_text(encoding="utf-8")
    assert "rcp/pyproject.toml" in build and "packages/" not in build
    helpers = (worktree / "tests" / "snap" / "helpers.py").read_text(encoding="utf-8")
    assert 'ROOT / "rcp" / "README.md"' in helpers
    assert 'ROOT / "rcp-core" / "README.md"' in helpers  # the split Path spelling moved too
    assert 'parents[2] / "rcp/src/rcp"' in helpers  # the split Path spelling of its source
    assert 'ROOT / "rcp" / "src"' in helpers  # the source-folder helper points at the moved source
    assert 'package / "pyproject.toml"' in helpers  # a tmp fixture's own manifest: untouched
    # the workspace manifest read stays at the root: in the target layout it is the workspace's, not
    # the root package's (the root-pyproject-helper only rewrites the pre-move package spellings)
    lock = (worktree / "uv.lock").read_text(encoding="utf-8")
    assert 'editable = "rcp"' in lock and 'editable = "rcp-core"' in lock
    assert 'editable = "."' not in lock
    changelog = (worktree / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "src/rcp_ndcg moved" in changelog  # the entries below Unreleased keep the spelling they shipped with

    # Its own MANIFEST.in keeps the package-relative source spelling.
    assert "recursive-include src/rcp *.py" in (worktree / "rcp" / "MANIFEST.in").read_text(encoding="utf-8")

    # HISTORY KEPT: committing the staged result and following a moved file reaches the fixture commit.
    git(worktree, "-c", "user.name=fixture", "-c", "user.email=f@example.invalid", "commit", "-qam", "the move")
    history = git(worktree, "log", "--follow", "--oneline", "--", "rcp/src/rcp/__init__.py")
    assert "fixture" in history


def test_the_harness_follows_the_distribution_rename(tmp_path: Path) -> None:
    """The harness leg of the rename: the unpublished distribution and its import package follow
    the others' (`packages/rcp-ndcg-test` -> `rcp-test`, `rcp_ndcg_test` -> `rcp_test`), the entry
    point name and target included, and the workspace the split regenerates lists the new member.
    """
    worktree = make_fixture(tmp_path, harness=True)
    run_layout(worktree)

    assert (worktree / "rcp-test" / "src" / "rcp_test" / "__init__.py").is_file()
    assert not (worktree / "rcp-ndcg-test").exists()
    manifest = (worktree / "rcp-test" / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "rcp-test"' in manifest
    assert "rcp-ndcg-test" not in manifest and "rcp_ndcg_test" not in manifest
    # the entry point follows: its name is the distribution's, its target the package's
    assert 'rcp-test = "rcp_test.engines:transport_for"' in manifest
    workspace = (worktree / "pyproject.toml").read_text(encoding="utf-8")
    assert 'members = ["rcp", "rcp-core", "rcp-test", "rcp-vllm"]' in workspace
    assert "rcp-test = { workspace = true }" in workspace
    # and the history followed the moves (commit the staged result first, like the performs test)
    git(
        worktree,
        "-c",
        "user.name=fixture",
        "-c",
        "user.email=f@example.invalid",
        "commit",
        "-qam",
        "the harness rename",
    )
    history = git(worktree, "log", "--follow", "--oneline", "--", "rcp-test/src/rcp_test/__init__.py")
    assert "fixture" in history
    # idempotent: the moved tree is already in the target layout
    assert "layout_move --check: the tree is in the target layout" in run_layout(worktree, "--check")


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
    (second / "rcp-core").mkdir()
    (second / "rcp-core" / "stale").write_text("x", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "tools/layout_move.py"], cwd=second, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 1
    assert "packages/rcp-ndcg-core: destination rcp-core already exists" in proc.stderr


def test_a_stale_old_path_left_behind_is_refused(tmp_path: Path) -> None:
    worktree = make_fixture(tmp_path)
    # `src/rcp` is package-relative exactly once: in the root package's own MANIFEST.in. Any other
    # MANIFEST.in (core's, say) that spells it names the path the move took away -- the sweep leaves
    # a MANIFEST's source spelling alone, and the stale scan refuses the one that cannot be right.
    stale = worktree / "packages" / "rcp-ndcg-core" / "MANIFEST.in"
    stale.write_text("recursive-include src/rcp *.py\n", encoding="utf-8")
    git(worktree, "add", "packages/rcp-ndcg-core/MANIFEST.in")
    proc = subprocess.run(
        [sys.executable, "tools/layout_move.py"], cwd=worktree, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 1
    assert "stale match" in proc.stderr
    assert "MANIFEST.in" in proc.stderr


def tool() -> ModuleType:
    """``tools/layout_move.py`` loaded by path: the transform is a script, not an installed module."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("layout_move_under_test", REAL_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_sweep_rewrites_the_module_and_leaves_the_metric_id() -> None:
    """After the rename `rcp_ndcg` is the metric's id far more often than the module's.

    The sweep rewrites the module in the contexts where the token is module syntax (a dotted
    reference, an import, a path segment) and leaves the id the metric's: ``Literal["rcp_ndcg", ...]``,
    ``--metrics rcp_ndcg``, ``report.value(..., "rcp_ndcg")``. Rewriting those is the regression this
    narrowing exists to prevent -- a correctness fix restored the id, and a bare-token rule undid it.
    """
    text = (
        'MetricName = Literal["rcp_ndcg", "qrel_ndcg"]\n'
        "from rcp_ndcg.eval import evaluate\n"
        "import rcp_ndcg\n"
        "import os, rcp_ndcg, rcp_ndcg.cli.main\n"
        "PUBLIC = rcp_ndcg.support.api_docs.PUBLIC_MODULES\n"
        'PROBE = "import rcp_ndcg"\n'
        'SOURCE = "rcp/src/rcp_ndcg"\n'
        'USAGE = ("--metrics", "rcp_ndcg")\n'
        'value = report.value("system", "rcp_ndcg", 10)\n'
    )
    rewritten, _hits = tool()._apply_rewrites(text, Path("rcp/src/rcp/module.py"))

    # The metric's id survives, in every context it is spelled in.
    assert 'MetricName = Literal["rcp_ndcg", "qrel_ndcg"]' in rewritten
    assert 'USAGE = ("--metrics", "rcp_ndcg")' in rewritten
    assert 'value = report.value("system", "rcp_ndcg", 10)' in rewritten
    # The module does not, in any of the forms that can only be the module.
    assert "from rcp.eval import evaluate" in rewritten
    assert "\nimport rcp\n" in rewritten
    assert "import os, rcp, rcp.cli.main" in rewritten
    assert "PUBLIC = rcp.support.api_docs.PUBLIC_MODULES" in rewritten
    assert 'PROBE = "import rcp"' in rewritten
    assert 'SOURCE = "rcp/src/rcp"' in rewritten


@pytest.mark.parametrize("source,destination", [("pyproject.toml", "rcp/pyproject.toml"), ("src", "rcp/src")])
def test_the_move_table_names_the_root_package(source: str, destination: str) -> None:
    module = tool()
    assert (source, destination) in module.MOVES
    assert module.OPTIONAL_SOURCES == frozenset(
        {"packages/rcp-ndcg-test", "rcp-ndcg-test", "rcp-test/src/rcp_ndcg_test"}
    )
    assert module.EXEMPT == frozenset(
        {
            # The transforms that must spell the old names: this one (its table is the move) and the
            # behaviour fingerprint's canonicalisation (its table maps a current spelling onto the
            # recorded one), each with the tests that assert the table.
            "tools/layout_move.py",
            "tests/test_layout_move.py",
            "rcp-test/src/rcp_test/fingerprint.py",
            "rcp-test/tests/test_fingerprint.py",
            "rcp-test/tests/test_wave_plugin_wheel.py",
        }
    ), "a new exemption is a transform's table, never a file that could be rewritten"


def test_the_checkout_is_in_the_target_layout() -> None:
    """``--check`` must pass on the real checkout: it is the tool's own claim that the move landed.

    The synthetic fixture cannot see a prose mention of a bare ``pyproject.toml`` in a workflow
    comment; the real tree carries one, and a rule that rewrites it would edit documentation.
    """
    proc = subprocess.run(
        [sys.executable, "tools/layout_move.py", "--check"],
        cwd=REAL_SCRIPT.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
