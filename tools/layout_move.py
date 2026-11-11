#!/usr/bin/env python3
"""The layout move as a re-runnable transform (drafts/layout-move.md item 1).

Moves this repository to the four-distribution tree:

    rcp-core/   (was packages/rcp-ndcg-core)
    rcp/             (was the root package: pyproject.toml, MANIFEST.in, README.md, src/)
    rcp-vllm/        (was packages/rcp-ndcg-vllm)
    rcp-test/        (was packages/rcp-ndcg-test, when that exists on the tip)
    pyproject.toml   the uv workspace only (members, tooling config)

Everything is rule-based -- a declared table of moves, of old-path -> new-path rewrites and of the
pyproject section split -- never a recorded patch, so the same script replays on a later tip in
minutes. It performs every ``git mv`` (history kept) and every path rewrite (CI, release.yml,
AGENTS.md, docs, mkdocs, tests and their path helpers, snapshots, constraints, MANIFEST.in, pyproject
workspace members, basedpyright/ruff includes, scripts). It prints every file it rewrote and refuses
(exit 1, listed) any old path left behind.

Usage::

    python tools/layout_move.py            # perform the move in this repository
    python tools/layout_move.py --check    # exit 0 iff the tree is in the target layout already

Idempotent: on an already-moved tree it changes nothing and says so. The two files of the transform
itself (this script and its test) declare an exemption from the rewrite sweep -- their old paths are
the declared table and its fixtures -- and are the only files allowed to spell them.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

__all__ = ["MOVES", "REWRITES", "STALE", "main", "run"]

MOVES: tuple[tuple[str, str], ...] = (
    # The root package into rcp/ (the files its distribution needs beside src/).
    ("pyproject.toml", "rcp/pyproject.toml"),
    ("MANIFEST.in", "rcp/MANIFEST.in"),
    ("README.md", "rcp/README.md"),
    ("src", "rcp/src"),
    # The package directories one level up. An optional source (a package a tip does not carry)
    # is skipped; a source next to an existing destination is refused.
    ("packages/rcp-ndcg-core", "rcp-core"),
    # The core's import package follows its distribution's name (`rcp_ndcg_core` -> `rcp_core`).
    ("rcp-core/src/rcp_ndcg_core", "rcp-core/src/rcp_core"),
    # So does the main package's (`rcp_ndcg` -> `rcp`), once the root source is in its new home.
    ("rcp/src/rcp_ndcg", "rcp/src/rcp"),
    ("packages/rcp-ndcg-vllm", "rcp-vllm"),
    # So does the serving package's (`rcp_ndcg_vllm` -> `rcp_vllm`).
    ("rcp-vllm/src/rcp_ndcg_vllm", "rcp-vllm/src/rcp_vllm"),
    ("packages/rcp-ndcg-test", "rcp-ndcg-test"),
    # So does the harness (`rcp_ndcg_test` -> `rcp_test`): the unpublished test distribution moves
    # once more (the rows above land it at the -ndcg spelling first), then its import package
    # follows its distribution's name like the others'.
    ("rcp-ndcg-test", "rcp-test"),
    ("rcp-test/src/rcp_ndcg_test", "rcp-test/src/rcp_test"),
)

OPTIONAL_SOURCES: frozenset[str] = frozenset({"packages/rcp-ndcg-test", "rcp-ndcg-test", "rcp-test/src/rcp_ndcg_test"})
"""A move source this tree is allowed not to carry (the tip predates the package, or a rename leg of
the harness above already happened)."""

LANDED_CARD_SOURCES: frozenset[str] = frozenset({"README.md"})
"""A move source the landing-card design replaced (docs-release Q1): the full README moved into
rcp/, and a NEW root README -- the short landing card -- was added at the old path on purpose.
Source beside destination is the target state for these, never a conflict."""

SPLIT_REGENERATED: frozenset[str] = frozenset({"pyproject.toml"})
"""A move source the pyproject split writes back as the workspace manifest: with the destination in
place and the source carrying no [project] table, the move is done and the pair is no conflict."""

EXEMPT: frozenset[str] = frozenset(
    {
        "tools/layout_move.py",
        "tests/test_layout_move.py",
        # The fingerprint canonicalisation is the other transform that must spell the old names: its
        # table maps them onto the recorded spelling, and its tests assert the mapping.
        "rcp-test/src/rcp_test/fingerprint.py",
        "rcp-test/tests/test_fingerprint.py",
        "rcp-test/tests/test_wave_plugin_wheel.py",
    }
)
"""Files the rewrite sweep and the stale scan skip: this transform and its test declare the old
paths in their tables and fixtures, and so does the behaviour fingerprint's canonicalisation, whose
whole purpose is to map a current spelling onto the recorded one."""

EXEMPT_PREFIXES: tuple[str, ...] = (
    "handover/",
    "rcp-test/corpora/",
    "rcp-test/pairs/",
    "rcp-test/src/rcp_test/fixtures/cases/",
    "rcp-test/tests/recipes/golden/",
)
"""Path prefixes the sweep and the stale scan skip: handover/ is temporary scaffolding (deleted in one
commit before the release), and its workstream reports describe the tree as it was when written; the
observation corpora -- the recipe keyings and the pairs corpus -- are the GPU waves' records, keyed by
the tree that recorded them (PLAN section 8); a case's expected values are derived from its input
texts, so a renamed text there re-records evidence; and a recipe golden is a recorded keying too -- its
``fingerprint.inputs`` name the plugin's modules in the spelling the wave recorded, which is why the
fingerprint canonicalises that spelling instead of the golden being rewritten. The engine-side plugin
modules under models/ and patches/ were exempt while their bytes were a fingerprint input a layout move
had no reason to touch; the rename moves the module names themselves, so their re-record is declared
with the rest of it."""

EXEMPT_SUFFIXES: tuple[str, ...] = (".jinja",)
"""File suffixes the sweep and the stale scan skip: a recipe template's bytes are a behaviour-fingerprint
input (`template_file`), so renaming inside one would cost a GPU re-record."""

# The rewrite rules, applied in order to the text of every tracked file outside EXEMPT. A rule is
# (name, pattern, replacement, files): ``files`` restricts the rule to paths matching any of its
# globs (None: every file). Patterns match old paths only; where a spelling is ambiguous (a bare
# ``pyproject.toml`` names every package's manifest; the bare ``rcp_ndcg`` is the metric's id far
# more often than the module) the rule is scoped to the contexts where the token is executable path
# syntax, and the bare spelling in prose is left to the owning page.
REWRITES: tuple[tuple[str, str, str, tuple[str, ...] | None], ...] = (
    # The rename on top of the move: the distributions' and import names drop `ndcg`
    # (`rcp-ndcg-core` -> `rcp-core`, `rcp_ndcg_core` -> `rcp_core`, `rcp-ndcg` -> `rcp`,
    # `rcp_ndcg` -> `rcp`, `rcp-ndcg-vllm` -> `rcp-vllm`, `rcp_ndcg_vllm` -> `rcp_vllm`,
    # `rcp-ndcg-test` -> `rcp-test`, `rcp_ndcg_test` -> `rcp_test`). The bare names are names, not
    # paths, so they rewrite wherever they are live -- the CHANGELOG's entries below Unreleased and
    # the ledger's historical source list keep the spellings they shipped with (skipped below).
    ("core-name", r"(?<![\w-])rcp-ndcg-core\b", "rcp-core", None),
    ("core-module", r"(?<![\w])rcp_ndcg_core\b", "rcp_core", None),
    # The main distribution's own scratch names carry a suffix the generic rule must not touch.
    ("main-scratch", r"rcp-ndcg-(uv|tests-|validate-)", r"rcp-\1", None),
    # The release pipeline's job and artifact names are built from the distribution's name.
    ("main-artifacts", r"(?<![\w-])(dist|publish)-rcp-ndcg(?![\w-])", r"\1-rcp", None),
    # The main name, whole-token: never a sibling distribution (`-vllm`, `-test`), never an artifact
    # tag or a `$id` (`rcp-ndcg.<name>.v1`), never the extension keys (`x-rcp-ndcg-*`), never the
    # public dataset ids (`rcp-ndcg-nanobeir`, ...) or the bucket (`rcp-ndcg-rfc`), and never the
    # repository URL (`cohere-ai/rcp-ndcg`), which is the operator's and redirects.
    ("main-name", r"(?<![\w./-])rcp-ndcg(?![-.\w])", "rcp", None),
    # The main module, in the contexts where the token can only be it. After the rename the bare
    # spelling is the metric's id far more often than the module's (`Literal["rcp_ndcg", ...]`,
    # `METRICS`, `--metrics rcp_ndcg`), so it is left to the owning page; the rules below are the
    # module syntax, and each says which form it is.
    #
    # A dotted reference: `rcp_ndcg.eval`, `rcp_ndcg.support.api_docs`, `rcp_ndcg.cli.main`. A
    # JSON or Python string escape is a word boundary, not an identifier (`\nrcp_ndcg.eval` is the
    # module, `delta_rcp_ndcg` and `rcp_ndcg10` are the metric's).
    ("main-dotted", r"(?:(?<!\w)|(?<=\\[ntr]))rcp_ndcg(?=\.)", "rcp", None),
    # An import: `import rcp_ndcg`, `import os, rcp_ndcg, ...`, `from rcp_ndcg import ...`.
    ("main-import", r"(\bimport\s+(?:[\w.]+\s*,\s*)*)rcp_ndcg(?![\w])", r"\1rcp", None),
    ("main-from", r"(\bfrom\s+)rcp_ndcg(?![\w])", r"\1rcp", None),
    # A path segment: the package's directory under a `src/` tree (`src/rcp_ndcg`,
    # `rcp/src/rcp_ndcg`), and the package's own directory, repo-relative (`rcp_ndcg/data/`,
    # `rcp_ndcg/pyproject.toml`). The metric's id follows no `src/` and is followed by no `/`.
    ("main-path", r"(\bsrc/)rcp_ndcg(?![\w])", r"\1rcp", None),
    ("main-path-dir", r"(?<![\w])rcp_ndcg(?=/)", "rcp", None),
    # The same path segment where a test spells the join as a Path division (`"src" / "rcp_ndcg"`):
    # the segment after a `/` operator is a directory, never a metric.
    ("main-path-split", r"""(/\s*)(["'])rcp_ndcg\2""", r"\1\2rcp\2", None),
    # The two config keys that name the module in quotes: ruff's first-party classification and
    # mkdocs' nav entry. Everywhere else a quoted bare name is the metric's id, so these two declared
    # positions are rules and the rest are not.
    ("main-isort", r'(known-first-party = \[)(")rcp_ndcg(")', r"\1\2rcp\3", ("pyproject.toml",)),
    ("main-nav", r'(")rcp_ndcg(":)', r"\1rcp\2", ("mkdocs.yml",)),
    # The serving distribution and its import package (`rcp-ndcg-vllm` -> `rcp-vllm`,
    # `rcp_ndcg_vllm` -> `rcp_vllm`). `serve.plugin` and the plugin's module names are fingerprint
    # inputs, but the fingerprint keys them in the recorded spelling (`rcp_test.fingerprint`), so the
    # rename moves no recorded key: no plugin recipe's corpus is stale and no wave re-runs for it.
    ("vllm-name", r"(?<![\w-])rcp-ndcg-vllm\b", "rcp-vllm", None),
    ("vllm-module", r"(?<!\w)rcp_ndcg_vllm\b", "rcp_vllm", None),
    # The unpublished harness's distribution and import names drop `ndcg`. `test-name`'s lookbehind
    # keeps `x-`-style prefixes and sibling distributions out, like `main-name`'s; `test-module`
    # matches the same escape boundary as `main-dotted` (`\nrcp_ndcg_test` in a record is the
    # module, never a fragment of another identifier).
    ("test-name", r"(?<![\w-])rcp-ndcg-test\b", "rcp-test", None),
    ("test-module", r"(?:(?<!\w)|(?<=\\[ntr]))rcp_ndcg_test(?!\w)", "rcp_test", None),
    # The package directories, in one string or split the way a test spells a Path. The core-name
    # rule above has already mapped the core's directory to `rcp-core`; `\1` keeps the others.
    ("packages", r"packages/(rcp-(?:ndcg-)?(?:core|vllm|test))\b", r"\1", None),
    ("packages-split", r"""(["'])packages\1\s*/\s*(['"])(rcp-(?:ndcg-)?(?:core|vllm|test))\2""", r"\2\3\2", None),
    # The root package's source: `src/rcp` (never `src/rcp_*` -- \b does not match before _), in one
    # string or split the way a test spells a Path. A MANIFEST.in spells its own source relative to
    # its directory and keeps it.
    ("root-src", r"(?<![\w./-])src/rcp(?![\w-])", "rcp/src/rcp", None),
    (
        "root-src-split",
        r"""(["'])src\1\s*/\s*(['"])rcp\2""",
        r"\1rcp/src/rcp\1",
        None,
    ),
    # A bare `pyproject.toml` in a script or workflow is executable path syntax: the one root-level
    # manifest is now rcp/pyproject.toml. (`x/pyproject.toml` is a package's own manifest.)
    ("bare-pyproject", r"(?<![\w./-])pyproject\.toml\b", "rcp/pyproject.toml", (".github/**", "**/*.sh")),
    # The root-file helpers in the tests point at the moved distribution files (the long README and
    # the package manifest are PyPI's now).
    ("root-readme-helper", r'(?<![\w./-])ROOT / "README\.md"', 'ROOT / "rcp" / "README.md"', None),
    ("root-readme-helper-repo", r'(?<![\w./-])REPO / "README\.md"', 'REPO / "rcp" / "README.md"', None),
    (
        "root-pyproject-helper",
        r'(?<![\w./-])(ROOT|REPO) / "pyproject\.toml"',
        r'\1 / "rcp" / "pyproject.toml"',
        ("tests/docs/**",),
    ),
    ("root-src-helper", r'(?<![\w./-])(ROOT|REPO) / "src"(?![\w.-])', r'\1 / "rcp" / "src"', ("tests/**",)),
    ("readme-pages", r'\["README\.md", "docs/', '["rcp/README.md", "docs/', ("tests/**",)),
    (
        "source-folders",
        r'\("src", "packages", "experiments"',
        '("rcp/src", "rcp-core", "rcp-vllm", "experiments"',
        ("tests/**",),
    ),
    # basedpyright's include lists the root package's source as a bare "src".
    ("basedpyright-include", r'include = \["src", ', 'include = ["rcp/src", ', ("*.toml",)),
    # The lock records every workspace member by editable path; the root package's was ".".
    ("lock-editable", r'editable = "\."', 'editable = "rcp"', ("uv.lock",)),
    # The build directories follow their distribution: the one glob that named the old packages/
    # tree names the four distributions (the main one carries no hyphen, so the glob cannot require
    # one; `/rcp*/build/` is a superset of `/rcp-*/build/`).
    ("gitignore-build", r"/packages/\*/build/", "/rcp*/build/", (".gitignore",)),
)

STALE: tuple[tuple[str, ...], ...] = (
    # Any old package path left anywhere, `packages/rcp-...` split or together.
    (r"packages/(rcp-(?:ndcg-)?(?:core|vllm|test))",),
    (r"""(["'])packages\1\s*/\s*['"]rcp-(?:ndcg-)?(?:core|vllm|test)""",),
    # The root package's source outside its one new home (its MANIFEST.in spells it relative).
    (r"(?<![\w./-])src/rcp(?![\w-])", "rcp/MANIFEST.in", "MANIFEST.in", "tests/docs/test_packaging.py"),
    (r"""(["'])src\1\s*/\s*['"]rcp["']""",),
    # The harness's distribution and import names, where they were live. The ledger's historical
    # source list, the shipped-entry homes the changelog test pins and the CHANGELOG's past entries
    # keep theirs; the corpora, the pairs corpus and the case fixtures keep theirs under EXEMPT_PREFIXES.
    (
        r"(?<![\w-])rcp-ndcg-test\b",
        "CHANGELOG.md",
        "tests/docs/test_changelog.py",
        "rcp-test/src/rcp_test/jobs/ledger.py",
        "rcp-test/tests/test_ledger.py",
    ),
    (
        r"(?:(?<!\w)|(?<=\\[ntr]))rcp_ndcg_test(?!\w)",
        "CHANGELOG.md",
        "tests/docs/test_changelog.py",
        "rcp-test/src/rcp_test/jobs/ledger.py",
        "rcp-test/tests/test_ledger.py",
    ),
)
"""What must be gone after the sweep: patterns that can only mean a path the move took away. Each
entry is (pattern, exempt paths...).  A bare ``pyproject.toml`` in prose (CHANGELOG history, a
docstring) is not one; the executable-context rewrite and the helpers above cover every such path."""


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)


def _tracked_files(root: Path) -> list[Path]:
    proc = _git(root, "ls-files", "-z")
    if proc.returncode != 0:
        raise SystemExit(f"git ls-files failed: {proc.stderr.strip()}")
    return [Path(name) for name in proc.stdout.split("\0") if name]


def _is_text(path: Path) -> bool:
    try:
        return b"\0" not in path.read_bytes()[:8192]
    except OSError:
        return False


def _matches_files(name: Path, globs: tuple[str, ...] | None) -> bool:
    """Whether ``name`` falls under one of the rule's file globs (a bare name matches at any depth)."""
    if globs is None:
        return True
    text = name.as_posix()
    return any(fnmatch.fnmatch(text, glob) or fnmatch.fnmatch(text, f"**/{glob}") for glob in globs)


def _subn_outside_comments(text: str, pattern: str, replacement: str) -> tuple[str, int]:
    """``re.subn`` over the lines that are not comments: prose is not executable path syntax.

    The bare-``pyproject.toml`` rule is scoped to workflows and scripts; a ``#`` comment there may
    name the manifest in prose ("all four pyproject.toml files"), and rewriting that edits
    documentation instead of a path.
    """
    lines = text.splitlines(keepends=True)
    count = 0
    for index, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        lines[index], hits = re.subn(pattern, replacement, line)
        count += hits
    return "".join(lines), count


def _apply_rewrites(text: str, name: Path) -> tuple[str, list[str]]:
    hits: list[str] = []
    for rule, pattern, replacement, files in REWRITES:
        if not _matches_files(name, files):
            continue
        if rule.startswith(("core-", "main-", "vllm-", "test-")) and name.as_posix() == "CHANGELOG.md":
            continue  # the entries below Unreleased are history: they keep the names they shipped with
        if rule in (
            "core-name",
            "main-name",
            "test-name",
            "test-module",
            "vllm-name",
            "vllm-module",
        ) and name.as_posix() in ("rcp-test/src/rcp_test/jobs/ledger.py", "rcp-test/tests/test_ledger.py"):
            continue  # both spell pre-rename trees' names as data: the source list, and the test that pins it
        if rule == "main-name" and name.as_posix() == "tests/judging/test_store.py":
            continue  # its parametrize list spells both identity tags as data: a pre-rename store carries the old one
        if rule.startswith("test-") and name.as_posix() == "tests/docs/test_changelog.py":
            continue  # it pins the shipped 0.0.1 entry's harness homes as names, never live paths
        if rule == "root-src" and name.name == "MANIFEST.in":
            continue  # `recursive-include src/rcp ...` is relative to its own directory
        if rule == "root-src" and name.as_posix() == "tests/docs/test_packaging.py":
            continue  # its NOTICE-path pattern lists the old spelling as data to recognise, never a live path
        if rule == "bare-pyproject":
            new, count = _subn_outside_comments(text, pattern, replacement)
        else:
            new, count = re.subn(pattern, replacement, text)
        if count:
            hits.append(f"{rule} x{count}")
            text = new
    return text, hits


def _drop_empty_parents(root: Path, source: str) -> None:
    """Remove the directories a move emptied (``packages/``), deepest first, only while literally empty."""
    parent = (root / source).parent
    while parent != root:
        if not parent.is_dir() or any(parent.iterdir()):
            return
        parent.rmdir()
        parent = parent.parent


def run(root: Path, *, check: bool = False) -> int:
    """Perform (or, with ``check``, verify) the layout move in ``root``.

    Inputs: the repository root.  Outputs: the exit status -- 0 when the tree is (or was brought) in
    the target layout with no old path left behind, 1 when an old path is left behind or a move
    conflicts; prints every move and every file it rewrote either way.
    """
    problems: list[str] = []
    done_work = False
    moved: list[str] = []
    already: list[str] = []
    skipped: list[str] = []

    # 1. The moves (git mv keeps the history).
    for source, destination in MOVES:
        src, dst = root / source, root / destination
        if not src.exists() and not dst.exists():
            if source in OPTIONAL_SOURCES:
                skipped.append(f"{source} (not in this tree)")
            else:
                problems.append(f"{source}: neither {source} nor {destination} exists")
            continue
        if not src.exists() and dst.exists():
            already.append(f"{source} -> {destination}")
            continue
        if dst.exists():
            if source in SPLIT_REGENERATED and not re.search(
                r"^\[project\]\s*$", src.read_text(encoding="utf-8"), re.M
            ):
                already.append(f"{source} -> {destination}")
                continue
            if source in LANDED_CARD_SOURCES:
                already.append(f"{source} -> {destination} (the root file is the landing card)")
                continue
            problems.append(f"{source}: destination {destination} already exists")
            continue
        if check:
            problems.append(f"{source} is unmoved (would move to {destination})")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        proc = _git(root, "mv", source, destination)
        if proc.returncode != 0:
            problems.append(f"git mv {source} {destination} failed: {proc.stderr.strip()}")
            continue
        done_work = True
        moved.append(f"{source} -> {destination}")
        if source.startswith("packages/"):
            # The `packages/` level empties; `rcp-core/src` is the target layout's own directory.
            _drop_empty_parents(root, source)

    # 2. The rewrite sweep over every tracked text file outside EXEMPT (before the split, so the
    #    moved manifest's kept sections are already clean of old paths when they reach the root).
    rewritten: list[str] = []
    touched: list[str] = []
    for name in _tracked_files(root):
        path = root / name
        if name.as_posix() in EXEMPT or name.as_posix().startswith(EXEMPT_PREFIXES):
            continue
        if name.suffix in EXEMPT_SUFFIXES:
            continue
        if not path.is_file() or not _is_text(path):
            continue
        text = path.read_text(encoding="utf-8")
        new, hits = _apply_rewrites(text, name)
        if new != text:
            if check:
                problems.append(f"{name.as_posix()}: {', '.join(hits)}")
                continue
            path.write_text(new, encoding="utf-8")
            done_work = True
            rewritten.append(f"{name.as_posix()}: {', '.join(hits)}")
            touched.append(name.as_posix())

    # 3. The root pyproject's split: its package tables travel with the root package into
    #    rcp/pyproject.toml; the workspace and tooling tables stay at the root, and the
    #    workspace tables are regenerated from the move table's destinations that exist.
    package_manifest = root / "rcp" / "pyproject.toml"
    workspace_manifest = root / "pyproject.toml"
    if package_manifest.is_file():
        workspace_text = workspace_manifest.read_text(encoding="utf-8") if workspace_manifest.is_file() else ""
        if re.search(r"^\[project\]\s*$", workspace_text, re.M) or not workspace_manifest.is_file():
            if check:
                problems.append("the root pyproject is not split (its package tables belong in rcp/)")
            else:
                package_text, workspace_text = _split_pyproject(
                    package_manifest.read_text(encoding="utf-8"), members=_members(root)
                )
                package_manifest.write_text(package_text, encoding="utf-8")
                workspace_manifest.write_text(workspace_text, encoding="utf-8")
                touched.extend(["pyproject.toml", "rcp/pyproject.toml"])
                done_work = True
                print("rewrote pyproject.toml: split (package -> rcp/pyproject.toml, workspace stays)")

    # 4. Stage only what the tool moved or rewrote, never the whole worktree: `git add -A` would
    #    stage an unrelated contributor's untracked files alongside the move.
    if not check:
        touched.extend(destination for _source, destination in MOVES if (root / destination).exists())
        if touched:
            staged = _git(root, "add", "--", *sorted(set(touched)))
            if staged.returncode != 0:
                print(f"layout_move: git add failed: {staged.stderr.strip()}", file=sys.stderr)
                return 1

    # 5. Nothing the move took away may survive anywhere it may not.
    for entry in STALE:
        pattern, *exempt = entry
        for name in _tracked_files(root):
            if name.as_posix() in exempt or name.as_posix() in EXEMPT:
                continue
            if name.as_posix().startswith(EXEMPT_PREFIXES) or name.suffix in EXEMPT_SUFFIXES:
                continue
            path = root / name
            if not path.is_file() or not _is_text(path):
                continue
            found = sum(1 for _ in re.finditer(pattern, path.read_text(encoding="utf-8")))
            if found:
                problems.append(f"{name.as_posix()}: {found} stale match(es) of {pattern!r}")

    # Report.
    for line in moved:
        print(f"moved   {line}")
    for line in already:
        print(f"moved   {line} (already done)")
    for line in skipped:
        print(f"skipped {line}")
    for line in rewritten:
        print(f"rewrote {line}")
    if problems:
        print("layout_move: refusals:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    if check:
        print("layout_move --check: the tree is in the target layout, no old path anywhere")
        return 0
    if done_work:
        total = len(moved) + len(rewritten)
        print(f"layout_move: done ({len(moved)} moves, {len(rewritten)} files rewritten, {total} changes)")
        print("layout_move: staged; commit the result")
    else:
        print("layout_move: already in the target layout; nothing to do")
    return 0


def _members(root: Path) -> list[str]:
    """The workspace members: the top-level directories the move map brings into existence, when the
    directory carries a pyproject.toml (sorted)."""
    members = set()
    for source, destination in MOVES:
        for name in (source, destination):
            top = name.split("/", 1)[0]
            if top and (root / top / "pyproject.toml").is_file():
                members.add(top)
    return sorted(members)


_PACKAGE_TABLES = ("build-system", "project", "tool.setuptools")
_WORKSPACE_TABLES = ("tool.uv", "tool.pytest", "tool.ruff", "tool.basedpyright", "dependency-groups")

_WORKSPACE_HEADER = """\
# The uv workspace root: the workspace members and the tooling config only. The distributions carry
# their own manifests (rcp/pyproject.toml and the project directories beside it); tools/
# layout_move.py keeps this split and the member list current.
"""


def _split_pyproject(text: str, *, members: list[str]) -> tuple[str, str]:
    """Partition the root manifest text into the package manifest and the workspace manifest.

    Top-level tables are assigned by name: ``build-system``, ``project*`` and ``tool.setuptools*``
    build the distribution; ``tool.uv*``, ``tool.pytest*``, ``tool.ruff*`` and ``tool.basedpyright*``
    are the workspace's tooling config. The ``tool.uv.workspace`` / ``tool.uv.sources`` tables are
    regenerated from the members (every destination of the move table this tree carries). A table
    neither side claims is refused, never dropped.
    """
    header, tables, order = _sections(text)
    package_parts: list[str] = []
    workspace_parts: list[str] = []
    for name in order:
        body = tables[name]
        if any(name.startswith(prefix) for prefix in _WORKSPACE_TABLES):
            if name in ("tool.uv.workspace", "tool.uv.sources", "tool.ruff.lint.isort"):
                continue  # regenerated below
            workspace_parts.append(body)
        elif any(name.startswith(prefix) for prefix in _PACKAGE_TABLES):
            package_parts.append(body)
        else:
            raise SystemExit(f"layout_move: pyproject table [{name}] is claimed by neither side; extend the split")
    package = header + "\n\n".join(part.strip("\n") for part in package_parts) + "\n"
    pieces = [_WORKSPACE_HEADER.strip("\n"), _workspace_tables(members).strip("\n")]
    pieces += [part.strip("\n") for part in workspace_parts]
    workspace = "\n\n".join(pieces) + "\n"
    return package, workspace


def _sections(text: str) -> tuple[str, dict[str, str], list[str]]:
    """Split ``text`` at top-level ``[table]`` headers: the pre-table header, each table's text and
    the tables' order of appearance."""
    header_lines: list[str] = []
    tables: dict[str, str] = {}
    order: list[str] = []

    def close(name: str | None, buffer: list[str]) -> None:
        if name is None:
            header_lines.extend(buffer)
        elif name in tables:
            tables[name] += "".join(buffer)  # a sub-table repeated: keep appending to its section
        else:
            tables[name] = "".join(buffer)
            order.append(name)

    current: str | None = None
    buffer: list[str] = []
    for line in text.splitlines(keepends=True):
        match = re.match(r"^\[([^\]]+)\]\s*$", line)
        if match:
            close(current, buffer)
            current = match.group(1)
            buffer = [line]
        else:
            buffer.append(line)
    close(current, buffer)
    return "".join(header_lines), tables, order


def _workspace_tables(members: list[str]) -> str:
    """The ``tool.uv.workspace`` / ``tool.uv.sources`` / ``tool.ruff.lint.isort`` tables: the members
    (the package name of a member directory is its directory name) and the imports classification the
    move forces (``src/rcp`` was first-party in the old src layout by ruff's inference; the
    sibling tops never were -- the pin keeps every import block byte-for-byte as before)."""
    listed = ", ".join(f'"{member}"' for member in members)
    sources = "\n".join(f"{member} = {{ workspace = true }}" for member in members)
    return f"""\
[tool.uv.workspace]
members = [{listed}]

[tool.uv.sources]
# An inter-member requirement resolves to the checkout, never to an index.
{sources}

[tool.ruff.lint.isort]
known-first-party = ["rcp"]
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="layout_move", description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify the tree only; change nothing")
    parser.add_argument("--repo", type=Path, default=None, help="the repository to transform (default: this one)")
    args = parser.parse_args(argv)
    root = (args.repo or Path(__file__).resolve().parents[1]).resolve()
    if not (root / ".git").exists():
        print(f"layout_move: {root} is not a git work tree", file=sys.stderr)
        return 2
    return run(root, check=args.check)


if __name__ == "__main__":
    sys.exit(main())
