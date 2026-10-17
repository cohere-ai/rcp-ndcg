#!/usr/bin/env python3
"""Check a release tree against a version: the manifests, the citation and the pins (one home).

``release.yml``'s tag pipeline and ``rc_build.sh``'s candidate build both run this script, so a candidate
can never pass the node gates with a check the tag would fail (g-rcjobs-5).  It checks:

* every workspace manifest declares the version (the release builds three; a stale ``rcp-ndcg-test``
  leaves ``uv.lock`` and the stated rule disagreeing);
* ``CITATION.cff``'s ``version:`` is the version;
* ``rcp-ndcg`` pins ``rcp-ndcg-core==<version>`` exactly, in every dependency scope;
* ``rcp-ndcg-vllm`` names no workspace sibling at all (decision 18; owner decision 15 settled the tag
  pipeline on the documented rule: the lean package pins no sibling, matching what ``rc_build.sh``
  already enforced);
* each ``--wave`` document's recorded RC commit is the tag's commit (decision 16 / g-rcjobs-7: the
  release checklist passes the wave documents whose evidence flipped the statuses).

Usage::

    python .github/scripts/check_release.py <version> [--root DIR] [--wave WAVE.json ...] [--commit SHA]

Exit code 0: every check agrees. 1: the first disagreement is printed and the rest are not run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

__all__ = ["check_citation", "check_manifests", "check_pins", "check_wave_commits", "main"]

MANIFESTS = (
    "rcp-ndcg/pyproject.toml",
    "rcp-ndcg-core/pyproject.toml",
    "rcp-ndcg-vllm/pyproject.toml",
    "rcp-ndcg-test/pyproject.toml",
)
"""The four workspace members; all four carry the release version (AGENTS.md: versions move together)."""

WORKSPACE_SIBLINGS = {"rcp-ndcg", "rcp-ndcg-core", "rcp-ndcg-test"}
"""Every distribution of this workspace except ``rcp-ndcg-vllm`` itself."""

# name, optional [extras], then the specifier: "rcp-ndcg[calibrate]==0.0.1" -> ("rcp-ndcg", "==0.0.1")
_REQUIREMENT_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")


def _canonical(name: str) -> str:
    """PEP 503: lowercase, runs of ``-_.`` to one ``-``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared(project: dict) -> list[str]:
    """Every dependency declaration of one manifest, runtime and extras."""
    declared = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        declared += extra
    return declared


def _pins_of(specs: list[str], name: str) -> list[str]:
    """The specifiers ``specs`` declare for ``name``, PEP 508/503 read (never by matching TOML text)."""
    out = []
    for requirement in specs:
        match = _REQUIREMENT_RE.match(requirement.split(";", 1)[0])
        if match is not None and _canonical(match.group(1)) == _canonical(name):
            out.append(re.sub(r"[\s()]", "", match.group(2)))  # "( == 0.0.1 )" is the exact pin "==0.0.1"
    return out


def check_manifests(root: Path, version: str) -> None:
    """Every workspace manifest declares ``version``; raise ``SystemExit`` naming the first that does not."""
    for relative in MANIFESTS:
        manifest = root / relative
        if not manifest.is_file():
            raise SystemExit(f"{relative} is not in this tree: the workspace has four members")
        declared = tomllib.loads(manifest.read_text(encoding="utf-8"))["project"]["version"]
        if declared != version:
            raise SystemExit(f"{relative} declares {declared!r}, and the release version is {version!r}")
    print(f"all four workspace manifests carry {version}")


def check_citation(root: Path, version: str) -> None:
    """``CITATION.cff`` declares ``version``; raise ``SystemExit`` otherwise."""
    text = (root / "CITATION.cff").read_text(encoding="utf-8")
    found = re.search(r"^version:\s*(\S+)\s*$", text, re.M)
    if found is None:
        raise SystemExit("CITATION.cff declares no version: the citation is not tied to the release")
    if found.group(1) != version:
        raise SystemExit(f"CITATION.cff declares {found.group(1)!r}, and the release version is {version!r}")
    print(f"CITATION.cff version {version} matches")


def check_pins(root: Path, version: str) -> None:
    """The sibling pins: ``rcp-ndcg`` pins its core exactly; ``rcp-ndcg-vllm`` names no sibling at all.

    One rule for both pipelines (owner decision 15, g-rcjobs-5(d)): the documented rule governs -- the
    lean package pins no sibling -- so the tag pipeline refuses the lockstep pin exactly as ``rc_build.sh``
    and decision 18 already did.
    """
    core_manifest = root / "rcp-ndcg/pyproject.toml"
    if not core_manifest.is_file():
        raise SystemExit(f"{core_manifest} is not in this tree: it must pin rcp-ndcg-core=={version}")
    project = tomllib.loads(core_manifest.read_text(encoding="utf-8"))["project"]
    runtime = list(project.get("dependencies", []))
    declared = _declared(project)
    expected = f"=={version}"
    if not _pins_of(runtime, "rcp-ndcg-core"):
        raise SystemExit(
            f"rcp-ndcg/pyproject.toml does not declare rcp-ndcg-core: it must pin rcp-ndcg-core{expected} exactly"
        )
    pins = _pins_of(declared, "rcp-ndcg-core")
    if any(pin != expected for pin in pins):
        raise SystemExit(f"rcp-ndcg-core must be pinned {expected} exactly in rcp-ndcg/pyproject.toml; found {pins}")
    print(f"rcp-ndcg/pyproject.toml pins rcp-ndcg-core{expected}")

    vllm_manifest = root / "rcp-ndcg-vllm/pyproject.toml"
    if not vllm_manifest.is_file():
        raise SystemExit(f"{vllm_manifest} is not in this tree: the release builds it")
    project = tomllib.loads(vllm_manifest.read_text(encoding="utf-8"))["project"]
    declared = _declared(project)
    siblings = [
        requirement.strip()
        for requirement in declared
        if (match := _REQUIREMENT_RE.match(requirement.split(";", 1)[0])) is not None
        and _canonical(match.group(1)) in WORKSPACE_SIBLINGS
    ]
    if siblings:
        raise SystemExit(
            "rcp-ndcg-vllm/pyproject.toml must name no sibling package (decision 18: no lockstep "
            f"version pin between rcp-ndcg and rcp-ndcg-vllm), found {siblings}"
        )
    print("rcp-ndcg-vllm/pyproject.toml names no sibling package")


def check_wave_commits(root: Path, version: str, waves: Sequence[str], commit: str | None) -> None:
    """Every wave document's recorded RC commit is the tag's commit (decision 16 / g-rcjobs-7).

    ``commit`` defaults to the tree's own HEAD (``git rev-parse HEAD`` in ``--root``): the tag pipeline
    checks out the tag, so the recorded RC commit must equal it.  Each ``--wave`` document must carry the
    ``rc`` record the bootstrap writes (name, commit, version, manifest sha256); a document without it
    predates the recording and cannot tie the tag to the tested RC.
    """
    if not waves:
        return
    if commit is None:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
        commit = completed.stdout.strip()
        if completed.returncode != 0 or not commit:
            raise SystemExit(f"no --commit given and {root} is not a git checkout: pass --commit <sha>")
    for path in waves:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        rc = document.get("rc")
        required = ("name", "commit", "version", "manifest_sha256")
        if not isinstance(rc, dict) or not all(rc.get(field) for field in required):
            raise SystemExit(
                f"{path}: no rc record ({', '.join(required)}): the wave document cannot identify the tested "
                "RC, so the tag cannot be tied to it (decision 16)"
            )
        if rc["version"] != version:
            raise SystemExit(f"{path}: rc.version is {rc['version']!r}, and the release version is {version!r}")
        if rc["commit"] != commit:
            raise SystemExit(
                f"{path}: rc.commit is {rc['commit']!r}, and the tag's commit is {commit!r}: the tag must be "
                "the tested RC's commit (decision 16)"
            )
        print(f"{path}: tested RC {rc['name']} at {rc['commit']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("version", help="the release version (the tag without its leading v)")
    parser.add_argument("--root", default=".", help="the tree to check (default: the current directory)")
    parser.add_argument(
        "--wave",
        action="append",
        default=[],
        help="a wave.json whose rc record must name the tag's commit (decision 16); repeatable",
    )
    parser.add_argument(
        "--commit",
        default=None,
        help="the commit the tag names (default: git rev-parse HEAD in --root); compared with each wave's rc.commit",
    )
    args = parser.parse_args(argv)
    root = Path(args.root)
    check_manifests(root, args.version)
    check_citation(root, args.version)
    check_pins(root, args.version)
    check_wave_commits(root, args.version, args.wave, args.commit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
