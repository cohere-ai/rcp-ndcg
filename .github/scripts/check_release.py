#!/usr/bin/env python3
"""Check a release tree against a version: the manifests, the citation and the pins (one home).

``release.yml``'s tag pipeline and ``rc_build.sh``'s candidate build both run this script, so a candidate
can never pass the node gates with a check the tag would fail (g-rcjobs-5).  It checks:

* every workspace manifest declares the version (the release builds three; a stale ``rcp-ndcg-test``
  leaves ``uv.lock`` and the stated rule disagreeing);
* ``CITATION.cff``'s ``version:`` is the version;
* ``rcp-ndcg`` pins ``rcp-ndcg-core==<version>`` exactly, in every dependency scope;
* ``rcp-ndcg-vllm`` either names no workspace sibling at all (``--refuse-vllm-sibling``: decision 18,
  what ``rc_build.sh`` runs) or, when it names ``rcp-ndcg``, pins it exactly (``release.yml``'s rule).

Usage::

    python .github/scripts/check_release.py <version> [--refuse-vllm-sibling] [--root DIR]

Exit code 0: every check agrees. 1: the first disagreement is printed and the rest are not run.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

__all__ = ["check_citation", "check_manifests", "check_pins", "main"]

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


def check_pins(root: Path, version: str, *, refuse_vllm_sibling: bool) -> None:
    """The sibling pins: ``rcp-ndcg`` pins its core exactly; ``rcp-ndcg-vllm``'s rule is the caller's.

    ``refuse_vllm_sibling`` is decision 18's rule (``rc_build.sh``): the lean package names no workspace
    sibling at all.  Without it, release.yml's rule applies: ``rcp-ndcg-vllm`` may name ``rcp-ndcg``, but
    only as the exact pin.  Which rule governs the tag pipeline is the owner's open decision (15).
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
    if refuse_vllm_sibling:
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
        return
    pins = _pins_of(declared, "rcp-ndcg")
    if not pins:
        print("rcp-ndcg-vllm/pyproject.toml does not depend on rcp-ndcg")
        return
    if any(pin != expected for pin in pins):
        raise SystemExit(f"rcp-ndcg must be pinned {expected} exactly in rcp-ndcg-vllm/pyproject.toml; found {pins}")
    print(f"rcp-ndcg-vllm/pyproject.toml pins rcp-ndcg{expected}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("version", help="the release version (the tag without its leading v)")
    parser.add_argument("--root", default=".", help="the tree to check (default: the current directory)")
    parser.add_argument(
        "--refuse-vllm-sibling",
        action="store_true",
        help="refuse any workspace sibling in rcp-ndcg-vllm's manifest (decision 18; rc_build.sh's rule)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root)
    check_manifests(root, args.version)
    check_citation(root, args.version)
    check_pins(root, args.version, refuse_vllm_sibling=args.refuse_vllm_sibling)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
