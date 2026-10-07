#!/usr/bin/env python3
"""Complete the reference venv's own distributions' missing dependencies, to a fixed point.

The bootstrap installs the recipes' reference requirements with ``--no-deps`` (the reference must never
resolve or replace the image's torch stack: its dependency tree is not registered in the image, and
the wheelhouse's CPU torch must never land over the image's CUDA build).  What ``--no-deps``
therefore cannot pull -- the reference venv's OWN distributions' declared dependencies
(sentence-transformers needs scikit-learn, scipy, joblib, threadpoolctl) -- is completed here, each
missing requirement installed ``--no-deps`` from the staged wheelhouse only, to a fixed point.

The image's own distributions are never completed -- completing them would shadow the image's CUDA
stack -- and an ``extra ==`` marker means nothing asked for that requirement.  Runs in the reference
venv's python (stdlib only):

    reference_deps.py <WHEELHOUSE>

Exits 0 at the fixed point (what it installed is reported on stderr); exits 1 with the requirement
names and the way out when the wheelhouse cannot satisfy a missing dependency.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path

__all__ = ["canonical", "main", "plan_more", "requirement_name"]

_MAX_ROUNDS = 20
"""A completion round installs the still-missing owners' needs; runaway loops fail loudly instead."""


def canonical(name: str) -> str:
    """PEP 503's canonical distribution name (``re.sub(r\"[-_.]+\", \"-\", name).lower()``)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    """The canonical distribution name a requirement string names (``name[extras] <spec> ; marker``)."""
    head = requirement.partition(";")[0].strip()
    head = head.split("[", 1)[0]
    head = re.split(r"[<>=!~ (]", head, maxsplit=1)[0]
    return canonical(head)


def plan_more(owned: dict[str, list[str]], visible: set[str], scheduled: set[str]) -> list[str]:
    """The requirement strings to install next, one round of the fixed point.

    Inputs: ``owned`` maps each distribution the reference venv OWNS (installed into it, not the
    image's) to its ``Requires-Dist`` strings; ``visible`` is every distribution this venv can see
    (image included: a dependency they cover is satisfied and their own needs are never completed);
    ``scheduled`` is what an earlier round already installed.  Output: the missing requirement
    strings (markers stripped) of the OWNED distributions, deduplicated in sorted-owner order; a
    requirement whose marker mentions ``extra`` is skipped (nothing asks for extras).  Units: none.
    """
    planned: list[str] = []
    seen: set[str] = set()
    for name in sorted(owned):
        for requirement in owned[name]:
            if not requirement.strip() or _requests_an_extra(requirement):
                continue
            dep = requirement_name(requirement)
            if not dep or dep in visible or dep in scheduled:
                continue
            spec = requirement.partition(";")[0].strip()
            if spec not in seen:
                seen.add(spec)
                planned.append(spec)
    return planned


def _requests_an_extra(requirement: str) -> bool:
    """``name; extra == "x"`` means nothing asked for the ``x`` extra."""
    _, sep, marker = requirement.partition(";")
    return bool(sep) and "extra" in marker


def _owned_and_visible() -> tuple[dict[str, list[str]], set[str]]:
    """What is installed in this venv (owned, with its declared requirements) and what is visible at
    all (the image's distributions included)."""
    prefix = Path(sys.prefix).resolve()
    owned: dict[str, list[str]] = {}
    visible: set[str] = set()
    for dist in distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        visible.add(canonical(name))
        if Path(dist.locate_file("")).resolve().is_relative_to(prefix):
            owned[canonical(name)] = [requirement for requirement in (dist.requires or []) if requirement]
    return owned, visible


def main(argv: list[str] | None = None) -> int:
    """The CLI: complete this venv's own missing dependencies from ``<WHEELHOUSE>``, to a fixed point."""
    parser = argparse.ArgumentParser(
        prog="reference_deps.py",
        description="Complete the reference venv's own missing dependencies from a staged wheelhouse.",
    )
    parser.add_argument("wheelhouse", help="the staged wheelhouse directory (the only index)")
    args = parser.parse_args(argv)
    wheelhouse = Path(args.wheelhouse)
    installed: set[str] = set()
    for _round in range(_MAX_ROUNDS):
        owned, visible = _owned_and_visible()
        planned = plan_more(owned, visible, installed)
        if not planned:
            print(
                f"reference_deps: the venv's own dependencies are complete ({len(installed)} installed: "
                f"{', '.join(sorted(installed)) or 'none'})",
                file=sys.stderr,
            )
            return 0
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "--no-deps",
                    "--no-index",
                    "--find-links",
                    str(wheelhouse),
                    *planned,
                ]
            )
        except subprocess.CalledProcessError:
            print(
                f"reference_deps: the wheelhouse cannot satisfy {planned}: add them to "
                "requirements-reference.txt or stage their wheels in the wheelhouse",
                file=sys.stderr,
            )
            return 1
        installed.update(requirement_name(spec) for spec in planned)
    print(
        f"reference_deps: no fixed point after {_MAX_ROUNDS} rounds (the needs keep growing: {planned})",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
