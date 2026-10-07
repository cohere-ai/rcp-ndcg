#!/usr/bin/env python3
"""Complete the reference venv's own distributions' missing dependencies, to a fixed point.

The bootstrap installs the recipes' reference requirements with ``--no-deps`` (the reference must never
resolve or replace the image's torch stack: its dependency tree is not registered in the image, and
the wheelhouse's CPU torch must never land over the image's CUDA build).  What ``--no-deps``
therefore cannot pull -- the reference venv's OWN distributions' declared dependencies
(sentence-transformers needs scikit-learn, scipy, joblib, threadpoolctl) -- is completed here, each
missing requirement installed ``--no-deps`` from the staged wheelhouse only, to a fixed point.

The image's own distributions are never completed -- installing over one would shadow the image's CUDA
stack -- so a need the image's seen version cannot satisfy is a hard error naming the way out.  An
``extra ==`` marker means nothing asked for that requirement (``extra !=`` is a real need).  Runs in
the reference venv's python (stdlib only), as ``reference_deps.py <WHEELHOUSE>``: exits 0 at the fixed
point (what it installed is reported on stderr), exits 1 with the requirement names and the way out
when the wheelhouse cannot satisfy a missing dependency.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path

__all__ = ["UnsatisfiableImageRequirement", "canonical", "main", "plan_more", "requirement_name"]

_MAX_ROUNDS = 20
"""A completion round installs the still-missing owners' needs; runaway loops fail loudly instead."""


class UnsatisfiableImageRequirement(RuntimeError):
    """An owned distribution's need the visible IMAGE distribution cannot satisfy: installing over an
    image distribution is refused (it would shadow the image's CUDA stack), so the caller fails loudly,
    with ``REFERENCE_REQUIREMENTS`` and its own venv as the way out."""


def canonical(name: str) -> str:
    """PEP 503's canonical distribution name (``re.sub(r\"[-_.]+\", \"-\", name).lower()``)."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str:
    """The canonical distribution name a requirement string names (``name[extras] <spec> ; marker``)."""
    head = requirement.partition(";")[0].strip()
    head = head.split("[", 1)[0]
    head = re.split(r"[<>=!~ (]", head, maxsplit=1)[0]
    return canonical(head)


def plan_more(
    owned: dict[str, list[str]],
    visible: dict[str, list[str]],
    scheduled: set[str] = frozenset(),
) -> list[str]:
    """The requirement strings to install next, one round of the fixed point.

    Inputs: ``owned`` maps each distribution the reference venv OWNS (installed into it -- the image's
    distributions are never owners) to its ``Requires-Dist`` strings; ``visible`` maps every visible
    distribution (image included) to the versions it is seen at; every name in ``scheduled`` was
    already installed this run (pip checked its specifier at install).  Output: the missing requirement
    strings (markers stripped) of the OWNED distributions, deduplicated in sorted-owner order.  A
    requirement whose marker positively asks for an ``extra`` is skipped (nothing asks for extras);
    ``extra !=`` is a real need.  A visible IMAGE distribution whose version cannot satisfy a need
    raises :class:`UnsatisfiableImageRequirement` (never installed over); a visible OWN distribution at
    a stale version is re-planned, which upgrades the venv's own copy.  With ``packaging`` the
    specifier check is exact; without it a dotted-numeric comparison decides and anything unparseable
    counts as satisfied (conservative -- never blocks a real install).  Units: none.
    """
    planned: list[str] = []
    seen: set[str] = set()
    for name in sorted(owned):
        for requirement in owned[name]:
            if not requirement.strip() or _requests_extra(requirement):
                continue
            dep = requirement_name(requirement)
            if not dep or dep in scheduled:
                continue
            versions = visible.get(dep)
            if versions and _satisfies(requirement, versions):
                continue
            if versions and dep not in owned:
                raise UnsatisfiableImageRequirement(
                    f"the image's {dep}=={', '.join(versions)} does not satisfy {requirement!r}; "
                    "replacing an image distribution is refused (that would shadow the image's CUDA "
                    "stack) - a paper reference that needs other versions gets REFERENCE_REQUIREMENTS "
                    "and its own venv"
                )
            spec = requirement.partition(";")[0].strip()
            if spec not in seen:
                seen.add(spec)
                planned.append(spec)
    return planned


def _requests_extra(requirement: str) -> bool:
    """``name; extra == 'x'`` (or ``extra in``) means nothing asked for that extra; ``extra !=`` does
    not -- with no extras requested, that need exists."""
    _, sep, marker = requirement.partition(";")
    return bool(sep) and re.search(r"\bextra\s*(?:===|==|=)|\bextra\s+in\b", marker) is not None


def _satisfies(requirement: str, versions: list[str]) -> bool:
    """Whether one of ``versions`` satisfies ``requirement``'s specifier (its marker excluded)."""
    spec_text = requirement.partition(";")[0].strip()
    try:
        from packaging.requirements import Requirement

        wanted = Requirement(spec_text)
        return any(wanted.specifier.contains(version, prereleases=True) for version in versions)
    except Exception:  # noqa: BLE001 - no packaging (or an odd requirement): the conservative check
        return _loose_satisfies(spec_text, versions)


def _loose_satisfies(spec_text: str, versions: list[str]) -> bool:
    """A conservative specifier check without ``packaging``: dotted-numeric releases, "as is" strings;
    anything it cannot parse counts as satisfied (never blocks a real install)."""
    parsed = re.findall(r"(==|!=|<=|>=|<|>)\s*([A-Za-z0-9.*+!_-]+)", spec_text)
    if not parsed:
        return True
    for operator, target in parsed:
        if "*" in target:
            if operator not in ("==", "!="):
                return True
            hit = any(_wildcard_match(version, target, negate=(operator == "!=")) for version in versions)
            if not hit:
                return False
            continue
        if not any(_compare(version, operator, target) for version in versions):
            return False
    return True


def _release(version: str) -> tuple[int, ...] | None:
    """A version's dotted-numeric release (other shapes: None -> the caller stays conservative)."""
    core = version.split("+")[0].lstrip("vV").split("!")[-1]
    release = re.match(r"(\d+(?:\.\d+)*)", core)
    if release is None:
        return None
    return tuple(int(part) for part in release.group(1).split("."))


def _compare(version: str, operator: str, target: str) -> bool:
    """One comparison of releases (zero-padded); a shape this cannot parse counts as satisfying."""
    left, right = _release(version), _release(target)
    if left is None or right is None:
        return True
    width = max(len(left), len(right))
    left = left + (0,) * (width - len(left))
    right = right + (0,) * (width - len(right))
    if operator == "==":
        return left == right
    if operator == "!=":
        return left != right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    return left >= right


def _wildcard_match(version: str, target: str, *, negate: bool) -> bool:
    """``== 2.1.*`` style matching on the release's prefix (its pruning decides ``!=``)."""
    release = _release(version)
    prefix = _release(target)
    if release is None or prefix is None:
        return True
    matched = release[: len(prefix)] == prefix
    return (not matched) if negate else matched


def _owned_and_visible() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """What is installed in this venv (owned, with its declared requirements) and everything visible
    (name -> the versions seen; the image's distributions included but never completed)."""
    prefix = Path(sys.prefix).resolve()
    owned: dict[str, list[str]] = {}
    visible: dict[str, list[str]] = {}
    for dist in distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        key = canonical(name)
        visible.setdefault(key, []).append(str(dist.metadata.get("Version") or ""))
        if Path(dist.locate_file("")).resolve().is_relative_to(prefix):
            owned[key] = [requirement for requirement in (dist.requires or []) if requirement]
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
        try:
            planned = plan_more(owned, visible, installed)
        except UnsatisfiableImageRequirement as error:
            print(f"reference_deps: {error}", file=sys.stderr)
            return 1
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
