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
the reference venv's python (stdlib only), as ``reference_deps.py <WHEELHOUSE>...``: exits 0 at the fixed
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

__all__ = ["UnsatisfiableImageRequirement", "canonical", "main", "plan_more", "requirement_name", "satisfies"]

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
    strings (markers stripped) of the OWNED distributions whose full PEP 508 marker holds for THIS
    interpreter (the reference venv's python, no extra requested -- :func:`_marker_allows`),
    deduplicated in sorted-owner order.  The false-marked ``tomli; python_version < '3.11'`` of GPU-E1
    is never planned again.  A visible IMAGE distribution whose version cannot satisfy a need
    raises :class:`UnsatisfiableImageRequirement` (never installed over); a visible OWN distribution at
    a stale version is re-planned, which upgrades the venv's own copy.  With ``packaging`` the
    specifier check is exact; without it a dotted-numeric comparison decides and anything unparseable
    counts as satisfied (conservative -- never blocks a real install).  Units: none.
    """
    planned: list[str] = []
    seen: set[str] = set()
    for name in sorted(owned):
        for requirement in owned[name]:
            if not requirement.strip() or not _marker_allows(requirement):
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


def satisfies(requirement: str, version: str) -> bool:
    """Whether one version satisfies a requirement's specifier (its marker excluded).

    Inputs: a requirement string (``name[extras] <specifier> ; marker``) and one version.  Output:
    ``True`` when the specifier admits the version (exact with ``packaging``; the conservative
    dotted-numeric check without it, where anything unparseable counts as satisfied).  Units: none.
    """
    return _satisfies(requirement, [version])


def _marker_allows(requirement: str) -> bool:
    """Whether one ``Requires-Dist`` string's PEP 508 marker holds for THIS interpreter (the reference
    venv's python): a requirement with no marker always does; one with a marker holds when the marker
    evaluates true with no extra requested (``extra == 'x'`` is never true, ``extra != 'x'`` is).

    With ``packaging`` the evaluation is exact (:func:`packaging.markers.Marker.evaluate` over the
    interpreter's default environment plus ``extra: ''``).  Without it a local evaluator decides the
    common environment markers (python_version, python_full_version, sys_platform, platform_system,
    platform_machine, os_name, implementation names); anything it cannot resolve counts as allowed
    (conservative -- never blocks a real install).  Units: none.
    """
    _, sep, marker = requirement.partition(";")
    if not sep:
        return True
    marker = marker.strip()
    try:
        from packaging.markers import Marker, default_environment
    except ImportError:
        return _loose_marker_allows(marker)
    try:
        return bool(Marker(marker).evaluate({**default_environment(), "extra": ""}))
    except Exception:  # noqa: BLE001 - an unparseable marker is a need (never blocks a real install)
        return True


def _loose_marker_allows(marker: str) -> bool:
    """The marker check without ``packaging``: the common environment markers, evaluated locally.

    ``python_version`` and ``python_full_version`` compare numerically on their release segments; every
    other comparison is string equality or ordering.  A variable this evaluator does not know, an
    operator it cannot parse, or a trailing fragment makes the whole (sub-)expression count as allowed
    (the need stands -- installing is today's behaviour, and pip's own metadata is trusted to be sane).
    Units: none.
    """
    tokens = _marker_tokens(marker)
    value, index = _marker_or(tokens, 0)
    return bool(value) if index >= len(tokens) else True


def _marker_tokens(marker: str) -> list[tuple[str, str]]:
    """A marker's tokens as ``(kind, text)`` pairs; a fragment the grammar cannot read ends the scan."""
    pattern = re.compile(r"\s*(<=|>=|==|!=|~=|<|>|\(|\)|'[^']*'|\"[^\"]*\"|[A-Za-z_][A-Za-z0-9_.]*)")
    tokens: list[tuple[str, str]] = []
    position = 0
    while position < len(marker):
        match = pattern.match(marker, position)
        if match is None:
            break
        text = match.group(1)
        if text[0] in "'\"":
            tokens.append(("value", text[1:-1]))
        else:
            kind = "op" if text in _MARKER_OPS else "paren" if text in "()" else "name"
            tokens.append((kind, text))
        position = match.end()
    return tokens


_MARKER_OPS = frozenset({"<=", ">=", "==", "!=", "~=", "<", ">", "in", "not"})


def _marker_or(tokens: list[tuple[str, str]], index: int) -> tuple[bool, int]:
    """``a and b or c``: and-exprs joined by ``or`` (PEP 508 binds ``and`` tighter)."""
    value, index = _marker_and(tokens, index)
    while index < len(tokens) and tokens[index] == ("name", "or"):
        right, index = _marker_and(tokens, index + 1)
        value = value or right
    return value, index


def _marker_and(tokens: list[tuple[str, str]], index: int) -> tuple[bool, int]:
    """Marker items joined by ``and``; an unparseable item leaves the need standing (True)."""
    value, index = _marker_item(tokens, index)
    while index < len(tokens) and tokens[index] == ("name", "and"):
        right, index = _marker_item(tokens, index + 1)
        value = value and right
    return value, index


def _marker_item(tokens: list[tuple[str, str]], index: int) -> tuple[bool, int]:
    """One comparison (``var op var``, ``var in var``, ``var not in var``) or a parenthesised group."""
    if index < len(tokens) and tokens[index] == ("paren", "("):
        value, index = _marker_or(tokens, index + 1)
        if index < len(tokens) and tokens[index] == ("paren", ")"):
            return value, index + 1
        return True, index  # unbalanced: the need stands
    if index + 2 < len(tokens) and tokens[index][0] in ("name", "value"):
        left = _marker_value(tokens[index])
        operator = tokens[index + 1][1]
        index += 2
        if operator == "not" and index < len(tokens) and tokens[index] == ("name", "in"):
            operator, index = "not in", index + 1
        if index < len(tokens) and tokens[index][0] in ("name", "value"):
            right = _marker_value(tokens[index])
            return _marker_compare(left, operator, right), index + 1
    return True, index  # unparseable: the need stands


def _marker_value(token: tuple[str, str]) -> str | None:
    """One marker operand: an environment variable's value or a literal string; an unknown variable is
    ``None`` (its comparison then leaves the need standing)."""
    kind, text = token
    if kind == "value":
        return text
    import os
    import platform
    import sys

    values: dict[str, str | None] = {
        "python_version": ".".join(str(part) for part in sys.version_info[:2]),
        "python_full_version": platform.python_version(),
        "sys_platform": sys.platform,
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "os_name": os.name,
        "implementation_name": platform.python_implementation(),
        "platform_python_implementation": platform.python_implementation(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "extra": "",
    }
    return values.get(text)


def _marker_compare(left: str | None, operator: str, right: str | None) -> bool:
    """One marker comparison; a side this evaluator cannot resolve allows the need (True)."""
    if left is None or right is None:
        return True
    if operator == "in":
        return left in right
    if operator == "not in":
        return left not in right
    if operator == "~=" and _release(left) is not None and _release(right) is not None:
        prefix = (_release(right) or ())[:-1]
        return _compare(left, ">=", right) and (_release(left) or ())[: len(prefix)] == prefix
    if operator in ("<", "<=", ">", ">=") and _release(left) is not None and _release(right) is not None:
        return _compare(left, operator, right)
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
    if operator == ">=":
        return left >= right
    return True  # ~= on non-versions and anything else: the need stands


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
    """The CLI: complete this venv's own missing dependencies from the ``<WHEELHOUSE>...`` directories,
    to a fixed point."""
    parser = argparse.ArgumentParser(
        prog="reference_deps.py",
        description="Complete the reference venv's own missing dependencies from staged wheelhouse(s).",
    )
    parser.add_argument("wheelhouse", nargs="+", help="the staged wheelhouse director(ies) (the only index)")
    args = parser.parse_args(argv)
    wheelhouses = [Path(path) for path in args.wheelhouse]
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
                    *[flag for path in wheelhouses for flag in ("--find-links", str(path))],
                    *planned,
                ]
            )
        except subprocess.CalledProcessError:
            print(
                f"reference_deps: the wheelhouse(s) {[str(path) for path in wheelhouses]} cannot satisfy "
                f"{planned}: add them to the family's reference.lock or stage their wheels in the wheelhouse",
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
