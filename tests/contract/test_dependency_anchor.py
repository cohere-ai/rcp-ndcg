"""The dependency anchor: the lock stays inside the windows the shipped engine release declares.

The anchor is ``vllm`` ``v0.33.0`` -- the tag the serving image is built from (``vllm/vllm-openai:v0.33.0``; the
same tag ``tests/contract/test_engine_pins.py`` cites). The release names the tag and the commit the tag points to,
never a checkout path, so this module reads the root manifest and ``uv.lock`` and nothing outside the repository:
the lock is checked offline, on a machine that has no engine tree.

Three failures it catches:

* :func:`test_the_anchor_names_the_release_it_is_read_at` -- the record's provenance drifts: a tag or commit that
  is not the release, a row citing a file the release does not carry, a source that is not ``file:line``.
* :func:`test_the_record_and_the_manifest_declare_the_same_windows` -- the record and the manifest disagree in
  **either** direction: a row no declaration enforces, a declaration with no row, or a window whose specifier the
  two spell differently.
* :func:`test_the_lock_satisfies_every_anchor_window` -- a resolution leaves a window: torch off ``==2.14.0``,
  transformers out of ``[5.16.1,5.21.0)``, huggingface-hub below ``1.31.0``, tokenizers below ``0.23.2``,
  setuptools out of ``[78.1.1,81.0.0)``. This is the mutation the anchor exists to stop.

The record's rows arrive with the version that satisfies them (its header says why), so the coverage of the third
test grows with the ladder: each commit that moves a package declares its window in the same step.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from tests.contract.surface import REPO

ANCHOR = Path(__file__).parent / "dependency-anchor.toml"
MANIFEST = REPO / "pyproject.toml"
LOCK = REPO / "uv.lock"

RELEASE = "v0.33.0"
"""The engine release the anchor is read at: the tag the serving image is built from."""

COMMIT = "4ba787ec9959e566d3b587a796dff07673324da5"
"""The commit the release tag points to (``git rev-parse v0.33.0^{commit}`` in the engine's repository)."""


def _canonical(name: str) -> str:
    """The PEP 503 spelling of a distribution name, so the record, the manifest and the lock compare as one name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _anchor() -> dict:
    """The committed record: its ``release`` provenance and its ``windows`` rows."""
    return tomllib.loads(ANCHOR.read_text(encoding="utf-8"))


def _recorded() -> dict[str, SpecifierSet]:
    """``{canonical name: window}`` of the record's rows."""
    return {_canonical(name): SpecifierSet(row["specifier"]) for name, row in _anchor()["windows"].items()}


def _declared() -> dict[str, SpecifierSet]:
    """``{canonical name: window}`` of the root manifest's ``[tool.uv] constraint-dependencies``."""
    manifest = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    out: dict[str, SpecifierSet] = {}
    for spec in manifest["tool"]["uv"]["constraint-dependencies"]:
        requirement = Requirement(spec)
        out[_canonical(requirement.name)] = requirement.specifier
    return out


def _locked() -> dict[str, list[str]]:
    """``{canonical name: [version, ...]}`` of every package ``uv.lock`` pins (one entry per fork)."""
    lock = tomllib.loads(LOCK.read_text(encoding="utf-8"))
    out: dict[str, list[str]] = {}
    for package in lock["package"]:
        out.setdefault(_canonical(package["name"]), []).append(package["version"])
    return out


def test_the_anchor_names_the_release_it_is_read_at() -> None:
    """The record cites the tag, the commit it points to, and a file of the release for every row."""
    release = _anchor()["release"]
    assert release["distribution"] == "vllm"
    assert release["tag"] == RELEASE, "the anchor is read at the shipped engine release"
    assert release["commit"] == COMMIT, "the commit must be the one the tag points to, not a moving branch"
    files = set(release["files"])
    for name, row in _anchor()["windows"].items():
        path, _, line = row["source"].rpartition(":")
        assert path in files, f"{name}: cites {path!r}, which the release record does not list"
        assert line.isdigit() and int(line) > 0, f"{name}: {row['source']!r} is not a file:line citation"


def test_the_record_and_the_manifest_declare_the_same_windows() -> None:
    """Set equality in both directions: no row without a declaration, no declaration without a row."""
    recorded, declared = _recorded(), _declared()
    assert recorded, "the record declares no window, so the lock is checked against nothing"
    assert set(recorded) == set(declared), (
        "the record and the manifest disagree: "
        f"recorded but not declared {sorted(set(recorded) - set(declared))}, "
        f"declared but not recorded {sorted(set(declared) - set(recorded))}"
    )
    differing = {name: (recorded[name], declared[name]) for name in recorded if recorded[name] != declared[name]}
    assert not differing, f"the record and the manifest spell a window differently: {differing}"


def test_the_lock_satisfies_every_anchor_window() -> None:
    """Every version ``uv.lock`` pins for an anchored package lies inside the release's window for it."""
    locked = _locked()
    violations: list[str] = []
    for name, window in sorted(_recorded().items()):
        versions = locked.get(name)
        if versions is None:
            violations.append(f"{name}: not in the lock, so the window {window} checks nothing")
            continue
        outside = [version for version in versions if Version(version) not in window]
        violations.extend(f"{name}: {version} is outside {window} (the engine release's window)" for version in outside)
    assert not violations, "the lock leaves the engine release's windows:\n" + "\n".join(violations)
