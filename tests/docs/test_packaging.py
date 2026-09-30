"""The extras in pyproject.toml say what the code needs, and ``rcp-ndcg doctor`` checks what the extras install."""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version

from tests.docs._markdown import ROOT

PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
EXTRAS = PYPROJECT["project"]["optional-dependencies"]


def _requirement(extra: str, name: str) -> Requirement:
    matches = [Requirement(spec) for spec in EXTRAS[extra] if Requirement(spec).name.lower() == name.lower()]
    assert len(matches) == 1, f"[{extra}] declares {name} {len(matches)} times"
    return matches[0]


def test_the_mteb_extra_installs_an_mteb_the_integration_runs_on() -> None:
    """The integration imports mteb 2.x modules, and ViDoRe v3 needs 2.10.5."""
    floors = [Version(spec.version) for spec in _requirement("mteb", "mteb").specifier if spec.operator == ">="]
    assert floors and min(floors) >= Version("2.10.5")


def test_doctor_checks_the_packages_the_extras_install() -> None:
    from rcp_ndcg.cli.doctor import _EXTRAS

    local = {Requirement(spec).name.lower() for spec in EXTRAS["local"]}
    assert "faiss-cpu" not in local, "nothing imports faiss"
    assert "faiss" not in _EXTRAS["local"]
    core = {Requirement(spec).name.lower() for spec in PYPROJECT["project"]["dependencies"]}
    assert {"bm25s", "pystemmer"} <= core, "BM25 runs on any install"
    assert {"bm25s", "Stemmer"}.isdisjoint(_EXTRAS["local"])


def test_no_extra_restates_a_core_dependency() -> None:
    core = {Requirement(spec).name.lower() for spec in PYPROJECT["project"]["dependencies"]}
    for extra, specs in EXTRAS.items():
        restated = {Requirement(spec).name.lower() for spec in specs} & core
        assert not restated, f"[{extra}] restates core dependencies {sorted(restated)}"


def test_every_runtime_dependency_is_imported_somewhere() -> None:
    """A declared dependency nothing imports is weight on every install."""
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for folder in ("src", "packages", "experiments", "examples")
        for path in (ROOT / folder).rglob("*.py")
    )
    # distribution -> the module it provides, where the names differ; fsspec protocols load their backend lazily.
    module = {
        "pyyaml": "yaml",
        "pillow": "PIL",
        "python-dotenv": "dotenv",
        "rcp-ndcg-core": "rcp_ndcg_core",
        "pystemmer": "Stemmer",
    }
    lazily_loaded = {"gcsfs"}
    for spec in PYPROJECT["project"]["dependencies"]:
        name = Requirement(spec).name.lower()
        if name in lazily_loaded:
            continue
        imported = module.get(name, name.replace("-", "_"))
        assert re.search(rf"^\s*(import|from) {re.escape(imported)}\b", sources, re.M), f"nothing imports {name}"


CONSTRAINTS = ROOT / "requirements-constraints.txt"


def _pins(text: str) -> dict[str, str]:
    """``{name: "version ; marker"}`` of a requirements file, comments dropped."""
    pins = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            requirement = Requirement(line)
            (spec,) = requirement.specifier
            pins[requirement.name.lower()] = f"{spec.version} ; {requirement.marker}"
    return pins


def test_the_constraints_file_is_the_locks_export_for_the_coordinators_extras() -> None:
    """The file a rendered job installs against pins exactly what uv.lock pins, for the extras it installs."""
    from rcp_ndcg.runners.script import COORDINATOR_EXTRAS

    text = CONSTRAINTS.read_text(encoding="utf-8")
    command = text.splitlines()[1].lstrip("# ").split()
    assert command[:2] == ["uv", "export"] and "--frozen" in command
    extras = [command[i + 1] for i, word in enumerate(command) if word == "--extra"]
    assert sorted(extras) == sorted(COORDINATOR_EXTRAS)
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked: dict[str, set[str]] = {}
    for package in lock["package"]:
        locked.setdefault(package["name"].lower(), set()).add(package["version"])
    pins = _pins(text)
    assert "torch" in pins and "rcp-ndcg" not in pins and "rcp-ndcg-core" not in pins
    stale = {name: pin for name, pin in pins.items() if pin.split(" ;")[0] not in locked.get(name, set())}
    assert not stale, f"regenerate requirements-constraints.txt with the command in its header: {stale}"
    uv = shutil.which("uv")
    if uv is not None:  # the exact export, when uv is at hand
        argv = [uv, *command[1 : command.index("-o")], "--no-header", "-q"]
        exported = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=True).stdout
        assert _pins(exported) == pins


def test_the_release_workflow_publishes_both_distributions_with_trusted_publishing() -> None:
    import yaml

    text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(text)
    assert workflow[True]["push"]["tags"] == ["v*"]  # YAML reads the key `on` as true
    publish = workflow["jobs"]["publish"]
    assert publish["environment"] == "pypi" and publish["permissions"] == {"id-token": "write"}
    assert any(step.get("uses", "").startswith("pypa/gh-action-pypi-publish@") for step in publish["steps"])
    assert "secrets." not in text  # trusted publishing: no token anywhere
    assert "uv build --all-packages" in text and "requirements-constraints.txt" in text
