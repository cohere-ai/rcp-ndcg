"""The extras in pyproject.toml say what the code needs, and ``rcp-ndcg doctor`` checks what the extras install."""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib

import pytest
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

    declared = {Requirement(spec).name.lower() for specs in EXTRAS.values() for spec in specs}
    assert "faiss-cpu" not in declared, "nothing imports faiss"
    mapped = {module for modules in _EXTRAS.values() for module in modules}
    assert "faiss" not in mapped
    core = {Requirement(spec).name.lower() for spec in PYPROJECT["project"]["dependencies"]}
    assert {"bm25s", "pystemmer"} <= core, "BM25 runs on any install"
    assert {"bm25s", "Stemmer"}.isdisjoint(mapped)


def test_extra_names_have_one_home() -> None:
    """The extras ``EXTRA_FOR_MODULE`` names are exactly the runtime extras ``pyproject.toml`` declares.

    One home for the extra names: a module's install hint may not name an extra that does not exist, and a
    declared runtime extra that no module maps to would never reach an install hint. ``dev`` and ``docs`` are
    the tooling extras: nothing the package imports belongs to them, so no module maps to them.
    """
    from rcp_ndcg.errors import EXTRA_FOR_MODULE

    named = set(EXTRA_FOR_MODULE.values())
    tooling = {"dev", "docs"}
    assert named == set(EXTRAS) - tooling, (
        f"EXTRA_FOR_MODULE and pyproject.toml disagree: {sorted(named ^ (set(EXTRAS) - tooling))}"
    )


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
    """``{name: version}`` of a requirements file's exact pins, comments, markers and layout dropped.

    The comparison is the semantic one the release check makes (which package is pinned to which version), so a
    newer uv's re-serialisation of the same export -- marker spacing, quoting, conjunction order, the ``# via``
    comments -- does not fail it; a moved version does.
    """
    pins = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            requirement = Requirement(line)
            (spec,) = requirement.specifier
            pins[requirement.name.lower()] = spec.version
    return pins


def test_the_pins_comparison_is_semantic_not_textual() -> None:
    """A newer uv's re-serialisation of the same export compares equal; only a version drift is a difference."""
    committed = "torch==2.8.0 ; platform_system == 'Linux' and platform_machine == 'x86_64'\nnumpy==2.3.0\n"
    reserialised = (
        'torch==2.8.0;platform_machine == "x86_64" and platform_system == "Linux"  # via rcp-ndcg\n'
        "numpy==2.3.0; python_version >= '3.11'\n"
    )
    assert _pins(reserialised) == _pins(committed) == {"torch": "2.8.0", "numpy": "2.3.0"}


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
    stale = {name: pin for name, pin in pins.items() if pin not in locked.get(name, set())}
    assert not stale, f"regenerate requirements-constraints.txt with the command in its header: {stale}"
    uv = shutil.which("uv")
    if uv is not None:  # the exact export, when uv is at hand
        argv = [uv, *command[1 : command.index("-o")], "--no-header", "-q"]
        exported = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=True).stdout
        assert _pins(exported) == pins


RELEASE_PUBLISH_JOBS = {
    # job -> (GitHub environment, PyPI project): PyPI identifies a pending trusted publisher by owner, repository,
    # workflow file and environment only, so each package publishes through its own environment.
    "publish-core": ("pypi-core", "rcp-ndcg-core"),
    "publish-rcp-ndcg": ("pypi", "rcp-ndcg"),
    "publish-vllm": ("pypi-vllm", "rcp-ndcg-vllm"),
}


def _release_workflow() -> tuple[dict, str]:
    import yaml

    text = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    return yaml.safe_load(text), text


def _assert_one_environment_per_package(workflow: dict) -> None:
    """The release contract: one build, then one publish job per package, each in its own GitHub environment."""
    jobs = workflow["jobs"]
    assert workflow[True]["push"]["tags"] == ["v*"]  # YAML reads the key `on` as true
    assert set(jobs) == {"build", *RELEASE_PUBLISH_JOBS, "github-release"}
    build = str(jobs["build"])
    assert "uv build --all-packages" in build and "packages/rcp-ndcg-vllm" in build, (
        "rcp-ndcg-vllm is outside the uv workspace: the build job must build it from its own directory"
    )
    assert "twine check" in build and "requirements-constraints.txt" in build
    # One artifact per package, each holding only that package's sdist and wheel.
    uploads = [s for s in jobs["build"]["steps"] if str(s.get("uses", "")).startswith("actions/upload-artifact@")]
    artifacts = {upload["with"]["name"]: upload["with"]["path"] for upload in uploads}
    assert artifacts == {
        f"dist-{project}": f"dist/{project.replace('-', '_')}-*" for _, project in RELEASE_PUBLISH_JOBS.values()
    }
    for job, (environment, project) in RELEASE_PUBLISH_JOBS.items():
        publish = jobs[job]
        assert publish["environment"] == environment, f"{job} must publish {project} from environment {environment!r}"
        assert publish["permissions"] == {"id-token": "write"}
        downloads = [s for s in publish["steps"] if str(s.get("uses", "")).startswith("actions/download-artifact@")]
        assert len(downloads) == 1, f"{job} downloads only its own artifact"
        assert downloads[0]["with"]["name"] == f"dist-{project}"
        directory = downloads[0]["with"]["path"]
        assert directory == f"dist/{project}", f"{job} works in a directory that holds only {project}'s files"
        published = [s for s in publish["steps"] if str(s.get("uses", "")).startswith("pypa/gh-action-pypi-publish@")]
        assert len(published) == 1 and published[0]["with"]["packages-dir"] == directory
    assert jobs["publish-core"]["needs"] == "build"
    assert jobs["publish-rcp-ndcg"]["needs"] == "publish-core"  # it pins the core exactly: the core goes first
    assert jobs["publish-vllm"]["needs"] == "build"
    assert set(jobs["github-release"]["needs"]) == set(RELEASE_PUBLISH_JOBS)
    tokenised = {name for name, job in jobs.items() if (job.get("permissions") or {}).get("id-token") == "write"}
    assert tokenised == set(RELEASE_PUBLISH_JOBS), "id-token: write belongs to exactly the publish jobs"


def test_the_release_workflow_publishes_three_packages_one_environment_each() -> None:
    workflow, text = _release_workflow()
    assert "secrets." not in text  # trusted publishing: no token anywhere
    _assert_one_environment_per_package(workflow)


def test_the_release_workflow_check_fails_when_two_environments_are_swapped() -> None:
    workflow, _ = _release_workflow()
    workflow["jobs"]["publish-core"]["environment"] = "pypi"
    workflow["jobs"]["publish-rcp-ndcg"]["environment"] = "pypi-core"
    with pytest.raises(AssertionError):
        _assert_one_environment_per_package(workflow)


def test_both_distributions_ship_the_license_and_the_notice() -> None:
    """NOTICE attributes the third-party code (Apache-2.0 section 4(d)); each distribution carries the same copy."""
    core = ROOT / "packages" / "rcp-ndcg-core"
    for folder in (ROOT, core):
        project = tomllib.loads((folder / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        assert project["license-files"] == ["LICENSE", "NOTICE"]
    for name in ("LICENSE", "NOTICE"):
        assert (core / name).read_bytes() == (ROOT / name).read_bytes(), f"packages/rcp-ndcg-core/{name} is stale"
    assert "smart_resize" in (ROOT / "NOTICE").read_text(encoding="utf-8")
