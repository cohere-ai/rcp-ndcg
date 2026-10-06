"""The extras in pyproject.toml say what the code needs, and ``rcp-ndcg doctor`` checks what the extras install."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
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


WORKFLOW_ACTIONS = {
    "actions/checkout": "11d5960a326750d5838078e36cf38b85af677262",  # v4.4.0
    "astral-sh/setup-uv": "d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86",  # v5.4.2
    "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",  # v4.6.2
    "actions/download-artifact": "d3f86a106a0bac45b974a628896c90dbdf5c8093",  # v4.3.0
    "pypa/gh-action-pypi-publish": "dc37677b2e1c63e2034f94d8a5b11f265b73ba33",  # v1.14.2 (release/v1)
}


def test_every_workflow_action_is_pinned_to_a_full_commit_sha() -> None:
    """R22: every ``uses:`` is a full 40-hex commit SHA (a tag is mutable); its tag stays in a trailing comment."""
    pinned = 0
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not (stripped.startswith("- uses:") or stripped.startswith("uses:")):
                continue
            uses = stripped.removeprefix("- ").removeprefix("uses:").strip()
            action, comment = (uses.split(" #", 1) + [""])[:2]
            name, _, sha = action.rpartition("@")
            assert name, f"{path.name}:{number}: not an action: {stripped!r}"
            assert re.fullmatch(r"[0-9a-f]{40}", sha), (
                f"{path.name}:{number}: {name} is not pinned to a full 40-hex commit SHA ({sha!r})"
            )
            assert comment.strip().startswith("v"), (
                f"{path.name}:{number}: {name} pins a SHA without its release tag in a trailing comment"
            )
            pinned += 1
    assert pinned >= len(WORKFLOW_ACTIONS), "no workflow action found to pin"


def test_the_release_workflow_checks_the_vllm_packages_rcp_ndcg_pin(tmp_path) -> None:
    """When ``rcp-ndcg-vllm`` depends on ``rcp-ndcg``, the release check requires exactly ``==<tag version>``.

    The check runs the workflow's own step against a manifest in ``tmp_path``; it must pass without the
    dependency (and without the package), and refuse any other specifier.
    """
    workflow, _ = _release_workflow()
    steps = workflow["jobs"]["build"]["steps"]
    step = next(step for step in steps if "rcp-ndcg-vllm pins" in str(step.get("name", "")))
    body = str(step["run"]).split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]

    def check(manifest: str | None, version: str = "0.0.1") -> subprocess.CompletedProcess[str]:
        tree = tmp_path / f"check-{check.calls:03d}-{version}"
        check.calls += 1
        package = tree / "packages" / "rcp-ndcg-vllm"
        package.mkdir(parents=True)
        if manifest is not None:
            (package / "pyproject.toml").write_text(manifest, encoding="utf-8")
        return subprocess.run([sys.executable, "-", version], input=body, capture_output=True, text=True, cwd=tree)

    check.calls = 0

    exact = '[project]\nname = "rcp-ndcg-vllm"\ndependencies = ["rcp-ndcg==0.0.1"]\n'
    assert check(exact).returncode == 0, "an exact pin must pass"
    assert check('[project]\ndependencies = ["rcp-ndcg[calibrate]==0.0.1"]\n').returncode == 0
    assert check('[project]\ndependencies = ["rcp_ndcg==0.0.1"]\n').returncode == 0, "a _ name normalises"
    assert check('[project]\ndependencies = ["rcp-ndcg == 0.0.1"]\n').returncode == 0, "spaces normalise"
    assert check('[project]\ndependencies = ["rcp-ndcg (==0.0.1)"]\n').returncode == 0, "parentheses normalise"
    dotted = check('[project]\ndependencies = ["rcp.ndcg==0.0.1"]\n')
    assert dotted.returncode == 0 and "pins rcp-ndcg==0.0.1" in dotted.stdout, "a . name normalises (PEP 503)"
    dotted_loose = check('[project]\ndependencies = ["RCP.NDCG==0.0.2"]\n')
    assert dotted_loose.returncode == 1 and "==0.0.1" in dotted_loose.stderr, "a . name must still be checked"
    assert check(None).returncode == 0, "no package, nothing to check"
    assert check('[project]\ndependencies = ["numpy"]\n').returncode == 0, "no dependency, nothing to check"
    for wrong in ("rcp-ndcg>=0.0.1", "rcp-ndcg", "rcp-ndcg[calibrate]", "rcp-ndcg==0.0.2"):
        result = check(f'[project]\ndependencies = ["{wrong}"]\n')
        assert result.returncode == 1, f"{wrong} must be refused"
        assert "==0.0.1" in result.stderr


def test_the_release_workflow_check_fails_when_two_environments_are_swapped() -> None:
    workflow, _ = _release_workflow()
    workflow["jobs"]["publish-core"]["environment"] = "pypi"
    workflow["jobs"]["publish-rcp-ndcg"]["environment"] = "pypi-core"
    with pytest.raises(AssertionError):
        _assert_one_environment_per_package(workflow)


def test_both_distributions_ship_the_license_and_the_notice() -> None:
    """NOTICE attributes the third-party code (Apache-2.0 section 4(d)); each distribution carries the same copy.

    Three distributions today (``rcp-ndcg``, ``rcp-ndcg-core``, ``rcp-ndcg-vllm``), plus the vLLM plugin
    distributions under ``packages/rcp-ndcg-vllm/plugins/*`` when the plugin lanes have landed them.
    """
    folders = [ROOT, ROOT / "packages" / "rcp-ndcg-core", ROOT / "packages" / "rcp-ndcg-vllm"]
    plugins = ROOT / "packages" / "rcp-ndcg-vllm" / "plugins"
    if plugins.is_dir():
        folders += sorted(
            folder for folder in plugins.iterdir() if folder.is_dir() and (folder / "pyproject.toml").is_file()
        )
    for folder in folders:
        project = tomllib.loads((folder / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        assert project["license-files"] == ["LICENSE", "NOTICE"], folder
    for name in ("LICENSE", "NOTICE"):
        for folder in folders[1:]:
            assert (folder / name).read_bytes() == (ROOT / name).read_bytes(), (
                f"{folder.relative_to(ROOT)}/{name} is stale"
            )
    assert "smart_resize" in (ROOT / "NOTICE").read_text(encoding="utf-8")
