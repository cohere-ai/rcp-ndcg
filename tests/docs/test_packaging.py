"""The extras in pyproject.toml say what the code needs, and ``rcp-ndcg doctor`` checks what the extras install."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from tests.docs._markdown import ROOT

PYPROJECT = tomllib.loads((ROOT / "rcp-ndcg" / "pyproject.toml").read_text(encoding="utf-8"))
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
        for folder in ("rcp-ndcg/src", "rcp-ndcg-core", "rcp-ndcg-vllm", "experiments", "examples")
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
CHECK_CONSTRAINTS = ROOT / ".github" / "scripts" / "check_constraints.py"


def _check_constraints_module():
    """The one implementation of the constraints comparison (CI and the release run this script, not a copy)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("check_constraints", CHECK_CONSTRAINTS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_constraints_check_is_semantic_not_textual(tmp_path: Path) -> None:
    """The check CI and the release run compares pins, not text (a newer uv's re-serialisation does not fail it;
    a moved, added or dropped pin does)."""
    committed = "torch==2.8.0 ; platform_system == 'Linux' and platform_machine == 'x86_64'\nnumpy==2.3.0\n"
    reserialised = (
        'torch==2.8.0;platform_machine == "x86_64" and platform_system == "Linux"  # via rcp-ndcg\n'
        "numpy==2.3.0; python_version >= '3.11'\n"
    )
    assert _check_constraints_module().pins(reserialised) == {"torch": "2.8.0", "numpy": "2.3.0"}
    base = tmp_path / "committed.txt"
    other = tmp_path / "exported.txt"
    base.write_text(committed, encoding="utf-8")

    def run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CHECK_CONSTRAINTS), str(other), str(base)], capture_output=True, text=True
        )

    other.write_text(reserialised, encoding="utf-8")
    assert run().returncode == 0, "the same pins re-serialised must compare equal"
    other.write_text(reserialised.replace("numpy==2.3.0", "numpy==2.4.0"), encoding="utf-8")
    assert run().returncode == 1, "a moved pin must be refused"
    other.write_text(reserialised + "scipy==1.16.0\n", encoding="utf-8")
    assert run().returncode == 1, "an added pin must be refused"


def test_ci_and_the_release_both_run_the_constraints_check() -> None:
    """One implementation, run on every pull request and at every release -- the tag cannot be the first run."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    _, release_text = _release_workflow()
    invocation = ".github/scripts/check_constraints.py"
    assert invocation in ci, "CI must run the constraints check"
    assert invocation in release_text, "the release must run the constraints check"


def test_the_constraints_file_is_the_locks_export_for_the_coordinators_extras() -> None:
    """The file a rendered job installs against pins exactly what uv.lock pins, for the extras it installs."""
    from rcp_ndcg.runners.script import COORDINATOR_EXTRAS

    text = CONSTRAINTS.read_text(encoding="utf-8")
    command = text.splitlines()[1].lstrip("# ").split()
    assert command == list(_check_constraints_module().EXPORT_ARGV), (
        "the header and the check script must record the same export command (one home)"
    )
    assert command[:2] == ["uv", "export"] and "--frozen" in command
    extras = [command[i + 1] for i, word in enumerate(command) if word == "--extra"]
    assert sorted(extras) == sorted(COORDINATOR_EXTRAS)
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked: dict[str, set[str]] = {}
    for package in lock["package"]:
        locked.setdefault(package["name"].lower(), set()).add(package["version"])
    pins = _check_constraints_module().pins(text)
    assert "torch" in pins and "rcp-ndcg" not in pins and "rcp-ndcg-core" not in pins
    stale = {name: pin for name, pin in pins.items() if pin not in locked.get(name, set())}
    assert not stale, f"regenerate requirements-constraints.txt with the command in its header: {stale}"
    uv = shutil.which("uv")
    if uv is not None:  # the exact export, when uv is at hand
        result = subprocess.run([sys.executable, str(CHECK_CONSTRAINTS)], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


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


#: The shared release checks the workflow's build job runs (and rc_build.sh calls too).
CHECK_RELEASE = ROOT / ".github" / "scripts" / "check_release.py"


def _release_tree(
    tmp_path: Path,
    versions: dict[str, str],
    *,
    core: str = '"rcp-ndcg-core==0.0.1"',
    vllm: str | None = '[project]\nname = "rcp-ndcg-vllm"\nversion = "0.0.1"\ndependencies = ["rcp-ndcg==0.0.1"]\n',
    core_in_extras: bool = False,
    extra_core: str | None = None,
) -> Path:
    """A fixture release tree: the four manifests (each at its ``versions`` entry), the citation and the
    given core/vllm declarations; ``vllm=None`` leaves the manifest the release builds absent."""
    tree = tmp_path / f"tree-{len(list(tmp_path.iterdir()))}"
    declaration = f"[project.optional-dependencies]\ndev = [{core}]" if core_in_extras else f"dependencies = [{core}]"
    if extra_core is not None:
        declaration += f"\n[project.optional-dependencies]\nprobe = [{extra_core}]"
    for package, version in versions.items():
        if package == "rcp-ndcg-vllm":
            continue  # written from the vllm parameter below (or left absent)
        directory = tree / package
        directory.mkdir(parents=True)
        if package == "rcp-ndcg":
            body = f'[project]\nname = "rcp-ndcg"\nversion = "{version}"\n{declaration}\n'
        else:
            body = f'[project]\nname = "{package}"\nversion = "{version}"\n'
        (directory / "pyproject.toml").write_text(body, encoding="utf-8")
    if vllm is not None:
        (tree / "rcp-ndcg-vllm").mkdir(parents=True, exist_ok=True)
        (tree / "rcp-ndcg-vllm" / "pyproject.toml").write_text(vllm, encoding="utf-8")
    (tree / "CITATION.cff").write_text(f"cff-version: 1.2.0\nversion: {versions['rcp-ndcg']}\n", encoding="utf-8")
    return tree


def _run_release_check(tree: Path, version: str = "0.0.1", *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECK_RELEASE), version, "--root", str(tree), *args], capture_output=True, text=True
    )


def _assert_one_environment_per_package(workflow: dict) -> None:
    """The release contract: one build, then one publish job per package, each in its own GitHub environment."""
    jobs = workflow["jobs"]
    assert workflow[True]["push"]["tags"] == ["v*"]  # YAML reads the key `on` as true
    assert set(jobs) == {"build", *RELEASE_PUBLISH_JOBS, "github-release"}
    build = str(jobs["build"])
    # The published packages are built by name (rcp-ndcg-vllm from its own directory, outside the uv
    # workspace); --all-packages is refused so a future workspace member (rcp-ndcg-test) is never swept
    # into the release build.
    assert "--all-packages" not in build, "the release builds the three published packages by name"
    # One --package per invocation (uv refuses a repeated --package): the build names the two workspace
    # members it publishes, so the unpublished test package -- also a member -- can never be swept in.
    for name in ("rcp-ndcg-core", "rcp-ndcg", "rcp-ndcg-vllm"):
        assert f"--package {name} --out-dir dist" in build, name
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
    # rcp-ndcg-vllm names no sibling pin (the lean package pins nothing), but every moment of the rollout
    # must install, so it publishes only after both siblings are on PyPI (PyPI files are immutable; a
    # published rcp-ndcg-vllm whose siblings failed would be uninstallable forever).
    assert set(jobs["publish-vllm"]["needs"]) == {"build", "publish-core", "publish-rcp-ndcg"}
    assert set(jobs["github-release"]["needs"]) == set(RELEASE_PUBLISH_JOBS)
    tokenised = {name for name, job in jobs.items() if (job.get("permissions") or {}).get("id-token") == "write"}
    assert tokenised == set(RELEASE_PUBLISH_JOBS), "id-token: write belongs to exactly the publish jobs"


def test_the_release_workflow_publishes_three_packages_one_environment_each() -> None:
    workflow, text = _release_workflow()
    assert "secrets." not in text  # trusted publishing: no token anywhere
    # The unpublished test package is never built or published: no build line, no publish job (the build
    # comment may name it as the member --all-packages would sweep in).
    assert "uv build --package rcp-ndcg-test" not in text
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


def test_the_release_workflow_pins_each_sibling_at_the_tags_version(tmp_path) -> None:
    """The release check requires ``rcp-ndcg-core==<tag>`` in this manifest and, where ``rcp-ndcg-vllm`` depends
    on ``rcp-ndcg``, exactly ``rcp-ndcg==<tag>`` there.

    The build job runs ``.github/scripts/check_release.py`` (the one home rc_build.sh shares); the test drives
    the script against manifests in ``tmp_path``: a manifest the release builds but cannot find is a failure
    ("nothing to check" is never a pass), and no dependency means nothing to pin.
    """
    workflow, _ = _release_workflow()
    steps = workflow["jobs"]["build"]["steps"]
    step = next(step for step in steps if "manifests, the citation and the pins" in str(step.get("name", "")))
    assert "check_release.py" in str(step["run"]), "the build job runs the shared release checks"

    def check(
        manifest: str | None,
        core: str = '"rcp-ndcg-core==0.0.1"',
        version: str = "0.0.1",
        core_in_extras: bool = False,
        extra_core: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        tree = _release_tree(
            tmp_path,
            dict.fromkeys(("rcp-ndcg", "rcp-ndcg-core", "rcp-ndcg-vllm", "rcp-ndcg-test"), version),
            core=core,
            vllm=manifest,
            core_in_extras=core_in_extras,
            extra_core=extra_core,
        )
        return _run_release_check(tree, version)

    exact = '[project]\nname = "rcp-ndcg-vllm"\nversion = "0.0.1"\ndependencies = ["rcp-ndcg==0.0.1"]\n'
    assert check(exact).returncode == 0, "an exact pin must pass"
    assert check('[project]\nversion = "0.0.1"\ndependencies = ["rcp-ndcg[calibrate]==0.0.1"]\n').returncode == 0
    assert check('[project]\nversion = "0.0.1"\ndependencies = ["rcp_ndcg==0.0.1"]\n').returncode == 0, (
        "a _ name normalises"
    )
    assert check('[project]\nversion = "0.0.1"\ndependencies = ["rcp-ndcg == 0.0.1"]\n').returncode == 0, (
        "spaces normalise"
    )
    assert check('[project]\nversion = "0.0.1"\ndependencies = ["rcp-ndcg (==0.0.1)"]\n').returncode == 0, (
        "parentheses normalise"
    )
    dotted = check('[project]\nversion = "0.0.1"\ndependencies = ["rcp.ndcg==0.0.1"]\n')
    assert dotted.returncode == 0 and "pins rcp-ndcg==0.0.1" in dotted.stdout, "a . name normalises (PEP 503)"
    dotted_loose = check('[project]\nversion = "0.0.1"\ndependencies = ["RCP.NDCG==0.0.2"]\n')
    assert dotted_loose.returncode == 1 and "==0.0.1" in dotted_loose.stderr, "a . name must still be checked"
    assert check('[project]\nversion = "0.0.1"\ndependencies = ["numpy"]\n').returncode == 0, (
        "no dependency, nothing to pin"
    )
    missing = check(None)
    assert missing.returncode == 1 and "pyproject.toml" in missing.stderr, (
        "a manifest the release builds may not be absent"
    )
    # The root manifest pins the core the same way: exact at the tag's version, however the TOML is spelled.
    assert check(exact, core='"rcp_ndcg_core == 0.0.1"').returncode == 0, "spaced and _ spelling normalise"
    assert (
        check(exact, core='"rcp-ndcg-core==0.0.1"  # a trailing comment in the list does not matter\n').returncode == 0
    ), "only the dependencies list is read"
    unpinned = check(exact, core='"numpy"')
    assert unpinned.returncode == 1 and "rcp-ndcg-core" in unpinned.stderr, "a root manifest must pin the core"
    extras_only = check(exact, core='"rcp-ndcg-core==0.0.1"', core_in_extras=True)
    assert extras_only.returncode == 1 and "rcp-ndcg-core" in extras_only.stderr, (
        "the required pin must be a runtime dependency: `pip install rcp-ndcg` resolves the core unpinned otherwise"
    )
    mixed = check(exact, core='"rcp-ndcg-core==0.0.1"', extra_core='"rcp-ndcg-core==0.0.2"')
    assert mixed.returncode == 1 and "==0.0.1" in mixed.stderr, (
        "an exact pin must not hide a stale one in an extra: `pip install rcp-ndcg[probe]` would not resolve"
    )
    stale = check(exact, core='"rcp-ndcg-core==0.0.2"')
    assert stale.returncode == 1 and "==0.0.1" in stale.stderr, "a wrong core pin must be refused"
    for wrong in ("rcp-ndcg>=0.0.1", "rcp-ndcg", "rcp-ndcg[calibrate]", "rcp-ndcg==0.0.2"):
        result = check(f'[project]\nversion = "0.0.1"\ndependencies = ["{wrong}"]\n')
        assert result.returncode == 1, f"{wrong} must be refused"
        assert "==0.0.1" in result.stderr


def test_the_release_workflow_checks_every_workspace_manifests_version(tmp_path) -> None:
    """The tag's version is that of all four workspace manifests, not only the three the release builds.

    The build job runs ``.github/scripts/check_release.py``; the test drives it against a tree of four
    manifests in ``tmp_path``: the unpublished ``rcp-ndcg-test`` included, because a stale version there
    leaves the lock and the stated release rule disagreeing.
    """
    workflow, _ = _release_workflow()
    steps = workflow["jobs"]["build"]["steps"]
    step = next(step for step in steps if "manifests, the citation and the pins" in str(step.get("name", "")))
    assert "check_release.py" in str(step["run"])
    packages = ("rcp-ndcg", "rcp-ndcg-core", "rcp-ndcg-vllm", "rcp-ndcg-test")

    def check(versions: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return _run_release_check(_release_tree(tmp_path, versions))

    at_tag = dict.fromkeys(packages, "0.0.1")
    assert check(at_tag).returncode == 0, "four manifests at the tag's version must pass"
    stale = dict(at_tag, **{"rcp-ndcg-test": "0.0.2"})
    result = check(stale)
    assert result.returncode == 1 and "rcp-ndcg-test" in result.stderr, "a stale fourth manifest must fail"
    missing = {package: "0.0.1" for package in packages[:-1]}
    assert check(missing).returncode == 1, "a missing workspace manifest must fail"


def test_the_release_workflow_check_fails_when_two_environments_are_swapped() -> None:
    workflow, _ = _release_workflow()
    workflow["jobs"]["publish-core"]["environment"] = "pypi"
    workflow["jobs"]["publish-rcp-ndcg"]["environment"] = "pypi-core"
    with pytest.raises(AssertionError):
        _assert_one_environment_per_package(workflow)


def test_both_distributions_ship_the_license_and_the_notice() -> None:
    """NOTICE attributes the third-party code (Apache-2.0 section 4(d)); each distribution carries the same copy.

    Four distributions (``rcp-ndcg``, ``rcp-ndcg-core``, ``rcp-ndcg-vllm`` and the unpublished
    ``rcp-ndcg-test``), one merged NOTICE byte-identical in all of them (docs-release Q4); the folded model
    plugins' attributions are part of it (their distributions are gone, layout-move item 3).
    """
    folders = [ROOT / "rcp-ndcg", ROOT / "rcp-ndcg-core", ROOT / "rcp-ndcg-vllm", ROOT / "rcp-ndcg-test"]
    for folder in folders:
        project = tomllib.loads((folder / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        assert project["license-files"] == ["LICENSE", "NOTICE"], folder
        for name in ("LICENSE", "NOTICE"):
            assert (folder / name).read_bytes() == (ROOT / name).read_bytes(), (
                f"{folder.relative_to(ROOT)}/{name} is stale"
            )
    assert "smart_resize" in (ROOT / "NOTICE").read_text(encoding="utf-8")


#: Repository paths NOTICE names (its upstream paths, such as ``src/transformers/...``, are not checked).
_NOTICE_PATH = re.compile(
    r"(?<![\w./-])((?:rcp-ndcg/src/rcp_ndcg|src/rcp_ndcg|rcp-ndcg-vllm/src|rcp-ndcg-test/tests|tests"
    r"|experiments|packages)/[A-Za-z0-9_./-]+\.(?:py|jinja))\b"
)

#: A folded model file's header names its upstream (a fold note, an upstream SPDX line or the
#: HuggingFace copyright): the attribution test collects these files automatically.
_THIRD_PARTY_HEADER = re.compile(r"Folded from|SPDX-FileCopyrightText|Copyright .* HuggingFace")


#: The audited derived files under the recipes and plugins: each ports, adapts or restates third-party code
#: (a model card's usage code, a checkpoint's remote code, vLLM internals), so NOTICE must name it. Every
#: template and every vendored module of a recipe directory is added automatically below.
_DERIVED_RECIPE_AND_PLUGIN_FILES = (
    "rcp-ndcg-test/tests/fixtures/recipes/fixture-vl-embed/reference.py",
    # the family references (decision 34: one reference file per family, parameterised by variant)
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/ctxl-rerank-v2-instruct-multilingual/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/harrier-oss-v1/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/jina-reranker-v3/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-embedding/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-reranker/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-vl-embedding/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-vl-reranker/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/topk-embed-v1/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/zerank/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/pplx/chunk_rule.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/pplx/hf_config.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/pplx/model.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/pplx/pooler.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/pplx/pooling_core.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/topk/config.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/topk/model.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/models/topk/weights.py",
)


def test_every_repository_path_the_notice_names_exists() -> None:
    """An attribution that names a moved or misspelled file attributes nothing."""
    named = set(_NOTICE_PATH.findall((ROOT / "NOTICE").read_text(encoding="utf-8")))
    assert named, "NOTICE names no repository path"
    missing = sorted(path for path in named if not (ROOT / path).is_file())
    assert not missing, f"NOTICE names files that do not exist: {missing}"


def test_the_notice_attributes_every_third_party_recipe_and_plugin_file() -> None:
    """Every recipe template, every vendored recipe module, every audited port and every folded model
    file is named in NOTICE."""
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    recipes = ROOT / "rcp-ndcg-vllm" / "src" / "rcp_ndcg_vllm" / "recipes"
    automatic = {
        path.relative_to(ROOT).as_posix()
        for path in recipes.glob("*/*")
        if path.suffix == ".jinja" or (path.suffix == ".py" and path.name != "reference.py")
    }
    models = ROOT / "rcp-ndcg-vllm" / "src" / "rcp_ndcg_vllm" / "models"
    folded = {
        path.relative_to(ROOT).as_posix()
        for path in models.rglob("*.py")
        if _THIRD_PARTY_HEADER.search("\n".join(path.read_text(encoding="utf-8").splitlines()[:15]))
    }
    assert folded, "no folded model file declares its upstream; update this test if the fold changed"
    expected = automatic | folded | set(_DERIVED_RECIPE_AND_PLUGIN_FILES)
    missing = sorted(path for path in expected if path not in notice)
    assert not missing, f"NOTICE does not attribute: {missing}"


# The dependency gates of tests/: every test module that gates on an import gets a CI job that opens the gate.
# "always" names a dependency of the package itself: every job's environment has it. The rest name the extra
# that provides the import (one home: rcp_ndcg.errors.EXTRA_FOR_MODULE), or "direct:" for a distribution that no
# extra names.
DEPENDENCY_GATES = {
    "PIL": "always",
    "pyarrow": "always",
    "fsspec": "always",
    "torch": "calibrate",
    "huggingface_hub": "hf",
    "pypdfium2": "data",
    "mteb": "mteb",
    "datasets": "mteb",  # the join gate's mteb 2.21 query path builds a datasets.Dataset
    "transformers": "mteb",  # mteb's own dependency (uv.lock)
    "mcp": "direct:mcp",
    "rcp_ndcg_vllm": "direct:./rcp-ndcg-vllm",  # example 09; the sibling installs from the checkout
}


def test_every_dependency_gate_in_tests_opens_in_ci() -> None:
    """No test gates on a package that no CI job installs (the two readers and the SDK round trip among them).

    A new ``pytest.importorskip`` must land in ``DEPENDENCY_GATES`` -- with the job that opens it, or the row
    is a lie -- and the table may never contradict the product's own ``EXTRA_FOR_MODULE``.
    """
    from rcp_ndcg.errors import EXTRA_FOR_MODULE

    found = set()
    for path in sorted((ROOT / "tests").rglob("*.py")):
        found.update(re.findall(r"importorskip\(\s*[\"']([A-Za-z0-9_]+)", path.read_text(encoding="utf-8")))
    assert found == set(DEPENDENCY_GATES), f"gates and the table disagree: {sorted(found ^ set(DEPENDENCY_GATES))}"
    core = {Requirement(spec).name.lower() for spec in PYPROJECT["project"]["dependencies"]}
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]
    requirement_name = re.compile(r"[A-Za-z0-9._-]+")
    deps_of = {
        p["name"].lower(): {
            requirement_name.match((d["name"] if isinstance(d, dict) else d).replace("_", "-").lower()).group()
            for d in p.get("dependencies", [])
        }
        for p in lock
    }
    for name, provider in DEPENDENCY_GATES.items():
        distribution = {"PIL": "pillow"}.get(name, name.replace("_", "-")).lower()
        if provider == "always":
            assert distribution in core, f"{name}: not a core dependency, so no job is guaranteed to have it"
        elif provider.startswith("direct:"):
            pass  # no extra names it: the install line below is its only home
        elif (mapped := EXTRA_FOR_MODULE.get(name)) is not None:
            assert mapped == provider, f"{name}: the table says {provider}, the product says {mapped}"
        else:
            # No product module imports it directly (nothing maps it): the provider extra must pull it in.
            assert distribution in deps_of.get(provider, set()), f"{name}: [{provider}] does not depend on it (uv.lock)"
    # Every provider the table names is opened by one CI job that also runs the whole tests/ tree: the tokens
    # come from that job's own run steps (a shell comment is not an install; tokens do not pool across jobs).
    workflow = __import__("yaml").safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    providers = {p.removeprefix("direct:") for p in DEPENDENCY_GATES.values() if p != "always"}
    openings = []
    for job in workflow["jobs"].values():
        code = "\n".join(
            "\n".join(ln for ln in str(step.get("run", "")).splitlines() if not ln.strip().startswith("#"))
            for step in job.get("steps", [])
        )
        if not re.search(r"pytest tests/(?:\s|$)", code):
            continue
        tokens = {t for line in re.findall(r"cpu-env\.sh([^\n]*)", code) for t in line.split() if t != "dev"}
        for spec in EXTRAS.get("dev", []):
            tokens |= set(Requirement(spec).extras or ())
        for install in re.findall(r"pip install([^\n]*)", code):
            tokens |= {t for t in install.split() if not t.startswith("-")}
        openings.append(tokens)
    assert any(providers <= tokens for tokens in openings), (
        f"no job both installs every gate's provider and runs tests/: {sorted(providers)} vs {openings}"
    )


def _index_installed_spec(command: str) -> str | None:
    """The ``rcp-ndcg-vllm`` spec one install command would resolve from an index, or ``None``.

    A spec is index-resolved exactly when it is the bare package name (optionally with extras) and the
    command is not editable: ``rcp-ndcg-vllm``. A path (``./rcp-ndcg-vllm``) or a built wheel
    (``dist/rcp_ndcg_vllm-0.0.1-py3-none-any.whl``) carries a separator or suffix and installs this
    checkout's bytes; ``-e`` makes even the bare name an editable install of the checkout.
    """
    tokens = command.split()
    if "-e" in tokens or "--editable" in tokens:
        return None
    for token in tokens:
        spec = token.strip("'\"")
        if spec.startswith(("rcp-ndcg-vllm", "rcp_ndcg_vllm")) and "/" not in spec and not spec.endswith(".whl"):
            return spec
    return None


def test_every_ci_install_of_the_serving_recipes_package_is_local() -> None:
    """Every CI job that installs ``rcp-ndcg-vllm`` installs THIS checkout's package, never an index
    release: editable (``-e``), a local path (``./rcp-ndcg-vllm``) or the wheel built in the same run
    (the ``wheel-smoke`` job's ``dist/rcp_ndcg_vllm-*.whl``).

    The recipes live beside the package in the checkout (the layout move made them package data), so
    the non-editable path install is the deliberate form there (a non-editable install without the
    package data left ``iter_recipes()`` without a recipe root, and the recipe-backed case validation
    failed the gated job, CI run 37822235213). An index install would test a published version instead
    of the tree. The synthetic cases below keep the check honest: it must be able to fail (the previous
    form of this test filtered on the package name and then asserted the package name was present, so
    it passed for ``uv pip install rcp-ndcg-vllm``).
    """
    # The check can fail: a bare name without -e is an index install, whatever the workflow.
    assert _index_installed_spec("uv pip install rcp-ndcg-vllm") == "rcp-ndcg-vllm"
    assert _index_installed_spec("uv pip install 'rcp-ndcg-vllm[test]'") == "rcp-ndcg-vllm[test]"
    # ...and the local forms it must accept really are accepted.
    assert _index_installed_spec("uv pip install -e rcp-ndcg-core -e rcp-ndcg -e 'rcp-ndcg-vllm[test]'") is None
    assert _index_installed_spec("uv pip install mcp ./rcp-ndcg-vllm") is None
    wheel_install = "uv pip install --python v/bin/python dist/rcp_ndcg_vllm-0.0.1-py3-none-any.whl"
    assert _index_installed_spec(wheel_install) is None

    workflow = __import__("yaml").safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    installs = [
        command.strip()
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        # a shell line continuation folds the next line into the same install command
        for command in re.findall(r"pip install[^\n]*", str(step.get("run", "")).replace("\\\n", " "))
        if "rcp-ndcg-vllm" in command or "rcp_ndcg_vllm" in command
    ]
    assert installs, "no CI job installs rcp-ndcg-vllm"
    offenders = [(command, spec) for command in installs if (spec := _index_installed_spec(command)) is not None]
    assert offenders == [], f"CI installs rcp-ndcg-vllm from an index, not this checkout: {offenders}"


def test_the_folded_models_test_suites_run_in_ci() -> None:
    """Every folded-model test suite runs in a CI job (six topk and three pplx modules executed nowhere
    before this). The plugins fold into rcp-ndcg-vllm (layout-move item 3); their tests live under
    ``rcp-ndcg-vllm/tests/models`` and run in the vllm-plugins job's own venv."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "rcp-ndcg-vllm/tests/models" in ci, "a CI job must collect the folded models' test suites"


def test_the_notice_summary_names_every_licence_its_entries_name() -> None:
    """The opening summary says which licences the entries carry; an entry under a licence the summary
    omits (an MIT checkpoint beside the Apache-2.0 ones) makes the summary false for that file."""
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    summary, entries = notice.split("\n\n", 2)[2].split("\n\n", 1)
    named = set(re.findall(r"Licence: ([A-Za-z0-9.-]+)", entries))
    assert named, "no NOTICE entry names its licence"
    missing = sorted(licence for licence in named if licence not in " ".join(summary.split()))
    assert not missing, f"the NOTICE summary omits the entries' licence(s) {missing}"


#: The pydantic config keys the records rely on; they first exist in pydantic 2.11.
PYDANTIC_211_KEYS = ("validate_by_name=", "validate_by_alias=", "serialize_by_alias=")
PYDANTIC_211 = Version("2.11")


def test_the_pydantic_floor_covers_the_config_keys_the_code_uses() -> None:
    """Pydantic before 2.11 ignores unknown ``ConfigDict`` keys without a warning.

    ``validate_by_name``, ``validate_by_alias`` and ``serialize_by_alias`` arrived in 2.11, and the
    records rely on them: with an older pydantic, ``Query(text=..., id=...)`` fails validation and every
    record with a ``schema`` alias is dumped under its field name (``schema_name``), violating the
    committed JSON Schemas. The declared floor must admit the version that implements them.
    """
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for folder in ("rcp-ndcg/src", "rcp-ndcg-core/src")
        for path in (ROOT / folder).rglob("*.py")
    )
    used = [key for key in PYDANTIC_211_KEYS if key in sources]
    assert used, "the scan found no pydantic 2.11 config keys; update this test if the models changed"
    for member in ("rcp-ndcg-core", "rcp-ndcg", "rcp-ndcg-vllm", "rcp-ndcg-test"):
        manifest = tomllib.loads((ROOT / member / "pyproject.toml").read_text(encoding="utf-8"))
        requirements = [Requirement(spec) for spec in manifest["project"]["dependencies"]]
        pydantic = [req for req in requirements if req.name.lower() == "pydantic"]
        assert len(pydantic) == 1, f"{member} declares pydantic {len(pydantic)} times"
        floors = [Version(spec.version) for spec in pydantic[0].specifier if spec.operator == ">="]
        assert floors and min(floors) >= PYDANTIC_211, f"{member} declares {pydantic[0]}"


def test_every_member_build_directory_is_ignored() -> None:
    """An in-tree build (``pip wheel ./rcp-ndcg``) leaves ``build/`` output behind.

    ``/rcp-ndcg-*/build/`` covers the three hyphenated members; the main distribution's directory must be
    ignored too, or its build output shows up in ``git status`` and in ``git add -A``.
    """
    for member in ("rcp-ndcg", "rcp-ndcg-core", "rcp-ndcg-vllm", "rcp-ndcg-test"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", f"{member}/build/lib/x.py"], cwd=ROOT, capture_output=True
        )
        assert result.returncode == 0, f"{member}/build/ is not ignored"


def test_the_wheel_smoke_gates_run_the_freeze_check_and_every_dry_run() -> None:
    """AGENTS.md's release gates: the ``--no-deps`` freeze check and a dry-run of every recipe, from a wheel.

    The CI tests standing in for them build the wheel with ``uv build --offline`` and turned a failed
    build into a skip, so both gates reported green without running. The freeze check and the
    every-recipe dry-run must be steps of the tag pipeline and of CI's wheel-smoke job.
    """
    for name in ("release.yml", "ci.yml"):
        workflow = __import__("yaml").safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))
        runs = [str(step.get("run", "")) for job in workflow["jobs"].values() for step in job.get("steps", [])]
        assert any("check_no_deps_install.py" in run and "dist/rcp_ndcg_vllm-*.whl" in run for run in runs), (
            f"{name}: no step runs the --no-deps freeze check on the built rcp-ndcg-vllm wheel"
        )
        assert any("iter_recipes()" in run and "--dry-run" in run and "for " in run for run in runs), (
            f"{name}: no step dry-runs every recipe from the built wheel"
        )


def test_the_wheel_build_fixture_fails_in_ci_instead_of_skipping(tmp_path: Path) -> None:
    """A wheel build that fails offline must fail CI, not skip the freeze and recipe checks away.

    The fixture is exercised with an empty uv cache and no network, which is the vllm-plugins job's
    situation: the build cannot resolve its build requirements there, and in CI that must be an error.
    """
    env = {**os.environ, "CI": "1", "UV_OFFLINE": "1", "UV_CACHE_DIR": str(tmp_path / "empty-uv-cache")}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "rcp-ndcg-vllm/tests/models/test_wheel_contract.py"
            "::test_the_wheel_is_pure_python_and_declares_only_the_image_deps",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode != 0, result.stdout
    assert "skipped" not in result.stdout


def test_the_vllm_plugins_job_fetches_the_offline_builds_requirements() -> None:
    """release-2: the fail-not-skip fixtures make the vllm-plugins job red unless its own steps put the
    wheel build's requirements in the uv cache.

    The fixtures build the wheel with ``uv build --offline``; the isolated build resolves
    ``setuptools>=77`` and ``wheel`` (``rcp-ndcg-vllm/pyproject.toml``'s build-system requires), the lock
    holds no ``wheel`` and setup-uv prunes cached wheels before saving, so the job must fetch them itself
    before the model tests run.
    """
    workflow = __import__("yaml").safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    job = workflow["jobs"]["vllm-plugins"]
    runs = [str(step.get("run", "")) for step in job["steps"]]
    installs = [run for run in runs if "uv pip install" in run]
    assert any("wheel" in install and "setuptools" in install for install in installs), (
        "the vllm-plugins job never installs the offline build's requirements (setuptools>=77, wheel): "
        "its wheel fixtures fail CI when the build cannot run"
    )


def test_the_gated_job_type_checks_with_the_integrations_installed() -> None:
    """The dev-only ``types`` job sees mteb, datasets and transformers as ``Unknown``.

    Every error the integrations carry is invisible there, so the gated job (which installs the
    ``mteb`` and ``data`` extras) must run basedpyright with them resolved.
    """
    workflow = __import__("yaml").safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    runs = [str(step.get("run", "")) for step in workflow["jobs"]["gated"]["steps"]]
    assert any("cpu-env.sh" in run and "mteb" in run for run in runs), "the gated job must install the mteb extra"
    assert any("basedpyright" in run for run in runs), "the gated job must type-check with the integrations resolved"


def test_every_network_marked_test_file_runs_in_a_network_enabled_job() -> None:
    """A network-marked test file no CI job opens is dead coverage: it skips everywhere and reports green.

    The check was dead in CI once (``tests/test_cases_package.py``); this keeps that fix from being the
    last one. Each marked file must be named by a step of a ci.yml job that sets
    ``RCP_NDCG_NETWORK_TESTS=1`` (a step env or an export in the step's shell).
    """
    workflow = __import__("yaml").safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    enabled_runs: list[str] = []
    for job in workflow["jobs"].values():
        steps = job.get("steps", [])
        enabled = any(
            str((step.get("env") or {}).get("RCP_NDCG_NETWORK_TESTS")) == "1"
            or "RCP_NDCG_NETWORK_TESTS=1" in str(step.get("run", ""))
            for step in steps
        )
        if enabled:
            enabled_runs.extend(str(step.get("run", "")) for step in steps)
    marked = [
        path.relative_to(ROOT).as_posix()
        for path in sorted((ROOT / "tests").rglob("*.py"))
        if re.search(r"^\s*@?pytest\.mark\.network\b", path.read_text(encoding="utf-8"), re.M)
    ]
    assert marked, "no network-marked test file found; update this test if the markers changed"

    def named_by(run: str, path: str) -> bool:
        if re.search(rf"(?<![\w./-]){re.escape(path)}(?![\w])", run):
            return True
        parent = path.rsplit("/", 1)[0]
        return re.search(rf"(?<![\w./-]){re.escape(parent)}(?![\w./-])", run) is not None

    uncovered = [path for path in marked if not any(named_by(run, path) for run in enabled_runs)]
    assert not uncovered, f"network-marked test files that no network-enabled CI job opens: {uncovered}"
