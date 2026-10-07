"""The extras in pyproject.toml say what the code needs, and ``rcp-ndcg doctor`` checks what the extras install."""

from __future__ import annotations

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
    # rcp-ndcg-vllm pins rcp-ndcg==<version> exactly (and the core through it): an install must resolve at every
    # instant of the rollout, so rcp-ndcg-vllm publishes only after both siblings are on PyPI.
    assert set(jobs["publish-vllm"]["needs"]) == {"build", "publish-core", "publish-rcp-ndcg"}
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


def test_the_release_workflow_pins_each_sibling_at_the_tags_version(tmp_path) -> None:
    """The release check requires ``rcp-ndcg-core==<tag>`` in this manifest and, where ``rcp-ndcg-vllm`` depends
    on ``rcp-ndcg``, exactly ``rcp-ndcg==<tag>`` there.

    The check runs the workflow's own step against manifests in ``tmp_path``; a manifest the release builds but
    cannot find is a failure ("nothing to check" is never a pass), and no dependency means nothing to pin.
    """
    workflow, _ = _release_workflow()
    steps = workflow["jobs"]["build"]["steps"]
    step = next(step for step in steps if "pins its sibling" in str(step.get("name", "")))
    body = str(step["run"]).split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]

    def check(
        manifest: str | None,
        core: str = '"rcp-ndcg-core==0.0.1"',
        version: str = "0.0.1",
        core_in_extras: bool = False,
        extra_core: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        tree = tmp_path / f"check-{check.calls:03d}-{version}"
        check.calls += 1
        package = tree / "packages" / "rcp-ndcg-vllm"
        package.mkdir(parents=True)
        declaration = (
            f"[project.optional-dependencies]\ndev = [{core}]" if core_in_extras else f"dependencies = [{core}]"
        )
        if extra_core is not None:
            declaration += f"\n[project.optional-dependencies]\nprobe = [{extra_core}]"
        (tree / "pyproject.toml").write_text(f'[project]\nname = "rcp-ndcg"\n{declaration}\n', encoding="utf-8")
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
    assert check('[project]\ndependencies = ["numpy"]\n').returncode == 0, "no dependency, nothing to pin"
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
    "datasets": "data",
    "mteb": "mteb",
    "transformers": "mteb",  # mteb's own dependency (uv.lock)
    "mcp": "direct:mcp",
    "rcp_ndcg_vllm": "direct:./packages/rcp-ndcg-vllm",  # example 09; the sibling installs from the checkout
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


def test_the_plugin_test_suites_run_in_ci() -> None:
    """Every test suite under ``packages/rcp-ndcg-vllm/plugins/*/tests`` runs in a CI job (six topk and three pplx
    modules executed nowhere before this). The plugins fold into rcp-ndcg-vllm with the layout move and then run
    under the package's own suite."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "packages/rcp-ndcg-vllm/plugins/*/tests" in ci, "a CI job must collect the plugins' test suites"
