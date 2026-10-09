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


def _assert_one_environment_per_package(workflow: dict) -> None:
    """The release contract: one build, then one publish job per package, each in its own GitHub environment."""
    jobs = workflow["jobs"]
    assert workflow[True]["push"]["tags"] == ["v*"]  # YAML reads the key `on` as true
    assert set(jobs) == {"build", *RELEASE_PUBLISH_JOBS, "github-release"}
    build = str(jobs["build"])
    # One --package per invocation (uv refuses a repeated --package), one per published member: --all-packages
    # is refused so the unpublished fourth member (rcp-ndcg-test, also a workspace member) is never swept into
    # the release build.
    assert "--all-packages" not in build, "the release builds the three published packages by name"
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
    # rcp-ndcg-vllm pins rcp-ndcg==<version> exactly (and the core through it): an install must resolve at every
    # instant of the rollout, so rcp-ndcg-vllm publishes only after both siblings are on PyPI.
    assert set(jobs["publish-vllm"]["needs"]) == {"build", "publish-core", "publish-rcp-ndcg"}
    assert set(jobs["github-release"]["needs"]) == set(RELEASE_PUBLISH_JOBS)
    tokenised = {name for name, job in jobs.items() if (job.get("permissions") or {}).get("id-token") == "write"}
    assert tokenised == set(RELEASE_PUBLISH_JOBS), "id-token: write belongs to exactly the publish jobs"


def test_the_release_workflow_publishes_three_packages_one_environment_each() -> None:
    workflow, text = _release_workflow()
    assert "secrets." not in text  # trusted publishing: no token anywhere
    # The unpublished test package is never built or published: no build line, no publish job.
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
        package = tree / "rcp-ndcg-vllm"
        package.mkdir(parents=True)
        declaration = (
            f"[project.optional-dependencies]\ndev = [{core}]" if core_in_extras else f"dependencies = [{core}]"
        )
        if extra_core is not None:
            declaration += f"\n[project.optional-dependencies]\nprobe = [{extra_core}]"
        (tree / "rcp-ndcg").mkdir()
        (tree / "rcp-ndcg" / "pyproject.toml").write_text(
            f'[project]\nname = "rcp-ndcg"\n{declaration}\n', encoding="utf-8"
        )
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
    r"\b((?:rcp-ndcg/src/rcp_ndcg|rcp-ndcg-vllm/src|tests|experiments)/[A-Za-z0-9_./-]+\.(?:py|jinja))\b"
)

#: The audited derived files under the recipes and plugins: each ports, adapts or restates third-party code
#: (a model card's usage code, a checkpoint's remote code, vLLM internals), so NOTICE must name it. Every
#: template and every vendored module of a recipe directory is added automatically below.
_DERIVED_RECIPE_AND_PLUGIN_FILES = (
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/ctxl-rerank-v2-instruct-multilingual-1b/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/ctxl-rerank-v2-instruct-multilingual-2b/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/ctxl-rerank-v2-instruct-multilingual-6b/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/jina-embeddings-v5-text-small/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/jina-reranker-v3/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/octen-embedding-8b/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/pplx-embed-v2-context-9b-preview/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-embedding-0.6b/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-reranker-0.6b/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-reranker-4b/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-reranker-8b/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-vl-embedding-2b/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-vl-reranker-2b/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/topk-embed-v1-small/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/zembed-1-embedding/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/zerank-1-reranker/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/zerank-1-small-reranker/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/zerank-2-reranker/reference.py",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/ctxl-rerank-v2-instruct-multilingual-1b/template.jinja",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/ctxl-rerank-v2-instruct-multilingual-2b/template.jinja",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/ctxl-rerank-v2-instruct-multilingual-6b/template.jinja",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-reranker-0.6b/template.jinja",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-reranker-4b/template.jinja",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-reranker-8b/template.jinja",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-vl-reranker-2b/template.jinja",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/zerank-1-reranker/template.jinja",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/zerank-1-small-reranker/template.jinja",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/zerank-2-reranker/template.jinja",
    "rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/qwen3-vl-embedding-2b/qwen3_vl_embedding.py",
)


def test_the_plugin_test_suites_run_in_ci() -> None:
    """Every test suite under ``rcp-ndcg-vllm/plugins/*/tests`` runs in a CI job (six topk and three pplx
    modules executed nowhere before this). The plugins fold into rcp-ndcg-vllm with the layout move and then run
    under the package's own suite."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "rcp-ndcg-vllm/plugins/*/tests" in ci, "a CI job must collect the plugins' test suites"


def test_the_notice_summary_names_every_licence_its_entries_name() -> None:
    """The opening summary says which licences the entries carry; an entry under a licence the summary
    omits (an MIT checkpoint beside the Apache-2.0 ones) makes the summary false for that file."""
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    summary, entries = notice.split("\n\n", 2)[2].split("\n\n", 1)
    named = set(re.findall(r"Licence: ([A-Za-z0-9.-]+)", entries))
    assert named, "no NOTICE entry names its licence"
    missing = sorted(licence for licence in named if licence not in " ".join(summary.split()))
    assert not missing, f"the NOTICE summary omits the entries' licence(s) {missing}"
