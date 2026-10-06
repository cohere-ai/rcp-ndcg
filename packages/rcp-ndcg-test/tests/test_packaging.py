"""The packaging contract of the unpublished test package.

The package is never published: no release build, no publish job, no reference from the published
metadata, no entry in the constraints file. The dev tooling reaches it through a dependency group and
the workspace; the release workflow builds exactly the three published distributions.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "packages" / "rcp-ndcg-test"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _release_workflow() -> str:
    return _read(ROOT / ".github" / "workflows" / "release.yml")


def test_the_release_workflow_never_builds_or_publishes_the_test_package() -> None:
    text = _release_workflow()
    assert "rcp-ndcg-test" not in text
    assert "packages/rcp-ndcg-test" not in text
    # The guard that keeps a future workspace member from being swept into the release build: the
    # three published packages are built by name, never with --all-packages.
    assert "--all-packages" not in text
    for package in ("rcp-ndcg", "rcp-ndcg-core", "rcp-ndcg-vllm"):
        assert package in text, f"the release must keep building {package}"


def test_no_published_package_names_the_test_package() -> None:
    for project in (ROOT, ROOT / "packages" / "rcp-ndcg-core", ROOT / "packages" / "rcp-ndcg-vllm"):
        data = tomllib.loads(_read(project / "pyproject.toml"))
        extras = data.get("project", {}).get("optional-dependencies", {})
        for extra, group_requirements in extras.items():
            assert not any("rcp-ndcg-test" in requirement for requirement in group_requirements), (
                f"the published extra {extra!r} of {project.name or 'the root project'} names "
                "the unpublished test package"
            )
    constraints = _read(ROOT / "requirements-constraints.txt")
    assert "rcp-ndcg-test" not in constraints, (
        "the constraints file installs on the node: the test package is not on PyPI"
    )


def test_the_root_reaches_it_through_a_dependency_group_and_the_workspace() -> None:
    data = tomllib.loads(_read(ROOT / "pyproject.toml"))
    groups = data.get("dependency-groups", {})
    assert any("rcp-ndcg-test" in requirement for requirement in groups.get("dev", [])), (
        "the dev dependency group is the one channel from the root's tooling to the unpublished package"
    )
    sources = data["tool"]["uv"]["sources"]
    assert sources.get("rcp-ndcg-test") == {"workspace": True}
    assert sources.get("rcp-ndcg-vllm", {}).get("path") == "packages/rcp-ndcg-vllm"


def test_the_lock_installs_it_into_the_dev_environment() -> None:
    lock = tomllib.loads(_read(ROOT / "uv.lock"))
    entries = {package["name"]: package for package in lock["package"]}
    entry = entries.get("rcp-ndcg-test")
    assert entry is not None and entry["version"] == "0.0.1", "uv.lock pins the workspace member"
    assert entry.get("source", {}).get("editable", "").endswith("packages/rcp-ndcg-test"), entry.get("source")
    dependency_names = {item["name"] for item in entry.get("dependencies", [])}
    assert {"rcp-ndcg", "rcp-ndcg-vllm"} <= dependency_names


def test_the_package_depends_on_the_product_and_the_harness_at_the_release_version() -> None:
    data = tomllib.loads(_read(PACKAGE / "pyproject.toml"))
    assert data["project"]["name"] == "rcp-ndcg-test"
    assert data["project"]["dependencies"] == [
        "rcp-ndcg==0.0.1",
        "rcp-ndcg-vllm==0.0.1",
        "numpy>=1.26",
        "pydantic>=2.0",
        "PyYAML>=6.0",
    ]


def test_it_carries_the_repositorys_license_and_notice() -> None:
    for name in ("LICENSE", "NOTICE"):
        assert (PACKAGE / name).read_bytes() == (ROOT / name).read_bytes(), f"packages/rcp-ndcg-test/{name} is stale"


def test_the_package_declares_its_types() -> None:
    assert (PACKAGE / "src" / "rcp_ndcg_test" / "py.typed").is_file()
