"""The public surface is pinned: every change to it is a deliberate, reviewed diff.

What is pinned (``tests/contract/snapshots/``): the Python API (``__all__`` of the public modules listed in
``surface.PUBLIC_MODULES``, signatures, pydantic fields), the CLI tree (commands, options, defaults), the MCP tool
manifest, the exit codes, and the packaging (console scripts, entry-point groups, heavy imports). ``schemas/`` (the
exported JSON Schemas) is compared byte for byte with a fresh export. Every other module is internal.

The update flow::

    pytest tests/contract                       # fails with a classified diff (BREAKING / ADDITIVE)
    pytest tests/contract --update-snapshots    # rewrites snapshots/ and schemas/ (or RCP_NDCG_UPDATE_SNAPSHOTS=1)
    git diff tests/contract/snapshots schemas   # the reviewer reads this diff

A snapshot change is part of the change's review and belongs in the CHANGELOG under "Public surface".
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from tests.contract.diffing import classify
from tests.contract.surface import COLLECTORS, PUBLIC_MODULES, REPO, all_modules, collect_python, render

SNAPSHOTS = Path(__file__).parent / "snapshots"
SCHEMAS = REPO / "schemas"
UPDATE_HINT = "Run `pytest tests/contract --update-snapshots` and add a CHANGELOG.md entry under 'Public surface'."

#: Names exported with two different defining modules (two homes for one concept). May only shrink.
#: ``Family`` names two unrelated concepts: the IRT judgement family
#: (``rcp_ndcg_core.schemas.Family``) and the recipe family (``rcp_ndcg_vllm.recipe.Family``,
#: decision 34: one family, many sizes); the packages are separate distributions and the recipe
#: model is never imported by the product. ``apply`` is the patch interface, not a second concept:
#: every module in ``rcp_ndcg_vllm.patches.PATCH_MODULES`` implements it (one interface, one
#: implementation per patch), so each patch module defines it by design.
KNOWN_SECOND_HOMES: frozenset[str] = frozenset({"Family", "apply"})


def _update() -> bool:
    return os.environ.get("RCP_NDCG_UPDATE_SNAPSHOTS") == "1"


@pytest.fixture()
def update_snapshots(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-snapshots")) or _update()


@pytest.fixture(scope="module")
def python_surface() -> dict:
    return collect_python()


@pytest.fixture(scope="module")
def every_module() -> dict:
    """Every non-underscore module, public or internal: the hygiene checks cover them all."""
    return collect_python(all_modules())


@pytest.mark.parametrize("surface", sorted(COLLECTORS))
def test_surface_matches_snapshot(surface: str, update_snapshots: bool, python_surface: dict) -> None:
    current = python_surface if surface == "python_api" else COLLECTORS[surface]()
    path = SNAPSHOTS / f"{surface}.json"
    if update_snapshots:
        SNAPSHOTS.mkdir(parents=True, exist_ok=True)
        path.write_text(render(current), encoding="utf-8")
        return
    assert path.exists(), f"no snapshot for {surface}. {UPDATE_HINT}"
    pinned = json.loads(path.read_text(encoding="utf-8"))
    current = json.loads(render(current))
    if pinned != current:
        lines = classify(surface, pinned, current)
        shown = "\n".join(lines[:60]) + (f"\n... and {len(lines) - 60} more" if len(lines) > 60 else "")
        pytest.fail(f"the {surface} surface changed:\n{shown}\n{UPDATE_HINT}", pytrace=False)


def test_exported_schemas_are_current(tmp_path: Path, update_snapshots: bool) -> None:
    from rcp_ndcg.schemas import export

    export(tmp_path)
    if update_snapshots:
        shutil.rmtree(SCHEMAS, ignore_errors=True)
        shutil.copytree(tmp_path, SCHEMAS)
        return
    fresh = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    committed = {p.name: p.read_bytes() for p in SCHEMAS.glob("*.json")}
    assert sorted(fresh) == sorted(committed), f"exported vs committed schema files differ. {UPDATE_HINT}"
    stale = [name for name in fresh if fresh[name] != committed[name]]
    assert not stale, f"stale exported schemas: {stale}. {UPDATE_HINT}"


def test_the_public_modules_exist() -> None:
    assert set(PUBLIC_MODULES) <= set(all_modules())


def test_every_module_declares_all(every_module: dict) -> None:
    """A module declares its surface; a module that should not have one is named with a leading ``_``."""
    missing = sorted(name for name, entry in every_module.items() if entry.get("missing_all"))
    assert not missing, f"modules without __all__: {missing}"


def test_one_home_per_concept(every_module: dict) -> None:
    """A name may be re-exported, never re-defined: every exported name has one defining module."""
    homes: dict[str, set[str]] = {}
    for entry in every_module.values():
        for name, described in entry.get("all", {}).items():
            if "defined_in" in described:
                homes.setdefault(name, set()).add(described["defined_in"])
    second = {name for name, where in homes.items() if len(where) > 1}
    new = sorted(second - KNOWN_SECOND_HOMES)
    assert not new, {name: sorted(homes[name]) for name in new}
    fixed = sorted(KNOWN_SECOND_HOMES - second)
    assert not fixed, f"remove from KNOWN_SECOND_HOMES: {fixed}"
