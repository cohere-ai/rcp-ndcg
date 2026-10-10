"""The public surface is documented: every pinned name has a page, and the generated pages are current.

The snapshot (``snapshots/python_api.json``, the ``__all__`` of ``surface.PUBLIC_MODULES``) is the definition of
what is public. ``rcp-ndcg docs api --out docs/reference/api`` generates one page per public module from that
snapshot and the modules' docstrings (each name with its kind, signature and one-line role), so the long tail of
the surface is documented mechanically; the curated pages under ``docs/`` carry the narrative. This module fails
when a pinned name is on no page under ``docs/``, when a pinned module has no page, or when the committed pages
differ from a fresh generation (so docs cannot drift from the surface).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from rcp_ndcg.support.api_docs import PUBLIC_MODULES, page_name, write_pages
from tests.contract.surface import REPO

SNAPSHOTS = Path(__file__).parent / "snapshots"
SNAPSHOT = SNAPSHOTS / "python_api.json"
PAGES = REPO / "docs" / "reference" / "api"
REGENERATE = "Regenerate with `rcp-ndcg docs api --out docs/reference/api`."


def _update() -> bool:
    return os.environ.get("RCP_NDCG_UPDATE_SNAPSHOTS") == "1"


@pytest.fixture()
def update_snapshots(request: pytest.FixtureRequest) -> bool:
    """The contract suite's own flag (defined in ``test_public_surface``): the generated pages are rewritten by
    the same ``--update-snapshots`` run that rewrites the snapshots."""
    return bool(request.config.getoption("--update-snapshots")) or _update()


def _snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def public_names() -> list[str]:
    """Every name the pinned Python surface exports, across the public modules."""
    return sorted({name for entry in _snapshot().values() for name in entry.get("all", {})})


def test_every_public_name_is_documented() -> None:
    """Every pinned name is mentioned on a page under ``docs/`` (the generated pages cover the long tail)."""
    names = public_names()
    assert names, "the python_api snapshot pins no names; the check would be vacuous"
    text = "\n".join(path.read_text(encoding="utf-8") for path in sorted((REPO / "docs").rglob("*.md")))
    missing = [name for name in names if not re.search(rf"\b{re.escape(name)}\b", text)]
    assert not missing, f"public names on no docs page: {missing}. {REGENERATE}"


def test_the_generated_pages_cover_the_pinned_modules() -> None:
    """One generated page per public module, and no page without a module."""
    committed = sorted(path.name for path in PAGES.glob("*.md"))
    assert committed == sorted(page_name(module) for module in PUBLIC_MODULES), REGENERATE


def test_the_generated_pages_are_current(tmp_path: Path, update_snapshots: bool) -> None:
    """A fresh generation equals the committed pages, byte for byte (the docs cannot drift from the surface)."""
    if update_snapshots:  # the contract suite's own flag rewrites its generated artifacts, pages included
        write_pages(PAGES, snapshot=SNAPSHOT)
        return
    written = write_pages(tmp_path, snapshot=SNAPSHOT)
    stale = [path.name for path in written if (PAGES / path.name).read_bytes() != path.read_bytes()]
    assert not stale, f"stale generated API pages: {stale}. {REGENERATE}"


def test_the_pinned_modules_are_the_public_modules() -> None:
    """The snapshot pins exactly the modules :data:`PUBLIC_MODULES` names (the one list of the surface)."""
    assert sorted(_snapshot()) == sorted(PUBLIC_MODULES)
