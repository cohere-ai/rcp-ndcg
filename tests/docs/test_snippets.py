"""Every Python snippet of the documentation runs.

The Python blocks of one page (``docs/`` and the core package's README) run in order, in one namespace, in a fresh
working directory. A block preceded by ``<!-- snippet: skip (reason) -->`` is not run (it needs an LLM endpoint or
your own files), and one preceded by ``<!-- snippet: network -->`` runs only with ``RCP_NDCG_NETWORK_TESTS=1``. In
the root README, every Python block is ``<!-- snippet: example examples/<file>.py -->``: a verbatim part of that
example, which ``test_examples`` runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.docs._markdown import ROOT, code_blocks

MARKERS = {None, "skip", "network", "example"}
PAGES = [*sorted((ROOT / "docs").rglob("*.md")), ROOT / "packages" / "rcp-ndcg-core" / "README.md"]


def _python_blocks(page: Path):
    return [b for b in code_blocks(page.read_text(encoding="utf-8")) if b.language in ("python", "py")]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_snippet_markers_are_known(page: Path) -> None:
    assert {b.marker for b in _python_blocks(page)} <= MARKERS


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_offline_snippets_run(page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = [b for b in _python_blocks(page) if b.marker is None]
    if not blocks:
        pytest.skip("no offline Python snippet")
    monkeypatch.chdir(tmp_path)
    namespace: dict = {"__name__": "__doc_snippet__"}
    for block in blocks:
        exec(compile(block.text, f"{page.relative_to(ROOT)}:{block.line}", "exec"), namespace)


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("RCP_NDCG_NETWORK_TESTS"), reason="set RCP_NDCG_NETWORK_TESTS=1 (HF Hub)")
@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_network_snippets_run(page: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    blocks = [b for b in _python_blocks(page) if b.marker == "network"]
    if not blocks:
        pytest.skip("no network snippet")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    if any("import mteb" in block.text for block in blocks):
        pytest.importorskip("mteb")
    monkeypatch.chdir(tmp_path)
    for block in blocks:
        exec(compile(block.text, f"{page.relative_to(ROOT)}:{block.line}", "exec"), {"__name__": "__doc_snippet__"})


OTHER_PAGES = [ROOT / "README.md"]


@pytest.mark.parametrize("page", OTHER_PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_readme_snippets_are_parts_of_examples(page: Path) -> None:
    for block in _python_blocks(page):
        assert block.marker == "example", f"line {block.line}: mark it <!-- snippet: example examples/<file>.py -->"
        example = ROOT / block.argument
        assert example.is_file(), f"line {block.line}: no example {block.argument!r}"
        assert block.text.strip() in example.read_text(encoding="utf-8"), f"line {block.line}: not in {block.argument}"
