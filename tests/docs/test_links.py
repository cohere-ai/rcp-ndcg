"""Every link in the repository's Markdown resolves, and the documentation navigation lists every page."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.docs._markdown import REPO_URL, ROOT, anchors, links, markdown_files

FILES = markdown_files()


def _broken_links(path: Path, text: str | None = None) -> list[str]:
    """The link targets of ``path`` (or of ``text``, read as if it were at ``path``) that do not resolve."""
    text = (ROOT / path).read_text(encoding="utf-8") if text is None else text
    broken: list[str] = []
    for target in links(text):
        if target.startswith(REPO_URL):
            target_path, _, anchor = target[len(REPO_URL) :].partition("#")
            resolved = ROOT / target_path
        elif target.startswith(("http://", "https://", "mailto:")):
            continue  # external: not checked offline
        else:
            target_path, _, anchor = target.partition("#")
            resolved = (ROOT / path).parent / target_path if target_path else ROOT / path
        if not resolved.exists():
            broken.append(f"{target} (no such file)")
        elif anchor and resolved.suffix == ".md" and anchor not in anchors(resolved.read_text(encoding="utf-8")):
            broken.append(f"{target} (no such heading)")
    return broken


@pytest.mark.parametrize("path", FILES, ids=str)
def test_every_link_resolves(path: Path) -> None:
    assert _broken_links(path) == []


def test_a_broken_link_is_reported() -> None:
    text = "[a](data.md) [b](data.md#loading-a-dataset) [x](missing.md) [y](data.md#nope) `[z](inline-code.md)`\n"
    assert _broken_links(Path("docs/index.md"), text) == ["missing.md (no such file)", "data.md#nope (no such heading)"]


def _nav_pages(entries: list) -> list[str]:
    pages: list[str] = []
    for entry in entries:
        for value in entry.values() if isinstance(entry, dict) else [entry]:
            pages.extend(_nav_pages(value) if isinstance(value, list) else [value])
    return pages


def test_the_navigation_lists_every_page_and_only_existing_ones() -> None:
    config = yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))
    nav = _nav_pages(config["nav"])
    pages = sorted(str(p.relative_to(ROOT / "docs")) for p in (ROOT / "docs").rglob("*.md"))
    assert sorted(nav) == pages
