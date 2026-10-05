"""The README is PyPI's long description, and PyPI does not resolve relative paths.

Images (``<img src=`` and the ``<picture>`` ``srcset=``, quoted or unquoted, ``<a href=`` too) and Markdown link
targets — regular links, ``![alt](...)`` images and ``[text]: target`` reference definitions alike — must be
absolute:
``https://raw.githubusercontent.com/...`` for images and ``https://github.com/cohere-ai/rcp-ndcg/blob/main/...``
for links to repository files. Paths that resolve in a checkout (``docs/data.md``) render as broken links on PyPI.
All scans run outside code (fenced blocks and inline code), which PyPI does not render as HTML or links.
"""

from __future__ import annotations

import re

from tests.docs._markdown import ROOT, links, prose

ABSOLUTE = ("http://", "https://", "mailto:")

#: ``<img src=...>``, ``<source ... srcset=...>`` and ``<a href=...>`` (``xlink:href=`` included) of the README, in
#: both quote styles and unquoted, with optional spaces around ``=``; the negative lookbehind keeps ``data-src=``
#: and friends out while still matching at the very start of the text.
_ATTRIBUTES = re.compile(
    r"(?<![\w-])(?:src|srcset|href)\s*=\s*"
    r"(?:\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)'|(?P<bare>[^\s>]+))"
)

#: Markdown images (``![alt](target)``): :func:`tests.docs._markdown.links` deliberately skips them.
_IMAGES = re.compile(r"!\[[^\]]*\]\((?P<url>[^)\s]+)(?:\s+\"[^\"]*\")?\)")

#: Markdown link-reference definitions (``[text]: target``): neither ``links`` nor ``_IMAGES`` scans them.
_REFS = re.compile(r"(?m)^\s{0,3}\[[^\]]+\]:\s+(?:<)?(?P<url>[^\s>]+)")


def _attribute_urls(text: str) -> list[str]:
    """Every URL an HTML ``src``, ``srcset`` or ``href`` attribute names; a ``srcset`` may list several candidates."""
    urls = []
    for match in _ATTRIBUTES.finditer(text):
        value = match.group("double") or match.group("single") or match.group("bare") or ""
        for candidate in value.split(","):
            url = candidate.strip().split(" ")[0] if candidate.strip() else ""
            if url:
                urls.append(url)
    return urls


def _image_urls(text: str) -> list[str]:
    """Every target a Markdown image or link-reference definition names, outside code."""
    return _IMAGES.findall(prose(text)) + _REFS.findall(prose(text))


def test_readme_html_sources_are_absolute() -> None:
    text = prose((ROOT / "README.md").read_text(encoding="utf-8"))
    sources = _attribute_urls(text)
    assert sources, "the README is expected to embed images"
    relative = [url for url in sources if not url.startswith((*ABSOLUTE, "#"))]
    assert relative == [], f"PyPI does not resolve relative image paths: {relative}"


def test_readme_markdown_links_are_absolute() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = links(text) + _image_urls(text)
    assert targets, "the README is expected to carry links"
    relative = [target for target in targets if not target.startswith((*ABSOLUTE, "#"))]
    assert relative == [], f"PyPI does not resolve relative link or image targets: {relative}"
