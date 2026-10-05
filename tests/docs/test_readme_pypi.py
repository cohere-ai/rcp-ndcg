"""The README is PyPI's long description, and PyPI does not resolve relative paths.

Images (``<img src=`` and the ``<picture>`` ``srcset=``, quoted or unquoted) and Markdown link targets — regular
links and ``![alt](...)`` images alike — must therefore be absolute: ``https://raw.githubusercontent.com/...`` for
images and ``https://github.com/cohere-ai/rcp-ndcg/blob/main/...`` for links to repository files. Paths that resolve
in a checkout (``docs/data.md``) render as broken links on PyPI.
"""

from __future__ import annotations

import re

from tests.docs._markdown import ROOT, links, prose

ABSOLUTE = ("http://", "https://", "mailto:")

#: ``<img src=...>`` and ``<source ... srcset=...>`` of the README, in both quote styles and unquoted.
_ATTRIBUTES = re.compile(r"\b(?:src|srcset)=" r"(?:\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)'|(?P<bare>[^\s>]+))")

#: Markdown images (``![alt](target)``): :func:`tests.docs._markdown.links` deliberately skips them.
_IMAGES = re.compile(r"!\[[^\]]*\]\((?P<url>[^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _attribute_urls(text: str) -> list[str]:
    """Every URL an HTML ``src`` or ``srcset`` attribute names; a ``srcset`` may list several candidates."""
    urls = []
    for match in _ATTRIBUTES.finditer(text):
        value = match.group("double") or match.group("single") or match.group("bare") or ""
        for candidate in value.split(","):
            url = candidate.strip().split(" ")[0] if candidate.strip() else ""
            if url:
                urls.append(url)
    return urls


def _image_urls(text: str) -> list[str]:
    """Every target a Markdown image names, outside code (``links`` skips images, so they are scanned here)."""
    return _IMAGES.findall(prose(text))


def test_readme_html_sources_are_absolute() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    sources = _attribute_urls(text)
    assert sources, "the README is expected to embed images"
    relative = [url for url in sources if not url.startswith(ABSOLUTE)]
    assert relative == [], f"PyPI does not resolve relative image paths: {relative}"


def test_readme_markdown_links_are_absolute() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = links(text) + _image_urls(text)
    assert targets, "the README is expected to carry links"
    relative = [target for target in targets if not target.startswith((*ABSOLUTE, "#"))]
    assert relative == [], f"PyPI does not resolve relative link or image targets: {relative}"
