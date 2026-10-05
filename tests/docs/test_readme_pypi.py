"""The README is PyPI's long description, and PyPI does not resolve relative paths.

Images (``<img src=`` and the ``<picture>`` ``srcset=``) and Markdown link targets must therefore be absolute:
``https://raw.githubusercontent.com/...`` for images and ``https://github.com/cohere-ai/rcp-ndcg/blob/main/...``
for links to repository files. Paths that resolve in a checkout (``docs/data.md``) render as broken links on PyPI.
"""

from __future__ import annotations

import re

from tests.docs._markdown import ROOT, links

ABSOLUTE = ("http://", "https://", "mailto:")

#: ``<img src="...">`` and ``<source ... srcset="...">`` of the README, values of both quote styles.
_ATTRIBUTES = re.compile(r"\b(?:src|srcset)=(?P<quote>['\"])(?P<url>[^'\"]*)(?P=quote)")


def _attribute_urls(text: str) -> list[str]:
    """Every URL an HTML ``src`` or ``srcset`` attribute names; a ``srcset`` may list several candidates."""
    urls = []
    for match in _ATTRIBUTES.finditer(text):
        for candidate in match.group("url").split(","):
            url = candidate.strip().split(" ")[0] if candidate.strip() else ""
            if url:
                urls.append(url)
    return urls


def test_readme_html_sources_are_absolute() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    sources = _attribute_urls(text)
    assert sources, "the README is expected to embed images"
    relative = [url for url in sources if not url.startswith(ABSOLUTE)]
    assert relative == [], f"PyPI does not resolve relative image paths: {relative}"


def test_readme_markdown_links_are_absolute() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = links(text)
    assert targets, "the README is expected to carry links"
    relative = [target for target in targets if not target.startswith((*ABSOLUTE, "#"))]
    assert relative == [], f"PyPI does not resolve relative link targets: {relative}"
