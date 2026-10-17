"""The primitives page's comments match what its snippets return.

``tests/docs/test_snippets`` runs the page's Python blocks but never checks their comments; the comment on
``select_opponents`` states how many windows the call returns, so this test runs the page and compares.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.docs._markdown import ROOT, code_blocks

PAGE = ROOT / "docs" / "concepts" / "primitives.md"

_COUNT = re.compile(r"\b(\d+|one|two|three|four|five) windows\b")
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def _run_page(page: Path, tmp_path: Path, monkeypatch) -> dict:  # noqa: ANN001 - pytest's monkeypatch fixture
    """Run the page's Python blocks in order, as ``test_snippets`` does, and return their namespace."""
    monkeypatch.chdir(tmp_path)
    namespace: dict = {"__name__": "__doc_snippet__"}
    for block in code_blocks(page.read_text(encoding="utf-8")):
        if block.language in ("python", "py"):
            exec(compile(block.text, f"{page.name}:{block.line}", "exec"), namespace)
    return namespace


def test_the_select_opponents_comment_matches_the_returned_windows(
    tmp_path: Path,
    monkeypatch,  # noqa: ANN001 - pytest's monkeypatch fixture
) -> None:
    """The comment's window count is ``len(select_opponents(...))``, run against the tiny world."""
    namespace = _run_page(PAGE, tmp_path, monkeypatch)
    windows = namespace["windows"]
    comment = next(line for line in PAGE.read_text(encoding="utf-8").splitlines() if "select_opponents(" in line)
    match = _COUNT.search(comment)
    assert match, f"the comment states no window count: {comment}"
    claimed = _WORDS[match.group(1)] if match.group(1) in _WORDS else int(match.group(1))
    assert len(windows) == claimed, f"comment says {claimed} windows, the call returns {len(windows)}"
