"""Every documented harness command parses.

``test_commands`` checks the ``rcp-ndcg`` command tree only. The pages that drive the unpublished harness
(``python -m rcp_ndcg_test.<tool> ...``) are shell blocks nothing validates, so this module calls each tool's
own ``main`` with the documented argv, placeholders replaced, and the tool's run stubbed out: a documented
command whose flags the parser refuses fails here, not at the reader's terminal.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

from tests.docs._markdown import ROOT, code_blocks, markdown_files

SHELL = {"bash", "sh", "shell", "console", "zsh"}
PLACEHOLDER = re.compile(r"<[^>]*>")
EQUIVALENCE = ["python", "-m", "rcp_ndcg_test.equivalence"]


def _documented_commands(prefix: list[str]) -> list[tuple[Path, int, list[str]]]:
    """The argv after ``prefix`` in every shell block that runs it (continuation lines joined, comments dropped)."""
    commands: list[tuple[Path, int, list[str]]] = []
    for path in markdown_files():
        text = (ROOT / path).read_text(encoding="utf-8")
        for block in code_blocks(text):
            if block.language not in SHELL:
                continue
            for line in block.text.replace("\\\n", " ").splitlines():
                try:
                    tokens = shlex.split(PLACEHOLDER.sub("dummy", line), comments=True)
                except ValueError:
                    continue
                if tokens[: len(prefix)] == prefix:
                    commands.append((path, block.line, tokens[len(prefix) :]))
    return commands


def test_documented_equivalence_commands_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented ``python -m rcp_ndcg_test.equivalence`` argv parses (stages and placeholders included)."""
    import rcp_ndcg_test.equivalence.__main__ as equivalence

    monkeypatch.setattr(equivalence, "load_recipe", lambda recipe: object())
    monkeypatch.setattr(equivalence, "run", lambda *args, **kwargs: {"recipe": "dummy", "passed": True})
    commands = _documented_commands(EQUIVALENCE)
    assert commands, "no documented equivalence command to check"
    failures: list[str] = []
    for path, line, argv in commands:
        try:
            code = equivalence.main(argv)
        except SystemExit as exit_code:  # argparse refused the documented command
            failures.append(f"{path}:{line}: exit {exit_code.code} for {' '.join(argv)}")
        else:
            if code != 0:
                failures.append(f"{path}:{line}: exit {code} for {' '.join(argv)}")
    assert failures == [], f"documented commands do not parse: {failures}"
