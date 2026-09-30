"""Reading the repository's Markdown files: their links and their fenced code blocks."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO_URL = "https://github.com/cohere-ai/rcp-ndcg/blob/main/"

_FENCE = re.compile(r"^(```+|~~~+)\s*([\w-]*)")
_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_MARKER = re.compile(r"<!--\s*snippet:\s*(\w+)\s*(.*?)\s*-->")


def markdown_files() -> list[Path]:
    """Every Markdown file of the repository (tracked, or new and not ignored), relative to the root."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        files = [Path(line) for line in out.splitlines() if line]
    except (OSError, subprocess.CalledProcessError):  # not a git checkout: walk the tree
        files = [p.relative_to(ROOT) for p in ROOT.rglob("*.md") if ".venv" not in p.parts]
    return sorted(f for f in files if (ROOT / f).exists())


@dataclass(frozen=True)
class Block:
    """A fenced code block: its language, its text, the line it starts on, and its snippet marker and argument."""

    language: str
    text: str
    line: int
    marker: str | None
    argument: str = ""


def code_blocks(text: str) -> list[Block]:
    """The fenced code blocks of a Markdown text, in order.

    A block's marker is the ``<!-- snippet: <word> -->`` comment on the line right before its fence, if any.
    """
    blocks: list[Block] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = _FENCE.match(lines[i].strip())
        if not match:
            i += 1
            continue
        fence, language = match.group(1), match.group(2).lower()
        previous = lines[i - 1].strip() if i > 0 else ""
        marker = _MARKER.match(previous)
        body: list[str] = []
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith(fence):
            body.append(lines[j])
            j += 1
        indent = len(lines[i]) - len(lines[i].lstrip())
        blocks.append(
            Block(
                language,
                "\n".join(line[indent:] for line in body),
                i + 1,
                marker.group(1) if marker else None,
                marker.group(2) if marker else "",
            )
        )
        i = j + 1
    return blocks


def prose(text: str, *, keep_inline_code: bool = False) -> str:
    """The text without its fenced code blocks and (unless kept) its inline code spans."""
    out: list[str] = []
    in_block, fence = False, ""
    for line in text.splitlines():
        match = _FENCE.match(line.strip())
        if match and (not in_block or line.strip().startswith(fence)):
            in_block, fence = (not in_block), match.group(1)
            continue
        if not in_block:
            out.append(line if keep_inline_code else re.sub(r"`[^`]*`", "", line))
    return "\n".join(out)


def links(text: str) -> list[str]:
    """The link targets of a Markdown text, outside code."""
    return _LINK.findall(prose(text))


def slug(heading: str) -> str:
    """The anchor GitHub and MkDocs give a heading."""
    heading = re.sub(r"[`*_]", "", heading.strip().lower())
    heading = re.sub(r"[^\w\s-]", "", heading)
    return re.sub(r"\s", "-", heading)


def anchors(text: str) -> set[str]:
    """The heading anchors of a Markdown text."""
    return {slug(line.lstrip("#")) for line in prose(text).splitlines() if re.match(r"#{1,6}\s", line)}
