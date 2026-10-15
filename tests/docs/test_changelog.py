"""The release notes carry each change once, under its own heading."""

from __future__ import annotations

from collections import Counter

from tests.docs._markdown import ROOT

CHANGELOG = ROOT / "CHANGELOG.md"


def test_no_change_appears_twice_in_the_release_notes() -> None:
    bullets = [line for line in CHANGELOG.read_text(encoding="utf-8").splitlines() if line.startswith("- ")]
    repeated = sorted(text for text, count in Counter(bullets).items() if count > 1)
    assert repeated == []


def test_the_text_limits_section_keeps_only_its_prose() -> None:
    lines = CHANGELOG.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("### Changed from the paper's code"))
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("#"):
            break
        body.append(line)
    assert [line for line in body if line.startswith("- ")] == []
