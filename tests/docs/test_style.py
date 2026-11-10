"""The documentation's prose is in Simplified Technical English (ASD-STE100): 20 words per sentence at most.

The check reads every hand-written page in ``docs/``; the generated API pages are skipped. It skips the fenced
code blocks, the tables, the headings and the HTML, and it treats each bullet item as its own sentence. An
inline code span counts as one word, because a backticked command or identifier is one technical term. A
sentence that a test pins is in ``ALLOWED`` with its reason, so the allowlist is the record of the exceptions,
not a silent filter.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
GENERATED = DOCS / "reference" / "api"

#: The sentences a test pins word for word, with the reason. A sentence here is not a style defect: another
#: test asserts its text, so a shorter sentence would fail that test.
ALLOWED = {
    # tests/docs/test_image_processor_docs.py matches this sentence's three model names literally.
    "the test suite checks this over a grid",
}

SENTENCE = re.compile(r"(?<=[.!?])\s+")
BULLET = re.compile(r"^(?:[-*]|\d+\.)\s+")
CODE_SPAN = re.compile(r"`[^`\n]*`")


def _mask_code(text: str) -> str:
    """A code span is one word, so its own characters must not split a sentence. The mask keeps the text's
    length, so a sentence's offsets are the same in the masked and the original text."""
    return CODE_SPAN.sub(lambda match: "X" * len(match.group(0)), text)


def _words(sentence: str) -> list[str]:
    """The sentence's words: an inline code span is one technical term, not a run of words."""
    return CODE_SPAN.sub("CODE", sentence).split()


def _prose(text: str) -> list[str]:
    """The page's prose paragraphs: the fenced blocks, the tables, the headings, the HTML, the display math
    and the code-only lines are not prose, and a paragraph is joined from its wrapped lines. Each bullet item
    starts its own paragraph, so a list is not one long sentence."""
    paragraphs: list[str] = []
    block: list[str] = []
    fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            fence = not fence
            continue
        if fence or not stripped or stripped.startswith(("#", "|", "<", "$$")):
            if block:
                paragraphs.append(" ".join(block))
                block = []
            continue
        if stripped.count("`") > 4 and len(stripped) - stripped.count("`") < len(stripped) // 2:
            # A signature or a usage line is code in prose clothing: the backticks hold most of the line.
            continue
        if BULLET.match(stripped):
            if block:
                paragraphs.append(" ".join(block))
                block = []
            stripped = BULLET.sub("", stripped)
        block.append(stripped)
    if block:
        paragraphs.append(" ".join(block))
    return paragraphs


def _sentences(paragraph: str) -> list[str]:
    """A sentence ends in a terminal punctuation mark; a bullet label or a fragment has none and is not one."""
    masked = _mask_code(paragraph)
    sentences: list[str] = []
    start = 0
    for match in SENTENCE.finditer(masked):
        sentence = paragraph[start : match.start()].strip()
        if sentence and sentence[-1] in ".!?":
            sentences.append(sentence)
        start = match.end()
    sentence = paragraph[start:].strip()
    if sentence and sentence[-1] in ".!?":
        sentences.append(sentence)
    return sentences


def test_every_docs_sentence_is_at_most_twenty_words() -> None:
    over: dict[str, list[str]] = {}
    for path in sorted(DOCS.rglob("*.md")):
        if path.is_relative_to(GENERATED):
            continue
        for paragraph in _prose(path.read_text(encoding="utf-8")):
            for sentence in _sentences(paragraph):
                words = _words(sentence)
                if len(words) <= 20:
                    continue
                if any(allowed in sentence.lower() for allowed in ALLOWED):
                    continue
                over.setdefault(str(path.relative_to(ROOT)), []).append(f"{len(words)}: {sentence}")
    assert not over, "docs sentences over 20 words (ASD-STE100; see docs/reference/writing-style.md): " + repr(over)


def test_the_style_page_states_the_limit_and_the_check() -> None:
    """The rule has one home, and it names its own enforcement."""
    page = (DOCS / "reference" / "writing-style.md").read_text(encoding="utf-8")
    assert "ASD-STE100" in page
    assert "20 words" in page
    assert "tests/docs/test_style.py" in page


def test_a_long_sentence_is_reported() -> None:
    """The check is real: a sentence over the limit fails, and the allowlist does not hide it."""
    paragraph = " ".join(["word"] * 21) + "."
    assert len(_words(_sentences(paragraph)[0])) == 21
    assert not any(allowed in paragraph.lower() for allowed in ALLOWED)


def test_an_inline_code_span_counts_as_one_word() -> None:
    """A backticked command or identifier is one technical term, not a run of words."""
    sentence = "Run `rcp-ndcg run start my_run.yaml --runner kubernetes --dry-run` now."
    assert len(_words(sentence)) == 3
    assert len(_sentences(sentence)) == 1


def test_each_bullet_item_is_its_own_sentence() -> None:
    """A list is not one long sentence: the block flushes at each bullet line."""
    page = "- first item.\n- second item.\n"
    assert _prose(page) == ["first item.", "second item."]
