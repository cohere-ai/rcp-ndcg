"""Deterministic long-text builder for the qwen3-vl-embedding-2b generated cases.

Writes the four long-text case files (text-long-under, text-long-over,
text-mixed-length-batch, image-text-long-over). Inputs are built from a seeded
generator - random.Random(seed) over the fixed WORDLIST below, one sentence per
13 words - and measured against the recipe's tokenizer
(Qwen/Qwen3-VL-Embedding-2B, revision 9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda;
pass its tokenizer.json via --tokenizer; download it into lane scratch, never
into the repo). Re-running this script with the same tokenizer regenerates the
case files byte-for-byte and re-verifies every token count.

Lengths are measured on the full rendered prompt, frame included:

    system header (14 tokens) + content + assistant tail (5) + anchor (1)

The recipe's frame adds 20 fixed tokens (measured with this tokenizer), so the
content budget under client.max_tokens = 8192 is 8172. The long_under document
targets a full prompt of 8150 (within 5% under the budget, no cut); the
long_over documents target 9600 (text) and ~8850 (image+text) so the client's
declared cut fires and the anchor must survive it.

Usage:
    python make_texts.py --tokenizer <scratch>/tokenizer/tokenizer.json
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

SEED = 20261006
RECIPE = "qwen3-vl-embedding-2b"
MAX_TOKENS = 8192
TOKENIZER_REVISION = "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"
CASE_DIR = Path(__file__).resolve().parent

WORDS = (
    "time year people way day man thing woman life child world school state family "
    "student group country problem hand part place case week company system program "
    "question work government number night point home water room mother area money "
    "story fact month lot right study book eye job word business issue side kind "
    "head house service friend father power hour game line end member law car city "
    "community name president team minute idea kid body information back parent "
    "face others level office door health person art war history party result "
    "change morning reason research girl guy moment air teacher force education"
).split()

QUERIES = {
    "long_under": "What was the economic impact of the policy change described in the report?",
    "long_over": "Summarise the main argument of the accompanying document.",
    "mixed_length": ["When did the project begin?", "Who reviewed the final report?"],
    "image_text_over": "What does this page show?",
}


def tokenizer_from(path: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def special(tokenizer, name: str) -> str:
    """The added token spelled "<|" + name + "|>", read from the tokenizer
    (special tokens are never typed literally in this file)."""
    want = "<|" + name + "|>"
    for token in tokenizer.get_added_tokens_decoder().values():
        if token.content == want:
            return token.content
    raise SystemExit(f"tokenizer lacks the special token {want!r}")


def frame(instruction: str, tokenizer) -> tuple[str, str, int]:
    """The recipe's fixed frame: returns (head, tail, fixed_tokens). The anchor
    (the tokenizer post-processor's end-of-text token, appended with
    add_special_tokens=true) is counted in fixed_tokens."""
    im_start, im_end = special(tokenizer, "im_start"), special(tokenizer, "im_end")
    head = f"{im_start}system\n{instruction}{im_end}\n{im_start}user\n"
    tail = f"{im_end}\n{im_start}assistant\n"
    anchor = len(tokenizer.encode("", add_special_tokens=True).ids)
    fixed = len(tokenizer.encode(head, add_special_tokens=False).ids) + len(
        tokenizer.encode(tail, add_special_tokens=False).ids
    )
    return head, tail, fixed + anchor


def prompt_tokens(text: str, tokenizer, head: str, tail: str) -> int:
    """Full rendered prompt: head + content + tail, plus the post-processor's
    anchor (add_special_tokens=true appends it)."""
    return len(tokenizer.encode(head + text + tail, add_special_tokens=True).ids)


def content_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def make_text(seed: int, prompt_target: int, tokenizer, head: str, tail: str) -> tuple[str, int]:
    """random.Random(seed) text whose full prompt token count is the largest
    count <= prompt_target. Returns (text, prompt_tokens)."""
    rng = random.Random(seed)
    words: list[str] = []
    while True:
        word = rng.choice(WORDS)
        if len(words) % 13 == 12:
            word = word.capitalize() + "."
        candidate = words + [word]
        if prompt_tokens(" ".join(candidate), tokenizer, head, tail) > prompt_target:
            break
        words = candidate
    return " ".join(words), prompt_tokens(" ".join(words), tokenizer, head, tail)


def wrap(text: str, width: int = 100) -> str:
    """Greedy word wrap: break only at spaces so a YAML folded block scalar
    (single newlines fold back to one space) reconstitutes the exact text."""
    lines: list[str] = []
    current = ""
    for word in text.split(" "):
        candidate = f"{current} {word}" if current else word
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def emit(path: Path, body: list[str], long_texts: dict[str, str]) -> None:
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    import yaml

    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    flat = {doc["id"]: doc.get("text") or "" for doc in parsed["inputs"]["documents"]}
    for doc_id, text in long_texts.items():
        assert flat[doc_id] == text, f"{path.name}: document {doc_id} did not round-trip"
    print(f"wrote {path.relative_to(CASE_DIR)}")


def expected_block() -> list[str]:
    return [
        "expected:",
        "  kind: similarity_matrix",
        "  values: null",
        "  tolerance: {abs: 0.002}",
        "  origin: reference",
        "  status: pending_gpu",
    ]


def notes_block(notes: str) -> list[str]:
    lines = ["notes: >-"]
    lines.extend("  " + part for part in wrap(notes, 96).splitlines())
    return lines


def doc_block(doc_id: str, text: str, image: str | None = None) -> list[str]:
    lines = [f"  - id: {doc_id}"]
    if image is not None:
        lines.append(f"    image: {image}")
    lines.append("    text: >-")
    lines.extend("      " + part for part in wrap(text).splitlines())
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tokenizer",
        default=os.environ.get("RCP_QWEN3VL_TOKENIZER", ""),
        help="path to the pinned revision's tokenizer.json (download to lane scratch)",
    )
    args = ap.parse_args()
    if not args.tokenizer:
        raise SystemExit("pass --tokenizer <path to the pinned revision's tokenizer.json>")
    tokenizer = tokenizer_from(args.tokenizer)
    instruction = "Represent the user's input."
    head, tail, fixed = frame(instruction, tokenizer)
    print(f"frame fixed tokens (head+tail+anchor): {fixed}")

    cases: list[tuple[Path, list[str], dict[str, str]]] = []

    # --- text-long-under: one document just under the budget (no cut) ---------
    text, count = make_text(SEED + 1, 8150, tokenizer, head, tail)
    cases.append(
        (
            CASE_DIR / "text-long-under.yaml",
            [
                f"id: {RECIPE}/text-long-under",
                f"recipe: {RECIPE}",
                "role: embed",
                "source:",
                "  kind: generated",
                "strata:",
                "  modality: text",
                "  length: long_under",
                "  batch: single",
                "inputs:",
                f"  instruction: {instruction}",
                "  queries:",
                "    - id: q1",
                f"      text: {QUERIES['long_under']}",
                "  documents:",
                *doc_block("d1", text),
                *expected_block(),
                *notes_block(
                    f"Generated long_under cell. Seeded generator (random.Random({SEED + 1}) over the "
                    f"WORDLIST in make_texts.py, one sentence per 13 words); lengths measured with the "
                    f"recipe's tokenizer at revision {TOKENIZER_REVISION}. Full prompt = {count} tokens "
                    f"(content {content_tokens(text, tokenizer)} + fixed frame {fixed}), inside 5% under "
                    f"max_tokens {MAX_TOKENS} with no cut. anchor: last (the post-processor's end-of-text "
                    f"token) must survive; origin reference, filled by the GPU wave."
                ),
            ],
            {"d1": text},
        )
    )

    # --- text-long-over: one document over the budget (client cuts) -----------
    text, count = make_text(SEED + 2, 9600, tokenizer, head, tail)
    cases.append(
        (
            CASE_DIR / "text-long-over.yaml",
            [
                f"id: {RECIPE}/text-long-over",
                f"recipe: {RECIPE}",
                "role: embed",
                "source:",
                "  kind: generated",
                "strata:",
                "  modality: text",
                "  length: long_over",
                "  batch: single",
                "inputs:",
                f"  instruction: {instruction}",
                "  queries:",
                "    - id: q1",
                f"      text: {QUERIES['long_over']}",
                "  documents:",
                *doc_block("d1", text),
                *expected_block(),
                *notes_block(
                    f"Generated long_over cell. Seeded generator (random.Random({SEED + 2}), same "
                    f"construction as text-long-under); full prompt = {count} tokens, {count - MAX_TOKENS} "
                    f"over max_tokens {MAX_TOKENS}. The client cuts the content span to fit (on_overflow: "
                    f"cut, frame reserved first); the anchor: last must survive the cut, and the wave "
                    f"records the cut and uncut vectors."
                ),
            ],
            {"d1": text},
        )
    )

    # --- text-mixed-length-batch: four document lengths in one batch ----------
    docs = []
    long_texts = {}
    targets = [8150, 1520, 170, 35]
    for i, target in enumerate(targets):
        text, count = make_text(SEED + 10 + i, target, tokenizer, head, tail)
        docs.append((f"d{i + 1}", text, count))
    query_lines = ["  queries:"]
    for i, q in enumerate(QUERIES["mixed_length"]):
        query_lines.append(f"    - id: q{i + 1}")
        query_lines.append(f"      text: {q}")
    counts = ", ".join(str(c) for _, _, c in docs)
    cases.append(
        (
            CASE_DIR / "text-mixed-length-batch.yaml",
            [
                f"id: {RECIPE}/text-mixed-length-batch",
                f"recipe: {RECIPE}",
                "role: embed",
                "source:",
                "  kind: generated",
                "strata:",
                "  modality: text",
                "  length: mixed",
                "  batch: mixed_length",
                "inputs:",
                f"  instruction: {instruction}",
                *query_lines,
                "  documents:",
                *(line for doc_id, text, _ in docs for line in doc_block(doc_id, text)),
                *expected_block(),
                *notes_block(
                    f"Generated mixed_length batch: four documents spanning short to budget-filling "
                    f"(full prompt tokens per document: {counts}; the 8150-token document sits just under "
                    f"max_tokens {MAX_TOKENS}, so no item exceeds the budget and no cut fires). Seeded "
                    f"generator random.Random({SEED + 10}+i), same construction as text-long-under; "
                    f"the wave records per-item vectors for the whole batch."
                ),
            ],
            {doc_id: text for doc_id, text, _ in docs},
        )
    )

    # --- image-text-long-over: page image + long text over the budget ---------
    text, count = make_text(SEED + 20, 7050, tokenizer, head, tail)
    cases.append(
        (
            CASE_DIR / "image-text-long-over.yaml",
            [
                f"id: {RECIPE}/image-text-long-over",
                f"recipe: {RECIPE}",
                "role: embed",
                "source:",
                "  kind: generated",
                "strata:",
                "  modality: mixed",
                "  length: long_over",
                "  batch: single",
                "inputs:",
                f"  instruction: {instruction}",
                "  queries:",
                "    - id: q1",
                f"      text: {QUERIES['image_text_over']}",
                "  documents:",
                *doc_block("d1", text, image="media/page.png"),
                *expected_block(),
                *notes_block(
                    f"Generated long_over cell in the mixed modality: a 1700x2200 page image plus a long "
                    f"text. Text content = {count - 2 * fixed} content tokens (prompt text span {count} "
                    f"with the frame); the image adds ~1776 image-pad tokens at the recipe's pixel budget "
                    f"(research-measured for 1700x2200 with images_kwargs max_pixels 1843200; 1240 at the "
                    f"checkpoint default - both totals exceed {MAX_TOKENS}). The text span must cut "
                    f"around the intact image pads and the anchor: last must survive. Media: synthetic "
                    f"(media/make_media.py), no third-party material. Wave: send the image through the "
                    f"chat-messages shape (image_url part); request_shape text covers text-only frames."
                ),
            ],
            {"d1": text},
        )
    )

    for path, body, long_texts in cases:
        emit(path, body, long_texts)


if __name__ == "__main__":
    main()
