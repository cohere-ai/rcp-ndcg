#!/usr/bin/env python3
"""Deterministic synthetic media for the ``topk-embed-v1-small`` cases.

Every image is generated from geometric shapes with ``random.Random(seed)`` -- no photographs,
no third-party material, nothing to licence (the files carry the repository's Apache-2.0 licence
like the rest of the case package; they are this script's own output, reproducible from the
recorded seeds). Re-running rewrites the PNGs byte-identically (same Pillow version; the
committed files are the provenance, the script is the audit trail). Never run by the tests.

The images exercise the recipe's pixel budget (serve.mm_processor_kwargs: min_pixels 65,536,
max_pixels 1,310,720 = the reference's image_token_budget 1280 patches):

- card_page.png      850x1100  the model-card case's substitute for the card's user-provided
                               "page.png" (a scanned-page look: title bar, text lines, a table)
- doc_small_1.png    256x256   = min_pixels exactly -> kept, 8 * 8 = 64 patch tokens expected
- doc_small_2.png    512x384   196,608 px, within budget -> 16 * 12 = 192 patch tokens expected
- doc_small_3.png    448x336   150,528 px -> the processor resizes to 32-px multiples
- doc_max_pixels.png 1280x1024 = max_pixels exactly -> N = 40 * 32 = 1280 patch tokens,
                               prompt 4 + 1280 + 3 = 1287 tokens (the R4 probe check)

Usage: python make_media.py [--out DIR]   (default: this file's directory)
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

GREYS = ((245, 245, 242), (232, 230, 226), (210, 208, 202), (188, 186, 180))
ACCENTS = ((36, 90, 154), (170, 62, 42), (52, 122, 74), (140, 96, 28))


def _page(draw, rng: random.Random, width: int, height: int, *, title: bool, table: bool) -> None:
    """A scanned-page look: a title bar, justified text lines, optionally a table grid."""
    margin = width // 14
    top = margin
    if title:
        bar_h = height // 22
        draw.rectangle((margin, top, width - margin, top + bar_h), fill=ACCENTS[rng.randrange(4)])
        top += bar_h + height // 30
    body_bottom = height - margin
    if table:
        body_bottom = top + int((height - 2 * margin) * 0.55)
        cols, rows = 5, 7
        cw, ch = (width - 2 * margin) / cols, (body_bottom - top) / rows
        for r in range(rows):
            for c in range(cols):
                x0, y0 = margin + c * cw, top + r * ch
                shade = GREYS[rng.randrange(4)] if (r + c) % 3 else ACCENTS[3]
                draw.rectangle((x0 + 2, y0 + 2, x0 + cw - 2, y0 + ch - 2), fill=shade)
        top = body_bottom + height // 40
    y = top
    line_h = max(6, height // 90)
    while y < body_bottom - line_h * 2:
        indent = margin if rng.random() > 0.12 else margin + width // 12
        extent = width - margin - rng.randrange(max(1, width // 6))
        draw.rectangle((indent, y, max(indent + line_h, extent), y + line_h), fill=GREYS[rng.randrange(3)])
        y += int(line_h * 1.9)


def _blocks(draw, rng: random.Random, width: int, height: int) -> None:
    """A slide-like pattern: a header band and a scatter of coloured blocks."""
    draw.rectangle((0, 0, width, height // 7), fill=ACCENTS[rng.randrange(4)])
    for _ in range(9):
        w, h = rng.randrange(width // 8, width // 2), rng.randrange(height // 12, height // 4)
        x, y = rng.randrange(0, max(1, width - w)), rng.randrange(height // 6, max(2, height - h))
        draw.rectangle(
            (x, y, x + w, y + h), fill=GREYS[rng.randrange(4)] if rng.random() < 0.6 else ACCENTS[rng.randrange(4)]
        )


def make(path: Path, width: int, height: int, seed: int, style: str) -> None:
    """Render one deterministic RGB image; the seed is recorded in the script (this call site)."""
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    img = Image.new("RGB", (width, height), GREYS[0])
    draw = ImageDraw.Draw(img)
    if style == "page":
        _page(draw, rng, width, height, title=True, table=True)
    elif style == "slide":
        _blocks(draw, rng, width, height)
    else:
        raise ValueError(style)
    img.save(path, format="PNG")


IMAGES = (  # (filename, width, height, seed, style)
    ("card_page.png", 850, 1100, 3101, "page"),
    ("doc_small_1.png", 256, 256, 3102, "slide"),
    ("doc_small_2.png", 512, 384, 3103, "slide"),
    ("doc_small_3.png", 448, 336, 3104, "slide"),
    ("doc_max_pixels.png", 1280, 1024, 3105, "page"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="write the case media (synthetic, deterministic)")
    parser.add_argument("--out", default=Path(__file__).resolve().parent, type=Path)
    args = parser.parse_args()
    for name, width, height, seed, style in IMAGES:
        make(args.out / name, width, height, seed, style)
        print(f"wrote {name} ({width}x{height}, seed {seed}, style {style})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
