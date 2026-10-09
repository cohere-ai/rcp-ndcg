"""The fixture vision embedder's reference, run as a subprocess (never imported by the harness).

Independent of the product: it renders the document prompt from its own constants and sizes images with its
own processor rule -- the fixture "card": Qwen2-VL's ``smart_resize`` (patch 14 x merge 2 = 28 px per token
edge) under the card's pixel budget 784..200704 px, each image costing its merged patches plus its two vision
markers, the media before the text (the order the card builds its inputs in).

Modes: ``render`` and ``embed`` as the fixture embedder's (text rows; deterministic vectors), and ``media``:
per pairs row and side that carries media, ``{"index", "side", "placement", "media": [{"kind", "width",
"height", "tokens"}]}`` -- what the card's model consumes; the card takes one image per document, and a side
with more is ``{"index", "side", "refused"}``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
from pathlib import Path

PREFIX = "doc: "
SUFFIX = " [END]"
EMBED_TAG = "embed"
FACTOR = 28
MIN_PIXELS = 784
MAX_PIXELS = 200704


def card_resize(height: int, width: int) -> tuple[int, int]:
    """The card processor's resize: both edges to the factor, the area into the pixel budget."""
    h_bar, w_bar = round(height / FACTOR) * FACTOR, round(width / FACTOR) * FACTOR
    if h_bar * w_bar > MAX_PIXELS:
        beta = math.sqrt((height * width) / MAX_PIXELS)
        h_bar = max(FACTOR, math.floor(height / beta / FACTOR) * FACTOR)
        w_bar = max(FACTOR, math.floor(width / beta / FACTOR) * FACTOR)
    elif h_bar * w_bar < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        h_bar = math.ceil(height * beta / FACTOR) * FACTOR
        w_bar = math.ceil(width * beta / FACTOR) * FACTOR
    return h_bar, w_bar


def media_facts(text: str, entries: list[dict]) -> dict:
    """One side as the card consumes it: the media in order, then the text."""
    from PIL import Image

    items = []
    for entry in entries:
        payload = base64.b64decode(str(entry["uri"]).split(",", 1)[1])
        with Image.open(io.BytesIO(payload)) as handle:
            width, height = handle.size
        resized_h, resized_w = card_resize(height, width)
        tokens = (resized_h // FACTOR) * (resized_w // FACTOR) + 2
        items.append({"kind": "image", "width": resized_w, "height": resized_h, "tokens": tokens})
    return {"placement": ["image"] * len(items) + (["text"] if text else []), "media": items}


def main() -> int:
    parser = argparse.ArgumentParser(description="the fixture vision embedder reference")
    parser.add_argument("--mode", required=True, choices=["render", "embed", "media"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--recipe", required=True, help="the resolved recipe JSON the harness passed")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from deterministic import vector

    pairs = [json.loads(line) for line in Path(args.pairs).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for index, row in enumerate(pairs):
        if args.mode == "render":
            rows.append({"index": index, "shape": "document", "text": PREFIX + row["documents"][0] + SUFFIX})
        elif args.mode == "embed":
            rows.append(
                {
                    "index": index,
                    "document_vectors": [
                        [float(x) for x in vector(PREFIX + document + SUFFIX, EMBED_TAG)]
                        for document in row["documents"]
                    ],
                }
            )
        else:
            media = row.get("media") or {}
            if media.get("query"):
                rows.append({"index": index, "side": "query", **media_facts(row["query"], media["query"])})
            for position, entries in enumerate(media.get("documents") or []):
                if len(entries or []) > 1:
                    rows.append({"index": index, "side": f"document {position}", "refused": "one image per document"})
                elif entries:
                    facts = media_facts(row["documents"][position], entries)
                    rows.append({"index": index, "side": f"document {position}", **facts})
    Path(args.out).write_text(json.dumps({"rows": rows}) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
