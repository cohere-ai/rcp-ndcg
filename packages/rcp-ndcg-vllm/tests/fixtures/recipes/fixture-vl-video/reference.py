"""The fixture vision-and-video embedder's reference, run as a subprocess (never imported by the harness).

Independent of the product: it renders the prompt from its own constants and sizes images with its own
processor rule -- the fixture "card": Qwen2-VL's ``smart_resize`` (patch 14 x merge 2 = 28 px per token
edge) under the card's pixel budget 784..200704 px, each image costing its merged patches plus its two
vision markers. The card consumes a side's parts in the GIVEN order (an interleaved-native card): a text
part stands where it stands, an image after the text stays after it; a side's text segments join into the
one text the card reads, standing where the first text part stood. A video is the card's declared frame
count (the sampling the recipe pins on both sides); its tokens are the processor's and are not counted
here.

Modes: ``render`` and ``embed`` as the fixture embedder's (text rows; deterministic vectors), and
``media``: per pairs row and side that carries media, ``{"index", "side", "placement", "media":
[{"kind", "width", "height", "tokens" or None}]}`` -- what the card's model consumes; a side with more
than ``MAX_IMAGES`` images or one video over ``MAX_VIDEOS`` is ``{"index", "side", "refused"}``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
from pathlib import Path

QUERY_PREFIX = "q: "
PREFIX = "doc: "
SUFFIX = " [END]"
EMBED_TAG = "embed"
FACTOR = 28
MIN_PIXELS = 784
MAX_PIXELS = 200704
MAX_IMAGES = 2
MAX_VIDEOS = 1
VIDEO_FRAMES = 4
"""The card's declared video sampling: the recipe's video_policy.num_frames, restated here as the card's
own constant (the recipe's test pins the two to the same number)."""


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
    """One side as the card consumes it: the parts in the GIVEN order, a side's text segments joined into
    the one text the card reads, standing where the first text part stood (the card's own input model)."""
    from PIL import Image

    items: list[dict] = []
    placement: list[str] = []
    placed_text = False
    texts: list[str] = []
    for entry in entries:
        kind = str(entry.get("kind", "image"))
        if kind == "text":
            texts.append(str(entry.get("text", "")))
            if not placed_text:
                placement.append("text")
                placed_text = True
            continue
        if kind == "video":
            placement.append("video")
            items.append({"kind": "video", "frames": VIDEO_FRAMES, "tokens": None})
            continue
        payload = base64.b64decode(str(entry["uri"]).split(",", 1)[1])
        with Image.open(io.BytesIO(payload)) as handle:
            width, height = handle.size
        resized_h, resized_w = card_resize(height, width)
        tokens = (resized_h // FACTOR) * (resized_w // FACTOR) + 2
        placement.append("image")
        items.append({"kind": "image", "width": resized_w, "height": resized_h, "tokens": tokens})
    if texts and not placed_text:
        placement.append("text")
        placed_text = True
    if text and not placed_text:
        placement.append("text")
    return {"placement": placement, "media": items}


def side_refusals(entries: list[dict]) -> str | None:
    """Why the card refuses this side outright: more images or videos than its capacity declares."""
    images = sum(1 for entry in entries if str(entry.get("kind", "image")) == "image")
    videos = sum(1 for entry in entries if entry.get("kind") == "video")
    if images > MAX_IMAGES:
        return f"{images} images on one side; the card reads at most {MAX_IMAGES}"
    if videos > MAX_VIDEOS:
        return f"{videos} videos on one side; the card reads at most {MAX_VIDEOS}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="the fixture vision-and-video embedder reference")
    parser.add_argument("--mode", required=True, choices=["render", "embed", "media"])
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", required=True)
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
                refused = side_refusals(list(media["query"]))
                side = {"index": index, "side": "query", **(refused and {"refused": refused} or {})}
                rows.append({**side, **({} if refused else media_facts(row["query"], media["query"]))})
            for position, entries in enumerate(media.get("documents") or []):
                entries = list(entries or [])
                refused = side_refusals(entries)
                if refused:
                    rows.append({"index": index, "side": f"document {position}", "refused": refused})
                elif entries:
                    facts = media_facts(row["documents"][position], entries)
                    rows.append({"index": index, "side": f"document {position}", **facts})
    Path(args.out).write_text(json.dumps({"rows": rows}) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
