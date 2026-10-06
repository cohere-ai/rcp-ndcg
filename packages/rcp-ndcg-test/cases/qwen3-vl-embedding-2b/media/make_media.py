"""Generate the synthetic media used by the qwen3-vl-embedding-2b cases.

All three files are drawn from scratch (no third-party material, no downloads),
so nothing here carries a licence constraint. Every generator is deterministic:
the images are pure functions of their pixel coordinates, the video of its frame
index. Re-running this script regenerates byte-identical PNGs (PIL writes a
deterministic stream) and a video whose frames are identical; only the MP4
container timestamps differ between encoders, which does not affect decode.

Files:
- stand_in.png  934x612 synthetic image standing in for the model card's demo
                photograph (that asset lives on a third-party bucket with no
                stated licence, so it is not redistributed here). 934x612 is the
                image size the research lane's token checks used; at the
                recipe's pixel budget it renders 551 image-pad tokens.
- page.png      1700x2200 synthetic page image (white background, dark
                text-like rules); at the recipe's pixel budget it renders
                1776 image-pad tokens (research check_tokens, page-image case).
- clip.mp4      448x256, 2 fps, 2 s (4 frames) synthetic clip. The recipe's
                served video sampling (checkpoint fps 2 / max_frames 768)
                samples all 4 frames; the reference samples fps 1 (2 frames).
                The reference/server video divergence is declared unmeasured in
                the recipe; this clip is small enough that both paths stay well
                inside every budget.

The images need only Pillow. The video needs imageio plus an ffmpeg binary
(imageio-ffmpeg bundles one); build it in a scratch venv, never the repo one:

    python -m venv <scratch>/venv && <scratch>/venv/bin/pip install imageio imageio-ffmpeg
    <scratch>/venv/bin/python media/make_media.py --only video

The committed media files are the test-time artefacts; this script is
provenance only (no test or case loads it at runtime).
"""

from __future__ import annotations

import argparse
from pathlib import Path

MEDIA_DIR = Path(__file__).resolve().parent


def stand_in_pixels(width: int = 934, height: int = 612):
    """Deterministic image content: a sky-to-ground gradient, a sun disc and a
    horizon band, drawn from pixel arithmetic only."""
    import numpy as np

    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    sky = (yy / height) * 120.0
    horizon = 0.62 * height
    r = 60.0 + sky * 1.1
    g = 90.0 + sky * 1.4
    b = 200.0 - sky * 0.8
    ground = yy > horizon
    r[ground] = 150.0 + (yy[ground] - horizon) / (height - horizon) * 70.0
    g[ground] = 120.0 + (yy[ground] - horizon) / (height - horizon) * 40.0
    b[ground] = 60.0
    # sun disc at (0.28w, 0.2h), radius 0.07*min(w,h)
    cx, cy, rad = 0.28 * width, 0.30 * height, 0.09 * min(width, height)
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 <= rad**2
    r[disc] = 250.0
    g[disc] = 235.0
    b[disc] = 160.0
    # horizon band
    band = (yy > horizon - 6) & (yy <= horizon)
    r[band] = 40.0
    g[band] = 40.0
    b[band] = 40.0
    return np.stack([r, g, b], axis=-1).astype("uint8")


def page_pixels(width: int = 1700, height: int = 2200):
    """A synthetic A4-shaped page: white ground, dark line blocks and a margin
    rule, all from coordinate arithmetic."""
    import numpy as np

    page = np.full((height, width, 3), 250, dtype="uint8")
    yy, xx = np.mgrid[0:height, 0:width]
    left, right = int(0.10 * width), int(0.90 * width)
    top = int(0.08 * height)
    for block in range(14):
        y0 = top + block * int(0.062 * height)
        y1 = y0 + int(0.030 * height)
        for line in range(4):
            ly = y1 - line * (y1 - y0) // 4 - 6
            in_line = (yy == ly) & (xx >= left) & (xx <= right)
            page[in_line] = 35
    head = (yy > top - 40) & (yy < top - 8) & (xx >= left) & (xx <= right)
    page[head] = 20
    return page


def write_png(name: str, array) -> None:
    from PIL import Image

    Image.fromarray(array, "RGB").save(MEDIA_DIR / name, format="PNG")
    print(f"wrote {name}: {array.shape[1]}x{array.shape[0]}")


def make_video(name: str = "clip.mp4", width: int = 448, height: int = 256, fps: int = 2, seconds: int = 2) -> None:
    """Four deterministic frames: a vertical gradient whose phase moves with the
    frame index and a per-frame counter block. Encoded with imageio/ffmpeg."""
    import imageio.v3 as iio
    import numpy as np

    frames = []
    for t in range(fps * seconds):
        yy, xx = np.mgrid[0:height, 0:width].astype(float)
        phase = t / (fps * seconds)
        r = (xx / width) * 255.0
        g = (yy / height) * 255.0 * (0.4 + 0.6 * phase)
        b = ((xx + yy) / (width + height)) * 255.0 * (0.5 + 0.5 * phase)
        frame = np.stack([r, g, b], axis=-1).astype("uint8")
        counter = (slice(t * 16, t * 16 + 16), slice(0, 16 + 8 * t))
        frame[counter] = 255
        frames.append(frame)
    iio.imwrite(MEDIA_DIR / name, frames, fps=fps, codec="libx264", macro_block_size=16)
    print(f"wrote {name}: {len(frames)} frames @ {fps} fps, {width}x{height}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["images", "video"], default=None)
    args = ap.parse_args()
    if args.only in (None, "images"):
        write_png("stand_in.png", stand_in_pixels())
        write_png("page.png", page_pixels())
    if args.only in (None, "video"):
        make_video()


if __name__ == "__main__":
    main()
