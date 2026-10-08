"""The synthetic media request set (OBSERVATIONS-SPEC section 1, media): the images every media recipe is asked.

One image per size bucket of the spec -- tiny, icon, an A4 page at 72, 150 and 300 dpi, a 16:9 slide, a very
tall receipt and an extreme aspect ratio -- plus one captioned page, each a pairs row whose document carries the
image inline (a ``data:`` URI of a PNG drawn here, deterministic in :data:`MEDIA_SET_VERSION` and the bucket),
so the media stage (:mod:`rcp_ndcg_test.equivalence.media`) compares them offline and the recorder sends them.
The protocol edges -- more images than the recipe's ``max_images`` in one request, an undecodable image -- are
bare requests of the corpus plan (:func:`media_edges`).  Versioned apart from the generator's sampling
(``GENERATOR_VERSION``), so adding or changing the media set never re-samples a recipe's text rows.

Public surface: :data:`MEDIA_SET_VERSION`, :data:`MEDIA_BUCKETS`, :func:`image_entry`, :func:`planned_media_rows`,
:func:`media_edges`.
"""

from __future__ import annotations

import base64
import hashlib
import io
from typing import Any

__all__ = ["MEDIA_BUCKETS", "MEDIA_SET_VERSION", "image_entry", "media_edges", "planned_media_rows"]

MEDIA_SET_VERSION = 1
"""The media set's version: any change to its images or rows bumps it."""

MEDIA_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("tiny", 16, 16),
    ("icon", 64, 64),
    ("a4_72dpi", 595, 842),
    ("a4_150dpi", 1240, 1754),
    ("a4_300dpi", 2480, 3508),
    ("slide_16_9", 1920, 1080),
    ("tall_receipt", 576, 4096),
    ("extreme_aspect", 3000, 20),
)
"""The image size buckets, ``(name, width, height)`` in pixels (the extreme aspect ratio stays under the
processors' 200:1 refusal)."""

_QUERY = "which page shows the quarterly chart"
_CAPTION = "Figure 2: the quarterly chart, with the totals per region"


def _png(name: str, width: int, height: int) -> bytes:
    """A deterministic PNG of ``width`` x ``height``: a background and two blocks, coloured from the name."""
    from PIL import Image, ImageDraw

    digest = hashlib.sha256(f"{MEDIA_SET_VERSION}/{name}".encode()).digest()
    image = Image.new("RGB", (width, height), (digest[0], digest[1], digest[2]))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, max(1, width // 2), max(1, height // 3)), fill=(digest[3], digest[4], digest[5]))
    draw.rectangle((width // 3, height // 2, width - 1, height - 1), fill=(digest[6], digest[7], digest[8]))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def image_entry(name: str, width: int, height: int, *, payload: bytes | None = None) -> dict[str, Any]:
    """One pairs media entry: an image of the bucket ``name`` inline, as a ``MediaRef`` object plus its kind."""
    data = payload if payload is not None else _png(name, width, height)
    return {
        "kind": "image",
        "uri": "data:image/png;base64," + base64.b64encode(data).decode("ascii"),
        "sha256": hashlib.sha256(data).hexdigest(),
        "mime": "image/png",
        "width": width,
        "height": height,
        "num_bytes": len(data),
    }


def planned_media_rows(recipe: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """The media rows the generator plans for one recipe (from the recipe, not from a pairs file), and their
    strata.

    Input: a loaded recipe.  Output: ``(rows, strata)`` -- for a recipe with image input, one row per
    :data:`MEDIA_BUCKETS` entry (an image-only document) and one captioned page, each ``{"query", "documents",
    "media", "strata", "source"}`` with the image on the document side (every media recipe takes document
    images); ``media:image:<bucket>`` and ``media:image+text`` present.  A text-only recipe gets no rows and
    the strata say why.
    """
    if "image" not in recipe.input:
        reason = "the recipe is text-only (recipe.input declares no image)"
        strata = {f"media:image:{name}": {"present": False, "reason": reason} for name, _, _ in MEDIA_BUCKETS}
        return [], {**strata, "media:image+text": {"present": False, "reason": reason}}
    rows: list[dict[str, Any]] = []
    strata: dict[str, dict[str, Any]] = {}
    for name, width, height in MEDIA_BUCKETS:
        rows.append(
            {
                "query": _QUERY,
                "documents": [""],
                "media": {"query": [], "documents": [[image_entry(name, width, height)]]},
                "strata": ["media:image", f"media:image:{name}"],
                "source": {"suite": "synthetic", "media_set": MEDIA_SET_VERSION, "bucket": name},
            }
        )
        strata[f"media:image:{name}"] = {"present": True, "width": width, "height": height}
    rows.append(
        {
            "query": _QUERY,
            "documents": [_CAPTION],
            "media": {"query": [], "documents": [[image_entry("a4_72dpi", 595, 842)]]},
            "strata": ["media:image", "media:image+text"],
            "source": {"suite": "synthetic", "media_set": MEDIA_SET_VERSION, "bucket": "a4_72dpi+caption"},
        }
    )
    strata["media:image+text"] = {"present": True}
    return rows, strata


def _media_body(recipe: Any, parts: list[dict[str, Any]]) -> dict[str, Any]:
    """A bare request carrying ``parts`` as one item on the recipe's media route."""
    if recipe.role == "rerank":
        return {"model": recipe.id, "query": _QUERY, "documents": [{"content": parts}]}
    body: dict[str, Any] = {"model": recipe.id, "messages": [{"role": "user", "content": parts}]}
    if recipe.role == "multi_vector":
        body["task"] = "token_embed"
    else:
        body["encoding_format"] = "float"
    return body


def media_edges(recipe: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """The media request set's protocol edges, sent bare: ``(bare rows, strata)``.

    ``edge:too_many_images`` -- one item with ``max_images + 1`` images (the engine's per-prompt limit
    refuses it); ``edge:corrupt_image`` -- one item whose PNG bytes do not decode.  ``media:request_set`` is
    present for a media recipe (its pairs rows carry the size buckets); ``media:video`` is absent with the
    reason (the generator writes no video container a video loader decodes).
    """
    if "image" not in recipe.input:
        reason = "the recipe is text-only"
        return [], {key: {"present": False, "reason": reason} for key in _EDGE_STRATA}
    image = {"type": "image_url", "image_url": {"url": image_entry("icon", 64, 64)["uri"]}}
    corrupt_uri = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nnot an image").decode("ascii")
    limit = int(recipe.client.get("max_images") or 0)
    rows = [
        {
            "request_id": "edge:too_many_images",
            "stratum": "edge:too_many_images",
            "probe": "edge:too_many_images",
            "layer": "protocol",
            "method": "POST",
            "body": _media_body(recipe, [image] * (limit + 1)),
        },
        {
            "request_id": "edge:corrupt_image",
            "stratum": "edge:corrupt_image",
            "probe": "edge:corrupt_image",
            "layer": "protocol",
            "method": "POST",
            "body": _media_body(recipe, [{"type": "image_url", "image_url": {"url": corrupt_uri}}]),
        },
    ]
    video = (
        "the generator writes no video container a video loader decodes (the media set is images drawn on CPU); "
        "a clip comes from a suite that carries one"
        if "video" in recipe.input
        else "the recipe declares no video input"
    )
    strata = {
        "media:request_set": {"present": True, "version": MEDIA_SET_VERSION, "buckets": len(MEDIA_BUCKETS)},
        "edge:too_many_images": {"present": True, "images": limit + 1},
        "edge:corrupt_image": {"present": True},
        "media:video": {"present": False, "reason": video},
    }
    return rows, strata


_EDGE_STRATA = ("media:request_set", "edge:too_many_images", "edge:corrupt_image", "media:video")
