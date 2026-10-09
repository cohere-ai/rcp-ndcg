"""The synthetic media request set (OBSERVATIONS-SPEC section 1, media): the images, clips and interleaved
rows every media recipe is asked.

One image per size bucket of the spec -- tiny, icon, an A4 page at 72, 150 and 300 dpi, a 16:9 slide, a very
tall receipt and an extreme aspect ratio -- plus one captioned page, each a pairs row whose document carries
the image inline (a ``data:`` URI of a PNG drawn here, deterministic in :data:`MEDIA_SET_VERSION` and the bucket),
so the media stage (:mod:`rcp_ndcg_test.equivalence.media`) compares them offline and the recorder sends them.
Beside the single-image rows: a query carrying an image (where the recipe's ``media_sides`` allows query media
and the recipe declares a query shape), a batch mixing a text-only and an image document, and -- for a recipe
whose ``max_images`` admits them -- a document with two images interleaved with text (text-image-text-image,
in order) and a document with ``max_images`` images; over the capacity stays the protocol edge
(:func:`media_edges`).  A video recipe plans a clip per :data:`VIDEO_CLIPS` size, alone and with text: a tiny
MJPEG AVI (RIFF) written here from frames drawn with PIL, its frame count the recipe's declared video policy's
``num_frames`` when it pins one, else :data:`VIDEO_FRAMES` (an fps policy's realised count follows from the
clip's rate and the declared fps), its header stating the size and rate so the product's
:func:`rcp_ndcg.data.media.probe_video_header` reads it and the engine's video loader decodes it.  The
protocol edges -- more images than the recipe's ``max_images`` in one request, an undecodable image -- are
bare requests of the corpus plan (:func:`media_edges`).  Versioned apart from the generator's sampling
(``GENERATOR_VERSION``), so adding or changing the media set never re-samples a recipe's text rows.

Public surface: :data:`MEDIA_SET_VERSION`, :data:`MEDIA_BUCKETS`, :data:`VIDEO_CLIPS`, :data:`VIDEO_FPS`,
:data:`VIDEO_FRAMES`, :func:`image_entry`, :func:`video_entry`, :func:`video_plan`, :func:`planned_media_rows`,
:func:`media_edges`.
"""

from __future__ import annotations

import base64
import hashlib
import io
import struct
from typing import Any

__all__ = [
    "MEDIA_BUCKETS",
    "MEDIA_SET_VERSION",
    "VIDEO_CLIPS",
    "VIDEO_FPS",
    "VIDEO_FRAMES",
    "image_entry",
    "media_edges",
    "planned_media_rows",
    "video_entry",
    "video_plan",
]

MEDIA_SET_VERSION = 3
"""The media set's version: any change to its images, clips or rows bumps it."""

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

VIDEO_CLIPS: tuple[tuple[str, int, int], ...] = (
    ("icon", 64, 64),
    ("page", 224, 224),
)
"""The clip sizes, ``(name, width, height)``: an icon-sized clip and a page-sized clip, each an MJPEG AVI
(RIFF) written here on CPU -- no container encoder, no codec dependency beyond PIL's JPEG."""

VIDEO_FPS = 8.0
"""The generated clips' frame rate: the number the AVI header states (``avih`` microseconds per frame,
``strh`` rate over scale) and every entry records."""

VIDEO_FRAMES = 64
"""The generated clips' total frames when the recipe declares the engine's fps rule instead of a pinned
``num_frames``: a policy that pins a count writes exactly that many frames, and an fps policy writes this
many (the engine's realised count follows from the clip's rate and the declared fps; a 64-frame/8 fps clip
at fps 2 realises 16)."""

_VIDEO_MIME = "video/x-msvideo"

_QUERY = "which page shows the quarterly chart"
_CAPTION = "Figure 2: the quarterly chart, with the totals per region"
_TEXT_BEFORE = "the chart opens the page, "
_TEXT_AFTER = " and the table closes it"


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


def _frame(seed: bytes, width: int, height: int, index: int, total: int) -> Any:
    """One deterministic frame: three distinct scenes over the clip, a bar moving with the frame's share."""
    from PIL import Image, ImageDraw

    scene = index * 3 // max(1, total)
    digest = hashlib.sha256(seed + bytes([scene])).digest()
    image = Image.new("RGB", (width, height), (digest[0], digest[1], digest[2]))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, max(1, width // 2), max(1, height // 3)), fill=(digest[3], digest[4], digest[5]))
    draw.rectangle((width // 3, height // 2, width - 1, height - 1), fill=(digest[6], digest[7], digest[8]))
    bar = max(1, width // 8)
    x = min((index + 1) * width // (total + 1), width - bar)
    draw.rectangle((x, 0, x + bar - 1, height - 1), fill=(digest[9], digest[10], digest[11]))
    return image


def mjpeg_avi(name: str, width: int, height: int, num_frames: int, *, fps: float = VIDEO_FPS) -> bytes:
    """A tiny MJPEG AVI (RIFF) of ``num_frames`` frames at ``width`` x ``height``, written on CPU.

    Each frame is drawn with PIL (:func:`_frame`, deterministic in :data:`MEDIA_SET_VERSION`, ``name`` and
    the frame's position) and encoded as a baseline JPEG; the container is the classic RIFF AVI the engines'
    video loaders read -- ``avih`` and a ``vids``/``MJPG`` ``strh`` in ``hdrl``, the JPEG frames as ``00dc``
    chunks in ``movi``, an ``idx1`` index, every chunk even-padded, and a full 40-byte BITMAPINFOHEADER
    (a truncated one makes OpenCV's AVI demuxer drop the first frame).  Units: bytes; ``fps`` in frames per
    second (the header states it as ``avih`` microseconds per frame and ``strh`` ``dwRate`` over ``dwScale``,
    with scale 1 so the rate is ``fps``).  The root distribution's tests carry a second writer by intent
    (``tests/conftest.py::write_mjpeg_avi``: cross-distribution); keep both headers BITMAPINFOHEADER-exact.
    """
    usec = round(1_000_000 / fps)
    jpegs = []
    for index in range(num_frames):
        buffer = io.BytesIO()
        _frame(f"{MEDIA_SET_VERSION}/{name}".encode(), width, height, index, num_frames).save(
            buffer, format="JPEG", quality=80
        )
        data = buffer.getvalue()
        if len(data) % 2:
            data += b"\x00"  # RIFF chunks are even-padded; JPEG decoders stop at the EOI marker
        jpegs.append(data)
    max_chunk = max(len(data) for data in jpegs) + 8

    def chunk(fourcc: bytes, payload: bytes) -> bytes:
        return fourcc + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) % 2 else b"")

    def listing(fourcc: bytes, body: bytes) -> bytes:
        return chunk(b"LIST", fourcc + body)

    # BITMAPINFOHEADER: the frame size and the MJPG compression tag the demuxer reads before decoding. All
    # 40 bytes its biSize declares -- a truncated one (biClrUsed/biClrImportant missing) makes OpenCV's AVI
    # demuxer drop the first frame of the clip while every JPEG still decodes standalone.
    strf = struct.pack(
        "<IiiHH4sIiiII",
        40,
        width,
        height,
        1,
        24,
        b"MJPG",
        width * height * 3,
        0,
        0,
        0,
        0,
    )
    strh = (
        b"vids"
        + b"MJPG"
        + struct.pack(
            "<IIHHIIIIIII",
            0,  # dwFlags
            0,
            0,  # wPriority, wLanguage
            0,  # dwInitialFrames
            1,  # dwScale
            int(fps),  # dwRate: frames per second over the scale
            0,  # dwStart
            num_frames,  # dwLength: frames in the stream
            max_chunk,  # dwSuggestedBufferSize
            0,  # dwQuality
            0,  # dwSampleSize: variable (compressed frames)
        )
        + struct.pack("<4H", 0, 0, width, height)  # rcFrame
    )
    avih = struct.pack(
        "<IIIIIIIIII4x4x4x4x",
        usec,  # dwMicroSecPerFrame
        max(1, max_chunk * num_frames),  # dwMaxBytesPerSec
        0,  # dwPaddingGranularity
        0x910,  # dwFlags: HASINDEX | ISINTERLEAVED | TRUSTCKTYPE
        num_frames,  # dwTotalFrames
        0,  # dwInitialFrames
        1,  # dwStreams
        max_chunk,  # dwSuggestedBufferSize
        width,  # dwWidth
        height,  # dwHeight
    )
    hdrl = listing(b"hdrl", chunk(b"avih", avih) + listing(b"strl", chunk(b"strh", strh) + chunk(b"strf", strf)))
    movi_body = b"".join(b"00dc" + struct.pack("<I", len(data)) + data for data in jpegs)
    offsets = []
    position = 0
    for data in jpegs:
        offsets.append(position)
        position += 8 + len(data)
    index_chunk = chunk(
        b"idx1",
        b"".join(
            b"00dc" + struct.pack("<III", 0x10, offset, len(data))  # AVIIF_KEYFRAME
            for data, offset in zip(jpegs, offsets, strict=True)
        ),
    )
    body = b"AVI " + hdrl + listing(b"movi", movi_body) + index_chunk
    return b"RIFF" + struct.pack("<I", len(body)) + body


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


def video_entry(name: str, width: int, height: int, num_frames: int, *, fps: float = VIDEO_FPS) -> dict[str, Any]:
    """One pairs media entry: an MJPEG AVI clip of the size ``name``, inline, with its container facts.

    ``num_frames`` is the clip's real frame count -- the recipe's declared video policy's ``num_frames``
    when it pins one, else :data:`VIDEO_FRAMES` (:func:`video_plan`), so the client's policy accepts the
    container and the engine's declared rule samples it; ``width``/``height``/``fps``/``duration_s`` are what
    :func:`rcp_ndcg.data.media.probe_video_header` reads back off the header.
    """
    payload = mjpeg_avi(name, width, height, num_frames, fps=fps)
    return {
        "kind": "video",
        "uri": f"data:{_VIDEO_MIME};base64," + base64.b64encode(payload).decode("ascii"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mime": _VIDEO_MIME,
        "width": width,
        "height": height,
        "num_bytes": len(payload),
        "num_frames": num_frames,
        "duration_s": num_frames / fps if fps else None,
        "fps": fps,
    }


def video_plan(recipe: Any) -> tuple[list[dict[str, Any]], str]:
    """The clips the media set writes for one recipe, and why it writes none.

    Input: a loaded recipe.  Output: ``(clips, reason)`` -- one record per :data:`VIDEO_CLIPS` size at the
    recipe's declared video policy's ``num_frames`` (``{"name", "width", "height", "num_frames"}``), or the
    empty list with the reason: the recipe declares no video input, or it declares none of the capacity and
    policy the wire needs (``max_videos``, a ``video_policy``; without a policy the client refuses a
    container -- the engine's own default sampling would decide what it is shown).
    """
    if "video" not in getattr(recipe, "input", ()):
        return [], "the recipe declares no video input"
    max_videos = int(recipe.client.get("max_videos") or 0)
    policy = recipe.client.get("video_policy")
    if not max_videos:
        return [], "the recipe declares no video capacity (client.max_videos: 0)"
    if policy is None:
        return [], "the recipe declares no video_policy: the client refuses a container (the engine's own "
        "default sampling would decide what it is shown)"
    return [
        {"name": name, "width": width, "height": height, "num_frames": int(policy.get("num_frames") or VIDEO_FRAMES)}
        for name, width, height in VIDEO_CLIPS
    ], ""


def _media_sides(recipe: Any) -> tuple[str, ...]:
    """The sides the recipe's client allows media on: its ``media_sides`` field (the product's default: both)."""
    sides = recipe.client.get("media_sides")
    return tuple(sides) if sides is not None else ("query", "document")


def _declares_query_shape(recipe: Any) -> bool:
    """Whether the client can encode a media query: a rerank pair carries both sides, an embed recipe needs
    a declared ``query`` shape (a one-shape template cannot fit the query's render)."""
    if recipe.role == "rerank":
        return True
    from ..equivalence import fitting

    return "query" in fitting.declared_shapes(recipe)


def _several_images_row(limit: int) -> dict[str, Any]:
    """One document carrying ``limit`` images (several images up to the recipe's per-request capacity)."""
    entries = [image_entry(name, width, height) for name, width, height in MEDIA_BUCKETS[:limit]]
    return {
        "query": _QUERY,
        "documents": [""],
        "media": {"query": [], "documents": [entries]},
        "strata": ["media:image", "media:several_images"],
        "source": {"suite": "synthetic", "media_set": MEDIA_SET_VERSION, "bucket": "several_images"},
    }


def planned_media_rows(recipe: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """The media rows the generator plans for one recipe (from the recipe, not from a pairs file), and their
    strata.

    Input: a loaded recipe.  Output: ``(rows, strata)`` -- for a recipe with image input, one row per
    :data:`MEDIA_BUCKETS` entry (an image-only document) and one captioned page; a batch mixing a text-only
    and an image document; a query carrying an image where the recipe's ``media_sides`` allows query media
    and the client can encode a media query; and, for a recipe whose ``max_images`` admits them, a document
    with two images interleaved with text (text-image-text-image, in order) and a document with
    ``max_images`` images.  For a recipe with video input and a declared policy (``max_videos`` and a
    ``video_policy``), a clip per :data:`VIDEO_CLIPS` size, alone and with text.  Each row is
    ``{"query", "documents", "media", "strata", "source"}`` with the media on the document side unless the
    row's stratum says otherwise; the media entries are ``image``/``video`` entries (a ``data:`` URI of bytes
    drawn here) and ``text`` entries (a part sequence's text segments, for the interleaved row).  Every
    stratum is recorded present or absent with the reason.  A text-only recipe gets no rows and the strata
    say why.
    """
    if "image" not in recipe.input:
        reason = "the recipe is text-only (recipe.input declares no image)"
        strata = {f"media:image:{name}": {"present": False, "reason": reason} for name, _, _ in MEDIA_BUCKETS}
        return [], {
            **strata,
            "media:image+text": {"present": False, "reason": reason},
            "media:query_image": {"present": False, "reason": reason},
            "media:mixed_batch": {"present": False, "reason": reason},
            "media:interleaved": {"present": False, "reason": reason},
            "media:several_images": {"present": False, "reason": reason},
            "media:video": {"present": False, "reason": reason},
            "media:video:icon": {"present": False, "reason": reason},
            "media:video+text": {"present": False, "reason": reason},
        }
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
    # One batch mixing a text-only and an image document: the client sends both sides of one row.
    rows.append(
        {
            "query": _QUERY,
            "documents": [_CAPTION, ""],
            "media": {"query": [], "documents": [[], [image_entry("icon", 64, 64)]]},
            "strata": ["media:image", "media:mixed_batch"],
            "source": {"suite": "synthetic", "media_set": MEDIA_SET_VERSION, "bucket": "mixed_batch"},
        }
    )
    strata["media:mixed_batch"] = {"present": True}
    # A query carrying an image: where the recipe's media_sides allow query media and the client can
    # encode a media query (a declared query shape; a rerank pair carries both sides).
    if "query" in _media_sides(recipe):
        if _declares_query_shape(recipe):
            rows.append(
                {
                    "query": _QUERY,
                    "documents": [_CAPTION],
                    "media": {"query": [image_entry("icon", 64, 64)], "documents": [[]]},
                    "strata": ["media:image", "media:query_image"],
                    "source": {"suite": "synthetic", "media_set": MEDIA_SET_VERSION, "bucket": "query_image"},
                }
            )
            strata["media:query_image"] = {"present": True, "side": "query"}
        else:
            strata["media:query_image"] = {
                "present": False,
                "reason": "the recipe declares no query shape (the client cannot encode a media query)",
            }
    else:
        sides = _media_sides(recipe)
        strata["media:query_image"] = {
            "present": False,
            "reason": f"media_sides allows media on the {' and '.join(sides) if sides else 'no'} side(s) only",
        }
    # Two images and up to max_images in one document: only where the recipe's per-request capacity admits
    # them.  Below the limit the client refuses a second image before the engine saw one, and the over-limit
    # request is the edge:too_many_images bare probe.
    limit = int(recipe.client.get("max_images") or 0)
    if limit >= 2:
        rows.append(
            {
                "query": _QUERY,
                "documents": [""],
                "media": {
                    "query": [],
                    "documents": [
                        [
                            {"kind": "text", "text": _TEXT_BEFORE},
                            image_entry("icon", 64, 64),
                            {"kind": "text", "text": _TEXT_AFTER},
                            image_entry("a4_72dpi", 595, 842),
                        ]
                    ],
                },
                "strata": ["media:image", "media:interleaved"],
                "source": {"suite": "synthetic", "media_set": MEDIA_SET_VERSION, "bucket": "interleaved"},
            }
        )
        strata["media:interleaved"] = {"present": True, "images": 2}
        rows.append(_several_images_row(limit))
        strata["media:several_images"] = {"present": True, "images": limit}
    else:
        reason = (
            f"the recipe's max_images is {limit}: the client refuses a second image in one request (the "
            "engine's per-prompt limit refuses it too; edge:too_many_images sends it bare)"
        )
        strata["media:interleaved"] = {"present": False, "reason": reason}
        strata["media:several_images"] = {"present": False, "reason": reason}
    # A video recipe's clips, alone and with text, at the declared sampling.
    clips, why = video_plan(recipe)
    if clips:
        for clip in clips:
            entry = video_entry(clip["name"], clip["width"], clip["height"], clip["num_frames"])
            with_text = clip["name"] != VIDEO_CLIPS[0][0]
            rows.append(
                {
                    "query": _QUERY,
                    "documents": [_CAPTION] if with_text else [""],
                    "media": {"query": [], "documents": [[entry]]},
                    "strata": ["media:video", f"media:video:{clip['name']}"]
                    + (["media:video+text"] if with_text else []),
                    "source": {"suite": "synthetic", "media_set": MEDIA_SET_VERSION, "bucket": clip["name"]},
                }
            )
            strata[f"media:video:{clip['name']}"] = {
                "present": True,
                "width": clip["width"],
                "height": clip["height"],
                "num_frames": clip["num_frames"],
            }
        strata["media:video"] = {
            "present": True,
            "clips": len(clips),
            "num_frames": clips[0]["num_frames"],
            "fps": VIDEO_FPS,
        }
        if any("media:video+text" in row["strata"] for row in rows):
            strata["media:video+text"] = {"present": True}
        else:
            strata["media:video+text"] = {
                "present": False,
                "reason": "the media set plans no with-text clip (one clip size)",
            }
    else:
        strata["media:video"] = {"present": False, "reason": why}
        strata["media:video:icon"] = {"present": False, "reason": why}
        strata["media:video+text"] = {"present": False, "reason": why}
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
    refuses it); ``edge:corrupt_image`` -- one item whose PNG bytes do not decode; for a recipe that takes
    video, ``edge:too_many_videos`` -- one item with ``max_videos + 1`` clips -- and ``edge:corrupt_video``
    -- one item whose container bytes do not decode.  ``media:request_set`` is present for a media recipe
    (its pairs rows carry the size buckets); ``media:video`` is present for a recipe whose pairs carry the
    clips (:func:`video_plan`), absent with the reason otherwise.
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
    clips, why = video_plan(recipe)
    video_limit = int(recipe.client.get("max_videos") or 0)
    video_strata: dict[str, dict[str, Any]]
    if "video" in getattr(recipe, "input", ()) and video_limit:
        clip = video_entry(*VIDEO_CLIPS[0], VIDEO_FRAMES)
        video_part = {"type": "video_url", "video_url": {"url": clip["uri"]}}
        corrupt_video = "data:video/mp4;base64," + base64.b64encode(b"not a container").decode("ascii")
        rows.extend(
            [
                {
                    "request_id": "edge:too_many_videos",
                    "stratum": "edge:too_many_videos",
                    "probe": "edge:too_many_videos",
                    "layer": "protocol",
                    "method": "POST",
                    "body": _media_body(recipe, [video_part] * (video_limit + 1)),
                },
                {
                    "request_id": "edge:corrupt_video",
                    "stratum": "edge:corrupt_video",
                    "probe": "edge:corrupt_video",
                    "layer": "protocol",
                    "method": "POST",
                    "body": _media_body(recipe, [{"type": "video_url", "video_url": {"url": corrupt_video}}]),
                },
            ]
        )
        video_strata = {
            "edge:too_many_videos": {"present": True, "videos": video_limit + 1},
            "edge:corrupt_video": {"present": True},
        }
    else:
        reason = (
            "the recipe declares no video input"
            if "video" not in getattr(recipe, "input", ())
            else f"the recipe's max_videos is {video_limit}: the client refuses a container outright"
        )
        video_strata = {
            "edge:too_many_videos": {"present": False, "reason": reason},
            "edge:corrupt_video": {"present": False, "reason": reason},
        }
    strata = {
        "media:request_set": {"present": True, "version": MEDIA_SET_VERSION, "buckets": len(MEDIA_BUCKETS)},
        "edge:too_many_images": {"present": True, "images": limit + 1},
        "edge:corrupt_image": {"present": True},
        **video_strata,
        "media:video": (
            {
                "present": True,
                "clips": len(clips),
                "num_frames": clips[0]["num_frames"],
                "fps": VIDEO_FPS,
                "note": "the pairs rows carry the clips; the engine decodes the MJPEG AVI containers",
            }
            if clips
            else {"present": False, "reason": why}
        ),
    }
    return rows, strata


_EDGE_STRATA = (
    "media:request_set",
    "edge:too_many_images",
    "edge:corrupt_image",
    "edge:too_many_videos",
    "edge:corrupt_video",
    "media:video",
)
