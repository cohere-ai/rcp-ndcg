"""Turning a :class:`CompletionInput` into OpenAI chat-completion messages.

The judge client (:class:`rcp_ndcg.llm.client.JudgeClient`) sends every
endpoint, self-hosted or hosted, the messages built here.

**Only prepared media is sent.** Every image and video frame reaches this module
already prepared by :func:`rcp_ndcg.data.prepare.prepare_content` -- resized as the
judge's processor would resize it, encoded, and inlined as a data URI -- and an
image that was not is refused. Inlining is the one form that works everywhere: a
hosted vendor cannot read a private bucket, and a self-hosted engine behind a
service address usually cannot either.

**A video goes out as what it is.** A container becomes one ``video_url`` block
(the OpenAI content part SGLang and vLLM both accept, as a data URI), sent
unchanged for the engine to decode and sample (``wire: video_url``); sampled
frames become one ``image_url`` block each, the form every OpenAI-compatible
endpoint accepts.
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from rcp_ndcg_core.content import Content, ImagePart, MediaRef, TextPart, VideoPart

from rcp_ndcg.errors import DataError
from rcp_ndcg.support.logging import get_logger

if TYPE_CHECKING:
    from rcp_ndcg.data.media import MediaResolver
    from rcp_ndcg.llm.client import CompletionInput

logger = get_logger(__name__)

#: Largest video container inlined into one request, in bytes. A data URI is
#: base64 (4/3 the size) and is re-sent with every window the clip appears in, so a
#: clip over this is refused by name rather than sent as a request the engine or a
#: proxy rejects -- or accepts after a long upload. Override with
#: ``RCP_NDCG_MAX_VIDEO_BYTES``; a corpus of long clips wants shorter clips.
MAX_VIDEO_BYTES = int(os.environ.get("RCP_NDCG_MAX_VIDEO_BYTES", str(64 * 1024 * 1024)))


class MediaCounts(NamedTuple):
    """What one prompt sends: ``image_url`` blocks and ``video_url`` blocks."""

    images: int
    videos: int


def media_counts(content: Content | None) -> MediaCounts:
    """How many image and video blocks *content* lowers to in :func:`build_messages`.

    The count the per-request gates compare against the server's limits; a video
    frame directory counts one image per frame, a container one video.
    """
    if content is None:
        return MediaCounts(0, 0)
    images = videos = 0
    for part in content.parts:
        if isinstance(part, ImagePart):
            images += 1
        elif isinstance(part, VideoPart):
            if part.frames:
                images += len(part.frames)
            else:
                videos += 1
    return MediaCounts(images, videos)


def build_messages(input: CompletionInput) -> list[dict[str, Any]]:
    """The ``messages`` array for one completion request.

    A text-only input sends a string ``content``; an input with media sends a list
    of text, ``image_url`` and ``video_url`` blocks.
    """
    content = input.user_content
    if content is None or not content.has_media:
        return [{"role": "user", "content": input.user_prompt}]
    return [{"role": "user", "content": _blocks(content)}]


def _blocks(content: Content) -> list[dict[str, Any]]:
    from rcp_ndcg.data.media import default_resolver

    resolver = default_resolver()

    blocks: list[dict[str, Any]] = []
    for part in content.parts:
        if isinstance(part, TextPart):
            if part.text:
                blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ImagePart):
            blocks.append(_image_block(part.ref))
        elif isinstance(part, VideoPart):
            if part.frames:
                blocks.extend(_image_block(ref) for ref in part.frames)
            else:
                assert part.ref is not None  # VideoPart validates container-or-frames
                blocks.append(_video_block(part.ref, resolver))
    return blocks


def _video_block(ref: MediaRef, resolver: MediaResolver) -> dict[str, Any]:
    """One ``video_url`` block carrying the whole container as a data URI."""
    from rcp_ndcg.data.media import VIDEO_MIME_BY_SUFFIX

    mime = ref.mime or VIDEO_MIME_BY_SUFFIX.get(Path(ref.uri).suffix.lower())
    if mime is None or not mime.startswith("video/"):
        raise DataError(
            f"{ref.uri}: cannot tell which video container this is (mime {ref.mime!r}); record `mime` at "
            f"ingest or use one of {sorted(VIDEO_MIME_BY_SUFFIX)}"
        )
    size = ref.num_bytes if ref.num_bytes is not None else resolver.local_path(ref).stat().st_size
    if size > MAX_VIDEO_BYTES:
        raise DataError(
            f"{ref.uri} is {size} bytes, over the {MAX_VIDEO_BYTES}-byte limit for an inlined video "
            "(`RCP_NDCG_MAX_VIDEO_BYTES`). Shorten or re-encode the clip at ingest, or raise the limit "
            "knowingly -- every window re-sends it."
        )
    path = resolver.local_path(ref)
    return {"type": "video_url", "video_url": {"url": _video_data_uri(ref.cache_key, str(path), mime)}}


def _image_block(ref: MediaRef) -> dict[str, Any]:
    """One ``image_url`` block carrying a prepared image."""
    from rcp_ndcg.data.prepare import is_prepared

    if not is_prepared(ref):
        raise DataError(
            f"{ref.uri}: an image reached the request unprepared. Every image and frame is sent as "
            "rcp_ndcg.data.prepare.prepare_content returns it (resized for the judge's processor, inlined)."
        )
    return {"type": "image_url", "image_url": {"url": ref.uri}}


#: Encoded video containers held per worker process. Far fewer than pages: a clip
#: is megabytes where a page is hundreds of KB, so the image cache's 512 entries
#: would be gigabytes per worker. A clip still recurs across the windows of one
#: query, which is what this small cache spans. Override with
#: ``RCP_NDCG_VIDEO_CACHE_SIZE``.
VIDEO_CACHE_SIZE = int(os.environ.get("RCP_NDCG_VIDEO_CACHE_SIZE", "16"))


@lru_cache(maxsize=VIDEO_CACHE_SIZE)
def _video_data_uri(cache_key: str, path: str, mime: str) -> str:
    """The data URI for one cached video container, keyed by :attr:`MediaRef.cache_key` (the content hash when
    there is one)."""
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


__all__ = [
    "MAX_VIDEO_BYTES",
    "MediaCounts",
    "build_messages",
    "media_counts",
]
