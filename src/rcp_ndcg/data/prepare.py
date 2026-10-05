"""Client-side media preparation: every image and video frame, sized as the judge's own processor would size it.

A judge endpoint only needs to be OpenAI-compatible: the engine runs stock, without media flags, because the client
prepares every image before it is sent, deterministically, and records what it sent. :func:`prepare_content` is the
one place this happens; the judge's chat wire (:mod:`rcp_ndcg.inference.adapters.chat`) sends only what it returns.

For each image and each sampled video frame, under an image policy with a pixel budget and a known processor family
(:meth:`~rcp_ndcg.data.resolution.ImagePolicy.resizes`):

1. **Decode and orient.** The stored bytes are decoded with Pillow and rotated by their EXIF orientation, as vLLM's
   loader and transformers' ``load_image`` do (vllm/multimodal/image.py:21-25 @ 3627a6a;
   transformers src/transformers/image_utils.py:510 @ 528c267).
2. **RGB.** An image with transparency is composited onto white, any other mode converted to RGB: vLLM's rule
   (vllm/multimodal/image.py:28-60 and multimodal/media/image.py:62-92 @ 3627a6a), and what transformers' processor
   does with ``do_convert_rgb`` for an opaque image. The engines differ on transparent images (SGLang drops the
   alpha channel); sending RGB removes the difference.
3. **Resize.** To :meth:`~rcp_ndcg.data.resolution.ImagePolicy.target_size` -- the processor's ``smart_resize`` under
   the budget -- with Pillow's BICUBIC filter, the processor's own ``resample``
   (models/qwen2_vl/image_processing_qwen2_vl.py:95 and image_processing_pil_qwen2_vl.py:90 @ 528c267). An image
   already at its target size is not resampled.
4. **Encode** losslessly as PNG, so the pixels the engine decodes are the pixels resized here, whatever image
   decoder the engine uses.

The engine then runs the same ``smart_resize`` on an image that is already at a fixed point of it: the budget is
checked to lie inside the engines' default budget (:data:`~rcp_ndcg.data.resolution.PROCESSORS`), so the resize
keeps the size, and both the torchvision and the Pillow resize return an equal-size image unchanged.

Without a budget, or when the judge declares no ``image_processor``, an image is sent **unchanged** (its stored bytes
and MIME type) and the engine's processor decides; the effective policy records that (``processor: null``).

Video: ``wire: frames`` samples the corpus's frames here (:func:`~rcp_ndcg.data.resolution.sample_video_part`) and
prepares each as an image; ``wire: video_url`` sends the container unchanged and the engine samples it.

:class:`MediaCensus` records every prepared image, frame and container once per judgement store, in
``preprocessing.jsonl`` beside the text cuts.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, NamedTuple

from rcp_ndcg_core.content import Content, ImagePart, MediaRef, VideoPart

from rcp_ndcg.data.media import decode_rgb, default_resolver
from rcp_ndcg.data.resolution import ImagePolicy, VideoPolicy, sample_video_part
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

#: The MIME type an unprepared image with no recorded ``mime`` is sent under.
DEFAULT_IMAGE_MIME = "image/png"

#: The encoding of every resized image: lossless, so the engine decodes exactly the resized pixels.
PREPARED_MIME = "image/png"

#: Prepared images held in memory (an LRU cache in the judging process). A page recurs in many windows of its
#: query, so the cache is sized to hold one query's pool: 512 pages, enough for a pool of 500, at roughly 1 to 1.5 MB
#: per encoded page (about 0.5 to 0.7 GB when full). Override with ``RCP_NDCG_IMAGE_CACHE_SIZE`` for a smaller
#: machine or a shallower pool.
IMAGE_CACHE_SIZE = int(os.environ.get("RCP_NDCG_IMAGE_CACHE_SIZE", "512"))

#: The census mechanism of a media record (beside the text cuts' ``doc_policy`` and ``window_budget``).
MEDIA_MECHANISM = "media"

MediaKind = Literal["image", "frame", "video"]


class PreparedMedia(NamedTuple):
    """One media item as stored and as sent.

    Attributes:
        kind: ``image`` (a page image), ``frame`` (a sampled video frame) or ``video`` (a container, engine-sampled).
        source: The reference as stored in the corpus.
        sent: What the request carries: a ``data:`` URI reference to the prepared bytes, with their hash, MIME
            type and size; for a container, the container reference unchanged.
        processor: The processor family whose resize was applied, or ``None`` when the item is sent unchanged.
        resized: Whether the size changed.
    """

    kind: MediaKind
    source: MediaRef
    sent: MediaRef
    processor: str | None
    resized: bool

    def as_row(self, *, corpus: str, doc_id: str) -> dict[str, Any]:
        """The census row: what was stored, what was sent, and how."""
        return {
            "mechanism": MEDIA_MECHANISM,
            "corpus": corpus,
            "doc_id": doc_id,
            "kind": self.kind,
            "uri": self.source.uri,
            "sha256": self.source.sha256,
            "source_width": self.source.width,
            "source_height": self.source.height,
            "sent_width": self.sent.width,
            "sent_height": self.sent.height,
            "sent_mime": self.sent.mime,
            "sent_sha256": self.sent.sha256,
            "processor": self.processor,
            "resized": self.resized,
            "sampling": "engine" if self.kind == "video" else "client",
        }


class PreparedContent(NamedTuple):
    """Content as the judge is sent it, and every media item in it."""

    content: Content
    media: list[PreparedMedia]


def is_prepared(ref: MediaRef) -> bool:
    """Whether ``ref`` is an image :func:`prepare_image` produced (its bytes inlined as a ``data:`` URI)."""
    return ref.uri.startswith("data:")


def prepare_image(ref: MediaRef, policy: ImagePolicy | None, *, kind: MediaKind = "image") -> PreparedMedia:
    """``ref`` as the judge is sent it under ``policy``.

    Args:
        ref: The stored image.
        policy: The effective image policy (its processor resolved from the judge); ``None`` for none.
        kind: ``image`` or ``frame``, for the record.

    Returns:
        :class:`PreparedMedia` whose ``sent`` inlines the bytes: the resized PNG when ``policy.resizes``, else the
        stored bytes unchanged.

    Raises:
        MediaError: the image cannot be fetched or decoded.
    """
    prepared = _prepared(ref.model_dump_json(), policy if policy is not None and policy.resizes else None)
    return prepared if kind == "image" else prepared._replace(kind=kind)


@lru_cache(maxsize=IMAGE_CACHE_SIZE)
def _prepared(ref_json: str, policy: ImagePolicy | None) -> PreparedMedia:
    ref = MediaRef.model_validate_json(ref_json)
    payload = default_resolver().bytes_of(ref)
    if policy is None:
        sent = _inline(payload, ref.mime or DEFAULT_IMAGE_MIME, width=ref.width, height=ref.height)
        return PreparedMedia(kind="image", source=ref, sent=sent, processor=None, resized=False)
    image = decode_rgb(ref, payload)
    height, width = policy.target_size(image.height, image.width)
    resized = (width, height) != image.size
    if resized:
        from PIL import Image

        image = image.resize((width, height), Image.Resampling.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    sent = _inline(buffer.getvalue(), PREPARED_MIME, width=width, height=height)
    source = ref.model_copy(update={"width": ref.width or image.width, "height": ref.height or image.height})
    return PreparedMedia(kind="image", source=source, sent=sent, processor=policy.processor, resized=resized)


def _inline(payload: bytes, mime: str, *, width: int | None, height: int | None) -> MediaRef:
    uri = f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")
    return MediaRef(
        uri=uri,
        sha256=hashlib.sha256(payload).hexdigest(),
        mime=mime,
        width=width,
        height=height,
        num_bytes=len(payload),
    )


def prepare_content(content: Content, image: ImagePolicy | None, video: VideoPolicy | None) -> PreparedContent:
    """``content`` as the judge is sent it: frames sampled, every image and frame prepared, containers unchanged.

    Args:
        content: One document as stored.
        image: The effective image policy (:meth:`ImagePolicy.for_processor`), or ``None``.
        video: The video policy, or ``None`` (every frame, or the container, as stored).

    Returns:
        :class:`PreparedContent`: the content whose image references are prepared ``data:`` URIs, and one
        :class:`PreparedMedia` per image, frame and container.

    Raises:
        VideoPolicyError: the video policy refuses a clip (:func:`sample_video_part`).
        MediaError: an image cannot be fetched or decoded.
    """
    if not content.has_media:
        return PreparedContent(content, [])
    parts: list[Any] = []
    media: list[PreparedMedia] = []
    for part in content.parts:
        if isinstance(part, ImagePart):
            prepared = prepare_image(part.ref, image)
            media.append(prepared)
            parts.append(part.model_copy(update={"ref": prepared.sent}))
        elif isinstance(part, VideoPart):
            shown = sample_video_part(part, video)
            if shown.frames:
                frames = [prepare_image(ref, image, kind="frame") for ref in shown.frames]
                media.extend(frames)
                parts.append(VideoPart(frames=[frame.sent for frame in frames], frame_indices=shown.frame_indices))
            else:
                assert shown.ref is not None
                media.append(PreparedMedia("video", shown.ref, shown.ref, processor=None, resized=False))
                parts.append(shown)
        else:
            parts.append(part)
    return PreparedContent(Content.from_parts(parts), media)


class MediaCensus:
    """Every media item a judgement store's passes sent, once per ``(corpus, document, source)``.

    Rows go to ``sink`` (the store's ``preprocessing.jsonl``, shared with the text cuts) as JSON lines. A resumed
    pass reads the rows already there and does not write them again. Referent: the media of a document, not its
    presentations: an image shown in fourteen windows is prepared identically fourteen times and recorded once.
    """

    def __init__(self, *, sink: str | Path | None = None) -> None:
        self.sink = Path(sink) if sink is not None else None
        self._seen: set[tuple[str, str, str]] = set()
        if self.sink is not None and self.sink.is_file():
            for line in self.sink.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("mechanism") == MEDIA_MECHANISM:
                    self._seen.add((row["corpus"], row["doc_id"], row["uri"]))

    def record(self, *, corpus: str, doc_id: str, media: list[PreparedMedia]) -> None:
        """Record ``media`` of document ``doc_id`` of ``corpus``, skipping what is already on record."""
        fresh = []
        for item in media:
            key = (corpus, doc_id, item.source.uri)
            if key in self._seen:
                continue
            self._seen.add(key)
            fresh.append(item.as_row(corpus=corpus, doc_id=doc_id))
        if fresh and self.sink is not None:
            with open(self.sink, "a", encoding="utf-8") as handle:
                handle.writelines(json.dumps(row, sort_keys=True) + "\n" for row in fresh)


__all__ = [
    "DEFAULT_IMAGE_MIME",
    "IMAGE_CACHE_SIZE",
    "MEDIA_MECHANISM",
    "MediaCensus",
    "PREPARED_MIME",
    "PreparedContent",
    "PreparedMedia",
    "is_prepared",
    "prepare_content",
    "prepare_image",
]
