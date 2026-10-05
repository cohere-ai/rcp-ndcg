"""Client-side media preparation: every image and video frame, sized as the judge's own processor would size it.

A judge endpoint only needs to be OpenAI-compatible: the engine runs stock, without media flags, because the client
prepares every image before it is sent, deterministically, and records what it sent. :func:`prepare_content` is the
one place this happens; the payload builder (:mod:`rcp_ndcg.llm._payload`) sends only what it returns.

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
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, NamedTuple

from rcp_ndcg_core.content import Content, ImagePart, MediaRef, VideoPart

from rcp_ndcg.data.media import decode_rgb, default_resolver
from rcp_ndcg.data.resolution import (
    ImagePolicy,
    MediaTokenCount,
    VideoPolicy,
    content_media_tokens,
    sample_video_part,
)
from rcp_ndcg.errors import DataError
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

    def as_row(self, *, corpus: str, doc_id: str, dropped: bool = False) -> dict[str, Any]:
        """The census row: what was stored, what was sent, and how.

        ``dropped=True`` records an item a request's text budget refused (never sent): the row holds the
        prepared state it was refused at, so a refusal is never silent.
        """
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
            "dropped": dropped,
        }


class PreparedContent(NamedTuple):
    """Content as the judge is sent it, and every media item in it."""

    content: Content
    media: list[PreparedMedia]


class PreparedRequest(NamedTuple):
    """One retrieval request's contents as the endpoint is sent them, with its exact media token counts.

    ``tokens`` is what the request's media costs the prompt, as the engine counts it -- each image and
    sampled frame its vision block (:data:`~rcp_ndcg.data.resolution.VISION_WRAPPER_TOKENS` plus the patch
    tokens), a container its temporal grid -- so the role's text budget can subtract it and never cut it.
    """

    contents: list[Content]
    """The request's contents, in order, with every image and frame prepared (:func:`prepare_content`)."""

    media: list[PreparedMedia]
    """Every prepared item, across all contents, in request order."""

    tokens: MediaTokenCount
    """The request's media token counts: exact where the sizes were recorded, a bound (counted in
    ``bounded``) where they were not."""


class MediaFit(NamedTuple):
    """What survives a request's text budget: the media to send, and what the budget dropped.

    The rule (:func:`fit_media_to_budget`) keeps every vision block whole; what it sends is exactly what the
    engine will count.
    """

    media: list[PreparedMedia]
    """The items to send, in part order, images possibly shrunk to the policy's minimum."""

    tokens: int
    """The exact token count of :attr:`media` as the engine counts it (unrecorded sizes are counted at their
    bound, so the count errs high)."""

    dropped: list[PreparedMedia]
    """The items the budget refused, in drop order (most expensive first); record them in the census."""


def fit_media_to_budget(
    media: Sequence[PreparedMedia], *, image: ImagePolicy, video: VideoPolicy | None, text_budget_tokens: int
) -> MediaFit:
    """What to send of a request's media when media alone exceed its text budget.

    A vision block is atomic -- the engine either sees a whole media item or none of it, never a cut
    through one: the vision start and end markers wrap the patch run, and a prompt cut between them
    would corrupt or orphan the block.

    The declared rule, in order, and never anything else:

    1. Media that fit the budget whole go whole.
    2. Otherwise every image and frame shrinks to the image policy's minimum pixel budget
       (``min_px`` as both bounds) and is re-prepared -- a container cannot shrink (the engine decodes it
       whole), and an image whose floored minimum is not a fixed point of the engine's resize is refused
       by :meth:`~rcp_ndcg.data.resolution.ImagePolicy.target_size` and so cannot shrink either.
    3. What still does not fit is dropped, most expensive first (ties keep the earlier part), until the
       remaining media fit. Items are dropped whole: tokens are never cut inside a vision block.

    Args:
        media: The request's prepared media (:func:`prepare_content` or :func:`prepare_request` returns them).
        image: The effective image policy, whose token counts the decision uses and whose minimum the
            shrink step targets.
        video: The video policy, for a container's temporal-grid count.
        text_budget_tokens: What the request's text budget leaves for media alone, in tokens.

    Returns:
        :class:`MediaFit`: the items to send with their exact token count, and the drops -- record them in
        the census (:meth:`MediaCensus.record` with ``dropped=True``), so a refusal is never silent.

    Raises:
        ConfigError: media whose token cost cannot be counted (a native policy, or no processor family) --
            a budget is never decided on a guess.
    """

    def count(items: Sequence[PreparedMedia]) -> int:
        return content_media_tokens(_content(items), image, video).tokens

    items = list(media)
    if not items:
        return MediaFit([], 0, [])
    tokens = count(items)
    if tokens <= text_budget_tokens:
        return MediaFit(items, tokens, [])
    if image.resizes:
        assert image.min_px is not None and image.processor is not None
        minimum = ImagePolicy(min_px=image.min_px, max_px=image.min_px, processor=image.processor)
        shrunk: list[PreparedMedia] = []
        for item in items:
            if item.kind == "video":
                shrunk.append(item)  # the engine decodes the container; nothing client-side to shrink
                continue
            try:
                shrunk.append(prepare_image(item.source, minimum, kind=item.kind))
            except DataError:
                shrunk.append(item)  # no fixed point at the floor; the drop step decides instead
        tokens = count(shrunk)
        if tokens <= text_budget_tokens:
            return MediaFit(shrunk, tokens, [])
        items = shrunk
    costs = [count([item]) for item in items]
    keep = list(range(len(items)))
    dropped: list[PreparedMedia] = []
    # drop whole items, most expensive first (ties keep the earlier part), until the rest fit
    while keep and sum(costs[index] for index in keep) > text_budget_tokens:
        dearest = max(keep, key=lambda index: (costs[index], index))
        keep.remove(dearest)
        dropped.append(items[dearest])
    kept = [items[index] for index in keep]
    return MediaFit(kept, sum(costs[index] for index in keep), dropped)


def _content(items: Sequence[PreparedMedia]) -> Content:
    """The content whose parts are exactly ``items``, for token counting."""
    parts: list[Any] = []
    for item in items:
        if item.kind == "video":
            parts.append(VideoPart(ref=item.sent))
        else:
            parts.append(ImagePart(ref=item.sent))
    return Content.from_parts(parts)


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


def prepare_request(
    contents: Sequence[Content], image: ImagePolicy | None, video: VideoPolicy | None
) -> PreparedRequest:
    """One retrieval request's contents as the endpoint is sent them, with the request's exact media token
    counts.

    The one preparation call the retrieval role clients make (an embedding item; a rerank pair's query and
    documents): every content is prepared through :func:`prepare_content` -- the judge's own path, so the
    frames sampled and the pixels sent are the same instrument the judge sees -- and the media token counts
    are summed over the request (:func:`~rcp_ndcg.data.resolution.content_media_tokens`, what the engine adds
    to the prompt for the media). The role's text budget subtracts ``tokens.tokens``; it never cuts the media:
    when media alone exceed the budget, :func:`fit_media_to_budget` decides what is sent instead.

    Args:
        contents: The request's contents, in order.
        image: The effective image policy (the role config's, under its ``image_processor``), or ``None``.
        video: The video policy, or ``None``.

    Returns:
        :class:`PreparedRequest`: the prepared contents, every prepared item, and the request's media token
        counts (exact where the references carry sizes; a bound, counted in ``bounded``, where they do not).

    Raises:
        VideoPolicyError: the video policy refuses a clip.
        MediaError: an image cannot be fetched or decoded.
        ConfigError: media whose token cost cannot be counted (a native policy, or no processor family).
    """
    prepared = [prepare_content(content, image, video) for content in contents]
    media = [item for one in prepared for item in one.media]
    tokens = 0
    bounded = 0
    for one in prepared:
        count = content_media_tokens(one.content, image or ImagePolicy.native(), video)
        tokens, bounded = tokens + count.tokens, bounded + count.bounded
    return PreparedRequest(
        contents=[one.content for one in prepared], media=media, tokens=MediaTokenCount(tokens, bounded)
    )


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

    def record(self, *, corpus: str, doc_id: str, media: list[PreparedMedia], dropped: bool = False) -> None:
        """Record ``media`` of document ``doc_id`` of ``corpus``, skipping what is already on record.

        ``dropped=True`` records items a request's text budget refused (:func:`fit_media_to_budget` returns
        them); they were never sent, and the row says so.
        """
        fresh = []
        for item in media:
            key = (corpus, doc_id, item.source.uri)
            if key in self._seen:
                continue
            self._seen.add(key)
            fresh.append(item.as_row(corpus=corpus, doc_id=doc_id, dropped=dropped))
        if fresh and self.sink is not None:
            with open(self.sink, "a", encoding="utf-8") as handle:
                handle.writelines(json.dumps(row, sort_keys=True) + "\n" for row in fresh)


__all__ = [
    "DEFAULT_IMAGE_MIME",
    "IMAGE_CACHE_SIZE",
    "MEDIA_MECHANISM",
    "MediaCensus",
    "MediaFit",
    "PREPARED_MIME",
    "PreparedContent",
    "PreparedMedia",
    "PreparedRequest",
    "fit_media_to_budget",
    "is_prepared",
    "prepare_content",
    "prepare_image",
    "prepare_request",
]
