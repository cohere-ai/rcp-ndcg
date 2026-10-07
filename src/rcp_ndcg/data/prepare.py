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
import os
import threading
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, NamedTuple

from rcp_ndcg_core.content import Content, ImagePart, MediaRef, VideoPart

from rcp_ndcg.data.media import DEFAULT_IMAGE_MIME, data_uri, decode_rgb, default_resolver
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

    content_tokens: tuple[MediaTokenCount, ...] = ()
    """Per content, its own media token counts (the same rule as :attr:`tokens`). A per-content slice of a
    request (:meth:`select`) needs them; :func:`prepare_request` fills them."""

    def per_content(self) -> tuple[PreparedRequest, ...]:
        """The preparation of each content on its own, in one pass: its media items (the flat list is in
        content order, so it slices by each content's media count) and its exact media token counts.

        The one slicing of a prepared request. A role client that prepares a whole request once and then
        fits each wire request's share of it (one pooling item, one rerank document, the rerank query) takes
        these slices instead of preparing the share again -- a second preparation would re-inline prepared
        bytes and record census rows against ``data:`` URIs -- and the census rows of the kept media are
        recorded per slice. One pass: a corpus encode is one request, so a slice that re-walked every
        content per item would be quadratic in the corpus.

        Raises:
            DataError: this request carries no per-content counts (it was not built by
                :func:`prepare_request`, which fills them).
        """
        if len(self.content_tokens) != len(self.contents):
            raise DataError(
                "a prepared request without per-content token counts cannot be sliced; construct prepared "
                "requests through prepare_request, which fills them",
                hint="prepare the request with prepare_request (it records each content's media token counts)",
            )
        slices: list[PreparedRequest] = []
        offset = 0
        for content, tokens in zip(self.contents, self.content_tokens, strict=True):
            count = sum(len(part.media_refs()) for part in content.parts)
            slices.append(
                PreparedRequest(
                    contents=[content],
                    media=self.media[offset : offset + count],
                    tokens=tokens,
                    content_tokens=(tokens,),
                )
            )
            offset += count
        return tuple(slices)


class MediaFit(NamedTuple):
    """What survives a request's text budget: the media to send, and what the budget dropped.

    The rule (:func:`fit_media_to_budget`) keeps every vision block whole; what it sends is exactly what the
    engine will count.
    """

    media: list[PreparedMedia]
    """The items to send, in part order, images possibly shrunk to the policy's minimum."""

    tokens: int
    """The token count of :attr:`media` as the engine counts it: exact where the sizes are recorded, the
    policy bound otherwise (so the count errs high)."""

    dropped: list[PreparedMedia]
    """The items the budget refused, in drop order (most expensive first); record them in the census."""

    decisions: tuple[MediaRef | None, ...] = ()
    """Per original item (in part order), the sent reference -- the kept item's (possibly shrunk)
    ``sent`` -- or ``None`` when the item was dropped. Positional, so identical items (one page prepared
    twice) are decided per occurrence, never per URI."""

    @property
    def dropped_positions(self) -> tuple[int, ...]:
        """The original indices of the dropped items, ascending (the census's doc ids)."""
        return tuple(index for index, decision in enumerate(self.decisions) if decision is None)


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
       (``min_px`` as both bounds) and is re-prepared, kept only if the shrunk size is one the declared
       budget itself keeps (:meth:`~rcp_ndcg.data.resolution.ImagePolicy.target_size` of the sent size
       returns it) -- a container cannot shrink (the engine decodes it whole), and a minimum whose floored
       size is not a fixed point of the engine's resize, or not one the declared budget keeps, is refused,
       and the item cannot shrink.
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
        return MediaFit([], 0, [], tuple())
    tokens = count(items)
    if tokens <= text_budget_tokens:
        return MediaFit(items, tokens, [], tuple(item.sent for item in items))
    if image.resizes:
        assert image.min_px is not None and image.processor is not None
        minimum = ImagePolicy(min_px=image.min_px, max_px=image.min_px, processor=image.processor)
        shrunk: list[PreparedMedia] = []
        for item in items:
            if item.kind == "video":
                shrunk.append(item)  # the engine decodes the container; nothing client-side to shrink
                continue
            try:
                candidate = prepare_image(item.source, minimum, kind=item.kind)
            except DataError:
                shrunk.append(item)  # no fixed point at the floor; the drop step decides instead
                continue
            sent_height, sent_width = candidate.sent.height or 0, candidate.sent.width or 0
            if image.target_size(sent_height, sent_width) == (sent_height, sent_width):
                shrunk.append(candidate)
            else:
                # flooring at the minimum landed below the declared budget, which would scale the image
                # back up: the sent size would leave the declared instrument, so the item cannot shrink
                shrunk.append(item)
        tokens = count(shrunk)
        if tokens <= text_budget_tokens:
            return MediaFit(shrunk, tokens, [], tuple(item.sent for item in shrunk))
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
    decisions = tuple(items[index].sent if index in keep else None for index in range(len(items)))
    return MediaFit(kept, sum(costs[index] for index in keep), dropped, decisions)


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
    uri = data_uri(mime, base64.b64encode(payload).decode("ascii"))
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
    per_content: list[MediaTokenCount] = []
    tokens = 0
    bounded = 0
    for one in prepared:
        count = content_media_tokens(one.content, image or ImagePolicy.native(), video)
        per_content.append(count)
        tokens, bounded = tokens + count.tokens, bounded + count.bounded
    return PreparedRequest(
        contents=[one.content for one in prepared],
        media=media,
        tokens=MediaTokenCount(tokens, bounded),
        content_tokens=tuple(per_content),
    )


def media_policies_for(config: Any) -> tuple[ImagePolicy | None, VideoPolicy | None]:
    """The effective media policies of a role config: the declared image policy under its processor family.

    The one rule every role client applies (``image_policy =
    config.image_policy.for_processor(config.image_processor)`` when both are declared); a policy without
    either family cannot be counted, and :func:`prepare_request` (through
    :func:`~rcp_ndcg.data.resolution.content_media_tokens`) refuses it rather than guessing.

    Args:
        config: A role config with the media fields (:class:`~rcp_ndcg.inference.config._MediaEndpoint`'s).

    Returns:
        ``(image_policy, video_policy)`` as the preparation and the token counting use them.
    """
    image = getattr(config, "image_policy", None)
    processor = getattr(config, "image_processor", None)
    if image is not None and processor is not None:
        image = image.for_processor(processor)
    return image, getattr(config, "video_policy", None)


def apply_media_fit(contents: Sequence[Content], fit: MediaFit) -> list[Content]:
    """The contents carrying exactly the media :func:`fit_media_to_budget` decided to send.

    A vision block is atomic: the fit shrinks whole items to the policy minimum and drops whole items (most
    expensive first, every drop recorded by the caller). This applies its positional decisions
    (:attr:`MediaFit.decisions` -- per original item in part order, the sent reference or ``None``) to the
    prepared contents: a kept item's part carries the decision's reference, a dropped item's part leaves the
    content. A content that loses every part it had becomes the empty content, which the caller's
    ``empty_doc`` policy then handles like any empty document.

    Args:
        contents: The prepared contents (:func:`prepare_request` returned them).
        fit: The budget's decision (:func:`fit_media_to_budget` returned it), whose decisions are positional
            over ``PreparedRequest.media``.

    Returns:
        One content per input, in order, with the sent media in place and the dropped media removed.

    Raises:
        DataError: the decisions cannot be matched to the prepared parts (a different request's fit).
    """
    kept: list[Content] = []
    applied = 0
    for content in contents:
        parts: list[Any] = []
        for part in content.parts:
            if not part.media_refs():
                parts.append(part)
                continue
            if isinstance(part, VideoPart):
                if part.frames:
                    # A dropped frame leaves with its sampled-index entry, so the sent video's frames and
                    # its provenance stay aligned: a fit that kept 1 of 10 frames must not ship a part
                    # that still claims 10 sampled source frames (the next count of it refuses).
                    sampled = part.frame_indices
                    pairs: list[tuple[Any, Any]] = []
                    for frame_position, _frame in enumerate(part.frames):
                        decision = fit.decisions[applied]
                        index = sampled[frame_position] if sampled and frame_position < len(sampled) else frame_position
                        applied += 1
                        if decision is not None:
                            pairs.append((decision, index))
                    if pairs:
                        update: dict[str, Any] = {"frames": [frame for frame, _ in pairs]}
                        if sampled is not None:
                            update["frame_indices"] = [index for _, index in pairs]
                        parts.append(part.model_copy(update=update))
                    continue
                # A ref-only container (``wire: video_url``): one prepared item, sent or dropped whole.
                decision = fit.decisions[applied]
                applied += 1
                if decision is not None:
                    parts.append(part.model_copy(update={"ref": decision}))
                continue
            if isinstance(part, ImagePart):
                decision = fit.decisions[applied]
                applied += 1
                if decision is not None:
                    parts.append(part.model_copy(update={"ref": decision}))
                continue
            parts.append(part)
        kept.append(Content.from_parts(parts))
    if applied != len(fit.decisions):
        raise DataError(
            f"applied {applied} media decision(s) for {len(fit.decisions)} prepared item(s); the parts of "
            "these contents do not carry the media :func:`prepare_request` prepared for them",
            hint="apply_media_fit is called with the media :func:`prepare_request` prepared for the same "
            "contents, in part order",
        )
    return kept


class MediaCensus:
    """Every media item a judgement store's passes sent, once per ``(corpus, document, source, outcome)``.

    Rows go to ``sink`` (the store's ``preprocessing.jsonl``, shared with the text cuts) as JSON lines. A resumed
    pass reads the rows already there and does not write them again. Referent: the media of a document, not its
    presentations: an image shown in fourteen windows is prepared identically fourteen times and recorded once.
    The dedup key names the outcome (``dropped`` or not) beside the source, so a budget that first kept an
    item and a later one that refused it are both on record -- a kept pass must not hide a later drop.

    The sink is read and appended through the census's one reader and writer
    (:func:`~rcp_ndcg.data.preprocess.read_census_rows`, :func:`~rcp_ndcg.data.preprocess.append_census_rows`):
    a torn last line is skipped with a warning, and appends run under the sink's writer lock. Within one
    census, ``record`` (a check-then-append) is serialised by the instance's lock, so concurrent recorders
    cannot double-write a row.
    """

    def __init__(self, *, sink: str | Path | None = None) -> None:
        self.sink = Path(sink) if sink is not None else None
        self._lock = threading.Lock()
        self._seen: set[tuple[str, str, str, bool]] = set()
        if self.sink is not None:
            from rcp_ndcg.data.preprocess import read_census_rows

            for row in read_census_rows(self.sink):
                if row.get("mechanism") == MEDIA_MECHANISM:
                    self._seen.add((row["corpus"], row["doc_id"], row["uri"], bool(row.get("dropped", False))))

    def recorded(self) -> tuple[tuple[str, str, str, bool], ...]:
        """What is on record, as ``(corpus, doc_id, uri, dropped)`` rows -- the dedup keys, ascending. A
        caller (a test, a run summary) reads this instead of the sink's lines or the private set."""
        with self._lock:
            return tuple(sorted(self._seen))

    def record(self, *, corpus: str, doc_id: str, media: list[PreparedMedia], dropped: bool = False) -> None:
        """Record ``media`` of document ``doc_id`` of ``corpus``, skipping what is already on record.

        ``dropped=True`` records items a request's text budget refused (:func:`fit_media_to_budget` returns
        them); they were never sent, and the row says so. A kept row and a dropped row of the same item are
        distinct records -- the outcome is part of the dedup key, so a drop after a kept pass of the same
        item is still recorded.
        """
        with self._lock:
            fresh = []
            for item in media:
                key = (corpus, doc_id, item.source.uri, dropped)
                if key in self._seen:
                    continue
                self._seen.add(key)
                fresh.append(item.as_row(corpus=corpus, doc_id=doc_id, dropped=dropped))
            if fresh and self.sink is not None:
                from rcp_ndcg.data.preprocess import append_census_rows

                # The one census append: the torn tail cut and the rows written under the sink's writer lock,
                # on every append (a peer killed after this writer started leaves a tail only its next append
                # merges).
                append_census_rows(self.sink, fresh)


__all__ = [
    "DEFAULT_IMAGE_MIME",
    "apply_media_fit",
    "media_policies_for",
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
