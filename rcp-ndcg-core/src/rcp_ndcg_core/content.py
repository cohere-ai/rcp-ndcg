"""Modality-agnostic content: the shape a query or document has.

A query or a document is an **ordered list of interleaved parts** -- text, image
or video -- rather than a string, and **media is carried by reference, not by
index or by bytes.**

An integer index into a side array of image bytes works inside a parquet shard
and nowhere else.  JSONL cannot hold bytes, and
this project already has an fsspec storage layer with a content-addressed cache
plus hash-keyed run manifests.  A :class:`MediaRef` carrying ``uri + sha256 +
mime + dims`` slots into all of that for free.

Importing this module costs pydantic, nothing else, so it stays usable from the
dependency-light public surface.

    >>> Content.from_text("a passage")
    >>> page = Content.from_image("gs://bucket/doc/page_1.png", sha256="ab...")
    >>> page.has_media
    True
    >>> page.text
    ''
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

# How text parts are joined when a multi-part document is flattened to a string.
TEXT_JOIN = "\n"


class Modality(StrEnum):
    """What a side of a dataset is made of.

    Declared in dataset metadata rather than sniffed from the rows: a corpus
    where the first thousand documents happen to be text-only must not be
    silently treated as a text corpus.
    """

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    # Interleaved text and media in one document, e.g. a page image with its
    # OCR transcript, or a passage with an inline figure.
    MULTIMODAL = "multimodal"


class MediaRef(BaseModel):
    """A reference to one media asset.

    ``uri`` goes through :mod:`rcp_ndcg.storage`, so a local path, a
    ``gs://`` object, an ``s3://`` object and an ``hf://`` file are all valid
    and interchangeable.

    ``sha256`` is the content hash of the *bytes at the uri*.  It is optional
    because a reader may not have paid to hash a 10 GiB corpus yet, but once
    present it is load-bearing in two places: verify-on-read in the media
    resolver, and the local cache key.  ``width`` / ``height`` let the token-cost estimator work
    without fetching a single byte.
    """

    model_config = ConfigDict(extra="forbid")

    uri: str
    sha256: str | None = None
    mime: str | None = None
    width: int | None = None
    height: int | None = None
    num_bytes: int | None = None

    # Video-only, and only ever populated for a container. Frame directories
    # describe themselves through ``VideoPart.frames``.
    num_frames: int | None = None
    duration_s: float | None = None
    fps: float | None = None

    @property
    def cache_key(self) -> str:
        """Stable identity for caching. Prefers the hash; falls back to the uri."""
        return self.sha256 or self.uri


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str

    def media_refs(self) -> list[MediaRef]:
        return []


class ImagePart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image"] = "image"
    ref: MediaRef
    # Provenance for images rendered from a paginated source, so a retrieved
    # page can be cited as "page 7 of this PDF" rather than as an opaque id.
    page: int | None = None

    def media_refs(self) -> list[MediaRef]:
        return [self.ref]


class VideoPart(BaseModel):
    """A video: a container reference (``ref``), pre-extracted frames (``frames``), or both.

    A directory of extracted JPEG frames is ``frames`` populated, with ``ref``
    absent or pointing at the directory. The judging payload sends a container
    whole to a judge that reads video and frames as images; nothing here decodes
    a container.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["video"] = "video"
    ref: MediaRef | None = None
    frames: list[MediaRef] = Field(default_factory=list)
    # Which frame indices of the source these frames were sampled at. Recorded
    # so a run at 8 frames and a run at 32 are distinguishable after the fact.
    frame_indices: list[int] | None = None

    @model_validator(mode="after")
    def _require_container_or_frames(self) -> Self:
        if self.ref is None and not self.frames:
            raise ValueError("VideoPart needs either a container `ref` or at least one entry in `frames`")
        return self

    def media_refs(self) -> list[MediaRef]:
        """The container then the frames -- every asset this part needs on disk."""
        return ([self.ref] if self.ref is not None else []) + list(self.frames)


Part = Annotated[TextPart | ImagePart | VideoPart, Field(discriminator="type")]


def split_text_across_parts(parts: Sequence[str], kept: str) -> list[tuple[str, bool]]:
    """The kept piece of every text part when the :data:`TEXT_JOIN`-joined text is cut to *kept*.

    One walk of the parts against the joined text, so a cut of the joined string lands on the parts in
    their own places: each piece is that part's own prefix, in order, and a piece's join is *kept* minus a
    trailing join newline (a cut that ends on a join drops it). The second element says whether the part
    survives in the cut content: a piece that is empty is dropped, unless its part's first character was
    inside the cut (an empty part whose join newline the cut reached is kept).

    ``kept`` must be a prefix of the joined text -- every caller cuts the joined text with a
    token-boundary prefix (``token_prefix``), so the distribution is exact, never a re-tokenisation.

    Args:
        parts: The text parts, in order, whose :data:`TEXT_JOIN` join is the text the cut applies to.
        kept: The kept prefix of the joined text.

    Returns:
        One ``(piece, kept)`` pair per part, in order; the pieces' join is *kept* -- minus the trailing
        join newline when the cut ends on one (the same rule :meth:`Content.truncated` states).

    Raises:
        ValueError: ``kept`` is not a prefix of the joined text (a caller bug: the cut must come from
            the joined text, or the pieces would not reassemble it).
    """
    full = TEXT_JOIN.join(parts)
    if not full.startswith(kept):
        raise ValueError(f"the kept text is not a prefix of the joined parts ({kept!r} vs {full!r})")
    out: list[tuple[str, bool]] = []
    start = 0  # the current text part's first character's position in the joined text
    for part in parts:
        piece = part[: max(0, len(kept) - start)]
        out.append((piece, bool(piece) or start < len(kept)))
        start += len(part) + len(TEXT_JOIN)
    return out


class Content(RootModel[list[Part]]):
    """An ordered list of parts -- the body of a query or a document.

    Serialises as a bare JSON array, so a JSONL line reads
    ``{"doc_id": "d1", "content": [{"type": "text", "text": "..."}]}`` with no
    wrapper object.
    """

    root: list[Part] = Field(default_factory=list)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_text(cls, text: str) -> Content:
        return cls(root=[TextPart(text=text)])

    @classmethod
    def from_image(cls, uri: str, *, page: int | None = None, **ref_kwargs) -> Content:
        """One image part: ``uri`` and the :class:`MediaRef` fields in ``ref_kwargs``."""
        return cls(root=[ImagePart(ref=MediaRef(uri=uri, **ref_kwargs), page=page)])

    @classmethod
    def from_parts(cls, parts: Sequence[Part]) -> Content:
        return cls(root=list(parts))

    # -- views -------------------------------------------------------------
    @property
    def parts(self) -> list[Part]:
        return self.root

    @property
    def text(self) -> str:
        """The text parts joined. Empty for an image-only document.

        Media contributes nothing here on purpose: a caller that only knows how
        to handle strings should see an image-only document as having no text,
        not as having a placeholder it might index or judge.  Callers that must
        distinguish "no text" from "no content" use :attr:`has_media`.
        """
        return TEXT_JOIN.join(part.text for part in self.root if isinstance(part, TextPart))

    @property
    def media(self) -> list[MediaRef]:
        """Every media reference, in order, flattening video frames."""
        return [ref for part in self.root for ref in part.media_refs()]

    @property
    def has_media(self) -> bool:
        return any(isinstance(part, ImagePart | VideoPart) for part in self.root)

    @property
    def has_text(self) -> bool:
        return any(isinstance(part, TextPart) and part.text for part in self.root)

    @property
    def modality(self) -> Modality:
        has_image = any(isinstance(p, ImagePart) for p in self.root)
        has_video = any(isinstance(p, VideoPart) for p in self.root)
        if self.has_text and (has_image or has_video):
            return Modality.MULTIMODAL
        if has_video:
            return Modality.VIDEO
        if has_image:
            return Modality.IMAGE
        return Modality.TEXT

    # -- transforms --------------------------------------------------------
    def with_text_prefix(self, prefix: str) -> Content:
        """A copy with *prefix* in front, as a text part.

        What asymmetric retrieval models need: ``"Query: "`` on one side,
        ``"Passage: "`` on the other. Prefixing ``.text`` and rebuilding would
        drop the images from an image-only document, so this prepends a part
        instead -- and merges into the leading text part when there is one, so a
        text-only document produces exactly the string it did before content parts
        existed.
        """
        if not prefix:
            return self
        if self.root and isinstance(self.root[0], TextPart):
            head = self.root[0]
            return Content(root=[TextPart(text=f"{prefix}{head.text}"), *self.root[1:]])
        return Content(root=[TextPart(text=prefix), *self.root])

    def truncated(self, max_chars: int | None) -> Content:
        """A copy whose :attr:`text` is the first *max_chars* characters of this one's, media untouched.

        The budget being spent is the context window, and a document's text is the
        only part of it that shortens meaningfully: half a page image is not a
        shorter document, it is a different one. So an image window that does not
        fit has to be made to fit by sending fewer documents or fewer pixels, and
        this method deliberately cannot do it -- which is why
        :func:`rcp_ndcg.judging.judging.window_tokens` raises instead of
        returning a smaller number.

        The budget is per document, so it is applied to the joined text rather than
        per part (:func:`split_text_across_parts` is the one walk that distributes it,
        text parts staying in their places around the media), and the newlines joining
        text parts count: the result's :attr:`text` is a verbatim prefix of this one's
        (a cut that ends on a joining newline drops it). An empty text part keeps its
        joining newline: the cut is a prefix of the joined text, empty parts included.

        Raises:
            ValueError: ``max_chars`` is negative (a caller bug; ``0`` legitimately cuts to nothing).
        """
        if max_chars is not None and max_chars < 0:
            raise ValueError(
                f"max_chars must be 0 or a positive number of characters (None keeps the text), got {max_chars}"
            )
        full = self.text
        if max_chars is None or not self.has_text or max_chars >= len(full):
            return self
        text_parts = [part.text for part in self.root if isinstance(part, TextPart)]
        pieces = iter(split_text_across_parts(text_parts, full[:max_chars]))
        parts: list[Part] = []
        for part in self.root:
            if isinstance(part, TextPart):
                piece, kept = next(pieces)
                if kept:
                    parts.append(TextPart(text=piece))
            else:
                parts.append(part)
        return Content(root=parts)

    # -- sequence protocol -------------------------------------------------
    def __iter__(self) -> Iterator[Part]:  # type: ignore[override]
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def __getitem__(self, index: int) -> Part:
        return self.root[index]

    def __bool__(self) -> bool:
        return bool(self.root)


__all__ = [
    "TEXT_JOIN",
    "Content",
    "ImagePart",
    "MediaRef",
    "Modality",
    "Part",
    "TextPart",
    "VideoPart",
    "split_text_across_parts",
]
