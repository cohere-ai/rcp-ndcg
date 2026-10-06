"""Turning a :class:`MediaRef` into bytes or a PIL image.

Content-addressed caching, which is what makes this cheap at corpus scale.  The
generic :func:`rcp_ndcg.storage.cache` is URI-keyed and therefore has to ask
the backend "did this change?" on every call -- one metadata round-trip per
object, which for 10^5 page images dominates the actual reads.  A ``MediaRef``
with a ``sha256`` cannot change by definition, so a cache hit needs no network
at all: the hash *is* the freshness check.

Verification is on write, not on read.  Bytes are hashed once as they enter the
cache and the file is only published under its hash if it matches; a later
reader trusting a cache hit is then trusting a check that already happened,
rather than paying to re-hash every image on every read.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import struct
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from rcp_ndcg_core.content import Content, MediaRef, VideoPart

from rcp_ndcg import storage
from rcp_ndcg.errors import DataError, MissingInputError, RcpNdcgError
from rcp_ndcg.support.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image

logger = get_logger(__name__)

#: Subdirectory of the shared cache holding media, keyed by content hash.
MEDIA_CACHE_DIRNAME = "media"

#: Overrides where media is cached.  Needed wherever the package is a dependency
#: rather than the checkout: see :func:`_media_cache_dir`.
MEDIA_CACHE_ENV = "RCP_NDCG_MEDIA_CACHE"


class MediaError(DataError):
    """A media asset could not be read or decoded, did not match its hash, or cannot be sent the way asked
    (exit code 12, like every :class:`DataError`). A missing file is a :class:`MissingInputError`."""


def _media_cache_dir() -> Path:
    """Where fetched media is kept: ``$RCP_NDCG_MEDIA_CACHE``, else ``media/`` under the package cache.

    The override lets a container or a Space point the media cache at a mounted volume on its own; the package
    cache is :func:`rcp_ndcg.support.paths.cache_dir`.
    """
    override = os.environ.get(MEDIA_CACHE_ENV)
    if override:
        return Path(override).expanduser()
    from rcp_ndcg.support.paths import cache_dir

    return cache_dir() / MEDIA_CACHE_DIRNAME


def sha256_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class MediaResolver:
    """Resolves media references to bytes or images, through a content-addressed cache in
    ``$RCP_NDCG_MEDIA_CACHE`` (default ``media/`` under the package cache)."""

    def __init__(self) -> None:
        self.cache_dir = _media_cache_dir()

    # -- cache layout ------------------------------------------------------
    def cache_path(self, ref: MediaRef) -> Path:
        """Where *ref* lives on disk once cached.

        Hashed refs shard by the first two hex characters, which keeps any one
        directory from growing to 10^5 entries. Unhashed refs fall back to a
        digest of the URI -- still stable, but it cannot detect the remote
        object changing underneath, which is why hashing at ingest matters.
        """
        if ref.sha256:
            digest = ref.sha256
            prefix = "by-hash"
        else:
            digest = hashlib.sha256(ref.uri.encode()).hexdigest()
            prefix = "by-uri"
        suffix = Path(ref.uri).suffix if len(Path(ref.uri).suffix) <= 6 else ""
        return self.cache_dir / prefix / digest[:2] / f"{digest}{suffix}"

    # -- resolution --------------------------------------------------------
    def bytes_of(self, ref: MediaRef) -> bytes:
        """The bytes behind *ref*, fetching and caching on first use.

        A ``data:`` URI carries its bytes inline (what the preparation inlines): decoded here, never
        fetched."""
        if ref.uri.startswith("data:"):
            header, _, payload = ref.uri.partition(",")
            if "base64" in header:
                return base64.b64decode(payload)
        return self.local_path(ref).read_bytes()

    def local_path(self, ref: MediaRef) -> Path:
        """A real file on disk holding *ref*, for libraries that need a path."""
        target = self.cache_path(ref)
        if target.exists():
            return target
        payload = self._fetch(ref)
        if ref.sha256:
            actual = sha256_of(payload)
            if actual != ref.sha256:
                raise MediaError(
                    f"content hash mismatch fetching {ref.uri}: declared {ref.sha256}, actual {actual}. "
                    "Refusing to cache it: every downstream cache key would then be wrong."
                )
        _atomic_write_bytes(target, payload)
        return target

    def _fetch(self, ref: MediaRef) -> bytes:
        try:
            return storage.read_bytes(ref.uri)
        except FileNotFoundError as exc:
            raise MissingInputError(
                f"media not found: {ref.uri}",
                hint="a dataset refers to its media by path: restore the file, or load the dataset again from where "
                "its media now is",
            ) from exc
        except RcpNdcgError:
            raise
        except Exception as exc:  # noqa: BLE001 - backend errors vary by protocol
            raise MediaError(f"failed to read media {ref.uri}: {exc}") from exc

    def image(self, ref: MediaRef) -> Image:
        """*ref* as an RGB :class:`PIL.Image.Image`, decoded by :func:`decode_rgb`."""
        return decode_rgb(ref, self.bytes_of(ref))

    def images_of(self, content: Content) -> list[Image]:
        """Every image in *content*, in order, video frames included."""
        return [self.image(ref) for ref in content.media]

    # -- hashing at ingest -------------------------------------------------
    def hydrate(self, ref: MediaRef) -> MediaRef:
        """Return *ref* with ``sha256``, ``num_bytes`` and dimensions filled in.

        What an ingest calls so that everything downstream -- the cache key, the
        token estimate, the duration check -- can work without opening the file
        again.
        """
        payload = self.bytes_of(ref) if ref.sha256 else self._fetch(ref)
        digest = ref.sha256 or sha256_of(payload)
        update: dict[str, Any] = {"sha256": digest, "num_bytes": ref.num_bytes or len(payload)}
        header = probe_video_header(payload)
        if header is not None:
            # A container: its header states size, length and frame count, which is
            # what prices it and what a duration limit is checked against.
            recorded = {"width": ref.width, "height": ref.height, "num_frames": ref.num_frames}
            recorded |= {"duration_s": ref.duration_s, "fps": ref.fps}
            update |= {name: getattr(header, name) if value is None else value for name, value in recorded.items()}
        elif ref.width is None or ref.height is None:
            update["width"], update["height"] = _probe_dimensions(payload)
        hydrated = ref.model_copy(update=update)
        if not ref.sha256:
            _atomic_write_bytes(self.cache_path(hydrated), payload)
        return hydrated


def _probe_dimensions(payload: bytes) -> tuple[int | None, int | None]:
    """Image dimensions from the header, without decoding the pixels."""
    from PIL import Image as PILImage

    try:
        with PILImage.open(io.BytesIO(payload)) as handle:
            return handle.width, handle.height
    except OSError:
        # A non-image asset legitimately has no dimensions readable this way; the
        # caller records None rather than guessing.
        return None, None


#: Image suffixes read as media, and the MIME type each is recorded and sent under.
IMAGE_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}

#: Container suffixes read as video, and the MIME type each is sent under.
VIDEO_MIME_BY_SUFFIX = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
}


class VideoHeader(NamedTuple):
    """What a container's header says about its video track, without decoding it.

    ``width`` / ``height`` in pixels, ``duration_s`` in seconds, ``fps`` in frames
    per second. A field the header does not state is ``None``.
    """

    width: int | None
    height: int | None
    num_frames: int | None
    duration_s: float | None
    fps: float | None


def probe_video_header(payload: bytes) -> VideoHeader | None:
    """Read a video container's header: MP4/MOV (ISO BMFF) and AVI.

    Pure Python over the bytes already in hand -- no decoder, no torch, no ffmpeg --
    because ingest needs frame count, duration and size to price a clip and to
    check a duration limit, and nothing more. Returns ``None`` for anything that is
    not one of these containers (including WebM/Matroska, whose clips are then
    recorded unprobed and priced at the policy's bound).

    Args:
        payload: The whole file.

    Returns:
        The video track's header fields, or ``None`` when the bytes are not a
        container this reads.
    """
    if payload[:4] == b"RIFF" and payload[8:12] == b"AVI ":
        return _avi_header(payload)
    if payload[4:8] in {b"ftyp", b"moov", b"mdat", b"free", b"wide", b"skip"}:
        return _isobmff_header(payload)
    return None


def _avi_header(payload: bytes) -> VideoHeader | None:
    """``avih`` (the main AVI header): microseconds per frame, total frames, size."""
    at = payload.find(b"avih", 12, 4096)
    if at < 0 or len(payload) < at + 48:
        return None
    usec_per_frame, _, _, _, total_frames = struct.unpack_from("<5I", payload, at + 8)
    width, height = struct.unpack_from("<2I", payload, at + 8 + 32)
    fps = 1e6 / usec_per_frame if usec_per_frame else None
    duration = total_frames / fps if fps else None
    return VideoHeader(width or None, height or None, total_frames or None, duration, fps)


def _boxes(payload: bytes, start: int, end: int):
    """``(type, body_start, body_end)`` for each ISO BMFF box in ``[start, end)``."""
    while start + 8 <= end:
        size, kind = struct.unpack_from(">I4s", payload, start)
        header = 8
        if size == 1:
            if start + 16 > end:
                return
            size = struct.unpack_from(">Q", payload, start + 8)[0]
            header = 16
        elif size == 0:
            size = end - start
        if size < header or start + size > end:
            return
        yield kind, start + header, start + size
        start += size


def _child(payload: bytes, start: int, end: int, kind: bytes) -> tuple[int, int] | None:
    for found, body, stop in _boxes(payload, start, end):
        if found == kind:
            return body, stop
    return None


def _isobmff_header(payload: bytes) -> VideoHeader | None:
    """``moov/trak`` of the first video track: ``tkhd`` size, ``mdhd`` length, ``stsz`` count."""
    moov = _child(payload, 0, len(payload), b"moov")
    if moov is None:
        return None
    for kind, body, stop in _boxes(payload, *moov):
        if kind != b"trak":
            continue
        mdia = _child(payload, body, stop, b"mdia")
        hdlr = _child(payload, *mdia, b"hdlr") if mdia else None
        if mdia is None or hdlr is None or payload[hdlr[0] + 8 : hdlr[0] + 12] != b"vide":
            continue
        width = height = num_frames = None
        duration = fps = None
        tkhd = _child(payload, body, stop, b"tkhd")
        if tkhd is not None:
            # Fixed-point 16.16 width and height are the box's last eight bytes.
            width, height = (value >> 16 for value in struct.unpack_from(">2I", payload, tkhd[1] - 8))
        mdhd = _child(payload, *mdia, b"mdhd")
        if mdhd is not None:
            version = payload[mdhd[0]]
            layout, offset = (">IQ", 20) if version == 1 else (">II", 12)
            timescale, units = struct.unpack_from(layout, payload, mdhd[0] + offset)
            duration = units / timescale if timescale else None
        minf = _child(payload, *mdia, b"minf")
        stbl = _child(payload, *minf, b"stbl") if minf else None
        stsz = _child(payload, *stbl, b"stsz") if stbl else None
        if stsz is not None:
            num_frames = struct.unpack_from(">I", payload, stsz[0] + 8)[0] or None
        if num_frames and duration:
            fps = num_frames / duration
        return VideoHeader(width or None, height or None, num_frames, duration, fps)
    return None


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    """Publish *payload* at *target* via a temp file in the same directory.

    Concurrent readers -- the N worker ranks that all resolve the same
    corpus at start-up -- would otherwise observe a half-written image.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.{os.getpid()}.", dir=target.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


_DEFAULT_RESOLVER: MediaResolver | None = None


def default_resolver() -> MediaResolver:
    """The process-wide resolver, so the cache is shared across call sites."""
    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is None:
        _DEFAULT_RESOLVER = MediaResolver()
    return _DEFAULT_RESOLVER


def store_media(
    payload: bytes, extension: str, *, root: str | None = None, width: int | None = None, height: int | None = None
) -> MediaRef:
    """Write ``payload`` once under its content hash and return a hashed :class:`MediaRef` to it.

    Args:
        payload: The encoded bytes (e.g. a PNG).
        extension: The file suffix, with its dot (``".png"``); :data:`IMAGE_MIME_BY_SUFFIX` gives its MIME type.
        root: The directory to write under (``<root>/<sha[:2]>/<sha><ext>``); ``None`` writes into the media
            cache in the resolver's own layout, so reading the reference back needs no second copy.
        width, height: The image's dimensions, when known.
    """
    mime = IMAGE_MIME_BY_SUFFIX.get(extension.lower())
    if mime is None:
        raise MediaError(f"unknown image type {extension!r}; known: {sorted(IMAGE_MIME_BY_SUFFIX)}")
    digest = sha256_of(payload)
    if root is None:
        target = str(MediaResolver().cache_path(MediaRef(uri=f"media{extension}", sha256=digest)))
    else:
        target = storage.join(root, digest[:2], f"{digest}{extension}")
    if not storage.exists(target):
        storage.write_bytes(target, payload)
    return MediaRef(
        uri=target,
        sha256=digest,
        mime=mime,
        width=width,
        height=height,
        num_bytes=len(payload),
    )


def decode_rgb(ref: MediaRef, payload: bytes) -> Image:
    """The image in ``payload``, EXIF-oriented and in RGB (transparency composited onto white), as vLLM loads it.

    The one decoder of stored images: the judge's media preparation and the embedding encoders both read images
    through it, so a model sees the same pixels whichever path sends them.

    Raises:
        MediaError: ``payload`` is not a decodable image.
    """
    from PIL import Image, ImageOps

    try:
        with Image.open(io.BytesIO(payload)) as handle:
            image = ImageOps.exif_transpose(handle)
            image.load()
    except OSError as exc:
        raise MediaError(f"could not decode {ref.uri} as an image: {exc}") from exc
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[3])
        return canvas
    return image.convert("RGB")


def content_parts_payload(content: Content) -> list[dict[str, Any]]:
    """Lower *content* into the OpenAI content-parts shape used over HTTP.

    ``[{"type": "text", "text": ...}, {"type": "image_url", "image_url": {"url": ...}}]``
    -- what Cohere's ``/embed``, vLLM's ``/pooling`` and ``/rerank``, and every
    OpenAI-compatible chat endpoint accept, so one lowering serves all of them.

    Interleaving is preserved: a caption before its page is a different input from
    the same caption after it, and the order is information the model uses.

    Images are inlined as base64 data URLs rather than passed as URLs. The corpus
    lives in a private bucket, so a URL would either not resolve for the server or
    would be a signed link that expires -- making a re-run of the same job depend
    on when it ran.
    """
    resolver = default_resolver()
    parts: list[dict[str, Any]] = []
    for part in content.parts:
        if part.type == "text":
            if part.text:
                parts.append({"type": "text", "text": part.text})
            continue
        if isinstance(part, VideoPart) and not part.frames:
            raise MediaError(
                f"{part.ref.uri if part.ref else 'a video part'} is a video container; this lowering sends "
                "images only. Ingest the clip as a frame directory (the `frames` reader) to embed it."
            )
        for ref in part.frames if isinstance(part, VideoPart) else part.media_refs():
            encoded = base64.b64encode(resolver.bytes_of(ref)).decode("ascii")
            parts.append(
                {"type": "image_url", "image_url": {"url": f"data:{ref.mime or 'image/png'};base64,{encoded}"}}
            )
    # An empty parts list is rejected by every one of these endpoints, and a
    # document that is genuinely empty should be embedded as empty, not dropped.
    return parts or [{"type": "text", "text": ""}]


__all__ = [
    "IMAGE_MIME_BY_SUFFIX",
    "MEDIA_CACHE_DIRNAME",
    "MediaError",
    "MediaResolver",
    "VIDEO_MIME_BY_SUFFIX",
    "VideoHeader",
    "content_parts_payload",
    "decode_rgb",
    "default_resolver",
    "probe_video_header",
    "sha256_of",
    "store_media",
]
