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
import json
import os
import struct
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


def media_reference_fingerprint(ref: MediaRef) -> dict[str, Any]:
    """What names one media reference for an identity: its content hash, or its URI with the object's size
    and mtime/etag.

    A reference with ``sha256`` cannot change, so the hash *is* its identity. A reader that did not hash
    (``hash_media: false``, the default) leaves the URI alone; the object behind it can be replaced in place
    (a re-encoded page, a re-cut clip), and an identity keyed by the URI alone would silently reuse an index,
    a checkpoint or a media-cache entry computed over the old bytes. The object's own size and change stamp
    -- ``mtime`` for a local file, the backend's etag/generation for a remote object -- enter beside the URI,
    so changed bytes change the key.

    What it does not detect: a same-size edit that preserves the change stamp (an object restored from a
    backup with its mtime, a copy written with an unchanged mtime), and a ``data:`` URI (its bytes are the
    URI).  ``hash_media: true`` hashes the bytes at ingest and detects everything.

    The lookup costs one stat (local) or one metadata call (remote) per reference per identity computation;
    an unreachable or missing object contributes its URI alone (the reader reports the missing media).
    """
    if ref.sha256:
        return {"uri": ref.uri, "sha256": ref.sha256}
    fingerprint: dict[str, Any] = {"uri": ref.uri}
    if ref.num_bytes is not None:
        fingerprint["num_bytes"] = ref.num_bytes
    if not ref.uri.startswith("data:"):
        try:
            info = storage.info(ref.uri)
        except Exception:  # noqa: BLE001 - a missing or unreachable object is the reader's error to report
            info = None
        if info:
            size = info.get("size")
            if size is not None:
                fingerprint["size"] = int(size)
            stamp = _change_stamp(info)
            if stamp is not None:
                fingerprint["etag"] = stamp
    return fingerprint


def _change_stamp(info: dict[str, Any]) -> str | None:
    """The backend's change stamp for one object: its etag/generation when it has one, else its mtime."""
    for key in ("ETag", "etag", "generation", "Generation", "mtime", "LastModified", "last_modified"):
        value = info.get(key)
        if value is not None:
            return str(value)
    return None


def content_identity(content: Content) -> str:
    """One content as an identity reads it: a text-only document is its text; a document with media is its
    parts plus every media reference's :func:`media_reference_fingerprint`.

    The one form every media-bearing identity uses (the index's corpus hash, the rerank checkpoint key), so
    an unhashed media item's size and change stamp enter all of them together.
    """
    payload = content.model_dump_json()
    if not content.has_media:
        return payload
    fingerprints = [json.dumps(media_reference_fingerprint(ref), sort_keys=True) for ref in content.media]
    return "\0".join([payload, *fingerprints])


class MediaResolver:
    """Resolves media references to bytes or images, through a content-addressed cache in
    ``$RCP_NDCG_MEDIA_CACHE`` (default ``media/`` under the package cache)."""

    def __init__(self) -> None:
        self.cache_dir = _media_cache_dir()

    # -- cache layout ------------------------------------------------------
    def cache_path(self, ref: MediaRef) -> Path:
        """Where *ref* lives on disk once cached.

        Hashed refs shard by the first two hex characters, which keeps any one
        directory from growing to 10^5 entries. Unhashed refs key by
        :func:`media_reference_fingerprint` -- the URI with the object's size and change stamp -- so a
        replaced object lands on a new path instead of serving the old bytes.
        """
        if ref.sha256:
            digest = ref.sha256
            prefix = "by-hash"
        else:
            # An unhashed ref keys by its fingerprint (uri + size + mtime/etag): the URI alone cannot detect
            # the object changing underneath, which is why hashing at ingest matters.
            fingerprint = json.dumps(media_reference_fingerprint(ref), sort_keys=True)
            digest = sha256_of(fingerprint.encode("utf-8"))
            prefix = "by-uri"
        suffix = Path(ref.uri).suffix if len(Path(ref.uri).suffix) <= 6 else ""
        return self.cache_dir / prefix / digest[:2] / f"{digest}{suffix}"

    # -- resolution --------------------------------------------------------
    def bytes_of(self, ref: MediaRef) -> bytes:
        """The bytes behind *ref*, fetching and caching on first use.

        A ``data:`` URI carries its bytes inline (what the preparation inlines): decoded here, never
        fetched. Only base64 inlining is produced by this package and understood here -- any other ``data:``
        form is a :class:`~rcp_ndcg.errors.MediaError` naming it, not a confused "file not found" (the URI
        carries its bytes; there is nothing to find).
        """
        if ref.uri.startswith("data:"):
            header, _, payload = ref.uri.partition(",")
            if "base64" in header:
                return base64.b64decode(payload)
            raise MediaError(
                f"unsupported data URI {ref.uri[:64]}...: only base64 data URIs carry bytes this package can "
                "read inline",
                hint="inline the bytes as data:<mime>;base64,<payload> (the preparation's own form), or store "
                "the media at a resolvable URI",
            )
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
        storage.publish_bytes(target, payload)
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
        # An inline ``data:`` payload never goes down the fetch path: its bytes are decodable right here
        # (with or without a recorded hash), and ``local_path`` would call a missing file.
        payload = self.bytes_of(ref) if ref.sha256 or ref.uri.startswith("data:") else self._fetch(ref)
        digest = ref.sha256 or sha256_of(payload)
        update: dict[str, Any] = {"sha256": digest, "num_bytes": ref.num_bytes or len(payload)}
        header = probe_video_header(payload)
        if header is not None:
            # A container: its header states size, length and frame count, which is
            # what prices it and what a duration limit is checked against.
            recorded = {"width": ref.width, "height": ref.height, "num_frames": ref.num_frames}
            recorded |= {"duration_s": ref.duration_s, "fps": ref.fps}
            update |= {name: getattr(header, name) if value is None else value for name, value in recorded.items()}
        elif ref.width is None and ref.height is None:
            # A partial record (a width without a height) is never wiped by an unreadable probe: probing
            # fills both only when it can, and recorded dimensions stay recorded.
            width, height = _probe_dimensions(payload)
            if width is not None and height is not None:
                update["width"], update["height"] = width, height
        hydrated = ref.model_copy(update=update)
        if not ref.sha256:
            storage.publish_bytes(self.cache_path(hydrated), payload)
        return hydrated


def image_dimensions(payload: bytes) -> tuple[int | None, int | None]:
    """An image's ``(width, height)`` from its header, without decoding the pixels; ``(None, None)`` when the
    bytes are not a readable image (a non-image asset legitimately has no dimensions this way).

    The public form of what :meth:`MediaResolver.hydrate` probes: a reader that already holds the bytes records
    the dimensions without a second file read, so the media policy prices the page it actually has.
    """
    from PIL import Image as PILImage

    try:
        with PILImage.open(io.BytesIO(payload)) as handle:
            return handle.width, handle.height
    except OSError:
        return None, None


def _probe_dimensions(payload: bytes) -> tuple[int | None, int | None]:
    """Image dimensions from the header, without decoding the pixels."""
    return image_dimensions(payload)


DEFAULT_IMAGE_MIME = "image/png"
"""The MIME type an image with no recorded ``mime`` is sent under (the one home; the preparation and the
wire lowerings import it from here)."""

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
    """``moov/trak`` of the first video track: ``tkhd`` size, ``mdhd`` length, ``stsz`` count.

    Returns ``None`` for anything that is not one of these containers -- including a truncated-but
    box-structured payload (a download cut short): the field peeks are guarded, and a read past a short box
    body is "recorded unprobed", never a bare ``struct.error`` or ``IndexError``.
    """
    moov = _child(payload, 0, len(payload), b"moov")
    if moov is None:
        return None
    try:
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
    except (IndexError, struct.error):
        # A truncated-but-box-structured container is probed as far as it goes: the promise is "recorded
        # unprobed", never a traceback.
        return None
    return None


_DEFAULT_RESOLVER: MediaResolver | None = None


#: Image magic numbers: what a raw media cell's own bytes say the format is (a parquet media column may hold
#: plain binary rather than the ``{"bytes", "path"}`` struct ``datasets`` writes).
_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"BM", ".bmp"),
    (b"II*\x00", ".tif"),
    (b"MM\x00*", ".tif"),
)


def media_extension(payload: bytes) -> str | None:
    """The file suffix a media payload's own bytes name, or ``None`` when it is neither a known image nor a
    known video container.

    Magic numbers only -- never a decode -- so a raw-binary cell gets the extension (and with it the MIME
    type) its bytes state, instead of a guess from the column's name.
    """
    for magic, suffix in _IMAGE_MAGIC:
        if payload.startswith(magic):
            return suffix
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return ".webp"
    if probe_video_header(payload) is not None:
        return ".mp4" if payload[4:8] == b"ftyp" else ".avi"
    return None


def default_resolver() -> MediaResolver:
    """The process-wide resolver, so the cache is shared across call sites."""
    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is None:
        _DEFAULT_RESOLVER = MediaResolver()
    return _DEFAULT_RESOLVER


def store_media(
    payload: bytes,
    extension: str,
    *,
    root: str | None = None,
    width: int | None = None,
    height: int | None = None,
    mime: str | None = None,
) -> MediaRef:
    """Write ``payload`` once under its content hash and return a hashed :class:`MediaRef` to it.

    Args:
        payload: The encoded bytes (e.g. a PNG).
        extension: The file suffix, with its dot (``".png"``); :data:`IMAGE_MIME_BY_SUFFIX` gives its MIME type.
        root: The directory to write under (``<root>/<sha[:2]>/<sha><ext>``); ``None`` writes into the media
            cache in the resolver's own layout, so reading the reference back needs no second copy.
        width, height: The image's dimensions, when known.
        mime: The MIME type to record, when the suffix is not one :data:`IMAGE_MIME_BY_SUFFIX` knows (a video
            container, say: ``mime="video/mp4"``); the suffix still names the file. An image suffix and an
            explicit ``mime`` is recorded as given -- the caller owns the pairing.

    Raises:
        MediaError: Neither the suffix nor ``mime`` names a known media type.
    """
    resolved = mime or IMAGE_MIME_BY_SUFFIX.get(extension.lower())
    if resolved is None:
        raise MediaError(
            f"unknown media type {extension!r}; known image types: {sorted(IMAGE_MIME_BY_SUFFIX)} -- pass mime= "
            "for any other kind (a video container)"
        )
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
        mime=resolved,
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

    ``[{'type': 'text', 'text': ...}, {'type': 'image_url', 'image_url': {'url': ...}}]``
    -- what Cohere's ``/embed``, vLLM's ``/pooling`` and ``/rerank``, and every
    OpenAI-compatible chat endpoint accept, so one lowering serves all of them
    (the chat-style embeddings input of a vision-language embedder included, 2e).

    Interleaving is preserved: a caption before its page is a different input from
    the same caption after it, and the order is information the model uses.

    Images are inlined as base64 data URLs rather than passed as URLs. The corpus
    lives in a private bucket, so a URL would either not resolve for the server or
    would be a signed link that expires -- making a re-run of the same job depend
    on when it ran.

    A video part is lowered per its role's video policy -- the policy has already
    been applied when the request is prepared: sampled frames (``wire: frames``)
    go out as image parts, a container (``wire: video_url``) as a ``video_url``
    part for the engine to decode.
    """
    resolver = default_resolver()
    parts: list[dict[str, Any]] = []
    for part in content.parts:
        if part.type == "text":
            if part.text:
                parts.append({"type": "text", "text": part.text})
            continue
        if isinstance(part, VideoPart) and not part.frames:
            if part.ref is None:
                raise MediaError(
                    "a video part with neither frames nor a container cannot be lowered: ingest the clip as a "
                    "frame directory (the `frames` reader) to embed it"
                )
            encoded = base64.b64encode(resolver.bytes_of(part.ref)).decode("ascii")
            parts.append({"type": "video_url", "video_url": {"url": data_uri(part.ref.mime or "video/mp4", encoded)}})
            continue
        for ref in part.frames if isinstance(part, VideoPart) else part.media_refs():
            encoded = base64.b64encode(resolver.bytes_of(ref)).decode("ascii")
            parts.append({"type": "image_url", "image_url": {"url": data_uri(ref.mime or DEFAULT_IMAGE_MIME, encoded)}})
    # An empty parts list is rejected by every one of these endpoints, and a
    # document that is genuinely empty should be embedded as empty, not dropped.
    return parts or [{"type": "text", "text": ""}]


def data_uri(mime: str, base64_payload: str) -> str:
    """The inline ``data:`` URI of a base64 payload: the one form the package inlines media as -- the
    preparation, the wire lowerings and the judge's video blocks all build it here, so every data URI the
    code produces decodes the same way (:meth:`MediaResolver.bytes_of` reads it back)."""
    return f"data:{mime};base64,{base64_payload}"


__all__ = [
    "DEFAULT_IMAGE_MIME",
    "IMAGE_MIME_BY_SUFFIX",
    "data_uri",
    "MEDIA_CACHE_DIRNAME",
    "MediaError",
    "MediaResolver",
    "VIDEO_MIME_BY_SUFFIX",
    "VideoHeader",
    "content_identity",
    "content_parts_payload",
    "decode_rgb",
    "default_resolver",
    "image_dimensions",
    "media_extension",
    "media_reference_fingerprint",
    "probe_video_header",
    "sha256_of",
    "store_media",
]
