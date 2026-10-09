"""Media resolution: content-addressed caching and verify-on-write."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from rcp_ndcg_core.content import Content, ImagePart, MediaRef

from rcp_ndcg.data.media import MediaError, MediaResolver, data_uri, default_resolver, sha256_of
from rcp_ndcg.errors import MissingInputError, classify

PIL = pytest.importorskip("PIL.Image")


@pytest.fixture
def png_bytes() -> bytes:
    buffer = io.BytesIO()
    PIL.new("RGB", (64, 32), color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def image_file(tmp_path, png_bytes) -> tuple[str, str]:
    path = tmp_path / "source" / "page_1.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(png_bytes)
    return str(path), sha256_of(png_bytes)


@pytest.fixture
def resolver(tmp_path, monkeypatch) -> MediaResolver:
    monkeypatch.setenv("RCP_NDCG_MEDIA_CACHE", str(tmp_path / "cache"))
    return MediaResolver()


@pytest.fixture
def offline_info(monkeypatch) -> None:
    """``storage.info`` raises as an unreachable backend does.

    An unhashed reference's fingerprint stats its object (one metadata call, by design); a test that names a
    ``gs://`` URI must not reach the network for it, and an unreachable object is the fallback's own case.
    """
    from rcp_ndcg import storage

    def refuse(uri):
        raise OSError(f"no backend for {uri} in this test")

    monkeypatch.setattr(storage, "info", refuse)


class TestCacheLayout:
    def test_hashed_refs_are_keyed_by_content(self, resolver):
        """Two URIs with the same content share one cache entry."""
        digest = "a" * 64
        first = resolver.cache_path(MediaRef(uri="gs://bucket/a.png", sha256=digest))
        second = resolver.cache_path(MediaRef(uri="s3://other/b.png", sha256=digest))
        assert first == second

    def test_hashed_refs_shard_by_prefix(self, resolver):
        digest = "ab" + "c" * 62
        assert resolver.cache_path(MediaRef(uri="x.png", sha256=digest)).parent.name == "ab"

    def test_unhashed_refs_are_keyed_by_uri(self, resolver, offline_info):
        first = resolver.cache_path(MediaRef(uri="gs://bucket/a.png"))
        second = resolver.cache_path(MediaRef(uri="gs://bucket/b.png"))
        assert first != second
        assert "by-uri" in str(first)

    def test_suffix_is_preserved_for_tooling(self, resolver):
        path = resolver.cache_path(MediaRef(uri="gs://bucket/page.png", sha256="a" * 64))
        assert path.suffix == ".png"

    def test_absurd_suffixes_are_dropped(self, resolver):
        path = resolver.cache_path(MediaRef(uri="gs://bucket/page.thisisnotanextension", sha256="a" * 64))
        assert path.suffix == ""


class TestResolution:
    def test_bytes_round_trip(self, resolver, image_file, png_bytes):
        uri, digest = image_file
        assert resolver.bytes_of(MediaRef(uri=uri, sha256=digest)) == png_bytes

    def test_first_read_populates_the_cache(self, resolver, image_file):
        uri, digest = image_file
        ref = MediaRef(uri=uri, sha256=digest)
        assert not resolver.cache_path(ref).exists()
        resolver.bytes_of(ref)
        assert resolver.cache_path(ref).exists()

    def test_cached_reads_survive_the_source_disappearing(self, resolver, image_file, tmp_path):
        """A cache hit must need nothing from the backend, network included."""
        uri, digest = image_file
        ref = MediaRef(uri=uri, sha256=digest)
        resolver.bytes_of(ref)
        (tmp_path / "source" / "page_1.png").unlink()
        assert len(resolver.bytes_of(ref)) > 0

    def test_image_decodes_to_rgb(self, resolver, image_file):
        uri, digest = image_file
        image = resolver.image(MediaRef(uri=uri, sha256=digest))
        assert image.mode == "RGB"
        assert image.size == (64, 32)

    def test_greyscale_is_converted(self, resolver, tmp_path):
        path = tmp_path / "grey.png"
        PIL.new("L", (8, 8), color=128).save(path)
        assert resolver.image(MediaRef(uri=str(path))).mode == "RGB"

    def test_missing_media_names_the_uri(self, resolver, tmp_path):
        with pytest.raises(MissingInputError, match="nope.png") as missing:
            resolver.bytes_of(MediaRef(uri=str(tmp_path / "nope.png")))
        assert classify(missing.value).exit_code == 4

    def test_images_of_content_preserves_order(self, resolver, tmp_path, png_bytes):
        refs = []
        for index in range(3):
            path = tmp_path / f"p{index}.png"
            PIL.new("RGB", (8 + index, 8)).save(path)
            refs.append(MediaRef(uri=str(path)))
        content = Content.from_parts([ImagePart(ref=ref) for ref in refs])
        assert [image.width for image in resolver.images_of(content)] == [8, 9, 10]


class TestVerification:
    def test_a_wrong_hash_is_refused_at_fetch(self, resolver, image_file):
        uri, _ = image_file
        with pytest.raises(MediaError, match="hash mismatch") as refused:
            resolver.bytes_of(MediaRef(uri=uri, sha256="b" * 64))
        assert classify(refused.value).exit_code == 12, "a refusal, not 'this is a bug'"

    def test_a_refused_fetch_is_not_cached(self, resolver, image_file):
        """Caching bytes under a hash they do not have poisons every consumer."""
        uri, _ = image_file
        bad = MediaRef(uri=uri, sha256="b" * 64)
        with pytest.raises(MediaError):
            resolver.bytes_of(bad)
        assert not resolver.cache_path(bad).exists()


class TestHydrate:
    def test_hydrate_fills_hash_size_and_dimensions(self, resolver, image_file, png_bytes):
        uri, digest = image_file
        hydrated = resolver.hydrate(MediaRef(uri=uri))
        assert hydrated.sha256 == digest
        assert hydrated.num_bytes == len(png_bytes)
        assert (hydrated.width, hydrated.height) == (64, 32)

    def test_hydrate_caches_under_the_discovered_hash(self, resolver, image_file):
        uri, _ = image_file
        hydrated = resolver.hydrate(MediaRef(uri=uri))
        assert resolver.cache_path(hydrated).exists()

    def test_hydrate_leaves_a_complete_ref_alone(self, resolver, image_file):
        uri, digest = image_file
        original = MediaRef(uri=uri, sha256=digest, width=64, height=32, num_bytes=1)
        assert resolver.hydrate(original) == original

    def test_hydrate_tolerates_a_non_image(self, resolver, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("not an image")
        hydrated = resolver.hydrate(MediaRef(uri=str(path)))
        assert hydrated.sha256 is not None
        assert hydrated.width is None


class TestWhereTheCacheLives:
    """Resolvable wherever this package is installed, not only in the checkout.

    The default used to come from the repository's ``[tool.rcp_ndcg] cache_dir``,
    which every consumer that installs this as a *dependency* -- the UI, an MCP
    server, a user's venv -- has no way to read: constructing a resolver there
    raised a ``KeyError`` about a config key the caller had never heard of.
    """

    def test_the_environment_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RCP_NDCG_MEDIA_CACHE", str(tmp_path / "elsewhere"))

        assert MediaResolver().cache_dir == tmp_path / "elsewhere"

    def test_a_tilde_is_expanded(self, monkeypatch):
        """A mounted volume is usually written as a path, and a literal ``~``
        directory next to the process is not what anyone meant."""
        monkeypatch.setenv("RCP_NDCG_MEDIA_CACHE", "~/media-cache")

        assert MediaResolver().cache_dir == Path.home() / "media-cache"

    def test_it_falls_back_to_the_user_cache(self, tmp_path, monkeypatch):
        """An installed package has no project to read a cache location from."""
        monkeypatch.delenv("RCP_NDCG_MEDIA_CACHE", raising=False)
        monkeypatch.delenv("RCP_NDCG_CACHE_DIR", raising=False)
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

        assert MediaResolver().cache_dir == tmp_path / "xdg" / "rcp-ndcg" / "media"


def test_stored_media_is_where_the_resolver_reads_it(resolver, png_bytes) -> None:
    """A released page image written into the cache must not be copied a second time when it is read."""
    from rcp_ndcg.data.media import store_media

    ref = store_media(png_bytes, ".png", width=64, height=32)

    assert Path(ref.uri) == resolver.cache_path(ref)
    assert resolver.bytes_of(ref) == png_bytes
    assert (ref.sha256, ref.mime, ref.num_bytes) == (sha256_of(png_bytes), "image/png", len(png_bytes))


def test_stored_media_is_recorded_under_its_registered_mime_type(resolver, png_bytes) -> None:
    """A '.jpg' cell was once recorded (and sent) as the unregistered 'image/jpg'."""
    from rcp_ndcg.data.media import store_media

    assert store_media(png_bytes, ".jpg").mime == "image/jpeg"
    with pytest.raises(MediaError, match="unknown media type"):
        store_media(png_bytes, ".xyz")


def test_the_resolver_decodes_as_the_engines_do(resolver, tmp_path) -> None:
    """One decoder for every path: transparency on white and the EXIF orientation, as the judge's preparation."""
    clear = tmp_path / "clear.png"
    PIL.new("RGBA", (4, 4), (0, 0, 0, 0)).save(clear)
    rotated = tmp_path / "rotated.jpg"
    exif = PIL.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise to display
    PIL.new("RGB", (40, 20)).save(rotated, exif=exif)

    assert resolver.image(MediaRef(uri=str(clear))).getpixel((0, 0)) == (255, 255, 255)
    assert resolver.image(MediaRef(uri=str(rotated))).size == (20, 40)


class TestDataUris:
    """One home for inline media (RFC sweep F4): the package produces data URIs with
    ``data_uri`` and reads them back with ``bytes_of`` -- and a data URI it cannot read is a typed media
    error, never a confused ``media not found``."""

    def test_data_uri_round_trips_through_bytes_of(self) -> None:
        import base64

        from rcp_ndcg.data.media import data_uri

        payload = b"inline-bytes"
        ref = MediaRef(uri=data_uri("image/png", base64.b64encode(payload).decode("ascii")))
        assert default_resolver().bytes_of(ref) == payload

    def test_a_non_base64_data_uri_is_a_media_error(self) -> None:
        ref = MediaRef(uri="data:text/plain,hello")
        with pytest.raises(MediaError, match="base64"):
            default_resolver().bytes_of(ref)

    def test_hydrate_inlines_a_data_uri_without_a_recorded_hash(self) -> None:
        """A data URI takes the bytes_of path unconditionally (its bytes are inline; there is nothing to
        fetch), with or without a recorded ``sha256``."""
        import base64

        payload = b"png-ish-bytes"
        ref = MediaRef(uri=data_uri("application/octet-stream", base64.b64encode(payload).decode("ascii")))
        hydrated = default_resolver().hydrate(ref)
        assert hydrated.sha256 and hydrated.num_bytes == len(payload)

    def test_hydrate_never_wipes_a_partial_dimension_record(self) -> None:
        """A ref with a recorded width keeps it when the probe cannot read the payload: an unreadable
        container never degrades an exact record into an unknown one."""
        payload = b"not-a-decodable-container"
        ref = MediaRef(
            uri=data_uri("application/octet-stream", __import__("base64").b64encode(payload).decode("ascii")),
            mime="video/x-unknown",
            width=1920,
            height=None,
        )
        hydrated = default_resolver().hydrate(ref)
        assert hydrated.width == 1920, "the recorded dimension stays recorded"

    def test_hydrate_keeps_a_recorded_width_beside_a_readable_payload(self) -> None:
        """A recorded dimension is a record: the probe fills dimensions only when none is recorded, so a
        width-only ref keeps its width even when the payload decodes to another size."""
        import base64
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (64, 48)).save(buffer, format="PNG")
        ref = MediaRef(
            uri=data_uri("image/png", base64.b64encode(buffer.getvalue()).decode("ascii")),
            mime="image/png",
            width=1920,
            height=None,
        )
        assert default_resolver().hydrate(ref).width == 1920


class TestTruncatedContainerHeaders:
    """A truncated-but-box-structured MP4/MOV is probed as far as it goes and recorded unprobed: never a
    bare ``struct.error`` or ``IndexError`` from the field peeks."""

    def test_a_truncated_isobmff_header_returns_none(self, tmp_path: Path) -> None:
        from rcp_ndcg.data.media import probe_video_header
        from tests.conftest import write_mp4_header

        path = write_mp4_header(tmp_path / "a.mp4", frames=30, timescale=1000, duration=2500, size=(320, 240))
        payload = path.read_bytes()
        truncated = bytes(payload[: len(payload) // 2])
        assert probe_video_header(truncated) is None
        assert probe_video_header(payload) is not None, "the intact header still probes"

    @pytest.mark.parametrize("cut", [0, 4])
    def test_a_box_whose_fields_were_cut_returns_none(self, cut: int) -> None:
        """The boxes are intact -- every size field is true -- but the video track's last box (``mdhd``)
        ends before the fields the probe peeks (a download cut inside the last box): the probe returns
        ``None``, where an unguarded peek raised ``IndexError`` (no version byte) or ``struct.error``."""
        import struct

        from rcp_ndcg.data.media import probe_video_header

        def box(kind: bytes, body: bytes) -> bytes:
            return struct.pack(">I", 8 + len(body)) + kind + body

        tkhd = box(b"tkhd", b"\0\0\0\x03" + b"\0" * 72 + struct.pack(">II", 320 << 16, 240 << 16))
        hdlr = box(b"hdlr", b"\0" * 8 + b"vide" + b"\0" * 12 + b"video\0")
        mdhd = box(b"mdhd", b"\0" * cut)  # the version byte and the timescale fields are gone
        payload = box(b"ftyp", b"isom\0\0\0\0isom") + box(b"moov", box(b"trak", tkhd + box(b"mdia", hdlr + mdhd)))

        assert probe_video_header(payload) is None


class TestAnUnhashedReferenceIsKeyedByItsObject:
    """A6: with ``hash_media: false`` the URI alone cannot detect the object changing, so the cache key
    records the object's size and change stamp beside it (the identity does too, through
    ``media_reference_fingerprint``)."""

    def test_the_cache_path_moves_when_the_object_changes(self, resolver, tmp_path) -> None:
        page = tmp_path / "page.png"
        page.write_bytes(b"first")
        ref = MediaRef(uri=str(page), mime="image/png")
        before = resolver.cache_path(ref)

        page.write_bytes(b"other")  # the same length: only the bytes (and the mtime) differ

        assert resolver.cache_path(ref) != before

    def test_a_replaced_object_is_refetched(self, resolver, tmp_path) -> None:
        page = tmp_path / "page.png"
        page.write_bytes(b"first")
        ref = MediaRef(uri=str(page), mime="image/png")
        assert resolver.bytes_of(ref) == b"first"

        page.write_bytes(b"second")

        assert resolver.bytes_of(ref) == b"second", "the replaced bytes are fetched, never the stale cache entry"

    def test_an_unreachable_object_keeps_its_uri_only(self, resolver, offline_info) -> None:
        """A URI that cannot be stat'ed still keys by itself (the reader reports the missing media)."""
        first = resolver.cache_path(MediaRef(uri="gs://YOUR-BUCKET/a.png"))
        second = resolver.cache_path(MediaRef(uri="gs://YOUR-BUCKET/b.png"))

        assert first != second
