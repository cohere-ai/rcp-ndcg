"""Video containers at ingest: header probing, hydration, and the two video readers.

Every clip is synthesised in-test (``tests/conftest.py``): a Motion-JPEG AVI that
real decoders read, and a header-only MP4 that pins the ISO BMFF parser. Where
``ffmpeg`` happens to be installed, the probe is also checked against files
ffmpeg itself wrote.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from rcp_ndcg_core.content import Content, ImagePart, MediaRef, TextPart, VideoPart

from rcp_ndcg.data.io import get_reader
from rcp_ndcg.data.media import MediaResolver, content_parts_payload, probe_video_header
from rcp_ndcg.errors import DataError
from tests.conftest import write_mp4_header


class TestProbe:
    def test_an_avi_header_states_size_length_and_frames(self, video_clip):
        header = probe_video_header(video_clip(frames=12, fps=4, size=(64, 48)).read_bytes())

        assert header == (64, 48, 12, 3.0, 4.0)

    def test_an_mp4_header_states_size_length_and_frames(self, tmp_path: Path):
        path = write_mp4_header(tmp_path / "a.mp4", frames=30, timescale=1000, duration=2500, size=(320, 240))

        assert probe_video_header(path.read_bytes()) == (320, 240, 30, 2.5, 12.0)

    def test_an_image_is_not_a_video(self, tmp_path: Path):
        from PIL import Image

        path = tmp_path / "p.png"
        Image.new("RGB", (4, 4)).save(path)

        assert probe_video_header(path.read_bytes()) is None

    @pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="needs ffmpeg")
    @pytest.mark.parametrize("suffix", [".mp4", ".mov"])
    def test_the_probe_agrees_with_ffprobe_on_a_real_encode(self, tmp_path: Path, video_clip, suffix: str):
        """The fixture decodes, and the parser reads what ffmpeg writes."""
        target = tmp_path / f"re{suffix}"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(video_clip(frames=12, fps=4)), "-c:v", "mpeg4", str(target)],
            check=True,
        )
        probed = subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-of", "json"]
            + ["-show_entries", "stream=width,height,nb_read_frames,duration", str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        (stream,) = json.loads(probed.stdout)["streams"]

        header = probe_video_header(target.read_bytes())

        assert (header.width, header.height, header.num_frames) == (
            stream["width"],
            stream["height"],
            int(stream["nb_read_frames"]),
        )
        assert header.duration_s == pytest.approx(float(stream["duration"]), abs=1e-3)


class TestHydration:
    def test_hydrating_a_container_records_its_header(self, tmp_path: Path, video_clip):
        ref = MediaRef(uri=str(video_clip(frames=8, fps=2, size=(32, 16))), mime="video/x-msvideo")

        hydrated = MediaResolver().hydrate(ref)

        assert hydrated.sha256 is not None
        assert (hydrated.width, hydrated.height, hydrated.num_frames) == (32, 16, 8)
        assert (hydrated.duration_s, hydrated.fps) == (4.0, 2.0)

    def test_a_recorded_field_wins_over_the_header(self, tmp_path: Path, video_clip):
        ref = MediaRef(uri=str(video_clip(frames=8)), num_frames=99)

        assert MediaResolver().hydrate(ref).num_frames == 99


class TestTheEmbeddingLowering:
    def test_a_video_url_wire_lowers_the_container_to_a_video_url_part(self, tmp_path: Path, video_clip):
        """2e: the lowering emits video -- a container (the role's ``video_policy: video_url`` wire leaves
        one) goes out as a ``video_url`` part, which the chat-style embeddings and pooling inputs take."""
        clip = video_clip()
        content = Content.from_parts([VideoPart(ref=MediaRef(uri=str(clip), mime="video/mp4"))])

        parts = content_parts_payload(content)

        assert len(parts) == 1
        assert parts[0]["type"] == "video_url"
        assert parts[0]["video_url"]["url"].startswith("data:video/mp4;base64,")

    def test_sampled_frames_lower_as_image_parts(self, tmp_path: Path):
        """The role's ``wire: frames`` policy leaves sampled frames; they lower as image parts, as today."""
        from PIL import Image

        frames = tmp_path / "frames"
        frames.mkdir()
        refs = []
        for index in range(2):
            path = frames / f"frame_{index}.png"
            Image.new("RGB", (8, 8), (index, 0, 0)).save(path, format="PNG")
            refs.append(MediaRef(uri=str(path)))
        content = Content.from_parts([VideoPart(frames=refs, frame_indices=[0, 1])])

        parts = content_parts_payload(content)

        assert [one["type"] for one in parts] == ["image_url", "image_url"]

    def test_the_declared_mechanisms_drop_empty_text_and_prefer_frames_over_ref(self, tmp_path: Path, video_clip):
        """The one lowering's declared mechanisms: an empty text part lowers to nothing, and a video part
        carrying both frames and a container lowers to its frames (the sampling the policy chose)."""
        from PIL import Image

        frame = tmp_path / "frame.png"
        Image.new("RGB", (8, 8), (1, 2, 3)).save(frame, format="PNG")
        content = Content.from_parts(
            [
                TextPart(text=""),
                VideoPart(ref=MediaRef(uri=str(video_clip()), mime="video/mp4"), frames=[MediaRef(uri=str(frame))]),
            ]
        )

        parts = content_parts_payload(content)

        assert [one["type"] for one in parts] == ["image_url"], "the empty text dropped, the frames over the ref"

    def test_the_judge_s_guards_are_hooks_and_change_no_block(self, tmp_path: Path):
        """A guard runs for every media item it covers and returns nothing: the blocks are the one
        lowering's, with and without guards."""
        from PIL import Image

        path = tmp_path / "page.png"
        Image.new("RGB", (8, 8), (1, 2, 3)).save(path, format="PNG")
        content = Content.from_parts([TextPart(text="a caption"), ImagePart(ref=MediaRef(uri=str(path)))])
        guarded: list[MediaRef] = []

        parts = content_parts_payload(content, image_guard=guarded.append)

        assert guarded == content.media and parts == content_parts_payload(content)

    def test_a_container_of_an_unknown_kind_is_refused_by_name(self, tmp_path: Path):
        """The one lowering resolves a container's mime (never a blind ``video/mp4``): the judge's rule is
        the served roles' too, so an unknown container is refused instead of sent as a wrong kind."""
        clip = tmp_path / "clip.xyz"
        clip.write_bytes(b"not a container")
        content = Content.from_parts([VideoPart(ref=MediaRef(uri=str(clip), num_bytes=15))])

        with pytest.raises(DataError, match="video container"):
            content_parts_payload(content)

    def test_an_inline_container_without_a_size_lowers_on_the_served_path(self) -> None:
        """The lowering computes a container's byte size only when a guard wants it: an already-inlined
        ``data:`` container (no file behind it, no recorded size) lowers on the served path, and the
        judge's cap reads its decoded size."""
        import base64

        payload = base64.b64encode(b"\x00\x00\x00\x18ftypmp42-inline-clip").decode("ascii")
        uri = f"data:video/mp4;base64,{payload}"
        content = Content.from_parts([VideoPart(ref=MediaRef(uri=uri, mime="video/mp4"))])

        (part,) = content_parts_payload(content)

        assert part == {"type": "video_url", "video_url": {"url": uri}}
        sizes: list[int] = []
        content_parts_payload(content, video_guard=lambda _ref, size: sizes.append(size))
        assert sizes == [len(base64.b64decode(payload))], "the guard sees the decoded byte size"


@pytest.fixture
def clip_dir(tmp_path: Path) -> Path:
    from tests.conftest import write_mjpeg_avi

    root = tmp_path / "clips"
    write_mjpeg_avi(root / "cooking" / "knot.avi", frames=6)
    write_mjpeg_avi(root / "sports" / "knot.avi", frames=10)
    (root / "notes.txt").write_text("not a clip")
    return root


@pytest.fixture
def frame_root(tmp_path: Path) -> Path:
    from PIL import Image

    root = tmp_path / "frames"
    for clip, count in (("movie_a/clip_1", 12), ("movie_a/clip_2", 3)):
        (root / clip).mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (8, 8), (index, 0, 0)).save(root / clip / f"frame_{index}.jpg")
    return root


class TestVideoDir:
    def test_ids_are_relative_paths_so_clips_do_not_collide(self, clip_dir: Path):
        docs = list(get_reader("videos", uri=str(clip_dir)).documents())

        assert [doc.id for doc in docs] == ["cooking/knot", "sports/knot"]
        assert all(isinstance(doc.content.parts[0], VideoPart) for doc in docs)

    def test_hash_media_records_each_clips_frame_count(self, clip_dir: Path):
        docs = list(get_reader("videos", uri=str(clip_dir), hash_media=True).documents())

        assert [doc.media[0].num_frames for doc in docs] == [6, 10]
        assert all(doc.media[0].mime == "video/x-msvideo" for doc in docs)


class TestFrameDir:
    def test_each_clip_directory_is_one_document(self, frame_root: Path):
        docs = {doc.id: doc for doc in get_reader("frames", uri=str(frame_root)).documents()}

        assert sorted(docs) == ["movie_a/clip_1", "movie_a/clip_2"]
        part = docs["movie_a/clip_1"].content.parts[0]
        assert isinstance(part, VideoPart) and part.ref is None
        assert part.frame_indices == list(range(12))

    def test_frames_are_in_numeric_order(self, frame_root: Path):
        doc = next(iter(get_reader("frames", uri=str(frame_root)).documents()))

        names = [Path(ref.uri).name for ref in doc.media]
        assert names[:3] == ["frame_0.jpg", "frame_1.jpg", "frame_2.jpg"]
        assert names[-1] == "frame_11.jpg"

    def test_frames_loose_in_the_root_are_refused(self, frame_root: Path):
        from PIL import Image

        Image.new("RGB", (8, 8)).save(frame_root / "stray.jpg")

        with pytest.raises(DataError, match="own directory"):
            list(get_reader("frames", uri=str(frame_root)).documents())


class TestIngest:
    def test_a_clip_directory_ingests_with_its_header_recorded(self, clip_dir: Path, tmp_path: Path):
        from click.testing import CliRunner

        from rcp_ndcg.cli.data import data_group

        out = tmp_path / "ingested"
        result = CliRunner().invoke(
            data_group,
            ["convert", "--format", "videos", "--source", str(clip_dir), "--out", str(out)]
            + ["--set", "hash_media=true", "--shape", "corpus"],
        )

        assert result.exit_code == 0, result.output
        written = "".join(path.read_text() for path in out.rglob("*.jsonl"))
        assert '"num_frames":6' in written.replace(" ", "")
