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
from rcp_ndcg_core.content import Content, MediaRef, VideoPart

from rcp_ndcg.data.io import get_reader
from rcp_ndcg.data.media import MediaError, MediaResolver, content_parts_payload, probe_video_header
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
    def test_a_container_is_refused_rather_than_sent_as_an_image(self, tmp_path: Path, video_clip):
        content = Content.from_parts([VideoPart(ref=MediaRef(uri=str(video_clip())))])

        with pytest.raises(MediaError, match="the `frames` reader"):
            content_parts_payload(content)


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
