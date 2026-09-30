"""Client-side media preparation, checked against the reference processors (``tests/data/_media_reference.py``).

The central guarantee: an image the client prepared is a fixed point of the engine's own resize under every
engine's default settings, so a stock engine started without media flags keeps exactly what the client sent.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import pytest
from rcp_ndcg_core.content import Content, ImagePart, MediaRef, VideoPart

from rcp_ndcg.data import Preprocessing
from rcp_ndcg.data.prepare import MediaCensus, prepare_content, prepare_image
from rcp_ndcg.data.resolution import PROCESSORS, ImagePolicy, VideoPolicy, smart_resize, uniform_frame_indices
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.llm._payload import build_messages
from rcp_ndcg.llm.client import CompletionInput
from tests.data import _media_reference as ref

EDGES = [1, 7, 27, 28, 29, 31, 33, 100, 333, 480, 512, 640, 719, 1000, 1080, 1414, 1920, 2200, 3000, 4000, 7000]
SIZES = [(h, w) for h in EDGES for w in EDGES if max(h, w) / min(h, w) <= 150]

#: Budgets per family: the engines' whole default range, the paper-style page budget, and a narrow one.
BUDGETS = {
    "qwen2_vl": [(3136, 1003520), (4 * 28 * 28, 1280 * 28 * 28), (256 * 28 * 28, 256 * 28 * 28)],
    "qwen2_5_vl": [(3136, 12845056), (4 * 28 * 28, 1280 * 28 * 28), (256 * 28 * 28, 2048 * 28 * 28)],
    "qwen3_vl": [(65536, 16777216), (65536, 1280 * 32 * 32), (256 * 32 * 32, 256 * 32 * 32)],
}
POLICIES = [
    ImagePolicy(min_px=low, max_px=high, processor=family)
    for family, budgets in BUDGETS.items()
    for low, high in budgets
]


def _png(path: Path, size: tuple[int, int], mode: str = "RGB", color=(200, 30, 30)) -> MediaRef:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color).save(path)
    return MediaRef(uri=str(path), mime="image/png", width=size[0], height=size[1])


def _decoded(sent: MediaRef):
    from PIL import Image

    header, payload = sent.uri.split(",", 1)
    assert header == f"data:{sent.mime};base64"
    return Image.open(io.BytesIO(base64.b64decode(payload)))


def _target(policy: ImagePolicy, height: int, width: int) -> tuple[int, int] | None:
    try:
        return policy.target_size(height, width)
    except DataError:
        return None


class TestTheResizeIsTheReferenceResize:
    @pytest.mark.parametrize("policy", POLICIES, ids=lambda p: p.descriptor)
    def test_every_size_resizes_as_the_transformers_processor(self, policy: ImagePolicy):
        factor = PROCESSORS[policy.processor].factor
        for height, width in SIZES:
            expected = ref.hf_smart_resize(height, width, factor, policy.min_px, policy.max_px)
            assert smart_resize(height, width, factor=factor, min_pixels=policy.min_px, max_pixels=policy.max_px) == (
                expected
            ), f"diverged at {height}x{width}"

    @pytest.mark.parametrize(
        ("processor", "size", "expected"),
        [
            ("qwen3_vl", (1000, 1000), (992, 992)),
            ("qwen3_vl", (2200, 1700), (1280, 992)),
            ("qwen3_vl", (100, 700), (96, 704)),
            ("qwen2_vl", (1000, 1000), (980, 980)),
            ("qwen2_vl", (2200, 1700), (1120, 868)),
            ("qwen2_vl", (60, 40), (84, 56)),
        ],
    )
    def test_golden_sizes(self, processor: str, size: tuple[int, int], expected: tuple[int, int]):
        """Known inputs under the page budget of 1280 tokens (the oracle's own output, pinned)."""
        factor = PROCESSORS[processor].factor
        low = PROCESSORS[processor].min_pixels
        policy = ImagePolicy(min_px=low, max_px=1280 * factor * factor, processor=processor)
        assert policy.target_size(*size) == expected == ref.hf_smart_resize(*size, factor, low, 1280 * factor**2)


class TestNoServerFlagsNeeded:
    """What the client sends, the engine keeps: a fixed point of each engine's resize at its default settings."""

    @pytest.mark.parametrize("policy", POLICIES, ids=lambda p: p.descriptor)
    @pytest.mark.parametrize("engine", ["vllm", "sglang"])
    def test_the_prepared_size_is_a_no_op_under_the_engine_default(self, policy: ImagePolicy, engine: str):
        factor, low, high = ref.ENGINE_DEFAULTS[(engine, policy.processor)]
        checked = 0
        for height, width in SIZES:
            target = _target(policy, height, width)
            if target is None:
                continue
            assert ref.hf_smart_resize(*target, factor, low, high) == target, f"{engine} resizes {target} again"
            checked += 1
        assert checked > 0.95 * len(SIZES)

    @pytest.mark.parametrize("policy", [p for p in POLICIES if p.processor != "qwen3_vl"], ids=lambda p: p.descriptor)
    def test_and_under_sglangs_own_resize_at_its_image_max_pixels_default(self, policy: ImagePolicy):
        """SGLang's module-level ``smart_resize`` (factor 28, ``SGLANG_IMAGE_MAX_PIXELS`` default) keeps it too."""
        for height, width in SIZES:
            target = _target(policy, height, width)
            if target is not None:
                assert ref.sglang_smart_resize(*target) == target

    def test_a_size_the_engine_would_resize_again_is_refused(self):
        """Under a budget at the engines' floor, flooring to the factor can land below it: refused, not sent."""
        policy = ImagePolicy(min_px=65536, max_px=65536, processor="qwen3_vl")
        assert ref.hf_smart_resize(300, 1000, 32, 65536, 65536) == (128, 448)  # 57344 px, under the floor
        with pytest.raises(DataError, match="resize again"):
            policy.target_size(300, 1000)

    def test_a_size_whose_aspect_the_engine_refuses_is_refused(self):
        """A 1:150 strip floors to 28 pixels high, a 1:219 image, which the processor rejects outright."""
        policy = ImagePolicy(min_px=4 * 28 * 28, max_px=1280 * 28 * 28, processor="qwen2_vl")
        with pytest.raises(ValueError, match="aspect ratio"):
            ref.hf_smart_resize(*ref.hf_smart_resize(12000, 80, 28, 3136, 1003520), 28, 3136, 1003520)
        with pytest.raises(DataError, match="refuses"):
            policy.target_size(12000, 80)

    @pytest.mark.parametrize(
        ("processor", "budget"),
        [("qwen3_vl", (4 * 28 * 28, 1280 * 28 * 28)), ("qwen2_vl", (3136, 12845056)), ("qwen3_vl", (65536, 2**25))],
    )
    def test_a_budget_outside_the_engine_default_is_refused(self, processor: str, budget: tuple[int, int]):
        with pytest.raises(ValueError, match="resize the prepared image again"):
            ImagePolicy(min_px=budget[0], max_px=budget[1], processor=processor)
        with pytest.raises(ConfigError, match="resize the prepared image again"):
            ImagePolicy(min_px=budget[0], max_px=budget[1]).for_processor(processor)

    def test_a_declared_processor_must_match_the_judges(self):
        with pytest.raises(ConfigError, match="image_processor"):
            ImagePolicy(min_px=65536, max_px=65536, processor="qwen3_vl").for_processor("qwen2_5_vl")


class TestPreparedImages:
    def test_the_sent_image_is_a_png_at_the_target_size(self, tmp_path: Path):
        policy = ImagePolicy(min_px=65536, max_px=1280 * 32 * 32, processor="qwen3_vl")
        source = _png(tmp_path / "page.png", (1700, 2200))

        prepared = prepare_image(source, policy)

        image = _decoded(prepared.sent)
        assert (image.height, image.width) == policy.target_size(2200, 1700) == (1280, 992)
        assert image.format == "PNG" and image.mode == "RGB"
        assert prepared.resized and prepared.processor == "qwen3_vl"
        assert (prepared.sent.height, prepared.sent.width) == (1280, 992)

    def test_preparation_is_deterministic(self, tmp_path: Path):
        policy = ImagePolicy(min_px=65536, max_px=1280 * 32 * 32, processor="qwen3_vl")
        first = prepare_image(_png(tmp_path / "a.png", (1001, 777)), policy)
        second = prepare_image(_png(tmp_path / "b.png", (1001, 777)), policy)
        assert first.sent.sha256 == second.sent.sha256

    def test_an_image_already_at_its_target_is_not_resampled(self, tmp_path: Path):
        from PIL import Image

        policy = ImagePolicy(min_px=65536, max_px=1280 * 32 * 32, processor="qwen3_vl")
        path = tmp_path / "p.png"
        noise = Image.effect_noise((992, 1280), 64).convert("RGB")
        noise.save(path)

        prepared = prepare_image(MediaRef(uri=str(path), mime="image/png"), policy)

        assert not prepared.resized
        assert _decoded(prepared.sent).tobytes() == noise.tobytes()

    def test_transparency_is_composited_onto_white(self, tmp_path: Path):
        policy = ImagePolicy(min_px=65536, max_px=65536, processor="qwen3_vl")
        source = _png(tmp_path / "t.png", (256, 256), mode="RGBA", color=(0, 0, 0, 0))

        image = _decoded(prepare_image(source, policy).sent)

        assert image.mode == "RGB" and image.getpixel((10, 10)) == (255, 255, 255)

    def test_the_exif_orientation_is_applied(self, tmp_path: Path):
        from PIL import Image

        path = tmp_path / "rotated.jpg"
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 degrees clockwise to display
        Image.new("RGB", (640, 320), (0, 128, 0)).save(path, exif=exif)
        policy = ImagePolicy(min_px=65536, max_px=1280 * 32 * 32, processor="qwen3_vl")

        prepared = prepare_image(MediaRef(uri=str(path), mime="image/jpeg"), policy)

        assert (prepared.sent.height, prepared.sent.width) == policy.target_size(640, 320)

    @pytest.mark.parametrize("policy", [None, ImagePolicy.native(), ImagePolicy(min_px=65536, max_px=65536)], ids=str)
    def test_an_unknown_processor_or_no_budget_sends_the_stored_bytes(self, tmp_path: Path, policy):
        source = _png(tmp_path / "raw.png", (333, 719))

        prepared = prepare_image(source, policy)

        assert base64.b64decode(prepared.sent.uri.split(",", 1)[1]) == Path(source.uri).read_bytes()
        assert prepared.processor is None and not prepared.resized
        assert prepared.as_row(corpus="c", doc_id="d")["processor"] is None


class TestFrames:
    @pytest.mark.parametrize("total", [2, 3, 8, 9, 16, 31, 100, 301, 1000])
    @pytest.mark.parametrize("wanted", [1, 2, 4, 7, 8, 16, 32])
    def test_frame_indices_are_the_engines(self, total: int, wanted: int):
        ours = uniform_frame_indices(total, wanted)
        assert ours == ref.vllm_frame_indices(total, duration=total / 2.0, num_frames=wanted, fps=-1)
        assert ours == ref.sglang_frame_indices(total, min(wanted, total))

    def test_frames_are_sampled_and_each_prepared_as_an_image(self, tmp_path: Path):
        frames = [_png(tmp_path / "clip" / f"{i:03d}.png", (640, 360), color=(i, i, i)) for i in range(10)]
        image = ImagePolicy(min_px=65536, max_px=256 * 32 * 32, processor="qwen3_vl")
        content = Content.from_parts([VideoPart(frames=frames, frame_indices=list(range(10)))])

        prepared = prepare_content(content, image, VideoPolicy(num_frames=4, wire="frames"))

        (part,) = prepared.content.parts
        assert part.frame_indices == [0, 3, 6, 9]
        assert [item.kind for item in prepared.media] == ["frame"] * 4
        assert all((f.height, f.width) == image.target_size(360, 640) for f in part.frames)
        assert all(_decoded(f).getpixel((0, 0)) == (i, i, i) for f, i in zip(part.frames, [0, 3, 6, 9], strict=True))

    def test_a_video_url_opt_in_sends_the_container_unchanged(self, tmp_path: Path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00\x01container" * 8)
        container = MediaRef(uri=str(clip), mime="video/mp4", num_bytes=clip.stat().st_size, num_frames=48)
        content = Content.from_parts([VideoPart(ref=container)])
        image = ImagePolicy(min_px=65536, max_px=65536, processor="qwen3_vl")

        prepared = prepare_content(content, image, VideoPolicy(num_frames=8, wire="video_url"))

        (item,) = prepared.media
        assert item.kind == "video" and item.sent == container
        assert item.as_row(corpus="c", doc_id="d")["sampling"] == "engine"
        (message,) = build_messages(CompletionInput(user_prompt="x", user_content=prepared.content))
        (block,) = message["content"]
        assert block["type"] == "video_url"
        assert base64.b64decode(block["video_url"]["url"].split(",", 1)[1]) == clip.read_bytes()


class TestRecords:
    def test_the_census_records_each_item_once_and_survives_a_resume(self, tmp_path: Path):
        page = _png(tmp_path / "p.png", (1700, 2200))
        policy = ImagePolicy(min_px=65536, max_px=1280 * 32 * 32, processor="qwen3_vl")
        prepared = prepare_content(Content.from_parts([ImagePart(ref=page)]), policy, None)
        sink = tmp_path / "preprocessing.jsonl"

        census = MediaCensus(sink=sink)
        census.record(corpus="c", doc_id="d1", media=prepared.media)
        census.record(corpus="c", doc_id="d1", media=prepared.media)
        MediaCensus(sink=sink).record(corpus="c", doc_id="d1", media=prepared.media)

        (row,) = [json.loads(line) for line in sink.read_text().splitlines()]
        assert (row["source_width"], row["source_height"], row["sent_width"], row["sent_height"]) == (
            1700,
            2200,
            992,
            1280,
        )
        assert (row["processor"], row["resized"], row["sent_mime"]) == ("qwen3_vl", True, "image/png")

    def test_the_family_key_names_the_budget_and_the_processor(self):
        declared = ImagePolicy(min_px=65536, max_px=1280 * 32 * 32)
        keys = {
            Preprocessing(image=declared).key,
            Preprocessing(image=declared.for_processor("qwen3_vl")).key,
            Preprocessing(image=declared.for_processor("qwen2_5_vl")).key,
            Preprocessing(image=ImagePolicy(min_px=65536, max_px=256 * 32 * 32).for_processor("qwen3_vl")).key,
        }
        assert len(keys) == 4
