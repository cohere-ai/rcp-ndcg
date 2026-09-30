"""The image and video policies: geometry, token accounting, frame sampling, and identity."""

from __future__ import annotations

import pytest
from rcp_ndcg_core.content import Content, ImagePart, MediaRef, TextPart, VideoPart

from rcp_ndcg.data import Preprocessing
from rcp_ndcg.data.resolution import (
    PROCESSORS,
    ImagePolicy,
    VideoPolicy,
    VideoPolicyError,
    content_media_tokens,
    sample_video_part,
    smart_resize,
    uniform_frame_indices,
)
from rcp_ndcg.errors import ConfigError, DataError, classify

QWEN = ImagePolicy(min_px=4 * 28 * 28, max_px=1280 * 28 * 28, processor="qwen2_vl")


def _qwen(max_px: int) -> ImagePolicy:
    return ImagePolicy(min_px=4 * 28 * 28, max_px=max_px, processor="qwen2_5_vl")


def _video(num_frames: int, wire: str = "frames", **kwargs) -> VideoPolicy:
    return VideoPolicy(num_frames=num_frames, wire=wire, **kwargs)


class TestSmartResize:
    def test_snaps_both_sides_to_the_factor(self):
        height, width = smart_resize(1000, 1000, factor=28, min_pixels=56 * 56, max_pixels=4096 * 28 * 28)
        assert height % 28 == 0
        assert width % 28 == 0

    def test_downscales_to_fit_the_budget(self):
        max_pixels = 256 * 28 * 28
        height, width = smart_resize(4000, 3000, factor=28, min_pixels=56 * 56, max_pixels=max_pixels)
        assert height * width <= max_pixels

    def test_upscales_to_meet_the_floor(self):
        min_pixels = 256 * 28 * 28
        height, width = smart_resize(60, 40, factor=28, min_pixels=min_pixels, max_pixels=4096 * 28 * 28)
        assert height * width >= min_pixels

    def test_preserves_aspect_ratio_approximately(self):
        height, width = smart_resize(1600, 800, factor=28, min_pixels=56 * 56, max_pixels=1280 * 28 * 28)
        assert 1.8 < height / width < 2.2

    def test_rejects_extreme_aspect_ratios(self):
        with pytest.raises(ValueError, match="aspect ratio"):
            smart_resize(10, 5000, factor=28, min_pixels=56 * 56, max_pixels=1280 * 28 * 28)

    def test_rejects_degenerate_dimensions(self):
        with pytest.raises(ValueError, match="positive"):
            smart_resize(0, 100, factor=28, min_pixels=56 * 56, max_pixels=1280 * 28 * 28)

    def test_matches_the_transformers_implementation(self):
        """Our port must agree with the processor, or token estimates are fiction."""
        transformers_qwen = pytest.importorskip("transformers.models.qwen2_vl.image_processing_qwen2_vl")
        for height, width in [(1000, 1000), (4000, 3000), (100, 700), (28, 28), (1568, 1104)]:
            expected = transformers_qwen.smart_resize(
                height, width, factor=28, min_pixels=56 * 56, max_pixels=1280 * 28 * 28
            )
            actual = smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=1280 * 28 * 28)
            assert actual == expected, f"diverged at {height}x{width}"


class TestPolicyValidation:
    def test_a_pixel_budget_needs_both_bounds(self):
        with pytest.raises(ValueError, match="both `min_px` and `max_px`"):
            ImagePolicy(max_px=1000)

    def test_a_pixel_budget_rejects_inverted_bounds(self):
        with pytest.raises(ValueError, match="exceeds max_px"):
            ImagePolicy(min_px=1000, max_px=100)

    def test_native_needs_nothing(self):
        assert ImagePolicy.native().is_native and not QWEN.is_native


class TestTokenAccounting:
    def test_pixel_budget_tokens_are_the_merged_patch_count(self):
        height, width = QWEN.target_size(1000, 1000)
        factor = PROCESSORS["qwen2_vl"].factor
        assert QWEN.image_tokens(1000, 1000) == (height // factor) * (width // factor)

    def test_a_bigger_budget_costs_more_tokens(self):
        small = _qwen(max_px=256 * 28 * 28)
        large = _qwen(max_px=2048 * 28 * 28)
        assert small.image_tokens(3000, 2000) < large.image_tokens(3000, 2000)

    def test_native_refuses_to_guess(self):
        with pytest.raises(ConfigError, match="native"):
            ImagePolicy.native().image_tokens(1000, 1000)

    def test_an_unknown_processor_refuses_to_guess(self):
        with pytest.raises(ConfigError, match="image_processor"):
            ImagePolicy(min_px=4 * 28 * 28, max_px=1280 * 28 * 28).image_tokens(1000, 1000)


class TestMaxImageTokens:
    """The bound a cost preflight uses when an image's size was never recorded."""

    @pytest.mark.parametrize("policy", [QWEN, _qwen(max_px=256 * 28 * 28)])
    @pytest.mark.parametrize("size", [(100, 100), (4000, 3000), (8000, 80), (80, 8000)])
    def test_no_image_can_exceed_the_bound(self, policy: ImagePolicy, size: tuple[int, int]):
        """A bound, not a guess -- the policy caps the geometry by construction."""
        assert policy.image_tokens(*size) <= policy.max_image_tokens

    def test_the_bound_is_tight(self):
        """Loose enough to be useless would be no better than a guess.

        It is not *attained*: ``smart_resize`` rounds each side down to a multiple
        of the patch factor, so a square image lands a little under the budget it
        is allowed to fill. Within a few percent is what a preflight needs.
        """
        assert QWEN.image_tokens(8000, 8000) > 0.95 * QWEN.max_image_tokens

    def test_native_refuses_to_bound(self):
        with pytest.raises(ConfigError, match="native"):
            _ = ImagePolicy.native().max_image_tokens


class TestContentMediaTokens:
    def _page(self, **kwargs) -> Content:
        return Content.from_image("gs://p/1.png", **kwargs)

    def test_text_only_content_costs_nothing(self):
        assert content_media_tokens(Content.from_text("hello"), QWEN) == (0, 0)

    def test_a_sized_image_prices_exactly(self):
        policy = QWEN
        count = content_media_tokens(self._page(width=1700, height=2200), policy)
        assert count == (policy.image_tokens(2200, 1700), 0)

    def test_an_unsized_image_is_bounded_and_counted(self):
        policy = QWEN
        assert content_media_tokens(self._page(), policy) == (policy.max_image_tokens, 1)

    def test_parts_add_up(self):
        policy = QWEN
        content = Content.from_parts(
            [
                TextPart(text="caption"),
                ImagePart(ref=MediaRef(uri="gs://p/1.png", width=1000, height=1000)),
                ImagePart(ref=MediaRef(uri="gs://p/2.png", width=1000, height=1000)),
            ]
        )
        assert content_media_tokens(content, policy).tokens == 2 * policy.image_tokens(1000, 1000)

    def test_a_frame_directory_is_priced_at_the_frame_budget(self):
        """300 extracted frames priced as 300 images over-states the cost by an
        order of magnitude; the model only sees the sampled ones."""
        frames = [MediaRef(uri=f"gs://v/f{index}.png", width=640, height=480) for index in range(300)]
        content = Content.from_parts([VideoPart(ref=None, frames=frames)])

        assert content_media_tokens(content, QWEN, _video(4)).tokens == 4 * QWEN.image_tokens(480, 640)

    def test_without_a_frame_budget_every_frame_counts(self):
        """The honest reading of "no sampling policy declared"."""
        policy = QWEN
        frames = [MediaRef(uri=f"gs://v/f{index}.png", width=640, height=480) for index in range(3)]
        content = Content.from_parts([VideoPart(ref=None, frames=frames)])

        assert content_media_tokens(content, policy).tokens == 3 * policy.image_tokens(480, 640)

    def test_a_probed_container_prices_exactly(self):
        clip = MediaRef(uri="gs://v/a.mp4", width=1280, height=720, num_frames=300)

        assert content_media_tokens(Content.from_parts([VideoPart(ref=clip)]), QWEN, _video(8, "video_url")) == (
            8 * QWEN.image_tokens(720, 1280),
            0,
        )

    def test_an_unsized_container_is_bounded_by_the_frame_budget_not_one_image(self):
        """Priced as one image-sized ref, a clip understates its cost num_frames-fold."""
        clip = MediaRef(uri="gs://v/a.mp4", num_frames=300)

        count = content_media_tokens(Content.from_parts([VideoPart(ref=clip)]), QWEN, _video(8, "video_url"))

        assert count == (8 * QWEN.max_image_tokens, 1)

    def test_a_container_without_a_frame_policy_is_refused(self):
        """The engine's default sampling would decide the cost; nothing here knows it."""
        with pytest.raises(ConfigError, match="video policy"):
            content_media_tokens(Content.from_parts([VideoPart(ref=MediaRef(uri="a.mp4"))]), QWEN)


class TestUniformSampling:
    """The rule both engines apply to a container, applied here to frame directories."""

    def test_eight_of_three_hundred_are_spread_not_a_prefix(self):
        assert uniform_frame_indices(300, 8) == [0, 42, 85, 128, 170, 213, 256, 299]

    def test_it_is_the_engines_linspace(self):
        import numpy as np

        for total, wanted in [(300, 8), (31, 4), (10, 3), (7, 6), (1000, 32)]:
            expected = np.linspace(0, total - 1, wanted, dtype=np.int64).tolist()
            assert uniform_frame_indices(total, wanted) == expected

    @pytest.mark.parametrize("total, wanted", [(3, 8), (5, 5), (1, 1), (1, 4)])
    def test_the_sampler_never_pads(self, total: int, wanted: int):
        assert uniform_frame_indices(total, wanted) == list(range(total))

    def test_one_frame_is_the_first(self):
        assert uniform_frame_indices(300, 1) == [0]

    def test_sampling_keeps_the_source_frame_indices(self):
        frames = [MediaRef(uri=f"f{index}.jpg") for index in range(10)]
        part = VideoPart(frames=frames, frame_indices=list(range(100, 110)))

        shown = sample_video_part(part, _video(3))

        assert shown.frame_indices == [100, 104, 109]
        assert [ref.uri for ref in shown.frames] == ["f0.jpg", "f4.jpg", "f9.jpg"]

    def test_misaligned_frame_indices_are_refused(self):
        part = VideoPart(frames=[MediaRef(uri="f0.jpg")], frame_indices=[0, 1])

        with pytest.raises(DataError, match="align"):
            sample_video_part(part, _video(2))


class TestDurationLimit:
    def test_a_clip_over_the_limit_is_refused(self):
        clip = VideoPart(ref=MediaRef(uri="long.mp4", duration_s=600.0, num_frames=18000))

        with pytest.raises(VideoPolicyError, match="600.0s exceeds") as refused:
            sample_video_part(clip, _video(8, "video_url", max_duration_s=60))
        assert classify(refused.value).exit_code == 12, "a refusal, not 'this is a bug'"

    def test_a_clip_of_unknown_length_is_refused(self):
        with pytest.raises(VideoPolicyError, match="never recorded"):
            clip = VideoPart(ref=MediaRef(uri="a.mp4", num_frames=300))
            sample_video_part(clip, _video(8, "video_url", max_duration_s=60))

    def test_a_clip_within_the_limit_is_admitted_unchanged(self):
        clip = VideoPart(ref=MediaRef(uri="short.mp4", duration_s=12.5, num_frames=300))

        assert sample_video_part(clip, _video(8, "video_url", max_duration_s=60)) == clip


class TestShortClips:
    """A clip that cannot supply the frame budget is refused, never shown whole."""

    def test_a_short_frame_directory_is_refused(self):
        part = VideoPart(frames=[MediaRef(uri=f"clip/f{i}.jpg") for i in range(3)])

        with pytest.raises(VideoPolicyError, match="3 frames, fewer than"):
            sample_video_part(part, _video(8))

    def test_a_short_container_is_refused(self):
        with pytest.raises(VideoPolicyError, match="3 frames, fewer than"):
            sample_video_part(VideoPart(ref=MediaRef(uri="a.mp4", num_frames=3)), _video(8, "video_url"))

    def test_a_container_of_unknown_length_is_refused(self):
        with pytest.raises(VideoPolicyError, match="hash_media"):
            sample_video_part(VideoPart(ref=MediaRef(uri="a.mkv")), _video(8, "video_url"))

    @pytest.mark.parametrize("frames", [8, 9])
    def test_exactly_enough_frames_is_admitted(self, frames: int):
        part = VideoPart(frames=[MediaRef(uri=f"clip/f{i}.jpg") for i in range(frames)])

        assert len(sample_video_part(part, _video(8)).frames) == 8

    def test_the_token_count_refuses_too(self):
        """Pricing goes through the same check, so a preflight fails where the run would."""
        content = Content.from_parts([VideoPart(ref=MediaRef(uri="a.mp4", num_frames=3))])

        with pytest.raises(VideoPolicyError):
            content_media_tokens(content, QWEN, _video(8, "video_url"))


class TestTheDeclaredWire:
    """Frames-as-images and a decoded container are different instruments."""

    def test_a_container_under_the_frames_wire_is_refused(self):
        with pytest.raises(VideoPolicyError, match="wire: video_url"):
            sample_video_part(VideoPart(ref=MediaRef(uri="a.mp4", num_frames=300)), _video(8))

    def test_a_frame_directory_under_the_video_wire_is_refused(self):
        with pytest.raises(VideoPolicyError, match="wire: frames"):
            sample_video_part(VideoPart(frames=[MediaRef(uri="f0.jpg")]), _video(8, "video_url"))

    def test_a_part_with_both_is_shown_as_the_declared_wire(self):
        part = VideoPart(
            ref=MediaRef(uri="a.mp4", num_frames=300), frames=[MediaRef(uri=f"f{i}.jpg") for i in range(4)]
        )

        assert sample_video_part(part, _video(8, "video_url")) == VideoPart(ref=part.ref)
        assert sample_video_part(part, _video(2)).ref is None

    def test_the_wire_is_in_the_family(self):
        assert Preprocessing(video=_video(8)).key != Preprocessing(video=_video(8, "video_url")).key


class TestVideoPolicy:
    def test_a_frame_count_and_a_wire_are_required(self):
        with pytest.raises(ValueError, match="num_frames"):
            VideoPolicy(wire="frames")
        with pytest.raises(ValueError, match="wire"):
            VideoPolicy(num_frames=8)

    def test_rate_based_sampling_is_not_accepted(self):
        """A rate shows a long clip more frames than a short one under one family."""
        with pytest.raises(ValueError, match="fps"):
            VideoPolicy(num_frames=8, wire="frames", fps=2)


class TestIdentity:
    """The policies reach the judgement family through :attr:`Preprocessing.key`."""

    def test_different_budgets_are_different_families(self):
        small = Preprocessing(image=_qwen(max_px=256 * 28 * 28))
        large = Preprocessing(image=_qwen(max_px=2048 * 28 * 28))
        assert small.key != large.key
        assert Preprocessing(image=QWEN).key == Preprocessing(image=QWEN.model_copy()).key

    def test_frame_counts_are_different_instruments(self):
        assert Preprocessing(video=_video(8)).key != Preprocessing(video=_video(32)).key

    def test_descriptor_names_the_budget(self):
        assert QWEN.descriptor == f"{4 * 28 * 28}-{1280 * 28 * 28}px qwen2_vl"
        assert ImagePolicy.native().descriptor == "native"

    def test_round_trips_through_json(self):
        policy = Preprocessing(image=QWEN, video=_video(8, "video_url", max_duration_s=60))
        assert Preprocessing.model_validate_json(policy.model_dump_json()) == policy
