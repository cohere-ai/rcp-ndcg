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
from rcp_ndcg_core.content import Content, ImagePart, MediaRef, TextPart, VideoPart

from rcp_ndcg.data import Preprocessing
from rcp_ndcg.data.prepare import (
    MediaCensus,
    MediaFit,
    PreparedMedia,
    apply_media_fit,
    fit_media_to_budget,
    prepare_content,
    prepare_image,
    prepare_request,
)
from rcp_ndcg.data.resolution import (
    PROCESSORS,
    ImagePolicy,
    VideoPolicy,
    content_media_tokens,
    smart_resize,
    uniform_frame_indices,
)
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.inference.adapters.chat import build_messages
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

    def test_a_frame_drop_keeps_the_sampled_indices_aligned(self, tmp_path: Path):
        """A media fit that drops frames drops their sampled-index entries with them: the sent part's frames
        and its ``frame_indices`` stay aligned (a part that kept 2 of 4 frames must not claim 4 indices)."""
        frames = [_png(tmp_path / "clip" / f"{i:03d}.png", (640, 360), color=(i, i, i)) for i in range(4)]
        content = Content.from_parts([VideoPart(frames=frames, frame_indices=[10, 20, 30, 40])])
        fit = MediaFit(media=[], tokens=0, dropped=[], decisions=(frames[0], None, frames[2], None))

        (out,) = apply_media_fit([content], fit)

        (part,) = out.parts
        assert [frame.uri for frame in part.frames] == [frames[0].uri, frames[2].uri]
        assert part.frame_indices == [10, 30]

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

        prepared = prepare_content(
            content, image, VideoPolicy(num_frames=8, wire="video_url", engine_video_pinning=True)
        )

        (item,) = prepared.media
        assert item.kind == "video" and item.sent == container
        assert item.as_row(corpus="c", doc_id="d")["sampling"] == "engine"
        (message,) = build_messages(CompletionInput(user_prompt="x", user_content=prepared.content))
        (block,) = message["content"]
        assert block["type"] == "video_url"
        assert base64.b64decode(block["video_url"]["url"].split(",", 1)[1]) == clip.read_bytes()


class TestPrepareRequest:
    """The one preparation call the retrieval role clients make: contents as sent, with their exact media
    token counts, so the role's text budget can subtract them (never cut them)."""

    POLICY = ImagePolicy(min_px=65536, max_px=1280 * 32 * 32, processor="qwen3_vl")

    def test_a_request_returns_prepared_contents_and_exact_counts(self, tmp_path: Path):
        pages = [_png(tmp_path / f"p{i}.png", (1700, 2200), color=(i, i, i)) for i in range(2)]
        contents = [
            Content.from_parts([TextPart(text="the query"), ImagePart(ref=pages[0])]),
            Content.from_parts([ImagePart(ref=pages[1])]),
        ]

        prepared = prepare_request(contents, self.POLICY, None)

        (query_image,) = (part.ref for part in prepared.contents[0].parts if isinstance(part, ImagePart))
        assert query_image.uri.startswith("data:image/png;base64,")
        assert [item.kind for item in prepared.media] == ["image", "image"]
        assert prepared.tokens == (2 * (self.POLICY.image_tokens(2200, 1700) + 2), 0)

    def test_the_counts_equal_the_judge_path_for_the_same_image(self, tmp_path: Path):
        """The reference measurement ran 27 image sizes through the real processors against the judge
        path's count with 0 mismatches; the retrieval path must return that same count for the same
        image, or it is a different instrument."""
        page = _png(tmp_path / "p.png", (1700, 2200))
        judged = prepare_content(Content.from_parts([ImagePart(ref=page)]), self.POLICY, None)

        prepared = prepare_request([Content.from_parts([ImagePart(ref=page)])], self.POLICY, None)

        assert prepared.tokens == content_media_tokens(judged.content, self.POLICY, None)
        assert prepared.tokens.tokens == self.POLICY.image_tokens(2200, 1700) + 2

    def test_content_without_media_counts_nothing(self):
        prepared = prepare_request([Content.from_text("hello")], self.POLICY, None)
        assert prepared.tokens == (0, 0)
        assert prepared.media == []

    def test_an_unrecorded_size_is_recovered_by_preparation(self, tmp_path: Path):
        """Preparation decodes the image, so the sent reference carries the size and the count is exact."""
        page = _png(tmp_path / "p.png", (1700, 2200))
        unrecorded = page.model_copy(update={"width": None, "height": None})

        prepared = prepare_request([Content.from_parts([ImagePart(ref=unrecorded)])], self.POLICY, None)

        assert prepared.tokens == (self.POLICY.image_tokens(2200, 1700) + 2, 0)

    def test_frames_are_sampled_and_counted_as_shown(self, tmp_path: Path):
        frames = [_png(tmp_path / "clip" / f"{i}.png", (640, 360), color=(i, i, i)) for i in range(10)]
        video = VideoPolicy(num_frames=4, wire="frames")

        prepared = prepare_request([Content.from_parts([VideoPart(frames=frames)])], self.POLICY, video)

        per_frame = self.POLICY.image_tokens(360, 640) + 2
        assert prepared.tokens == (4 * per_frame, 0)
        assert [item.kind for item in prepared.media] == ["frame"] * 4


class TestFitMediaToBudget:
    """A vision block is atomic: when media alone exceed a request's text
    budget, images shrink to the policy's minimum, then whole items are dropped with a census record. Tokens
    are never cut inside a block."""

    POLICY = ImagePolicy(min_px=65536, max_px=1280 * 32 * 32, processor="qwen3_vl")

    def _prepared(self, tmp_path: Path, sizes: list[tuple[int, int]]) -> list[PreparedMedia]:
        return [
            prepare_image(_png(tmp_path / f"p{i}.png", size, color=(i, i, i)), self.POLICY)
            for i, size in enumerate(sizes)
        ]

    def test_media_within_the_budget_go_whole(self, tmp_path: Path):
        media = self._prepared(tmp_path, [(2200, 1700)])

        fit = fit_media_to_budget(media, image=self.POLICY, video=None, text_budget_tokens=10_000)

        assert [item.sent for item in fit.media] == [item.sent for item in media]
        assert fit.tokens == self.POLICY.image_tokens(2200, 1700) + 2
        assert fit.dropped == []

    def test_an_oversized_image_is_shrunk_to_the_policy_minimum(self, tmp_path: Path):
        """2560x2560 under the page budget costs 1,225 patches; at the 65,536px floor, 64."""
        media = self._prepared(tmp_path, [(2560, 2560)])
        full = self.POLICY.image_tokens(2560, 2560) + 2
        minimum = ImagePolicy(min_px=65536, max_px=65536, processor="qwen3_vl")
        shrunk = minimum.image_tokens(2560, 2560) + 2
        assert shrunk < full

        fit = fit_media_to_budget(media, image=self.POLICY, video=None, text_budget_tokens=(full + shrunk) // 2)

        (item,) = fit.media
        assert (item.sent.height, item.sent.width) == (256, 256)
        assert fit.tokens == shrunk and fit.dropped == []

    def test_media_that_still_do_not_fit_are_dropped_most_expensive_first(self, tmp_path: Path):
        """Two pages shrink to 66 tokens each; a budget of 100 holds one whole block."""
        media = self._prepared(tmp_path, [(2560, 2560), (2560, 2560)])

        fit = fit_media_to_budget(media, image=self.POLICY, video=None, text_budget_tokens=100)

        assert fit.tokens == 66 <= 100
        assert len(fit.media) == 1 and len(fit.dropped) == 1
        assert fit.media[0].sent == fit.dropped[0].sent.model_copy(update={"sha256": fit.media[0].sent.sha256}) or (
            (fit.media[0].sent.height, fit.media[0].sent.width)
            == (fit.dropped[0].sent.height, fit.dropped[0].sent.width)
        )  # both shrunk to the same size first; the later one was dropped

    def test_a_shrink_that_leaves_the_declared_budget_is_refused(self, tmp_path: Path):
        """A min_px above the engines' floor: flooring at the minimum can land below it, and the declared
        budget would scale the image back up -- so the image cannot shrink within the declared instrument,
        and the budget drops it whole instead of sending a size the declaration does not describe."""
        policy = ImagePolicy(min_px=131072, max_px=1310720, processor="qwen3_vl")
        media = [prepare_image(_png(tmp_path / f"p{i}.png", (1000, 300), color=(i, i, i)), policy) for i in range(2)]
        whole = policy.image_tokens(300, 1000) + 2  # (288, 992): the declared budget keeps it

        fit = fit_media_to_budget(media, image=policy, video=None, text_budget_tokens=400)

        assert fit.tokens == whole <= 400
        assert len(fit.media) == 1 and len(fit.dropped) == 1
        assert all((item.sent.height, item.sent.width) == (288, 992) for item in fit.media + fit.dropped)

    def test_an_image_that_cannot_shrink_to_a_fixed_point_is_dropped_whole(self, tmp_path: Path):
        """2200x1700 floored to the 65,536px floor lands under it, which the engine would resize again --
        so the image cannot shrink, and the budget drops it rather than sending a size the engine changes."""
        media = self._prepared(tmp_path, [(2200, 1700), (2200, 1700)])
        whole = self.POLICY.image_tokens(2200, 1700) + 2

        fit = fit_media_to_budget(media, image=self.POLICY, video=None, text_budget_tokens=whole + 258)

        assert fit.tokens == whole <= whole + 258
        assert len(fit.media) == 1 and len(fit.dropped) == 1
        assert fit.media[0].sent == media[0].sent  # kept whole, never cut
        assert fit.dropped[0].sent == media[1].sent

    def test_a_container_cannot_shrink_so_it_is_dropped_whole(self, tmp_path: Path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"\x00" * 64)
        container = MediaRef(uri=str(clip), mime="video/mp4", num_bytes=64, num_frames=300)
        video = VideoPolicy(num_frames=8, wire="video_url", engine_video_pinning=True)
        prepared = prepare_content(Content.from_parts([VideoPart(ref=container)]), self.POLICY, video)

        fit = fit_media_to_budget(
            prepared.media, image=self.POLICY, video=video, text_budget_tokens=4 * self.POLICY.max_image_tokens
        )

        assert fit.media == [] and [item.kind for item in fit.dropped] == ["video"]

    def test_a_budget_too_small_for_even_the_minimum_drops_everything(self, tmp_path: Path):
        media = self._prepared(tmp_path, [(2560, 2560)])

        fit = fit_media_to_budget(media, image=self.POLICY, video=None, text_budget_tokens=2)

        assert fit.media == [] and fit.tokens == 0 and len(fit.dropped) == 1

    def test_drops_are_decided_on_exact_counts_not_guesses(self, tmp_path: Path):
        """An item with no recorded size is bounded, so the fit errs high and says so is not needed:
        the decision still never cuts a block."""
        media = [
            PreparedMedia(
                kind="image",
                source=MediaRef(uri="gs://p/x.png"),
                sent=MediaRef(uri="data:image/png;base64,AAAA", width=None, height=None),
                processor=self.POLICY.processor,
                resized=False,
            )
        ]

        fit = fit_media_to_budget(media, image=self.POLICY, video=None, text_budget_tokens=100)

        assert fit.media == [] and len(fit.dropped) == 1


class TestDroppedCensusRows:
    def test_a_dropped_item_is_recorded_as_not_sent(self, tmp_path: Path):
        page = _png(tmp_path / "p.png", (1700, 2200))
        policy = ImagePolicy(min_px=65536, max_px=1280 * 32 * 32, processor="qwen3_vl")
        item = prepare_image(page, policy)
        sink = tmp_path / "preprocessing.jsonl"

        census = MediaCensus(sink=sink)
        census.record(corpus="c", doc_id="d1", media=[item], dropped=True)

        (row,) = [json.loads(line) for line in sink.read_text().splitlines()]
        assert row["dropped"] is True and row["uri"] == page.uri
        assert row["sent_width"] == 992  # the size it was refused at, not a fiction

    def test_a_drop_after_a_kept_pass_of_the_same_item_is_still_recorded(self, tmp_path: Path):
        """The outcome is part of the dedup key: a budget that first kept an item and a later one that refused
        it are both on record -- a kept row must not hide the later drop, in the sink or on resume."""
        page = _png(tmp_path / "p.png", (1700, 2200))
        policy = ImagePolicy(min_px=65536, max_px=1280 * 32 * 32, processor="qwen3_vl")
        item = prepare_image(page, policy)
        sink = tmp_path / "preprocessing.jsonl"

        census = MediaCensus(sink=sink)
        census.record(corpus="c", doc_id="d1", media=[item])
        census.record(corpus="c", doc_id="d1", media=[item], dropped=True)
        MediaCensus(sink=sink).record(corpus="c", doc_id="d1", media=[item], dropped=True)  # a resumed pass

        rows = [json.loads(line) for line in sink.read_text().splitlines()]
        assert sorted(row["dropped"] for row in rows) == [False, True]

    def test_a_kept_row_says_it_was_sent(self, tmp_path: Path):
        page = _png(tmp_path / "p.png", (1700, 2200))
        policy = ImagePolicy(min_px=65536, max_px=1280 * 32 * 32, processor="qwen3_vl")
        item = prepare_image(page, policy)

        assert item.as_row(corpus="c", doc_id="d")["dropped"] is False


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


class TestCensusReader:
    def test_an_unterminated_but_parseable_row_is_torn_not_counted(self, tmp_path: Path):
        """The readers' definition of unfinished matches the cutter's: an unterminated last row (parseable or
        not) is skipped and recorded again -- the next append can never silently delete a row this read
        counted as on record."""
        from rcp_ndcg.data.preprocess import read_census_rows

        sink = tmp_path / "preprocessing.jsonl"
        sink.write_text('{"mechanism": "doc_policy", "corpus": "c"}\n{"mechanism": "doc_policy", "corpus": "d"}')
        rows = list(read_census_rows(sink))
        assert [row["corpus"] for row in rows] == ["c"], "the unterminated row is a torn write: absent"
        # And a complete line that is not a census row is refused with the typed error.
        sink.write_text('{"mechanism": "doc_policy", "corpus": "c"}\n{"foo": 1}\n')
        from rcp_ndcg.errors import DataError

        with pytest.raises(DataError, match="census row"):
            list(read_census_rows(sink))
