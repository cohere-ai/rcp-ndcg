"""How page images and video are prepared before a judge sees them -- declared, not implied.

Every vision-language model resizes its input, and the choice changes both the *cost* (token count) and the
*content* (what is legible at that resolution). Two judging passes at different pixel budgets are two different
measurements, so the policy is a declared, recorded object (part of
:class:`~rcp_ndcg.data.preprocess.Preprocessing`, and so of the judgement family) rather than a processor default
that happens to be in effect.

The client prepares every image itself (:mod:`rcp_ndcg.data.prepare`), so a stock engine needs no media flags:

* :class:`ImagePolicy` -- the pixel budget ``[min_px, max_px]`` and the judge's image processor family
  (:data:`ImageProcessor`, :data:`PROCESSORS`). Under a known family each image is resized exactly as that
  processor would (:func:`smart_resize`: both edges snapped to a multiple of the family's factor, aspect ratio
  kept), and the budget is checked to lie inside the engines' default budget, so the engine's own resize of the
  prepared image is a no-op. Without a budget, or without a known family, the image is sent unchanged and the
  processor decides, which also means its token cost is unknown.
* :class:`VideoPolicy` -- ``num_frames`` uniformly spaced frames per clip, and how they travel. With ``wire:
  frames`` the client samples the frames (:func:`uniform_frame_indices`) and sizes each by the image policy;
  with ``wire: video_url`` the container is sent unchanged and the engine samples it, which it may only do
  when the policy declares the engine pinned to the same frame count. :func:`sample_video_part` is the one
  place a video's frames are chosen, and :func:`content_media_tokens` counts exactly what it returns, as the
  engine will count it in the prompt.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rcp_ndcg_core.content import Content, ImagePart, MediaRef, TextPart, VideoPart

from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.support.identity import FieldRole

if TYPE_CHECKING:
    from rcp_ndcg.data.tokenizer import TextTokenizer

ImageProcessor = Literal["qwen2_vl", "qwen2_5_vl", "qwen3_vl"]
"""The image processor families whose resize the client reproduces (the judge config's ``image_processor``).

``qwen2_vl`` is Qwen2-VL, ``qwen2_5_vl`` is Qwen2.5-VL, and ``qwen3_vl`` is Qwen3-VL and the natively multimodal
Qwen3.5-397B and Qwen3.6-27B, whose checkpoints ship the same image processor with a 16-pixel patch."""


class ProcessorGeometry(NamedTuple):
    """How one processor family sizes an image and a video, and the budgets both engines apply when started
    without media flags.

    Attributes:
        factor: Pixels per token edge: the vision patch times the spatial merge. Both edges of a resized image
            are multiples of it, and each ``factor x factor`` block costs one token.
        min_pixels: The engines' default floor, in pixels, for images the client sends prepared.
        max_pixels: The engines' default ceiling, in pixels. Where vLLM and SGLang differ, the lower one.
        temporal_patch: Frames the vision tower merges in time, so a container of ``num_frames`` frames costs
            ``ceil(num_frames / temporal_patch)`` per-frame token runs (transformers' video processors patchify
            with ``temporal_patch_size = 2`` for every family here, padding an odd clip by repeating its last
            frame).
        video_min_pixels: The per-frame floor the engine's video accounting applies to a container's frames,
            in pixels.
        video_max_pixels: The per-frame ceiling, in pixels -- or, when :attr:`video_pixels_per_clip` is set,
            the whole clip's ceiling, which shrinks the per-frame resolution as the frame count grows.
        video_pixels_per_clip: Whether the video pixel budget constrains the whole clip together (the
            Qwen3-VL video processor's clip-level resize) rather than each frame independently (the
            Qwen2-VL families' video processors, and stock vLLM's video accounting, which sizes each frame
            by the checkpoint's image-processor size).
        video_timestamp_tokens: The tokens of the timestamp line the family's processor renders before each
            temporal group's vision block in the prompt (a declared bound; 0 when a family renders none).
            The bound covers every timestamp a clip of up to 99,999.9 s (~27.8 h) can carry; a longer clip
            (only possible with ``max_duration_s`` unset) adds a token per group, so declare
            ``max_duration_s`` for clips of that length.
    """

    factor: int
    min_pixels: int
    max_pixels: int
    temporal_patch: int = 2
    video_min_pixels: int | None = None
    video_max_pixels: int | None = None
    video_pixels_per_clip: bool = False
    video_timestamp_tokens: int = 0


PROCESSORS: dict[str, ProcessorGeometry] = {
    # Qwen2-VL checkpoints: patch 14 x merge 2 and {min,max}_pixels 3136..12845056 in preprocessor_config.json,
    # which vLLM applies; SGLang overrides the image ceiling to 1003520 for model_type qwen2_vl
    # (sglang python/sglang/srt/utils/hf_transformers/processor.py:279-281 @ 45c8ddd). A container's frames
    # are sized by that same checkpoint budget per frame (vLLM's video accounting passes the image
    # processor's size, qwen2_vl.py:1014 @ d0d6e5f3a); SGLang's video path caps per-frame pixels lower
    # (602112px, clip-dependent).
    "qwen2_vl": ProcessorGeometry(
        factor=28,
        min_pixels=56 * 56,
        max_pixels=28 * 28 * 1280,
        video_min_pixels=56 * 56,
        video_max_pixels=12845056,
    ),
    # Qwen2.5-VL checkpoints: the same processor and budget, which both engines apply as shipped.
    "qwen2_5_vl": ProcessorGeometry(
        factor=28,
        min_pixels=56 * 56,
        max_pixels=28 * 28 * 16384,
        video_min_pixels=56 * 56,
        video_max_pixels=28 * 28 * 16384,
    ),
    # Qwen3-VL, Qwen3.5-397B and Qwen3.6-27B checkpoints: patch 16 x merge 2 and size {shortest_edge: 65536,
    # longest_edge: 16777216} in preprocessor_config.json, which both engines apply as shipped. The video
    # processor ships its own per-clip budget, 4096..25165824 px (video_preprocessor_config.json), on the
    # same 2-frame temporal patch, and renders one timestamp line (a bound of 10 tokens) per temporal
    # group in the prompt.
    "qwen3_vl": ProcessorGeometry(
        factor=32,
        min_pixels=65536,
        max_pixels=16777216,
        video_min_pixels=4096,
        video_max_pixels=25165824,
        video_pixels_per_clip=True,
        video_timestamp_tokens=10,
    ),
}
"""Every :data:`ImageProcessor` family's geometry. The resize itself is transformers' ``Qwen2VLImageProcessor``
(``smart_resize`` with ``factor = patch_size * merge_size``, BICUBIC), which vLLM and SGLang both run for these
models with the checkpoint's own size: vLLM in ``Qwen2VLProcessingInfo._get_vision_size`` /
``Qwen3VLProcessingInfo._get_vision_info`` (vllm/model_executor/models/qwen2_vl.py:952-978,
qwen3_vl.py:957-1003 @ 3627a6a), SGLang through the HF processor (python/sglang/srt/multimodal/processors/
base_processor.py:838-927 @ 45c8ddd)."""


class VideoPolicy(BaseModel):
    """Which frames of a video the judge is shown, and which videos it may be shown at all.

    One sampling rule: ``num_frames`` frames at uniformly spaced indices over the
    whole clip (:func:`uniform_frame_indices`), the rule both serving engines apply
    to a decoded container. It is the only rule whose realised frame count is fixed
    by the policy -- a frames-per-second rule shows a 10-second clip 20 frames and a
    10-minute clip 1200, so two runs "at 2 fps" would share a judgement family while
    showing the judge different amounts of video.

    A clip with fewer frames than ``num_frames`` is refused, not shown whole: the
    engines disagree about it (SGLang rejects the request, vLLM resamples at its
    processor's own rate), so no single number of frames could be recorded for it.
    A container's frame count must therefore be recorded at ingest
    (``hash_media=True``) before it can be judged over ``video_url``.

    ``wire`` is how the judge receives the frames, and it is part of the
    instrument. ``frames`` (client-sampled, the default and the exact one): the
    client picks the frames from the corpus's pre-extracted frames and sends each,
    sized by the image policy, as its own image, which any OpenAI-compatible
    endpoint accepts -- so the frames counted are the frames sent.
    ``video_url`` (engine-sampled, an opt-in for models with a native video encoder):
    the container is sent unchanged and the engine decodes and samples it (Qwen-VL
    towers then merge frame pairs in time and see timestamps). A stock engine samples
    its own default number of frames (32 on vLLM), so ``video_url`` is refused unless
    :attr:`engine_video_pinning` declares the engine pinned -- to a uniform
    :attr:`num_frames`, or to the engine's own fps rule (:attr:`fps`, the vLLM v0.31.0
    ``Qwen3VLVideoBackend``: ``int(total_frames / original_fps * fps)`` frames clamped
    to its bounds, :func:`qwen3_vl_video_frame_indices`). The two are different
    measurements, so exactly one is declared: a policy that declares neither, or both,
    is refused rather than silently picking one. The same clips judged both ways are
    different measurements, so the wire is part of the policy and a corpus whose videos
    do not match it is refused.

    ``max_duration_s`` is a refusal, not a cut: uniform sampling of a long clip
    spreads the same frame budget ever thinner, so a corpus that declares it refuses
    any video whose recorded duration exceeds it -- or whose duration was never
    recorded -- rather than judging it at a density nobody chose.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_frames: int | None = Field(default=None, gt=0)
    """Frames shown per video, sampled uniformly over the clip: the rule for ``wire: frames``
    and a pinned ``wire: video_url`` engine. ``None`` with :attr:`fps` set: the engine's fps
    rule decides the realised count per clip."""

    fps: float | None = Field(default=None, gt=0)
    """The target sampling rate for a ``wire: video_url`` engine's own fps rule (the vLLM
    v0.31.0 ``Qwen3VLVideoBackend``: ``int(total_frames / original_fps * fps)`` frames clamped
    to its 30 fps ceiling and its frame bounds, :func:`qwen3_vl_video_frame_indices`). The
    client counts the frames the engine samples from each clip's recorded frame count and fps.
    ``None`` with :attr:`num_frames` set. Content: a different rate shows the model different
    frames."""

    wire: Literal["frames", "video_url"]
    """``frames``: sampled frames sent as images. ``video_url``: the container, decoded by the engine."""

    engine_video_pinning: bool = False
    """Whether the engine serving this corpus is pinned to sample exactly the declared frames per
    container: vLLM ``--media-io-kwargs '{"video": {"num_frames": N}}'`` for a pinned count, or
    ``'{"video": {"fps": N}}'`` for the engine's fps rule (the Qwen3-VL backend samples by fps and
    ignores ``num_frames``; see :func:`qwen3_vl_video_frame_indices`). Required for ``wire:
    video_url`` -- the engine's own default sampling would make the counted tokens and the recorded
    instrument describe frames nobody chose -- and refused under ``wire: frames``, which samples on
    the client."""

    max_duration_s: float | None = Field(default=None, gt=0)
    """Longest clip, in seconds, this corpus may be judged on; ``None`` for no limit.

    Checked against :attr:`MediaRef.duration_s`, which ingest records for
    containers (``hash_media=True``)."""

    @model_validator(mode="after")
    def _pinning_matches_the_wire(self) -> Self:
        """A container's frame count is the engine's to sample, so the declaration is required for
        ``video_url`` and meaningless under ``frames``; neither rule may pass silently."""
        if self.wire == "frames":
            if self.num_frames is None:
                raise ValueError(
                    "`wire: frames` samples on the client: declare `num_frames`, the frames it sends as images"
                )
            if self.fps is not None:
                raise ValueError(
                    "`fps` is the engine's container sampling rate (`wire: video_url`); `wire: frames` samples "
                    "`num_frames` uniform frames on the client -- declare one rule"
                )
            if self.engine_video_pinning:
                raise ValueError(
                    "`engine_video_pinning` declares how the engine samples a container, but `wire: frames` sends "
                    "sampled frames as images and the engine never samples. Drop the declaration."
                )
            return self
        if (self.num_frames is None) == (self.fps is None):
            raise ValueError(
                "`wire: video_url` sends the container for the engine to sample: declare exactly one of "
                "`num_frames` (a pinned uniform count) or `fps` (the engine's own rate, the vLLM v0.31.0 "
                "Qwen3-VL backend's rule)"
            )
        if self.num_frames is not None and self.num_frames < 2:
            raise ValueError(
                f"`wire: video_url` shows {self.num_frames} frame, but the declared instrument merges frames "
                "in time, which needs at least a temporal pair -- the one rule under which a clip's realised "
                "frame count is the policy's. A single frame is an image: declare `wire: frames` with "
                "num_frames >= 2, or judge the clip as an image."
            )
        if not self.engine_video_pinning:
            raise ValueError(
                "`wire: video_url` sends the container for the engine to sample, so the frame count is the "
                "engine's default (32 frames on vLLM), not the declared one. Declare `engine_video_pinning: "
                "true` and serve the engine pinned to the same sampling (`--media-io-kwargs` on vLLM), or "
                "declare `wire: frames`, which the client samples itself."
            )
        return self

    @property
    def descriptor(self) -> str:
        """Human-readable one-liner, e.g. ``frames-n8``, ``video_url-n8`` or ``video_url-f2``."""
        if self.fps is not None:
            return f"{self.wire}-f{self.fps:g}"
        return f"{self.wire}-n{self.num_frames}"

    #: Every field is content: the frame policy is part of the instrument (the preprocessing record and the
    #: judgement family), and the roles are declared so a media policy nested in an identity payload passes
    #: :func:`~rcp_ndcg.support.identity.check_declarations`.
    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "num_frames": FieldRole.CONTENT,
        "fps": FieldRole.CONTENT,
        "wire": FieldRole.CONTENT,
        "engine_video_pinning": FieldRole.CONTENT,
        "max_duration_s": FieldRole.CONTENT,
    }


def uniform_frame_indices(total_frames: int, num_frames: int) -> list[int]:
    """The frame indices uniform sampling shows, ``[0, total_frames)``.

    ``np.linspace(0, total - 1, n)`` truncated to integers -- what vLLM's default
    video loader (``VideoBackend.compute_frames_index_to_sample``,
    vllm/multimodal/video.py:236-238 @ 3627a6a) and SGLang's Qwen-VL
    ``preprocess_video`` (python/sglang/srt/multimodal/processors/qwen_vl.py:264 @ 45c8ddd)
    do to a decoded container. The first and last frame are included whenever
    ``num_frames`` is at least 2; with ``num_frames >= total_frames`` every frame is
    returned once.

    Args:
        total_frames: Frames available in the clip (> 0).
        num_frames: Frames requested (> 0).

    Returns:
        Strictly increasing indices, ``min(num_frames, total_frames)`` of them.
    """
    if total_frames <= 0 or num_frames <= 0:
        raise DataError(
            f"need positive frame counts, got total={total_frames}, requested={num_frames}",
            hint="sample at least one frame from a container that has one",
        )
    if num_frames >= total_frames:
        return list(range(total_frames))
    return [int(index) for index in np.linspace(0, total_frames - 1, num_frames, dtype=np.int64)]


def qwen3_vl_video_frame_indices(
    total_frames: int,
    original_fps: float,
    *,
    fps: float,
    min_frames: int = 4,
    max_frames: int = 768,
    max_fps: float = 30.0,
) -> list[int]:
    """The frame indices vLLM v0.31.0's ``Qwen3VLVideoBackend`` samples from a clip.

    A faithful port of ``Qwen3VLVideoBackend.compute_frames_index_to_sample``
    (vllm/multimodal/video.py:360-400 at the v0.31.0 tag): the target rate is clamped to the
    backend's 30 fps ceiling; the count is ``int(total_frames / original_fps * fps)``, then
    clamped to ``[min_frames, max_frames, total_frames]``; the indices are
    ``np.linspace(0, total_frames - 1, count).round()``. The backend ignores ``num_frames``
    entirely -- the count is fps-driven -- which is why the policy declares :attr:`VideoPolicy.fps`
    and the client counts a clip's frames with this function rather than a declared count.

    Args:
        total_frames: Frames in the clip (> 0), as ingest recorded them.
        original_fps: The clip's own rate (> 0), as ingest recorded it.
        fps: The target sampling rate (> 0); clamped to ``max_fps``.
        min_frames, max_frames: The backend's own bounds (4 and 768).
        max_fps: The backend's fps ceiling (30).

    Returns:
        The sampled indices, ascending, ``min(max(int(total/original*fps), min_frames), max_frames,
        total)`` of them.

    Raises:
        DataError: a non-positive frame count or rate, or a clip whose frame count/rate was never
            recorded -- the count cannot be computed and a guess would misprice the prompt.
    """
    if total_frames <= 0 or original_fps <= 0 or fps <= 0:
        raise DataError(
            f"need positive frame counts and rates, got total={total_frames}, original_fps={original_fps}, fps={fps}",
            hint="the clip's header must state its frame count and rate, and the policy its target rate",
        )
    target = min(fps, max_fps)
    num_frames = int(total_frames / original_fps * target)
    num_frames = min(max(num_frames, min_frames), max_frames, total_frames)
    return [int(index) for index in np.linspace(0, total_frames - 1, num_frames).round().astype(int)]


def _video_frame_indices(video: VideoPolicy, ref: MediaRef) -> list[int]:
    """The indices the engine samples from ``ref`` under ``video``'s declared rule.

    ``fps`` (the engine's own rule, :func:`qwen3_vl_video_frame_indices`): the clip's recorded frame
    count and rate are required. ``num_frames``: the uniform count (:func:`uniform_frame_indices`).
    """
    if ref.num_frames is None:
        raise VideoPolicyError(
            f"{ref.uri}: the video policy samples the engine's fps rule, but this container's frame count "
            "was never recorded, so its realised frame count cannot be counted. Ingest with `hash_media=True` "
            "(MP4, MOV and AVI headers are read; re-encode WebM/MKV to MP4)."
        )
    if video.fps is not None:
        if ref.fps is None:
            raise VideoPolicyError(
                f"{ref.uri}: the video policy samples at {video.fps:g} fps, but this container's own frame "
                "rate was never recorded, so the engine's sampled frame count cannot be counted. Ingest with "
                "`hash_media=True` so the header is probed."
            )
        return qwen3_vl_video_frame_indices(ref.num_frames, ref.fps, fps=video.fps)
    assert video.num_frames is not None  # the policy validator refuses neither rule
    return uniform_frame_indices(ref.num_frames, video.num_frames)


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Qwen-VL ``smart_resize``: snap both edges to ``factor`` and fit the pixel budget.

    A faithful port of transformers' ``smart_resize``
    (models/qwen2_vl/image_processing_qwen2_vl.py:63-89 @ 528c267; Apache-2.0, see NOTICE): round each edge to
    the nearest multiple of ``factor``; if the area then exceeds ``max_pixels``, scale
    down by ``sqrt(area / max_pixels)`` and floor each edge to the factor (at least one
    factor); if it falls below ``min_pixels``, scale up by ``sqrt(min_pixels / area)``
    and ceil each edge to the factor.

    Args:
        height, width: The image's size, in pixels.
        factor: The processor family's :attr:`ProcessorGeometry.factor`.
        min_pixels, max_pixels: The pixel budget.

    Returns:
        ``(height, width)`` the processor resizes to.

    Raises:
        DataError: a non-positive edge, or an aspect ratio above 200 (the processor refuses it too); both
            carry a hint naming the fix.
    """
    if min(height, width) <= 0:
        raise DataError(
            f"image dimensions must be positive, got {height}x{width}",
            hint="a recorded size is read from the stored image; this one is corrupt",
        )
    if max(height, width) / min(height, width) > 200:
        raise DataError(
            f"aspect ratio must be below 200, got {max(height, width) / min(height, width):.1f}",
            hint="crop or split the image at ingest so its aspect ratio is below 200 (the processors refuse it)",
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _budget_problem(min_px: int, max_px: int, processor: str, *, pinned: bool = False) -> str | None:
    """Why a stock engine serving ``processor`` would resize the budget ``[min_px, max_px]`` again, if it would.

    ``pinned``: the engine is pinned to exactly this budget (:attr:`ImagePolicy.engine_pixel_pinning`), so it
    keeps the prepared size whatever the family's stock range is -- no problem then."""
    if pinned:
        return None
    geometry = PROCESSORS[processor]
    if min_px < geometry.min_pixels or max_px > geometry.max_pixels:
        return (
            f"the pixel budget {min_px}-{max_px}px lies outside what a stock engine serving the {processor} "
            f"processor keeps ({geometry.min_pixels}-{geometry.max_pixels}px), so the engine would resize the "
            f"prepared image again. Declare a budget inside that range, or serve the engine pinned to this "
            f"budget and declare engine_pixel_pinning: true."
        )
    return None


class ImagePolicy(BaseModel):
    """The pixel budget every page image and video frame is resized to, and the processor whose resize is used.

    Attributes:
        min_px: The fewest pixels an image is scaled up to.
        max_px: The most pixels an image is scaled down to.
        processor: The judge's image processor family (:data:`ImageProcessor`). Left unset in a declared policy:
            the judging pass takes it from the judge config's ``image_processor`` (:meth:`for_processor`), so the
            recorded policy names it. ``None`` in an effective policy means the family is unknown, and images are
            sent unchanged.

        engine_pixel_pinning: Whether the engine serving this policy is pinned to exactly this pixel budget
            (vLLM ``--mm-processor-kwargs '{"images_kwargs": {"min_pixels": <min_px>, "max_pixels": <max_px>}}'``;
            a serving recipe checks that both sides carry the same numbers). Then the budget is the engine's
            own and may lie outside the family's stock range -- the Qwen3-VL-Embedding card's 4096 px floor
            under qwen3_vl's stock 65536 -- and the engine keeps the prepared size. ``False`` is the absence of
            the declaration (stored as ``None``, so a policy that never names it keeps its identity).

    Budget both or neither: without one, images go at their stored size and the engine's processor decides
    (:meth:`native`), so their token cost cannot be counted. With a known processor the budget must lie within
    the engines' default budget for it (:data:`PROCESSORS`), so the engine keeps the prepared size -- unless the
    engine is pinned to the budget itself (:attr:`engine_pixel_pinning`).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min_px: int | None = Field(default=None, gt=0)
    max_px: int | None = Field(default=None, gt=0)
    processor: ImageProcessor | None = None
    engine_pixel_pinning: Literal[True] | None = Field(default=None, exclude_if=lambda value: value is None)

    @field_validator("engine_pixel_pinning", mode="before")
    @classmethod
    def _false_is_no_declaration(cls, value: object) -> object:
        """``false`` declares nothing: stored as ``None``, so it never re-keys a policy that omits it."""
        return None if value is False else value

    if TYPE_CHECKING:
        # Frozen, so pydantic installs a field-value hash at class creation; declared here for type checkers.
        def __hash__(self) -> int: ...

    @model_validator(mode="after")
    def _both_or_neither(self) -> Self:
        if (self.min_px is None) != (self.max_px is None):
            raise ValueError("an image pixel budget needs both `min_px` and `max_px` (or neither, for native size)")
        if self.min_px is not None and self.max_px is not None:
            if self.min_px > self.max_px:
                raise ValueError(f"min_px ({self.min_px}) exceeds max_px ({self.max_px})")
            if self.processor is not None and (
                problem := _budget_problem(self.min_px, self.max_px, self.processor, pinned=self.pinned)
            ):
                raise ValueError(problem)
        elif self.engine_pixel_pinning:
            raise ValueError(
                "engine_pixel_pinning declares the engine pinned to this policy's pixel budget, and the policy "
                "declares none: declare min_px and max_px (the numbers the engine is pinned to)"
            )
        return self

    @property
    def pinned(self) -> bool:
        """Whether the engine is declared pinned to this policy's pixel budget (:attr:`engine_pixel_pinning`)."""
        return bool(self.engine_pixel_pinning)

    @property
    def is_native(self) -> bool:
        """Whether the image goes at its stored size (no pixel budget)."""
        return self.max_px is None

    @property
    def resizes(self) -> bool:
        """Whether the client resizes images: a pixel budget under a known processor family."""
        return not self.is_native and self.processor is not None

    @property
    def descriptor(self) -> str:
        """Human-readable one-liner, e.g. ``65536-1003520px qwen3_vl``, ``3136-1003520px``, ``native``, or
        ``4096-1843200px qwen3_vl pinned`` for a budget the engine is pinned to."""
        if self.is_native:
            return "native"
        processor = f" {self.processor}" if self.processor else ""
        return f"{self.min_px}-{self.max_px}px{processor}" + (" pinned" if self.pinned else "")

    def for_processor(self, processor: ImageProcessor | None) -> ImagePolicy:
        """This policy under the judge's processor family: the effective policy a judging pass records.

        Args:
            processor: The judge config's ``image_processor``; ``None`` when it declares none.

        Raises:
            ConfigError: the policy names a different processor than the judge, or the budget lies outside the
                engines' default budget for the judge's processor.
        """
        if self.processor is not None and processor is not None and self.processor != processor:
            raise ConfigError(
                f"the image policy names the {self.processor} processor, but the judge's image_processor is "
                f"{processor}",
                hint="leave the policy's own processor unset: the judge's image_processor (the role configs' "
                "image_processor field) decides it",
            )
        chosen = self.processor or processor
        if chosen is not None and self.min_px is not None and self.max_px is not None:
            problem = _budget_problem(self.min_px, self.max_px, chosen, pinned=self.pinned)
            if problem is not None:
                geometry = PROCESSORS[chosen]
                raise ConfigError(
                    problem,
                    hint=f"e.g. the image policy's pixel budget: {{min_px: {geometry.min_pixels}, max_px: "
                    f"{min(geometry.max_pixels, 1280 * geometry.factor**2)}}} (the judge declares it under "
                    "preprocessing.image, a role config as image_policy)",
                )
        return self.model_copy(update={"processor": chosen})

    def target_size(self, height: int, width: int) -> tuple[int, int]:
        """The ``(height, width)`` the client sends: the processor's resize under this budget, else unchanged.

        Raises:
            DataError: the image is one a stock engine serving this processor refuses or would resize again:
                an input whose aspect ratio is above 200 (the processor refuses it outright), a resized size
                whose aspect ratio the processor refuses, or flooring to the factor left it outside the
                engines' default budget, so the engine would resize the prepared image again. The message
                names the fix; nothing is sent that the engine would change.
        """
        if not self.resizes:
            return height, width
        assert self.min_px is not None and self.max_px is not None and self.processor is not None
        geometry = PROCESSORS[self.processor]
        try:
            target = smart_resize(height, width, factor=geometry.factor, min_pixels=self.min_px, max_pixels=self.max_px)
        except (ValueError, DataError) as exc:  # the input's aspect ratio is one the processor refuses
            raise DataError(
                f"a {height}x{width} image under the budget {self.descriptor} has an aspect ratio the "
                f"{self.processor} processor refuses ({exc}); the image must be cropped or split at ingest",
                hint="crop or split the image at ingest so its aspect ratio is below 200",
            ) from exc
        # The budget the serving engine applies to the prepared image: its pinned one (this policy's), else the
        # family's stock range.
        engine_min, engine_max = (
            (self.min_px, self.max_px) if self.pinned else (geometry.min_pixels, geometry.max_pixels)
        )
        try:
            kept = smart_resize(*target, factor=geometry.factor, min_pixels=engine_min, max_pixels=engine_max)
        except (ValueError, DataError) as exc:  # the resized image's aspect ratio is one the processor refuses
            raise DataError(
                f"a {height}x{width} image resizes to {target[0]}x{target[1]} under the budget {self.descriptor}, "
                f"which a stock engine serving the {self.processor} processor refuses ({exc}); crop the image at "
                "ingest",
                hint="crop the image at ingest",
            ) from exc
        if kept != target:
            raise DataError(
                f"a {height}x{width} image resizes to {target[0]}x{target[1]} under the budget {self.descriptor}, "
                f"which a stock engine serving the {self.processor} processor would resize again to "
                f"{kept[0]}x{kept[1]}; widen the budget, or crop the image at ingest",
                hint="widen the budget, or crop the image at ingest",
            )
        return target

    def _factor(self, what: str) -> int:
        if self.is_native:
            raise ConfigError(
                f"Cannot {what} for a native-size image policy: the processor decides the geometry.",
                hint="declare a pixel budget -- the judge's preprocessing.image or a role config's "
                "image_policy: {min_px, max_px} -- to get an estimate",
            )
        if self.processor is None:
            raise ConfigError(
                f"Cannot {what}: the config declares no image_processor, so images are sent unchanged and the "
                "engine's processor decides their geometry.",
                hint=f"set image_processor on the config (one of {', '.join(PROCESSORS)})",
            )
        return PROCESSORS[self.processor].factor

    def image_tokens(self, height: int, width: int) -> int:
        """How many tokens one image of this size costs under this policy.

        Raises:
            ConfigError: the policy is native or its processor is unknown: the engine decides the geometry, and a
                guess would silently corrupt a budget estimate.
        """
        factor = self._factor("count tokens")
        target_h, target_w = self.target_size(height, width)
        return (target_h // factor) * (target_w // factor)

    @property
    def max_image_tokens(self) -> int:
        """The most one image can cost, whatever its dimensions: the bound a cost preflight uses when an image's
        size was never recorded (an estimate that over-counts rather than under-counts).

        Raises:
            ConfigError: the policy is native or its processor is unknown (no ceiling to report).
        """
        factor = self._factor("bound tokens")
        assert self.max_px is not None
        return self.max_px // (factor**2)

    @classmethod
    def native(cls) -> ImagePolicy:
        """No budget: the image goes at its stored size and the processor decides."""
        return cls()

    #: Every field is content: the pixel budget and the processor family are part of the instrument (the
    #: preprocessing record and the judgement family), and the roles are declared so a media policy nested in
    #: an identity payload passes :func:`~rcp_ndcg.support.identity.check_declarations`.
    IDENTITY_ROLES: ClassVar[dict[str, FieldRole]] = {
        "min_px": FieldRole.CONTENT,
        "max_px": FieldRole.CONTENT,
        "processor": FieldRole.CONTENT,
        "engine_pixel_pinning": FieldRole.CONTENT,
    }


class MediaTokenCount(NamedTuple):
    """Token cost of some content's media, and how much of it was a bound.

    ``bounded`` counts the references whose dimensions were never recorded, so a
    caller can say "at most N tokens" honestly instead of presenting a ceiling as
    a measurement.
    """

    tokens: int
    bounded: int


#: The tokens around one media item's patch run in the rendered prompt: the processor's vision start and
#: vision end markers. The same two for every family here (``vision_start`` and ``vision_end`` wrap the
#: placeholder run the engine expands to the patch tokens).
VISION_WRAPPER_TOKENS = 2


class EngineMediaMismatch(NamedTuple):
    """An engine's media DELTA for one prepared probe image, against the client's counted media tokens.

    The delta is the engine's ``usage.prompt_tokens`` with the probe image minus its report for the same
    request without it -- the chat template, the special tokens and the probe's text cancel, so the delta
    reports the media block alone. The two numbers disagree when the engine was started with a media budget
    nobody declared, when its processor family is not the one the client reproduced, or when the served
    model changed under the same name: any of these means the counted tokens do not describe what the
    engine sees.
    """

    reported: int
    counted: int

    @property
    def difference(self) -> int:
        """How many tokens the engine reports above the client's count (negative: fewer)."""
        return self.reported - self.counted

    @property
    def message(self) -> str:
        """What the mismatch means, for a person or a record."""
        return (
            f"the engine's media block costs {self.reported:,} prompt tokens for the probe (its report with "
            f"the image minus its report without it), but the client counted {self.counted:,} "
            f"({self.difference:+,}): the served engine's media handling is not the one the counted tokens "
            "describe. Check that the engine runs the declared image_processor without media flags that "
            "resize again, and that its prompt-token report covers the probe requests."
        )


def engine_media_check(reported: int, counted: int) -> EngineMediaMismatch | None:
    """Compare an engine's media DELTA for one prepared probe image with the counted media tokens.

    The delta is the engine's ``usage.prompt_tokens`` for the probe request WITH the prepared image minus
    its report for the same request WITHOUT the media: everything the two requests share (a server-side
    chat template, the route's special tokens, the probe's text) cancels, so the delta reports the media
    block alone -- the same thing :func:`content_media_tokens` counts. ``None`` when the engine counts
    exactly what the client counted; the typed mismatch otherwise, which the caller records or raises: a
    mismatch means the engine's media handling is not what the declared policy and the counted tokens
    describe (a reconfigured engine, a mis-declared processor family), so every later count is suspect.

    Args:
        reported: The engine's media delta (its two prompt-token reports' difference).
        counted: The client's exact count of the prepared probe's media tokens.

    Returns:
        ``None`` when the two agree; the :class:`EngineMediaMismatch` when they do not.
    """
    if reported == counted:
        return None
    return EngineMediaMismatch(reported=reported, counted=counted)


class VideoPolicyError(DataError):
    """A video the declared frame policy refuses to show the judge (exit code 12, like every :class:`DataError`);
    the message names the fix."""


def sample_video_part(part: VideoPart, video: VideoPolicy | None) -> VideoPart:
    """*part* as the judge is shown it under the frame policy *video*.

    The one place a video's frames are chosen; token counting and the media
    preparation (:func:`rcp_ndcg.data.prepare.prepare_content`) both go through it,
    so the frames counted are the frames sent.

    * ``wire: frames`` -- the uniformly sampled subset of ``part.frames``, with
      ``frame_indices`` recording which source frames were kept. A container
      ``ref`` on the same part stays provenance and is not sent.
    * ``wire: video_url`` -- the container ``ref`` alone, unchanged. The engine
      decodes and samples it with its own video loader (engine-sampled).
    * **No frame policy** -- the part as it is: every frame, or the container at
      the engine's own defaults, which :func:`content_media_tokens` then refuses
      to count.

    Raises:
        VideoPolicyError: when the part lacks what its declared wire sends (frames
            for ``frames``, a container for ``video_url``); when it has fewer frames
            than ``num_frames`` (or, for a container, no recorded frame count); or
            when ``max_duration_s`` is declared and the video's duration exceeds it
            or was never recorded.
    """
    frame_policy = video
    if frame_policy is None:
        return part
    _check_duration(part, frame_policy)
    if frame_policy.wire == "video_url":
        if part.ref is None:
            raise VideoPolicyError(
                f"{part.frames[0].uri}: the video policy declares `wire: video_url`, but this video "
                "is a frame directory with no container. Declare `wire: frames` for frame-directory corpora."
            )
        _check_frame_count(part.ref.uri, part.ref.num_frames, frame_policy)
        return VideoPart(ref=part.ref)
    if not part.frames:
        assert part.ref is not None
        raise VideoPolicyError(
            f"{part.ref.uri}: the video policy declares `wire: frames`, but this video is a "
            "container with no extracted frames. Declare `wire: video_url` to have the engine decode it, or "
            "ingest the clip as frames (the `frames` reader)."
        )
    source = part.frame_indices if part.frame_indices is not None else list(range(len(part.frames)))
    if len(source) != len(part.frames):
        raise DataError(
            f"{part.frames[0].uri}: the video records {len(source)} frame_indices for {len(part.frames)} frames; "
            "they must align"
        )
    _check_frame_count(part.frames[0].uri.rsplit("/", 1)[0], len(part.frames), frame_policy)
    assert frame_policy.num_frames is not None  # the validator requires it for `wire: frames`
    keep = uniform_frame_indices(len(part.frames), frame_policy.num_frames)
    return VideoPart(frames=[part.frames[i] for i in keep], frame_indices=[source[i] for i in keep])


def _check_duration(part: VideoPart, frame_policy: VideoPolicy) -> None:
    limit = frame_policy.max_duration_s
    if limit is None:
        return
    duration = part.ref.duration_s if part.ref is not None else None
    where = part.ref.uri if part.ref is not None else (part.frames[0].uri if part.frames else "<empty>")
    if duration is None:
        raise VideoPolicyError(
            f"{where}: the video policy declares `max_duration_s: {limit:g}`, but this video's "
            "duration was never recorded. Ingest containers with `hash_media=True` so the header is probed, "
            "or drop `max_duration_s`."
        )
    if duration > limit:
        raise VideoPolicyError(
            f"{where}: {duration:.1f}s exceeds the declared `max_duration_s: {limit:g}`. "
            f"The declared frame policy ({frame_policy.descriptor}) over this clip would be sparser than the "
            "corpus declared; raise the limit knowingly, or split the clip at ingest."
        )


def _check_frame_count(where: str, available: int | None, frame_policy: VideoPolicy) -> None:
    """Refuse a clip that cannot supply the declared number of frames, or whose count is unknown.

    Under the engine's fps rule (:attr:`VideoPolicy.fps`) the realised count is per clip, so only the
    recorded count is required -- :func:`_video_frame_indices` computes it from the clip's own metadata."""
    if available is None:
        raise VideoPolicyError(
            f"{where}: the frame policy samples the engine's rule, but this container's frame count was never "
            "recorded, so it cannot be checked. Ingest with `hash_media=True` (MP4, MOV and AVI headers are "
            "read; re-encode WebM/MKV to MP4)."
        )
    if frame_policy.num_frames is None:
        return
    wanted = frame_policy.num_frames
    if available < wanted:
        raise VideoPolicyError(
            f"{where}: {available} frames, fewer than the declared `num_frames: {wanted}`. A short clip "
            "is not shown whole: the engines disagree about it, so its judgement would not be the one "
            "recorded. Lower num_frames knowingly, or drop the clip at ingest."
        )


def content_media_tokens(
    content: Content,
    image: ImagePolicy,
    video: VideoPolicy | None = None,
    *,
    tokenizer: TextTokenizer | None = None,
) -> MediaTokenCount:
    """What *content*'s media costs the prompt, as the engine counts it, under the image and frame policies.

    Each image costs its merged patch tokens plus the family's vision start and end markers
    (:data:`VISION_WRAPPER_TOKENS`); a sampled frame is its own image and costs its own wrapper; a container
    is the engine's own video accounting (never the image policy's -- the container is sent unchanged, so
    the client's pixel budget never reaches the engine): ``ceil(num_frames / temporal_patch)`` per-frame
    token runs under the family's video budget (:data:`PROCESSORS`) -- each frame sized independently for
    the Qwen2-VL families (stock vLLM's accounting), the whole clip budgeted together for ``qwen3_vl``,
    whose prompt renders one timestamp line and one vision block per temporal group inside the chat
    template's own vision pair. It uses each reference's recorded ``width`` / ``height`` where present --
    our own ingest records them, so a page corpus counts exactly -- and the family's budget ceiling where
    they are absent. It never fetches bytes: a preflight that downloaded the corpus to count it would cost
    more than the thing it is counting.

    A ``qwen3_vl`` container under the engine's fps rule (:attr:`VideoPolicy.fps`) is counted from the
    clip's recorded frame count and rate with :func:`qwen3_vl_video_frame_indices`, and its timestamp lines
    are counted EXACTLY when ``tokenizer`` is given (the engine tokenizes ``<{seconds:.1f} seconds>`` with
    the checkpoint's own tokenizer); without one the family's declared per-group bound is used and the
    count is a bound. The client passes its loaded tokenizer, so the count it gates with is exact.

    Videos are counted as shown (:func:`sample_video_part`, which refuses clips shorter than the frame
    budget), and ``bounded`` counts the references counted at a bound.

    Raises:
        VideoPolicyError: the video policy refuses a clip -- a container under the other wire, a clip
            shorter than the frame budget, over ``max_duration_s``, or one whose recorded frame count or
            rate the fps rule needs.
        ConfigError: for a container without a video policy -- the engine's own default sampling decides
            its cost, and nothing here can know it -- for an fps policy on a family whose fps rule is not
            ported, or for an image under a native policy or an unknown processor
            (:meth:`ImagePolicy.image_tokens`).
    """
    tokens = 0
    bounded = 0
    for part in content.parts:
        if isinstance(part, TextPart):
            continue
        if isinstance(part, ImagePart):
            cost, bound = _ref_tokens(part.ref, image)
            tokens, bounded = tokens + cost + VISION_WRAPPER_TOKENS, bounded + bound
            continue
        shown = sample_video_part(part, video)
        if shown.frames:
            for ref in shown.frames:
                cost, bound = _ref_tokens(ref, image)
                # each sampled frame is sent as its own image and gets its own vision block
                tokens, bounded = tokens + cost + VISION_WRAPPER_TOKENS, bounded + bound
            continue
        assert shown.ref is not None
        cost, bound = _container_tokens(shown.ref, image, video, tokenizer=tokenizer)
        tokens, bounded = tokens + cost, bounded + bound
    return MediaTokenCount(tokens=tokens, bounded=bounded)


def _ref_tokens(ref: MediaRef, image: ImagePolicy) -> tuple[int, int]:
    """One image's patch tokens (no wrapper), and whether the count was a bound."""
    if ref.width and ref.height:
        return image.image_tokens(ref.height, ref.width), 0
    return image.max_image_tokens, 1


def _container_tokens(
    ref: MediaRef, image: ImagePolicy, video: VideoPolicy | None, *, tokenizer: TextTokenizer | None = None
) -> tuple[int, int]:
    """One container's prompt tokens, as a stock engine's video accounting counts them.

    The engine samples the container (a pinned uniform count, or -- for ``qwen3_vl`` under :attr:`fps` --
    its own fps rule, :func:`qwen3_vl_video_frame_indices`) and patchifies it in time, so
    ``ceil(frames / temporal_patch)`` per-frame token runs are shown, not ``frames``. The frames' geometry
    is the family's own video budget (:data:`PROCESSORS`), never the image policy's -- the container is
    sent unchanged, so the client's pixel budget never reaches the engine:

    * the Qwen2-VL families size each frame independently by the checkpoint's per-frame budget -- stock
      vLLM's accounting, which passes the image processor's size for videos (qwen2_vl.py:1014 @
      d0d6e5f3a) -- under one vision block for the whole clip.
    * ``qwen3_vl`` constrains the whole clip (a clip-level budget that shrinks per-frame resolution as the
      frame count grows) and renders one timestamp line and one vision block per temporal group, inside the
      chat template's own vision pair around the video placeholder. The timestamp lines are counted exactly
      when ``tokenizer`` is given, else at the family's declared bound (and the count is a bound).
    """
    if video is None:
        raise ConfigError(
            f"{ref.uri} is a video container, but no video policy is declared, so the engine's default sampling "
            "would decide how many frames the judge sees. Declare `preprocessing.video: {num_frames: N, wire: "
            "video_url}`."
        )
    if image.processor is None:
        raise ConfigError(
            f"cannot count the tokens of the container {ref.uri}: the image policy is native or its processor "
            "is unknown, so the engine's video processor decides the geometry, and nothing here can count it.",
            hint=f"set image_processor in the judge config (one of {', '.join(PROCESSORS)})",
        )
    geometry = PROCESSORS[image.processor]
    assert geometry.video_min_pixels is not None and geometry.video_max_pixels is not None
    if geometry.video_pixels_per_clip:
        # the clip-level budget constrains all frames together and shrinks per-frame resolution as the
        # frame count grows; the prompt renders one timestamp line and one vision block per group, inside
        # the chat template's own vision pair
        indices = _video_frame_indices(video, ref) if video.fps is not None else None
        frames = len(indices) if indices is not None else video.num_frames
        assert frames is not None  # the policy validator refuses neither rule
        if indices is not None:
            timestamps, timestamp_bound = _container_timestamps(ref, indices, geometry, tokenizer)
        else:
            steps = math.ceil(frames / geometry.temporal_patch)
            timestamps, timestamp_bound = [geometry.video_timestamp_tokens] * steps, 1
        if ref.width and ref.height:
            height, width = _clip_frame_size(geometry, frames, ref.height, ref.width)
            per_group = (height // geometry.factor) * (width // geometry.factor)
            return (
                VISION_WRAPPER_TOKENS + sum(ts + VISION_WRAPPER_TOKENS + per_group for ts in timestamps),
                timestamp_bound,
            )
        # no recorded size: the per-clip ceiling bounds the whole clip's patch tokens (each merged token
        # covers temporal_patch x factor^2 pixels), plus each group's wrapper and timestamp
        clip_bound = geometry.video_max_pixels // (geometry.temporal_patch * geometry.factor**2)
        return (
            VISION_WRAPPER_TOKENS + clip_bound + sum(ts + VISION_WRAPPER_TOKENS for ts in timestamps),
            1,
        )
    if video.num_frames is None:
        raise ConfigError(
            f"cannot count the container {ref.uri}: the video policy samples at {video.fps:g} fps, and only "
            "the qwen3_vl family's fps rule is ported (the Qwen3-VL backend's clamp). Declare `num_frames` for "
            "this processor family, or pin the engine to a uniform count.",
            hint="the fps rule ported here is qwen3_vl_video_frame_indices; other families' fps sampling is not ported",
        )
    steps = math.ceil(video.num_frames / geometry.temporal_patch)
    per_frame, bound = _video_frame_tokens(ref, geometry)
    return steps * per_frame + VISION_WRAPPER_TOKENS, bound


def _container_timestamps(
    ref: MediaRef,
    indices: list[int],
    geometry: ProcessorGeometry,
    tokenizer: TextTokenizer | None,
) -> tuple[list[int], int]:
    """The tokens of the timestamp line the engine renders before each temporal group's vision block.

    A faithful port of vLLM's ``Qwen3VLMultiModalProcessor.get_video_repl``
    (qwen3_vl.py:1620-1700 at the tag): the sampled indices are padded to the temporal patch, each group's
    timestamp is the mean of its pair of source-frame seconds, and the line ``<{seconds:.1f} seconds>`` is
    tokenized independently. With a tokenizer the count is exact; without one the family's declared bound
    is returned and the caller records the count as a bound.
    """
    assert ref.fps is not None  # _video_frame_indices required it for the fps rule
    padded = list(indices)
    if len(padded) % geometry.temporal_patch:
        padded += [padded[-1]] * (geometry.temporal_patch - len(padded) % geometry.temporal_patch)
    seconds = [index / ref.fps for index in padded]
    groups = [
        (seconds[i] + seconds[i + geometry.temporal_patch - 1]) / 2
        for i in range(0, len(seconds), geometry.temporal_patch)
    ]
    if tokenizer is None:
        return [geometry.video_timestamp_tokens] * len(groups), 1
    return [len(tokenizer.ids(f"<{group:.1f} seconds>", add_special_tokens=False)) for group in groups], 0


def _video_frame_tokens(ref: MediaRef, geometry: ProcessorGeometry) -> tuple[int, int]:
    """One container frame's patch tokens under the family's per-frame video budget, as the engine's
    accounting sizes it (a faithful port of the pinned vLLM ``Qwen2VLProcessingInfo._get_vision_info``,
    qwen2_vl.py:989-1053 @ d0d6e5f3a, whose default size is the checkpoint's image-processor size).

    Returns:
        ``(tokens, bound)``: the frame's merged patch tokens, and whether it was a bound (the reference's
        size was never recorded, so the budget's ceiling bounds it).

    Raises:
        DataError: an aspect ratio above 200, which the video processors refuse.
    """
    assert geometry.video_min_pixels is not None and geometry.video_max_pixels is not None
    if ref.width and ref.height:
        try:
            height, width = smart_resize(
                ref.height,
                ref.width,
                factor=geometry.factor,
                min_pixels=geometry.video_min_pixels,
                max_pixels=geometry.video_max_pixels,
            )
        except (ValueError, DataError) as exc:  # the frame's aspect ratio is one the video processors refuse
            raise DataError(
                f"a {ref.width}x{ref.height} video frame has an aspect ratio the video processors refuse "
                "(above 200); crop or split the clip at ingest",
                hint="crop or split the clip at ingest so its aspect ratio is below 200",
            ) from exc
        return (height // geometry.factor) * (width // geometry.factor), 0
    assert geometry.video_max_pixels is not None
    return geometry.video_max_pixels // geometry.factor**2, 1


def _clip_frame_size(geometry: ProcessorGeometry, num_frames: int, height: int, width: int) -> tuple[int, int]:
    """The per-frame size the family's video processor picks for a whole clip of ``num_frames`` frames.

    The family's per-clip pixel budget (:attr:`ProcessorGeometry.video_min_pixels`,
    :attr:`ProcessorGeometry.video_max_pixels`) constrains all frames together, so the per-frame resolution
    shrinks as the frame count grows. A faithful port of transformers' Qwen3-VL video ``smart_resize``
    (models/qwen3_vl/video_processing_qwen3_vl.py:75-109 @ 2c8526d; Apache-2.0, see NOTICE): checked against
    the real ``Qwen3VLVideoProcessor`` with 0 mismatches over 65 frame-count and size combinations.

    Raises:
        DataError: an aspect ratio above 200, which the video processor refuses.
    """
    factor = geometry.factor
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height, width = int(height * scale), int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise DataError(
            f"a {num_frames}-frame clip with {height}x{width} frames has an aspect ratio the video processor "
            "refuses (above 200); crop or split the clip at ingest",
            hint="crop or split the clip at ingest so its aspect ratio is below 200",
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    assert geometry.video_min_pixels is not None and geometry.video_max_pixels is not None
    t_bar = round(num_frames / geometry.temporal_patch) * geometry.temporal_patch
    if t_bar * h_bar * w_bar > geometry.video_max_pixels:
        beta = math.sqrt((num_frames * height * width) / geometry.video_max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < geometry.video_min_pixels:
        beta = math.sqrt(geometry.video_min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


__all__ = [
    "PROCESSORS",
    "VISION_WRAPPER_TOKENS",
    "EngineMediaMismatch",
    "ImagePolicy",
    "ImageProcessor",
    "MediaTokenCount",
    "ProcessorGeometry",
    "VideoPolicy",
    "VideoPolicyError",
    "content_media_tokens",
    "engine_media_check",
    "qwen3_vl_video_frame_indices",
    "sample_video_part",
    "smart_resize",
    "uniform_frame_indices",
]
