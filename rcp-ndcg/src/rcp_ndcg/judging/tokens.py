"""The token approximations for estimates: text without a tokenizer, images without an image processor.

Text limits and the window budget are counted exactly with the judge's tokenizer
(:mod:`rcp_ndcg.data.tokenizer`); nothing is ever cut by this approximation. It serves only
:func:`~rcp_ndcg.judging.cost.estimate` (and the offline fake judge's usage) when no tokenizer is at hand:
two characters per token. For Latin-script text that is usually conservative (English prose runs nearer
four), so an estimate errs towards more tokens and more cost; CJK text and code can exceed it, so it is a
heuristic, not a bound.

Images and video frames sent to a judge that declares no ``image_processor`` (a hosted API, which sizes them
itself) have no countable token cost. Estimates then assume :data:`APPROX_TOKENS_PER_IMAGE` per image or frame
shown, about one document page at a hosted API's high detail; a judging pass never budgets text on it.
Every number derived here is labelled as an approximation where it is reported.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rcp_ndcg_core.content import Content

    from rcp_ndcg.data.resolution import VideoPolicy

#: Characters per token: a heuristic, usually conservative for Latin-script text (see the module docstring).
_CHARS_PER_TOKEN = 2.0

#: How estimates label numbers derived from :func:`approx_tokens`.
APPROXIMATION_NOTE = (
    f"approximate: {_CHARS_PER_TOKEN:g} characters per token, a heuristic, usually conservative for Latin-script text"
)


def approx_tokens(chars: int | float) -> int:
    """Approximate tokens for ``chars`` characters of text (rounded up; see the module docstring)."""
    if chars <= 0:
        return 0
    return math.ceil(chars / _CHARS_PER_TOKEN)


#: Tokens assumed per image or video frame when the judge declares no ``image_processor`` (estimates only).
APPROX_TOKENS_PER_IMAGE = 1_000

#: How estimates label image tokens derived from :func:`approx_media_tokens`.
IMAGE_APPROXIMATION_NOTE = (
    f"approximate: {APPROX_TOKENS_PER_IMAGE:,} tokens per image or video frame, since the judge declares no "
    "image_processor and the engine sizes the images itself"
)


def approx_media_tokens(content: Content, video: VideoPolicy | None) -> int:
    """Approximate tokens of ``content``'s images and video frames as shown (:data:`APPROX_TOKENS_PER_IMAGE` each).

    A video counts the frames the frame policy shows (:func:`~rcp_ndcg.data.resolution.sample_video_part`); a
    container sent whole counts the policy's ``num_frames``.

    Raises:
        ConfigError: a video container without a frame policy (the engine's own sampling decides its frames).
    """
    from rcp_ndcg_core.content import ImagePart, VideoPart

    from rcp_ndcg.data.resolution import sample_video_part
    from rcp_ndcg.errors import ConfigError

    shown = 0
    for part in content.parts:
        if isinstance(part, ImagePart):
            shown += 1
        elif isinstance(part, VideoPart):
            sampled = sample_video_part(part, video)
            if sampled.frames:
                shown += len(sampled.frames)
            elif video is not None:
                shown += video.num_frames
            else:
                raise ConfigError(
                    "a video container without a video policy: the engine's default sampling decides its frames",
                    hint="declare preprocessing.video: {num_frames: N, wire: video_url}",
                )
    return shown * APPROX_TOKENS_PER_IMAGE


__all__ = [
    "APPROXIMATION_NOTE",
    "APPROX_TOKENS_PER_IMAGE",
    "IMAGE_APPROXIMATION_NOTE",
    "approx_media_tokens",
    "approx_tokens",
]
