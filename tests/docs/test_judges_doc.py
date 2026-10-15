"""The judges page's per-request video limit can serve the shipped video schedules.

The page's example value is the one an operator copies into `--limit-mm-per-prompt`; every shipped video
window puts five documents in one request, so an example below five admits no window.
"""

from __future__ import annotations

import re

from rcp_ndcg.judging.schedule import RubricSchedule, TournamentSchedule
from tests.docs._markdown import ROOT, paragraph

PAGE = ROOT / "docs" / "concepts" / "judges.md"
VIDEO_LIMIT = re.compile(r'`\{"video": (\d+)\}`')


def test_the_video_limit_example_admits_a_video_window() -> None:
    """The example's value is at least every shipped video window, and the prose names the rule."""
    text = PAGE.read_text(encoding="utf-8")
    match = VIDEO_LIMIT.search(text)
    assert match, "the page states no per-request video limit example"
    limit = int(match.group(1))
    windows = {
        "tournament": TournamentSchedule.for_modality("video").window,
        "rubric": RubricSchedule.for_modality("video").window,
    }
    assert limit >= max(windows.values()), f"the example {limit} is below a video window {windows}"
    bullet = paragraph(text, "A `video_url` corpus")
    assert "max_videos" in bullet, "the client's own max_videos must allow the same count"
