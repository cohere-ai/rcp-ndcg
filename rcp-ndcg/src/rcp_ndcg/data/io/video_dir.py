"""A directory of video files as a document corpus: one clip per document.

The :mod:`~rcp_ndcg.data.io.image_dir` contract applied to containers -- ids are
paths relative to the root without the suffix, queries and qrels come from the
same optional sidecars, and hashing is opt-in. With ``hash_media=True`` each
container's header is also read (:func:`rcp_ndcg.data.media.probe_video_header`)
so its frame count, duration and size are recorded: that is what prices a clip
exactly and what the video policy's ``max_duration_s`` (``preprocessing.video``) is checked against.

Nothing is decoded here. A container is judged over ``wire: video_url``: it
travels unchanged as one ``video_url`` block, and the judge's engine decodes and
samples it. To have the client sample frames instead, ingest the clip as frames
(the ``frames`` reader).
"""

from __future__ import annotations

from typing import ClassVar

from rcp_ndcg_core.content import MediaRef, Modality, Part, VideoPart

from rcp_ndcg.data.io.image_dir import ImageDirReader
from rcp_ndcg.data.media import VIDEO_MIME_BY_SUFFIX


class VideoDirReader(ImageDirReader):
    """Reads a directory of video containers (MP4, MOV, M4V, WebM, MKV, AVI) as one-clip documents.

    Takes the same arguments as :class:`~rcp_ndcg.data.io.image_dir.ImageDirReader`.
    """

    name = "videos"
    mime_by_suffix: ClassVar[dict[str, str]] = VIDEO_MIME_BY_SUFFIX
    modality: ClassVar[Modality] = Modality.VIDEO

    def _part(self, ref: MediaRef) -> Part:
        return VideoPart(ref=ref)


__all__ = ["VideoDirReader"]
