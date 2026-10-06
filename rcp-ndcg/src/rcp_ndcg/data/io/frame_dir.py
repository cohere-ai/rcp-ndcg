"""Clip-level frame directories as a document corpus: one directory of frames per document.

The layout pre-extracted video benchmarks ship in -- every clip a directory of
JPEG/PNG frames -- read so that each clip becomes one
``VideoPart(frames=[...], frame_indices=[...])``::

    root/
      movie_a/clip_001/000001.jpg 000002.jpg ...   -> document "movie_a/clip_001"
      movie_a/clip_002/...                         -> document "movie_a/clip_002"

A clip is any directory that directly holds frame images; its id is that
directory's path relative to the root. Frames are ordered by the numbers in
their file names (``frame_2`` before ``frame_10``) and ``frame_indices`` records
each frame's position in that order, so after sampling the prompt still says
which frames of the clip were shown.

This is the video shape every OpenAI-compatible endpoint can judge: the sampled
frames are sent as images, and no video decoder is involved anywhere.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import ClassVar

from rcp_ndcg_core._records import Document
from rcp_ndcg_core.content import Content, Modality, VideoPart

from rcp_ndcg import storage
from rcp_ndcg.data.io.image_dir import ImageDirReader
from rcp_ndcg.errors import DataError

_NUMBER = re.compile(r"(\d+)")


def _natural_key(uri: str) -> list[int | str]:
    """``frame_2`` sorts before ``frame_10``."""
    return [int(chunk) if chunk.isdigit() else chunk for chunk in _NUMBER.split(uri)]


class FrameDirReader(ImageDirReader):
    """Reads a tree of per-clip frame directories as one-clip documents.

    Takes the same arguments as :class:`~rcp_ndcg.data.io.image_dir.ImageDirReader`;
    ``hash_media`` hashes and sizes every frame. ``frame_indices`` records each frame's number in
    its file name -- which frame of the source it was sampled at -- not its position in the directory.
    """

    name = "frames"
    modality: ClassVar[Modality] = Modality.VIDEO

    def documents(self) -> Iterator[Document]:
        for clip_id, uris in self._clips():
            frames = [self._ref(uri) for uri in uris]
            part = VideoPart(frames=frames, frame_indices=[_frame_number(uri) for uri in uris])
            yield Document(doc_id=clip_id, content=Content.from_parts([part]))

    def _clips(self) -> Iterator[tuple[str, list[str]]]:
        """``(clip id, frame uris in order)`` for every directory that holds frames."""
        clips: dict[str, list[str]] = {}
        for uri in self._media_uris():
            clip_id, _, _ = storage.relative(uri, self.uri).rpartition("/")
            if not clip_id:
                raise DataError(
                    f"{uri} sits directly in the frame root {self.uri}; a frame directory corpus holds one "
                    "directory per clip. Move each clip's frames into their own directory."
                )
            clips.setdefault(clip_id, []).append(uri)
        for clip_id in sorted(clips):
            yield clip_id, sorted(clips[clip_id], key=_natural_key)


def _frame_number(uri: str) -> int:
    """The frame's number in its file name (``000007.jpg`` -> 7): what ``frame_indices`` records.

    Raises:
        DataError: the file name holds no number -- the record could not say which frame it is.
    """
    stem = uri.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    match = _NUMBER.search(stem)
    if match is None:
        raise DataError(f"{uri}: a frame file name carries no number; frame_indices cannot name the source frame")
    return int(match.group(1))


__all__ = ["FrameDirReader"]
