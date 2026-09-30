"""A directory of images as a document corpus.

The simplest possible image dataset, and the one that makes an ad-hoc collection
of page renders immediately evaluable: point at a directory, get a corpus of
one-image documents whose ids are the filenames.

Queries and qrels come from optional sidecars, because a directory of images has
no way to express them.  With no sidecar this reader yields a corpus and no
queries -- honest, and still useful for building an index.

Hashing is opt-in.  Hashing 40k page images costs a full read of the corpus, so
it is a deliberate ingest-time decision rather than something that happens
silently on first use; without it, cache keys fall back to the URI, which cannot
notice the file changing underneath.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

from rcp_ndcg_core._records import ID, Document, Query
from rcp_ndcg_core.content import Content, ImagePart, MediaRef, Modality, Part

from rcp_ndcg import storage
from rcp_ndcg.data.io.base import DataShape, SourceReader, grade, iter_json_lines
from rcp_ndcg.data.media import default_resolver
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)


MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}


class ImageDirReader(SourceReader):
    """Reads a directory of images as one-image documents.

    Args:
        uri: The image directory. Searched recursively.
        queries_uri: Optional JSONL of ``{"query_id", "text"}`` records.
        qrels_uri: Optional JSONL of ``{"query_id", "qrels": {...}}`` records.
        hash_media: Read every image to record its ``sha256`` and dimensions.
            Off by default because it costs a full pass over the corpus.
        name: Dataset name; defaults to the directory name.
    """

    name = "images"
    shapes = frozenset({DataShape.CORPUS, DataShape.RANKING})

    #: The file suffixes read as media, and the MIME type each is recorded with.
    mime_by_suffix: ClassVar[dict[str, str]] = MIME_BY_SUFFIX
    #: What every document of this reader is made of.
    modality: ClassVar[Modality] = Modality.IMAGE

    def __init__(
        self,
        uri: str,
        *,
        queries_uri: str | None = None,
        qrels_uri: str | None = None,
        hash_media: bool = False,
        name: str | None = None,
    ) -> None:
        self.uri = str(uri).rstrip("/")
        self.queries_uri = queries_uri
        self.qrels_uri = qrels_uri
        self.hash_media = hash_media
        self.dataset_name = name or self.uri.rsplit("/", 1)[-1]

    def documents(self) -> Iterator[Document]:
        for uri in self._media_uris():
            doc_id = _doc_id_for(uri, self.uri, tuple(self.mime_by_suffix))
            yield Document(doc_id=doc_id, content=Content.from_parts([self._part(self._ref(uri))]))

    def _ref(self, uri: str) -> MediaRef:
        """One file as a reference, hashed and probed when ``hash_media`` is set."""
        ref = MediaRef(uri=uri, mime=_mime_for(uri, self.mime_by_suffix))
        return default_resolver().hydrate(ref) if self.hash_media else ref

    def _part(self, ref: MediaRef) -> Part:
        """The part one media file becomes."""
        return ImagePart(ref=ref)

    def queries(self) -> Iterator[Query]:
        if self.queries_uri is None:
            return
        for row in iter_json_lines(self.queries_uri):
            query_id = row.get("query_id") or row.get("_id") or row.get("id")
            if query_id is None:
                continue
            yield Query(
                query_id=str(query_id),
                query=row.get("text") or row.get("query") or "",
                instruction=row.get("instruction"),
            )

    def qrels(self) -> dict[ID, dict[ID, float]]:
        if self.qrels_uri is None:
            return {}
        out: dict[ID, dict[ID, float]] = {}
        for row in iter_json_lines(self.qrels_uri):
            query_id = row.get("query_id") or row.get("_id") or row.get("id")
            if query_id is None:
                continue
            judged = row.get("qrels") or {}
            out.setdefault(str(query_id), {}).update(
                {str(k): grade(v, source=self.qrels_uri) for k, v in judged.items()}
            )
        return out

    def _media_uris(self) -> Iterator[str]:
        suffixes = tuple(self.mime_by_suffix)
        for entry in sorted(storage.ls(self.uri, recursive=True)):
            if entry.lower().endswith(suffixes):
                yield entry


def _doc_id_for(uri: str, root: str, suffixes: tuple[str, ...]) -> str:
    """Path relative to the root, without the suffix.

    Relative rather than the bare filename so ``doc_a/page_1.png`` and
    ``doc_b/page_1.png`` do not collide -- which is exactly the layout a
    per-document PDF render produces.
    """
    relative = uri[len(root) :].lstrip("/") if uri.startswith(root) else uri.rsplit("/", 1)[-1]
    for suffix in suffixes:
        if relative.lower().endswith(suffix):
            return relative[: -len(suffix)]
    return relative


def _mime_for(uri: str, mime_by_suffix: dict[str, str] = MIME_BY_SUFFIX) -> str | None:
    lowered = uri.lower()
    for suffix, mime in mime_by_suffix.items():
        if lowered.endswith(suffix):
            return mime
    return None


__all__ = ["ImageDirReader"]
