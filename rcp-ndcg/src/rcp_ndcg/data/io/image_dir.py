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

from rcp_ndcg_core.content import Content, ImagePart, MediaRef, Modality, Part
from rcp_ndcg_core.records import ID, Document, Query

from rcp_ndcg import storage
from rcp_ndcg.data.io.base import DataShape, SourceReader, required_id, sidecar_qrels, unique_document_ids
from rcp_ndcg.data.media import IMAGE_MIME_BY_SUFFIX, default_resolver
from rcp_ndcg.storage.io import numbered_json_lines
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)


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
    mime_by_suffix: ClassVar[dict[str, str]] = IMAGE_MIME_BY_SUFFIX
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
        """One document per image; its id is the file's path below the root, without the suffix.

        Raises:
            DataError: Two files give one id (``page_2.png`` and ``page_2.jpg``).
        """
        ids = unique_document_ids(
            ((_doc_id_for(uri, self.uri, tuple(self.mime_by_suffix)), uri) for uri in self._media_uris()), self.uri
        )
        for doc_id, uri in ids.items():
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
        for line_number, row in numbered_json_lines(self.queries_uri):
            query_id = required_id(
                row,
                ("query_id", "_id", "id"),
                source=f"{self.queries_uri}:{line_number}",
                what="a query row",
            )
            yield Query(
                query_id=str(query_id),
                query=row.get("text") or row.get("query") or "",
                instruction=row.get("instruction"),
            )

    def qrels(self) -> dict[ID, dict[ID, float]]:
        if self.qrels_uri is None:
            return {}
        return sidecar_qrels(numbered_json_lines(self.qrels_uri), source=self.qrels_uri)

    def _media_uris(self) -> Iterator[str]:
        suffixes = tuple(self.mime_by_suffix)
        for entry in sorted(storage.ls(self.uri, recursive=True)):
            if entry.lower().endswith(suffixes):
                yield entry


def _doc_id_for(uri: str, root: str, suffixes: tuple[str, ...]) -> str:
    """Path relative to the root (as :func:`rcp_ndcg.storage.relative` normalises both), without the suffix.

    Relative rather than the bare filename so ``doc_a/page_1.png`` and
    ``doc_b/page_1.png`` do not collide -- which is exactly the layout a
    per-document PDF render produces -- and the same whether the root is spelled
    relative or absolute.
    """
    relative = storage.relative(uri, root)
    for suffix in suffixes:
        if relative.lower().endswith(suffix):
            return relative[: -len(suffix)]
    return relative


def _mime_for(uri: str, mime_by_suffix: dict[str, str] = IMAGE_MIME_BY_SUFFIX) -> str | None:
    lowered = uri.lower()
    for suffix, mime in mime_by_suffix.items():
        if lowered.endswith(suffix):
            return mime
    return None


__all__ = ["ImageDirReader"]
