"""HuggingFace datasets through ``datasets.load_dataset``, image columns included.

MTEB mirrors, ViDoRe, and anything else on the Hub.  Corpus-shaped.

Text and image columns are resolved independently and either may be absent, so a text corpus, a page-image corpus
(a ViDoRe split has an ``image`` column of PIL objects and no text at all) and an interleaved one all read through
one path.

Image columns arrive as decoded PIL objects, which is the wrong shape for a
reference-based wire format.  They are written out once to a content-addressed
sidecar directory and referenced by URI + hash, so the rest of the pipeline
never holds a decoded corpus in memory and the images are reusable by anything
that can read a file.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from rcp_ndcg_core._records import ID, Document, Query
from rcp_ndcg_core.content import Content, ImagePart, MediaRef, TextPart

from rcp_ndcg import storage
from rcp_ndcg.data.io.base import DataShape, SourceReader, grade, join_title
from rcp_ndcg.data.media import store_media
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

ID_COLUMNS = ("_id", "id", "corpus-id", "corpus_id", "doc_id", "document_id", "idx", "index")
TEXT_COLUMNS = ("text", "content", "sentence", "document", "body", "markdown", "doc", "passage")
TITLE_COLUMNS = ("title", "document_title", "doc_title", "corpus_title", "name", "heading")
IMAGE_COLUMNS = ("image", "images", "page_image", "img", "picture", "screenshot")
QUERY_COLUMNS = ("query", "text", "question", "anchor")

DocumentParts = Literal["auto", "text", "image", "both"]
DOCUMENT_PARTS: tuple[DocumentParts, ...] = ("auto", "text", "image", "both")


class HfReader(SourceReader):
    """Reads a HuggingFace dataset into the canonical types.

    Args:
        uri: The Hub repo id or local dataset path.
        corpus_split: Split holding the documents.
        queries_split: Split holding the queries, if any.
        qrels_split: Split holding the relevance judgements, if any.
        config: The dataset config name (HF ``name`` argument).
        media_out_uri: Where decoded images are written. Required to read an
            image column, because there is nowhere else for the bytes to go.
        document_parts: Which of a multi-column corpus's columns to read.
            ``auto`` takes whatever is there. The explicit values matter for
            corpora like ViDoRe v3 that ship OCR text *and* page images in one
            row: ``image`` is a visual retrieval run, ``text`` is the OCR
            baseline, and they are different experiments over the same corpus,
            not a detail of how the file is parsed.
        name: Dataset name; defaults to the repo id's last segment.
    """

    name = "hf"
    shapes = frozenset({DataShape.CORPUS, DataShape.RANKING})

    def __init__(
        self,
        uri: str,
        *,
        corpus_split: str = "corpus",
        queries_split: str | None = "queries",
        qrels_split: str | None = "qrels",
        config: str | None = None,
        media_out_uri: str | None = None,
        document_parts: DocumentParts = "auto",
        name: str | None = None,
        **load_kwargs: Any,
    ) -> None:
        if document_parts not in DOCUMENT_PARTS:
            raise ConfigError(f"document_parts must be one of {DOCUMENT_PARTS}, got {document_parts!r}")
        self.uri = uri
        self.corpus_split = corpus_split
        self.queries_split = queries_split
        self.qrels_split = qrels_split
        self.config = config
        self.media_out_uri = media_out_uri
        self.document_parts: DocumentParts = document_parts
        self.dataset_name = name or uri.rstrip("/").rsplit("/", 1)[-1]
        self.load_kwargs = load_kwargs

    # -- loading -----------------------------------------------------------
    def _split(self, split: str):
        from datasets import load_dataset

        kwargs = dict(self.load_kwargs)
        if self.config:
            kwargs["name"] = self.config
        return load_dataset(self.uri, split=split, **kwargs)

    def _columns(self) -> tuple[str | None, str | None, str | None]:
        """The (text, image, title) columns this reader will actually read.

        One resolution shared by :meth:`documents` and :meth:`meta`: a corpus read
        as images while its metadata claims multimodal is exactly the "metadata
        that disagrees with the data" failure the reader contract tests for.
        """
        split = self._split(self.corpus_split)
        columns = list(split.column_names)
        text_column = _first(columns, TEXT_COLUMNS)
        image_column = _first(columns, IMAGE_COLUMNS)
        title_column = _first(columns, TITLE_COLUMNS)

        if self.document_parts in {"text", "both"} and text_column is None:
            raise DataError(
                f"document_parts={self.document_parts!r} needs a text column, but "
                f"{self.uri}[{self.corpus_split}] has none. Columns: {columns}; recognised: {TEXT_COLUMNS}."
            )
        if self.document_parts in {"image", "both"} and image_column is None:
            raise DataError(
                f"document_parts={self.document_parts!r} needs an image column, but "
                f"{self.uri}[{self.corpus_split}] has none. Columns: {columns}; recognised: {IMAGE_COLUMNS}."
            )
        if self.document_parts == "text":
            image_column = None
        elif self.document_parts == "image":
            text_column, title_column = None, None
        return text_column, image_column, title_column

    # -- corpus ------------------------------------------------------------
    def documents(self) -> Iterator[Document]:
        split = self._split(self.corpus_split)
        columns = list(split.column_names)
        id_column = _first(columns, ID_COLUMNS)
        text_column, image_column, title_column = self._columns()

        if id_column is None:
            raise DataError(
                f"{self.uri}[{self.corpus_split}] has no id column. Columns: {columns}. Recognised: {ID_COLUMNS}."
            )
        if text_column is None and image_column is None:
            raise DataError(
                f"{self.uri}[{self.corpus_split}] has neither a text nor an image column. "
                f"Columns: {columns}. Recognised text: {TEXT_COLUMNS}; images: {IMAGE_COLUMNS}."
            )
        if image_column is not None and self.media_out_uri is None:
            raise ConfigError(
                f"{self.uri}[{self.corpus_split}] has an image column ({image_column!r}) but no "
                "`media_out_uri`. Decoded images need a destination so they can be referenced by URI "
                "rather than held in memory."
            )

        seen: set[str] = set()
        for row in split:
            doc_id = str(row[id_column])
            if doc_id in seen:
                raise DataError(f"duplicate document id {doc_id!r} in {self.uri}[{self.corpus_split}]")
            seen.add(doc_id)
            parts = []
            text = _text_of(row, text_column, title_column)
            if text:
                parts.append(TextPart(text=text))
            if image_column is not None:
                parts.extend(ImagePart(ref=ref) for ref in self._persist_images(row[image_column]))
            if not parts:
                continue
            yield Document(doc_id=doc_id, content=Content.from_parts(parts))

    def _persist_images(self, value: Any) -> list[MediaRef]:
        """Write decoded images out once and return references to them.

        Content-addressed: two rows carrying the same page write one file.
        """
        assert self.media_out_uri is not None
        images = value if isinstance(value, list) else [value]
        refs: list[MediaRef] = []
        for image in images:
            if image is None:
                continue
            payload, extension = _encode_hf_image(image)
            width, height = getattr(image, "width", None), getattr(image, "height", None)
            refs.append(store_media(payload, extension, root=self.media_out_uri, width=width, height=height))
        return refs

    def queries(self) -> Iterator[Query]:
        if self.queries_split is None:
            return
        split = self._split(self.queries_split)
        columns = list(split.column_names)
        id_column = _first(columns, ("_id", "id", "query-id", "query_id", "qid"))
        text_column = _first(columns, QUERY_COLUMNS)
        if id_column is None or text_column is None:
            raise DataError(f"{self.uri}[{self.queries_split}] needs an id and a text column. Columns: {columns}.")
        for row in split:
            yield Query(
                query_id=str(row[id_column]),
                query=row[text_column] or "",
                instruction=row.get("instruction") if hasattr(row, "get") else None,
            )

    def qrels(self) -> dict[ID, dict[ID, float]]:
        if self.qrels_split is None:
            return {}
        try:
            split = self._split(self.qrels_split)
        except Exception as exc:  # noqa: BLE001 - an absent split is normal, not a defect
            logger.debug(f"no qrels split {self.qrels_split!r} for {self.uri}: {exc}")
            return {}
        columns = list(split.column_names)
        query_column = _first(columns, ("query-id", "query_id", "qid", "_id"))
        doc_column = _first(columns, ("corpus-id", "corpus_id", "doc_id", "docid"))
        score_column = _first(columns, ("score", "relevance", "label", "grade"))
        if query_column is None or doc_column is None:
            raise DataError(f"{self.uri}[{self.qrels_split}] needs query and corpus id columns. Columns: {columns}.")
        out: dict[ID, dict[ID, float]] = {}
        source = f"{self.uri}[{self.qrels_split}]"
        for row in split:
            label = grade(row[score_column], source=source) if score_column else 1.0
            out.setdefault(str(row[query_column]), {})[str(row[doc_column])] = label
        return out


def _first(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    return next((candidate for candidate in candidates if candidate in columns), None)


def _text_of(row: Any, text_column: str | None, title_column: str | None) -> str:
    """A document's full text (title and body); text is shortened only by the judging pass, never at load."""
    if text_column is None:
        return ""
    value = row[text_column]
    if isinstance(value, list):
        # The fused part-list layout: a list of ``{"type": ..., "text": ...}`` parts.
        value = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in value
            if not isinstance(part, dict) or part.get("type") == "text"
        )
    return join_title(row[title_column] if title_column is not None else None, value)


def _encode_hf_image(image: Any) -> tuple[bytes, str]:
    """Bytes for a HF image cell, preferring the undecoded original.

    ``datasets`` hands back either a PIL image or a ``{"bytes", "path"}`` dict.
    Using the dict's bytes when present avoids a decode/re-encode round trip,
    which would change the hash of a byte-identical page.
    """
    if isinstance(image, dict):
        payload = image.get("bytes")
        if payload:
            path = str(image.get("path") or "")
            extension = f".{path.rsplit('.', 1)[-1].lower()}" if "." in path else ".png"
            return payload, extension
        if image.get("path"):
            return storage.read_bytes(str(image["path"])), f".{str(image['path']).rsplit('.', 1)[-1].lower()}"
        raise DataError(f"image cell has neither bytes nor a path: {sorted(image)}")

    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), ".png"


__all__ = [
    "DOCUMENT_PARTS",
    "IMAGE_COLUMNS",
    "TEXT_COLUMNS",
    "DocumentParts",
    "HfReader",
]
