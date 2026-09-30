"""PDFs rendered to page images at a recorded DPI.

One document per page, which is the unit visual retrieval actually retrieves.

**The DPI is the point.** A page rendered at 150 DPI and the same page at 300 DPI
are different images, and a model reads different amounts of small print off
them.  So the DPI is recorded in the metadata's provenance, and every rendered
page file is named with it (``page_0001_150dpi.png``) -- two ingests at different
DPIs into the same directory cannot collide, and a run cannot silently inherit
pages rendered for a different purpose.

Rendering needs ``pypdfium2``, which is not a core dependency: most users of this
package never touch a PDF, and the import error names the extra.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from rcp_ndcg_core._records import Document
from rcp_ndcg_core.content import Content, ImagePart, MediaRef

from rcp_ndcg import storage
from rcp_ndcg.data.io.base import DataShape, SourceReader, unique_document_ids
from rcp_ndcg.data.media import default_resolver, sha256_of
from rcp_ndcg.errors import ConfigError
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DPI = 150
#: pypdfium2 renders at a scale factor relative to 72 DPI (PDF user space).
PDF_BASE_DPI = 72


class PdfReader(SourceReader):
    """Renders PDFs to page images and yields one document per page.

    Args:
        uri: A single PDF, or a directory of PDFs (searched recursively).
        dpi: Render resolution (default :data:`DEFAULT_DPI`, 150); recorded in the
            metadata and in every page file name.
        out_uri: Where the page images are written; default ``<name>_pages``
            beside the source. Renders at different DPIs coexist in it.
        image_format: ``png`` (lossless, larger) or ``jpeg``.
        name: Dataset name; defaults to the source file or directory name.
    """

    name = "pdf"
    shapes = frozenset({DataShape.CORPUS})

    def __init__(
        self,
        uri: str,
        *,
        dpi: int = DEFAULT_DPI,
        out_uri: str | None = None,
        image_format: str = "png",
        name: str | None = None,
    ) -> None:
        if dpi <= 0:
            raise ConfigError(f"dpi must be positive, got {dpi}")
        if image_format not in {"png", "jpeg"}:
            raise ConfigError(f"image_format must be 'png' or 'jpeg', got {image_format!r}")
        self.uri = str(uri).rstrip("/")
        self.dpi = dpi
        self.image_format = image_format
        self.dataset_name = name or Path(self.uri).stem
        self.out_uri = out_uri or storage.join(storage.parent(self.uri), f"{self.dataset_name}_pages")
        self._rendered: dict[str, list[MediaRef]] = {}

    def documents(self) -> Iterator[Document]:
        """One document per page, ``<stem>#p<page>``; the stem is the PDF's path below the root, without ``.pdf``.

        Raises:
            DataError: Two PDFs give one stem (and would share a page directory).
        """
        stems = unique_document_ids(((_stem_for(uri, self.uri), uri) for uri in self._pdf_uris()), self.uri)
        for stem, pdf_uri in stems.items():
            for page_number, ref in enumerate(self._render(pdf_uri, stem), start=1):
                yield Document(
                    doc_id=f"{stem}#p{page_number}",
                    content=Content.from_parts([ImagePart(ref=ref, page=page_number)]),
                )

    def _pdf_uris(self) -> list[str]:
        if self.uri.lower().endswith(".pdf"):
            return [self.uri]
        return sorted(entry for entry in storage.ls(self.uri, recursive=True) if entry.lower().endswith(".pdf"))

    def _render(self, pdf_uri: str, stem: str) -> list[MediaRef]:
        """Render one PDF's pages, reusing any already on disk.

        Cached by output path: a re-run of an interrupted ingest must not
        re-render 40k pages, and the DPI is in the file name so the reuse is safe.
        """
        if pdf_uri in self._rendered:
            return self._rendered[pdf_uri]

        try:
            import pypdfium2
        except ImportError as exc:
            raise ImportError("PDF rendering needs pypdfium2: pip install 'rcp-ndcg[data]'") from exc

        local_pdf = storage.cache(pdf_uri)
        page_dir = storage.join(self.out_uri, stem)
        storage.makedirs(page_dir)
        refs: list[MediaRef] = []
        scale = self.dpi / PDF_BASE_DPI
        suffix = "png" if self.image_format == "png" else "jpg"

        document = pypdfium2.PdfDocument(str(local_pdf))
        try:
            for index in range(len(document)):
                # The DPI in the name is what keeps two renders of the same corpus apart.
                target = storage.join(page_dir, f"page_{index + 1:04d}_{self.dpi}dpi.{suffix}")
                if storage.exists(target):
                    refs.append(default_resolver().hydrate(MediaRef(uri=target, mime=f"image/{self.image_format}")))
                    continue
                # pypdfium2's `scale` is untyped with an int default; it takes a
                # float, which is the whole point of rendering at a chosen DPI.
                image = document[index].render(scale=scale).to_pil()  # type: ignore[arg-type]
                payload = _encode(image, self.image_format)
                storage.write_bytes(target, payload)
                refs.append(
                    MediaRef(
                        uri=target,
                        sha256=sha256_of(payload),
                        mime=f"image/{self.image_format}",
                        width=image.width,
                        height=image.height,
                        num_bytes=len(payload),
                    )
                )
        finally:
            document.close()

        logger.info(f"rendered {len(refs)} pages of {pdf_uri} at {self.dpi} DPI to {page_dir}")
        self._rendered[pdf_uri] = refs
        return refs


def _encode(image, image_format: str) -> bytes:
    import io

    buffer = io.BytesIO()
    if image_format == "jpeg":
        image.convert("RGB").save(buffer, format="JPEG", quality=92)
    else:
        image.save(buffer, format="PNG")
    return buffer.getvalue()


def _stem_for(pdf_uri: str, root: str) -> str:
    """The PDF's path below a directory root (its file name for a single PDF), without ``.pdf``."""
    relative = Path(pdf_uri).name if pdf_uri == root else storage.relative(pdf_uri, root)
    return relative[:-4] if relative.lower().endswith(".pdf") else relative


__all__ = ["DEFAULT_DPI", "PdfReader"]
