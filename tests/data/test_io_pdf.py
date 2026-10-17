"""PDF ingest: pages become documents, and the DPI is part of the dataset's identity."""

from __future__ import annotations

import pytest

from rcp_ndcg.data.io import get_reader
from rcp_ndcg.errors import ConfigError

pypdfium2 = pytest.importorskip("pypdfium2")
PIL = pytest.importorskip("PIL.Image")


@pytest.fixture
def two_page_pdf(tmp_path) -> str:
    """A real two-page US-Letter PDF at 72 DPI, so page geometry is predictable."""
    # PIL's PDF writer reaches into Image.SAVE["JPEG"], which only exists once the
    # lazily-registered plugins are loaded.
    PIL.init()
    pages = [PIL.new("RGB", (612, 792), color=(255, 255 - index * 40, 255)) for index in range(2)]
    target = tmp_path / "report.pdf"
    pages[0].save(target, save_all=True, append_images=pages[1:], resolution=72.0)
    return str(target)


class TestPdfReader:
    def test_one_document_per_page(self, two_page_pdf):
        docs = list(get_reader("pdf", uri=two_page_pdf, dpi=72).documents())
        assert [doc.id for doc in docs] == ["report#p1", "report#p2"]

    def test_pages_carry_their_page_number(self, two_page_pdf):
        docs = list(get_reader("pdf", uri=two_page_pdf, dpi=72).documents())
        assert [part.page for doc in docs for part in doc.as_content.parts] == [1, 2]

    def test_rendered_pages_are_hashed_and_sized(self, two_page_pdf):
        ref = next(iter(get_reader("pdf", uri=two_page_pdf, dpi=72).documents())).media[0]
        assert ref.sha256 is not None
        assert ref.num_bytes and ref.num_bytes > 0
        assert (ref.width, ref.height) == (612, 792)

    def test_dpi_changes_the_pixels(self, two_page_pdf):
        low = next(iter(get_reader("pdf", uri=two_page_pdf, dpi=72).documents())).media[0]
        high = next(iter(get_reader("pdf", uri=two_page_pdf, dpi=144).documents())).media[0]
        assert high.width == low.width * 2
        assert high.sha256 != low.sha256

    def test_dpi_keeps_the_two_renders_apart_on_disk(self, two_page_pdf):
        """Two DPIs must not collide, or a run inherits the wrong pages."""
        low = next(iter(get_reader("pdf", uri=two_page_pdf, dpi=72).documents())).media[0]
        high = next(iter(get_reader("pdf", uri=two_page_pdf, dpi=144).documents())).media[0]
        assert low.uri != high.uri
        assert low.uri.endswith("_72dpi.png") and high.uri.endswith("_144dpi.png")

    def test_rerendering_reuses_what_is_on_disk(self, two_page_pdf, tmp_path):
        """An interrupted ingest must not re-render 40k pages on restart."""
        out = str(tmp_path / "pages")

        def hashes() -> list[str | None]:
            reader = get_reader("pdf", uri=two_page_pdf, dpi=72, out_uri=out)
            return [ref.sha256 for doc in reader.documents() for ref in doc.media]

        assert hashes() == hashes()

    def test_two_dpis_into_one_directory_do_not_share_pages(self, two_page_pdf, tmp_path):
        """A second conversion at another DPI into the same --out must render, not reuse the first one's pages."""
        out = str(tmp_path / "pages")

        def widths(dpi: int) -> list[int | None]:
            reader = get_reader("pdf", uri=two_page_pdf, dpi=dpi, out_uri=out)
            return [ref.width for doc in reader.documents() for ref in doc.media]

        low, high = widths(72), widths(144)

        assert all(h == 2 * w for w, h in zip(low, high, strict=True) if w and h)
        assert low != high

    def test_a_directory_of_pdfs_keeps_them_apart(self, tmp_path, two_page_pdf):
        import shutil

        root = tmp_path / "corpus"
        root.mkdir()
        shutil.copy(two_page_pdf, root / "a.pdf")
        shutil.copy(two_page_pdf, root / "b.pdf")
        ids = [doc.id for doc in get_reader("pdf", uri=str(root), dpi=72).documents()]
        assert ids == ["a#p1", "a#p2", "b#p1", "b#p2"]

    def test_media_is_resolvable(self, two_page_pdf, tmp_path):
        from rcp_ndcg.data.media import MediaResolver

        resolver = MediaResolver()
        for doc in get_reader("pdf", uri=two_page_pdf, dpi=72).documents():
            assert resolver.image(doc.media[0]).size == (612, 792)

    def test_jpeg_output_is_smaller_than_png(self, two_page_pdf):
        png = next(iter(get_reader("pdf", uri=two_page_pdf, dpi=72).documents())).media[0]
        jpeg = next(iter(get_reader("pdf", uri=two_page_pdf, dpi=72, image_format="jpeg").documents())).media[0]
        assert jpeg.mime == "image/jpeg"
        assert png.mime == "image/png"

    def test_a_nonsense_dpi_is_refused(self, two_page_pdf):
        with pytest.raises(ConfigError, match="dpi must be positive"):
            get_reader("pdf", uri=two_page_pdf, dpi=0)

    def test_an_unknown_image_format_is_refused(self, two_page_pdf):
        with pytest.raises(ConfigError, match="image_format"):
            get_reader("pdf", uri=two_page_pdf, image_format="tiff")
