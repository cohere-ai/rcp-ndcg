"""The HuggingFace reader, and choosing what to read from a multi-column corpus.

ViDoRe v3 is the case that forces the question: every row carries OCR ``markdown``
*and* a page ``image``. Reading the page instead of the OCR is a different
experiment over the same judgements, so it is declared rather than sniffed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rcp_ndcg.data.io.hf import HfReader
from rcp_ndcg.errors import ConfigError, DataError

datasets = pytest.importorskip("datasets")
PIL = pytest.importorskip("PIL.Image")


@pytest.fixture
def multimodal_corpus(tmp_path: Path):
    """Two pages that each carry OCR text and an image, as ViDoRe v3 does."""
    rows = {
        "_id": ["p0", "p1"],
        "markdown": ["the invoice total is 42", "appendix B"],
        "image": [PIL.new("RGB", (40, 60), "white"), PIL.new("RGB", (40, 60), "black")],
    }
    split = datasets.Dataset.from_dict(rows).cast_column("image", datasets.Image())
    path = tmp_path / "corpus"
    split.save_to_disk(str(path))
    return datasets.load_from_disk(str(path))


@pytest.fixture
def reader_factory(multimodal_corpus, tmp_path: Path, monkeypatch):
    def build(**kwargs) -> HfReader:
        reader = HfReader(
            uri="vidore/fake_v3",
            corpus_split="corpus",
            queries_split=None,
            qrels_split=None,
            media_out_uri=str(tmp_path / "media"),
            **kwargs,
        )
        monkeypatch.setattr(type(reader), "_split", lambda self, split: multimodal_corpus, raising=False)
        return reader

    return build


class TestDocumentParts:
    def test_auto_reads_everything_there_is(self, reader_factory) -> None:
        docs = list(reader_factory().documents())
        assert all(doc.as_content.has_media for doc in docs)
        assert docs[0].text == "the invoice total is 42"

    def test_image_drops_the_ocr_text(self, reader_factory) -> None:
        """The visual run: what the judge sees is the page, not a transcription."""
        docs = list(reader_factory(document_parts="image").documents())
        assert all(doc.as_content.has_media for doc in docs)
        assert all(doc.text == "" for doc in docs)

    def test_text_drops_the_pages(self, reader_factory) -> None:
        """The OCR baseline: comparable to the visual run because the ids and
        qrels are identical."""
        docs = list(reader_factory(document_parts="text").documents())
        assert not any(doc.as_content.has_media for doc in docs)
        assert docs[0].text == "the invoice total is 42"

    def test_both_is_explicit_about_wanting_the_pair(self, reader_factory) -> None:
        docs = list(reader_factory(document_parts="both").documents())
        assert docs[0].text and docs[0].as_content.has_media

    def test_document_ids_are_the_same_whichever_parts_are_read(self, reader_factory) -> None:
        """What makes the text and vision runs comparable at all."""
        ids = {
            parts: [doc.id for doc in reader_factory(document_parts=parts).documents()]
            for parts in ("auto", "text", "image", "both")
        }
        assert len(set(map(tuple, ids.values()))) == 1

    def test_an_unknown_selector_is_refused_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="document_parts must be one of"):
            HfReader(uri="x", document_parts="pixels-please")  # type: ignore[arg-type]

    def test_asking_for_images_a_corpus_lacks_is_an_error(self, tmp_path: Path, monkeypatch) -> None:
        """Silently falling back to text would report a visual number for an OCR run."""
        split = datasets.Dataset.from_dict({"_id": ["p0"], "text": ["only text here"]})
        reader = HfReader(uri="x", corpus_split="corpus", queries_split=None, qrels_split=None, document_parts="image")
        monkeypatch.setattr(type(reader), "_split", lambda self, s: split, raising=False)

        with pytest.raises(DataError, match="needs an image column"):
            list(reader.documents())

    def test_asking_for_text_a_corpus_lacks_is_an_error(self, tmp_path: Path, monkeypatch) -> None:
        rows = {"_id": ["p0"], "image": [PIL.new("RGB", (8, 8))]}
        split = datasets.Dataset.from_dict(rows).cast_column("image", datasets.Image())
        reader = HfReader(
            uri="x",
            corpus_split="corpus",
            queries_split=None,
            qrels_split=None,
            document_parts="text",
            media_out_uri=str(tmp_path / "media"),
        )
        monkeypatch.setattr(type(reader), "_split", lambda self, s: split, raising=False)

        with pytest.raises(DataError, match="needs a text column"):
            list(reader.documents())


class TestMediaPersistence:
    def test_pages_are_written_content_addressed(self, reader_factory, tmp_path: Path) -> None:
        refs = [ref for doc in reader_factory(document_parts="image").documents() for ref in doc.media]

        assert all(ref.sha256 and ref.sha256 in ref.uri for ref in refs)
        assert all((ref.width, ref.height) == (40, 60) for ref in refs)

    def test_identical_pages_share_one_file(self, tmp_path: Path, monkeypatch) -> None:
        """Content addressing is what keeps a corpus with repeated pages from
        writing the same bytes twice."""
        same = PIL.new("RGB", (8, 8), "white")
        rows = {"_id": ["a", "b"], "image": [same, same.copy()]}
        split = datasets.Dataset.from_dict(rows).cast_column("image", datasets.Image())
        reader = HfReader(
            uri="x",
            corpus_split="corpus",
            queries_split=None,
            qrels_split=None,
            document_parts="image",
            media_out_uri=str(tmp_path / "media"),
        )
        monkeypatch.setattr(type(reader), "_split", lambda self, s: split, raising=False)

        uris = {ref.uri for doc in reader.documents() for ref in doc.media}
        assert len(uris) == 1

    def test_an_image_column_without_a_destination_is_refused(self, monkeypatch) -> None:
        """Decoded pixels have nowhere to go; holding a corpus of them in memory
        is the failure mode the URI indirection exists to prevent."""
        rows = {"_id": ["p0"], "image": [PIL.new("RGB", (8, 8))]}
        split = datasets.Dataset.from_dict(rows).cast_column("image", datasets.Image())
        reader = HfReader(uri="x", corpus_split="corpus", queries_split=None, qrels_split=None, document_parts="image")
        monkeypatch.setattr(type(reader), "_split", lambda self, s: split, raising=False)

        with pytest.raises(ConfigError, match="media_out_uri"):
            list(reader.documents())
