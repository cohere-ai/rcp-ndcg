"""The conformance suite every reader and writer is run through.

This is the mechanism the plan calls "one tested contract each".  The failure it
exists to prevent is the one in ``rerank``, where three independent
``SampleV1``-to-text extractors disagree and a teacher and a student read
different text from the same row.  A new reader is added to :data:`READER_CASES`
and is then held to the same invariants as every other one -- so a format that
loses text, drops media, reorders records between passes, or invents metadata it
cannot support fails here rather than three phases downstream.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rcp_ndcg_core._records import RankingExample
from rcp_ndcg_core.content import Content, ImagePart, MediaRef

from rcp_ndcg.data.io import available_readers, available_writers, get_reader, get_writer
from rcp_ndcg.data.io.base import SourceReader
from rcp_ndcg.errors import ConfigError, DataError, MissingInputError
from rcp_ndcg.testing import io_conformance

from tests.data.hub_stubs import hub_fixture

PIL = pytest.importorskip("PIL.Image")


# -- fixtures building one small dataset in each format ---------------------


@pytest.fixture
def ranking_jsonl(tmp_path) -> str:
    """Two queries, text documents, graded qrels."""
    path = tmp_path / "rank" / "data.jsonl"
    path.parent.mkdir(parents=True)
    records = [
        RankingExample(
            query_id="q1",
            query="what is a tortoise",
            doc_ids=["d1", "d2"],
            docs=["a tortoise is a reptile", "unrelated passage"],
            qrels={"d1": 2, "d2": 0},
        ),
        RankingExample(
            query_id="q2",
            query="how long do they live",
            doc_ids=["d3"],
            docs=["tortoises can live over a century"],
            qrels={"d3": 1},
        ),
    ]
    path.write_text("".join(record.serialize_jsonl() + "\n" for record in records))
    return str(path)


@pytest.fixture
def image_ranking_jsonl(tmp_path) -> str:
    """One query over two page images, so media survives the round trip."""
    pages = tmp_path / "pages"
    pages.mkdir()
    refs = []
    for index in range(2):
        page = pages / f"page_{index}.png"
        PIL.new("RGB", (32, 48), color=(index * 10, 0, 0)).save(page)
        refs.append(MediaRef(uri=str(page), mime="image/png", width=32, height=48))
    path = tmp_path / "img" / "data.jsonl"
    path.parent.mkdir(parents=True)
    record = RankingExample(
        query_id="q1",
        query="find the invoice total",
        doc_ids=["p0", "p1"],
        contents=[Content.from_parts([ImagePart(ref=ref, page=i + 1)]) for i, ref in enumerate(refs)],
        qrels={"p0": 1, "p1": 0},
    )
    path.write_text(record.serialize_jsonl() + "\n")
    return str(path)


@pytest.fixture
def beir_dir(tmp_path) -> str:
    root = tmp_path / "beir"
    (root / "qrels").mkdir(parents=True)
    (root / "corpus.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {"_id": "d1", "title": "Tortoises", "text": "a tortoise is a reptile"},
                {"_id": "d2", "title": "", "text": "unrelated passage"},
                {"_id": "d3", "title": "", "text": "tortoises can live over a century"},
            ]
        )
    )
    (root / "queries.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [{"_id": "q1", "text": "what is a tortoise"}, {"_id": "q2", "text": "how long do they live"}]
        )
    )
    (root / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t0\nq2\td3\t1\n")
    return str(root)


@pytest.fixture
def image_dir(tmp_path) -> str:
    root = tmp_path / "images"
    (root / "docA").mkdir(parents=True)
    (root / "docB").mkdir(parents=True)
    for doc in ("docA", "docB"):
        for page in (1, 2):
            PIL.new("RGB", (40, 60)).save(root / doc / f"page_{page}.png")
    queries = tmp_path / "queries.jsonl"
    queries.write_text(json.dumps({"query_id": "q1", "text": "the invoice total"}) + "\n")
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(json.dumps({"query_id": "q1", "qrels": {"docA/page_1": 1}}) + "\n")
    return str(root)


@pytest.fixture
def video_dir(tmp_path) -> str:
    """Two decodable clips plus sidecars, in the image-directory layout."""
    from tests.conftest import write_mjpeg_avi

    root = tmp_path / "videos"
    write_mjpeg_avi(root / "cooking" / "knot.avi", frames=6)
    write_mjpeg_avi(root / "sports" / "knot.avi", frames=4)
    (tmp_path / "video_queries.jsonl").write_text(json.dumps({"query_id": "q1", "text": "tying a knot"}) + "\n")
    (tmp_path / "video_qrels.jsonl").write_text(json.dumps({"query_id": "q1", "qrels": {"cooking/knot": 1}}) + "\n")
    return str(root)


@pytest.fixture
def frame_dir(tmp_path) -> str:
    """Two clips as per-clip frame directories."""
    root = tmp_path / "frames"
    for clip, count in (("clip_a", 5), ("clip_b", 3)):
        (root / clip).mkdir(parents=True)
        for index in range(count):
            PIL.new("RGB", (16, 12), color=(index * 30, 0, 0)).save(root / clip / f"{index:04d}.jpg")
    return str(root)


READER_CASES = {
    "jsonl": lambda fixture: get_reader("jsonl", uri=fixture("ranking_jsonl")),
    "jsonl_images": lambda fixture: get_reader("jsonl", uri=fixture("image_ranking_jsonl")),
    "beir": lambda fixture: get_reader("beir", uri=fixture("beir_dir")),
    "image_dir": lambda fixture: get_reader(
        "images",
        uri=fixture("image_dir"),
        queries_uri=fixture("image_dir").replace("/images", "/queries.jsonl"),
        qrels_uri=fixture("image_dir").replace("/images", "/qrels.jsonl"),
    ),
    # Corpus-only sources: nothing to search them for, which is a shape the
    # contract has to cover rather than a reader to leave untested.  Their absence
    # is how the base class's queries/examples mutual recursion survived.
    "image_dir_unqueried": lambda fixture: get_reader("images", uri=fixture("image_dir")),
    "pdf": lambda fixture: get_reader("pdf", uri=fixture("pdf_file"), dpi=72),
    "video_dir": lambda fixture: get_reader(
        "videos",
        uri=fixture("video_dir"),
        queries_uri=fixture("video_dir").replace("/videos", "/video_queries.jsonl"),
        qrels_uri=fixture("video_dir").replace("/videos", "/video_qrels.jsonl"),
        hash_media=True,
    ),
    "frame_dir": lambda fixture: get_reader("frames", uri=fixture("frame_dir")),
}


@pytest.fixture
def pdf_file(tmp_path) -> str:
    pytest.importorskip("pypdfium2")
    PIL.init()
    pages = [PIL.new("RGB", (612, 792), color=(255, 255 - index * 40, 255)) for index in range(2)]
    target = tmp_path / "report.pdf"
    pages[0].save(target, save_all=True, append_images=pages[1:], resolution=72.0)
    return str(target)


@pytest.fixture(params=sorted(READER_CASES))
def reader(request) -> SourceReader:
    """Each case builds only the fixtures it names, so one reader's optional
    dependency (``pypdfium2`` for ``pdf``) skips that reader, not the suite."""
    return READER_CASES[request.param](request.getfixturevalue)


# -- the contract ----------------------------------------------------------


class TestReaderContract:
    """Invariants every registered reader must satisfy.

    One definition, one home: :func:`rcp_ndcg.testing.io_conformance` is what a plugin author calls; these
    tests run every built-in reader through it and keep the granular invariants readable when one fails.
    """

    def test_every_reader_passes_the_shared_conformance(self, reader):
        io_conformance(reader)

    def test_declares_a_registry_name(self, reader):
        assert reader.name
        assert reader.name in available_readers()

    def test_declares_at_least_one_shape(self, reader):
        assert reader.shapes

    def test_the_source_locator_is_called_uri(self, reader):
        """``get_reader`` and ``data ingest`` pass it by keyword, so a reader that
        names it ``path`` is unreachable through the registry."""
        import inspect as inspect_module

        parameters = list(inspect_module.signature(type(reader).__init__).parameters)
        assert parameters[1] == "uri", f"{type(reader).__name__} takes {parameters[1]!r} rather than 'uri'"

    def test_examples_are_reiterable(self, reader):
        """A reader is read more than once; a generator consumed once is a bug."""
        first = [example.id for example in reader.examples()]
        second = [example.id for example in reader.examples()]
        assert first == second

    def test_examples_exist_exactly_when_something_is_judged(self, reader):
        """The ranking shape *is* the judged candidate lists.

        A corpus with no qrels -- a PDF, a bare image directory -- has no examples
        to give, and must say so by yielding none rather than by fabricating a
        candidate list or by recursing while trying to derive one.
        """
        assert bool(list(reader.examples())) == bool(reader.qrels())

    def test_documents_are_reiterable(self, reader):
        first = [doc.id for doc in reader.documents()]
        second = [doc.id for doc in reader.documents()]
        assert first == second
        assert first

    def test_document_ids_are_unique(self, reader):
        ids = [doc.id for doc in reader.documents()]
        assert len(ids) == len(set(ids))

    def test_query_ids_are_unique(self, reader):
        ids = [query.id for query in reader.queries()]
        assert len(ids) == len(set(ids))

    def test_every_document_has_text_or_media(self, reader):
        """An empty document is unembeddable and unjudgeable; none should exist."""
        for doc in reader.documents():
            assert doc.text or doc.has_media, f"{doc.id} has neither text nor media"

    def test_qrels_reference_known_ids(self, reader):
        doc_ids = {doc.id for doc in reader.documents()}
        query_ids = {query.id for query in reader.queries()}
        for query_id, judged in reader.qrels().items():
            if query_ids:
                assert query_id in query_ids, f"qrels reference unknown query {query_id}"
            for doc_id in judged:
                assert doc_id in doc_ids, f"qrels reference unknown doc {doc_id}"

    def test_examples_align_their_document_fields(self, reader):
        for example in reader.examples():
            assert example.doc_ids
            if example.docs is not None:
                assert len(example.docs) == len(example.doc_ids)
            if example.contents is not None:
                assert len(example.contents) == len(example.doc_ids)

    def test_doc_contents_never_raises(self, reader):
        """What every encoder and judge calls; it must work for any reader."""
        for example in reader.examples():
            contents = example.doc_contents
            assert len(contents) == len(example.doc_ids)

    def test_media_refs_are_resolvable(self, reader, tmp_path):
        from rcp_ndcg.data.media import MediaResolver, probe_video_header

        resolver = MediaResolver()
        for doc in reader.documents():
            for ref in doc.media:
                if ref.mime and ref.mime.startswith("video/"):
                    assert probe_video_header(resolver.bytes_of(ref)) is not None
                else:
                    assert resolver.image(ref).width > 0


class TestRoundTrip:
    """Reading what we wrote must give back what we had."""

    def test_jsonl_round_trip_preserves_text_and_qrels(self, tmp_path, ranking_jsonl):
        original = list(get_reader("jsonl", uri=ranking_jsonl).examples())
        out = str(tmp_path / "out" / "data.jsonl")
        assert get_writer("jsonl").write_examples(original, out) == len(original)

        restored = list(get_reader("jsonl", uri=out).examples())
        assert [ex.id for ex in restored] == [ex.id for ex in original]
        assert [ex.docs for ex in restored] == [ex.docs for ex in original]
        assert [ex.qrels for ex in restored] == [ex.qrels for ex in original]

    def test_jsonl_round_trip_preserves_media_refs(self, tmp_path, image_ranking_jsonl):
        original = list(get_reader("jsonl", uri=image_ranking_jsonl).examples())
        out = str(tmp_path / "out" / "data.jsonl")
        get_writer("jsonl").write_examples(original, out)

        restored = list(get_reader("jsonl", uri=out).examples())
        assert [ref.uri for ref in restored[0].doc_contents[0].media] == [
            ref.uri for ref in original[0].doc_contents[0].media
        ]
        assert restored[0].doc_contents[0].media[0].width == 32

    def test_beir_round_trip_preserves_the_corpus(self, tmp_path, beir_dir):
        source = get_reader("beir", uri=beir_dir)
        out = str(tmp_path / "beir-out")
        get_writer("beir").write_corpus(source.documents(), source.queries(), source.qrels(), out)

        restored = get_reader("beir", uri=out)
        assert {doc.id for doc in restored.documents()} == {doc.id for doc in source.documents()}
        assert restored.qrels() == source.qrels()

    def test_beir_writer_refuses_media_rather_than_dropping_it(self, tmp_path, image_ranking_jsonl):
        """Silently writing an empty document would be the worse outcome."""
        source = get_reader("jsonl", uri=image_ranking_jsonl)
        with pytest.raises(ConfigError, match="cannot express"):
            get_writer("beir").write_corpus(source.documents(), source.queries(), source.qrels(), str(tmp_path / "b"))

    def test_ranking_to_corpus_derivation(self, ranking_jsonl):
        """A ranking source serves the corpus shape without knowing about it."""
        reader = get_reader("jsonl", uri=ranking_jsonl)
        assert {doc.id for doc in reader.documents()} == {"d1", "d2", "d3"}
        assert {query.id for query in reader.queries()} == {"q1", "q2"}

    def test_corpus_to_ranking_derivation(self, beir_dir):
        """A corpus source is judgeable with no intervening retrieval run; the documents read as the
        canonical records (title a field, body the text)."""
        examples = list(get_reader("beir", uri=beir_dir).examples())
        assert {ex.id for ex in examples} == {"q1", "q2"}
        by_id = {ex.id: ex for ex in examples}
        assert set(by_id["q1"].doc_ids) == {"d1", "d2"}
        assert dict(zip(by_id["q1"].doc_ids, by_id["q1"].docs, strict=True))["d1"] == "a tortoise is a reptile"


class TestReaderTable:
    def test_every_format_is_listed_under_its_uri_scheme(self):
        assert available_readers() == ["beir", "frames", "hf", "images", "jsonl", "mteb", "pdf", "videos"]
        assert available_writers() == ["beir", "jsonl"]

    def test_an_unknown_format_names_the_alternatives(self):
        with pytest.raises(ConfigError, match="Available:"):
            get_reader("parquet_of_dreams")
        with pytest.raises(ConfigError, match="Available:"):
            get_writer("stone_tablet")


class TestImageDirSpecifics:
    def test_doc_ids_are_paths_so_pages_do_not_collide(self, image_dir):
        ids = {doc.id for doc in get_reader("images", uri=image_dir).documents()}
        assert ids == {"docA/page_1", "docA/page_2", "docB/page_1", "docB/page_2"}

    def test_ids_do_not_depend_on_how_the_root_is_spelled(self, image_dir, frame_dir, tmp_path, monkeypatch):
        """A relative root once gave bare file names (docA/page_1 and docB/page_1 collided) and absolute clip ids."""
        monkeypatch.chdir(tmp_path)
        for scheme, root in (("images", image_dir), ("frames", frame_dir)):
            absolute = [doc.id for doc in get_reader(scheme, uri=root).documents()]
            relative = [doc.id for doc in get_reader(scheme, uri=Path(root).name).documents()]
            assert relative == absolute
        assert sorted(absolute) == ["clip_a", "clip_b"]

    def test_two_files_with_one_id_are_refused(self, image_dir):
        PIL.new("RGB", (4, 4)).save(Path(image_dir) / "docA" / "page_1.jpg")

        with pytest.raises(DataError, match="are both document 'docA/page_1'"):
            list(get_reader("images", uri=image_dir).documents())

    def test_two_pdfs_with_one_stem_are_refused_before_rendering(self, tmp_path):
        """They would share a page directory, and the second would reuse the first's pages."""
        (tmp_path / "pdfs" / "a").mkdir(parents=True)
        for name in ("report.pdf", "report.PDF"):
            (tmp_path / "pdfs" / "a" / name).write_bytes(b"%PDF-1.4\n")
        if len(list((tmp_path / "pdfs" / "a").iterdir())) != 2:
            # A case-insensitive filesystem folds the two names into one file: the collision this test refuses
            # cannot be set up there, and reading the one file would render it (and need pypdfium2).
            pytest.skip("the filesystem holds the two names as one file, so there is no collision to refuse")

        with pytest.raises(DataError, match="are both document 'a/report'"):
            list(get_reader("pdf", uri=str(tmp_path / "pdfs")).documents())

    def test_hashing_is_opt_in(self, image_dir):
        unhashed = next(iter(get_reader("images", uri=image_dir).documents()))
        assert unhashed.media[0].sha256 is None

        hashed = next(iter(get_reader("images", uri=image_dir, hash_media=True).documents()))
        assert hashed.media[0].sha256 is not None
        assert (hashed.media[0].width, hashed.media[0].height) == (40, 60)


class TestJsonlSpecifics:
    def test_a_malformed_qrels_row_is_a_data_error(self, tmp_path):
        for name, line in (("corpus.jsonl", '{"doc_id": "d1", "text": "x"}'), ("qrels.jsonl", '{"query_id": "q1"}')):
            (tmp_path / name).write_text(line + "\n")

        with pytest.raises(DataError, match="a qrels row is"):
            get_reader("jsonl", uri=str(tmp_path)).qrels()

    def test_a_directory_with_one_jsonl_is_accepted(self, ranking_jsonl):
        from pathlib import Path

        directory = str(Path(ranking_jsonl).parent)
        assert len(list(get_reader("jsonl", uri=directory).examples())) == 2

    def test_an_ambiguous_directory_is_refused(self, tmp_path, ranking_jsonl):
        from pathlib import Path
        from shutil import copyfile

        directory = Path(ranking_jsonl).parent
        copyfile(ranking_jsonl, directory / "other.jsonl")
        with pytest.raises(ConfigError, match="name one explicitly"):
            list(get_reader("jsonl", uri=str(directory)).examples())


class TestBeirSpecifics:
    def test_title_is_its_own_field_and_the_body_the_text(self, beir_dir):
        """Nothing joins at read time: the join is a formatting decision (the MTEB join), made where a
        model's input is formatted, so a document reads the same whichever format held it."""
        docs = {doc.id: doc for doc in get_reader("beir", uri=beir_dir).documents()}
        assert docs["d1"].title == "Tortoises"
        assert docs["d1"].text == "a tortoise is a reptile"
        assert docs["d2"].title is None and docs["d2"].text == "unrelated passage"

    def test_headerless_qrels_are_accepted(self, tmp_path, beir_dir):
        from pathlib import Path

        qrels = Path(beir_dir) / "qrels" / "test.tsv"
        qrels.write_text("q1\td1\t2\nq2\td3\t1\n")
        assert get_reader("beir", uri=beir_dir).qrels() == {"q1": {"d1": 2}, "q2": {"d3": 1}}

    def test_a_missing_corpus_names_what_it_looked_for(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(MissingInputError, match="corpus.jsonl"):
            list(get_reader("beir", uri=str(tmp_path / "empty")).documents())


def test_a_reader_that_implements_nothing_fails_at_definition():
    """A half-implemented reader must not return an empty dataset silently.

    Caught when the class is defined rather than when it is read, because each
    shape's default is written in terms of the other: left to run, the two derive
    from each other until the stack ends.
    """
    from rcp_ndcg.data.io.base import DataShape as Shape
    from rcp_ndcg.data.io.base import SourceReader as Base

    with pytest.raises(TypeError, match="overrides neither"):

        class Hollow(Base):
            name = "hollow-test-only"
            shapes = frozenset({Shape.RANKING})


def test_encoded_image_bytes_are_stable(tmp_path):
    """The same page must hash the same way twice, or caching is pointless."""
    from rcp_ndcg.data.media import sha256_of

    buffer = io.BytesIO()
    PIL.new("RGB", (16, 16), color=(1, 2, 3)).save(buffer, format="PNG")
    first = sha256_of(buffer.getvalue())

    buffer = io.BytesIO()
    PIL.new("RGB", (16, 16), color=(1, 2, 3)).save(buffer, format="PNG")
    assert sha256_of(buffer.getvalue()) == first


def test_a_fractional_qrels_label_is_kept_as_a_float(beir_dir, tmp_path) -> None:
    """A continuous label (0.7) is a grade like any other: no floor, no refusal."""
    qrels = Path(beir_dir) / "qrels" / "test.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\td1\t0.7\nq1\td2\t2\n")
    reader = get_reader("beir", uri=beir_dir)

    assert reader.qrels() == {"q1": {"d1": 0.7, "d2": 2.0}}
    get_writer("beir").write_corpus(reader.documents(), reader.queries(), reader.qrels(), str(tmp_path / "out"))
    assert (tmp_path / "out" / "qrels" / "test.tsv").read_text().splitlines()[1:] == ["q1\td1\t0.7", "q1\td2\t2.0"]


def test_a_non_numeric_qrels_label_is_a_data_error(beir_dir) -> None:
    (Path(beir_dir) / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\tnan\n")
    with pytest.raises(DataError, match="not a finite number"):
        get_reader("beir", uri=beir_dir).qrels()


def test_hub_qrels_keep_fractional_labels(tmp_path: Path) -> None:
    """Hub qrels are floats: a 0.5 label once raised (and before that was floored to 0). The Hub reader
    keeps float grades end to end; mteb's own loader casts them to int32, ours never does."""
    from rcp_ndcg.data.io.hub import HubReader

    with hub_fixture("mteb-nfcorpus", tmp_path) as root:
        qrels = root / "qrels/test.jsonl"
        qrels.write_text(
            '{"query-id": "PLAIN-2", "corpus-id": "d1", "score": "0.5"}\n'
            '{"query-id": "PLAIN-2", "corpus-id": "d2", "score": "2"}\n'
        )
        reader = HubReader("mteb/nfcorpus", revision="c" * 40)
        assert reader.qrels() == {"PLAIN-2": {"d1": 0.5, "d2": 2.0}}


def test_image_dir_qrels_keep_fractional_labels(image_dir) -> None:
    """The sidecar qrels of an image corpus were read with ``int()``: 0.5 became 0 without a word."""
    qrels = Path(image_dir.replace("/images", "/qrels.jsonl"))
    qrels.write_text('{"query_id": "q1", "qrels": {"docA/page_1": 0.5}}\n')

    reader = get_reader("images", uri=image_dir, qrels_uri=str(qrels))

    assert reader.qrels() == {"q1": {"docA/page_1": 0.5}}


# ---------------------------------------------------------------------------
# Nothing is cut or defaulted silently: the readers' refusals
# ---------------------------------------------------------------------------


def test_an_idless_beir_row_is_refused_with_its_line(beir_dir) -> None:
    """A row without an id (or with an empty one) vanished with no message; a corpus that
    silently loses documents produces confident wrong numbers."""
    corpus = Path(beir_dir) / "corpus.jsonl"
    corpus.write_text(json.dumps({"title": "no id", "text": "body"}) + "\n")
    with pytest.raises(DataError, match="corpus.jsonl:1.*no id"):
        list(get_reader("beir", uri=beir_dir).documents())

    queries = Path(beir_dir) / "queries.jsonl"
    queries.write_text(json.dumps({"_id": "", "text": "empty id"}) + "\n")
    with pytest.raises(DataError, match="queries.jsonl:1"):
        list(get_reader("beir", uri=beir_dir).queries())


def test_an_idless_sidecar_row_is_refused_not_dropped(image_dir) -> None:
    qrels = Path(image_dir.replace("/images", "/qrels.jsonl"))
    qrels.write_text('{"query_id": "", "qrels": {"docA/page_1": 1}}\n')
    reader = get_reader("images", uri=image_dir, qrels_uri=str(qrels))
    with pytest.raises(DataError, match="qrels.jsonl:1.*no id"):
        reader.qrels()


def test_a_missing_beir_file_is_a_missing_input(tmp_path) -> None:
    """A bare FileNotFoundError slipped past the typed hierarchy (and load_dataset's promised
    raises list); a library caller catching RcpNdcgError missed it."""
    (tmp_path / "empty").mkdir()
    reader = get_reader("beir", uri=str(tmp_path / "empty"))
    with pytest.raises(MissingInputError, match="corpus"):
        list(reader.documents())
    with pytest.raises(MissingInputError, match="queries"):
        list(reader.queries())
    with pytest.raises(MissingInputError, match="qrels"):
        reader.qrels()


def test_a_labelled_pair_twice_is_refused_not_last_wins(beir_dir) -> None:
    """The in-memory path refuses a label given twice; the file readers silently took the last."""
    qrels = Path(beir_dir) / "qrels" / "test.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td1\t5\n")
    with pytest.raises(DataError, match="twice"):
        get_reader("beir", uri=beir_dir).qrels()


def test_a_ranking_row_labelling_a_pair_twice_is_refused_not_last_wins(ranking_jsonl, tmp_path) -> None:
    """Two ranking rows for one query, each labelling q1/d1, silently merged last-wins through
    the derived qrels of the ranking layout."""
    rows = list(Path(ranking_jsonl).read_text().splitlines())
    extra = json.loads(rows[0])
    extra["qrels"] = {"d1": 5}
    out = tmp_path / "dup.jsonl"
    out.write_text(rows[0] + "\n" + json.dumps(extra) + "\n")
    with pytest.raises(DataError, match="twice"):
        get_reader("jsonl", uri=str(out)).qrels()


def test_a_sidecar_qrels_shape_error_names_the_row(image_dir) -> None:
    """``jsonl:`` refused a malformed qrels row; ``images:`` crashed with a bare AttributeError
    on the same format -- one format, one error."""
    qrels = Path(image_dir.replace("/images", "/qrels.jsonl"))
    qrels.write_text('{"query_id": "q1", "qrels": [5]}\n')
    reader = get_reader("images", uri=image_dir, qrels_uri=str(qrels))
    with pytest.raises(DataError, match="qrels"):
        reader.qrels()


def test_a_rankings_file_that_is_one_json_array_is_a_data_error(tmp_path) -> None:
    """A single-line JSON array of otherwise-valid rows fell into the per-row branch of the
    rankings loader and raised a raw AttributeError past load_rankings' promised DataError."""
    from rcp_ndcg.data import load_rankings

    path = tmp_path / "runs.json"
    path.write_text(json.dumps([{"query_id": "q1", "doc_id": "d1", "score": 1.0}]))
    with pytest.raises(DataError, match="JSON object"):
        load_rankings(str(path))


def test_a_jsonl_dataset_name_stops_at_the_suffix_not_the_first_dot(tmp_path) -> None:
    assert get_reader("jsonl", uri=str(tmp_path / "nfcorpus.v2.jsonl")).dataset_name == "nfcorpus.v2"


def test_unknown_corpus_row_keys_are_refused_but_a_title_reads(tmp_path) -> None:
    """A corpus row's title is a field now (it round-trips); any other unknown key is still refused, never
    read past."""
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "corpus.jsonl").write_text(json.dumps({"doc_id": "d1", "title": "T", "text": "b"}) + "\n")
    (tmp_path / "c" / "queries.jsonl").write_text("")
    (document,) = list(get_reader("jsonl", uri=str(tmp_path / "c")).documents())
    assert (document.title, document.text) == ("T", "b")

    (tmp_path / "c" / "corpus.jsonl").write_text(json.dumps({"doc_id": "d1", "abstract": "x", "text": "b"}) + "\n")
    with pytest.raises(DataError, match="abstract"):
        list(get_reader("jsonl", uri=str(tmp_path / "c")).documents())


def test_frame_indices_record_the_source_frames(frame_dir) -> None:
    """``frame_indices`` says which frames of the source these frames were sampled at; positions
    in the directory cannot say that once the numbering has gaps (``000001, 000007, 000042``)."""
    clip = Path(frame_dir) / "clip_a"
    for stale in clip.iterdir():
        stale.unlink()
    for number in (1, 7, 42):
        PIL.new("RGB", (16, 12)).save(clip / f"{number:06d}.jpg")

    document = next(iter(get_reader("frames", uri=frame_dir).documents()))

    part = document.as_content.parts[0]
    assert part.frame_indices == [1, 7, 42]


def test_a_headerless_qrels_row_that_is_not_a_header_is_not_eaten(beir_dir) -> None:
    """A headerless file whose first label failed the digit check lost its first row as a "header":
    the same label with a header raised the typed DataError -- two behaviours for one input. Now the
    first row is only a header when it names the columns; a ``nan`` label is refused, a ``+1`` or
    ``1e-3`` label is a valid grade and stays.
    """
    qrels = Path(beir_dir) / "qrels" / "test.tsv"
    qrels.write_text("q1\td1\tnan\nq1\td2\t2\n")
    with pytest.raises(DataError, match="not a finite number"):
        get_reader("beir", uri=beir_dir).qrels()
    for first_label, expected in (("+1", 1.0), ("1e-3", 0.001)):
        qrels.write_text(f"q1\td1\t{first_label}\nq1\td2\t2\n")
        assert get_reader("beir", uri=beir_dir).qrels() == {"q1": {"d1": expected, "d2": 2.0}}


def test_the_beir_writer_refuses_a_media_query_too(tmp_path, beir_dir) -> None:
    """A media-bearing document was refused loudly; a media-bearing query was written as an
    empty-text query silently -- the silent half of the asymmetry."""
    source = get_reader("beir", uri=beir_dir)
    queries = list(source.queries())
    image = tmp_path / "query.png"
    PIL.new("RGB", (8, 8)).save(image)
    media_query = queries[0].model_copy(update={"text": "", "content": Content.from_image(str(image))})
    with pytest.raises(ConfigError, match="cannot express"):
        get_writer("beir").write_corpus(source.documents(), [media_query], source.qrels(), str(tmp_path / "b"))


def test_a_query_instruction_round_trips_through_beir(tmp_path, beir_dir) -> None:
    """A BRIGHT-style query written without its instruction reads back as bare text: the text
    the encoder or judge sees silently changed."""
    source = get_reader("beir", uri=beir_dir)
    queries = list(source.queries())
    queries[0] = queries[0].model_copy(update={"instruction": "Given a claim, find documents that refute it"})
    out = str(tmp_path / "beir-out")
    get_writer("beir").write_corpus(source.documents(), queries, source.qrels(), out)

    restored = {query.id: query for query in get_reader("beir", uri=out).queries()}
    assert restored["q1"].instruction == "Given a claim, find documents that refute it"
