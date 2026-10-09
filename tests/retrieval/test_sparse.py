"""BM25 with bm25s: stemming is stated in the config, recorded in the index identity, and never dropped silently."""

from __future__ import annotations

import sys

import pytest

from rcp_ndcg.errors import ConfigError, DataError, DependencyError, MissingInputError
from rcp_ndcg.retrieval import BM25Config, sparse
from rcp_ndcg.retrieval import _api as retrieval_api


class _FakeStemmerModule:
    """PyStemmer's surface: ``algorithms()`` and ``Stemmer(language)``."""

    def __init__(self) -> None:
        self.module = type(sys)("Stemmer")
        self.module.algorithms = lambda: ["english", "french", "german"]
        self.module.Stemmer = lambda language: ("stemmer", language)


def test_stemming_is_part_of_the_index_identity() -> None:
    stemmed = retrieval_api._identity(BM25Config(stemmer="english"), ["d1"], [])
    unstemmed = retrieval_api._identity(BM25Config(), ["d1"], [])

    assert stemmed != unstemmed


def test_a_requested_stemmer_without_pystemmer_is_a_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    from rcp_ndcg.retrieval import sparse

    monkeypatch.setitem(sys.modules, "Stemmer", None)

    with pytest.raises(DependencyError, match="PyStemmer") as caught:
        sparse.stemmer_for("english")
    assert "pip install --force-reinstall rcp-ndcg" in (caught.value.hint or "")


def test_an_unknown_stemmer_language_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from rcp_ndcg.retrieval import sparse

    monkeypatch.setitem(sys.modules, "Stemmer", _FakeStemmerModule().module)

    assert sparse.stemmer_for("german") == ("stemmer", "german")
    assert sparse.stemmer_for(None) is None
    with pytest.raises(ConfigError, match="klingon"):
        sparse.stemmer_for("klingon")


class TestTheIndexFormat:
    """The index is persisted in bm25s' own format (npz arrays + JSON parameters), never a pickle: an index
    directory comes from ordinary user paths (``retrieval index --out`` / ``retrieval search --index``), and
    unpickling one somebody else wrote would run their code."""

    def test_the_index_is_stored_without_pickle_and_still_searches(self, tmp_path: pytest.Path) -> None:
        sparse.build_bm25_index(["hares run across the fields", "tortoises move slowly"], tmp_path, stemmer=None)

        files = {entry.name for entry in (tmp_path / "bm25s").iterdir()}
        assert not any(name.endswith((".pkl", ".pickle")) for name in files), "no pickle on disk"
        assert "params.index.json" in files and "vocab.index.json" in files, "bm25s' npz + JSON format"
        assert "meta.json" in files, "the stemmer is recorded beside the model"

        hits = sparse.search_bm25(tmp_path, ["hares"], k=2)
        assert hits[0][0][0] == 0 and hits[0][0][1] > 0

    def test_an_index_directory_without_a_stored_model_is_missing_input(self, tmp_path: pytest.Path) -> None:
        (tmp_path / "bm25s").mkdir()
        with pytest.raises(MissingInputError, match="BM25 index not found"):
            sparse.search_bm25(tmp_path, ["hares"], k=2)

    def test_an_index_without_its_stemmer_record_is_missing_input(self, tmp_path: pytest.Path) -> None:
        """Stemming is stated, never inferred: a bare bm25s model without the meta.json that names its stemmer
        is refused instead of searched unstemmed (the read was a bare FileNotFoundError before)."""
        sparse.build_bm25_index(["hares run"], tmp_path, stemmer="english")
        (tmp_path / "bm25s" / "meta.json").unlink()

        with pytest.raises(MissingInputError, match="meta.json") as caught:
            sparse.search_bm25(tmp_path, ["hares"], k=2)
        assert "index()" in (caught.value.hint or "")

    def test_a_corrupt_stemmer_record_is_missing_input(self, tmp_path: pytest.Path) -> None:
        """A hand-edited or partially written meta.json is refused by name (the reads were bare
        JSONDecodeError / KeyError before), with the rebuild hint."""
        for content in ("not json", '{"x": 1}'):
            sparse.build_bm25_index(["hares run"], tmp_path, stemmer=None)
            (tmp_path / "bm25s" / "meta.json").write_text(content)

            with pytest.raises(MissingInputError, match="meta.json") as caught:
                sparse.search_bm25(tmp_path, ["hares"], k=2)
            assert "rebuild" in (caught.value.hint or "")

    def test_a_pickle_written_by_an_earlier_build_is_refused_with_the_fix(self, tmp_path: pytest.Path) -> None:
        (tmp_path / "bm25s").mkdir()
        (tmp_path / "bm25s" / "bm25.pkl").write_bytes(b"not really a pickle")

        with pytest.raises(MissingInputError, match="BM25 index not found") as caught:
            sparse.search_bm25(tmp_path, ["hares"], k=2)
        assert "rebuild" in (caught.value.hint or "") and "pickle" in (caught.value.hint or "")

    def test_a_stemmed_index_matches_an_unstemmed_query_inflection(self, tmp_path: pytest.Path) -> None:
        """The stemmer is read back from the index (the sweep's M16 mutation dropped that read and nothing
        failed): a query's inflection reaches the documents' stems."""
        corpus = ["the hares are running across the fields", "tortoises move slowly"]
        sparse.build_bm25_index(corpus, tmp_path, stemmer="english")

        stemmed_hits = sparse.search_bm25(tmp_path, ["runs"], k=2)

        assert stemmed_hits[0][0][0] == 0 and stemmed_hits[0][0][1] > 0, "'runs' stems to 'run' and matches"

    def test_a_corpus_of_only_stop_words_is_a_data_error(self, tmp_path: pytest.Path) -> None:
        """An empty vocabulary crashed inside bm25s with a bare ``ValueError: max() iterable argument is
        empty``; the corpus is refused by name, with where to look."""
        with pytest.raises(DataError, match="no indexable tokens") as caught:
            sparse.build_bm25_index(["the of and to a", "a the of"], tmp_path, stemmer=None)
        assert "stop" in (caught.value.hint or "")


class TestTheQuerySide:
    """A8/A9: a query with no indexable term is refused, never scored as ``k`` arbitrary zero-score
    documents, and ties at the cut follow the retrieval stack's one rule (score descending, lower row)."""

    @pytest.mark.parametrize("query", ["", "   ", "the a of"])
    def test_a_query_with_no_indexable_term_is_refused(self, tmp_path: pytest.Path, query: str) -> None:
        sparse.build_bm25_index(["alpha beta", "beta gamma"], tmp_path, stemmer=None)

        with pytest.raises(DataError, match="no indexable term") as caught:
            sparse.search_bm25(tmp_path, [query], k=2)

        assert "stop" in (caught.value.hint or "")

    def test_a_query_that_matches_no_document_is_refused(self, tmp_path: pytest.Path) -> None:
        """A8 (the verifier's repro): a query whose terms all occur in no document scored every row 0.0 and
        returned the cut's ``k`` arbitrary documents, which looks like a result."""
        sparse.build_bm25_index(["alpha beta", "beta gamma"], tmp_path, stemmer=None)

        with pytest.raises(DataError, match="matches no document") as caught:
            sparse.search_bm25(tmp_path, ["zeta"], k=2)

        assert "corpus" in (caught.value.hint or "")

    def test_ties_break_toward_the_lower_row_at_the_cut(self, tmp_path: pytest.Path) -> None:
        """Four documents carry the query term and one does not: ``k=3`` keeps rows 0, 1, 2 in ascending
        order. The model's own ``argpartition`` order returned rows 3, 1, 2 and dropped row 0, so the set at
        the cut -- and every nDCG computed from it -- moved with the library's partition."""
        sparse.build_bm25_index(["alpha", "alpha", "alpha", "alpha", "beta"], tmp_path, stemmer=None)

        hits = sparse.search_bm25(tmp_path, ["alpha"], k=3)

        assert [row for row, _ in hits[0]] == [0, 1, 2]
        assert len({score for _, score in hits[0]}) == 1

    def test_a_partial_match_still_ranks_by_score(self, tmp_path: pytest.Path) -> None:
        """The tie repair does not disturb the ordinary case: a document with both terms outranks one with a
        single term, and the cut keeps the best ``k``."""
        sparse.build_bm25_index(["alpha beta", "alpha", "gamma"], tmp_path, stemmer=None)

        hits = sparse.search_bm25(tmp_path, ["alpha beta"], k=2)

        assert [row for row, _ in hits[0]] == [0, 1]
        assert hits[0][0][1] > hits[0][1][1]


def test_a_missing_bm25s_package_names_the_reinstall(monkeypatch: pytest.MonkeyPatch) -> None:
    """The coverage gap (contract F8): the DependencyError path was untested. ``bm25s`` is a core dependency,
    but a stripped install must say what to do rather than fail with a bare ImportError."""
    monkeypatch.setitem(sys.modules, "bm25s", None)

    with pytest.raises(DependencyError, match="bm25s package, which is not installed") as caught:
        sparse._bm25s()

    assert "pip install --force-reinstall rcp-ndcg" in (caught.value.hint or "")
