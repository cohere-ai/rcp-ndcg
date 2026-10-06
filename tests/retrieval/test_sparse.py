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
