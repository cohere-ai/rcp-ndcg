"""BM25 with bm25s: stemming is stated in the config, recorded in the index identity, and never dropped silently."""

from __future__ import annotations

import sys

import pytest

from rcp_ndcg.errors import ConfigError, DependencyError
from rcp_ndcg.retrieval import BM25Config
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
