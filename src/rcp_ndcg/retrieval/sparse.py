"""BM25 with ``bm25s``, optionally with a Snowball stemmer from PyStemmer (both core dependencies).

Stemming is stated, never inferred from what happens to be installed: :class:`~rcp_ndcg.retrieval.BM25Config`
names the stemmer language (or none), the index identity records it, and a requested stemmer without PyStemmer is
a :class:`~rcp_ndcg.errors.DependencyError`.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import ConfigError, DependencyError
from rcp_ndcg.retrieval.base import BaseRetriever, to_texts

STOPWORDS = "en"
"""The bm25s stop list the corpus is tokenized with."""


def _bm25s() -> Any:
    """The ``bm25s`` module (an optional dependency)."""
    try:
        import bm25s
    except ImportError as exc:
        raise DependencyError(
            "BM25 needs the bm25s package, which is not installed",
            hint="bm25s is a dependency of rcp-ndcg: reinstall it (pip install --force-reinstall rcp-ndcg)",
        ) from exc
    return bm25s


def stemmer_for(language: str | None) -> Any:
    """The PyStemmer stemmer for a Snowball ``language`` (e.g. ``"english"``), or ``None`` for no stemming.

    Raises:
        DependencyError: A stemmer is requested and PyStemmer is not installed.
        ConfigError: PyStemmer has no stemmer for ``language``.
    """
    if language is None:
        return None
    try:
        import Stemmer
    except ImportError as exc:
        raise DependencyError(
            f"BM25 with stemmer {language!r} needs PyStemmer, which is not installed",
            hint="PyStemmer is a dependency of rcp-ndcg: reinstall it (pip install --force-reinstall rcp-ndcg), or "
            "set stemmer: null",
        ) from exc
    languages = sorted(Stemmer.algorithms())
    if language not in languages:
        raise ConfigError(f"PyStemmer has no stemmer {language!r}", hint=f"known: {', '.join(languages)}")
    return Stemmer.Stemmer(language)


class BM25SRetriever(BaseRetriever):
    """BM25 over a ``bm25s`` index, built by :meth:`build_index` and loaded by :meth:`from_index`."""

    def __init__(self, model: Any, corpus: list[str], stemmer: Any) -> None:
        self.model = model
        self.corpus = corpus
        self.stemmer = stemmer

    @property
    def name(self) -> str:
        return "bm25s"

    @classmethod
    def build_index(
        cls, corpus: Sequence[Content | str], dataset_dir: Path, *, stemmer: str | None = None, **_: Any
    ) -> None:
        """Build the index under ``dataset_dir/bm25s/``, stemming with the Snowball ``stemmer`` language (or not)."""
        stemmer_object = stemmer_for(stemmer)
        engine = _bm25s()
        bm_dir = Path(dataset_dir) / "bm25s"
        bm_dir.mkdir(parents=True, exist_ok=True)
        model = engine.BM25()
        model.index(engine.tokenize(to_texts(corpus), stopwords=STOPWORDS, stemmer=stemmer_object))
        with (bm_dir / "bm25.pkl").open("wb") as handle:
            pickle.dump(model, handle)
        (bm_dir / "meta.json").write_text(json.dumps({"stemmer": stemmer}, indent=2), encoding="utf-8")

    @classmethod
    def from_index(cls, dataset_dir: Path, corpus: Sequence[Content | str], **_: Any) -> BM25SRetriever:
        """Load the index under ``dataset_dir/bm25s/``, with the stemmer it was built with."""
        bm_dir = Path(dataset_dir) / "bm25s"
        model_path = bm_dir / "bm25.pkl"
        if not model_path.is_file():
            raise FileNotFoundError(f"BM25S index not found at {model_path}")
        meta = json.loads((bm_dir / "meta.json").read_text(encoding="utf-8"))
        stemmer = stemmer_for(meta["stemmer"])
        _bm25s()  # unpickling the model needs the package
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        return cls(model, to_texts(corpus), stemmer)

    def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """The top ``k`` documents for ``query``: ``{"id": row, "text": ..., "score": ...}``, best first."""
        k = min(k, len(self.corpus))  # bm25s refuses a k beyond the corpus
        tokens = _bm25s().tokenize(query, stemmer=self.stemmer)  # stop words are absent from the index anyway
        ids, scores = self.model.retrieve(tokens, k=k)
        results = [
            {"id": int(ids[0, i]), "text": self.corpus[int(ids[0, i])], "score": float(scores[0, i])}
            for i in range(ids.shape[1])
        ]
        results.sort(key=lambda result: result["score"], reverse=True)
        return results


__all__ = ["STOPWORDS", "BM25SRetriever", "stemmer_for"]
