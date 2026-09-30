"""BM25 with ``bm25s``, optionally with a Snowball stemmer from PyStemmer (both core dependencies): build an index
(:func:`build_bm25_index`) and search it (:func:`search_bm25`).

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

from rcp_ndcg.errors import ConfigError, DependencyError, MissingInputError

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


def build_bm25_index(corpus: Sequence[Content | str], dataset_dir: Path, *, stemmer: str | None) -> None:
    """Build a BM25 index of ``corpus`` under ``dataset_dir/bm25s/``, stemming with the Snowball ``stemmer``.

    Args:
        corpus: The documents, in index row order; BM25 reads their text (an image-only document is empty).
        dataset_dir: The index directory.
        stemmer: A PyStemmer language (``"english"``), or ``None`` for no stemming.
    """
    stemmer_object = stemmer_for(stemmer)
    engine = _bm25s()
    bm_dir = Path(dataset_dir) / "bm25s"
    bm_dir.mkdir(parents=True, exist_ok=True)
    model = engine.BM25()
    texts = [item if isinstance(item, str) else item.text for item in corpus]
    model.index(engine.tokenize(texts, stopwords=STOPWORDS, stemmer=stemmer_object))
    with (bm_dir / "bm25.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    (bm_dir / "meta.json").write_text(json.dumps({"stemmer": stemmer}, indent=2), encoding="utf-8")


def search_bm25(dataset_dir: Path, queries: Sequence[str], *, k: int) -> list[list[tuple[int, float]]]:
    """The top ``k`` rows of the index under ``dataset_dir/bm25s/`` for each query, best first.

    Args:
        dataset_dir: The index directory (:func:`build_bm25_index`); its stemmer is the one it was built with.
        queries: The query texts.
        k: Rows per query, at most the number of documents indexed.

    Returns:
        One list per query of ``(row, score)``, the score descending.

    Raises:
        MissingInputError: No index under ``dataset_dir``.
    """
    bm_dir = Path(dataset_dir) / "bm25s"
    model_path = bm_dir / "bm25.pkl"
    if not model_path.is_file():
        raise MissingInputError(
            f"BM25 index not found at {model_path}",
            hint="build it with index()",
            cli_hint="build it with `rcp-ndcg retrieval index`",
        )
    stemmer = stemmer_for(json.loads((bm_dir / "meta.json").read_text(encoding="utf-8"))["stemmer"])
    engine = _bm25s()  # unpickling the model needs the package
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    out = []
    for query in queries:
        tokens = engine.tokenize(query, stemmer=stemmer)  # stop words are absent from the index anyway
        ids, scores = model.retrieve(tokens, k=k)
        hits = [(int(ids[0, i]), float(scores[0, i])) for i in range(ids.shape[1])]
        out.append(sorted(hits, key=lambda hit: hit[1], reverse=True))
    return out


__all__ = ["STOPWORDS", "build_bm25_index", "search_bm25", "stemmer_for"]
