"""BM25 with ``bm25s``, optionally with a Snowball stemmer from PyStemmer (both core dependencies): build an index
(:func:`build_bm25_index`) and search it (:func:`search_bm25`).

Stemming is stated, never inferred from what happens to be installed: :class:`~rcp_ndcg.retrieval.BM25Config`
names the stemmer language (or none), the index identity records it, and a requested stemmer without PyStemmer is
a :class:`~rcp_ndcg.errors.DependencyError`. The stemmer is stored beside the model (``meta.json``) and read back
at search time, so a search uses the index's own stemmer, never whatever happens to be configured.

The search applies the retrieval stack's one tie rule (score descending, then the lower row -- the same rule
:func:`rcp_ndcg.retrieval.topk.numpy_topk` and :meth:`rcp_ndcg.data.Rankings.top` apply): the model's own
``argpartition`` order is never the cut. A query with no indexable term (empty, or only stop words after
the ``en`` list and the stemmer) is refused, never scored as ``depth`` arbitrary zero-score documents.

The index is persisted with bm25s' own format (npz arrays and JSON parameters), never a pickle: an index
directory comes from ordinary user paths (``retrieval index --out``, ``retrieval search --index``), and
unpickling one somebody else wrote would run their code.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import ConfigError, DataError, DependencyError, MissingInputError

STOPWORDS = "en"
"""The bm25s stop list the corpus is tokenized with."""

_MODEL_PARAMS = "params.index.json"
"""The parameters file of bm25s' own save format (``BM25.save``); the marker of a stored model."""

_LEGACY_PICKLE = "bm25.pkl"
"""The pickle name of the earlier build's format, refused with a rebuild hint."""


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

    The model is built in a temporary directory beside the target and swapped in with one rename, so a reader
    sees either the previous model or the complete new one -- and a build that dies leaves the previous one
    intact (the caller holds the index's publication lock, so the swap is not racing another build).

    Args:
        corpus: The documents, in index row order; BM25 reads their text (an image-only document is empty).
        dataset_dir: The index directory.
        stemmer: A PyStemmer language (``"english"``), or ``None`` for no stemming.

    Raises:
        DataError: No document of the corpus has an indexable token (every one is empty, or its tokens are all
            stop words): bm25s builds an empty vocabulary and would crash inside itself.
    """
    stemmer_object = stemmer_for(stemmer)
    engine = _bm25s()
    target = Path(dataset_dir) / "bm25s"
    Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    texts = [item if isinstance(item, str) else item.text for item in corpus]
    tokenized = engine.tokenize(texts, stopwords=STOPWORDS, stemmer=stemmer_object, show_progress=False)
    if not any(tokenized.ids):
        raise DataError(
            "the corpus has no indexable tokens: every document is empty, or only stop words after removal",
            hint="check the corpus texts; BM25 tokenizes with the 'en' stop list and the configured stemmer",
        )
    model = engine.BM25()
    model.index(tokenized, show_progress=False)
    built = Path(tempfile.mkdtemp(prefix=".bm25s.", dir=dataset_dir))
    try:
        model.save(str(built), allow_pickle=False)
        (built / "meta.json").write_text(json.dumps({"stemmer": stemmer}, indent=2), encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)  # one rename replaces the directory (os.replace needs an absent target)
        os.replace(built, target)
    finally:
        shutil.rmtree(built, ignore_errors=True)


def search_bm25(dataset_dir: Path, queries: Sequence[str], *, k: int) -> list[list[tuple[int, float]]]:
    """The top ``k`` rows of the index under ``dataset_dir/bm25s/`` for each query, best first.

    Args:
        dataset_dir: The index directory (:func:`build_bm25_index`); its stemmer is the one it was built with.
        queries: The query texts.
        k: Rows per query, at most the number of documents indexed.

    Returns:
        One list per query of ``(row, score)``, ordered by score descending, then by the lower row (the
        retrieval stack's one tie rule); the cut resolves a tie class by the lower row too.

    Raises:
        MissingInputError: No stored model under ``dataset_dir`` (one written by the earlier build's pickle
            format is refused with a rebuild hint: it is not loaded, so its code never runs).
        DataError: A query has no indexable term (empty, or only stop words after the ``en`` list and the
            stemmer): scoring it would return ``k`` arbitrary zero-score documents that look like a result.
    """
    bm_dir = Path(dataset_dir) / "bm25s"
    model_path = bm_dir / _MODEL_PARAMS
    if not model_path.is_file():
        hint = "build it with index()"
        if (bm_dir / _LEGACY_PICKLE).is_file():
            hint = (
                "this index was written in an earlier build's pickle format, which is no longer read "
                "(loading it would run its code); rebuild it with index()"
            )
        raise MissingInputError(
            f"BM25 index not found at {model_path}",
            hint=hint,
            cli_hint="build it with `rcp-ndcg retrieval index`",
        )
    meta_path = bm_dir / "meta.json"
    if not meta_path.is_file():
        raise MissingInputError(
            f"the BM25 index at {bm_dir} has no meta.json naming its stemmer",
            hint="build it with index() (stemming is stated, never inferred from the model)",
            cli_hint="build it with `rcp-ndcg retrieval index`",
        )
    try:
        stemmer = stemmer_for(json.loads(meta_path.read_text(encoding="utf-8"))["stemmer"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise MissingInputError(
            f"the BM25 index's {meta_path} is unreadable or does not name a stemmer",
            hint="rebuild the index with index()",
            cli_hint="rebuild it with `rcp-ndcg retrieval index`",
        ) from exc
    engine = _bm25s()
    model = engine.BM25.load(str(bm_dir), allow_pickle=False, load_corpus=False)
    num_docs = int(model.scores["num_docs"])
    from rcp_ndcg.retrieval.topk import select_topk

    out = []
    for query in queries:
        tokens = engine.tokenize(query, stopwords=STOPWORDS, stemmer=stemmer, show_progress=False)
        if not any(tokens.ids):
            raise DataError(
                f"the query {query[:200]!r} has no indexable term: BM25 would score every document 0.0",
                hint=f"the {STOPWORDS!r} stop list and the index's stemmer leave nothing to score; drop the "
                "empty query from the run, or check the text the reader produced for it",
            )
        # Every row, unsorted: the model's own argpartition order is never the cut. Scoring is the same
        # O(num_docs) pass the model does for any k; only the selection below is ours.
        result = model.retrieve(tokens, k=num_docs, sorted=False, show_progress=False)
        rows = np.asarray(result.documents[0], dtype=np.int64)
        scores_by_row = np.zeros(num_docs, dtype=np.float32)
        scores_by_row[rows] = np.asarray(result.scores[0], dtype=np.float32)
        kept_scores, kept_rows = select_topk(
            scores_by_row[None, :], np.arange(num_docs, dtype=np.int64)[None, :], min(k, num_docs)
        )
        out.append([(int(row), float(score)) for score, row in zip(kept_scores[0], kept_rows[0], strict=True)])
    return out


__all__ = ["STOPWORDS", "build_bm25_index", "search_bm25", "stemmer_for"]
