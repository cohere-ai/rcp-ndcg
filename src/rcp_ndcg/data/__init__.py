"""The data layer: datasets and rankings, one way in.

* :func:`load_dataset` turns a URI (``hf://``, ``suite:``, ``beir:``, ``jsonl:``, ``images:``, ``videos:``,
  ``frames:``) into a :class:`Dataset`: float qrels, released gains, candidate pools, excluded ids, and
  queries and corpus read on demand.
* :func:`load_rankings` turns a run file (parquet, TREC, JSONL, CSV) into :class:`Rankings`.
* In memory: :meth:`Dataset.from_records` and :meth:`Rankings.from_records` take plain records (dicts, or the row
  models :class:`QueryRow`, :class:`DocumentRow`, :class:`QrelRow`, :class:`RankingRow`) and validate them strictly.
  pandas is an output format only (``to_pandas``); a frame goes in as ``frame.to_dict("records")``.
* The formats themselves are the readers and writers of :mod:`rcp_ndcg.data.io`.
* :func:`validate` checks a dataset (and rankings) against the scoring protocol before scoring.
* Media: :class:`MediaResolver` turns a :class:`~rcp_ndcg_core.content.MediaRef` into bytes or an image through
  the storage layer and a content-addressed cache.
* :class:`Preprocessing` is what a judge is shown: the text policy (:class:`TextPolicy`), the chunk geometry
  (:class:`ChunkPolicy`), the pixel budget of images (:class:`ImagePolicy`) and the frames of a video
  (:class:`VideoPolicy`). Text limits count tokens of the judge's tokenizer (:func:`load_tokenizer`,
  :class:`TextTokenizer`).

The content model (:class:`~rcp_ndcg_core.content.Content` and its parts) lives in ``rcp_ndcg_core.content``.
"""

from rcp_ndcg.data.dataset import (
    SUITES,
    VIDORE_NATIVE_LANGUAGE,
    Dataset,
    DocumentRow,
    QrelRow,
    QueryRow,
    Suite,
    load_dataset,
)
from rcp_ndcg.data.media import MediaError, MediaResolver, default_resolver
from rcp_ndcg.data.preprocess import ChunkPolicy, Preprocessing, TextPolicy
from rcp_ndcg.data.rankings import DEFAULT_SYSTEM, RankingRow, Rankings, load_rankings
from rcp_ndcg.data.resolution import ImagePolicy, VideoPolicy
from rcp_ndcg.data.tokenizer import TextTokenizer, load_tokenizer
from rcp_ndcg.data.validate import ValidationCheck, ValidationReport, validate

__all__ = [
    "ChunkPolicy",
    "ImagePolicy",
    "Preprocessing",
    "TextPolicy",
    "TextTokenizer",
    "VideoPolicy",
    "DEFAULT_SYSTEM",
    "SUITES",
    "VIDORE_NATIVE_LANGUAGE",
    "Dataset",
    "DocumentRow",
    "MediaError",
    "MediaResolver",
    "QrelRow",
    "QueryRow",
    "RankingRow",
    "Rankings",
    "Suite",
    "ValidationCheck",
    "ValidationReport",
    "default_resolver",
    "load_dataset",
    "load_rankings",
    "load_tokenizer",
    "validate",
]
