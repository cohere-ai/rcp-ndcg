"""The MTEB Hub layout: exactly what mteb's ``push_dataset_to_hub`` writes, plus the extras mteb ignores.

Configs, per subset ``s`` (the default subset has no prefix; the qrels config is named ``qrels``, not
``default``):

===================  =====================================================
``{s-}corpus``       ``id`` (string), ``title`` (string), ``text`` (string)
``{s-}queries``      ``id``, ``text``, ``instruction`` only when a query carries one
``{s-}qrels``        ``query-id``, ``corpus-id``, ``score`` (int64), plus ``gain``/``theta`` when given
``{s-}top_ranked``   ``query-id``, ``corpus-ids`` (list of strings); written when the data has a pool
``{s-}excluded``     ``query-id``, ``excluded-corpus-ids``; rcp-ndcg's extra, mteb reads no such config
===================  =====================================================

Each config is a parquet file at ``{config}/{split}-00000-of-00001.parquet`` (the ``push_dataset_to_hub``
shard name for a one-shard split), and the ``README.md`` carries the ``configs:`` front matter that
``datasets.load_dataset`` -- and through it mteb's ``RetrievalDatasetLoader`` -- reads the directory with.
With ``card=`` (a mteb ``TaskMetadata`` or the fields of one, ``[mteb]`` extra) the README is the full card
mteb's own template renders; without it the README is the front matter alone, so the layout still loads.

**The extras ride only where mteb ignores them.**  mteb's loader keeps three columns of the qrels, so the
calibrated ``gain``/``theta`` columns travel on the same table and drop there.  mteb reads no ``excluded``
config, so it holds the exclusions verbatim -- and because ``top_ranked`` is the pool mteb *does* read, the
exclusions are also folded out of it (out of the corpus, when the data has no pool): a model scored inside
mteb never sees an excluded document.  A grade that is not a whole number is refused: the ``score`` column is
written as int64 and mteb's loader casts it to int32 at load, where a fractional value fails; export integer
grades and keep the continuous signal in ``gain``/``theta``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import yaml
from rcp_ndcg_core._records import ID, Document, Query

from rcp_ndcg import storage
from rcp_ndcg.data.io.base import DataShape, SinkWriter
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.support.logging import get_logger

if TYPE_CHECKING:
    import pyarrow as pa

logger = get_logger(__name__)

SHARD = "{split}-00000-of-00001.parquet"
"""The parquet file of one config: the ``push_dataset_to_hub`` name for a one-shard split."""

DEFAULT_SUBSET = "default"
"""The subset name that leaves the config names unprefixed."""


class MtebWriter(SinkWriter):
    """Writes the MTEB Hub layout (see the module docstring) into a directory."""

    name: ClassVar[str] = "mteb"
    shapes: ClassVar[frozenset[DataShape]] = frozenset({DataShape.CORPUS})

    def write_dataset(
        self,
        dataset: Any,
        uri: str,
        *,
        subset: str | None = None,
        split: str | None = None,
        card: Mapping[str, Any] | Any | None = None,
    ) -> int:
        """Write a :class:`~rcp_ndcg.data.Dataset`: its qrels, gains, thetas, pools and exclusions included.

        Args:
            dataset: The dataset. Its ``subset`` and ``split`` (``"default"`` and ``"test"`` where the data
                model does not carry them yet) name the configs and the parquet split; ``subset=`` and
                ``split=`` override them.
            uri: The directory to write (a local path or a storage URI).
            subset: Overrides the dataset's subset.
            split: Overrides the dataset's split.
            card: A mteb ``TaskMetadata`` or the fields of one, for the README's card (the ``[mteb]`` extra).

        Returns:
            The number of corpus rows written.
        """
        if subset is not None and dataset.subsets:
            raise ConfigError(
                f"subset={subset!r} applies to one dataset, and this is a suite of {len(dataset.subsets)} "
                "subsets; every part writes under its own name",
                hint="drop the subset override (a suite's configs are named after its parts), or write one part alone",
            )
        parts = tuple(dataset.subsets) if dataset.subsets else (dataset,)
        if len(parts) == 1:
            part = parts[0]
            return self.write_corpus(
                part.corpus.values(),
                part.queries.values(),
                part.qrels,
                uri,
                candidates=part.candidates,
                excluded=part.excluded or None,
                gains=part.gains,
                thetas=part.thetas,
                card=card,
                subset=subset if subset is not None else dataset.subset,
                split=split if split is not None else dataset.split,
            )
        # A suite: every subset's configs into one directory (a published repository holds one config per
        # subset), and one README over all of them.
        rows = 0
        configs: list[dict[str, Any]] = []
        for part in parts:
            part_split = split if split is not None else part.split
            tables = _subset_tables(
                part.corpus.values(),
                part.queries.values(),
                part.qrels,
                candidates=part.candidates,
                excluded=part.excluded or None,
                gains=part.gains,
                thetas=part.thetas,
                subset=part.name,
            )
            _write_configs(tables, uri, part_split)
            configs += _configs_of(tables, part_split)
            rows += len(tables[f"{_config_prefix(part.name)}corpus"])
        storage.makedirs(uri)
        storage.write_text(storage.join(uri, "README.md"), _readme(configs, card))
        logger.info(f"wrote MTEB layout to {uri}: {len(configs)} configs over {len(parts)} subsets")
        return rows

    def write_corpus(  # noqa: PLR0913 - one parameter per config, all of them the writer's input
        self,
        documents: Iterable[Document],
        queries: Iterable[Query],
        qrels: dict[ID, dict[ID, float]],
        uri: str,
        *,
        candidates: dict[ID, list[ID]] | None = None,
        excluded: dict[ID, list[ID]] | None = None,
        gains: dict[ID, dict[ID, float]] | None = None,
        thetas: dict[ID, dict[ID, float]] | None = None,
        card: Mapping[str, Any] | Any | None = None,
        subset: str = DEFAULT_SUBSET,
        split: str = "test",
    ) -> int:
        """Write the corpus, queries, qrels and (when present) pool and exclusion configs under *uri*.

        Args:
            documents: The corpus (``Document`` or :class:`~rcp_ndcg.data.DocumentRow` records); the title is
                the record's own ``title`` field where the data model carries one, else the corpus has none.
            queries: The queries (an ``instruction`` column only when a query carries one).
            qrels: ``{query_id: {doc_id: grade}}``; grades must be whole numbers (the ``score`` column is
                int64 and mteb's loader casts it to int32, where a fractional value fails; the continuous gains
                travel in the ``gain``/``theta`` columns).
            uri: The directory to write.
            candidates: ``{query_id: [doc_id, ...]}`` -- the pool written as ``{s-}top_ranked``; when absent,
                the pool is the corpus minus each query's exclusions (and no ``top_ranked`` without
                exclusions, which is what a plain retrieval task is).
            excluded: ``{query_id: [doc_id, ...]}`` -- written as ``{s-}excluded`` and folded out of
                ``top_ranked``.
            gains: ``{query_id: {doc_id: gain}}``, the ``gain`` column mteb ignores.
            thetas: ``{query_id: {doc_id: theta}}``, the ``theta`` column mteb ignores.
            card: A mteb ``TaskMetadata`` or the fields of one, for the README's card (the ``[mteb]`` extra).
            subset: The subset the configs are prefixed with; ``"default"`` (or ``""``) writes unprefixed
                names.
            split: The split name of the parquet files (the republished datasets use the eval split
                ``test``).

        Returns:
            The number of corpus rows written.

        Raises:
            DataError: A grade is not a whole number, a qrel, pool, exclusion, gain or theta names an unknown
                id, or no qrels or no queries were given.
            ConfigError: A record carries media, which the layout's text columns cannot hold (BEIR refuses
                the same way; ``store_media``-backed image corpora are an ingestion concern, not an export).
        """
        corpus = list(documents)
        query_rows = list(queries)

        tables = _subset_tables(
            corpus,
            query_rows,
            qrels,
            candidates=candidates,
            excluded=excluded,
            gains=gains,
            thetas=thetas,
            subset=subset,
        )
        _write_configs(tables, uri, split)
        storage.write_text(storage.join(uri, "README.md"), _readme(_configs_of(tables, split), card))
        logger.info(f"wrote MTEB layout to {uri}: {len(tables)} configs, {len(corpus)} documents")
        return len(corpus)


def _title_of(document: Any) -> str:
    """The document's own ``title``, as the MTEB column holds it (``""`` when the source has none; the text
    stays the body, and nothing joins them here)."""
    return (document.title or "").strip()


# -- the tables -----------------------------------------------------------------


def _doc_id(document: Any) -> str:
    return str(document.doc_id)


def _query_id(query: Any) -> str:
    return str(query.query_id)


def _carries_media(record: Any) -> bool:
    if getattr(record, "has_media", False):
        return True
    content = getattr(record, "content", None)
    return content is not None and content.has_media


def _tables(
    corpus: list[Any],
    query_rows: list[Any],
    qrels: dict[ID, dict[ID, float]],
    pools: dict[ID, list[ID]],
    drops: dict[ID, list[ID]],
    gains: dict[ID, dict[ID, float]] | None,
    thetas: dict[ID, dict[ID, float]] | None,
    prefix: str,
) -> dict[str, pa.Table]:
    """The configs of one subset: corpus, queries, qrels, and the pool and exclusions when there are any."""
    tables: dict[str, pa.Table] = {
        f"{prefix}corpus": _corpus_table(corpus),
        f"{prefix}queries": _queries_table(query_rows),
        f"{prefix}qrels": _qrels_table(qrels, gains, thetas),
    }
    corpus_ids = [_doc_id(document) for document in corpus]
    if pools:
        tables[f"{prefix}top_ranked"] = _pool_table(pools, drops)
    elif drops:
        tables[f"{prefix}top_ranked"] = _pool_table({q: corpus_ids for q in qrels}, drops)
    if drops:
        tables[f"{prefix}excluded"] = _list_table(drops, "excluded-corpus-ids")
    return tables


def _subset_tables(
    documents: Iterable[Document],
    queries: Iterable[Query],
    qrels: dict[ID, dict[ID, float]],
    *,
    candidates: dict[ID, list[ID]] | None,
    excluded: dict[ID, list[ID]] | None,
    gains: dict[ID, dict[ID, float]] | None,
    thetas: dict[ID, dict[ID, float]] | None,
    subset: str,
) -> dict[str, pa.Table]:
    """The validated tables of one subset: the writer's checks, then its configs."""
    corpus = list(documents)
    query_rows = list(queries)
    if not qrels:
        raise DataError(
            "the MTEB layout needs qrels: every query of the queries config is filtered to them at load",
            hint="export a corpus without labels as `beir` or `jsonl` instead",
        )
    if not query_rows:
        raise DataError("no queries given: the MTEB layout needs the queries config")
    corpus_ids = [_doc_id(document) for document in corpus]
    query_ids = {_query_id(query) for query in query_rows}
    _check_ids(qrels, set(corpus_ids), query_ids, what="qrels")
    pools = _pool_of(candidates, set(corpus_ids), query_ids, what="candidates")
    drops = _pool_of(excluded, set(corpus_ids), query_ids, what="excluded")
    _check_pairs(gains, qrels, "gains")
    _check_pairs(thetas, qrels, "thetas")
    return _tables(corpus, query_rows, qrels, pools, drops, gains, thetas, _config_prefix(subset))


def _write_configs(tables: dict[str, pa.Table], uri: str, split: str) -> None:
    import pyarrow.parquet as pq

    for config, table in tables.items():
        path = storage.join(uri, config, SHARD.format(split=split))
        with storage.open_path(path, "wb") as handle:
            pq.write_table(table, handle)


def _configs_of(tables: dict[str, pa.Table], split: str) -> list[dict[str, Any]]:
    """The README's ``configs:`` entries of the tables: what ``load_dataset`` reads the directory with."""
    return [
        {"config_name": config, "data_files": [{"split": split, "path": f"{config}/{split}-*"}]}
        for config in sorted(tables)
    ]


def _config_prefix(subset: str) -> str:
    return "" if subset in ("", DEFAULT_SUBSET) else f"{subset}-"


def _corpus_table(documents: list[Any]) -> pa.Table:
    import pyarrow as pa

    for document in documents:
        if _carries_media(document):
            raise ConfigError(
                f"document {_doc_id(document)!r} carries media, which the MTEB text columns cannot express; "
                "rcp-ndcg's media travel as content references, not as parquet pixels",
                hint="export the text to `mteb` and the media to `jsonl`, or publish media through mteb's own layout",
            )
    return pa.table(
        {
            "id": pa.array([_doc_id(document) for document in documents], pa.string()),
            "title": pa.array([_title_of(document) for document in documents], pa.string()),
            "text": pa.array([str(document.text or "") for document in documents], pa.string()),
        }
    )


def _queries_table(queries: list[Any]) -> pa.Table:
    import pyarrow as pa

    for query in queries:
        if _carries_media(query):
            raise ConfigError(
                f"query {_query_id(query)!r} carries media, which the MTEB text columns cannot express",
                hint="export the text to `mteb` and the media to `jsonl`",
            )
    columns: dict[str, pa.Array] = {
        "id": pa.array([_query_id(query) for query in queries], pa.string()),
        "text": pa.array([str(query.text or "") for query in queries], pa.string()),
    }
    instructions = [query.instruction for query in queries]
    if any(instruction is not None for instruction in instructions):
        columns["instruction"] = pa.array(
            [instruction if isinstance(instruction, str) else None for instruction in instructions], pa.string()
        )
    return pa.table(columns)


def _qrels_table(
    qrels: dict[ID, dict[ID, float]],
    gains: dict[ID, dict[ID, float]] | None,
    thetas: dict[ID, dict[ID, float]] | None,
) -> pa.Table:
    """``query-id, corpus-id, score`` (int64) -- the columns the loader keeps -- plus ``gain``/``theta`` when
    the data carries them (rows without a value read as null; mteb's ``select_columns`` drops both)."""
    import pyarrow as pa

    pairs = [(query_id, doc_id) for query_id, judged in qrels.items() for doc_id in judged]
    columns: dict[str, pa.Array] = {
        "query-id": pa.array([str(query_id) for query_id, _ in pairs], pa.string()),
        "corpus-id": pa.array([str(doc_id) for _, doc_id in pairs], pa.string()),
        "score": pa.array([_integer_grade(qrels[q][d], q, d) for q, d in pairs], pa.int64()),
    }
    if gains is not None:
        columns["gain"] = pa.array([_value_of(gains, q, d) for q, d in pairs], pa.float64())
    if thetas is not None:
        columns["theta"] = pa.array([_value_of(thetas, q, d) for q, d in pairs], pa.float64())
    return pa.table(columns)


def _integer_grade(value: float, query_id: ID, doc_id: ID) -> int:
    """The whole-number grade the ``score`` column casts to; a fractional or non-finite one is refused, never
    floored.

    mteb casts the column to int32 at load and a fractional value fails that cast, so refusing here is what
    the loader would do, at write time and with the pair named.
    """
    if not math.isfinite(value):
        raise DataError(
            f"qrels: query {query_id!r}, document {doc_id!r}: the grade {value!r} is not a finite number",
            details={"query_id": str(query_id), "doc_id": str(doc_id), "grade": value},
        )
    integer = int(value)
    if value != integer:
        raise DataError(
            f"qrels: query {query_id!r}, document {doc_id!r}: the grade {value!r} is not a whole number; "
            "mteb's qrels are integers only",
            hint="export an integer grade for the pair (round or re-scale it), and keep the continuous "
            "signal in the `gain`/`theta` columns, which mteb ignores",
            details={"query_id": str(query_id), "doc_id": str(doc_id), "grade": value},
        )
    return integer


def _value_of(table: dict[ID, dict[ID, float]] | None, query_id: str, doc_id: str) -> float | None:
    value = table.get(query_id, {}).get(doc_id) if table else None
    return float(value) if value is not None else None


def _pool_table(pools: dict[ID, list[ID]], drops: dict[ID, list[ID]] | None) -> pa.Table:
    """``{query-id, corpus-ids}``: the pool per query, with its exclusions folded out."""
    import pyarrow as pa

    folded = drops or {}
    return pa.table(
        {
            "query-id": pa.array([str(query_id) for query_id in pools], pa.string()),
            "corpus-ids": pa.array(
                [
                    [str(doc_id) for doc_id in ids if str(doc_id) not in set(folded.get(query_id, ()))]
                    for query_id, ids in pools.items()
                ],
                pa.list_(pa.string()),
            ),
        }
    )


def _list_table(drops: dict[ID, list[ID]], column: str) -> pa.Table:
    import pyarrow as pa

    return pa.table(
        {
            "query-id": pa.array([str(query_id) for query_id in drops], pa.string()),
            column: pa.array([[str(doc_id) for doc_id in ids] for ids in drops.values()], pa.list_(pa.string())),
        }
    )


# -- validation ------------------------------------------------------------------


def _check_ids(qrels: dict[ID, dict[ID, float]], corpus_ids: set[str], query_ids: set[str], *, what: str) -> None:
    """A qrel (or pool entry) for an id the corpus or queries lack is refused, never trimmed."""
    unknown_queries = sorted(str(query_id) for query_id in qrels if str(query_id) not in query_ids)
    if unknown_queries:
        raise DataError(
            f"{what} name queries that are not in queries: {unknown_queries[:5]}",
            details={"what": what, "unknown_queries": unknown_queries[:20]},
        )
    unknown_docs = sorted(
        {str(doc_id) for judged in qrels.values() for doc_id in judged if str(doc_id) not in corpus_ids}
    )
    if unknown_docs:
        raise DataError(
            f"{what} name documents that are not in the corpus: {unknown_docs[:5]}",
            details={"what": what, "unknown_documents": unknown_docs[:20]},
        )


def _pool_of(
    table: dict[ID, list[ID]] | None, corpus_ids: set[str], query_ids: set[str], *, what: str
) -> dict[ID, list[ID]]:
    """``{query_id: [doc_id, ...]}`` as strings, refusing unknown ids."""
    if not table:
        return {}
    fake = {query_id: {str(doc_id): 0.0 for doc_id in ids} for query_id, ids in table.items()}
    _check_ids(fake, corpus_ids, query_ids, what=what)
    return {query_id: [str(doc_id) for doc_id in ids] for query_id, ids in table.items()}


def _check_pairs(table: dict[ID, dict[ID, float]] | None, qrels: dict[ID, dict[ID, float]], what: str) -> None:
    if not table:
        return
    unknown = sorted(
        (str(query_id), str(doc_id))
        for query_id, judged in table.items()
        for doc_id in judged
        if str(doc_id) not in qrels.get(query_id, {})
    )
    if unknown:
        raise DataError(
            f"{what} label pairs the qrels do not hold: {unknown[:3]}",
            details={"what": what, "unknown_pairs": unknown[:20]},
        )


# -- the README ------------------------------------------------------------------


def _readme(configs: list[dict[str, Any]], card: Mapping[str, Any] | Any | None) -> str:
    """The README: the ``configs:`` front matter ``load_dataset`` reads, and mteb's card body when asked for."""
    front = yaml.safe_dump({"configs": configs}, sort_keys=False, default_flow_style=False).strip()
    if card is None:
        return f"---\n{front}\n---\n"
    return _task_card(card, configs)


def _task_card(card: Mapping[str, Any] | Any, configs: list[dict[str, Any]]) -> str:
    """The card mteb's own template renders for the task metadata, with the configs in the front matter."""
    try:
        from mteb.abstasks.task_metadata import TaskMetadata
    except ImportError as exc:  # pragma: no cover - the card is an [mteb]-extra feature
        raise ConfigError(
            "writing the dataset card needs the mteb package",
            hint="install the [mteb] extra -- pip install 'rcp-ndcg[mteb]' -- or write the export without a card",
        ) from exc
    if isinstance(card, Mapping):
        if "dataset" not in card:
            raise ConfigError(
                "the card fields need a 'dataset': {'path': ..., 'revision': ...}, as mteb's TaskMetadata requires",
            )
        try:
            meta = TaskMetadata(**dict(card))
        except Exception as exc:
            raise ConfigError(f"the card fields are not a mteb TaskMetadata: {exc}") from exc
    else:
        meta = card
    rendered = meta.generate_dataset_card()
    # the card data stores arbitrary metadata fields by name: huggingface_hub's CardData.__init__(**kwargs)
    # updates __dict__, so `configs` is not a declared attribute but is exactly what the YAML renders.
    rendered.data.configs = configs  # type: ignore[reportAttributeAccessIssue]
    return rendered.content


__all__ = ["MtebWriter"]
