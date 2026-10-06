"""Rankings: the systems' scores for (query, document) pairs, and :func:`load_rankings`.

A :class:`Rankings` is one table, ``system, dataset, query_id, doc_id, score``. ``system`` names the ranker (one
file often holds several); ``dataset`` names the subset a query belongs to and is empty when the query ids are
unique on their own. Higher scores are better. A ranking given only as an order (a list of document ids) becomes
scores ``n, n-1, ..., 1``, so the order survives every tie rule unchanged.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, field_validator

from rcp_ndcg import storage
from rcp_ndcg.errors import ConfigError, DataError, MissingInputError

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_SYSTEM = "system"
"""The system name of rankings that do not name one."""

RankingsFormat = Literal["auto", "parquet", "trec", "jsonl", "csv"]

_COLUMNS = {
    "system": ("system", "model", "run", "tag", "run_id"),
    "dataset": ("dataset", "subset"),
    "query_id": ("query_id", "query-id", "qid", "query"),
    "doc_id": ("doc_id", "corpus-id", "corpus_id", "docid", "docno"),
    "score": ("score", "rerank_score", "sim"),
}


class RankingRow(BaseModel):
    """One score of one system for one (query, document): a record of :meth:`Rankings.from_records`.

    Attributes:
        query_id: The query id.
        doc_id: The document id.
        score: The system's score; higher is better; finite.
        system: The system (ranker) name; default :data:`DEFAULT_SYSTEM`.
        dataset: The dataset (subset) the query belongs to; ``""`` when the query ids are unique on their own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", coerce_numbers_to_str=True)

    query_id: str
    doc_id: str
    score: float
    system: str = DEFAULT_SYSTEM
    dataset: str = ""

    @field_validator("score")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("not finite")
        return value


class Rankings:
    """The scores of one or more systems on the queries of one or more datasets: one immutable table.

    The table (:meth:`to_pandas`) has the columns ``system, dataset, query_id, doc_id, score``: higher is better,
    every score finite, and at most one score per (system, dataset, query, document). ``dataset`` is ``""`` when the
    query ids need no subset to be unique. The accessors (:meth:`queries`, :meth:`for_query`) read it as
    ``{query_id: {doc_id: score}}``.

    Build it from records (:meth:`from_records`; a pandas frame via ``frame.to_dict("records")``), from
    ``{query_id: {doc_id: score}}`` (:meth:`from_scores`) or orders (:meth:`from_orders`), or read one with
    :func:`load_rankings`; write it with :meth:`save`.
    """

    COLUMNS: tuple[str, ...] = ("system", "dataset", "query_id", "doc_id", "score")

    __slots__ = ("_index", "_table")

    def __init__(self, table: pd.DataFrame) -> None:
        """Wrap a table with the :attr:`COLUMNS` (use the ``from_*`` constructors to build one).

        Raises:
            DataError: A column is missing, a score is not finite, or a (system, dataset, query, document) is
                scored twice.
        """
        import pandas as pd

        missing = [column for column in self.COLUMNS if column not in table.columns]
        if missing:
            raise DataError(f"a rankings table needs the columns {list(self.COLUMNS)}; missing {missing}")
        frame = pd.DataFrame(
            {
                **{column: table[column].astype(str) for column in self.COLUMNS[:4]},
                # A column in gives a column out; pandas annotates to_numeric's result for every input kind.
                "score": cast("pd.Series", pd.to_numeric(table["score"], errors="coerce")).astype(float),
            }
        ).reset_index(drop=True)
        not_finite = frame[~frame["score"].map(math.isfinite)]
        if not not_finite.empty:
            row = not_finite.iloc[0]
            where = f"{row['dataset']}/{row['query_id']}" if row["dataset"] else row["query_id"]
            raise DataError(
                f"system {row['system']!r}, query {where!r}: the score of document {row['doc_id']!r} is not finite",
                hint="drop the documents a system did not score before building the rankings",
                details={"system": row["system"], "dataset": row["dataset"], "query_id": row["query_id"]},
            )
        duplicated = frame.duplicated(list(self.COLUMNS[:4]))
        if duplicated.any():
            row = frame[duplicated].iloc[0]
            raise DataError(
                f"duplicate score for query {row['query_id']!r}, document {row['doc_id']!r} ({row['system']})"
            )
        self._table = frame
        self._index: dict[tuple[str, str], dict[str, dict[str, float]]] | None = None

    # -- construction ------------------------------------------------------

    @classmethod
    def from_records(cls, records: Iterable[RankingRow | Mapping[str, Any]]) -> Rankings:
        """Rankings from :class:`RankingRow` records, validated strictly.

        Args:
            records: Dicts (or :class:`RankingRow`) with ``query_id``, ``doc_id``, ``score`` and optional ``system``
                and ``dataset``; e.g. ``frame.to_dict("records")`` of a pandas frame with those columns.

        Returns:
            The :class:`Rankings`.

        Raises:
            DataError: A record has an unknown key, lacks a field, or holds a score that is no finite number; or a
                (system, dataset, query, document) is scored twice.
        """
        import pandas as pd

        from rcp_ndcg.data._rows import validate_rows

        rows = validate_rows(RankingRow, records, what="rankings")
        frame = pd.DataFrame(
            [(r.system, r.dataset, r.query_id, r.doc_id, r.score) for r in rows], columns=list(cls.COLUMNS)
        )
        return cls(frame)

    @classmethod
    def from_scores(
        cls, scores: Mapping[str, Mapping[str, float]], *, system: str = DEFAULT_SYSTEM, dataset: str = ""
    ) -> Rankings:
        """One system's rankings from ``{query_id: {doc_id: score}}``."""
        return cls.from_records(
            {"system": system, "dataset": dataset, "query_id": q, "doc_id": d, "score": s}
            for q, docs in scores.items()
            for d, s in docs.items()
        )

    @classmethod
    def from_orders(
        cls, orders: Mapping[str, Sequence[str]], *, system: str = DEFAULT_SYSTEM, dataset: str = ""
    ) -> Rankings:
        """One system's rankings from ``{query_id: [doc_id, ...]}`` (best first): scores ``n, ..., 1``."""
        return cls.from_scores({q: _order_scores(docs) for q, docs in orders.items()}, system=system, dataset=dataset)

    @classmethod
    def concat(cls, parts: Iterable[Rankings]) -> Rankings:
        """One rankings holding every row of ``parts`` (a system appearing in two parts must not score one pair
        twice)."""
        import pandas as pd

        frames = [part._table for part in parts]
        return cls(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(cls.COLUMNS)))

    # -- the table ---------------------------------------------------------

    def to_pandas(self) -> pd.DataFrame:
        """The table ``system, dataset, query_id, doc_id, score`` (a copy)."""
        return self._table.copy()

    def __len__(self) -> int:
        return len(self._table)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rankings):
            return NotImplemented
        keys = list(self.COLUMNS[:4])
        mine = self._table.sort_values(keys).reset_index(drop=True)
        theirs = other._table.sort_values(keys).reset_index(drop=True)
        return mine.equals(theirs)

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        systems = self.systems
        shown = ", ".join(systems[:5]) + (f", ... ({len(systems)} systems)" if len(systems) > 5 else "")
        queries = self._table.groupby(["dataset", "query_id"]).ngroups if len(self) else 0
        return f"Rankings({len(self)} scores, {queries} queries, systems=[{shown}])"

    def save(self, path: str | Path, *, format: RankingsFormat = "auto") -> str:  # noqa: A002 (as load_rankings)
        """Write the rankings where :func:`load_rankings` reads them back.

        Args:
            path: A local path or a storage URI.
            format: ``parquet`` or ``csv`` (the table of :meth:`to_pandas`), ``jsonl`` (one row per query:
                ``system``, ``dataset``, ``query_id``, ``scores``), or ``"auto"`` from the extension.

        Returns:
            The path written.
        """
        uri = str(path)
        kind = _format_of(uri) if format == "auto" else format
        storage.makedirs(storage.parent(uri))
        if kind == "parquet":
            with storage.open_path(uri, "wb") as handle:
                self._table.to_parquet(handle, index=False)
        elif kind == "csv":
            with storage.open_path(uri, "w") as handle:
                self._table.to_csv(handle, index=False, sep="\t" if uri.endswith(".tsv") else ",")
        elif kind == "jsonl":
            lines = [
                json.dumps({"system": system, "dataset": dataset, "query_id": query_id, "scores": docs})
                for (system, dataset), queries in self._grouped().items()
                for query_id, docs in queries.items()
            ]
            storage.write_text(uri, "".join(line + "\n" for line in lines))
        else:
            raise DataError(f"cannot write rankings as {kind!r}; use parquet, csv or jsonl")
        return uri

    # -- accessors ---------------------------------------------------------

    @property
    def systems(self) -> list[str]:
        """The system names, in input order."""
        return list(dict.fromkeys(self._table["system"]))

    @property
    def datasets(self) -> list[str]:
        """The dataset names, in input order (``""``: rankings whose query ids need no dataset)."""
        return list(dict.fromkeys(self._table["dataset"]))

    def queries(self, *, system: str | None = None, dataset: str | None = None) -> dict[str, dict[str, float]]:
        """``{query_id: {doc_id: score}}`` of one system on one dataset.

        Args:
            system: The system; may be omitted when the rankings hold one.
            dataset: The dataset (subset) whose queries to read: the rows that name it, else the rows that name no
                dataset (see :meth:`resolve_dataset`). ``None``: the one dataset the system's rows name.

        Returns:
            The system's scores on that dataset; empty when other systems rank the dataset and this one does not.

        Raises:
            DataError: The system is unknown or not named among several; ``dataset`` is ``None`` and the system's
                rows name several datasets; or no row can rank ``dataset``.
        """
        name = self._one_system(system)
        key = self._dataset_key(name, dataset)
        return {q: dict(docs) for q, docs in self._grouped().get((name, key), {}).items()}

    def for_query(self, query_id: str, *, system: str | None = None, dataset: str | None = None) -> dict[str, float]:
        """``{doc_id: score}`` of one query (``system`` and ``dataset`` as in :meth:`queries`); empty when the
        system did not rank it."""
        name = self._one_system(system)
        return dict(self._grouped().get((name, self._dataset_key(name, dataset)), {}).get(query_id, {}))

    def resolve_dataset(self, dataset: str) -> str | None:
        """The ``dataset`` value of the rows that rank ``dataset``'s queries.

        Rows that name ``dataset`` rank it. Without them, rows that name no dataset (``""``, e.g. a TREC run) rank
        any dataset, since their query ids are taken to be unique on their own.

        Args:
            dataset: A dataset (subset) name.

        Returns:
            ``dataset`` when rows name it, else ``""`` when some rows name no dataset, else ``None``.
        """
        datasets = self.datasets
        if dataset in datasets:
            return dataset
        return "" if "" in datasets else None

    def _dataset_key(self, system: str, dataset: str | None) -> str:
        if dataset is None:
            named = list(dict.fromkeys(ds for s, ds in self._grouped() if s == system))
            if len(named) > 1:
                raise DataError(
                    f"the rankings of system {system!r} span {len(named)} datasets {named}; name one",
                    hint="pass dataset=<one of them>",
                    cli_hint="keep only the rows of the dataset the command reads (one file per dataset)",
                    details={"system": system, "datasets": named},
                )
            return named[0] if named else ""
        key = self.resolve_dataset(dataset)
        if key is None:
            raise no_rankings_error(dataset, self.datasets)
        return key

    def top(self, depth: int) -> Rankings:
        """The ``depth`` best-scored documents of every query (ties broken by document id, descending).

        Raises:
            ConfigError: ``depth`` is not positive: zero would return an empty table, and a negative one would
                silently keep all but the last ``|depth|`` rows (pandas' ``head`` semantics).
        """
        if depth <= 0:
            raise ConfigError(f"depth must be positive, got {depth}")
        ordered = self._table.sort_values(["score", "doc_id"], ascending=False, kind="mergesort")
        kept = ordered.groupby(["system", "dataset", "query_id"], sort=False).head(depth)
        return Rankings(kept.sort_index())

    def _grouped(self) -> dict[tuple[str, str], dict[str, dict[str, float]]]:
        if self._index is None:
            index: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
            columns = (self._table[column].tolist() for column in self.COLUMNS)
            for system, dataset, query_id, doc_id, score in zip(*columns, strict=True):
                index.setdefault((system, dataset), {}).setdefault(query_id, {})[doc_id] = score
            self._index = index
        return self._index

    def _one_system(self, system: str | None) -> str:
        systems = self.systems
        if system is not None:
            if system not in systems:
                raise DataError(f"no rankings of system {system!r}; systems: {systems}")
            return system
        if len(systems) != 1:
            raise DataError(f"the rankings hold {len(systems)} systems; name one of {systems}")
        return systems[0]


def no_rankings_error(
    dataset: str, datasets: Sequence[str], *, system: str | None = None, hint: str | None = None
) -> DataError:
    """The error for a reader that finds no rows naming ``dataset``: shared by :meth:`Rankings.queries` and
    :func:`rcp_ndcg.eval.evaluate`, so both name the datasets the rows do name.

    Args:
        dataset: The dataset (subset) whose rows were wanted.
        datasets: The dataset names the rows do name.
        system: The system whose rows were wanted, when the reader scores one system at a time; the message names
            it so a file of several systems says which one is missing.
        hint: The next step, when the reader knows more than :meth:`Rankings.queries` does (e.g. the exact subset
            name the ``dataset`` column must hold); ``None`` keeps the rankings' own hint.

    Returns:
        The :class:`~rcp_ndcg.errors.DataError` to raise (exit 12 on the command line).
    """
    details: dict[str, Any] = {"dataset": dataset, "datasets": list(datasets)}
    if system is not None:
        details = {"system": system, **details}
    return DataError(
        f"{'' if system is None else f'system {system!r}: '}no rankings of dataset {dataset!r}; "
        f"the rankings name the datasets {list(datasets)}",
        hint=hint or "rank the dataset's queries, or name one of those datasets",
        details=details,
    )


def _from_frame(frame: pd.DataFrame) -> Rankings:
    """Rankings from a file's table: ``query_id``, ``doc_id``, ``score``, optional ``system`` and ``dataset`` (or
    their aliases, e.g. the HF column names ``query-id``, ``corpus-id``, ``model``)."""
    renamed = frame.rename(columns=_column_map(list(frame.columns)))
    if "system" not in renamed.columns:
        renamed = renamed.assign(system=DEFAULT_SYSTEM)
    if "dataset" not in renamed.columns:
        renamed = renamed.assign(dataset="")
    renamed = renamed.assign(
        system=renamed["system"].fillna(DEFAULT_SYSTEM).replace("", DEFAULT_SYSTEM),
        dataset=renamed["dataset"].fillna(""),
    )
    return Rankings(renamed)


def _from_file_rows(rows: Iterable[Mapping[str, Any]], uri: str) -> Rankings:
    """Rankings from rows parsed out of a file (TREC, JSONL): unset system and dataset take their defaults."""
    records = [{key: value for key, value in row.items() if value is not None} for row in rows]
    try:
        return Rankings.from_records(records)
    except DataError as exc:
        raise DataError(f"{uri}: {exc.message}", hint=exc.hint, details=exc.details) from None


def _order_scores(docs: Sequence[str]) -> dict[str, float]:
    if len(set(docs)) != len(docs):
        raise DataError(f"a ranking lists a document twice: {[d for d in docs if docs.count(d) > 1][:5]}")
    return {doc_id: float(len(docs) - rank) for rank, doc_id in enumerate(docs)}


def _column_map(columns: list[str]) -> dict[str, str]:
    """``{source column: canonical column}`` for the first matching alias of each canonical column."""
    out: dict[str, str] = {}
    for canonical, aliases in _COLUMNS.items():
        found = next((c for c in aliases if c in columns), None)
        if found is not None:
            out[found] = canonical
        elif canonical in ("query_id", "doc_id", "score"):
            raise DataError(f"a rankings table needs a {canonical!r} column (one of {aliases}); columns: {columns}")
    return out


def load_rankings(
    path: str | Path,
    *,
    format: RankingsFormat = "auto",  # noqa: A002 (spec name)
    dataset: str | None = None,
) -> Rankings:
    """Load rankings from a file.

    The ``dataset`` column names each row's subset. A file of a suite whose subsets share query ids (BRIGHT,
    ViDoRe v3, NanoBEIR) needs it, since a query id alone does not say which subset it belongs to. A format without
    the column (a TREC run) holds one subset per file: pass ``dataset=`` to name it, and join the files with
    :meth:`Rankings.concat`.

    Formats:

    * ``parquet``, ``csv`` (also ``.tsv``): a table with ``query_id``, ``doc_id``, ``score`` and optional
      ``system``, ``dataset`` (alias ``subset``). The HF column names (``query-id``, ``corpus-id``, ``model``) are
      accepted too.
    * ``trec``: a TREC run, ``qid Q0 doc_id rank score tag``; the tag is the system. It names no dataset.
    * ``jsonl``: rows ``{"query_id", "scores": {doc_id: score}}``, ``{"query_id", "doc_ids": [...]}`` (an order;
      with an aligned ``"scores"`` list when the scores are known), or flat ``{"query_id", "doc_id", "score"}``;
      each optionally with ``"system"`` and ``"dataset"``. A single JSON document ``{query_id: {doc_id: score}}``
      or ``{query_id: [doc_id, ...]}`` is read the same way.

    Args:
        path: A local path or a storage URI (``gs://``, ``s3://``).
        format: One of the formats above; ``"auto"`` picks it from the extension.
        dataset: The dataset (subset) of every row, for a file whose rows name none.

    Returns:
        The :class:`Rankings`.

    Raises:
        MissingInputError: The file does not exist.
        DataError: The file is malformed, or ``dataset`` is given for a file whose rows name a dataset.
    """
    rankings = _load(str(path), format)
    if dataset is None:
        return rankings
    named = [name for name in rankings.datasets if name]
    if named:
        raise DataError(
            f"{path}: its rows already name the datasets {named}; dataset={dataset!r} applies to a file that names none"
        )
    return Rankings(rankings.to_pandas().assign(dataset=dataset))


def _load(uri: str, format: RankingsFormat) -> Rankings:  # noqa: A002 (as load_rankings)
    if not storage.exists(uri):
        raise MissingInputError(f"rankings file not found: {uri}")
    kind = _format_of(uri) if format == "auto" else format
    if kind == "parquet":
        import pandas as pd

        with storage.open_path(uri, "rb") as handle:
            return _from_frame(pd.read_parquet(handle))
    if kind == "csv":
        import pandas as pd

        with storage.open_path(uri, "r") as handle:
            frame = pd.read_csv(handle, sep="\t" if uri.endswith(".tsv") else ",", dtype=str, keep_default_na=False)
        return _from_frame(frame)
    if kind == "trec":
        return _from_file_rows(_trec_rows(storage.read_text(uri), uri), uri)
    if kind == "jsonl":
        return _from_json_text(storage.read_text(uri), uri)
    raise DataError(f"unknown rankings format {format!r}; expected auto, parquet, trec, jsonl or csv")


def _format_of(uri: str) -> str:
    suffix = uri.rsplit(".", 1)[-1].lower() if "." in uri.rsplit("/", 1)[-1] else ""
    formats = {"parquet": "parquet", "csv": "csv", "tsv": "csv", "jsonl": "jsonl", "json": "jsonl"}
    formats.update(dict.fromkeys(("trec", "run", "txt"), "trec"))
    if suffix not in formats:
        raise DataError(f"cannot tell the rankings format of {uri!r}; pass format= (parquet, trec, jsonl, csv)")
    return formats[suffix]


def _trec_rows(text: str, uri: str) -> Iterator[dict[str, Any]]:
    for number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 6:
            raise DataError(f"{uri}:{number}: a TREC run line has 6 fields (qid Q0 doc rank score tag), got {line!r}")
        query_id, _, doc_id, _, score, tag = fields
        yield {"system": tag, "query_id": query_id, "doc_id": doc_id, "score": score}


def _from_json_text(text: str, uri: str) -> Rankings:
    text = text.strip()
    if not text:
        return Rankings.from_records([])
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        document = None
    if isinstance(document, dict) and all(isinstance(v, list | dict) for v in document.values()):
        return _from_mapping(document, uri)  # {query_id: {doc_id: score}} or {query_id: [doc_id, ...]}
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"{uri}:{number}: not a JSON object: {exc}") from exc
        if not isinstance(record, dict):
            raise DataError(f"{uri}:{number}: a rankings row is a JSON object, got {line[:200]}")
        rows.extend(_json_record_rows(record, f"{uri}:{number}"))
    return _from_file_rows(rows, uri)


def _from_mapping(document: Mapping[str, Any], uri: str) -> Rankings:
    rows = []
    for query_id, docs in document.items():
        scores = _order_scores(list(docs)) if isinstance(docs, list) else docs
        rows += [{"query_id": query_id, "doc_id": d, "score": s} for d, s in scores.items()]
    return _from_file_rows(rows, uri)


def _json_record_rows(record: Mapping[str, Any], where: str) -> list[dict[str, Any]]:
    query_id = record.get("query_id", record.get("id"))
    if query_id is None:
        raise DataError(f"{where}: a rankings row needs a 'query_id'")
    common = {"query_id": query_id, "system": record.get("system"), "dataset": record.get("dataset")}
    if "doc_id" in record:
        if "score" not in record:
            raise DataError(f"{where}: a rankings row with a 'doc_id' needs its 'score'")
        return [{**common, "doc_id": record["doc_id"], "score": record["score"]}]
    scores = record.get("scores")
    order = record.get("doc_ids", record.get("ranking"))
    if isinstance(scores, Mapping):
        pairs = scores.items()
    elif order is not None and isinstance(scores, list):
        if len(scores) != len(order):
            raise DataError(f"{where}: 'scores' and 'doc_ids' differ in length")
        pairs = zip(order, scores, strict=True)
    elif order is not None:
        pairs = _order_scores(list(order)).items()
    else:
        raise DataError(f"{where}: a rankings row needs 'scores', 'doc_ids' or 'doc_id'")
    return [{**common, "doc_id": d, "score": s} for d, s in pairs]


__all__ = ["DEFAULT_SYSTEM", "RankingRow", "Rankings", "RankingsFormat", "load_rankings", "no_rankings_error"]
