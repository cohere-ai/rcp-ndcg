"""``rcp-ndcg data``: fetch, inspect, validate and convert datasets.

``inspect`` and ``validate`` take ``--dataset`` as every command does: a URI that
:func:`rcp_ndcg.data.load_dataset` reads (``hf://``, ``suite:``, ``beir:``, ``jsonl:``, ``images:``, ...), with
``--subset`` and ``--revision`` for a Hub dataset. ``inspect --dataset suite:<name>`` without a subset lists the
suite's subsets.

``fetch`` downloads a public suite (``nanobeir``, ``bright``, ``vidore``, ``trecdl``, whose HuggingFace repository
:data:`rcp_ndcg.data.SUITES` names) or ``hf://<org>/<repo>`` into the Hub cache.

``convert`` reads a source through the reader registry (:mod:`rcp_ndcg.data.io`) and writes it in a layout
``load_dataset`` reads (``jsonl:<out>`` or ``beir:<out>``).
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain
from pathlib import Path
from typing import Any, Literal

import click
from pydantic import BaseModel, Field

from rcp_ndcg.cli._args import DatasetInput
from rcp_ndcg.cli.command import command
from rcp_ndcg.data.validate import ValidationReport
from rcp_ndcg.errors import DataError, UsageError

_FETCH_HELP = (
    "A suite (nanobeir, bright, vidore, trecdl), hf://<org>/<repo>, or tiny (the example dataset that ships with "
    "the package)."
)


def _suite(dataset: str) -> str | None:
    from rcp_ndcg.data import SUITES

    name = dataset.removeprefix("suite:")
    return name if name in SUITES else None


def _repo_id(dataset: str) -> str | None:
    """The HuggingFace dataset repository of a suite name (an ``hf://`` URI goes through
    :func:`_hub_locator`)."""
    from rcp_ndcg.data import SUITES

    suite = _suite(dataset)
    return SUITES[suite].repo if suite is not None else None


def _hub_locator(dataset: str) -> tuple[str, str | None, str | None]:
    """``(repo, subset, revision)`` of an ``hf://<org>/<repo>[/<subset>][@<revision>]`` URI.

    The data layer's own URI grammar (:mod:`rcp_ndcg.data.dataset`, :mod:`rcp_ndcg.data.revisions`): a third
    path segment is the subset and ``@rev`` the revision. A URI outside the grammar is a UsageError naming
    the form, never a repository id handed to the Hub (which answered with an untyped HFValidationError,
    classified as an internal error).
    """
    locator = dataset.removeprefix("hf://").strip("/")
    path, _, revision = locator.partition("@")
    parts = path.split("/")
    if len(parts) not in (2, 3) or not all(parts):
        raise UsageError(
            f"{dataset!r} is not hf://<org>/<repo>[/<subset>][@<revision>]",
            hint="e.g. hf://org/name, hf://org/name/subset@main; --subset and --revision name the same parts",
        )
    return "/".join(parts[:2]), parts[2] if len(parts) == 3 else None, revision or None


def _one_value(what: str, from_uri: str | None, given: str | None) -> str | None:
    """The one value of *what*: the URI's when only it is given, the flag's when only it is, both when equal."""
    if from_uri is not None and given is not None and from_uri != given:
        raise UsageError(f"two {what}s for one dataset: {from_uri!r} in the URI and {given!r}")
    return from_uri if from_uri is not None else given


# ----------------------------------------------------------------------------------------------------------------
# fetch
# ----------------------------------------------------------------------------------------------------------------


class DataFetchRequest(BaseModel):
    dataset: str = Field(description=_FETCH_HELP)
    subset: str | None = Field(default=None, description="Only this subset directory (default: the whole repo).")
    revision: str | None = Field(default=None, description="Hub revision: a commit, tag or branch (default main).")
    out: str | None = Field(default=None, description="Copy into this directory instead of the Hub cache.")


class DataFetch(BaseModel):
    """Where the downloaded files are."""

    dataset: str
    repo_id: str | None = Field(description="The HuggingFace repository; null for the packaged example.")
    subset: str | None
    revision: str | None
    path: str = Field(description="The local directory holding the repository's files.")
    files: int = Field(description="Number of files under that directory.")


@command("data fetch", request=DataFetchRequest, result=DataFetch, read_only=False)
def data_fetch(request: DataFetchRequest) -> DataFetch:
    """Download a public suite or a HuggingFace dataset repository into the Hub cache (or --out); `tiny` gives the
    packaged example dataset (copied into --out)."""
    if request.dataset == "tiny":
        return _fetch_example(request)
    repo_id = _repo_id(request.dataset)
    subset, revision = request.subset, request.revision
    if repo_id is None and request.dataset.startswith("hf://"):
        repo_id, uri_subset, uri_revision = _hub_locator(request.dataset)
        subset = _one_value("subset", uri_subset, subset)
        revision = _one_value("revision", uri_revision, revision)
    if repo_id is None:
        raise UsageError(f"cannot fetch {request.dataset!r}", hint=_FETCH_HELP)
    from huggingface_hub import snapshot_download

    patterns = None if subset is None else [f"{subset}/*", f"{subset}__*/*"]
    path = Path(
        snapshot_download(
            repo_id, repo_type="dataset", revision=revision, allow_patterns=patterns, local_dir=request.out
        )
    )
    files = sum(1 for item in path.rglob("*") if item.is_file() and ".cache" not in item.parts)
    return DataFetch(
        dataset=request.dataset,
        repo_id=repo_id,
        subset=subset,
        revision=revision,
        path=str(path),
        files=files,
    )


def _fetch_example(request: DataFetchRequest) -> DataFetch:
    import shutil

    from rcp_ndcg import examples

    if request.subset is not None or request.revision is not None:
        raise UsageError(
            "the packaged example has no subsets or revisions",
            hint="copy it with --dataset tiny --out DIR and read its files",
        )
    source = examples.tiny()
    target = source
    if request.out is not None:
        target = Path(request.out)
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.is_file():
                shutil.copy2(item, target / item.name)
    files = sum(1 for item in target.iterdir() if item.is_file())
    return DataFetch(dataset="tiny", repo_id=None, subset=None, revision=None, path=str(target), files=files)


# ----------------------------------------------------------------------------------------------------------------
# inspect
# ----------------------------------------------------------------------------------------------------------------


class DataInspectRequest(DatasetInput):
    """``--dataset URI`` (``--subset``, ``--revision``): any dataset ``load_dataset`` reads."""


class Stats(BaseModel):
    """Minimum, mean and maximum of a quantity."""

    min: float
    mean: float
    max: float


class DatasetSummary(BaseModel):
    """What a dataset holds: counts, candidate pools, labels and gains; or a suite's subsets."""

    dataset: str
    kind: Literal["dataset", "suite-subsets"]
    subset: str | None = None
    revision: str | None = None
    subsets: list[str] | None = Field(default=None, description="The suite's subsets (kind suite-subsets).")
    queries: int | None = Field(default=None, description="Queries with a candidate pool or a positive label.")
    judged_documents: int | None = Field(default=None, description="(query, document) pairs with a gain.")
    positive_labels: int | None = Field(default=None, description="(query, document) pairs with a positive label.")
    labels: dict[str, int] | None = Field(default=None, description="Count of each positive human label value.")
    pool_depth: Stats | None = Field(default=None, description="Candidates per query.")
    gains: Stats | None = Field(default=None, description="Calibrated gains g(theta) in [0, 1] over the pools.")
    excluded: int | None = Field(default=None, description="(query, document) pairs removed by the protocol.")


def _stats(values: list[float]) -> Stats | None:
    return Stats(min=min(values), mean=sum(values) / len(values), max=max(values)) if values else None


def _label(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _positive(dataset: Any) -> dict[str, dict[str, float]]:
    return {q: {d: g for d, g in docs.items() if g > 0} for q, docs in dataset.qrels.items()}


def _suite_subsets(request: DatasetInput) -> list[str] | None:
    """The subsets of ``suite:<name>`` named without one, from the suite table (nothing is downloaded)."""
    from rcp_ndcg.data import SUITES

    scheme, _, name = request.dataset.partition(":")
    if scheme == "suite" and request.subset is None and name in SUITES:
        return list(SUITES[name].subsets)
    return None


@command("data inspect", request=DataInspectRequest, result=DatasetSummary)
def data_inspect(request: DataInspectRequest) -> DatasetSummary:
    """Summarise a dataset: queries, candidate pools, label and gain distributions, excluded ids."""
    subsets = _suite_subsets(request)
    if subsets is None:
        dataset = request.load()
        if dataset.subsets:
            subsets = [part.name for part in dataset.subsets]
        else:
            return _dataset_summary(request, dataset)
    return DatasetSummary(dataset=request.dataset, kind="suite-subsets", revision=request.revision, subsets=subsets)


def _dataset_summary(request: DataInspectRequest, dataset: Any) -> DatasetSummary:
    """The summary of one loaded dataset (not a suite)."""
    positive = _positive(dataset)
    candidates = dataset.candidates or {}
    gains = dataset.gains or {}
    labels: dict[str, int] = {}
    for docs in positive.values():
        for value in docs.values():
            labels[_label(value)] = labels.get(_label(value), 0) + 1
    return DatasetSummary(
        dataset=request.dataset,
        kind="dataset",
        subset=dataset.name,
        revision=request.revision,
        queries=len(set(candidates) | {query for query, docs in positive.items() if docs}),
        judged_documents=sum(len(docs) for docs in gains.values()),
        positive_labels=sum(len(docs) for docs in positive.values()),
        labels=dict(sorted(labels.items())),
        pool_depth=_stats([float(len(pool)) for pool in candidates.values()]),
        gains=_stats([gain for docs in gains.values() for gain in docs.values()]),
        excluded=sum(len(ids) for ids in dataset.excluded.values()),
    )


# ----------------------------------------------------------------------------------------------------------------
# validate
# ----------------------------------------------------------------------------------------------------------------


class DataValidateRequest(DatasetInput):
    rankings: str | None = Field(
        default=None, description="Rankings to check against the dataset (parquet, TREC run, JSONL or CSV)."
    )


@command("data validate", request=DataValidateRequest, result=ValidationReport)
def data_validate(request: DataValidateRequest) -> ValidationReport:
    """Check a dataset (and optionally rankings) against the scoring protocol; exit 12 on an error."""
    from rcp_ndcg.data import load_rankings, validate

    if _suite_subsets(request) is not None:
        raise UsageError("validating a suite needs --subset", hint="list them with `rcp-ndcg data inspect`")
    dataset = request.load()
    if dataset.subsets:
        raise UsageError("validating a suite needs --subset", hint="list them with `rcp-ndcg data inspect`")
    report = validate(dataset, load_rankings(request.rankings) if request.rankings is not None else None)
    if not report.ok:
        raise DataError(
            f"{request.dataset}: {', '.join(report.errors)}",
            hint="the numbers computed from this input would be wrong; see details.checks",
            details=report.model_dump(mode="json"),
        )
    return report


# ----------------------------------------------------------------------------------------------------------------
# convert and formats
# ----------------------------------------------------------------------------------------------------------------


class DataConvertRequest(BaseModel):
    source: str = Field(description="What to read: a path or URI the reader understands.")
    format: str = Field(description="The reader; see `rcp-ndcg data formats`.")
    out: str = Field(description="Destination directory or file (local path or URI).")
    to: str = Field(default="jsonl", description="The writer.")
    shape: Literal["corpus", "ranking"] | None = Field(
        default=None, description="What to write; default the reader's native shape (the other is derived)."
    )
    set: list[str] = Field(
        default_factory=list,
        description="Reader option KEY=VALUE (repeatable), e.g. --set split=test or --set dpi=150 (pdf).",
    )
    limit: int | None = Field(default=None, ge=1, description="Stop after this many records (a smoke conversion).")
    dry_run: bool = Field(default=False, description="Report what would be written; write nothing.")


class Conversion(BaseModel):
    """What `data convert` wrote (or, with --dry-run, would write)."""

    source: str
    out: str
    reader: str
    writer: str
    shape: Literal["corpus", "ranking"]
    written: int | None = Field(description="Records written; null for a dry run.")
    queries: int | None = Field(
        default=None, description="Queries written (corpus shape, after --limit trims); null for a dry run."
    )
    labels: int | None = Field(
        default=None,
        description="(query, document) labels written (corpus shape, after --limit trims); null for a dry run.",
    )
    limit: int | None = Field(
        default=None,
        description="The --limit cap the conversion ran under (a smoke conversion); null without it.",
    )
    dry_run: bool


def _scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def _options(pairs: list[str], *, flag: str) -> dict[str, Any]:
    """``key=value`` pairs as a dict, with ints, floats, booleans and ``null`` coerced."""
    options: dict[str, Any] = {}
    for pair in pairs:
        key, separator, raw = pair.partition("=")
        if not separator or not key.strip():
            raise UsageError(
                f"{flag} expects KEY=VALUE, got {pair!r}",
                hint="pass one pair per option: --opt KEY=VALUE",
            )
        options[key.strip()] = _scalar(raw.strip())
    return options


def _target_shape(requested: str | None, reader: Any, writer: Any) -> Any:
    from rcp_ndcg.data.io import DataShape

    if requested is not None:
        target = DataShape(requested)
    elif DataShape.RANKING in reader.shapes and DataShape.RANKING in writer.shapes:
        target = DataShape.RANKING
    else:
        target = DataShape.CORPUS
    if target not in reader.shapes:
        raise UsageError(
            f"reader {reader.name!r} cannot serve the {target} shape (it declares {sorted(reader.shapes)})",
            hint="pass a shape both sides take, or pass --shape",
        )
    if target not in writer.shapes:
        raise UsageError(
            f"writer {writer.name!r} cannot write the {target} shape (it declares {sorted(writer.shapes)})",
            hint="pass a shape both sides take, or pass --shape",
        )
    return target


def _limited(records: Iterator[Any], limit: int | None) -> Iterator[Any]:
    for index, record in enumerate(records):
        if limit is not None and index >= limit:
            return
        yield record


def _nonempty(records: Iterator[Any], reader: Any, shape: Any) -> Iterator[Any]:
    """Refuse to write an empty dataset, before the sink is opened (an empty file is an artefact that gets
    mistaken for real data later)."""
    from rcp_ndcg.data.io import DataShape

    iterator = iter(records)
    try:
        first = next(iterator)
    except StopIteration:
        hint = (
            "the ranking shape is derived from judged candidate lists, so a source without qrels yields nothing: "
            "convert it with --shape corpus"
            if shape is DataShape.RANKING
            else "check the split names and the source"
        )
        raise DataError(f"reader {reader.name!r} yielded no {shape} records", hint=hint) from None
    return chain([first], iterator)


@command("data convert", request=DataConvertRequest, result=Conversion, read_only=False)
def data_convert(request: DataConvertRequest) -> Conversion:
    """Convert between dataset formats -- the readers and writers of the rcp_ndcg.readers and
    rcp_ndcg.writers entry-point groups; `rcp-ndcg data formats` lists them."""
    from rcp_ndcg.data.io import (
        DataShape,
        available_readers,
        available_writers,
        get_reader,
        get_writer,
        reader_options,
        unknown_reader_options,
    )

    if request.format not in available_readers():
        raise UsageError(f"unknown format {request.format!r}", hint=f"readers: {', '.join(available_readers())}")
    if request.to not in available_writers():
        raise UsageError(f"unknown writer {request.to!r}", hint=f"writers: {', '.join(available_writers())}")
    options = _options(request.set, flag="--set")
    unknown = unknown_reader_options(request.format, options)
    if unknown:
        raise UsageError(
            f"the {request.format} reader takes no option {', '.join(map(repr, unknown))}",
            hint=f"its options (--set KEY=VALUE): {', '.join(sorted(reader_options(request.format) or ()))}",
        )

    reader = get_reader(request.format, uri=request.source, **options)
    writer = get_writer(request.to)
    shape = _target_shape(request.shape, reader, writer)
    written: int | None = None
    queries: int | None = None
    labels: int | None = None
    if not request.dry_run:
        if shape is DataShape.RANKING:
            records = _nonempty(_limited(reader.examples(), request.limit), reader, shape)
            written = writer.write_examples(records, request.out)
        else:
            written, queries, labels = _write_corpus(writer, reader, request, shape)
    return Conversion(
        source=request.source,
        out=request.out,
        reader=reader.name,
        writer=writer.name,
        shape="ranking" if shape is DataShape.RANKING else "corpus",
        written=written,
        queries=queries,
        labels=labels,
        limit=request.limit,
        dry_run=request.dry_run,
    )


def _write_corpus(writer: Any, reader: Any, request: DataConvertRequest, shape: Any) -> tuple[int, int, int]:
    """Write the corpus shape, carrying every table the source has, and return ``(documents, queries, labels)``.

    A writer that exposes ``write_dataset`` (the MTEB layout) gets the Dataset built through the one reader
    path, so its pools, exclusions, gains, thetas, subset, split and task instruction are written. Any other
    writer gets the tables its ``write_corpus`` accepts, and a table it cannot express is refused with a hint
    rather than dropped silently. With ``--limit`` the documents, labels and queries are trimmed together:
    writing every query and qrel beside N documents names documents the corpus lacks, and the result does not
    load (``Dataset`` refuses it: qrels name documents that are not in the corpus).
    """
    from rcp_ndcg.data.dataset import _from_reader

    documents = list(_limited(reader.documents(), request.limit))
    records = _nonempty(iter(documents), reader, shape)
    qrels = reader.qrels()
    queries = list(reader.queries())
    if request.limit is not None:
        kept = {document.doc_id for document in documents}
        qrels = {q: {d: g for d, g in judged.items() if d in kept} for q, judged in qrels.items()}
        qrels = {q: judged for q, judged in qrels.items() if judged}
        queries = [query for query in queries if query.query_id in qrels]
    if hasattr(writer, "write_dataset"):
        dataset = (
            _from_reader(reader, uri=request.source)
            if request.limit is None
            else _limited_dataset(reader, documents, queries, qrels)
        )
        written = writer.write_dataset(dataset, request.out)
    else:
        _refuse_unexpressible_tables(writer, reader)
        written = writer.write_corpus(records, queries, qrels, request.out)
    return written, len(queries), sum(len(judged) for judged in qrels.values())


def _limited_dataset(reader: Any, documents: list[Any], queries: list[Any], qrels: Any) -> Any:
    """The Dataset of a ``--limit`` corpus conversion: the N documents with the labels, queries, pools and
    exclusions trimmed to them (a smoke conversion must not name documents its corpus lacks)."""
    from rcp_ndcg.data.dataset import Dataset

    kept = {document.doc_id for document in documents}
    gains = {q: {d: g for d, g in judged.items() if d in kept} for q, judged in (reader.gains() or {}).items()}
    thetas = {q: {d: t for d, t in judged.items() if d in kept} for q, judged in (reader.thetas() or {}).items()}
    candidates = {
        q: [doc_id for doc_id in ids if doc_id in kept]
        for q, ids in (reader.candidates() or {}).items()
        if q in qrels
    }
    excluded = {
        q: [doc_id for doc_id in ids if doc_id in kept]
        for q, ids in reader.excluded().items()
        if q in qrels
    }
    rows = [
        {
            "query_id": query_id,
            "doc_id": doc_id,
            "grade": grade,
            "gain": gains.get(query_id, {}).get(doc_id),
            "theta": thetas.get(query_id, {}).get(doc_id),
        }
        for query_id, judged in qrels.items()
        for doc_id, grade in judged.items()
    ]
    provenance = reader.provenance
    return Dataset.from_records(
        name=reader.dataset_name,
        queries=queries,
        corpus=documents,
        qrels=rows,
        candidates=candidates or None,
        excluded=excluded or None,
        subset=provenance.subset,
        split=provenance.split,
        task=reader.task,
        task_instruction=reader.task_instruction,
    )


def _refuse_unexpressible_tables(writer: Any, reader: Any) -> None:
    """Refuse a conversion whose source carries a table the writer's ``write_corpus`` cannot express.

    The tables checked are the ones that change what a model sees or how it is scored -- the exclusions, the
    calibrated gains and thetas, and the task instruction. A source's candidate lists are not counted: the
    corpus shape is derived from them (the ranking-to-corpus derivation), so they are consumed, not dropped.
    """
    dropped = [
        name
        for name, value in (
            ("excluded", reader.excluded()),
            ("gains", reader.gains()),
            ("thetas", reader.thetas()),
            ("task_instruction", reader.task_instruction),
        )
        if value
    ]
    if dropped:
        raise DataError(
            f"writer {writer.name!r} cannot carry the {', '.join(dropped)} this source has: converting "
            "would drop them silently",
            hint="export to `mteb`, whose layout carries pools, exclusions, gains, thetas and the task "
            "instruction, or trim the source first",
        )


class DataFormatsRequest(BaseModel):
    """No inputs."""


class DataFormats(BaseModel):
    """The registered readers and writers."""

    readers: list[str]
    writers: list[str]


@command("data formats", request=DataFormatsRequest, result=DataFormats)
def data_formats(request: DataFormatsRequest) -> DataFormats:
    """List the registered readers and writers."""
    from rcp_ndcg.data.io import available_readers, available_writers

    return DataFormats(readers=available_readers(), writers=available_writers())


@click.group(name="data", help="Fetch, inspect, validate and convert datasets.")
def data_group() -> None:
    """``rcp-ndcg data``."""


for _command in (data_fetch, data_inspect, data_validate, data_convert, data_formats):
    data_group.add_command(_command)


__all__ = [
    "Conversion",
    "DataConvertRequest",
    "DataFetch",
    "DataFetchRequest",
    "DataFormats",
    "DataFormatsRequest",
    "DataInspectRequest",
    "DataValidateRequest",
    "DatasetSummary",
    "Stats",
    "data_group",
]
