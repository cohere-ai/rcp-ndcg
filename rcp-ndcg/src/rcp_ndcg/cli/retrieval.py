"""``rcp-ndcg retrieval``: first-stage indexing and search, system reranking and reciprocal rank fusion.

Thin adapters over :mod:`rcp_ndcg.retrieval`: a dataset (``--dataset``, a :func:`~rcp_ndcg.data.load_dataset` URI)
in, rankings (``--out``, a parquet, JSONL or CSV file of :func:`~rcp_ndcg.data.load_rankings`) out. A retriever or
reranker is a YAML file of :data:`~rcp_ndcg.retrieval.RetrieverConfig` or :data:`~rcp_ndcg.retrieval.RerankerConfig`
(``--retriever``/``--reranker``) whose fields ``--set KEY=VALUE`` overrides (dotted, repeatable), e.g.::

    rcp-ndcg retrieval search --dataset beir:data/nfcorpus --retriever bm25.yaml --out bm25.parquet

with ``bm25.yaml`` holding ``kind: bm25``.

The full-width store and the ex-post Matryoshka sweep live here too: ``retrieval store`` encodes a corpus's
and its queries' vectors once at the checkpoint's full width, and ``retrieval sweep`` applies the declared
MRL head per ``k`` and scores from the stored vectors, so one forward pass evaluates every declared output
dimension (``--dims`` selects a subset; the per-k rankings name the system ``<model>@<k>``).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import click
from pydantic import BaseModel, Field

from rcp_ndcg.cli._args import DatasetInput, load_yaml_config
from rcp_ndcg.cli.command import command
from rcp_ndcg.data import Rankings

_RETRIEVER_SET_HELP = "Override a field of the retriever YAML: KEY=VALUE, dotted, repeatable."
_RERANKER_SET_HELP = "Override a field of the reranker YAML: KEY=VALUE, dotted, repeatable."
_OUT_HELP = "Rankings file to write: .parquet, .jsonl or .csv (local or a storage URI)."


class RankingsFile(BaseModel):
    """A rankings file that a command wrote."""

    out: str = Field(description="Where the rankings are.")
    systems: list[str] = Field(description="The systems in the file.")
    queries: int = Field(description="Number of ranked queries.")


class IndexBuild(BaseModel):
    """An index on disk."""

    index: str = Field(description="The index directory (pass it to `retrieval search --index`).")
    dataset: str
    num_documents: int
    identity: str


def _role_config(value: str, overrides: list[str], *, which: str) -> Any:
    """A retriever or reranker config from a YAML path, or the shorthand ``recipe:<id>`` (docs-firstcontact Q1:
    one string that expands to the mapping form, with the URL from ``--set ...base_url=`` or serve-by-role)."""
    from rcp_ndcg.retrieval import validate_reranker, validate_retriever

    validate = validate_retriever if which == "retriever" else validate_reranker
    if not value.startswith("recipe:"):
        return validate(load_yaml_config(value, overrides))
    from rcp_ndcg.errors import ConfigError
    from rcp_ndcg.inference.recipes import recipe_role, shorthand_config
    from rcp_ndcg.support.config import apply_overrides

    mapping = shorthand_config(value)
    data: Any = mapping
    if which == "retriever":
        role = recipe_role(str(mapping["recipe"]))
        if role == "rerank":
            raise ConfigError(
                f"--retriever {value}: the recipe is a reranker, not a retriever",
                hint="pass it to retrieval rerank --reranker, or name a recipe with role embed or multi_vector",
            )
        data = {"kind": "late_interaction" if role == "multi_vector" else "dense", "encoder": mapping}
    return validate(apply_overrides(data, overrides))


def _retriever(path: str, overrides: list[str]) -> Any:
    return _role_config(path, overrides, which="retriever")


def _written(rankings: Any, out: str) -> RankingsFile:
    rankings.save(out)
    return RankingsFile(
        out=out,
        systems=rankings.systems,
        queries=rankings.to_pandas().groupby(["dataset", "query_id"]).ngroups,
    )


# ----------------------------------------------------------------------------------------------------------------
# index and search
# ----------------------------------------------------------------------------------------------------------------


class RetrievalIndexRequest(DatasetInput):
    retriever: str = Field(description="Retriever YAML (RetrieverConfig: kind bm25, dense or late_interaction).")
    set: list[str] = Field(default_factory=list, description=_RETRIEVER_SET_HELP)
    out: str = Field(description="The index directory (created).")


@command("retrieval index", request=RetrievalIndexRequest, result=IndexBuild, read_only=False)
def retrieval_index(request: RetrievalIndexRequest) -> IndexBuild:
    """Build a sparse, dense or late-interaction index of a dataset's corpus."""
    from rcp_ndcg.retrieval import index

    retriever = _retriever(request.retriever, request.set)  # config errors before the dataset loads
    built = index(request.load(), retriever, out=request.out)
    return IndexBuild(
        index=built.path, dataset=built.dataset, num_documents=built.num_documents, identity=built.identity
    )


class RetrievalSearchRequest(DatasetInput):
    retriever: str | None = Field(
        default=None, description="Retriever YAML: build (or reuse) its index, then search. Or pass --index."
    )
    index: str | None = Field(default=None, description="An index directory from `retrieval index`.")
    set: list[str] = Field(default_factory=list, description=_RETRIEVER_SET_HELP)
    out: str = Field(description=_OUT_HELP)
    depth: int = Field(default=150, ge=1, description="Documents retrieved per query.")


@command("retrieval search", request=RetrievalSearchRequest, result=RankingsFile, read_only=False)
def retrieval_search(request: RetrievalSearchRequest) -> RankingsFile:
    """First-stage retrieval of a dataset's queries into a rankings file."""
    from rcp_ndcg.errors import UsageError
    from rcp_ndcg.retrieval import load_index, retrieve, search

    if (request.retriever is None) == (request.index is None):
        raise UsageError(
            "pass exactly one of --retriever and --index",
            hint="index a corpus with `rcp-ndcg retrieval index --retriever`, or pass --retriever to search it "
            "directly",
        )
    if request.index is not None and request.set:
        raise UsageError(
            "--set overrides the --retriever YAML; an --index was built with its retriever already",
            hint="drop --set and search the index, or rebuild it from a --retriever with the override",
        )
    dataset = request.load()
    if request.index is not None:
        rankings = search(load_index(request.index), dataset, depth=request.depth)
    else:
        assert request.retriever is not None
        rankings = retrieve(dataset, _retriever(request.retriever, request.set), depth=request.depth)
    return _written(rankings, request.out)


# ----------------------------------------------------------------------------------------------------------------
# rerank and fuse
# ----------------------------------------------------------------------------------------------------------------


class RetrievalRerankRequest(DatasetInput):
    rankings: str = Field(description="Rankings file to rerank (parquet, TREC, JSONL or CSV).")
    system: str | None = Field(default=None, description="The system of a rankings file that holds several.")
    reranker: str = Field(description="Reranker YAML (RerankerConfig: api and model).")
    set: list[str] = Field(
        default_factory=list, description=_RERANKER_SET_HELP + " E.g. --set base_url=http://host:8000."
    )
    out: str = Field(description=_OUT_HELP)
    depth: int = Field(default=150, ge=1, description="Rerank the top N candidates of each query.")
    checkpoints: str | None = Field(
        default=None, description="A directory recording each scored query; a rerun skips them (served backends)."
    )


@command("retrieval rerank", request=RetrievalRerankRequest, result=RankingsFile, read_only=False)
def retrieval_rerank(request: RetrievalRerankRequest) -> RankingsFile:
    """Rescore rankings with a served /rerank endpoint or a hosted rerank API."""
    from rcp_ndcg.data import load_rankings
    from rcp_ndcg.retrieval import rerank

    reranker = _role_config(request.reranker, request.set, which="reranker")
    rankings = rerank(
        request.load(),
        load_rankings(request.rankings),
        reranker,
        depth=request.depth,
        system=request.system,
        out=request.checkpoints,
    )
    return _written(rankings, request.out)


class RetrievalFuseRequest(BaseModel):
    rankings: list[str] = Field(min_length=2, description="Rankings files to fuse (repeat: at least two).")
    out: str = Field(description=_OUT_HELP)
    depth: int = Field(default=150, ge=1, description="Documents kept per query after fusion.")
    rrf_k: int = Field(default=60, ge=1, description="Reciprocal rank fusion constant k in 1 / (k + rank).")


@command("retrieval fuse", request=RetrievalFuseRequest, result=RankingsFile, read_only=False)
def retrieval_fuse(request: RetrievalFuseRequest) -> RankingsFile:
    """Reciprocal rank fusion of two or more rankings (every system of each file)."""
    from rcp_ndcg.data import load_rankings
    from rcp_ndcg.retrieval import fuse

    fused = fuse([load_rankings(path) for path in request.rankings], rrf_k=request.rrf_k, depth=request.depth)
    return _written(fused, request.out)


# ----------------------------------------------------------------------------------------------------------------
# the full-width store and the ex-post MRL sweep
# ----------------------------------------------------------------------------------------------------------------


class StoreBuild(BaseModel):
    """A full-width embedding store on disk."""

    store: str = Field(description="The store directory (pass it to `retrieval sweep --store`).")
    dataset: str
    identity: str = Field(description="The store's content identity (the full-width retrieval identity).")
    full_width: int = Field(description="The stored vectors' width (the checkpoint's own output width).")
    count: int = Field(description="Corpus rows stored.")
    query_count: int = Field(description="Query rows stored.")
    mrl_kind: str = Field(description="The declared Matryoshka kind (truncation, projection or none).")
    mrl_dims: list[int] = Field(description="The declared output dimensions the sweep may select.")
    mrl_range: list[int] | None = Field(
        default=None,
        description="The declared [min, max] output-dimension range, when the card gives a range; a range "
        "store needs explicit --dims (it cannot be enumerated).",
    )


class RetrievalStoreRequest(DatasetInput):
    retriever: str = Field(description="Retriever YAML (RetrieverConfig: kind dense or late_interaction).")
    set: list[str] = Field(default_factory=list, description=_RETRIEVER_SET_HELP)
    out: str = Field(description="The store directory (created).")


@command("retrieval store", request=RetrievalStoreRequest, result=StoreBuild, read_only=False)
def retrieval_store(request: RetrievalStoreRequest) -> StoreBuild:
    """Encode a corpus's and its queries' vectors once at the checkpoint's full width."""
    from rcp_ndcg.retrieval import build_store

    retriever = _retriever(request.retriever, request.set)  # config errors before the dataset loads
    record = build_store(request.load(), retriever, out=request.out)
    return StoreBuild(
        store=record.path,
        dataset=record.dataset,
        identity=record.identity,
        full_width=record.full_width,
        count=record.count,
        query_count=record.query_count,
        mrl_kind=record.mrl_kind,
        mrl_dims=list(record.mrl_dims),
        mrl_range=list(record.mrl_range) if record.mrl_range is not None else None,
    )


class RetrievalSweepRequest(BaseModel):
    store: str = Field(description="An embedding store directory from `retrieval store`.")
    dims: list[int] = Field(
        default_factory=list,
        description="The k values to sweep (repeatable); default: every declared mrl_dim of the store (a "
        "store that declares only mrl_range needs explicit dims).",
    )
    out_dir: str = Field(description="Directory for the per-k rankings files (created).")
    dataset: str | None = Field(
        default=None,
        description="The dataset to evaluate against (URI); omit to write the per-k rankings only.",
    )
    subset: str | None = Field(default=None, description="The subset of a hf:// dataset.")
    revision: str | None = Field(default=None, description="The Hub revision of the data.")
    depth: int = Field(default=150, ge=1, description="Documents scored per query.")
    k: list[int] = Field(default_factory=lambda: [10], description="Evaluation cutoffs (repeatable).")
    metrics: list[Literal["rcp_ndcg", "qrel_ndcg"]] = Field(
        default_factory=lambda: ["rcp_ndcg", "qrel_ndcg"], description="Metrics to score (repeatable)."
    )
    bootstrap: int = Field(default=1000, ge=0, description="Bootstrap resamples of the report and comparison.")
    seed: int = Field(default=0, description="The bootstrap seed.")


class RetrievalSweepResult(BaseModel):
    """A sweep's per-k rankings and (with a dataset) its evaluation and comparison files."""

    store: str
    systems: list[str] = Field(description="The per-k systems, in sweep order (``<model>@<k>``).")
    rankings: list[str] = Field(description="The per-k rankings files, in the same order.")
    report: str | None = Field(default=None, description="The evaluation report (JSON), when a dataset was given.")
    comparison: str | None = Field(default=None, description="The k-pair comparison (JSON), with two or more k.")


def _safe_name(system: str) -> str:
    """A rankings filename for a system (a served model name may hold ``/``)."""
    return re.sub(r"[^A-Za-z0-9._@-]", "_", system)


@command("retrieval sweep", request=RetrievalSweepRequest, result=RetrievalSweepResult, read_only=False)
def retrieval_sweep(request: RetrievalSweepRequest) -> RetrievalSweepResult:
    """Apply the declared MRL head per k to a full-width store and score every cut (one forward pass)."""
    from rcp_ndcg.retrieval import load_store, sweep

    record, corpus, queries = load_store(request.store)
    ranked = sweep(record, corpus, queries, dims=request.dims or None, depth=request.depth)
    out_dir = Path(request.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for rows in ranked:
        path = out_dir / f"{_safe_name(rows.systems[0])}.parquet"
        rows.save(path)
        paths.append(str(path))
    report_path: str | None = None
    comparison_path: str | None = None
    if request.dataset is not None:
        from rcp_ndcg.data import load_dataset
        from rcp_ndcg.eval import compare, evaluate

        dataset = load_dataset(request.dataset, subset=request.subset, revision=request.revision)
        report = evaluate(
            Rankings.concat(ranked),
            dataset=dataset,
            k=request.k,
            metrics=request.metrics,
            bootstrap=request.bootstrap,
            seed=request.seed,
        )
        report_file = out_dir / "report.json"
        report_file.write_text(report.model_dump_json(by_alias=True, indent=2), encoding="utf-8")
        report_path = str(report_file)
        if len(ranked) >= 2:
            comparison = compare(report, metric=request.metrics[0], bootstrap=request.bootstrap, seed=request.seed)
            comparison_file = out_dir / "comparison.json"
            comparison_file.write_text(comparison.model_dump_json(by_alias=True, indent=2), encoding="utf-8")
            comparison_path = str(comparison_file)
    return RetrievalSweepResult(
        store=request.store,
        systems=[rows.systems[0] for rows in ranked],
        rankings=paths,
        report=report_path,
        comparison=comparison_path,
    )


@click.group(name="retrieval", help="First-stage retrieval, system reranking and rank fusion.")
def retrieval_group() -> None:
    """``rcp-ndcg retrieval``."""


for _command in (retrieval_index, retrieval_search, retrieval_rerank, retrieval_fuse, retrieval_store, retrieval_sweep):
    retrieval_group.add_command(_command)


__all__ = [
    "IndexBuild",
    "RankingsFile",
    "RetrievalFuseRequest",
    "RetrievalIndexRequest",
    "RetrievalRerankRequest",
    "RetrievalSearchRequest",
    "RetrievalStoreRequest",
    "RetrievalSweepRequest",
    "RetrievalSweepResult",
    "StoreBuild",
    "retrieval_group",
]
