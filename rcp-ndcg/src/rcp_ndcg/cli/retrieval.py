"""``rcp-ndcg retrieval``: first-stage indexing and search, system reranking and reciprocal rank fusion.

Thin adapters over :mod:`rcp_ndcg.retrieval`: a dataset (``--dataset``, a :func:`~rcp_ndcg.data.load_dataset` URI)
in, rankings (``--out``, a parquet, JSONL or CSV file of :func:`~rcp_ndcg.data.load_rankings`) out. A retriever or
reranker is a YAML file of :data:`~rcp_ndcg.retrieval.RetrieverConfig` or :data:`~rcp_ndcg.retrieval.RerankerConfig`
(``--retriever``/``--reranker``) whose fields ``--set KEY=VALUE`` overrides (dotted, repeatable), e.g.::

    rcp-ndcg retrieval search --dataset beir:data/nfcorpus --retriever bm25.yaml --out bm25.parquet

with ``bm25.yaml`` holding ``kind: bm25``.
"""

from __future__ import annotations

from typing import Any

import click
from pydantic import BaseModel, Field

from rcp_ndcg.cli._args import DatasetInput, load_yaml_config
from rcp_ndcg.cli.command import command

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
    rrf_k: int = Field(default=60, ge=0, description="Reciprocal rank fusion constant k in 1 / (k + rank).")


@command("retrieval fuse", request=RetrievalFuseRequest, result=RankingsFile, read_only=False)
def retrieval_fuse(request: RetrievalFuseRequest) -> RankingsFile:
    """Reciprocal rank fusion of two or more rankings (every system of each file)."""
    from rcp_ndcg.data import load_rankings
    from rcp_ndcg.retrieval import fuse

    fused = fuse([load_rankings(path) for path in request.rankings], rrf_k=request.rrf_k, depth=request.depth)
    return _written(fused, request.out)


@click.group(name="retrieval", help="First-stage retrieval, system reranking and rank fusion.")
def retrieval_group() -> None:
    """``rcp-ndcg retrieval``."""


for _command in (retrieval_index, retrieval_search, retrieval_rerank, retrieval_fuse):
    retrieval_group.add_command(_command)


__all__ = [
    "IndexBuild",
    "RankingsFile",
    "RetrievalFuseRequest",
    "RetrievalIndexRequest",
    "RetrievalRerankRequest",
    "RetrievalSearchRequest",
    "retrieval_group",
]
