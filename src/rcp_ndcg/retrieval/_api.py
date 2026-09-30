"""First-stage retrieval, reranking and fusion over a :class:`~rcp_ndcg.data.Dataset`, returning Rankings.

:func:`index`, :func:`search`, :func:`retrieve` (= index + search, reusing an index of the same identity),
:func:`rerank`, :func:`fuse` (reciprocal rank fusion). The configs are in :mod:`rcp_ndcg.retrieval.config`.

Every function takes and returns the data layer's types; the retrievers, encoders and rerankers behind them are
the modules of this package.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from rcp_ndcg.data.dataset import Dataset
from rcp_ndcg.data.rankings import Rankings
from rcp_ndcg.errors import ConfigError, CredentialsError, DataError, IdentityError, MissingInputError
from rcp_ndcg.retrieval.config import (
    BM25Config,
    DenseConfig,
    EncoderConfig,
    LateInteractionConfig,
    Local,
    LocalEncoder,
    OpenAICompatibleEncoder,
    OpenAICompatibleReranker,
    RerankerConfig,
    RetrieverConfig,
)
from rcp_ndcg.support.identity import combine_digests, hash_payload, hash_strings, identity_payload

_RETRIEVER: TypeAdapter[BM25Config | DenseConfig | LateInteractionConfig] = TypeAdapter(RetrieverConfig)

_LOCAL_RERANKERS: tuple[tuple[str, str], ...] = (
    ("qwen/qwen3-reranker", "qwen_og"),
    ("zeroentropy/zerank", "zerank"),
    ("contextualai/ctxl-rerank", "contextual"),
    ("jinaai/jina-reranker", "jina_hf"),
)
"""The in-process rerankers: model-id prefix -> the implementation that scores it."""

_BATCHED: frozenset[str] = frozenset({"qwen_og", "contextual", "cohere", "voyage"})
"""The rerankers that take a batch size: documents per forward pass, or per API request."""

_DEFAULT_BATCH: dict[str, int] = {"qwen_og": 8, "contextual": 8, "cohere": 100, "voyage": 20}


class Index(BaseModel):
    """A built index: where it is, what it indexes, and its identity."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    schema_name: Literal["rcp-ndcg.index.v1"] = Field(default="rcp-ndcg.index.v1", alias="schema")
    path: str
    dataset: str
    retriever: RetrieverConfig
    identity: str
    num_documents: int


# ---------------------------------------------------------------------------
# index, search, retrieve
# ---------------------------------------------------------------------------


def index(dataset: Dataset, retriever: RetrieverConfig, *, out: str | Path) -> Index:
    """Build an index of a dataset's corpus.

    Args:
        dataset: The dataset; its corpus is read.
        retriever: The retriever.
        out: The index directory (created).

    Returns:
        The :class:`Index`; ``<out>/index.json`` records it.
    """
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    doc_ids, contents = _corpus(dataset)
    if isinstance(retriever, BM25Config):
        _bm25_class(retriever).build_index(contents, root, stemmer=retriever.stemmer)
    else:
        from rcp_ndcg.retrieval.encoder import EncodeRole

        embeddings = _encoder(retriever.encoder).encode(
            contents, role=EncodeRole.DOCUMENT, batch_size=retriever.encoder.batch_size
        )
        np.save(root / "vectors.npy", embeddings.vectors)
        if embeddings.offsets is not None:
            np.save(root / "offsets.npy", embeddings.offsets)
    built = Index(
        path=str(root),
        dataset=dataset.name,
        retriever=retriever,
        identity=_identity(retriever, doc_ids, contents),
        num_documents=len(doc_ids),
    )
    (root / "index.json").write_text(built.model_dump_json(by_alias=True, indent=2), encoding="utf-8")
    return built


def search(index: Index, dataset: Dataset, *, depth: int = 150) -> Rankings:
    """Search an index with a dataset's queries.

    Args:
        index: The index (from :func:`index`, or :func:`load_index`).
        dataset: The dataset whose queries are searched; its corpus must be the one indexed.
        depth: Documents per query.

    Returns:
        :class:`~rcp_ndcg.data.Rankings` with one system, named after the retriever.

    Raises:
        IdentityError: The dataset's corpus is not the indexed one.
    """
    if depth <= 0:
        raise ConfigError(f"depth must be positive, got {depth}")
    doc_ids, contents = _corpus(dataset)
    if _identity(index.retriever, doc_ids, contents) != index.identity:
        raise IdentityError(
            f"the index at {index.path} was built over another corpus or retriever than {dataset.name!r}",
            hint="rebuild it with index(), or use retrieve(), which rebuilds when the identity differs",
            cli_hint="rebuild it with `rcp-ndcg retrieval index`",
        )
    queries = dataset.queries
    query_ids = list(queries)
    root, retriever = Path(index.path), index.retriever
    if isinstance(retriever, BM25Config):
        engine = _bm25_class(retriever).from_index(root, contents)
        hits = engine.retrieve_batch([queries[q].format_query() for q in query_ids], k=depth)
        scores = {
            q: {doc_ids[h["id"]]: float(h["score"]) for h in hits_q} for q, hits_q in zip(query_ids, hits, strict=True)
        }
    else:
        from rcp_ndcg.retrieval.encoder import Embeddings, EncodeRole
        from rcp_ndcg.retrieval.topk import score_topk

        vectors = np.load(root / "vectors.npy")
        offsets = np.load(root / "offsets.npy") if (root / "offsets.npy").exists() else None
        documents = Embeddings(vectors=vectors, offsets=offsets)
        encoded = _encoder(retriever.encoder).encode(
            [queries[q].format_content() for q in query_ids],
            role=EncodeRole.QUERY,
            batch_size=retriever.encoder.batch_size,
        )
        top_scores, top_indices = score_topk(documents, encoded, depth)
        scores = {
            q: {doc_ids[int(i)]: float(s) for s, i in zip(row_s, row_i, strict=True) if int(i) >= 0}
            for q, row_s, row_i in zip(query_ids, top_scores, top_indices, strict=True)
        }
    return Rankings.from_scores(scores, system=_system_name(retriever))


def load_index(path: str | Path) -> Index:
    """The :class:`Index` recorded in ``<path>/index.json``."""
    record = Path(path) / "index.json"
    if not record.is_file():
        raise MissingInputError(
            f"no index at {path}",
            hint="build one with rcp_ndcg.retrieval.index",
            cli_hint="build one with `rcp-ndcg retrieval index`",
        )
    return Index.model_validate_json(record.read_text(encoding="utf-8"))


def retrieve(
    dataset: Dataset, retriever: RetrieverConfig, *, depth: int = 150, out: str | Path | None = None
) -> Rankings:
    """Index (unless an index of the same identity exists under ``out``) and search.

    Args:
        dataset: The dataset.
        retriever: The retriever.
        depth: Documents per query.
        out: The index directory; defaults to ``indexes/<identity>`` under the package cache.

    Returns:
        :class:`~rcp_ndcg.data.Rankings` with one system.
    """
    doc_ids, contents = _corpus(dataset)
    identity = _identity(retriever, doc_ids, contents)
    if out is None:
        from rcp_ndcg.support.paths import cache_dir

        out = cache_dir() / "indexes" / identity[:16]
    record = Path(out) / "index.json"
    built = load_index(out) if record.is_file() else None
    if built is None or built.identity != identity:
        built = index(dataset, retriever, out=out)
    return search(built, dataset, depth=depth)


# ---------------------------------------------------------------------------
# rerank and fuse
# ---------------------------------------------------------------------------


def rerank(
    dataset: Dataset,
    rankings: Rankings,
    reranker: RerankerConfig,
    *,
    depth: int = 150,
    system: str | None = None,
    out: str | Path | None = None,
) -> Rankings:
    """Rescore each query's top ``depth`` candidates with a reranker.

    Args:
        dataset: The dataset (queries and corpus).
        rankings: The candidates.
        reranker: The reranker.
        depth: Candidates per query.
        system: The system of ``rankings`` to rerank (may be omitted when it has one).
        out: A checkpoint directory: each scored query is recorded as it finishes, and a rerun skips them
            (served backends).

    Returns:
        :class:`~rcp_ndcg.data.Rankings` with one system named after the reranker's model.
    """
    from rcp_ndcg_core._records import RankingExample

    from rcp_ndcg.retrieval.cross_encoder import rerank_examples

    candidates = rankings.top(depth).queries(system=system)
    corpus, queries = dataset.corpus, dataset.queries
    missing = sorted({d for docs in candidates.values() for d in docs if d not in corpus})
    if missing:
        raise DataError(f"{len(missing)} ranked documents are not in {dataset.name!r}'s corpus, e.g. {missing[:3]}")
    examples = []
    for query_id, scores in candidates.items():
        if query_id not in queries:
            raise DataError(f"query {query_id!r} of the rankings is not in {dataset.name!r}")
        order = sorted(scores, key=lambda d: (scores[d], d), reverse=True)
        query = queries[query_id]
        examples.append(
            RankingExample(
                query_id=query_id,
                query=query.text,
                instruction=query.instruction,
                doc_ids=order,
                contents=[corpus[d].as_content for d in order],
            )
        )
    scored = rerank_examples(examples, _rerank_settings(reranker), checkpoint_dir=out) or []
    return Rankings.from_scores(
        {e.id: dict(zip(e.doc_ids, e.scores or [], strict=True)) for e in scored}, system=reranker.model
    )


def fuse(rankings: Sequence[Rankings], *, rrf_k: int = 60, depth: int = 150, system: str = "rrf") -> Rankings:
    """Reciprocal rank fusion of several rankings: ``sum_r 1 / (rrf_k + rank_r)``, ranks from 1.

    Args:
        rankings: One :class:`~rcp_ndcg.data.Rankings` per system (every system of each is fused).
        rrf_k: The RRF constant.
        depth: Documents per query after fusion.
        system: The fused system's name.

    Returns:
        :class:`~rcp_ndcg.data.Rankings` with one system; its scores are the RRF scores.
    """
    from rcp_ndcg_core._records import RankingExample

    from rcp_ndcg.retrieval.fusion import reciprocal_rank_fusion

    runs = []
    for ranking in rankings:
        for name in ranking.systems:
            runs.append(
                [
                    RankingExample(query_id=q, query="", doc_ids=sorted(docs, key=lambda d: (docs[d], d), reverse=True))
                    for q, docs in ranking.queries(system=name).items()
                ]
            )
    try:
        fused = reciprocal_rank_fusion(runs, top_k=depth, rrf_k=rrf_k)
    except ValueError as exc:
        raise DataError(str(exc)) from exc
    return Rankings.from_scores({e.id: dict(zip(e.doc_ids, e.scores or [], strict=True)) for e in fused}, system=system)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _corpus(dataset: Dataset) -> tuple[list[str], list[Any]]:
    """The corpus's ids, sorted, and their contents: an index's row order.

    The rows are in id order, so the top-k's tie-break toward the lower row
    (:func:`~rcp_ndcg.retrieval.topk.select_topk`) is a tie-break toward the lower document id, whatever order
    the dataset lists its corpus in.
    """
    corpus = dataset.corpus
    if not corpus:
        raise DataError(f"{dataset.name!r} has no corpus to index")
    doc_ids = sorted(corpus)
    return doc_ids, [corpus[doc_id].as_content for doc_id in doc_ids]


def _corpus_hash(contents: Sequence[Any], doc_ids: Sequence[str]) -> str:
    """Identity of a corpus, for deciding whether an index still matches it.

    A text-only document hashes as its plain string; a document with media hashes as its full parts array, since
    two page corpora with the same (empty) text are otherwise indistinguishable.
    """
    bodies = [content.text if not content.has_media else content.model_dump_json() for content in contents]
    return combine_digests(hash_strings(doc_ids), hash_strings(bodies))


def _identity(retriever: RetrieverConfig, doc_ids: list[str], contents: list[Any]) -> str:
    """What an index is: the retriever's content fields (``IDENTITY_ROLES``) and the corpus."""
    return combine_digests(hash_payload(identity_payload(retriever)), _corpus_hash(contents, doc_ids))


def _system_name(retriever: RetrieverConfig) -> str:
    if isinstance(retriever, BM25Config):
        return "bm25"
    return retriever.encoder.model


def _bm25_class(retriever: BM25Config) -> Any:
    from rcp_ndcg.retrieval.sparse import BM25SRetriever

    return BM25SRetriever


def _key(env: str | None) -> str | None:
    if env is None:
        return None
    if env not in os.environ:
        raise CredentialsError(f"the API key variable {env} is not set", hint=f"export {env}=...")
    return os.environ[env]


def _encoder(config: EncoderConfig) -> Any:
    """The :class:`~rcp_ndcg.retrieval.encoder.Encoder` that runs ``config``."""
    if isinstance(config, LocalEncoder) and config.engine == "hf":
        from rcp_ndcg.retrieval.encoders import TorchDenseEncoder

        return TorchDenseEncoder(
            model_name=config.model,
            revision=config.revision,
            batch_size=config.batch_size or 32,
            document_prefix=config.doc_prompt,
        )
    if isinstance(config, LocalEncoder) or (isinstance(config, OpenAICompatibleEncoder) and config.pooling == "token"):
        from rcp_ndcg.retrieval.encoders import VllmEncoder

        served = isinstance(config, OpenAICompatibleEncoder)
        return VllmEncoder(
            model_name=config.model,
            pooling_task="token_embed" if config.pooling == "token" else "embed",
            mode="http" if served else "offline",
            revision=None if served else config.revision,
            api_base=config.base_url if served else None,
            api_key=_key(config.api_key_env) if served else None,
            timeout_s=config.timeout_s if served else None,
            connect_timeout_s=config.connect_timeout_s if served else None,
            max_retries=config.max_retries if served else None,
            batch_size=config.batch_size or 64,
            query_prefix=config.query_prompt or "",
            document_prefix=config.doc_prompt or "",
        )
    from rcp_ndcg.retrieval.encoders import HostedApiEncoder

    return HostedApiEncoder(
        vendor="openai" if isinstance(config, OpenAICompatibleEncoder) else config.provider,
        model=config.model,
        base_url=config.base_url,
        api_key=_key(config.api_key_env),
        batch_size=config.batch_size,
        timeout_s=config.timeout_s,
        connect_timeout_s=config.connect_timeout_s,
        max_retries=config.max_retries,
        query_prefix=getattr(config, "query_prompt", None) or "",
        document_prefix=getattr(config, "doc_prompt", None) or "",
    )


def _local_reranker(model: str) -> str:
    """The in-process implementation of a reranker model id."""
    family = next((f for prefix, f in _LOCAL_RERANKERS if model.lower().startswith(prefix)), None)
    if family is None:
        known = ", ".join(prefix for prefix, _ in _LOCAL_RERANKERS)
        raise ConfigError(
            f"no in-process reranker for {model!r} (known families: {known})",
            hint="serve the model (vllm serve --runner pooling) and use provider: openai_compatible",
        )
    return family


def _rerank_settings(config: RerankerConfig) -> Any:
    from rcp_ndcg.retrieval.cross_encoder import RerankSettings

    if isinstance(config, Local):
        framework = _local_reranker(config.model)
        if config.batch_size is not None and framework not in _BATCHED:
            raise ConfigError(f"{config.model} batches on its own and takes no batch_size", hint="drop batch_size")
        batch = config.batch_size or _DEFAULT_BATCH.get(framework, 8)
        return RerankSettings(
            model_name=config.model, framework=framework, revision=config.revision, external_batch_size=batch
        )
    if isinstance(config, OpenAICompatibleReranker):
        return RerankSettings(
            model_name=config.model,
            framework="vllm",
            api_base=config.base_url,
            api_key=_key(config.api_key_env),
            concurrency=config.concurrency,
            timeout_s=config.timeout_s,
            connect_timeout_s=config.connect_timeout_s,
            max_retries=config.max_retries,
        )
    return RerankSettings(
        model_name=config.model,
        framework=config.provider,
        api_base=config.base_url,
        api_key=_key(config.api_key_env),
        request_size=config.batch_size or _DEFAULT_BATCH[config.provider],
        timeout_s=config.timeout_s,
        connect_timeout_s=config.connect_timeout_s,
        max_retries=config.max_retries,
    )
