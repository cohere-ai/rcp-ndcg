"""First-stage retrieval, reranking and fusion over a :class:`~rcp_ndcg.data.Dataset`, returning Rankings.

:func:`index`, :func:`search`, :func:`retrieve` (= index + search, reusing an index of the same identity),
:func:`rerank`, :func:`fuse` (reciprocal rank fusion). The configs are in :mod:`rcp_ndcg.retrieval.config`.

Every model is reached over the shared inference transport through its role client -- an
:class:`~rcp_ndcg.inference.clients.EmbeddingClient` (dense), a
:class:`~rcp_ndcg.inference.clients.PoolingClient` (late interaction) or a
:class:`~rcp_ndcg.inference.clients.RerankClient` (rerank) -- so a hosted API, a gateway and a run's own engine
all take the same code path, and every function takes and returns the data layer's types.

Reranking checkpoints per query: every scored query is recorded under ``out`` (``rank000.jsonl``, one JSON
record, flushed and fsynced) as it finishes, and a rerun with the same reranker over the same candidates skips
the queries the checkpoint already holds. The record format is the served path's historical one; the key is
the reranker's content identity plus the exact texts sent (the earlier release keyed on ids and historical
budget constants only -- such checkpoints are scored again, not resumed).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from rcp_ndcg_core._records import DocumentTitle

from rcp_ndcg.data.dataset import Dataset
from rcp_ndcg.data.rankings import Rankings
from rcp_ndcg.errors import ConfigError, DataError, IdentityError, MissingInputError
from rcp_ndcg.inference.clients import EmbeddingClient, PoolingClient, RerankClient
from rcp_ndcg.inference.types import Embeddings, EncodeRole
from rcp_ndcg.retrieval.config import (
    BM25Config,
    DenseConfig,
    RerankerConfig,
    RetrieverConfig,
    ServedEmbedding,
)
from rcp_ndcg.support.identity import combine_digests, hash_payload, hash_strings, identity_payload, short
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)


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
    doc_ids, contents = _corpus(dataset, title=_title_mode(retriever))
    if isinstance(retriever, BM25Config):
        from rcp_ndcg.retrieval import sparse

        sparse.build_bm25_index(contents, root, stemmer=retriever.stemmer)
    elif isinstance(retriever, DenseConfig):
        embeddings = _encode(
            retriever.encoder,
            contents,
            EncodeRole.DOCUMENT,
            instruction=dataset.task_instruction_for("document"),
        )
        np.save(root / "vectors.npy", embeddings.as_matrix())
    else:
        embeddings = _encode(
            retriever.encoder,
            contents,
            EncodeRole.DOCUMENT,
            instruction=dataset.task_instruction_for("document"),
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
    doc_ids, contents = _corpus(dataset, title=_title_mode(index.retriever))
    if _identity(index.retriever, doc_ids, contents) != index.identity:
        raise IdentityError(
            f"the index at {index.path} was built over another corpus or retriever than {dataset.name!r}",
            hint="rebuild it with index(), or use retrieve(), which rebuilds when the identity differs",
            cli_hint="rebuild it with `rcp-ndcg retrieval index`",
        )
    queries = dataset.queries
    query_ids = list(queries)
    root, retriever = Path(index.path), index.retriever
    instruction = dataset.task_instruction_for("query")
    if isinstance(retriever, BM25Config):
        from rcp_ndcg.retrieval import sparse

        # The sparse path has no role client (no stage to place the instruction): the generic default is
        # applied here, through the one formatter (the core record's), so the two paths read the same text.
        hits = sparse.search_bm25(
            root,
            [queries[q].format_query(task_instruction=instruction) for q in query_ids],
            k=min(depth, len(doc_ids)),
        )
        scores = {q: {doc_ids[row]: score for row, score in hits_q} for q, hits_q in zip(query_ids, hits, strict=True)}
    elif isinstance(retriever, DenseConfig):
        from rcp_ndcg.retrieval.topk import score_topk

        vectors = np.load(root / "vectors.npy")
        documents = Embeddings(vectors=vectors)
        encoded = _encode(
            retriever.encoder,
            [queries[q].format_content() for q in query_ids],
            EncodeRole.QUERY,
            instruction=instruction,
        )
        top_scores, top_indices = score_topk(documents, encoded, depth)
        scores = {
            q: {doc_ids[int(i)]: float(s) for s, i in zip(row_s, row_i, strict=True) if int(i) >= 0}
            for q, row_s, row_i in zip(query_ids, top_scores, top_indices, strict=True)
        }
    else:
        from rcp_ndcg.retrieval.topk import score_topk

        vectors = np.load(root / "vectors.npy")
        offsets = np.load(root / "offsets.npy") if (root / "offsets.npy").exists() else None
        documents = Embeddings(vectors=vectors, offsets=offsets)
        encoded = _encode(
            retriever.encoder,  # type: ignore[arg-type]  # the kind's union: ServedPooling here
            [queries[q].format_content() for q in query_ids],
            EncodeRole.QUERY,
            instruction=instruction,
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
    try:
        return Index.model_validate_json(record.read_text(encoding="utf-8"))
    except ValidationError as exc:
        from rcp_ndcg.retrieval.config import _OLD_SHAPE_HINT
        from rcp_ndcg.support.config import config_error

        raise config_error(
            exc,
            model=Index,
            source=f"{record}",
            hint=f"this index was written before the api rewiring; rebuild it with rcp_ndcg.retrieval.index. "
            f"{_OLD_SHAPE_HINT}",
        ) from exc


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
    doc_ids, contents = _corpus(dataset, title=_title_mode(retriever))
    identity = _identity(retriever, doc_ids, contents)
    if out is None:
        from rcp_ndcg.support.paths import cache_dir

        out = cache_dir() / "indexes" / identity[:16]
    record = Path(out) / "index.json"
    built: Index | None = None
    if record.is_file():
        try:
            built = load_index(out)
        except ConfigError:
            pass  # an index.json of another shape (or a corrupt one) is a cache miss: rebuild over it
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
        rankings: The candidates: the rows that name ``dataset``, else the rows that name no dataset
            (:meth:`~rcp_ndcg.data.Rankings.resolve_dataset`).
        reranker: The reranker.
        depth: Candidates per query.
        system: The system of ``rankings`` to rerank (may be omitted when it has one).
        out: A checkpoint directory: each scored query is recorded as it finishes, and a rerun skips them.

    Returns:
        :class:`~rcp_ndcg.data.Rankings` with one system named after the reranker's model, its rows naming the
        dataset the candidates' rows name.

    Raises:
        ConfigError: ``depth`` is not positive (as :func:`search` refuses it).
        DataError: A ranked document is not in the corpus, or a ranked query is not in the dataset.
    """
    from rcp_ndcg_core._records import RankingExample

    if depth <= 0:
        raise ConfigError(f"depth must be positive, got {depth}")
    candidates = rankings.top(depth).queries(system=system, dataset=dataset.name)
    corpus, queries = dataset.corpus, dataset.queries
    missing = sorted({d for docs in candidates.values() for d in docs if d not in corpus})
    if missing:
        raise DataError(f"{len(missing)} ranked documents are not in {dataset.name!r}'s corpus, e.g. {missing[:3]}")
    title = _title_mode(reranker)
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
                contents=[corpus[d].model_content(title=title) for d in order],
            )
        )
    scored = _rerank_examples(
        examples,
        reranker,
        task_instruction=dataset.task_instruction_for("query"),
        checkpoint_dir=out,
    )
    return Rankings.from_scores(
        {e.id: dict(zip(e.doc_ids, e.scores or [], strict=True)) for e in scored},
        system=reranker.model,
        dataset=rankings.resolve_dataset(dataset.name) or "",
    )


def fuse(rankings: Sequence[Rankings], *, rrf_k: int = 60, depth: int = 150, system: str = "rrf") -> Rankings:
    """Reciprocal rank fusion of several rankings: ``sum_r 1 / (rrf_k + rank_r)``, ranks from 1.

    Rows of different datasets are fused apart, so the rankings of a suite keep their subsets.

    Args:
        rankings: One :class:`~rcp_ndcg.data.Rankings` per system (every system of each is fused).
        rrf_k: The RRF constant.
        depth: Documents per query after fusion.
        system: The fused system's name.

    Returns:
        :class:`~rcp_ndcg.data.Rankings` with one system; its scores are the RRF scores.

    Raises:
        DataError: No rankings to fuse, or two rankings that share a subset ranking different queries.
        ConfigError: ``depth`` or ``rrf_k`` is not positive.
    """
    from rcp_ndcg_core._records import RankingExample

    from rcp_ndcg.retrieval.fusion import reciprocal_rank_fusion

    datasets = list(dict.fromkeys(name for ranking in rankings for name in ranking.datasets))
    if not datasets:
        raise DataError("fuse needs rankings to fuse; got none")
    fused_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        runs = [
            [
                RankingExample(query_id=q, query="", doc_ids=sorted(docs, key=lambda d: (docs[d], d), reverse=True))
                for q, docs in ranking.queries(system=name, dataset=dataset).items()
            ]
            for ranking in rankings
            if ranking.resolve_dataset(dataset) is not None  # a ranking with no rows for the subset stays out
            for name in ranking.systems
        ]
        fused = reciprocal_rank_fusion(runs, top_k=depth, rrf_k=rrf_k)
        fused_rows += [
            {"system": system, "dataset": dataset, "query_id": e.id, "doc_id": d, "score": score}
            for e in fused
            for d, score in zip(e.doc_ids, e.scores or [], strict=True)
        ]
    return Rankings.from_records(fused_rows)


# ---------------------------------------------------------------------------
# The rerank driver: the clients score, this module checkpoints and aligns.
# ---------------------------------------------------------------------------


def _checkpoint_key(
    config: RerankerConfig,
    example: Any,
    *,
    tokenizer_sha256: str | None = None,
    task_instruction: str | None = None,
) -> str:
    """What a checkpointed query's scores are valid for: the reranker's content identity and the exact texts.

    The payload is :func:`~rcp_ndcg.support.identity.identity_payload` of the config (the model, its revision,
    the wire adapter, the recipe, the instruction mode, the activation switch and the budgets), the
    tokenizer's SHA-256, and what goes over the wire for this query: its id, its raw text and instruction,
    the run's task instruction, and the candidate ids with a digest of their contents. A rerun after any of
    these changed -- or over a different candidate set or depth -- computes another key and scores the query
    again.

    The key is not the earlier release's (that payload named only the model, revision, the historical budget
    constants and the ids, so a rerun after any content change silently resumed stale scores). A checkpoint
    written before this key existed is therefore re-scored, not resumed; nothing is released yet, so no
    checkpoint in the wild carries the old key.

    Args:
        config: The reranker; its CONTENT fields and tokenizer digest enter the payload.
        example: The query to score, with its documents populated (the digest reads ``doc_contents``).
        tokenizer_sha256: The config's tokenizer digest, resolved once by the caller; ``None`` resolves it
            here (an identity-like cost per query otherwise).
        task_instruction: The run's task instruction (``Dataset.task_instruction``): the model read it, so a
            rerun with another one scores the query again.
    """
    payload = identity_payload(config)
    digest = tokenizer_sha256 if tokenizer_sha256 is not None else config.identity_extra().get("tokenizer_sha256")
    if digest:
        payload["tokenizer_sha256"] = digest
    payload.update(
        {
            "query_id": str(example.id),
            "query": example.as_content.model_dump_json(),
            "query_instruction": example.instruction,
            "task_instruction": task_instruction,
            "doc_ids": [str(doc_id) for doc_id in example.doc_ids],
            "docs": hash_strings([content.model_dump_json() for content in example.doc_contents]),
        }
    )
    return short(hash_payload(payload), 16)


def _iter_checkpoint_records(ckpt_dir: Path) -> Iterable[dict[str, Any]]:
    """Yield score records from every ``rank*.jsonl`` in *ckpt_dir*.

    Tolerates a truncated trailing line -- the common crash signature is a process
    that died mid-flush, and one unparseable line should not invalidate the work
    before it.
    """
    if not ckpt_dir.exists():
        return
    for shard in sorted(ckpt_dir.glob("rank*.jsonl")):
        with shard.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _checkpoint_scores(ckpt_dir: Path) -> dict[str, dict[str, float]]:
    """``{checkpoint key: {doc_id: score}}`` across all rank checkpoints (:func:`_checkpoint_key`).

    Per *query* rather than per pair: a rerank request is atomic at the query level, so a partially-written query
    has nothing usable to resume from. Records without a key are not reused, and one with a value that is not a
    number is dropped whole (the reader takes historical and hand-edited files as it finds them; one bad value
    poisons only its own record, which the rerun then scores again).
    """
    by_key: dict[str, dict[str, float]] = {}
    for record in _iter_checkpoint_records(ckpt_dir):
        if isinstance(record, dict) and isinstance(record.get("k"), str) and isinstance(record.get("s"), dict):
            scores: dict[str, float] = {}
            for doc_id, score in record["s"].items():
                try:
                    scores[str(doc_id)] = float(score)
                except (TypeError, ValueError):
                    break
            else:
                by_key[record["k"]] = scores
    return by_key


def _reusable_scores(
    by_key: dict[str, dict[str, float]], examples: Sequence[Any], keys: Sequence[str]
) -> dict[str, dict[str, float]]:
    """The checkpointed scores a rerun may reuse: those covering their example's whole candidate set.

    The writer records one complete score map per query, but the reader takes historical and hand-edited files
    as it finds them: a record that misses one of its example's documents is dropped here and the query is
    scored again, instead of resuming into a refusal whose hint -- rerun the rerank -- replayed the identical
    failure forever.
    """
    out: dict[str, dict[str, float]] = {}
    for example, key in zip(examples, keys, strict=True):
        scores = by_key.get(key)
        if scores is not None and all(str(doc_id) in scores for doc_id in example.doc_ids):
            out[key] = scores
    return out


def _rerank_examples(
    examples: list[Any],
    config: RerankerConfig,
    *,
    task_instruction: str | None = None,
    checkpoint_dir: str | Path | None,
) -> list[Any]:
    """Score every example through the rerank client, checkpointing per query.

    Args:
        examples: :class:`~rcp_ndcg_core._records.RankingExample` records with their documents populated.
        config: The reranker.
        task_instruction: The run's task instruction (``Dataset.task_instruction``), placed by the config's
            ``instruction`` mode; part of the checkpoint key, because the model read it.
        checkpoint_dir: When given, each scored query is appended to ``<dir>/rank000.jsonl`` (one record
            ``{"q", "k", "s"}``, flushed and fsynced) as it finishes, and the queries the directory already holds
            are skipped.

    Returns:
        The examples in input order with ``scores`` set, aligned to ``doc_ids``.

    Raises:
        DataError: A document has no score. The candidate set is part of the run's identity, so a document is
            neither dropped nor given a made-up score.
    """
    client = RerankClient(config)
    tokenizer_sha256 = config.identity_extra().get("tokenizer_sha256")
    keys = [
        _checkpoint_key(config, example, tokenizer_sha256=tokenizer_sha256, task_instruction=task_instruction)
        for example in examples
    ]
    meta = {
        str(example.id): (key, [str(doc_id) for doc_id in example.doc_ids])
        for example, key in zip(examples, keys, strict=True)
    }
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    by_key = _checkpoint_scores(ckpt_dir) if ckpt_dir is not None else {}
    if by_key:
        by_key = _reusable_scores(by_key, examples, keys)
    pending = [example for example, key in zip(examples, keys, strict=True) if key not in by_key]
    if by_key and len(pending) < len(examples):
        logger.info(f"rerank resume: {len(examples) - len(pending)}/{len(examples)} queries already scored")

    ckpt_fh = (ckpt_dir / "rank000.jsonl").open("a") if ckpt_dir is not None else None
    try:

        def checkpoint(query_id: str, scores: tuple[float, ...]) -> None:
            """One scored query: record it (old record format), flush, fsync -- a crash costs the in-flight ones."""
            key, doc_ids = meta[query_id]
            scored = {doc_id: float(score) for doc_id, score in zip(doc_ids, scores, strict=True)}
            by_key[key] = scored
            if ckpt_fh is not None:
                ckpt_fh.write(json.dumps({"q": query_id, "k": key, "s": scored}) + "\n")
                # Flush + fsync per query so a crash on the next one keeps this one.
                ckpt_fh.flush()
                os.fsync(ckpt_fh.fileno())

        try:
            client.rerank_many(pending, instruction=task_instruction, checkpoint=checkpoint)
        finally:
            client.close()
    finally:
        if ckpt_fh is not None:
            ckpt_fh.close()
    return _apply_scores(examples, keys, by_key)


def _apply_scores(
    examples: Sequence[Any],
    keys: Sequence[str],
    by_key: dict[str, dict[str, float]],
) -> list[Any]:
    """Attach scores to *examples* in input order (``keys``: each example's :func:`_checkpoint_key`).

    Raises:
        DataError: A document has no score. The candidate set is part of the run's identity, so a document is
            neither dropped nor given a made-up score.
    """
    out: list[Any] = []
    for example, key in zip(examples, keys, strict=True):
        scored = by_key.get(key, {})
        missing = [str(doc_id) for doc_id in example.doc_ids if str(doc_id) not in scored]
        if missing:
            raise DataError(
                f"query {example.id!r}: the reranker returned no score for {len(missing)} documents, e.g. "
                f"{missing[:3]}",
                hint="rerun the rerank; scored queries are kept in the checkpoint directory",
            )
        out.append(example.model_copy(update={"scores": [scored[str(doc_id)] for doc_id in example.doc_ids]}))
    return out


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _corpus(dataset: Dataset, *, title: DocumentTitle = "join") -> tuple[list[str], list[Any]]:
    """The corpus's ids, sorted, and their contents: an index's row order.

    The rows are in id order, so the top-k's tie-break toward the lower row
    (:func:`~rcp_ndcg.retrieval.topk.select_topk`) is a tie-break toward the lower document id, whatever order
    the dataset lists its corpus in. Each document is materialised as the content a model reads
    (:meth:`~rcp_ndcg.data.DocumentRow.model_content`): MTEB's title join, or the title separately where the
    step's config declares ``title: separate``.
    """
    corpus = dataset.corpus
    if not corpus:
        raise DataError(f"{dataset.name!r} has no corpus to index")
    doc_ids = sorted(corpus)
    return doc_ids, [corpus[doc_id].model_content(title=title) for doc_id in doc_ids]


def _title_mode(config: Any) -> DocumentTitle:
    """The title mode a retrieval step's config declares: the encoder's (or the reranker's) ``title`` field,
    else MTEB's join (``None`` declares nothing)."""
    endpoint = getattr(config, "encoder", config)
    return getattr(endpoint, "title", None) or "join"


def _corpus_hash(contents: Sequence[Any], doc_ids: Sequence[str]) -> str:
    """Identity of a corpus, for deciding whether an index still matches it.

    A text-only document hashes as its plain string; a document with media hashes as its full parts array, since
    two page corpora with the same (empty) text are otherwise indistinguishable.
    """
    bodies = [content.text if not content.has_media else content.model_dump_json() for content in contents]
    return combine_digests(hash_strings(doc_ids), hash_strings(bodies))


def _identity(retriever: RetrieverConfig, doc_ids: list[str], contents: list[Any]) -> str:
    """What an index is: the retriever's content fields (``IDENTITY_ROLES``), the encoder's tokenizer digest and
    the corpus.

    The tokenizer digest (``identity_extra()``) is spliced in at the encoder, as the step identities splice it:
    what cuts the text is content, and the tokenizer's *name* is not. Two configs that declare different
    tokenizer bytes never share an index, so a corpus indexed before the text-budget mechanism lands is not
    silently reused once budgets become content-bearing.
    """
    payload = identity_payload(retriever)
    encoder = getattr(retriever, "encoder", None)
    if encoder is not None:
        payload = payload.copy()
        payload["encoder"] = {**payload.get("encoder", {}), **encoder.identity_extra()}
    return combine_digests(hash_payload(payload), _corpus_hash(contents, doc_ids))


def _system_name(retriever: RetrieverConfig) -> str:
    if isinstance(retriever, BM25Config):
        return "bm25"
    return retriever.encoder.model


def _no_base_url() -> ConfigError:
    """The refusal a served encoder without a URL gets, with where the URL may come from."""
    return ConfigError(
        "the served encoder has no base_url",
        hint="give encoder.base_url, or start its engine with serve.encoder in the run config (the job then "
        "passes the URL at runtime)",
    )


def _encode(config: Any, contents: Sequence[Any], role: EncodeRole, *, instruction: str | None = None) -> Embeddings:
    """Encode *contents* through the encoder config's role client, its transport closed after the call.

    Any pooling config (:class:`~rcp_ndcg.retrieval.config.ServedPooling` or a third-party
    :class:`~rcp_ndcg.retrieval.config.PluginPooling`) runs the
    :class:`~rcp_ndcg.inference.clients.PoolingClient` (ragged); every embedding config -- served, hosted, or
    a third-party :class:`~rcp_ndcg.retrieval.config.PluginEmbedding` -- the
    :class:`~rcp_ndcg.inference.clients.EmbeddingClient`, a hosted profile at the vendor's public URL.
    ``instruction`` is the side's task instruction (``Dataset.task_instruction_for``), placed by the
    config's ``instruction`` mode.

    Raises:
        ConfigError: a pooling encoder's ``base_url`` is unset: give it, or start its engine with
            ``serve.encoder`` (the job then passes the URL at runtime); a served one without a public root
            likewise.
    """
    from rcp_ndcg.inference.config import PoolingEndpoint

    if isinstance(config, PoolingEndpoint):
        if config.base_url is None:
            raise _no_base_url()
        client: Any = PoolingClient(config)
    else:
        if isinstance(config, ServedEmbedding) and config.base_url is None:
            raise _no_base_url()
        client = EmbeddingClient(config)
    try:
        return client.encode(contents, role, instruction=instruction)
    finally:
        client.close()  # the client base's sync close (an injected sender closes nothing)


__all__ = ["Index", "fuse", "index", "load_index", "rerank", "retrieve", "search"]
