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
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rcp_ndcg.data.dataset import Dataset
from rcp_ndcg.data.media import content_identity
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
from rcp_ndcg.storage import local_dir, publication_lock, publish
from rcp_ndcg.storage.artifacts import artifact_ref
from rcp_ndcg.support.identity import combine_digests, hash_payload, hash_strings, identity_payload, short
from rcp_ndcg.support.logging import get_logger

logger = get_logger(__name__)

#: The index's behaviour version.  Bumped deliberately when what an index *is* changes without any config
#: field changing -- the selection rule, the payload layout, the tie rule -- and never for a release as such:
#: it enters the index identity, so an index built by other numbers is rebuilt, and a release that changes no
#: retrieval number does not invalidate every resume and judgement pool.  A judgement's identity is not
#: versioned here: it is prompt- and model-defined.
INDEX_BEHAVIOUR_VERSION = "1"

#: The retrieve step's behaviour version: the numbers a retrieve output was computed with (the index's own
#: version covers the index; this covers the first-stage selection and the ranking it writes).
RETRIEVE_BEHAVIOUR_VERSION = "1"

#: The rerank step's behaviour version: the numbers a rerank output -- and its checkpoint key -- were
#: computed with.
RERANK_BEHAVIOUR_VERSION = "1"

#: The record's own file, never part of the payload it describes.
_RECORD = "index.json"


class Index(BaseModel):
    """A built index: where it is, what it indexes, and its identity.

    Attributes:
        path: The directory this record was loaded from (``load_index``) or built in (``index``): the I/O
            root of the payload.  ``index.json``'s own ``path`` records where the index was *built* --
            provenance -- and :func:`load_index` returns the record with the directory it read, so a copied,
            moved or restored index directory is searched where it now is.
        dataset: The dataset the corpus came from.
        retriever: The retriever that built the payload.
        identity: The index identity: the retriever's content fields, the tokenizer digest, the corpus and
            :data:`INDEX_BEHAVIOUR_VERSION`.
        num_documents: Corpus rows in the payload.
        behaviour_version: :data:`INDEX_BEHAVIOUR_VERSION` as recorded, for diagnostics.
        payload: ``{relative name: sha256}`` of every payload file (``vectors.npy``, ``offsets.npy``, the
            ``bm25s/`` model files).  :func:`search` recomputes it, so a killed or concurrent build -- whose
            record describes another payload -- is refused, never scored.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    schema_name: Literal["rcp-ndcg.index.v1"] = Field(default="rcp-ndcg.index.v1", alias="schema")
    path: str
    dataset: str
    retriever: RetrieverConfig
    identity: str
    num_documents: int
    behaviour_version: str = INDEX_BEHAVIOUR_VERSION
    payload: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# index, search, retrieve
# ---------------------------------------------------------------------------


def _payload_digest(root: Path) -> dict[str, str]:
    """``{relative name: sha256}`` of every payload file under *root*, sorted.

    The payload is everything an index directory holds beside its record: ``vectors.npy`` and
    ``offsets.npy`` for a dense or late-interaction index, the ``bm25s/`` model files for a sparse one.
    Files another writer's atomic publish left behind (``*.tmp``) are not part of it.
    """
    if not root.is_dir():
        return {}
    return {
        str(path.relative_to(root)): artifact_ref(path).sha256
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != _RECORD and not path.name.endswith(".tmp")
    }


def _payload_matches(record: Index, root: Path) -> bool:
    """Whether the payload under *root* is byte-for-byte the one *record* describes."""
    return bool(record.payload) and _payload_digest(root) == record.payload


def _verify_payload(record: Index, root: Path) -> None:
    """Refuse a payload the record does not describe: a killed or concurrent build, or a hand-edited index.

    The record is written last, so a build that died between its payload and its record leaves the *previous*
    record beside a partial payload; without this check the next search would score whatever bytes it found
    under the old identity.  Nothing is scored that the record does not describe.

    Raises:
        MissingInputError: A payload file the record names is absent (or the directory holds no payload):
            a half-written or emptied index, with the rebuild hint.
        DataError: The record carries no payload digest (an index written before the digest existed), or a
            payload file is unrecorded or has different bytes.
    """
    actual = _payload_digest(root)
    if actual and actual == record.payload:
        return
    missing = sorted(set(record.payload) - set(actual))
    if missing or not actual:
        raise MissingInputError(
            f"the index at {root} is missing {missing[:5] if missing else 'its whole payload'}",
            hint="the build was interrupted or the payload was removed; rebuild it with index(), or use "
            "retrieve(), which rebuilds when the payload is missing",
            cli_hint="rebuild it with `rcp-ndcg retrieval index`",
        )
    extra = sorted(set(actual) - set(record.payload))
    changed = sorted(name for name in set(actual) & set(record.payload) if actual[name] != record.payload[name])
    detail = "; ".join(
        part
        for part in (
            f"unrecorded {extra[:5]}" if extra else "",
            f"changed {changed[:5]}" if changed else "",
            "the record describes no payload" if not record.payload else "",
        )
        if part
    )
    raise DataError(
        f"the index at {root} does not match its record ({detail})",
        hint="the build was interrupted or the payload was replaced; rebuild it with index(), or use retrieve(), "
        "which rebuilds when the payload does not match",
        cli_hint="rebuild it with `rcp-ndcg retrieval index`",
    )


def _clear_payload(root: Path) -> None:
    """Remove the payload files an earlier build left, so the new record describes exactly the new build.

    A dense rebuild must not leave a late-interaction build's ``offsets.npy`` beside its vectors (the search
    would slice the new vectors by the old offsets), and a sparse rebuild must not leave a dense build's
    ``vectors.npy``.  Called after the new payload is computed (a failed encode leaves the old index intact)
    and before it is written.
    """
    for name in ("vectors.npy", "offsets.npy"):
        (root / name).unlink(missing_ok=True)
    shutil.rmtree(root / "bm25s", ignore_errors=True)


def _publish_array(target: Path, array: np.ndarray) -> None:
    """Write one payload array atomically: a temp file beside it, then one rename."""

    def write(tmp: Path) -> None:
        with tmp.open("wb") as handle:  # np.save appends .npy to a *name*, never to a file object
            np.save(handle, array)

    publish(target, write)


def index(dataset: Dataset, retriever: RetrieverConfig, *, out: str | Path) -> Index:
    """Build an index of a dataset's corpus.

    The payload is written atomically (a temp file and one rename per file, a directory swap for the sparse
    model) under an exclusive lock on the index directory, and ``index.json`` is written last, carrying the
    sha256 of every payload file.  A killed or concurrent build therefore leaves a record that does not
    describe its payload, which :func:`search` refuses instead of scoring.

    Args:
        dataset: The dataset; its corpus is read.
        retriever: The retriever.
        out: The index directory (created).  A local or shared-filesystem path: a remote URI is refused
            (an object store cannot be renamed into place).

    Returns:
        The :class:`Index`; ``<out>/index.json`` records it.

    Raises:
        ConfigError: ``out`` is a remote URI.
        DataError: The encoder answered zero-width document vectors (every document was empty and the config's
            ``empty_doc: omit_zero`` sends none).
    """
    root = local_dir(out, "the index directory")
    root.mkdir(parents=True, exist_ok=True)
    doc_ids, contents = _corpus(dataset)
    with publication_lock(root):
        if isinstance(retriever, BM25Config):
            from rcp_ndcg.retrieval import sparse

            _clear_payload(root)
            sparse.build_bm25_index(contents, root, stemmer=retriever.stemmer)
        elif isinstance(retriever, DenseConfig):
            embeddings = _encode(retriever.encoder, contents, EncodeRole.DOCUMENT)
            _refuse_empty_vectors(embeddings, side="document")
            _clear_payload(root)
            _publish_array(root / "vectors.npy", embeddings.as_matrix())
        else:
            embeddings = _encode(retriever.encoder, contents, EncodeRole.DOCUMENT)
            _refuse_empty_vectors(embeddings, side="document")
            if not embeddings.is_multi_vector:
                raise DataError(
                    "the late-interaction encoder answered one vector per document, not one per token: a "
                    "pooled answer is not a multi-vector index",
                    hint="serve the checkpoint's token_embed task (the pooling wire requests it and refuses a "
                    "pooled answer), or use a dense retriever for a pooled endpoint",
                )
            _clear_payload(root)
            _publish_array(root / "vectors.npy", embeddings.vectors)
            if embeddings.offsets is not None:
                _publish_array(root / "offsets.npy", embeddings.offsets)
        built = Index(
            path=str(root),
            dataset=dataset.name,
            retriever=retriever,
            identity=_identity(retriever, doc_ids, contents),
            num_documents=len(doc_ids),
            behaviour_version=INDEX_BEHAVIOUR_VERSION,
            payload=_payload_digest(root),
        )
        record = built.model_dump_json(by_alias=True, indent=2)
        publish(root / _RECORD, lambda tmp: tmp.write_text(record, encoding="utf-8"))
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
        DataError: The payload is not the one the record describes (a killed or concurrent build, a replaced
            or truncated file); nothing is scored from it. Also: the dataset holds no queries, or the encoder
            answered zero-width query vectors (every query was empty and ``empty_doc: omit_zero`` sends none).
        ConfigError: ``depth`` is not positive.
    """
    if depth <= 0:
        raise ConfigError(f"depth must be positive, got {depth}")
    _verify_payload(index, Path(index.path))
    doc_ids, contents = _corpus(dataset)
    if _identity(index.retriever, doc_ids, contents) != index.identity:
        raise IdentityError(
            f"the index at {index.path} was built over another corpus or retriever than {dataset.name!r}",
            hint="rebuild it with index(), or use retrieve(), which rebuilds when the identity differs",
            cli_hint="rebuild it with `rcp-ndcg retrieval index`",
        )
    queries = dataset.queries
    query_ids = list(queries)
    if not query_ids:
        raise DataError(
            f"{dataset.name!r} holds no queries: there is nothing to search for",
            hint="check the dataset's query source (an empty split, or a reader that dropped the queries)",
        )
    root, retriever = Path(index.path), index.retriever
    if isinstance(retriever, BM25Config):
        from rcp_ndcg.retrieval import sparse

        hits = sparse.search_bm25(root, [queries[q].format_query() for q in query_ids], k=min(depth, len(doc_ids)))
        scores = {q: {doc_ids[row]: score for row, score in hits_q} for q, hits_q in zip(query_ids, hits, strict=True)}
    elif isinstance(retriever, DenseConfig):
        from rcp_ndcg.retrieval.topk import score_topk

        vectors = np.load(root / "vectors.npy")
        documents = Embeddings(vectors=vectors)
        encoded = _encode(retriever.encoder, [queries[q].format_content() for q in query_ids], EncodeRole.QUERY)
        _refuse_empty_vectors(encoded, side="query")
        top_scores, top_indices = score_topk(documents, encoded, depth)
        scores = {
            q: {doc_ids[int(i)]: float(s) for s, i in zip(row_s, row_i, strict=True) if int(i) >= 0}
            for q, row_s, row_i in zip(query_ids, top_scores, top_indices, strict=True)
        }
    else:
        from rcp_ndcg.retrieval.topk import score_topk

        vectors = np.load(root / "vectors.npy")
        offsets = np.load(root / "offsets.npy") if (root / "offsets.npy").exists() else None
        if offsets is None:
            raise DataError(
                f"the late_interaction index at {root} has no offsets.npy: it is not a multi-vector index",
                hint="a pooled answer must not become a late-interaction index; rebuild it with index() over a "
                "token_embed endpoint",
                cli_hint="rebuild it with `rcp-ndcg retrieval index`",
            )
        documents = Embeddings(vectors=vectors, offsets=offsets)
        encoded = _encode(
            retriever.encoder,  # type: ignore[arg-type]  # the kind's union: ServedPooling here
            [queries[q].format_content() for q in query_ids],
            EncodeRole.QUERY,
        )
        _refuse_empty_vectors(encoded, side="query")
        top_scores, top_indices = score_topk(documents, encoded, depth)
        scores = {
            q: {doc_ids[int(i)]: float(s) for s, i in zip(row_s, row_i, strict=True) if int(i) >= 0}
            for q, row_s, row_i in zip(query_ids, top_scores, top_indices, strict=True)
        }
    return Rankings.from_scores(scores, system=_system_name(retriever))


def load_index(path: str | Path) -> Index:
    """The :class:`Index` recorded in ``<path>/index.json``, reading its payload from *path*.

    ``index.json``'s own ``path`` field records where the index was *built* -- provenance -- while the
    payload is read from the directory this function was given: a copied, moved, mounted or restored index
    directory is searched where it now is, and never silently reads the directory it was built in.

    Args:
        path: The index directory (a local or shared-filesystem path).

    Returns:
        The record, its ``path`` set to the directory it was read from.

    Raises:
        ConfigError: ``path`` is a remote URI, or the record was written by an older shape (the hint shows
            the new one); a non-UTF-8 or unreadable record is a cache miss, not a crash.
        MissingInputError: No ``index.json`` under ``path``.
    """
    root = local_dir(path, "the index directory")
    record = root / _RECORD
    if not record.is_file():
        raise MissingInputError(
            f"no index at {path}",
            hint="build one with rcp_ndcg.retrieval.index",
            cli_hint="build one with `rcp-ndcg retrieval index`",
        )
    try:
        loaded = Index.model_validate_json(record.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, OSError) as exc:
        from rcp_ndcg.support.config import config_error

        raise config_error(
            exc,
            model=Index,
            source=f"{record}",
            hint="the record is not readable UTF-8 JSON (a torn or corrupt write); rebuild it with "
            "rcp_ndcg.retrieval.index",
        ) from exc
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
    return loaded.model_copy(update={"path": str(root)})


def retrieve(
    dataset: Dataset, retriever: RetrieverConfig, *, depth: int = 150, out: str | Path | None = None
) -> Rankings:
    """Index (unless an index of the same identity exists under ``out``) and search.

    An index under ``out`` is reused only when its identity matches *and* its payload is the one its record
    describes; a missing, truncated or mismatched payload is a cache miss and is rebuilt, never scored.

    Args:
        dataset: The dataset.
        retriever: The retriever.
        depth: Documents per query.
        out: The index directory (a local or shared-filesystem path); defaults to ``indexes/<identity>``
            under the package cache.

    Returns:
        :class:`~rcp_ndcg.data.Rankings` with one system.

    Raises:
        ConfigError: ``out`` is a remote URI.
    """
    doc_ids, contents = _corpus(dataset)
    identity = _identity(retriever, doc_ids, contents)
    if out is None:
        from rcp_ndcg.support.paths import cache_dir

        out = cache_dir() / "indexes" / identity[:16]
    root = local_dir(out, "the index directory")
    record = root / _RECORD
    built: Index | None = None
    if record.is_file():
        try:
            candidate = load_index(root)
        except ConfigError:
            candidate = None  # an index.json of another shape (or a corrupt one) is a cache miss: rebuild it
        else:
            if candidate.identity == identity and _payload_matches(candidate, Path(candidate.path)):
                built = candidate
    if built is None:
        built = index(dataset, retriever, out=root)
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
        DataError: The rankings hold no candidates for the dataset, a ranked document is not in the corpus,
            a ranked query is not in the dataset, or two candidates share a query id.
    """
    from rcp_ndcg_core._records import RankingExample

    if depth <= 0:
        raise ConfigError(f"depth must be positive, got {depth}")
    candidates = rankings.top(depth).queries(system=system, dataset=dataset.name)
    if not candidates:
        raise DataError(
            f"the rankings hold no candidates for {dataset.name!r}" + (f" of system {system!r}" if system else ""),
            hint="check that the rankings' rows name the dataset (and its subset, e.g. hr__english), or "
            "rerank the system that ranked it",
        )
    corpus, queries = dataset.corpus, dataset.queries
    missing = sorted({d for docs in candidates.values() for d in docs if d not in corpus})
    if missing:
        raise DataError(f"{len(missing)} ranked documents are not in {dataset.name!r}'s corpus, e.g. {missing[:3]}")
    examples = []
    for query_id, scores in candidates.items():
        if query_id not in queries:
            raise DataError(f"query {query_id!r} of the rankings is not in {dataset.name!r}")
        order = sorted(scores, key=lambda d: (-scores[d], d))  # score descending, then the lower document id
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
    scored = _rerank_examples(examples, reranker, checkpoint_dir=out)
    return Rankings.from_scores(
        {e.id: dict(zip(e.doc_ids, e.scores or [], strict=True)) for e in scored},
        system=reranker.model,
        dataset=rankings.resolve_dataset(dataset.name) or "",
    )


def fuse(rankings: Sequence[Rankings], *, rrf_k: int = 60, depth: int = 150, system: str = "rrf") -> Rankings:
    """Reciprocal rank fusion of several rankings: ``sum_r 1 / (rrf_k + rank_r)``, ranks from 1.

    Rows of different datasets are fused apart, so the rankings of a suite keep their subsets.  Every system
    of every input is a ranker of each subset its own rows cover: a file assembled from several systems (a
    :meth:`~rcp_ndcg.data.Rankings.concat`) whose systems cover different subsets fuses each subset from the
    systems that rank it, and a system with no rows for a subset stays out of that subset's fusion.  Ties
    break by earliest appearance, systems in argument order (deterministic for a fixed input order).

    Args:
        rankings: One :class:`~rcp_ndcg.data.Rankings` per input (every system of each is fused).
        rrf_k: The RRF constant.
        depth: Documents per query after fusion.
        system: The fused system's name.

    Returns:
        :class:`~rcp_ndcg.data.Rankings` with one system; its scores are the RRF scores.

    Raises:
        DataError: No rankings to fuse, an input holds no rows, or two rankings that share a subset ranking
            different queries.
        ConfigError: ``depth`` or ``rrf_k`` is not positive.
    """
    from rcp_ndcg_core._records import RankingExample

    from rcp_ndcg.retrieval.fusion import reciprocal_rank_fusion

    if depth <= 0:
        raise ConfigError(
            f"depth must be positive, got {depth}",
            hint="pass the documents per query kept after fusion",
        )
    if rrf_k <= 0:
        raise ConfigError(
            f"rrf_k must be positive, got {rrf_k}",
            hint="the RRF constant k is at least 1; the paper's is 60",
        )
    for position, ranking in enumerate(rankings, start=1):
        if not ranking.systems:
            raise DataError(
                f"rankings input {position} of {len(rankings)} holds no rows: fusing it would silently "
                "contribute nothing and the result would look like a real fusion",
                hint="pass rankings files that hold rows; an empty one is usually a failed run",
            )
    datasets = list(dict.fromkeys(name for ranking in rankings for name in ranking.datasets))
    if not datasets:
        raise DataError("fuse needs rankings to fuse; got none")
    fused_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        runs: list[list[Any]] = []
        for ranking in rankings:
            if ranking.resolve_dataset(dataset) is None:
                continue  # a file whose rows rank no subset of this name stays out of it
            for name in ranking.systems:
                scores = ranking.queries(system=name, dataset=dataset)
                if not scores:
                    continue  # this system ranks no query of the subset: not a ranker of it
                runs.append(
                    [
                        RankingExample(
                            query_id=q, query="", doc_ids=sorted(docs, key=lambda d: (docs[d], d), reverse=True)
                        )
                        for q, docs in scores.items()
                    ]
                )
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


def _checkpoint_key(config: RerankerConfig, example: Any, *, tokenizer_sha256: str | None = None) -> str:
    """What a checkpointed query's scores are valid for: the reranker's content identity and the exact texts.

    The payload is :func:`~rcp_ndcg.support.identity.identity_payload` of the config (the model, its revision,
    the wire adapter, the recipe, the instruction mode, the activation switch and the budgets), the
    tokenizer's SHA-256, :data:`RERANK_BEHAVIOUR_VERSION`, and what goes over the wire for this query: its id,
    its raw text and instruction, and the candidate ids with a digest of their contents. A rerun after any of
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
    """
    payload = identity_payload(config)
    payload["behaviour_version"] = RERANK_BEHAVIOUR_VERSION
    digest = tokenizer_sha256 if tokenizer_sha256 is not None else config.identity_extra().get("tokenizer_sha256")
    if digest:
        payload["tokenizer_sha256"] = digest
    payload.update(
        {
            "query_id": str(example.id),
            "query": example.as_content.model_dump_json(),
            "query_instruction": example.instruction,
            "doc_ids": [str(doc_id) for doc_id in example.doc_ids],
            "docs": hash_strings([content_identity(content) for content in example.doc_contents]),
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
    checkpoint_dir: str | Path | None,
) -> list[Any]:
    """Score every example through the rerank client, checkpointing per query.

    Args:
        examples: :class:`~rcp_ndcg_core._records.RankingExample` records with their documents populated.
        config: The reranker.
        checkpoint_dir: When given, each scored query is appended to ``<dir>/rank000.jsonl`` (one record
            ``{"q", "k", "s"}``, flushed and fsynced) as it finishes, and the queries the directory already holds
            are skipped.

    Returns:
        The examples in input order with ``scores`` set, aligned to ``doc_ids``.

    Raises:
        DataError: A document has no score. The candidate set is part of the run's identity, so a document is
            neither dropped nor given a made-up score; or two candidates share a query id, which would
            attribute one query's scores to another.
    """
    client = RerankClient(config)
    tokenizer_sha256 = config.identity_extra().get("tokenizer_sha256")
    ids = [str(example.id) for example in examples]
    duplicates = sorted({query_id for query_id in ids if ids.count(query_id) > 1})
    if duplicates:
        raise DataError(
            f"the rerank candidates hold {len(duplicates)} query id(s) more than once, e.g. {duplicates[:3]}: "
            "a query's scores are checkpointed and applied under its id, so duplicates would be attributed "
            "to one another",
            hint="make the query ids unique in the dataset, or rerank the subsets separately",
        )
    keys = [_checkpoint_key(config, example, tokenizer_sha256=tokenizer_sha256) for example in examples]
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
            client.rerank_many(pending, checkpoint=checkpoint)
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


def _refuse_empty_vectors(embeddings: Embeddings, *, side: str) -> None:
    """Refuse a zero-width embedding buffer: every item was empty and the config's ``empty_doc: omit_zero``
    sends none.

    A zero-width buffer is an all-omitted batch (``Embeddings.concat`` names it too): scoring it used to die
    inside :func:`~rcp_ndcg.retrieval.topk.numpy_topk` with a late-interaction message, or build a zero-width
    index that failed on the next search.

    Args:
        embeddings: The encoder's answer.
        side: ``"document"`` or ``"query"``, for the message.

    Raises:
        DataError: The buffer holds items but no width.
    """
    if embeddings.num_items and embeddings.dim < 1:
        raise DataError(
            f"the encoder answered zero-width {side} vectors: every {side} was empty and the config's "
            "empty_doc: omit_zero sends none",
            hint="declare empty_doc: send (or send_text) so an empty item still gets a vector, or drop the "
            "empty items from the dataset",
        )


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

    A text-only document hashes as its plain string; a document with media hashes as its parts plus each
    media reference's fingerprint (:func:`rcp_ndcg.data.media.content_identity`), since two page corpora with
    the same (empty) text are otherwise indistinguishable -- and an unhashed reference's size and change
    stamp are what make a replaced object at the same URI move the identity.
    """
    bodies = [content.text if not content.has_media else content_identity(content) for content in contents]
    return combine_digests(hash_strings(doc_ids), hash_strings(bodies))


def _identity(retriever: RetrieverConfig, doc_ids: list[str], contents: list[Any]) -> str:
    """What an index is: the retriever's content fields (``IDENTITY_ROLES``), the encoder's tokenizer digest,
    the corpus, and :data:`INDEX_BEHAVIOUR_VERSION`.

    The tokenizer digest (``identity_extra()``) is spliced in at the encoder, as the step identities splice it:
    what cuts the text is content, and the tokenizer's *name* is not. Two configs that declare different
    tokenizer bytes never share an index, so a corpus indexed before the text-budget mechanism lands is not
    silently reused once budgets become content-bearing. The behaviour version is the last part: a change to
    what an index *is* that moves no config field still re-keys every index (the package version is not used --
    every release would invalidate every resume and judgement pool).
    """
    payload = identity_payload(retriever)
    payload = {**payload, "behaviour_version": INDEX_BEHAVIOUR_VERSION}
    encoder = getattr(retriever, "encoder", None)
    if encoder is not None:
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


def _encode(config: Any, contents: Sequence[Any], role: EncodeRole) -> Embeddings:
    """Encode *contents* through the encoder config's role client, its transport closed after the call.

    Any pooling config (:class:`~rcp_ndcg.retrieval.config.ServedPooling` or a third-party
    :class:`~rcp_ndcg.retrieval.config.PluginPooling`) runs the
    :class:`~rcp_ndcg.inference.clients.PoolingClient` (ragged); every embedding config -- served, hosted, or
    a third-party :class:`~rcp_ndcg.retrieval.config.PluginEmbedding` -- the
    :class:`~rcp_ndcg.inference.clients.EmbeddingClient`, a hosted profile at the vendor's public URL.

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
        return client.encode(contents, role)
    finally:
        client.close()  # the client base's sync close (an injected sender closes nothing)


__all__ = [
    "INDEX_BEHAVIOUR_VERSION",
    "RERANK_BEHAVIOUR_VERSION",
    "RETRIEVE_BEHAVIOUR_VERSION",
    "Index",
    "fuse",
    "index",
    "load_index",
    "rerank",
    "retrieve",
    "search",
]
