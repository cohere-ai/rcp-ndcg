"""The full-width embedding store and the ex-post Matryoshka sweep.

:func:`build_store` encodes a dataset's corpus and queries ONCE at the checkpoint's full width (the
configured ``dimensions``/``mrl_dim`` selection is stripped for the forward pass, so ``k`` never reaches
the engine) and persists both through :mod:`rcp_ndcg.data.embedding_store`. :func:`sweep` applies the
declared MRL head to the stored vectors per ``k`` and scores each cut with the retrieval scorer (dense
inner product or MaxSim), so every ``k`` costs one head application and one scoring pass -- never another
forward pass. The per-k rankings name the system ``<model>@<k>``, which is what keys the derived artifact
and its evaluation.

The store's identity is the retrieval identity of the full-width encoder plus a full-width marker; a
k-selected index keeps its own identity (``dimensions``/``mrl_dim`` are CONTENT), so the two artifacts
never collide.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from rcp_ndcg.data.dataset import Dataset
from rcp_ndcg.data.embedding_store import (
    EmbeddingStore,
    StoredVectors,
    load_embedding_store,
    save_embedding_store,
)
from rcp_ndcg.data.mrl import MrlHead
from rcp_ndcg.data.rankings import Rankings
from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.inference.types import Embeddings, EncodeRole
from rcp_ndcg.retrieval._api import _corpus, _encode, _identity
from rcp_ndcg.retrieval.config import BM25Config, DenseConfig, LateInteractionConfig, RetrieverConfig
from rcp_ndcg.support.identity import combine_digests, hash_payload, hash_strings, identity_payload, short

#: The request-shaping fields whose digest the store records as its ``prompt_digest``.
_PROMPT_FIELDS: tuple[str, ...] = (
    "query_prompt",
    "doc_prompt",
    "template",
    "request_shape",
    "add_generation_prompt",
    "empty_doc",
    "empty_doc_text",
    "on_overflow",
    "chunk",
    "aggregation",
    "max_tokens",
    "query_max_tokens",
)

#: The full-width marker the store's identity carries (a cut index never shares a store's key).
_FULL_WIDTH_MARKER = "full_width"


def build_store(dataset: Dataset, retriever: RetrieverConfig, *, out: str | Path) -> EmbeddingStore:
    """Encode a dataset's corpus and queries at full width and persist the store.

    The retriever's ``dimensions``/``mrl_dim`` selection is stripped for this pass only: the store holds
    the checkpoint's own vectors, and the declared head (kind, set, projection) is recorded for the sweep.
    The corpus and query vectors must share one width; a retriever with no vectors (BM25) is refused.

    Args:
        dataset: The dataset; its corpus and queries are encoded.
        retriever: The dense or late-interaction retriever whose encoder computes the vectors.
        out: The store directory (created).

    Returns:
        The :class:`~rcp_ndcg.data.EmbeddingStore` record; ``<out>/store.json`` records it.

    Raises:
        ConfigError: ``retriever`` is BM25 (it has no vectors to store).
        DataError: the encoder answered corpus and query vectors of different widths, or an empty width.
    """
    if isinstance(retriever, BM25Config):
        raise ConfigError(
            "a BM25 retriever has no vectors to store: the store persists embedding vectors",
            hint="build a store from a dense or late_interaction retriever (an encoder with vectors), or "
            "use the BM25 index",
        )
    assert isinstance(retriever, (DenseConfig, LateInteractionConfig))
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    doc_ids, contents = _corpus(dataset)
    queries = dataset.queries
    query_ids = list(queries)
    query_contents = [queries[query_id].format_content() for query_id in query_ids]
    full = _full_width(retriever)
    corpus_embeddings = _encode(full.encoder, contents, EncodeRole.DOCUMENT)
    query_embeddings = _encode(full.encoder, query_contents, EncodeRole.QUERY)
    if corpus_embeddings.dim < 1:
        raise DataError(
            "the encoder answered zero-width corpus vectors: there is no full width to store",
            hint="check the encoder's reply shape (the served model's own output width)",
        )
    if query_embeddings.num_items and query_embeddings.dim != corpus_embeddings.dim:
        raise DataError(
            f"the encoder answered {corpus_embeddings.dim}-wide corpus vectors and "
            f"{query_embeddings.dim}-wide query vectors; one endpoint's embeddings share a width",
            hint="one encoder computes both sides of a store",
        )
    encoder = retriever.encoder
    record = EmbeddingStore(
        path=str(root),
        dataset=dataset.name,
        identity=_store_identity(full, doc_ids, contents),
        layout="ragged" if corpus_embeddings.is_multi_vector else "single",
        model=encoder.model,
        revision=encoder.revision,
        recipe=encoder.recipe,
        prompt_digest=_prompt_digest(encoder),
        tokenizer=encoder.tokenizer,
        tokenizer_sha256=encoder.identity_extra().get("tokenizer_sha256"),
        max_tokens=encoder.max_tokens,
        full_width=corpus_embeddings.dim,
        dtype=str(np.dtype(corpus_embeddings.vectors.dtype)),
        mrl_kind=encoder.mrl_kind or "none",
        mrl_dims=tuple(encoder.mrl_dims or ()),
        mrl_range=encoder.mrl_range,
        mrl_projection=encoder.mrl_projection,
        count=len(doc_ids),
        query_count=len(query_ids),
        doc_ids=list(doc_ids),
        query_ids=query_ids,
    )
    save_embedding_store(
        root,
        record,
        StoredVectors(vectors=corpus_embeddings.vectors, offsets=corpus_embeddings.offsets),
        StoredVectors(vectors=query_embeddings.vectors, offsets=query_embeddings.offsets),
    )
    return record


def load_store(path: str | Path) -> tuple[EmbeddingStore, Embeddings, Embeddings]:
    """The store at ``path``: its record and its corpus and query vectors as the clients' type.

    Raises:
        MissingInputError: no store under ``path``.
        DataError: the store's vectors disagree with its record (a cut store, a count mismatch).
    """
    record, corpus, queries = load_embedding_store(path)
    return (
        record,
        Embeddings(vectors=corpus.vectors, offsets=corpus.offsets),
        Embeddings(vectors=queries.vectors, offsets=queries.offsets),
    )


def sweep(
    store: EmbeddingStore,
    corpus: Embeddings,
    queries: Embeddings,
    *,
    dims: Sequence[int] | None = None,
    depth: int = 150,
) -> list[Rankings]:
    """Apply the declared MRL head per ``k`` and score every cut from the stored vectors.

    One head application and one scoring pass per ``k``; no model is called (the store holds the full-width
    vectors from the one forward pass). Each ranking's system is ``<model>@<k>``.

    Args:
        store: The store record (its ``mrl_kind``, ``mrl_dims``/``mrl_range`` and ``mrl_projection``
            configure the head).
        corpus: The stored corpus vectors (full width).
        queries: The stored query vectors (full width).
        dims: The ``k`` values to sweep, each selectable under the store's declaration; ``None`` sweeps
            every declared dimension, in declaration order (refused when the store declares a range: a
            range cannot be enumerated).
        depth: Documents per query.

    Returns:
        One :class:`~rcp_ndcg.data.Rankings` per selected ``k``, in the given (or declared) order.

    Raises:
        ConfigError: ``depth`` is not positive, the store declares no head, ``dims`` is unset for a range
            declaration, or a requested ``k`` is not selectable.
        DataError: the head cannot apply (a projection source that does not line up).
    """
    if depth <= 0:
        raise ConfigError(f"depth must be positive, got {depth}", hint="pass the number of documents per query")
    head = MrlHead(
        kind=store.mrl_kind,
        dims=store.mrl_dims,
        mrl_range=store.mrl_range,
        projection=store.mrl_projection,
    )
    if not head.dims and head.mrl_range is None:
        raise ConfigError(
            f"the store declares no mrl_dims or mrl_range: there is nothing to sweep for {store.model}",
            hint="build the store from an encoder that declares mrl_kind with mrl_dims or mrl_range (the "
            "card's set), or declare them on the retriever",
        )
    if dims is None:
        if not store.mrl_dims:
            raise ConfigError(
                f"the store declares {head.declaration} and no mrl_dims: a range cannot be enumerated, so "
                "the sweep needs explicit dims",
                hint="pass --dims with the k values to evaluate (each inside the declared range)",
            )
        selected = tuple(store.mrl_dims)
    else:
        selected = tuple(dims)
        if not selected:
            raise ConfigError(
                "dims is empty: there is no k to sweep",
                hint="pass the k values to evaluate (each selectable under the store's declaration), or "
                "leave dims unset to sweep every declared mrl_dims",
            )
        unknown = sorted({k for k in selected if not head.supports(k)})
        if unknown:
            raise ConfigError(
                f"dims {unknown} are not in the store's declared {head.declaration}",
                hint=f"select k values from the store's declared {head.declaration}",
            )
    from rcp_ndcg.retrieval.topk import score_topk

    rankings: list[Rankings] = []
    for k in selected:
        cut_corpus = Embeddings(vectors=head.apply(corpus.vectors, k), offsets=corpus.offsets)
        cut_queries = Embeddings(vectors=head.apply(queries.vectors, k), offsets=queries.offsets)
        top_scores, top_indices = score_topk(cut_corpus, cut_queries, depth)
        scores = {
            query_id: {
                store.doc_ids[int(index)]: float(score)
                for score, index in zip(row_scores, row_indices, strict=True)
                if int(index) >= 0
            }
            for query_id, row_scores, row_indices in zip(store.query_ids, top_scores, top_indices, strict=True)
        }
        rankings.append(Rankings.from_scores(scores, system=f"{store.model}@{k}", dataset=store.dataset))
    return rankings


def _full_width(retriever: DenseConfig | LateInteractionConfig) -> DenseConfig | LateInteractionConfig:
    """The retriever with the Matryoshka SELECTION stripped: the forward pass is always full width."""
    encoder = retriever.encoder
    stripped = encoder.model_copy(update={"dimensions": None, "mrl_dim": None})
    return retriever.model_copy(update={"encoder": stripped})


def _prompt_digest(encoder: object) -> str:
    """The digest of the request-shaping fields of ``encoder`` (the prompts, template, budget, shapes)."""
    payload = identity_payload(encoder)  # type: ignore[arg-type]  # a role config by contract
    prompt = {field: payload[field] for field in _PROMPT_FIELDS if field in payload}
    return short(hash_payload(prompt), 16)


def _store_identity(
    retriever: DenseConfig | LateInteractionConfig, doc_ids: Sequence[str], contents: Sequence[object]
) -> str:
    """The retrieval identity of the full-width retriever plus the full-width marker."""
    return combine_digests(
        _identity(retriever, list(doc_ids), list(contents)),  # type: ignore[arg-type]  # the index's own corpus hash
        hash_strings([_FULL_WIDTH_MARKER]),
    )


__all__ = ["build_store", "load_store", "sweep"]
