"""The full-width embedding store: corpus and query vectors from one forward pass, plus their provenance.

The store is what makes the Matryoshka sweep cheap: every vector is persisted at the checkpoint's full
width (``mrl_dim``/``dimensions`` never reach the engine when it is built), and the ex-post sweep applies
the declared MRL head per ``k`` to the stored vectors, so the forward pass happens once for every ``k``.
A store is keyed by the retrieval identity plus a full-width marker (:func:`rcp_ndcg.retrieval.build_store`),
so a cut store can never be mistaken for a full-width one -- and this reader refuses vectors narrower
than the record's ``full_width`` for the same reason.

Layout::

    <root>/store.json            the typed record (schema, provenance, ids)
    <root>/corpus.npy            (count, full_width), or the flat ragged buffer
    <root>/corpus_offsets.npy    (count + 1,) for a late-interaction store
    <root>/queries.npy           (query_count, full_width), or the flat ragged buffer
    <root>/queries_offsets.npy   (query_count + 1,) for a late-interaction store

Bytes and URIs go through :mod:`rcp_ndcg.storage`, so a store can live on a local disk or an object store.
The typed record and this reader live here; the encoding and the sweep are wired in
:mod:`rcp_ndcg.retrieval`.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from rcp_ndcg.data.mrl import MrlKind, MrlProjection
from rcp_ndcg.errors import DataError, MissingInputError
from rcp_ndcg.storage import exists, join, publish_bytes, read_bytes, read_text

STORE_SCHEMA = "rcp-ndcg.embedding-store.v1"
"""The store record's schema id (carried in its ``schema`` field)."""


@dataclass(frozen=True)
class StoredVectors:
    """Vectors read from a store: single-vector ``(count, width)`` or a ragged flat buffer with offsets.

    The data layer's own container (``rcp_ndcg.inference.types.Embeddings`` lives above this layer); the
    retrieval wiring wraps one of these into the client's ``Embeddings``.

    Attributes:
        vectors: ``(count, full_width)`` when single-vector; ``(total_vectors, full_width)`` when ragged,
            in item order.
        offsets: ``None`` when single-vector; otherwise ``(count + 1,)``, item ``i`` owning
            ``vectors[offsets[i]:offsets[i + 1]]``.
    """

    vectors: np.ndarray
    offsets: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise DataError(f"stored vectors must be 2-D (got shape {self.vectors.shape})")
        if self.offsets is None:
            return
        if self.offsets.ndim != 1 or len(self.offsets) < 1 or int(self.offsets[0]) != 0:
            raise DataError(f"stored offsets must be a 1-D array starting at 0, got shape {self.offsets.shape}")
        if int(self.offsets[-1]) != len(self.vectors) or np.any(np.diff(self.offsets) < 0):
            raise DataError("stored offsets must be non-decreasing and end at the vector count")

    @property
    def is_multi_vector(self) -> bool:
        """Whether the buffer is ragged (one slice of vectors per item)."""
        return self.offsets is not None

    @property
    def num_items(self) -> int:
        """Items described by the buffer."""
        return len(self.vectors) if self.offsets is None else len(self.offsets) - 1

    @property
    def dim(self) -> int:
        """The stored width (``0`` for an empty buffer)."""
        return int(self.vectors.shape[1])


class EmbeddingStore(BaseModel):
    """A full-width embedding store's record: what is stored and the provenance of the vectors.

    Attributes:
        schema_name: ``rcp-ndcg.embedding-store.v1`` (the ``schema`` field).
        path: The store directory.
        dataset: The dataset the corpus and queries were read from (its name, as the index records it).
        identity: The store's content identity: the retrieval identity of the full-width encoder plus a
            full-width marker (:func:`rcp_ndcg.retrieval.build_store`). ``k`` never enters it.
        layout: ``"single"`` (one vector per item) or ``"ragged"`` (late interaction: a flat token-vector
            buffer with offsets).
        model: The served model name the vectors came from.
        revision: The checkpoint revision, when the encoder declares one.
        recipe: The serving recipe id, when the encoder declares one.
        prompt_digest: The digest of the request-shaping fields (the prompts, the template, the budget and
            the shapes) that produced the vectors.
        tokenizer: The tokenizer spec the text budget counted in, when declared.
        tokenizer_sha256: The tokenizer's content digest, when resolvable.
        max_tokens: The declared text budget (tokens), when declared.
        full_width: The width of every stored vector -- the checkpoint's own output width, never a cut.
        dtype: The stored dtype (``float32``, or ``float16`` for a late-interaction store in its transfer
            precision).
        mrl_kind: The declared Matryoshka kind (``truncation``, ``projection`` or ``none``).
        mrl_dims: The declared set of supported output dimensions (``k`` values the sweep may select).
        mrl_range: The declared closed range of output dimensions, when the card gives a range instead of a
            set; the sweep then needs explicit ``dims`` (a range cannot be enumerated).
        mrl_projection: Where a projection kind's learned matrices live, when declared.
        count: Corpus rows (documents).
        query_count: Query rows.
        doc_ids: The corpus ids in stored row order.
        query_ids: The query ids in stored row order.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    schema_name: Literal["rcp-ndcg.embedding-store.v1"] = Field(default=STORE_SCHEMA, alias="schema")  # type: ignore[assignment]
    path: str = Field(min_length=1)
    dataset: str
    identity: str = Field(min_length=1)
    layout: Literal["single", "ragged"]
    model: str = Field(min_length=1)
    revision: str | None = None
    recipe: str | None = None
    prompt_digest: str = Field(min_length=1)
    tokenizer: str | None = None
    tokenizer_sha256: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    full_width: int = Field(ge=1)
    dtype: str = Field(min_length=1)
    mrl_kind: MrlKind = "none"
    mrl_dims: tuple[int, ...] = ()
    mrl_range: tuple[int, int] | None = None
    mrl_projection: MrlProjection | None = None
    count: int = Field(ge=0)
    query_count: int = Field(ge=0)
    doc_ids: list[str]
    query_ids: list[str]


def _npy_bytes(array: np.ndarray) -> bytes:
    """One array as ``.npy`` bytes (through :func:`numpy.save`, so the reader is plain ``numpy.load``)."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _load_npy(uri: str) -> np.ndarray:
    """One ``.npy`` array from a local path or a storage URI."""
    return np.load(io.BytesIO(read_bytes(uri)))


def _check(record: EmbeddingStore, corpus: StoredVectors, queries: StoredVectors) -> None:
    """The store's own consistency: full width, declared dtype, declared counts and ids, one layout.

    Raises:
        DataError: the vectors disagree with the record -- the shape a hand-edited or cut store takes.
    """
    if (
        record.full_width < 1
        or corpus.dim != record.full_width
        or (queries.num_items and queries.dim != record.full_width)
    ):
        raise DataError(
            f"the store declares full_width {record.full_width}, but its corpus vectors are {corpus.dim}-wide "
            f"and its query vectors are {queries.dim}-wide",
            hint="a store holds full-width vectors; a cut store cannot be read as full width (rebuild it "
            "with rcp_ndcg.retrieval.build_store)",
        )
    for side, vectors, expected, ids in (
        ("corpus", corpus, record.count, record.doc_ids),
        ("queries", queries, record.query_count, record.query_ids),
    ):
        if vectors.num_items != expected or len(ids) != expected:
            raise DataError(
                f"the store declares {expected} {side} row(s) but holds {vectors.num_items} vector set(s) "
                f"and {len(ids)} id(s)",
                hint="rebuild the store; the record and the arrays must describe the same rows",
            )
        if str(np.dtype(vectors.vectors.dtype)) != record.dtype:
            raise DataError(
                f"the store declares dtype {record.dtype!r} but its {side} vectors are "
                f"{np.dtype(vectors.vectors.dtype)}",
                hint="rebuild the store; the record and the arrays must share one dtype",
            )
        if vectors.is_multi_vector != (record.layout == "ragged"):
            raise DataError(
                f"the store declares layout {record.layout!r} but its {side} vectors "
                f"{'are ragged' if vectors.is_multi_vector else 'are single-vector'}",
                hint="rebuild the store; the layout follows the retriever kind",
            )


def save_embedding_store(
    path: str | Path, record: EmbeddingStore, corpus: StoredVectors, queries: StoredVectors
) -> None:
    """Write ``record`` and its two vector buffers under ``path`` (through :mod:`rcp_ndcg.storage`).

    Raises:
        DataError: the vectors disagree with the record (a cut store, a count mismatch, two layouts).
    """
    _check(record, corpus, queries)
    publish_bytes(join(path, "corpus.npy"), _npy_bytes(corpus.vectors))
    if corpus.offsets is not None:
        publish_bytes(join(path, "corpus_offsets.npy"), _npy_bytes(corpus.offsets))
    publish_bytes(join(path, "queries.npy"), _npy_bytes(queries.vectors))
    if queries.offsets is not None:
        publish_bytes(join(path, "queries_offsets.npy"), _npy_bytes(queries.offsets))
    publish_bytes(join(path, "store.json"), record.model_dump_json(by_alias=True, indent=2).encode("utf-8"))


def load_embedding_store(path: str | Path) -> tuple[EmbeddingStore, StoredVectors, StoredVectors]:
    """Read the store at ``path``: its record and both vector buffers.

    Raises:
        MissingInputError: no ``store.json`` under ``path``.
        DataError: the vectors disagree with the record (a cut store, a count mismatch, two layouts).
    """
    record_path = join(path, "store.json")
    if not exists(record_path):
        raise MissingInputError(
            f"no embedding store at {path}",
            hint="build one with rcp_ndcg.retrieval.build_store",
            cli_hint="build one with `rcp-ndcg retrieval store`",
        )
    record = EmbeddingStore.model_validate_json(read_text(record_path))
    # A store rebuilt in place across layouts can leave the other layout's offsets behind: only the
    # record's declared layout reads them, so a stale ``*_offsets.npy`` is never mistaken for the data.
    ragged = record.layout == "ragged"
    corpus_offsets = join(path, "corpus_offsets.npy")
    query_offsets = join(path, "queries_offsets.npy")
    corpus = StoredVectors(
        vectors=_load_npy(join(path, "corpus.npy")),
        offsets=_load_npy(corpus_offsets) if ragged and exists(corpus_offsets) else None,
    )
    queries = StoredVectors(
        vectors=_load_npy(join(path, "queries.npy")),
        offsets=_load_npy(query_offsets) if ragged and exists(query_offsets) else None,
    )
    _check(record, corpus, queries)
    return record, corpus, queries


__all__ = [
    "STORE_SCHEMA",
    "EmbeddingStore",
    "StoredVectors",
    "load_embedding_store",
    "save_embedding_store",
]
