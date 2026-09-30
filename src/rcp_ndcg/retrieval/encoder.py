"""One encoder contract: content in, vectors out, whatever runs it.

An :class:`Encoder` takes :class:`~rcp_ndcg_core.content.Content` and returns
:class:`Embeddings`. Three things follow from that shape:

* **A text corpus and a page corpus go down one path.** ``Content`` is the same
  type either way, so nothing between the reader and the index branches on
  modality. An encoder that cannot see pixels says so once, in
  :attr:`Encoder.supports_media`, and fails with an explanation instead of
  silently embedding the empty string an image-only document reads as.
* **Single-vector and late-interaction are the same call.** :class:`Embeddings`
  carries an optional offsets array, so ColPali-style per-token output is a
  ragged buffer rather than a separate code path.
* **Where the model runs is a config key.** An encoder config's ``provider``
  picks an implementation (:mod:`rcp_ndcg.retrieval.encoders`); the caller's
  code is identical.

Role belongs to the *call*, not the encoder. Asymmetric embedders need to know
whether they are seeing a query or a document, but that is a property of the
batch; one encoder object for both sides keeps the query and document paths from
drifting apart.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import CapabilityError


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Every row of ``vectors`` scaled to unit L2 norm (a zero row stays zero), as float32."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


class EncodeRole(StrEnum):
    """Which side of the retrieval pair a batch is.

    Asymmetric embedders prepend different instructions per side, and getting it
    wrong costs recall without raising -- so it is a required argument rather
    than a default.
    """

    QUERY = "query"
    DOCUMENT = "document"


@dataclass(frozen=True)
class Embeddings:
    """Vectors for a batch of items, single- or multi-vector.

    One type for both because the difference is a layout detail, not a different
    kind of answer.  Multi-vector output is stored ragged -- all vectors
    concatenated, plus an offsets array -- rather than padded to a rectangle: a
    page can be 40 patches or 4000, and padding to the maximum wastes most of
    the buffer while making a padded row indistinguishable from a real one.

    Attributes:
        vectors: ``(N, D)`` when single-vector; ``(total_vectors, D)`` when
            ragged, in item order.
        offsets: ``None`` when single-vector; otherwise ``(N + 1,)``, so item
            ``i`` owns ``vectors[offsets[i]:offsets[i + 1]]``.
    """

    vectors: np.ndarray
    offsets: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(
                f"Embeddings.vectors must be 2-D (got shape {self.vectors.shape}). "
                "Multi-vector output is stored flat with `offsets`, not as a 3-D array."
            )
        if self.offsets is None:
            return
        if self.offsets.ndim != 1 or len(self.offsets) < 1:
            raise ValueError(f"offsets must be a 1-D array of length num_items + 1, got shape {self.offsets.shape}")
        if int(self.offsets[0]) != 0:
            raise ValueError(f"offsets must start at 0, got {int(self.offsets[0])}")
        if int(self.offsets[-1]) != len(self.vectors):
            raise ValueError(
                f"offsets end at {int(self.offsets[-1])} but there are {len(self.vectors)} vectors; "
                "every vector must belong to exactly one item"
            )
        if np.any(np.diff(self.offsets) < 0):
            raise ValueError("offsets must be non-decreasing")

    # -- shape -------------------------------------------------------------
    @property
    def is_multi_vector(self) -> bool:
        return self.offsets is not None

    @property
    def num_items(self) -> int:
        return len(self.vectors) if self.offsets is None else len(self.offsets) - 1

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def as_matrix(self) -> np.ndarray:
        """The ``(N, D)`` matrix, for the single-vector consumers (the top-k search).

        Raises on multi-vector rather than pooling silently: collapsing a
        late-interaction model to one vector per document is a different
        retrieval method with different numbers, and it should be asked for.
        """
        if self.offsets is not None:
            raise ValueError(
                f"as_matrix() on multi-vector embeddings ({self.num_items} items, "
                f"{len(self.vectors)} vectors). Use MaxSim scoring, or pool explicitly first."
            )
        return self.vectors

    # -- construction ------------------------------------------------------
    @classmethod
    def single(cls, vectors: np.ndarray) -> Embeddings:
        return cls(vectors=np.ascontiguousarray(vectors, dtype=np.float32))

    @classmethod
    def ragged(cls, per_item: Sequence[np.ndarray]) -> Embeddings:
        """Build from one ``(T_i, D)`` array per item (width 0 when every item is empty)."""
        lengths = [len(item) for item in per_item]
        width = next((int(item.shape[1]) for item in per_item if len(item)), 0)
        stacked = (
            np.concatenate([np.asarray(item, dtype=np.float32) for item in per_item if len(item)], axis=0)
            if any(lengths)
            else np.zeros((0, width), dtype=np.float32)
        )
        offsets = np.zeros(len(per_item) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        return cls(vectors=np.ascontiguousarray(stacked, dtype=np.float32), offsets=offsets)

    @classmethod
    def empty(cls, dim: int, *, multi_vector: bool = False) -> Embeddings:
        """The zero-item value, so an empty shard needs no special-casing."""
        vectors = np.zeros((0, dim), dtype=np.float32)
        return cls(vectors=vectors, offsets=np.zeros(1, dtype=np.int64) if multi_vector else None)

    # -- transforms --------------------------------------------------------
    def l2_normalized(self) -> Embeddings:
        """Unit-norm every vector, so an inner product is a cosine.

        Applies per *vector*, not per item, which is what MaxSim needs too.
        """
        return Embeddings(vectors=l2_normalize(self.vectors), offsets=self.offsets)

    def concat(self, other: Embeddings) -> Embeddings:
        """Append *other*'s items after this one's."""
        if self.is_multi_vector != other.is_multi_vector:
            raise ValueError("cannot concatenate single-vector and multi-vector embeddings")
        if self.num_items and other.num_items and self.dim != other.dim:
            raise ValueError(f"dimension mismatch: {self.dim} vs {other.dim}")
        vectors = np.concatenate([self.vectors, other.vectors], axis=0) if other.num_items else self.vectors
        if self.offsets is None or other.offsets is None:
            return Embeddings(vectors=vectors)
        shifted = other.offsets[1:] + int(self.offsets[-1])
        return Embeddings(vectors=vectors, offsets=np.concatenate([self.offsets, shifted]))


class Encoder(ABC):
    """Maps content to vectors. The one contract every backend implements.

    Subclasses declare their capabilities as class attributes so a caller can
    refuse an impossible job *before* embedding a corpus, rather than discovering
    it in the numbers afterwards.
    """

    #: Whether this encoder can see pixels. ``False`` makes an image corpus a
    #: loud failure instead of a silent one: an image-only document's text is
    #: ``""``, which embeds perfectly happily and retrieves nothing.
    #:
    #: Instance-level with a class default, because for some backends it is a
    #: property of the configuration rather than of the code -- one vendor's
    #: endpoint takes images and another's does not, through the same client.
    supports_media: bool = False

    #: Whether output is one vector per token/patch rather than per item.
    #: Instance-level for the same reason: vLLM's pooling task decides it.
    is_multi_vector: bool = False

    @abstractmethod
    def encode(
        self,
        contents: Sequence[Content],
        *,
        role: EncodeRole,
        batch_size: int | None = None,
    ) -> Embeddings:
        """Embed *contents*, in order.

        Args:
            contents: Queries or documents as content parts.
            role: Which side these are. Required: an asymmetric embedder given
                the wrong side loses recall without raising.
            batch_size: Per-forward batch; the encoder's own default when
                ``None``.
        """

    # -- helpers for implementations --------------------------------------
    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        role: EncodeRole,
        batch_size: int | None = None,
    ) -> Embeddings:
        """Convenience for text-only callers and tests."""
        return self.encode([Content.from_text(text) for text in texts], role=role, batch_size=batch_size)

    def check_media(self, contents: Sequence[Content]) -> None:
        """Refuse media this encoder cannot see, naming the first offender.

        Called by text-only implementations before doing any work. The failure
        this prevents is quiet rather than loud: an image-only document's
        ``.text`` is ``""``, so a text encoder embeds it happily and produces an
        index in which those documents are simply never retrieved.
        """
        if self.supports_media:
            return
        offender = next(
            ((index, content) for index, content in enumerate(contents) if content.has_media),
            None,
        )
        if offender is None:
            return
        index, content = offender
        raise CapabilityError(
            f"{type(self).__name__} cannot encode media, but item {index} carries "
            f"{len(content.media)} media reference(s), e.g. {content.media[0].uri}. "
            "Use a vision encoder, or ingest a text rendering of this corpus."
        )


__all__ = ["EncodeRole", "Embeddings", "Encoder"]
