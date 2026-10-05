"""One encoder contract: content in, vectors out, whatever runs it.

An :class:`Encoder` takes :class:`~rcp_ndcg_core.content.Content` and returns
:class:`~rcp_ndcg.inference.types.Embeddings`. Three things follow from that shape:

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

The role types (:class:`~rcp_ndcg.inference.types.EncodeRole`,
:class:`~rcp_ndcg.inference.types.Embeddings`, :func:`~rcp_ndcg.inference.types.l2_normalize`) live in
:mod:`rcp_ndcg.inference.types` now and are re-exported here, so every import from this module keeps working.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rcp_ndcg_core.content import Content

from rcp_ndcg.errors import CapabilityError
from rcp_ndcg.inference.types import Embeddings, EncodeRole, l2_normalize  # noqa: F401  # re-exported


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


__all__ = ["Embeddings", "EncodeRole", "Encoder", "l2_normalize"]
