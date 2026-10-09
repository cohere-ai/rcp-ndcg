"""The postprocessing home: what happens to a model's output after the model answered.

One home for every client- and judge-side postprocess of model output, so a consumer (the retrieval
index, the score pooling, the calibration projection) reads one mechanism, not three:

* :func:`l2_normalize` -- unit-norm rows (a zero row stays zero), float32-computed, dtype-preserving;
* the chunk-score aggregation -- a document's score is its best chunk's
  (:func:`max_pool_scores_by_document`), a rubric window's verdicts pool per criterion
  (:func:`max_pool_rubric_window_by_document`), and the chunk->document mapping is read by
  :func:`document_id_for_chunk` / :func:`document_ids_from_chunks`;
* the late-interaction skip ids (:func:`skip_keep_mask`, :func:`kept_vector_count`) -- the positions
  whose document vectors are dropped before MaxSim, and the declared count a reply is checked against
  when the served plugin applies the rule engine-side. The image-position rule lives with the caller
  that knows which documents carry media: a media request's positions are the engine's chat-template
  render (which the client cannot tokenise), so a media document's vectors are kept whole (exempt,
  never skipped) and the deviation is recorded -- unless the plugin applied the rule engine-side, in
  which case the reply is the kept set and the count check replaces the exemption.

The Matryoshka head (:func:`~rcp_ndcg.data.mrl.mrl_cut`, the learned projection) lives in
:mod:`rcp_ndcg.data.mrl`; it is a postprocess of the reply like the functions here, and it has one home.

Nothing here re-reads a request: the functions take what the reply or the client already holds, so a
change is decided once and recorded once (the role clients' :class:`~rcp_ndcg.data.text_budget.ProcessingRecord`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from rcp_ndcg.data.text_policy import CHUNK_ID_SEPARATOR
from rcp_ndcg.errors import DataError


def document_id_for_chunk(chunk_id: str, chunk_mapping: Mapping[str, str] | None) -> str:
    """The document a chunk belongs to (the id itself when there is no mapping).

    Raises:
        DataError: a non-empty mapping that does not cover *chunk_id*, or maps it to an empty id.
    """
    if not chunk_mapping:
        return chunk_id
    try:
        document_id = chunk_mapping[chunk_id]
    except KeyError as exc:
        raise DataError(f"chunk_mapping has no document ID for chunk {chunk_id!r}") from exc
    if not document_id:
        raise DataError(f"chunk_mapping maps chunk {chunk_id!r} to an empty document ID")
    return document_id


def document_ids_from_chunks(chunk_ids: Sequence[str], chunk_mapping: Mapping[str, str] | None) -> list[str]:
    """Unique document ids in first-chunk order."""
    return list(dict.fromkeys(document_id_for_chunk(chunk_id, chunk_mapping) for chunk_id in chunk_ids))


def max_pool_scores_by_document(
    chunk_scores: Mapping[str, float], chunk_mapping: Mapping[str, str] | None
) -> dict[str, float]:
    """A document's score is its best chunk's; documents in first-chunk order."""
    pooled: dict[str, float] = {}
    for chunk_id, score in chunk_scores.items():
        document_id = document_id_for_chunk(chunk_id, chunk_mapping)
        if document_id not in pooled or score > pooled[document_id]:
            pooled[document_id] = score
    return pooled


def max_pool_rubric_window_by_document(
    chunk_criteria: Mapping[str, Mapping[str, int]], chunk_mapping: Mapping[str, str] | None
) -> dict[str, dict[str, int]]:
    """Pool one rubric window's verdicts from chunks onto documents.

    A document passes criterion ``Ck`` in the window when any of its chunks in
    that window does. Pooling never crosses windows: each window stays one
    placement in the 2PL likelihood.

    Raises:
        DataError: a chunk without verdicts, or sibling chunks with different criteria.
    """
    pooled: dict[str, dict[str, int]] = {}
    for chunk_id, criteria in chunk_criteria.items():
        if not criteria:
            raise DataError(f"chunk {chunk_id!r} has no rubric criteria")
        document_id = document_id_for_chunk(chunk_id, chunk_mapping)
        current = pooled.get(document_id)
        if current is None:
            pooled[document_id] = dict(criteria)
            continue
        if current.keys() != criteria.keys():
            raise DataError(
                f"chunks for document {document_id!r} have inconsistent rubric criteria: "
                f"{sorted(current)} != {sorted(criteria)}"
            )
        for criterion, value in criteria.items():
            current[criterion] = max(current[criterion], value)
    return pooled


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Every row of ``vectors`` scaled to unit L2 norm (a zero row stays zero).

    Computed in float32 (a float16 sum of squares loses most of its three
    decimal digits), then cast back to the input dtype: float32 in, float32
    out; a float16 buffer stays float16 so a multi-vector index keeps its
    transfer precision end to end.
    """
    source = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(source, axis=1, keepdims=True)
    normalized = source / np.maximum(norms, 1e-12)
    if np.asarray(vectors).dtype == np.float16:
        return normalized.astype(np.float16)
    return normalized


def skip_keep_mask(token_ids: Sequence[int], skip_ids: Sequence[int]) -> list[int]:
    """The positions of ``token_ids`` a ``skip_ids`` keep-mask keeps, as indices into the reply's vectors.

    The late-interaction keep-rule (2, the topk hand-off): the document's vectors at the skip ids are
    dropped before MaxSim. When the served plugin applies the rule engine-side
    (``PoolingEndpoint.document_skip_engine_side``) the reply already carries only the kept vectors, so
    the caller does not slice: it counts them with :func:`kept_vector_count` and checks the reply. When
    it does not, the caller slices with this mask. A MEDIA document whose render the client cannot
    tokenise is the caller's per-item decision either way.

    Args:
        token_ids: The token id of every returned vector position (the client's tokenisation of what
            the engine read).
        skip_ids: The ids whose document vectors are dropped.

    Returns:
        The kept positions, ascending.

    Raises:
        DataError: ``token_ids`` is empty (there is nothing to keep a mask over).
    """
    if not token_ids:
        raise DataError(
            "skip_keep_mask needs the reply's token ids; none were given",
            hint="a document under document_skip_token_ids returned no prompt tokens: the engine's usage or the "
            "sent render must carry the positions the skip list is applied to (check the pooling route's "
            "prompt-token report and the declared tokenizer)",
        )
    skip = frozenset(skip_ids)
    return [position for position, token in enumerate(token_ids) if token not in skip]


def kept_vector_count(token_ids: Sequence[int], skip_ids: Sequence[int], *, media_tokens: int = 0) -> int:
    """The declared count of vectors a keep-rule leaves for one document -- what a reply is checked against.

    When the served plugin applies the rule engine-side (``PoolingEndpoint.document_skip_engine_side``)
    the wire carries only the kept vectors, and the client cannot count them by the positions it sent: it
    counts the declared kept positions instead. The count has the document's two shapes:

    * a TEXT document: ``token_ids`` are the ids of the sent render (the client's own tokenisation), and
      the count is the ids outside the skip list;
    * a MEDIA document: ``token_ids`` are the leading fixed head's ids (the template segments sent as a
      system message, ``media_head_as_system``) and ``media_tokens`` is the prepared media block's counted
      tokens (the vision wrapper plus the patch run, :func:`~rcp_ndcg.data.resolution.content_media_tokens`).
      The block's own positions are the processor's structural tokens, which the declared rule never names,
      so they are kept whole -- and a reply that disagrees with this count is refused by the caller, never
      absorbed.

    Args:
        token_ids: The token ids the rule is applied to (a text render, or a media document's head).
        skip_ids: The ids whose vectors the rule drops.
        media_tokens: The media block's counted tokens; 0 for a text document.

    Returns:
        The number of kept vectors (``>= 0``).

    Raises:
        DataError: both ``token_ids`` and ``media_tokens`` are empty (there is nothing to count), or
            ``media_tokens`` is negative.
    """
    if media_tokens < 0:
        raise DataError(
            f"kept_vector_count needs media_tokens of at least 0, got {media_tokens}",
            hint="media_tokens is the prepared media block's counted tokens (a count, never a delta); pass 0 "
            "for a text document",
        )
    if not token_ids and media_tokens == 0:
        raise DataError(
            "kept_vector_count has nothing to count: no token ids and no media tokens were given",
            hint="pass the sent render's ids (a text document), or a media document's head ids and its counted "
            "media block (media_tokens)",
        )
    head = len(skip_keep_mask(token_ids, skip_ids)) if token_ids else 0
    return head + media_tokens


__all__ = [
    "CHUNK_ID_SEPARATOR",
    "document_id_for_chunk",
    "document_ids_from_chunks",
    "kept_vector_count",
    "l2_normalize",
    "max_pool_rubric_window_by_document",
    "max_pool_scores_by_document",
    "skip_keep_mask",
]
