"""The contextual model's id-level contract and per-chunk segmentation, in pure Python.

The plugin's pooler (:func:`rcp_ndcg_vllm.models.pplx.pooling_core.pool_sequence`) answers a
DOCUMENT row with one vector per chunk -- the segments between the ``<|chunk_sep|>`` boundary
markers of the ``[D] ``-prefixed render -- and a QUERY row with one vector (the whole-sequence
mean).  The rule lives here, free of vLLM and torch, because two callers read it: the pooler
itself, and the observation corpus's acceptance check (``rcp_ndcg_test.observe.corpus``), which
expects the declared per-chunk count of every vector reply a per-chunk recipe records
(:func:`chunk_count`).  One home: a change to the segmentation moves both.

Semantics (measured against the pinned revision
``b667039ee8b438a6350fbc91bbcecd86f9d363ba`` of the checkpoint and its remote code):

- A **query** is ``[248077] + tokenized text``; the pool is the mean over the whole
  sequence, the ``[Q] `` prefix token included (the reference's masked-mean pool; serving
  has no padding, so the mask is all ones).
- A **document** is one request holding all of its chunks: ``tokenize("[D] " + chunk0 +
  "<|chunk_sep|>" + chunk1 + ..., split_special_tokens=True)``. The ``[D] `` prefix
  renders as the two literal tokens ``[62724, 60]`` and each ``<|chunk_sep|>`` as the one
  added id ``248079``. The pool is one **span mean per chunk**, the prefix tokens and the
  boundary markers excluded from every span. A one-chunk document has no marker and is a
  single span; a query is told apart from it by the leading ``248077`` — the leading token
  id is the role disambiguation rule (the one-chunk case).
- An **empty chunk** (two consecutive markers, or a trailing marker) yields a zero vector
  for that chunk, which is the reference's own behaviour for a chunk with no tokens.
- A chunk's text containing the literal ``<|chunk_sep|>`` cannot be distinguished, on the
  id wire, from a boundary marker (the marker is an added token, not a special one, so it
  renders as ``BOUNDARY_TOKEN_ID`` in both paths): the plugin segments there — one extra
  row, the token excluded — where the reference's char-span pooling keeps it inside the
  chunk. Declared contract, not a fallback: the recipe's chunker must not emit the marker
  string as chunk content.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "BOUNDARY_TOKEN_ID",
    "DOCUMENT_PREFIX_TOKEN_IDS",
    "PREFIX_SPACE_TOKEN_ID",
    "QUERY_PREFIX_TOKEN_ID",
    "chunk_count",
    "document_body",
    "document_spans",
    "is_engine_dummy",
]

#: The ``[Q] `` query prefix, one added token of the checkpoint's tokenizer, prepended by
#: the reference implementation as a single id before the query's own tokens.
QUERY_PREFIX_TOKEN_ID = 248077

#: The ``[D] `` document prefix, rendered by the reference tokenization (``truncation=False``,
#: ``split_special_tokens=True``) as the two literal tokens ``'[D', ']'`` — measured, not
#: assumed; this is why the client must send token ids rather than text.
DOCUMENT_PREFIX_TOKEN_IDS = (62724, 60)

#: The ``[D] `` prefix's trailing space as its own token, measured at the pinned tokenizer: the
#: render is ``"[D] " + content`` and the space stands alone exactly when the content's first token
#: is an added token that cannot carry it (or when the content is empty); a content-initial space
#: merges into the next token (ids 256/262/...), so a leading 220 in the body is always the
#: prefix's. The reference's char spans start at char 4 and never pool it.
PREFIX_SPACE_TOKEN_ID = 220

#: The ``<|chunk_sep|>`` boundary marker, one added token in both the reference and a
#: server-side tokenization of the joined text.
BOUNDARY_TOKEN_ID = 248079


def is_engine_dummy(token_ids: list[int]) -> bool:
    """Whether a token-id row is one of vLLM's engine-side dummy inputs (no role prefix, leading id 0).

    At the pinned engine (v0.31.0) two dummy paths reach the pooler, and both open with token id 0:

    - the kernel warmup builds its dummy requests' ids as ``list(range(prompt_len))`` with
      ``prompt_len = decode_query_len + 1`` (``vllm/v1/worker/gpu/warmup.py:256-257``, sent as a real
      scheduler prefill at ``:330-345``); a pooling model has ``decode_query_len == 1``, so the row is the
      measured ``[0, 1]``;
    - the pooler sizing grid is all zeros (``vllm/v1/worker/gpu/pool/pooling_runner.py:178-183``; the v1
      runner's ``_dummy_pooler_run_task`` builds the same zero grid at
      ``vllm/v1/worker/gpu_model_runner.py:6339-6352``).

    A real request always opens with a role prefix (``[Q] `` = 248077 or ``[D] `` = ``(62724, 60)``), so a
    leading 0 can only be an engine dummy: the warmup output is discarded, and the row pools as one span so
    the engine can size the pooler. An empty row is the zero-length case of the same grid.
    """
    return not token_ids or token_ids[0] == 0


def document_body(token_ids: Sequence[int]) -> list[int]:
    """The body a DOCUMENT row's chunk spans are read from: the ``[D] `` prefix and its own trailing space removed.

    Inputs: the row's full token ids, the role prefix included.  Output: the ids after the two-token
    ``[D] `` prefix and, when the prefix's trailing space stands as its own token
    (:data:`PREFIX_SPACE_TOKEN_ID`), after that too -- the positions the reference's char spans start after.
    Raises:
        ValueError: The ids do not open with the document prefix (they are not a document render).
    """
    ids = list(token_ids)
    if tuple(ids[: len(DOCUMENT_PREFIX_TOKEN_IDS)]) != DOCUMENT_PREFIX_TOKEN_IDS:
        raise ValueError(
            "an input's first token id must be the query prefix "
            f"{QUERY_PREFIX_TOKEN_ID} ([Q] ) or the document prefix "
            f"{list(DOCUMENT_PREFIX_TOKEN_IDS)} ('[D', ']'), got {ids[:4]!r}; the "
            "client must send token ids with the role prefix prepended, per the plugin's "
            "README contract"
        )
    body = ids[len(DOCUMENT_PREFIX_TOKEN_IDS) :]
    return body[1:] if body[:1] == [PREFIX_SPACE_TOKEN_ID] else body


def document_spans(body: Sequence[int]) -> list[tuple[int, int]]:
    """Half-open token spans of the chunks in the prefix-stripped document ids.

    The spans run between boundary markers; everything the tokenizer rendered from the
    ``[D] `` prefix was already stripped by :func:`document_body`. Consecutive markers (an
    empty chunk) or a trailing marker give an empty span, which pools to the zero vector
    the reference produces for a chunk with no tokens.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for index, token_id in enumerate(body):
        if token_id == BOUNDARY_TOKEN_ID:
            spans.append((start, index))
            start = index + 1
    spans.append((start, len(body)))
    return spans


def chunk_count(token_ids: Sequence[int]) -> int:
    """The number of vectors the pooler answers one prompt row with: one per document chunk, one for a query.

    The declared per-chunk count the observation corpus's acceptance check expects of every vector reply a
    per-chunk recipe records: a document row's chunks are the segments between its boundary markers (the
    same :func:`document_spans` the pooler pools), while a query row and an engine dummy pool to one vector.

    Inputs: one prompt row's token ids, the role prefix included, as the engine rendered them.
    Output: the vector count, at least 1.
    Raises:
        ValueError: The ids open with neither role prefix and are not an engine dummy (the same refusal
            :func:`~rcp_ndcg_vllm.models.pplx.pooling_core.pool_sequence` makes).
    """
    ids = list(token_ids)
    if is_engine_dummy(ids) or ids[0] == QUERY_PREFIX_TOKEN_ID:
        return 1
    return len(document_spans(document_body(ids)))
