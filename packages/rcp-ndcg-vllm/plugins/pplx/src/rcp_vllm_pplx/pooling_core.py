"""The pooling contract of pplx-embed-v2-context-9b-preview, in pure torch.

Everything in this module runs without vLLM: the segmentation, the span pooling and the
int8 head are the parts of the served model that stock vLLM v0.31.0 cannot express, and
they are where fidelity to the reference implementation lives. The vLLM-side wiring lives
in :mod:`rcp_vllm_pplx.pooler` and :mod:`rcp_vllm_pplx.model`.

Semantics (r-pplx research lane, measured against the pinned revision
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
  id is the role disambiguation rule (an unstated one-chunk case in the r-pplx audit).
- An **empty chunk** (two consecutive markers, or a trailing marker) yields a zero vector
  for that chunk, which is the reference's own behaviour for a chunk with no tokens.
- A chunk's text containing the literal ``<|chunk_sep|>`` cannot be distinguished, on the
  id wire, from a boundary marker (the marker is an added token, not a special one, so it
  renders as ``BOUNDARY_TOKEN_ID`` in both paths): the plugin segments there — one extra
  row, the token excluded — where the reference's char-span pooling keeps it inside the
  chunk. Declared contract, not a fallback: the recipe's chunker must not emit the marker
  string as chunk content.
- The head is ``Linear(4096→2048, bias=False)`` kept in fp32 (the checkpoint keeps the
  projection out of the quantised path via ``_keep_in_fp32_modules``) followed by
  ``round(tanh(x)·127).clamp(-128, 127)`` — the model card's "unnormalized int8-quantized
  embeddings". The optional L2 normalisation is the request's ``use_activation`` and
  belongs to the pooler head, not here.
"""

from __future__ import annotations

__all__ = [
    "BOUNDARY_TOKEN_ID",
    "DOCUMENT_PREFIX_TOKEN_IDS",
    "QUERY_PREFIX_TOKEN_ID",
    "PplxInt8Projection",
    "pool_sequence",
]

import torch

#: The ``[Q] `` query prefix, one added token of the checkpoint's tokenizer, prepended by
#: the reference implementation as a single id before the query's own tokens.
QUERY_PREFIX_TOKEN_ID = 248077

#: The ``[D] `` document prefix, rendered by the reference tokenization (``truncation=False``,
#: ``split_special_tokens=True``) as the two literal tokens ``'[D', ']'`` — measured, not
#: assumed; this is why the client must send token ids rather than text.
DOCUMENT_PREFIX_TOKEN_IDS = (62724, 60)

#: The ``<|chunk_sep|>`` boundary marker, one added token in both the reference and a
#: server-side tokenization of the joined text.
BOUNDARY_TOKEN_ID = 248079


def _is_warmup_dummy(token_ids: list[int]) -> bool:
    """Whether a token-id row is vLLM's pooler warm-up input.

    The pooling runner's ``_dummy_pooler_run_task`` sizes the pooler with an all-zero
    token grid before the engine serves anything; the warm-up output is discarded. A real
    request can never be all zeros — its first id is always a role prefix — so this is a
    narrow, documented exception, not a fallback for real inputs.
    """
    return not any(token_ids)


def _document_spans(body: list[int]) -> list[tuple[int, int]]:
    """Half-open token spans of the chunks in the prefix-stripped document ids.

    The spans run between boundary markers; everything the tokenizer rendered from the
    ``[D] `` prefix was already stripped by the caller. Consecutive markers (an empty
    chunk) or a trailing marker give an empty span, which pools to the zero vector the
    reference produces for a chunk with no tokens.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for index, token_id in enumerate(body):
        if token_id == BOUNDARY_TOKEN_ID:
            spans.append((start, index))
            start = index + 1
    spans.append((start, len(body)))
    return spans


def pool_sequence(hidden_states: torch.Tensor, token_ids_cpu: torch.Tensor) -> torch.Tensor:
    """Pool one sequence's hidden states into one vector per chunk (or the query vector).

    Args:
        hidden_states: ``(seq_len, hidden_size)`` last states of one sequence, on the
            model's device, in the model's dtype.
        token_ids_cpu: ``(seq_len,)`` int64 CPU tensor of the sequence's prompt token ids
            (``PoolingMetadata.get_prompt_token_ids_cpu()``, the only tensor vLLM's v1
            pooling runner materialises; ``prompt_token_ids`` stays ``None`` there).

    Returns:
        ``(n_chunks, hidden_size)`` float32 — one span-mean row per document chunk, or one
        whole-sequence mean row for a query. An empty chunk's row is the zero vector.

    Raises:
        ValueError: The ids start with neither role prefix and are not the warm-up dummy
            (a real client sent an input that does not follow the contract).
    """
    if token_ids_cpu.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            f"prompt token ids ({token_ids_cpu.shape[0]}) do not align with the pooled "
            f"hidden states ({hidden_states.shape[0]}); refusing to pool a shifted sequence"
        )
    token_ids = [int(t) for t in token_ids_cpu.tolist()]
    hidden = hidden_states.to(torch.float32)

    if _is_warmup_dummy(token_ids):
        # The pooler warm-up's all-zero grid: pool as one span so the engine can size the
        # pooler; the result is thrown away by vLLM.
        return hidden.mean(dim=0, keepdim=True)

    if token_ids[0] == QUERY_PREFIX_TOKEN_ID:
        # Reference query pool: masked mean over the whole sequence, the [Q] prefix token
        # included. Serving requests carry no padding, so the mask is all ones.
        return hidden.mean(dim=0, keepdim=True)

    if tuple(token_ids[: len(DOCUMENT_PREFIX_TOKEN_IDS)]) == DOCUMENT_PREFIX_TOKEN_IDS:
        # Reference document pool: span mean per chunk, prefix and boundary markers
        # excluded from every span. Computed as a single index_add so the device work is
        # one scatter, with the segment map built from the CPU ids (no d2h sync).
        prefix_len = len(DOCUMENT_PREFIX_TOKEN_IDS)
        body = token_ids[prefix_len:]
        body_hidden = hidden[prefix_len:]
        spans = _document_spans(body)
        n_chunks = len(spans)
        # -1 marks the boundary markers themselves: they are excluded from every span
        # (the reference pools only the chunk tokens' char spans), unlike the prefix
        # tokens they sit behind.
        segment_map = torch.full((len(body),), -1, dtype=torch.int64)
        for chunk_index, (start, end) in enumerate(spans):
            if end > start:
                segment_map[start:end] = chunk_index
        pooled_positions = segment_map >= 0
        pooled_index = segment_map[pooled_positions]
        pooled_hidden = body_hidden[pooled_positions.to(hidden.device)]
        counts = torch.bincount(pooled_index, minlength=n_chunks).clamp(min=1)
        sums = torch.zeros(n_chunks, hidden.shape[-1], dtype=hidden.dtype, device=hidden.device)
        sums = sums.index_add(0, pooled_index.to(hidden.device), pooled_hidden)
        # Empty chunks have count 0 -> clamped to 1 with a zero sum -> the zero vector the
        # reference's own empty-chunk branch (span (0, 0)) produces.
        return sums / counts.to(hidden.device).unsqueeze(1)

    raise ValueError(
        "an input's first token id must be the query prefix "
        f"{QUERY_PREFIX_TOKEN_ID} ([Q] ) or the document prefix "
        f"{list(DOCUMENT_PREFIX_TOKEN_IDS)} ('[D', ']'), got {token_ids[:4]!r}; the "
        "client must send token ids with the role prefix prepended, per the plugin's "
        "README contract"
    )


class PplxInt8Projection(torch.nn.Module):
    """The checkpoint's embedding head: fp32 linear, then the int8 tanh quantiser.

    The reference implementation keeps ``contextual_projection`` (a bias-free
    ``Linear(4096, 2048)``) in fp32 via its ``_keep_in_fp32_modules`` and then applies
    ``round(tanh(x)·127).clamp(-128, 127)`` — the card's unnormalized int8-quantized
    embeddings. The module is created in the model's ``head_dtype`` (fp32 for pooling
    models by default), which keeps the projection in fp32 whatever the serving dtype is,
    as the checkpoint does. The ``round`` is ``torch.round`` (half-to-even); the clamp is
    kept from the reference although ``tanh(x)·127`` already lies in [-127, 127].
    """

    def __init__(self, hidden_size: int, embedding_dim: int, *, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, embedding_dim, bias=False, dtype=dtype)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """``(…, hidden_size)`` to ``(…, embedding_dim)`` int8-valued floats, in fp32."""
        projected = self.linear(pooled.to(self.linear.weight.dtype))
        return torch.round(torch.tanh(projected) * 127.0).clamp(-128.0, 127.0)
