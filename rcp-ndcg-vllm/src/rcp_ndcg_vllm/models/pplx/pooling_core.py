"""The pooling contract of pplx-embed-v2-context-9b-preview, in pure torch.

Everything in this module runs without vLLM: the span pooling and the int8 head are the
parts of the served model that stock vLLM v0.31.0 cannot express, and they are where
fidelity to the reference implementation lives. The id-level contract and the per-chunk
segmentation they read live in :mod:`rcp_ndcg_vllm.models.pplx.chunk_rule` (pure Python,
so the observation corpus's acceptance check shares the one home); the vLLM-side wiring
lives in :mod:`rcp_ndcg_vllm.models.pplx.pooler` and
:mod:`rcp_ndcg_vllm.models.pplx.model`.

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
    "PREFIX_SPACE_TOKEN_ID",
    "QUERY_PREFIX_TOKEN_ID",
    "PplxInt8Projection",
    "pool_sequence",
]

import torch

from rcp_ndcg_vllm.models.pplx.chunk_rule import (
    BOUNDARY_TOKEN_ID,
    DOCUMENT_PREFIX_TOKEN_IDS,
    PREFIX_SPACE_TOKEN_ID,
    QUERY_PREFIX_TOKEN_ID,
    document_body,
    document_spans,
    is_engine_dummy,
)


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
        ValueError: The ids start with neither role prefix and are not an engine dummy (a real client sent an
            input that does not follow the contract).
    """
    if token_ids_cpu.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            f"prompt token ids ({token_ids_cpu.shape[0]}) do not align with the pooled "
            f"hidden states ({hidden_states.shape[0]}); refusing to pool a shifted sequence"
        )
    token_ids = [int(t) for t in token_ids_cpu.tolist()]
    hidden = hidden_states.to(torch.float32)

    if is_engine_dummy(token_ids):
        # The engine's dummy inputs: the pooler sizing grid (all zeros) and the kernel warmup
        # (list(range(prompt_len)) = [0, 1] for a pooling model). Pool as one span so the engine can size
        # the pooler; the result is thrown away by vLLM.
        return hidden.mean(dim=0, keepdim=True)

    if token_ids[0] == QUERY_PREFIX_TOKEN_ID:
        # Reference query pool: masked mean over the whole sequence, the [Q] prefix token
        # included. Serving requests carry no padding, so the mask is all ones.
        return hidden.mean(dim=0, keepdim=True)

    if tuple(token_ids[: len(DOCUMENT_PREFIX_TOKEN_IDS)]) == DOCUMENT_PREFIX_TOKEN_IDS:
        # Reference document pool: span mean per chunk, prefix and boundary markers
        # excluded from every span. Computed as a single index_add so the device work is
        # one scatter, with the segment map built from the CPU ids (no d2h sync). The
        # prefix's trailing space (chunk_rule.PREFIX_SPACE_TOKEN_ID), when it stands as its
        # own token, belongs to no chunk: the reference's char spans start after it, and
        # without this the empty document pools the space token -- a nonzero vector where
        # the reference pools the zero vector -- and every document starting with an added
        # token gains one token in its first chunk's mean.
        body = document_body(token_ids)
        body_hidden = hidden[len(token_ids) - len(body) :]
        spans = document_spans(body)
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
