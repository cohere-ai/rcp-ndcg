"""The plugin's vLLM pooler: per-chunk span pooling wired into vLLM's token pooling.

vLLM's own ``TokenPooler`` runs two stages: a *pooling method* over the batch's hidden
states (``TokenPoolingFn``), then a *head* (projector, Matryoshka cut, optional
activation). The plugin replaces the first stage with ``PplxChunkPool`` — the per-chunk
segmentation of :mod:`rcp_ndcg_vllm.models.pplx.pooling_core` — and builds the second from vLLM's own
``TokenEmbeddingPoolerHead`` so the request-level knobs (``dimensions``, the L2
normalisation behind ``use_activation``) behave exactly as they do for the in-tree token
poolers, with the plugin's int8 projection as the head's projector.

Modeled on vLLM's ``StepPool`` (vLLM ``tokwise/methods.py``, consumed by
``models/jina.py``): a ``TokenPoolingMethod`` that subclasses ``AllPool`` — inheriting the
chunked-prefill caching — and additionally asks for the prompt token ids, which it reads
from their CPU copy (the v1 pooling runner materialises only that tensor;
``prompt_token_ids`` stays ``None`` there, so ``get_prompt_token_ids()`` would raise).
"""

from __future__ import annotations

__all__ = ["PplxChunkPool", "build_pooler"]

from collections.abc import Set

import torch
from vllm.config import ModelConfig
from vllm.model_executor.layers.pooler import PoolingParamsUpdate
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.tokwise.heads import TokenEmbeddingPoolerHead
from vllm.model_executor.layers.pooler.tokwise.methods import (
    AllPool,
    TokenPoolingMethodOutputItem,
)
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from rcp_ndcg_vllm.models.pplx.pooling_core import pool_sequence


class PplxChunkPool(AllPool):
    """Per-chunk span pooling over the flattened batch, segmented on the boundary id.

    ``AllPool`` (vLLM ``tokwise/methods.py``) does the batch work — splitting the
    flattened hidden states per request and caching chunked-prefill pieces until a
    sequence is complete; this subclass adds the role-aware span pooling that stock
    vLLM cannot express.
    """

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"token_embed"}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        # The segmentation reads the prompt token ids; this flag is what makes vLLM's
        # pooling runner build their CPU copy (PoolingParamsUpdate.requires_token_ids).
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> list[TokenPoolingMethodOutputItem]:
        pooled_data = super().forward(hidden_states, pooling_metadata)
        token_ids_cpu = pooling_metadata.get_prompt_token_ids_cpu()
        pooled: list[TokenPoolingMethodOutputItem] = []
        for data, token_ids in zip(pooled_data, token_ids_cpu, strict=True):
            # Unfinished chunked prefill: AllPool already returned None for the sequence.
            pooled.append(None if data is None else pool_sequence(data, token_ids))
        return pooled


def build_pooler(model_config: ModelConfig, *, projector: torch.nn.Module) -> TokenPooler:
    """The plugin's ``TokenPooler``: chunk pool + int8 head + the request's activation.

    Mirrors vLLM's ``pooler_for_token_embed`` (``tokwise/poolers.py``) with the plugin's
    pooling method and the checkpoint's own projection in place of the
    sentence-transformers projector fallback: ``head_dtype`` (fp32 for pooling models)
    casts the span means, the projector applies the int8 tanh head, and
    ``activation=PoolerNormalize()`` L2-normalises unless the request sends
    ``use_activation: false`` — the reference's ``normalize_embeddings`` flag, on by
    default as the card's examples use it.

    Args:
        model_config: vLLM's ``ModelConfig``; only ``head_dtype`` and ``pooler_config``
            are read.

        projector: The model's ``PplxInt8Projection``, registered on the model so the
            checkpoint's ``contextual_projection.weight`` loads into it.
    """
    head = TokenEmbeddingPoolerHead(
        head_dtype=model_config.head_dtype,
        projector=projector,
        activation=PoolerNormalize(),
    )
    return TokenPooler(pooling=PplxChunkPool(), head=head)
