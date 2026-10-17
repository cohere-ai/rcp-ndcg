"""The plugin's one token-embed head construction: vLLM's ``TokenPooler`` around a pooling method.

Both plugin pooling methods assemble the same head -- vLLM's ``TokenEmbeddingPoolerHead`` over the
checkpoint's trained projection, with ``PoolerNormalize()`` as the request's activation -- so the assembly
lives here once, mirroring vLLM's ``pooler_for_token_embed`` (``tokwise/poolers.py``): ``head_dtype`` (fp32
for pooling models) casts the pooled vectors, the projector applies the checkpoint's head, and the activation
L2-normalises unless the request sends ``use_activation: false``.

Callers: :func:`~rcp_ndcg_vllm.models.pplx.pooler.build_pooler` (the contextual model's per-chunk spans) and
:func:`~rcp_ndcg_vllm.models.keep_pooler.build_keep_pooler` (both late-interaction families' declared
keep-rule).
"""

from __future__ import annotations

__all__ = ["token_embed_pooler"]

import torch
from vllm.config import ModelConfig
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.tokwise.heads import TokenEmbeddingPoolerHead
from vllm.model_executor.layers.pooler.tokwise.methods import TokenPoolingMethod
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler


def token_embed_pooler(
    model_config: ModelConfig, *, projector: torch.nn.Module, pooling: TokenPoolingMethod
) -> TokenPooler:
    """vLLM's token-embed head around ``pooling``.

    Args:
        model_config: vLLM's ``ModelConfig``; only ``head_dtype`` is read.

        projector: The model's trained projection, registered on the model so the checkpoint's head loads
            into it.

        pooling: The plugin's pooling method (``PplxChunkPool`` for the contextual model, ``KeepPool`` for
            the late-interaction families).
    """
    head = TokenEmbeddingPoolerHead(
        head_dtype=model_config.head_dtype,
        projector=projector,
        activation=PoolerNormalize(),
    )
    return TokenPooler(pooling=pooling, head=head)
