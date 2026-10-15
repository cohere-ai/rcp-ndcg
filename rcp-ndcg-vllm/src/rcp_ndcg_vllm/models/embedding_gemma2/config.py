# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``EmbeddingGemma2Model`` config hook, folded from vLLM commit 8cbd5d03 (the fold's source).

Folded from vLLM commit ``8cbd5d03006c33185f402249ff2b448efd594986``
(``vllm/model_executor/models/config.py``, ``class EmbeddingGemma2ModelConfig``, its ``MODELS_CONFIG_MAP``
entry): the stock v0.31.0 image the recipe serves carries no
``EmbeddingGemma2Model`` entry, so ``register_embedding_gemma2`` adds this one when the running vLLM
lacks it. The body is upstream's, unchanged: the Gemma4 hook first (heterogeneous head dimensions --
the full-attention layers' head is larger than the sliding ones), then TRITON_ATTN as the declared
default backend when none was requested (upstream's "validated backend for heterogeneous head
dimensions"), then the 8192 context cap for a checkpoint that declares none. This module imports vLLM
by design and is only imported by :func:`rcp_ndcg_vllm.models.register` inside the engine process
(``LAZY_MODEL_MODULES``); importing the plugin never touches it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.model_executor.models.config import Gemma4Config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

__all__ = ["EmbeddingGemma2ModelConfig"]

logger = init_logger(__name__)


class EmbeddingGemma2ModelConfig(Gemma4Config):
    """Upstream's config hook for the folded ``EmbeddingGemma2Model`` architecture."""

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        Gemma4Config.verify_and_update_config(vllm_config)
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        attn_config = vllm_config.attention_config
        if attn_config.backend is None:
            attn_config.backend = AttentionBackendEnum.TRITON_ATTN
            logger.info(
                "EmbeddingGemma2: attention backend not specified; defaulting to "
                "TRITON_ATTN (validated backend for heterogeneous head dimensions)."
            )
        else:
            logger.info("EmbeddingGemma2: using attention backend %s", attn_config.backend)

        model_config = vllm_config.model_config
        if model_config.max_model_len > 8192 and getattr(model_config, "original_max_model_len", None) in (None, -1):
            model_config.max_model_len = 8192
