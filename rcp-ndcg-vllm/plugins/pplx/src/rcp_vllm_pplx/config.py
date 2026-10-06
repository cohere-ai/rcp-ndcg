"""The config handler the plugin binds for the ``PplxContextualModel`` architecture.

The checkpoint declares its bidirectional attention contract with ``is_causal: false`` in
``text_config``; vLLM reads that flag in two places whose configs differ:

- per full-attention layer, ``qwen3_next.py`` reads ``hf_text_config.is_causal`` and picks
  ``AttentionType.ENCODER_ONLY`` (bidirectional) when it is false;
- ``ModelConfig.attn_type`` reads ``hf_config.is_causal`` (the top-level config) for a
  pooling model whose default sequence pooling type is not ``CLS``, and drives chunked
  prefill and the KV-cache shape from it.

The checkpoint sets the text config but not the top-level flag, so the handler forces
``is_causal = False`` on both — the model's fixed contract, made explicit the way vLLM's
own ``ColQwen3_5Config`` (``models/config.py``) makes ColQwen3.5's explicit. A future
checkpoint revision that flips or drops the flag therefore changes nothing until the
plugin is re-pinned, instead of silently serving causal attention.
"""

from __future__ import annotations

__all__ = ["PplxContextualConfig"]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.model import ModelConfig

_IS_CAUSAL = False
"""The model's attention contract: bidirectional on the full-attention layers."""


class PplxContextualConfig:
    """Force the bidirectional attention contract of pplx-embed-v2-context-9b-preview.

    A ``VerifyAndUpdateConfig``-shaped handler bound in
    ``MODELS_CONFIG_MAP["PplxContextualModel"]`` by :func:`rcp_vllm_pplx.register`; vLLM
    calls ``verify_and_update_model_config`` while constructing the ``ModelConfig``.
    """

    @staticmethod
    def verify_and_update_model_config(model_config: ModelConfig) -> None:
        """Force ``is_causal = False`` on the top-level HF config and the text config."""
        model_config.hf_config.is_causal = _IS_CAUSAL
        model_config.hf_text_config.is_causal = _IS_CAUSAL

    @staticmethod
    def verify_and_update_config(vllm_config: VllmConfig) -> None:
        """The vllm-config-level hook of the same handler: nothing to do.

        The contract is set on the HF configs, which the ``ModelConfig``-level hook
        covers; vLLM calls this variant from ``VllmConfig`` construction and its
        ``VerifyAndUpdateConfig`` base defines it as a no-op, which this mirrors.
        """
        return
