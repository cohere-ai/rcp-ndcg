"""The token-embed pooling chain, as pure torch.

``token_embed_pool`` is the numeric contract the plugin's served model
implements — projection, Matryoshka slice, L2 normalisation — restated without
vLLM so the CPU equivalence test can execute it.  It mirrors the served chain
exactly:

- ``TokenEmbeddingPoolerHead.forward_chunk`` (vLLM v0.31.0,
  vllm/model_executor/layers/pooler/tokwise/heads.py:74-100): cast the pooled
  data to ``head_dtype`` (float32 for pooling models,
  vllm/config/model.py:_get_head_dtype), apply the projector
  (``custom_text_proj``, passed by ColQwen3_5Model.__init__, colqwen3_5.py:
  177-189), slice ``dimensions``, then apply ``PoolerNormalize``
  (F.normalize(p=2, dim=-1), activations.py:108-110) when
  ``use_activation`` — which defaults to True for ``token_embed``
  (vllm/pooling_params.py:_set_default_parameters).
- ``ColQwen3_5Model`` returns the backbone's per-token hidden states
  unchanged (token pooling type ALL), so the projector input is one hidden
  state per prompt token, in prompt order.
"""

from __future__ import annotations

import torch

__all__ = ["token_embed_pool"]


def token_embed_pool(
    hidden_states: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor | None = None,
    output_dim: int | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Project per-token hidden states to embedding space, slice, L2-normalise.

    Args:
        hidden_states: ``[n_tokens, hidden_size]`` backbone states (any float
            dtype; cast to ``proj_weight``'s dtype first, as the pooler casts
            to ``head_dtype``).
        proj_weight: ``[output_size, hidden_size]`` projection matrix (the
            checkpoint's ``head.weight``).
        proj_bias: optional ``[output_size]`` bias; the checkpoint has none
            and the served class zero-initialises its bias, so ``None`` and a
            zero bias are equivalent (see ``model.TopkEmbedModel``).
        output_dim: the MRL prefix length (the checkpoint's ``output_dim``
            config knob, <= proj_weight.shape[0]); ``None`` keeps every
            dimension.  Sliced after projection, before normalisation.
        normalize: L2-normalise the output rows (the checkpoint's
            ``normalize: true``).

    Returns:
        ``[n_tokens, output_dim]`` float32 per-token embeddings in the same
        order as the input tokens.
    """
    pooled = hidden_states.to(proj_weight.dtype)
    embeddings = torch.nn.functional.linear(pooled, proj_weight, proj_bias)
    if output_dim is not None:
        if not 0 < output_dim <= embeddings.shape[-1]:
            raise ValueError(f"output_dim must be in 1..{embeddings.shape[-1]}, got {output_dim}")
        embeddings = embeddings[..., :output_dim]
    if normalize:
        # PoolerNormalize (F.normalize, p=2, dim=-1), applied unconditionally
        # for token_embed with the default use_activation=True.
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return embeddings
