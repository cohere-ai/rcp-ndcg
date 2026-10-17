"""The tiny-config equivalence of the pooling chain.

The plugin's forward-affecting surface is the token-embed pooling chain
(projection -> MRL slice -> L2 normalise) inherited from ColQwen3_5Model's
pooler wiring, and the weight mapping proven in test_weight_mapping.py.  This
suite proves the chain numerically on a tiny randomly initialised config of
the same architecture (Linear hidden->dim, MRL slice, L2 normalise), with
weights shared between the reference chain — the model card's
``modeling_topk_embed.py:76-82`` (``self.head(hidden).float()``, MRL
``output_dim`` slice, ``F.normalize``) — and the plugin's pure-torch chain
(``rcp_ndcg_vllm.models.topk.pooling.token_embed_pool``, which mirrors vLLM's
``TokenEmbeddingPoolerHead``).

What this suite cannot prove on a CPU box without vLLM: the backbone (stock
vLLM Qwen3.5 code, unchanged by the plugin) and the real weights — those are
the GPU wave's served-vs-reference equivalence (README, "What the GPU wave
must check").
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from rcp_ndcg_vllm.models.topk.pooling import token_embed_pool  # noqa: I001 - conftest prepends src/

# Tiny "same architecture" dimensions: hidden_size -> dim projection with an
# MRL prefix (the checkpoint's are 2048 -> 2048 with MRL prefixes 64..2048).
HIDDEN = 16
DIM = 8
N_TOKENS = 7
TOLERANCE = 1e-6  # float32 tolerance for the identical-math comparison


def reference_chain(
    hidden_states: torch.Tensor,
    head_weight: torch.Tensor,
    output_dim: int | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """The model card's vector path (modeling_topk_embed.py:76-82): project,
    cast the result to float32, MRL-slice, L2-normalise.

    The reference computes the head matmul in the model's dtype and casts the
    RESULT to float32; with float32 inputs this is the same math as the
    served fp32-head chain, and with bf16 operands it is the head-rounding
    regime the GPU wave measures on real weights.
    """
    vectors = (hidden_states @ head_weight.T).float()
    if output_dim is not None:
        vectors = vectors[..., :output_dim]
    if normalize:
        vectors = F.normalize(vectors, p=2, dim=-1)
    return vectors


def test_chain_matches_the_reference_exactly_fp32() -> None:
    """Shared float32 weights, fp32 head: the served chain equals the
    reference chain within float32 tolerance."""
    generator = torch.Generator().manual_seed(42)
    hidden = torch.randn(N_TOKENS, HIDDEN, generator=generator)
    head_weight = torch.randn(DIM, HIDDEN, generator=generator)

    served = token_embed_pool(hidden, head_weight)
    reference = reference_chain(hidden, head_weight)

    assert torch.allclose(served, reference, rtol=0, atol=TOLERANCE)
    # Per-token rows are L2-normalised (PoolerNormalize) and one vector per
    # input token, in prompt order (token_embed pooling type ALL).
    assert served.shape == (N_TOKENS, DIM)
    assert torch.allclose(served.norm(dim=-1), torch.ones(N_TOKENS), rtol=1e-6, atol=1e-6)


def test_chain_matches_the_reference_with_mrl_slice() -> None:
    """MRL: slicing after projection, before normalisation — the reference's
    order (modeling_topk_embed.py:78-82) — is what the served pooler does too
    (heads.py:94-96), so a prefix slice of dims equals the reference."""
    generator = torch.Generator().manual_seed(7)
    hidden = torch.randn(N_TOKENS, HIDDEN, generator=generator)
    head_weight = torch.randn(DIM, HIDDEN, generator=generator)
    for output_dim in (DIM, 4):
        served = token_embed_pool(hidden, head_weight, output_dim=output_dim)
        reference = reference_chain(hidden, head_weight, output_dim=output_dim)
        assert served.shape == (N_TOKENS, output_dim)
        assert torch.allclose(served, reference, rtol=0, atol=TOLERANCE)


def test_chain_matches_the_reference_without_normalization() -> None:
    """normalize=false (the checkpoint's knob) skips only the normalisation."""
    generator = torch.Generator().manual_seed(3)
    hidden = torch.randn(N_TOKENS, HIDDEN, generator=generator)
    head_weight = torch.randn(DIM, HIDDEN, generator=generator)
    served = token_embed_pool(hidden, head_weight, normalize=False)
    reference = F.linear(hidden, head_weight)
    assert torch.allclose(served, reference, rtol=0, atol=TOLERANCE)


def test_chain_rejects_mrl_wider_than_the_projection() -> None:
    """output_dim above the projection size is refused (the reference's
    TopkEmbedConfig.__post_init__ enforces the same bound)."""
    hidden = torch.randn(2, HIDDEN)
    head_weight = torch.randn(DIM, HIDDEN)
    with pytest.raises(ValueError, match="output_dim"):
        token_embed_pool(hidden, head_weight, output_dim=DIM + 1)


def test_bf16_head_rounding_is_bounded() -> None:
    """The head rounding on CPU: the reference runs the head matmul on bf16 operands and
    casts the result; the served pooler casts the data to a float32 head
    first.  The two agree to bf16 rounding of the operands — measured here on
    a tiny config to bound the delta the GPU wave should see from this source
    (per-token cosines stay within 1e-3 of 1 for a 2048-unit head; the tiny
    config below is noisier, so the bound is stated at 1e-2)."""
    generator = torch.Generator().manual_seed(11)
    hidden = torch.randn(N_TOKENS, HIDDEN, generator=generator)
    head_weight = torch.randn(DIM, HIDDEN, generator=generator)

    served = token_embed_pool(hidden, head_weight.to(torch.float32))
    reference = reference_chain(hidden.to(torch.bfloat16), head_weight.to(torch.bfloat16))

    cosine = F.cosine_similarity(served, reference, dim=-1)
    assert (cosine >= 1.0 - 1e-2).all(), cosine
