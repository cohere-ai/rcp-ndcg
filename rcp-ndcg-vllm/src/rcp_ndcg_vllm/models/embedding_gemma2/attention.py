"""The plugin's ``TORCH_SDPA`` attention backend: encoder-only attention over torch SDPA.

E2 round 10. vLLM v0.31.0 (the stock image the recipe serves) refuses every backend the
embeddinggemma-2 recipe tried: TRITON_ATTN cannot
compile its prefill kernel for the full layers' head_dim 512, FA4 caps head_dim at 256, and
``TORCH_SDPA`` is not a decoder backend on the image -- its enum value is the empty string ("this tag
is only used for ViT"), so the manual-selection path raises ``ValueError: Backend TORCH_SDPA must be
registered before use. Use register_backend(Backend.TORCH_SDPA, 'your.module.YourClass')``
(``vllm/v1/attention/backends/registry.py``). FLEX_ATTENTION is the only image backend that clears
the heterogeneous head dimensions, and it dies in its own ``torch.compile``/CUDAGraph interaction
(``torch._dynamo.exc.InternalTorchDynamoError: RuntimeError: Error: accessing tensor output of
CUDAGraphs that has been overwritten by a subsequent run`` at
``torch/nn/attention/flex_attention.py:2369``): its ``create_block_mask`` is compiled with
``mode="reduce-overhead"``, so the block mask handed to the compiled ``flex_attention`` is a
CUDA-graph output a later run of the block-mask graph overwrites -- and ``--enforce-eager`` does not
stop it, because the flex backend compiles independently of the engine's cudagraph mode.

The engine's own error names the way out: register a class under ``Backend.TORCH_SDPA``. This module
is that class, an encoder-only backend over
:func:`torch.nn.functional.scaled_dot_product_attention` -- no compile, no block mask, no KV cache
(the model passes ``AttentionType.ENCODER_ONLY`` for every layer and vLLM builds no cache for it),
and no head-dimension cap (SDPA has no 256 limit; head_dim 512 runs on its math kernel). The
attention is non-causal within each request, and the sliding layers get the checkpoint's own window:
the vendored model passes ``per_layer_sliding_window = config.sliding_window + 1`` (513) because
vLLM's encoder-only window masks ``|i - j| <= W - 1``, which is exactly the HF mask
(``create_bidirectional_sliding_window_mask`` uses ``config.sliding_window`` = 512,
``|i - j| <= 512``); ``self.sliding_window`` here is vLLM's W, so the mask is ``W - 1``.

The backend is registered by the ``torch-sdpa-attention`` patch, which the embeddinggemma-2 recipe
opts into (``serve.patches``); the engine starts with ``--attention-backend TORCH_SDPA``. The
registration path is the one home: :func:`register_torch_sdpa_backend` builds it from this module's
``__name__`` and the class's name, and vLLM resolves the class lazily when a layer asks for the
backend (``AttentionBackendEnum.get_class`` -> ``resolve_obj_by_qualname``). Every other recipe's
engine runs without the opt-in, so the enum override never lands there.

Removal condition: delete this module, the patch and the recipe's ``--attention-backend TORCH_SDPA``
pin together with the fold, when ``engine.image`` moves to a vLLM image whose registry carries the
model, whose own backends clear head_dim 512, and whose transformers ships ``embedding_gemma2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn.functional as F
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

__all__ = [
    "TorchSdpaAttentionBackend",
    "TorchSdpaAttentionImpl",
    "TorchSdpaAttentionMetadata",
    "TorchSdpaAttentionMetadataBuilder",
    "register_torch_sdpa_backend",
]


class TorchSdpaAttentionBackend(AttentionBackend):
    """Encoder-only attention through ``torch.nn.functional.scaled_dot_product_attention``.

    The class clears exactly what the embeddinggemma-2 engine validates: no head-size cap (the
    model's sliding layers are 256 and its full layers 512), bf16/float32 activations, encoder-only
    attention, a sliding window and non-causal attention. It runs no kernel of its own and compiles
    nothing, so it has no CUDA-graph support to declare (``AttentionCGSupport.NEVER`` is the
    :class:`~vllm.v1.attention.backend.AttentionMetadataBuilder` default).
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]

    #: The backend's forward reads no KV cache (encoder-only layers keep none), so vLLM calls
    #: :meth:`TorchSdpaAttentionImpl.do_kv_cache_update` separately; it is a no-op.
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_name() -> str:
        # The enum member the plugin registers the class under: ``--attention-backend TORCH_SDPA``.
        return "TORCH_SDPA"

    @staticmethod
    def get_impl_cls() -> type[TorchSdpaAttentionImpl]:
        return TorchSdpaAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TorchSdpaAttentionMetadataBuilder]:
        return TorchSdpaAttentionMetadataBuilder

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """Encoder-only attention (the only type this backend implements)."""
        return attn_type in (AttentionType.ENCODER_ONLY,)

    @classmethod
    def supports_sliding_window(cls) -> bool:
        """The model's sliding layers pass ``per_layer_sliding_window`` (513); SDPA gets a mask."""
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        """Encoder attention is bidirectional; the backend never applies a causal mask."""
        return True

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """No cap: SDPA has no 256-head-dim kernel limit (the model's full layers are 512)."""
        return []

    @staticmethod
    def use_cascade_attention(*args: object, **kwargs: object) -> bool:
        return False


@dataclass
class TorchSdpaAttentionMetadata(AttentionMetadata):
    """The per-batch metadata the SDPA impl reads: the packed requests' token boundaries.

    Attributes:
        query_start_loc: ``(num_reqs + 1,)`` device offsets into the packed token dimension; request
            ``i`` owns ``[query_start_loc[i], query_start_loc[i + 1])``.
        num_reqs: The number of requests in the batch.
        num_actual_tokens: The number of real (unpadded) tokens in the packed dimension.
    """

    query_start_loc: torch.Tensor
    num_reqs: int
    num_actual_tokens: int


class TorchSdpaAttentionMetadataBuilder(AttentionMetadataBuilder[TorchSdpaAttentionMetadata]):
    """Copies the batch's request boundaries out of the engine's common metadata.

    Encoder-only attention keeps no KV cache and no block table, so nothing else is needed: the
    engine builds one metadata per batch (the runner's encoder-only model state) and hands it to
    every layer of the group.
    """

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TorchSdpaAttentionMetadata:
        return TorchSdpaAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
        )


class TorchSdpaAttentionImpl(AttentionImpl):
    """Per-request ``scaled_dot_product_attention`` over the packed q/k/v.

    Shapes follow vLLM's attention contract: ``query`` is ``[num_tokens, num_heads, head_size]``,
    ``key``/``value`` are ``[num_tokens, num_kv_heads, head_size]`` and ``output`` is
    ``[num_tokens, num_heads, head_size]``; the requests are packed along the token dimension and
    split by :attr:`TorchSdpaAttentionMetadata.query_start_loc`. Grouped-query attention is left to
    SDPA's ``enable_gqa``. The only mask is the sliding layers' window (``|i - j| <= W - 1``, see the
    module docstring); everything else attends every token of its own request, non-causally.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs: object,
    ) -> None:
        if attn_type not in (AttentionType.ENCODER_ONLY,):
            raise NotImplementedError(f"TORCH_SDPA is an encoder-only backend, got {attn_type}")
        if alibi_slopes is not None:
            raise NotImplementedError("TORCH_SDPA does not implement ALiBi slopes")
        if logits_soft_cap is not None:
            raise NotImplementedError("TORCH_SDPA does not implement a logits soft cap")
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("TORCH_SDPA does not implement KV-cache sharing")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) is not divisible by num_kv_heads ({self.num_kv_heads})"
        )
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """No-op: encoder-only layers keep no KV cache (the engine binds none)."""

    def _sliding_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """The sliding layers' bidirectional window, ``|i - j| <= W - 1`` (vLLM's W is HF's + 1)."""
        assert self.sliding_window is not None
        positions = torch.arange(seq_len, device=device)
        return (positions[:, None] - positions[None, :]).abs() <= self.sliding_window - 1

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TorchSdpaAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend each request's tokens to its own, writing the result into ``output``."""
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("TORCH_SDPA does not implement fused output quantization")
        if attn_metadata is None:
            # Profiling run: no metadata exists yet, the output values are unused.
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        out = output[:num_actual_tokens]
        enable_gqa = self.num_kv_heads != self.num_heads
        # One host round-trip per layer (the boundaries are the same for every layer): the requests
        # are packed along the token dimension and SDPA runs once per request.
        boundaries = attn_metadata.query_start_loc[: attn_metadata.num_reqs + 1].tolist()
        for index in range(attn_metadata.num_reqs):
            start, end = boundaries[index], boundaries[index + 1]
            if end <= start:
                # A zero-length request is padding (it only occurs in a padded capture batch).
                continue
            # [seq, heads, head] -> [1, heads, seq, head]
            q = query[start:end].transpose(0, 1).unsqueeze(0)
            k = key[start:end].transpose(0, 1).unsqueeze(0)
            v = value[start:end].transpose(0, 1).unsqueeze(0)
            mask = self._sliding_mask(end - start, q.device) if self.sliding_window is not None else None
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale, enable_gqa=enable_gqa)
            out[start:end] = attended.squeeze(0).transpose(0, 1)
        return output


def register_torch_sdpa_backend() -> None:
    """Register :class:`TorchSdpaAttentionBackend` under ``Backend.TORCH_SDPA`` (idempotent).

    Called by the ``torch-sdpa-attention`` patch at engine startup (the ``vllm.general_plugins``
    entry point), before any model layer asks the registry for a backend. The override only exists
    in a process that opted the patch in, so no other recipe's engine is touched.
    """
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(
        AttentionBackendEnum.TORCH_SDPA,
        f"{__name__}.{TorchSdpaAttentionBackend.__name__}",
    )
