"""The ``torch-sdpa-attention`` patch: register the plugin's encoder-only SDPA backend.

E2 round 10: the embeddinggemma-2 recipe's engine starts with ``--attention-backend TORCH_SDPA``, and
the pinned image ships no decoder ``TORCH_SDPA`` class -- its enum value is the empty string ("this
tag is only used for ViT"), so the manual-selection path raises ``Backend TORCH_SDPA must be
registered before use`` (``vllm/v1/attention/backends/registry.py``). This patch registers
:class:`~rcp_ndcg_vllm.models.embedding_gemma2.attention.TorchSdpaAttentionBackend` under that enum
member, which is the mechanism the engine's own error names.

Opt-in: the embeddinggemma-2 recipe declares ``serve.patches: [..., torch-sdpa-attention]``, so only
that engine process overrides the member and only that recipe's fingerprint keys this module. The
backend class itself lives in the plugin's model package (its own module, so a change to it moves the
recipe's fingerprint) and imports vLLM/torch by design; this patch imports it inside :func:`apply`,
so importing :mod:`rcp_ndcg_vllm.patches` stays free of the engine stack.

Removal condition: delete this patch (the recipe's ``serve.patches`` entry with it) when
``engine.image`` moves to a vLLM image whose own backends clear the model's head dimensions.
"""

from __future__ import annotations

__all__ = ["PATCH_NAME", "apply"]

#: The opt-in name (``serve.patches``; the recipe schema validates it against ``PATCH_NAMES``).
PATCH_NAME = "torch-sdpa-attention"


def apply() -> None:
    """Register the plugin's encoder-only SDPA backend as ``Backend.TORCH_SDPA`` (idempotent)."""
    from rcp_ndcg_vllm.models.embedding_gemma2.attention import (  # noqa: PLC0415
        register_torch_sdpa_backend,
    )

    register_torch_sdpa_backend()
