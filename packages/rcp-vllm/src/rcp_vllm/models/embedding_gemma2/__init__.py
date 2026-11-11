"""The embeddinggemma-2 plugin package: the ``TORCH_SDPA`` backend its recipe's patch registers.

E2 r1-E2 round 10 shipped a transformers/vLLM backport here (``fold/``, the four transformers 5.19.0
``embedding_gemma2`` modules and vLLM commit ``8cbd5d03``'s model, plus ``config.py``'s config hook and
``install_fold``/``register_embedding_gemma2``): the 0.31 image the verifying waves ran carried no
``EmbeddingGemma2Model`` and its transformers (5.17.0) no ``embedding_gemma2`` classes, so the plugin
supplied both.  The fold is **deleted** (the parity decision: the cross-image cosine 0.999972 is inside
the recipe's measured bf16 floor 0.999882, and the 0.33 image's transformers 5.20.2 ships the classes
while the v0.33.0 tag's registry carries the architecture, so the backport was inert on the pinned
image).  Nothing here installs or registers anything any more; the recipe's engine uses the image's own
model, and the plugin contributes only the SDPA backend below.

:mod:`.attention` is what survives, because the recipe's ``torch-sdpa-attention`` patch stays (D4: the
probe differs from the native backend outside its null-controlled gate): the model's heterogeneous head
dimensions (sliding 256, full 512) leave no backend the pinned image can run, and the backend class
there is registered under ``Backend.TORCH_SDPA`` by that patch.  The module imports vLLM/torch by
design, so it is registry-lazy (``rcp_vllm.models.LAZY_MODEL_MODULES``), and this package's
``__init__`` imports nothing.
"""

from __future__ import annotations

__all__: list[str] = []
