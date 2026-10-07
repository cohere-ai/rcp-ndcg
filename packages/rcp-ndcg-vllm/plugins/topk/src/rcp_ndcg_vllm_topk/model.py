"""The vLLM model class for ``topk-io/topk-embed-v1-small``.

``TopkEmbedModel`` subclasses the native ``ColQwen3_5Model`` (the stock
late-interaction model on the same Qwen3.5 backbone) and overrides two things:
the checkpoint-name mapping (class attribute ``hf_to_vllm_mapper``, replacing
the stock ColPali-convention mapper) and ``load_weights``, which calls the
stock loader and then marks the projection's zero bias as loaded (the
checkpoint ships none).  Everything forward-affecting — the forward pass, the
multimodal processor registration, the token-embed pooler wiring (projection
-> MRL slice -> L2 normalise), the zero-bias construction — is inherited
unchanged, which is what keeps the served numbers faithful.

Serving contract (why each inherited piece is the right one):

- Forward: inherited verbatim.  The reference implementation runs the six
  full-attention layers NON-causally (the checkpoint's ``text_config``
  carries ``is_causal: false``), and vLLM already honours that: the stock
  ``Qwen3NextAttention`` constructor (vLLM v0.31.0,
  vllm/model_executor/models/qwen3_next.py:335-343) reads
  ``getattr(config, "is_causal", True)`` from the text config and builds
  ``AttentionType.ENCODER_ONLY`` (bidirectional, no KV cache needed for a
  pooling model) when it is false.  The in-tree ColQwen3.5 models reach the
  same state through a registry config handler that sets ``is_causal``
  (vllm/model_executor/models/config.py:937 ColQwen3_5Config); this
  checkpoint carries the flag itself, and out-of-tree architectures get no
  config handler (``MODELS_CONFIG_MAP`` is keyed by in-tree architecture
  name), so no attention swap is needed or performed.  The linear-attention
  (GatedDeltaNet) layers are recurrent and causal in both implementations.
- Projection: inherited construction (the load-tracker marking is the
  plugin's ``load_weights``, below).  ``ColQwen3_5Model.__init__`` builds
  ``custom_text_proj = nn.Linear(hidden_size, embed_dim, bias=True,
  dtype=head_dtype)`` with a zero-initialised bias (colqwen3_5.py:177-183).
  This checkpoint's ``head`` has no bias tensor; a zero bias adds nothing to
  the matmul, so the served projection is score-equivalent to the reference's
  ``bias=False`` head, and a future checkpoint revision that ships
  ``head.bias`` is loaded over it instead of silently dropped.
- Pooling: inherited wiring.  ``embed_dim`` resolves from the config's ``dim``
  (2048; colqwen3_5.py:162-169) and the module is handed to
  ``pooler_for_token_embed`` as the projector (colqwen3_5.py:187), giving
  per-token vectors with float32 head arithmetic, the ``dimensions`` MRL
  slice and L2 normalisation — the reference computes the head in bf16 and
  casts the result to fp32, so per-token vectors agree up to bf16 rounding of
  the head operands; the served-vs-reference equivalence on the GPU wave
  measures that delta, and rcp-ndcg transfers float16 on
  the client, which dominates it.
- Multimodal: inherited registration.  The ``@MULTIMODAL_REGISTRY`` decorator
  stores its factories as a class attribute on ``ColQwen3_5Model``, which this
  subclass inherits, so the checkpoint's own ``Qwen3VLProcessor``
  (processor_config.json) drives image documents without extra code.

Version guard: importing this module asserts the installed vLLM is in the
tested range (see ``guard.py``) *before* the vLLM imports, so an unsupported
engine fails with an explicit message rather than a missing-symbol traceback.
"""

from __future__ import annotations

from .guard import ensure_vllm_version

# Refuse untested vLLM lines before any vLLM import (see guard.py).
ensure_vllm_version()

from collections.abc import Iterable  # noqa: E402

import torch  # noqa: E402
from vllm.model_executor.models.colqwen3_5 import (  # noqa: E402
    ColQwen3_5Model,
)
from vllm.model_executor.models.qwen3_5 import (  # noqa: E402
    Qwen3_5ForConditionalGeneration,
)
from vllm.model_executor.models.utils import WeightsMapper  # noqa: E402

from .weights import (  # noqa: E402
    PROJECTION_SOURCE_PREFIX,
    PROJECTION_TARGET_PREFIX,
    mark_zero_initialised,
)

__all__ = ["TopkEmbedModel"]


class TopkEmbedModel(ColQwen3_5Model):
    """``topk-io/topk-embed-v1-small``: multimodal late-interaction pooling.

    Served through ``/pooling`` with ``task: token_embed``: one
    L2-normalised 2048-dim vector per prompt token, float32 head arithmetic
    (``head_dtype`` defaults to float32 for pooling runners), scored
    client-side by fp32 MaxSim.  The differences from ``ColQwen3_5Model`` are
    the checkpoint-name mapping below and the zero-bias marking in
    ``load_weights``; every forward-affecting behaviour is inherited (module
    docstring).
    """

    # This checkpoint follows the Qwen3-VL naming convention
    # (`model.language_model.*`, `model.visual.*`), not ColPali's
    # (`language_model.*`), so the plugin restores the mapper that
    # Qwen3_5ForConditionalGeneration itself ships for that convention
    # (qwen3_vl.py:1819-1826 via qwen3_5.py:476-479) instead of the
    # ColQwen3_5Model override (colqwen3_5.py:135-143), and adds the trained
    # projection rename `head.` -> `custom_text_proj.` (the stock class's
    # `_PROJ_LAYER_NAMES` would never match `head.weight`, which is the first
    # reason a flags-only serve of this checkpoint cannot load).  The mapping
    # is restated as data in weights.py and cross-checked by the tests.
    hf_to_vllm_mapper = Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={
            PROJECTION_SOURCE_PREFIX: PROJECTION_TARGET_PREFIX,
        }
    )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the checkpoint, then claim the projection's zero bias as initialized.

        The checkpoint's ``head`` is bias-less while ``ColQwen3_5Model`` builds
        ``custom_text_proj`` with a zero-initialised bias (score-equivalent to
        ``bias=False``).  vLLM v0.31.0's load tracker refuses a parameter the
        checkpoint never supplied (``model_loader/default_loader.py:
        track_weights_loading``: serving this checkpoint on the stock v0.31.0
        image died on ``{'custom_text_proj.bias'}``), so the returned set is annotated under
        both qualnames exactly as the in-tree projection loader marks a shipped
        one (``colqwen3_5.py:load_weights``).  A checkpoint revision that ships
        ``head.bias`` is loaded over the zeros first and needs no annotation.
        """
        loaded = super().load_weights(weights)
        return mark_zero_initialised(loaded)
