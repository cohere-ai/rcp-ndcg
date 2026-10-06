"""The out-of-tree vLLM model class for pplx-embed-v2-context-9b-preview.

``PplxContextualForPooling`` subclasses vLLM's own ``Qwen3_5ForCausalLMBase`` — the
text-only Qwen3.5 hybrid backbone (gated-delta-net linear attention interleaved with
full attention), which already carries everything the checkpoint's text stack needs:
``embed_input_ids``, the hybrid flags, the mamba state helpers and the stacked GDN
weight mappings. Only what must differ is written here, each difference commented with
the reason and the in-tree line it follows:

- **No vision tower.** The checkpoint ships the Qwen3.5-VL vision tower (333
  ``visual.*`` tensors, 1.82 GB fp32) but the embedding path is text-only (the remote
  code takes no images), so the class subclasses the text-only causal-LM base rather
  than ``Qwen3_5ForConditionalGeneration`` and drops the tower's weights in the mapper
  (``orig_to_new_prefix={..., None}``), the way vLLM drops ``mtp.``.
- **No lm_head / logits processor.** The checkpoint has no ``lm_head`` and pooling never
  reads one; ``no_init_weights`` with ``targets=(LogitsProcessor, ParallelLMHead)``
  replaces both with a ``StageMissingLayer`` exactly as vLLM's own
  ``_create_pooling_model_cls`` (``models/adapters.py``) does for converted pooling
  models — saving the 248320×4096 fp32 vocabulary head (~4 GB) this checkpoint would
  otherwise allocate unused.
- **Weight prefix mapping.** The checkpoint saves the text backbone under
  ``language_model.*`` where this class keeps it under ``model.*`` (measured by the
  r-pplx lane; vLLM's ``ColQwen3_5Model`` ships the analogous
  ``language_model.`` → ``language_model.model.`` map for its VL nesting).
- **The chunk head.** ``contextual_projection.weight`` (fp32 ``[2048, 4096]``,
  bias-free) loads into the model's ``PplxInt8Projection``, which doubles as the pooler
  head's projector — the ``custom_text_proj`` pattern of vLLM's ``ColQwen3_5Model``.
- **Bidirectional attention.** ``is_causal=False`` is forced on both HF configs by the
  plugin's config handler (``rcp_vllm_pplx.config``), so vLLM's qwen3-next layers run
  ``AttentionType.ENCODER_ONLY`` on the full-attention layers.
"""

from __future__ import annotations

__all__ = ["PplxContextualForPooling"]

# The guard's second import moment, hoisted above every vllm import: the registry
# imports this module lazily through the "module:Class" string, and the guard (which
# itself imports no vllm) must refuse an out-of-range engine before that engine's
# diverging surface can raise a raw ImportError instead of the guard's RuntimeError.
from rcp_vllm_pplx.version_guard import require_vllm_version

require_vllm_version()

from collections.abc import Iterable  # noqa: E402

import torch  # noqa: E402
from vllm.config import VllmConfig  # noqa: E402
from vllm.model_executor.layers.logits_processor import LogitsProcessor  # noqa: E402
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead  # noqa: E402
from vllm.model_executor.model_loader.weight_utils import default_weight_loader  # noqa: E402
from vllm.model_executor.models.interfaces_base import default_pooling_type  # noqa: E402
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLMBase  # noqa: E402
from vllm.model_executor.models.utils import (  # noqa: E402
    AutoWeightsLoader,
    StageMissingLayer,
    WeightsMapper,
    no_init_weights,
)

from rcp_vllm_pplx.pooler import build_pooler  # noqa: E402
from rcp_vllm_pplx.pooling_core import PplxInt8Projection  # noqa: E402


@default_pooling_type(seq_pooling_type="CLS", tok_pooling_type="ALL")
class PplxContextualForPooling(Qwen3_5ForCausalLMBase):
    """``PplxContextualModel`` as a vLLM pooling model: one embedding per chunk.

    The ``seq_pooling_type="CLS"`` decoration keeps vLLM's ``ModelConfig.attn_type`` on
    the encoder-only path (matching the model's bidirectional contract) and the
    ``tok_pooling_type="ALL"`` default matches the token-wise task the pooler serves;
    neither pooling method is otherwise used — the plugin's ``PplxChunkPool`` replaces
    them (see :mod:`rcp_vllm_pplx.pooler`).
    """

    is_pooling_model = True

    # The checkpoint's naming mapped onto this class's modules; ``None`` drops the
    # weight (the WeightsMapper contract for unused tensors, as vLLM maps ``mtp.``).
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # The checkpoint's text backbone (measured r-pplx: `language_model.*`); the
            # base class's own map (qwen3_5.py:323-325) keeps the community
            # `model.language_model.*` variant first so both nestings load.
            "model.language_model.": "model.",
            "language_model.": "model.",
            # The Qwen3.5-VL vision tower ships in the checkpoint and is unused by the
            # embedding path (no image branch in the reference; the card's usage is
            # text-only) — dropped, saving 1.82 GB fp32.
            "visual.": None,
            "model.visual.": None,
            # No MTP weights in the checkpoint (measured); mirrors qwen3_5.py:324.
            "mtp.": None,
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # Replace the generation-only head with a missing-stage placeholder before the
        # base class builds it: the checkpoint has no lm_head (tie_word_embeddings is
        # false and the checkpoint carries no lm_head tensors) and pooling never calls
        # compute_logits. Follows vLLM's _create_pooling_model_cls (adapters.py:145-148,
        # the function at :128).
        with no_init_weights(
            self,
            lambda mod: StageMissingLayer("output", mod),
            targets=(LogitsProcessor, ParallelLMHead),
        ):
            super().__init__(vllm_config=vllm_config, prefix=prefix)

        config = vllm_config.model_config.hf_config
        text_config = vllm_config.model_config.hf_text_config
        head_dtype = vllm_config.model_config.head_dtype

        hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "the checkpoint's text config declares no hidden_size; cannot size the contextual_projection"
            )
        # Nothing silently defaults here: embedding_dim is the checkpoint's own field
        # (config.json "embedding_dim": 2048); a missing one is an error with a hint.
        embedding_dim = getattr(config, "embedding_dim", None)
        if embedding_dim is None:
            raise ValueError(
                "the checkpoint's config declares no embedding_dim; cannot size the "
                "contextual_projection (expected config.json 'embedding_dim': 2048)"
            )
        self.embedding_dim: int = int(embedding_dim)

        self.contextual_projection = PplxInt8Projection(
            hidden_size=hidden_size,
            embedding_dim=self.embedding_dim,
            dtype=head_dtype,
        )

        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None  # noqa: S101  (a pooling model always has one)
        self.pooler = build_pooler(vllm_config.model_config, projector=self.contextual_projection)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the checkpoint, routing ``contextual_projection`` by hand.

        The projection head has no ``model.*`` home (it is the pooler head's projector),
        so it is intercepted by name and loaded like vLLM's ``ColQwen3_5Model`` loads
        ``custom_text_proj`` — with both qualnames (the model attribute and the pooler
        path) marked loaded so the weight validator accepts the shared module.
        """
        projection_weights: list[tuple[str, torch.Tensor]] = []
        model_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if "contextual_projection" in name:
                projection_weights.append((name, weight))
            else:
                model_weights.append((name, weight))

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(model_weights, mapper=self.hf_to_vllm_mapper)

        for name, weight in projection_weights:
            param_name = name.split(".")[-1]
            param = getattr(self.contextual_projection.linear, param_name, None)
            if param is None:
                raise ValueError(
                    f"the checkpoint's {name!r} has no counterpart on the projection "
                    f"head (contextual_projection.linear.{param_name}); the checkpoint "
                    "revision does not match the plugin's pinned one"
                )
            # Move the checkpoint tensor onto the parameter, as ColQwen3_5Model does for
            # custom_text_proj: the parameter keeps its registration; only the tensor
            # travels. (Sharded loads are not wired: the recipe serves this model on one
            # GPU, the same constraint ColQwen3_5's own projection loader has.)
            device_weight = weight.to(device=param.device, dtype=param.dtype)
            default_weight_loader(param, device_weight)
            loaded.add(f"contextual_projection.linear.{param_name}")
            # The same module is the pooler head's projector; mark that path too.
            loaded.add(f"pooler.head.projector.linear.{param_name}")

        return loaded
