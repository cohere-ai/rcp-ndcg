"""The out-of-tree vLLM model class for ``perplexity-ai/pplx-embed-v2-late-0.6b``.

``PplxLateMultiVectorModel`` subclasses vLLM's own ``ColQwen3_5Model`` -- the
stock late-interaction pooling model on the same Qwen3.5 backbone (the
ColBERT-style linear head, the bidirectional full-attention layers, the vision
tower, the ``token_embed`` pooler wiring and the multimodal processor
registration are all inherited unchanged, each difference commented with the
reason and the in-tree line it follows). It overrides only ``load_weights``,
because the stock weight pipeline cannot see this checkpoint's trained head:

- **The architecture name.** The checkpoint's ``architectures[0]`` is
  ``Qwen3_5Model``, which vLLM v0.31.0's registry does not carry (its qwen3_5
  family is registered under ``Qwen3_5ForCausalLM``, ``ColQwen3_5``,
  ``Qwen3_5ForConditionalGeneration``, ...); the plugin's ``register``
  (``rcp_vllm_pplx.__init__``) adds that one name, pointing at this class.
- **The weight naming.** The checkpoint saves the text backbone top-level as
  ``language_model.*`` and the vision tower as ``visual.*`` -- exactly the
  ColPali convention the inherited ``hf_to_vllm_mapper`` maps
  (``language_model.`` -> ``language_model.model.``; ``visual.`` matches
  as-is; no ``mtp.`` weights exist to drop). The absent ``lm_head`` is the
  tied alias of ``embed_tokens`` (``tie_word_embeddings: true``) and loads as
  its skipped alias.
- **The Dense head.** The trained projection ships as a SEPARATE
  sentence-transformers module file, ``1_Dense/model.safetensors`` (one
  tensor, ``linear.weight`` [128, 1024], fp32). The stock loader's discovery
  globs ``*.safetensors`` in the snapshot root non-recursively
  (default_loader.py:226-233 at v0.31.0), so the head never enters the weight
  iterator; this class fetches that file itself -- from the checkpoint
  directory when serving a local path, else from the same Hub snapshot the
  engine already downloaded (vLLM's own ``download_weights_from_hf`` pulls
  every ``*.safetensors`` including subdirectory paths; the in-tree precedent
  for a model class reading an extra safetensors file is
  ``mimo_audio.py:1262-1272``) -- and renames the tensor onto
  ``custom_text_proj.weight``, the name the inherited loader's
  ``_PROJ_LAYER_NAMES`` intercepts and the pooler head's projector shares.
- **The zero bias.** The checkpoint's head is bias-less while the inherited
  constructor builds ``custom_text_proj`` with a zero-initialised bias
  (score-equivalent); the returned loaded set is annotated under both
  qualnames so vLLM's load tracker accepts it, exactly as the in-tree
  projection loader marks a shipped bias (colqwen3_5.py:load_weights).
- **Bidirectional attention.** ``is_causal=False`` rides the checkpoint's own
  ``text_config`` (vLLM's qwen3-next layers read it and build
  ``AttentionType.ENCODER_ONLY``, qwen3_next.py:336-343 at v0.31.0); the
  in-tree ColQwen3_5 config handler is not registered for this out-of-tree
  architecture and none is needed.
- **No config class registration.** ``model_type qwen3_5`` is native to the
  engine's transformers line, and vLLM's own config registry
  (transformers_utils/config.py: qwen3_5 -> Qwen3_5Config) parses it and
  registers it with transformers' AutoConfig -- ``trust_remote_code`` stays
  false and nothing remote exists to execute (the checkpoint carries no
  auto_map).

Version guard: importing this module asserts the installed vLLM is in the
tested range (see ``version_guard.py``) *before* the vLLM imports, so an
unsupported engine fails with an explicit message rather than a
missing-symbol traceback.
"""

from __future__ import annotations

__all__ = ["PplxLateMultiVectorModel"]

# The guard's second import moment, hoisted above every vllm import: the registry
# imports this module lazily through the "module:Class" string, and the guard (which
# itself imports no vllm) must refuse an out-of-range engine before that engine's
# diverging surface can raise a raw ImportError instead of the guard's RuntimeError.
from rcp_vllm_pplx.version_guard import require_vllm_version

require_vllm_version()

from collections.abc import Iterable  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from vllm.model_executor.models.colqwen3_5 import ColQwen3_5Model  # noqa: E402

from rcp_vllm_pplx.late_data import (  # noqa: E402
    DENSE_HEAD_BIAS_TENSOR,
    DENSE_HEAD_FILE,
    DENSE_HEAD_TENSOR,
    PROJECTION_BIAS_TARGET_NAME,
    PROJECTION_TARGET_NAME,
    map_checkpoint_name,
    mark_zero_initialised,
)


def _dense_head_file(model: str, revision: str | None) -> Path:
    """Locate the checkpoint's own ``1_Dense/model.safetensors``.

    A local model directory carries the file beside config.json; a Hub id
    resolves through the same HF cache the engine's snapshot download already
    populated (the download's ``allow_patterns`` match subdirectory paths, so
    the file is there; ``huggingface_hub`` honours ``HF_HUB_OFFLINE``). Raises
    with a hint when the file is absent, never a bare stack.
    """
    local = Path(model)
    if local.is_dir():
        candidate = local / DENSE_HEAD_FILE
        if not candidate.is_file():
            raise FileNotFoundError(
                f"the checkpoint directory {model!r} carries no {DENSE_HEAD_FILE}: the "
                "sentence-transformers Dense module file this plugin loads the projection "
                "head from is missing (expected beside config.json at the pinned revision)"
            )
        return candidate
    from huggingface_hub import hf_hub_download  # noqa: PLC0415  (only the Hub path needs it)

    try:
        return Path(hf_hub_download(model, DENSE_HEAD_FILE, revision=revision))
    except Exception as error:  # noqa: BLE001  (any Hub/cache failure is the same loud contract)
        raise FileNotFoundError(
            f"cannot resolve {DENSE_HEAD_FILE!r} of {model!r} (revision {revision!r}): the "
            "plugin loads the Dense head from the checkpoint's own sentence-transformers "
            "module file; the engine's weight download does fetch it (allow_patterns match "
            "subdirectory paths), so this usually means the engine never resolved the "
            f"checkpoint. Underlying error: {type(error).__name__}: {error}"
        ) from error


class PplxLateMultiVectorModel(ColQwen3_5Model):
    """``perplexity-ai/pplx-embed-v2-late-0.6b``: multimodal late-interaction pooling.

    Served through ``/pooling`` with ``task: token_embed``: one L2-normalised
    128-dim vector per prompt token (float32 head arithmetic, ``head_dtype``
    defaulting to fp32 for pooling runners), scored client-side by fp32
    MaxSim. Every forward-affecting behaviour is inherited from
    ``ColQwen3_5Model`` -- the projection construction (hidden -> embed_dim,
    the biasless checkpoint equivalent to its zero bias), the token-embed
    pooler wiring (projection -> L2 normalise), the multimodal processor
    registration for image documents and the bidirectional full-attention
    layers; the class's own delta is the weight loading above.
    """

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the checkpoint, routing the checkpoint's separate Dense-head file by hand.

        The head file never enters the stock weight iterator (the module
        docstring's "The Dense head"), so it is read here, shape-checked
        against the projection module the constructor built, renamed onto
        :data:`rcp_vllm_pplx.late_data.PROJECTION_TARGET_NAME` and chained in
        ahead of the backbone tensors; the stock loader then resolves every
        backbone name through the inherited mapper and the head through its
        projection branch, and the zero-initialised bias is claimed for the
        load tracker.
        """
        model_config = self.model_config  # set by the Qwen3_5ForConditionalGeneration base
        head_path = _dense_head_file(model_config.model, model_config.revision)
        from safetensors.torch import load_file  # noqa: PLC0415  (safetensors ships with vLLM)

        state = load_file(str(head_path), device="cpu")
        if DENSE_HEAD_TENSOR not in state:
            raise ValueError(
                f"the checkpoint's {DENSE_HEAD_FILE} carries no tensor "
                f"{DENSE_HEAD_TENSOR!r} (it has: {sorted(state)}); the checkpoint revision "
                "does not match the plugin's pinned one"
            )
        head_weight = state[DENSE_HEAD_TENSOR]
        expected_shape = tuple(self.custom_text_proj.weight.shape)
        if tuple(head_weight.shape) != expected_shape:
            raise ValueError(
                f"the checkpoint's Dense head {DENSE_HEAD_TENSOR!r} has shape "
                f"{tuple(head_weight.shape)}, but the served projector expects "
                f"{expected_shape} (serve.hf_overrides embed_dim against the checkpoint's "
                "1_Dense/config.json out_features); the checkpoint revision does not match "
                "the recipe's pinned one"
            )
        projection = [(PROJECTION_TARGET_NAME, head_weight)]
        # A future checkpoint revision that ships a head bias (1_Dense/config.json
        # bias: true) loads over the constructor's zeros through the same rename;
        # this pinned revision ships none, so the load is a no-op here.
        if DENSE_HEAD_BIAS_TENSOR in state:
            projection.append((PROJECTION_BIAS_TARGET_NAME, state[DENSE_HEAD_BIAS_TENSOR]))

        named_weights = list(weights)
        for name, _ in named_weights:
            if map_checkpoint_name(name) is None:
                continue
            if not name.startswith(("language_model.", "visual.", "mtp.")):
                raise ValueError(
                    f"the checkpoint tensor {name!r} is outside the mapped name spaces this "
                    "plugin pins (language_model.*, visual.*, mtp.*); the checkpoint revision "
                    "does not match the plugin's pinned one"
                )
        loaded = super().load_weights(iter([*projection, *named_weights]))
        return mark_zero_initialised(loaded)
