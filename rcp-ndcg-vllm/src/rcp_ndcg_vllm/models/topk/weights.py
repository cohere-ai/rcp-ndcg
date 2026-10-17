"""Checkpoint-to-vLLM weight-name mapping for topk-embed-v1-small.

The checkpoint at revision e54485ebab921f2c18c4d092b3f4c40dcca26781 ships 618
tensors under three prefixes (safetensors header, re-derived at build time and
asserted in tests):

- ``head.weight``                       (1 tensor)  the trained projection
- ``model.language_model.*``            (320 tensors)
- ``model.visual.*``                    (297 tensors)

vLLM's ``Qwen3_5ForConditionalGeneration`` module tree expects
``language_model.model.*`` and ``visual.*`` relative to the root, and
``ColQwen3_5Model`` (the base class of the plugin's model) assumes the
ColQwen3.5 checkpoint convention ``language_model.*`` / ``visual.*``.  The
plugin therefore overrides ``hf_to_vllm_mapper`` with the Qwen3-VL-shaped
mapping (the one ``Qwen3_5ForConditionalGeneration`` itself inherits, restated
below) plus the projection rename that is the whole point of this plugin:
``head.`` -> ``custom_text_proj.``.

The real mapper is built in ``rcp_ndcg_vllm.models.topk.model`` from vLLM's own
``WeightsMapper`` objects.  This module restates the resulting name mapping as
plain data so the CPU test suite (no vLLM import) can verify the exact names
the served model will load, and so a vLLM release that changes the upstream
mapper is caught by the cross-check test instead of silently corrupting
weight loading.
"""

from __future__ import annotations

__all__ = [
    "PROJECTION_SOURCE_PREFIX",
    "PROJECTION_TARGET_PREFIX",
    "CHECKPOINT_TO_VLLM_PREFIXES",
    "IGNORED_CHECKPOINT_PREFIXES",
    "ZERO_INITIALISED_PARAMETERS",
    "map_checkpoint_name",
    "mark_zero_initialised",
]

# The trained projection of topk-embed-v1-small: `head` (Linear,
# hidden_size -> dim, no bias in the checkpoint).  vLLM's ColQwen3_5Model
# exposes the projection as `custom_text_proj` and passes that module to the
# token-embed pooler as its projector, so the checkpoint name must be mapped
# onto it (ColQwen3_5Model._PROJ_LAYER_NAMES only matches
# "custom_text_proj"/"embedding_proj_layer"; vllm/model_executor/models/
# colqwen3_5.py:210-217 at v0.31.0).
PROJECTION_SOURCE_PREFIX = "head."
PROJECTION_TARGET_PREFIX = "custom_text_proj."

# Restatement of the upstream prefix table the plugin's model class merges in
# (vllm/model_executor/models/qwen3_vl.py:1819-1826, inherited by
# Qwen3_5ForConditionalGeneration at vllm/model_executor/models/qwen3_5.py:
# 476-479, at v0.31.0): the Qwen3-VL checkpoint convention is exactly this
# checkpoint's (`model.language_model.*`, `model.visual.*`), while the stock
# ColQwen3_5Model mapper assumes ColPali's `language_model.*` convention and
# would leave `model.language_model.*` unresolved.
CHECKPOINT_TO_VLLM_PREFIXES: dict[str, str | None] = {
    "model.visual.": "visual.",
    "lm_head.": "language_model.lm_head.",
    "model.language_model.": "language_model.model.",
    # Multi-token-prediction weights: this checkpoint has none (census above);
    # the upstream table drops them if a future revision ships some.
    "mtp.": None,
    # The plugin's own delta (see PROJECTION_*_PREFIX above).
    "head.": "custom_text_proj.",
}

# Checkpoint prefixes that must be dropped, restated for the CPU-side test.
IGNORED_CHECKPOINT_PREFIXES = ("mtp.",)


def map_checkpoint_name(name: str) -> str | None:
    """Map one checkpoint weight name to its vLLM name; ``None`` means drop.

    Prefix-based, like vLLM's ``WeightsMapper.map_name`` for
    ``orig_to_new_prefix`` tables: the first table prefix that the name starts
    with is replaced once.  Order matches the upstream mapper (visual and
    lm_head before language_model; head last), which cannot collide because
    the checkpoint's name spaces are disjoint (one tensor starts with
    ``head.``, none with both ``model.visual.`` and ``model.language_model.``).
    """
    for prefix, new_prefix in CHECKPOINT_TO_VLLM_PREFIXES.items():
        if name.startswith(prefix):
            if new_prefix is None:
                return None
            return name.replace(prefix, new_prefix, 1)
    return name


#: The served module parameters the checkpoint never supplies: the projection's
#: bias, whose two qualnames share one parameter (the model attribute and the
#: pooler head's projector).  ColQwen3_5Model builds ``custom_text_proj`` with
#: ``bias=True`` and zero-initialises it (colqwen3_5.py:177-183 at v0.31.0) so a
#: bias-less checkpoint behaves exactly as ``bias=False`` -- and a future
#: revision that ships ``head.bias`` is loaded over the zeros by the in-tree
#: projection loader.
ZERO_INITIALISED_PARAMETERS = ("custom_text_proj.bias", "pooler.head.projector.bias")


def mark_zero_initialised(loaded: set[str]) -> set[str]:
    """Mark the projection's zero-initialised bias as initialized in a load tracker's set.

    Inputs: the ``set[str]`` ``ColQwen3_5Model.load_weights`` returns (every parameter the
    checkpoint's tensors were loaded into).  Output: the same set, with
    :data:`ZERO_INITIALISED_PARAMETERS` added.  The reason: vLLM v0.31.0's load tracker
    (``model_loader/default_loader.py:track_weights_loading``) raises for any model
    parameter outside that set, but this checkpoint is bias-less -- the bias is the
    constructor's zeros (score-equivalent, module docstring) and a shipped ``head.bias``
    is already in the set before this call, so the marking is idempotent either way.
    """
    loaded.update(ZERO_INITIALISED_PARAMETERS)
    return loaded
