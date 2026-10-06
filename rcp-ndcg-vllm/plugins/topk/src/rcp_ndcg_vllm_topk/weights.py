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

The real mapper is built in ``rcp_ndcg_vllm_topk.model`` from vLLM's own
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
    "map_checkpoint_name",
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
