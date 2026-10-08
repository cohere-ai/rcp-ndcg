"""Checkpoint facts for ``perplexity-ai/pplx-embed-v2-late-0.6b``, as plain data.

This module imports no vLLM and no torch: the CPU test suite verifies the exact
names the served model will load against the checkpoint's safetensors census
(derived from the Hub file headers at the pinned revision), and a vLLM release
that changes the upstream mapper is caught by the cross-check test instead of
silently corrupting weight loading.

The checkpoint (revision ``8fc2de24534aa3610d85fa59c463313a5f096455``) ships
305 tensors under two prefixes and one head file:

- ``language_model.*``   (152 tensors: ``embed_tokens`` [248320, 1024], the 150
  layer tensors, ``norm``)
- ``visual.*``           (153 tensors, the Qwen3.5-VL vision tower)
- ``1_Dense/model.safetensors`` (the trained Dense head: ``linear.weight``
  [128, 1024]; this revision ships no bias) -- a SEPARATE file the stock vLLM
  weight discovery never reads: ``DefaultModelLoader._prepare_weights`` globs
  ``hf_folder/*.safetensors`` non-recursively (vllm/model_executor/
  model_loader/default_loader.py:221 at v0.31.0), so the head must be
  loaded by the plugin itself and renamed onto the projector module
  ``custom_text_proj`` that the inherited ``ColQwen3_5Model`` exposes to its
  token-embed pooler (vllm/model_executor/models/colqwen3_5.py:177-187).

A flags-only serve cannot serve this checkpoint through the name either: the
registry does not carry ``Qwen3_5Model``, and the only fallback resolution left
is vLLM's transformers-backend wrapper (a generic ``AutoModel`` host), whose
construction crashes on this checkpoint's config and which has no path to the
Dense head or the multi-vector contract. The plugin's registration makes the
name resolve to the class in ``rcp_vllm_pplx.late`` instead.

There is no ``lm_head`` and no ``mtp`` in the checkpoint: the LM head is tied
to ``embed_tokens`` (config.json ``tie_word_embeddings: true``) and loads as
the tied alias vLLM's ``AutoWeightsLoader`` already handles
(vllm/model_executor/models/utils.py:202-215, :475-486 at v0.31.0).
"""

from __future__ import annotations

import re

__all__ = [
    "ARCHITECTURE",
    "CHECKPOINT_TENSOR_RE",
    "DENSE_HEAD_BIAS_TENSOR",
    "DENSE_HEAD_FILE",
    "DENSE_HEAD_TENSOR",
    "IGNORED_CHECKPOINT_PREFIXES",
    "PROJECTION_BIAS_TARGET_NAME",
    "PROJECTION_TARGET_NAME",
    "ZERO_INITIALISED_PARAMETERS",
    "map_checkpoint_name",
    "mark_zero_initialised",
]

#: The checkpoint's ``architectures[0]`` (config.json at the pinned revision) --
#: the name the plugin registers with vLLM's model registry. vLLM v0.31.0's own
#: registry carries the qwen3_5 family under other names (``Qwen3_5ForCausalLM``,
#: ``ColQwen3_5``, ``Qwen3_5ForConditionalGeneration``, ...; vllm/model_executor/
#: models/registry.py:203-204, :284, :596) and not this one, so a flags-only
#: serve falls through to the transformers-backend fallback, which cannot serve
#: this checkpoint (see the module docstring).
ARCHITECTURE = "Qwen3_5Model"

#: The plugin module and class the registry imports lazily for that
#: architecture (the ``"module:Class"`` string vLLM's plugin registration takes).
MODEL_QUALNAME = "rcp_vllm_pplx.late:PplxLateMultiVectorModel"

#: The checkpoint's own Dense-head module file (sentence-transformers' save
#: format: one file per module), inside the model repo at the pinned revision.
DENSE_HEAD_FILE = "1_Dense/model.safetensors"

#: The single tensor inside that file (parsed from the file's safetensors
#: header at the pinned revision: ``linear.weight``, F32, [128, 1024]).
DENSE_HEAD_TENSOR = "linear.weight"

#: A shipped head bias, loaded over the constructor's zeros when a checkpoint
#: revision ships one (1_Dense/config.json ``bias: true``); this revision ships
#: no bias tensor.
DENSE_HEAD_BIAS_TENSOR = "linear.bias"

#: The vLLM parameters the head tensors load into: the inherited
#: ``ColQwen3_5Model`` exposes the projection as ``custom_text_proj`` and hands
#: it to the token-embed pooler as its projector. The stock loader reaches that
#: module only under its ``_PROJ_LAYER_NAMES`` ("custom_text_proj",
#: "embedding_proj_layer" -- colqwen3_5.py:210-214 at v0.31.0), so the head
#: tensors are renamed to these names before the stock loader sees them.
PROJECTION_TARGET_NAME = "custom_text_proj.weight"
PROJECTION_BIAS_TARGET_NAME = "custom_text_proj.bias"

#: Restatement of the prefix mapping the plugin's model class inherits
#: (vllm/model_executor/models/colqwen3_5.py:135-143 at v0.31.0), so the tests
#: can verify the served mapping without importing vLLM: exactly this
#: checkpoint's naming convention (top-level ``language_model.*`` maps onto the
#: class's ``language_model.model.*``; ``visual.*`` matches as-is; this
#: checkpoint has no ``mtp.*`` weights to drop).
CHECKPOINT_PREFIXES: dict[str, str | None] = {
    "language_model.": "language_model.model.",
    "mtp.": None,
}

#: The checkpoint prefixes that must be dropped, restated for the CPU-side test.
IGNORED_CHECKPOINT_PREFIXES = ("mtp.",)

#: Every checkpoint tensor the backbone loader must resolve or explicitly drop:
#: the two prefixes above (the head file is loaded separately, by name).
CHECKPOINT_TENSOR_RE = re.compile(r"^(?:language_model|visual)\..+")

#: The served module parameters the checkpoint never supplies: the projection's
#: bias, whose two qualnames share one parameter (the model attribute and the
#: pooler head's projector). ``ColQwen3_5Model`` builds ``custom_text_proj``
#: with ``bias=True`` and zero-initialises it (colqwen3_5.py:177-183 at
#: v0.31.0), so this checkpoint's bias-less Dense head behaves exactly as
#: ``bias=False`` -- and a future checkpoint revision that ships a head bias
#: loads over the zeros instead of being silently dropped.
ZERO_INITIALISED_PARAMETERS = ("custom_text_proj.bias", "pooler.head.projector.bias")


def map_checkpoint_name(name: str) -> str | None:
    """Map one checkpoint weight name to its vLLM name; ``None`` means drop.

    Prefix-based, like vLLM's ``WeightsMapper.map_name`` for
    ``orig_to_new_prefix`` tables: the first table prefix that the name starts
    with is replaced once. The checkpoint's name spaces are disjoint (only
    ``language_model.*`` and ``visual.*`` exist, measured), so no ordering
    hazard exists; the projection tensor never passes through this mapper
    (the plugin intercepts it by its target name before the stock loader).
    """
    for prefix, new_prefix in CHECKPOINT_PREFIXES.items():
        if name.startswith(prefix):
            if new_prefix is None:
                return None
            return name.replace(prefix, new_prefix, 1)
    return name


def mark_zero_initialised(loaded: set[str]) -> set[str]:
    """Mark the projection's zero-initialised bias as initialized in a load tracker's set.

    Inputs: the ``set[str]`` ``ColQwen3_5Model.load_weights`` returns (every parameter the
    checkpoint's tensors were loaded into).  Output: the same set, with
    :data:`ZERO_INITIALISED_PARAMETERS` added.  The reason: vLLM v0.31.0's load tracker
    (``model_loader/default_loader.py:track_weights_loading``) raises for any model
    parameter outside that set, but this checkpoint is bias-less -- the bias is the
    constructor's zeros (score-equivalent, see the module docstring of
    :mod:`rcp_vllm_pplx.late`) and a shipped head bias is already in the set before this
    call, so the marking is idempotent either way.
    """
    loaded.update(ZERO_INITIALISED_PARAMETERS)
    return loaded
