"""rcp-ndcg-vllm: serve pplx-embed-v2-context-9b-preview on stock vLLM v0.31.x.

Registered through the ``vllm.general_plugins`` entry point (``rcp_ndcg_vllm.models.pplx``): every
vLLM process calls :func:`register`, which (a) refuses a vLLM outside the validated range,
(b) registers the checkpoint's configuration class with transformers' ``AutoConfig`` (so
the engine parses ``config.json`` locally and never executes the checkpoint's remote
config code -- see :mod:`rcp_ndcg_vllm.models.pplx.hf_config`) and (c) registers the out-of-tree model
class and its vLLM config handler. Registering is idempotent — vLLM warns that plugins
"can be loaded for multiple times in different processes" and guards against a second
load within one process.
"""

from __future__ import annotations

__all__ = ["HF_MODEL_TYPE", "PLUGIN_ARCHITECTURE", "PLUGIN_NAME", "register"]

#: The entry-point name (what ``VLLM_PLUGINS`` filters by).
PLUGIN_NAME = "rcp_ndcg_vllm.models.pplx"
#: The HF architecture the plugin registers (the checkpoint's ``architectures[0]``).
PLUGIN_ARCHITECTURE = "PplxContextualModel"
#: The checkpoint config's ``model_type`` (config.json), what ``AutoConfig.register`` keys on.
HF_MODEL_TYPE = "pplx_contextual_qwen3_5"


def register() -> None:
    """Register the HF config class, the model class and its config handler.

    The transformers config class is registered BEFORE the architecture: ``AutoConfig``
    consults its registry the first time the checkpoint's ``config.json``
    (``auto_map`` -> remote code) is resolved, while the ``ModelConfig`` is built. With
    ``model_type`` registered locally, transformers takes the explicit-local-code path
    and never fetches or executes the checkpoint's remote ``configuration_`` module.
    ``exist_ok=True`` keeps repeated plugin loads idempotent (a second registration
    overwrites with the same class).

    The model class is registered as a lazy ``"module:Class"`` string, as vLLM's
    plugin documentation prescribes, so importing it (and with it vLLM's CUDA-touching
    model stack) stays out of every process that does not load the model.

    Raises:
        RuntimeError: The running vLLM is outside the validated range
            (:mod:`rcp_ndcg_vllm.models.version_guard`).
    """
    from rcp_ndcg_vllm.models.version_guard import require_vllm_version  # noqa: PLC0415

    require_vllm_version()

    from transformers import AutoConfig  # noqa: PLC0415

    from rcp_ndcg_vllm.models.pplx.hf_config import PplxContextualConfig  # noqa: PLC0415

    AutoConfig.register(HF_MODEL_TYPE, PplxContextualConfig, exist_ok=True)

    from vllm import ModelRegistry  # noqa: PLC0415  (the plugin runs inside vLLM)
    from vllm.model_executor.models.config import (  # noqa: PLC0415
        MODELS_CONFIG_MAP,
    )

    from rcp_ndcg_vllm.models.pplx.config import PplxModelConfigHandler  # noqa: PLC0415

    if PLUGIN_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            PLUGIN_ARCHITECTURE,
            "rcp_ndcg_vllm.models.pplx.model:PplxContextualForPooling",
        )

    # The map is consulted by ModelConfig._try_verify_and_update_model_config before the
    # engine resolves anything else; EngineArgs.__post_init__ loads the general plugins
    # first, so a plugin-added entry is seen in every serve path.
    MODELS_CONFIG_MAP[PLUGIN_ARCHITECTURE] = PplxModelConfigHandler
