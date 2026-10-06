"""rcp-ndcg-vllm-pplx: serve pplx-embed-v2-context-9b-preview on stock vLLM v0.31.x.

Registered through the ``vllm.general_plugins`` entry point (``rcp_vllm_pplx``): every
vLLM process calls :func:`register`, which (a) refuses a vLLM outside the validated range
and (b) registers the out-of-tree model class and its config handler. Registering is
idempotent — vLLM warns that plugins "can be loaded for multiple times in different
processes" and guards against a second load within one process.
"""

from __future__ import annotations

__all__ = ["PLUGIN_ARCHITECTURE", "PLUGIN_NAME", "register"]

#: The entry-point name (what ``VLLM_PLUGINS`` filters by).
PLUGIN_NAME = "rcp_vllm_pplx"
#: The HF architecture the plugin registers (the checkpoint's ``architectures[0]``).
PLUGIN_ARCHITECTURE = "PplxContextualModel"


def register() -> None:
    """Register the model class and its config handler with the running vLLM.

    The model class is registered as a lazy ``"module:Class"`` string, as vLLM's plugin
    documentation prescribes, so importing it (and with it vLLM's CUDA-touching model
    stack) stays out of every process that does not load the model. Re-registration is a
    deliberate no-op (``ModelRegistry.register_model`` overwrites the same key).

    Raises:
        RuntimeError: The running vLLM is outside the validated range
            (``rcp_vllm_pplx.version_guard``).
    """
    from rcp_vllm_pplx.version_guard import require_vllm_version  # noqa: PLC0415

    require_vllm_version()

    from vllm import ModelRegistry  # noqa: PLC0415  (the plugin runs inside vLLM)
    from vllm.model_executor.models.config import (  # noqa: PLC0415
        MODELS_CONFIG_MAP,
    )

    from rcp_vllm_pplx.config import PplxContextualConfig  # noqa: PLC0415

    if PLUGIN_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            PLUGIN_ARCHITECTURE,
            "rcp_vllm_pplx.model:PplxContextualForPooling",
        )

    # The map is consulted by ModelConfig._try_verify_and_update_model_config before the
    # engine resolves anything else; EngineArgs.__post_init__ loads the general plugins
    # first, so a plugin-added entry is seen in every serve path.
    MODELS_CONFIG_MAP[PLUGIN_ARCHITECTURE] = PplxContextualConfig
