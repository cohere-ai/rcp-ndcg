"""The ``vllm.general_plugins`` entry-point callable.

vLLM loads this group in every engine process at startup
(vllm/plugins/__init__.py:36-75 at v0.31.0, ``DEFAULT_PLUGINS_GROUP =
"vllm.general_plugins"``; executed by ``load_general_plugins`` from
``EngineArgs.from_cli_args``, the engine core and the workers, all before the
model architecture resolves).  Registering a ``"<module>:<class>"`` string
keeps the model class lazy: the registry imports
``rcp_ndcg_vllm_topk.model`` only when a request for the ``TopkEmbedModel``
architecture actually needs it (vllm/model_executor/models/registry.py:1122
``register_model`` at v0.31.0), which avoids importing CUDA-touching modules
in forked subprocesses.
"""

from __future__ import annotations

__all__ = ["MODEL_ARCHITECTURE", "MODEL_QUALNAME", "register"]

MODEL_ARCHITECTURE = "TopkEmbedModel"
MODEL_QUALNAME = "rcp_ndcg_vllm_topk.model:TopkEmbedModel"


def register() -> None:
    """Register ``TopkEmbedModel`` with vLLM's model registry.

    Runs once per vLLM process (the loader tolerates repeated runs).  The
    version guard raises before the registry is touched so an unsupported
    engine fails with an explicit message instead of importing classes whose
    contracts may have moved.
    """
    # Refuse untested vLLM lines before importing anything from vLLM (see
    # guard.py); the registry import below needs the very internals the guard
    # pins to the tested range.
    from .guard import ensure_vllm_version

    ensure_vllm_version()

    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(MODEL_ARCHITECTURE, MODEL_QUALNAME)
