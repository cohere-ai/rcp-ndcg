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

The callable also registers the checkpoint's configuration class with
transformers' ``AutoConfig`` (see ``config.py`` for why the engine must never
execute the checkpoint's remote config code on the stock image).
"""

from __future__ import annotations

__all__ = ["MODEL_ARCHITECTURE", "MODEL_QUALNAME", "MODEL_TYPE", "register"]

MODEL_ARCHITECTURE = "TopkEmbedModel"
MODEL_QUALNAME = "rcp_ndcg_vllm_topk.model:TopkEmbedModel"
MODEL_TYPE = "topk_embed"


def register() -> None:
    """Register the model architecture and the config class with vLLM.

    Runs once per vLLM process (the loader tolerates repeated runs).  The
    version guard raises before the registries are touched so an unsupported
    engine fails with an explicit message instead of importing classes whose
    contracts may have moved.
    """
    # Refuse untested vLLM lines before importing anything from vLLM (see
    # guard.py); the registry imports below need the very internals the guard
    # pins to the tested range.
    from .guard import ensure_vllm_version

    ensure_vllm_version()

    # Register the config class BEFORE the architecture: transformers'
    # AutoConfig consults its registry the first time the checkpoint's
    # config.json (auto_map -> remote code importing `fla`, absent from the
    # engine image) is resolved, which happens while the ModelConfig below is
    # built.  exist_ok=True keeps repeated plugin loads idempotent (the loader
    # runs once per process; a second registration overwrites with the same
    # class).  See config.py for why the remote code must never execute.
    from transformers import AutoConfig

    from .config import MODEL_TYPE, TopkEmbedConfig

    AutoConfig.register(MODEL_TYPE, TopkEmbedConfig, exist_ok=True)

    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(MODEL_ARCHITECTURE, MODEL_QUALNAME)
