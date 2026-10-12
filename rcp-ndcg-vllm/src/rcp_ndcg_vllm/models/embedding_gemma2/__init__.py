"""The embeddinggemma-2 fold: the classes the engine image the recipe pins lacks.

E2 r1: ``google/embeddinggemma-2`` served by the image failed at startup -- the image's transformers
does not know the checkpoint's ``model_type: embedding_gemma2``, so ``AutoConfig`` refuses
``config.json``.  The owner's 2026-10-10 decision: backport the architecture into the engine
environment through the plugin.  This module ships transformers 5.19.0's four ``embedding_gemma2``
modules (``fold/``, byte-identical bodies plus a header each: configuration 207 lines, modeling 933,
processing 417, video processing 381) and, at engine startup (the ``vllm.general_plugins`` entry
point), loads each under its upstream module name, registers the config class with ``AutoConfig``,
the processor and video-processor classes with ``AutoProcessor`` / ``AutoVideoProcessor`` and the two
model classes with ``AutoModel`` (the folded vLLM model's ``AutoModel.from_config`` calls resolve its
text backbone and vision tower through it).

E2 round 4: serve failed again -- ``Model architectures ['EmbeddingGemma2Model'] are not supported
for now`` -- and the wave's serve log's supported-architecture list is exactly the vLLM v0.31.0
tag's (plus the plugin's own registrations) while the nightly commit the image tag names
(``8cbd5d03006c33185f402249ff2b448efd594986``) carries both the model module and its registry entry:
the pinned image is a v0.31.0-based build, so vLLM's registry does NOT carry the model class (the r1
reading, inferred from the PR, was wrong).  The plugin therefore also ships vLLM's own
``embedding_gemma2.py`` from that commit (``fold/embedding_gemma2.py``, 1142 lines, byte-identical
plus a header) and :func:`register_embedding_gemma2` registers it lazily -- with upstream's config
hook (``.config``: the Gemma4 heterogeneous-head-dimension handling, TRITON_ATTN as the declared
default backend, the 8192 cap) -- when the running vLLM lacks them, and only then.

Two details are deliberate:

* the fold's classes are loaded under their upstream names so their relative imports
  (``from ...configuration_utils import PreTrainedConfig``) resolve inside transformers; their
  ``__module__`` is then set to this package. transformers 5.17's ``_LazyAutoMapping.register``
  refuses to map a key whose ``__module__`` starts with ``"transformers."`` (``auto_factory.py``,
  the native-config guard), and a folded class is local code -- it ships in this wheel -- so it
  must present itself as such for ``AutoProcessor.register`` to take effect. Without the
  registration ``processor_class_from_name`` finds nothing, and ``AutoProcessor.from_pretrained``
  silently falls back to the bare tokenizer, dropping the checkpoint's processor.
* an engine image whose transformers already carries the classes -- and whose registry carries the
  architecture -- is left alone: the import probe returns before anything is installed and the
  registration checks the registry first, so both steps are inert there (the removal condition in
  ``fold/``'s headers).
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "FOLD_MODULES",
    "FOLD_PACKAGE",
    "HF_MODEL_TYPE",
    "MODEL_ARCHITECTURE",
    "MODEL_QUALNAME",
    "install_fold",
    "register_embedding_gemma2",
]

#: The checkpoint's ``model_type`` (config.json), what ``AutoConfig.register`` keys on.
HF_MODEL_TYPE = "embedding_gemma2"
#: The upstream package the folded modules are installed as.
FOLD_PACKAGE = "transformers.models.embedding_gemma2"
#: The architecture the checkpoint's ``config.json`` declares and the vLLM registry resolves.
MODEL_ARCHITECTURE = "EmbeddingGemma2Model"
#: The folded model class, as vLLM's registry imports it (lazy ``"module:Class"`` string).
MODEL_QUALNAME = "rcp_ndcg_vllm.models.embedding_gemma2.fold.embedding_gemma2:EmbeddingGemma2Model"
#: The fold's own source modules, as this package ships them (the behaviour fingerprint keys exactly
#: these for a recipe that declares the ``EmbeddingGemma2Config`` registration; the model
#: registration adds the folded vLLM model and the config hook -- see
#: :data:`~rcp_ndcg_vllm.models.ARCHITECTURE_MODULES`).
FOLD_MODULES: tuple[str, ...] = (
    "rcp_ndcg_vllm.models.embedding_gemma2",
    "rcp_ndcg_vllm.models.embedding_gemma2.fold.configuration_embedding_gemma2",
    "rcp_ndcg_vllm.models.embedding_gemma2.fold.modeling_embedding_gemma2",
    "rcp_ndcg_vllm.models.embedding_gemma2.fold.processing_embedding_gemma2",
    "rcp_ndcg_vllm.models.embedding_gemma2.fold.video_processing_embedding_gemma2",
)

#: ``(upstream module name, file in fold/)``, in import order (the processing module imports the
#: config module's classes through their upstream names, and the modeling module the config's).
_UPSTREAM_MODULES: tuple[tuple[str, str], ...] = (
    ("configuration_embedding_gemma2", "configuration_embedding_gemma2.py"),
    ("processing_embedding_gemma2", "processing_embedding_gemma2.py"),
    ("video_processing_embedding_gemma2", "video_processing_embedding_gemma2.py"),
    ("modeling_embedding_gemma2", "modeling_embedding_gemma2.py"),
)


def install_fold() -> None:
    """Install the folded modules and register their classes with transformers (idempotent).

    Called by the ``embeddinggemma2-transformers-fold`` patch, which the embeddinggemma-2 recipe opts
    into; an engine serving another recipe never runs this. Inert (returns immediately) when the
    running transformers already carries ``transformers.models.embedding_gemma2``: an engine image
    that ships the classes is the fold's removal condition, and shadowing its modules would be worse
    than doing nothing.
    """
    import importlib.util
    import sys
    import types

    try:
        import transformers.models.embedding_gemma2  # noqa: F401, PLC0415 -- the probe, not a use
    except ImportError:
        pass
    else:
        return
    if FOLD_PACKAGE in sys.modules:
        return

    from transformers import AutoConfig, AutoModel, AutoProcessor, AutoVideoProcessor  # noqa: PLC0415

    package = types.ModuleType(FOLD_PACKAGE)
    package.__path__ = []
    sys.modules[FOLD_PACKAGE] = package
    here = Path(__file__).parent / "fold"
    for module_name, filename in _UPSTREAM_MODULES:
        qualified = f"{FOLD_PACKAGE}.{module_name}"
        spec = importlib.util.spec_from_file_location(qualified, here / filename)
        if spec is None or spec.loader is None:  # pragma: no cover - a missing fold file is a broken wheel
            raise RuntimeError(f"rcp-ndcg-vllm: the embeddinggemma-2 fold file {filename!r} is missing")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        setattr(package, module_name, module)

    from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import (  # noqa: PLC0415
        EmbeddingGemma2Config,
        EmbeddingGemma2TextConfig,
    )
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (  # noqa: PLC0415
        EmbeddingGemma2Model,
        EmbeddingGemma2TextModel,
    )
    from transformers.models.embedding_gemma2.processing_embedding_gemma2 import (  # noqa: PLC0415
        EmbeddingGemma2Processor,
    )
    from transformers.models.embedding_gemma2.video_processing_embedding_gemma2 import (  # noqa: PLC0415
        EmbeddingGemma2VideoProcessor,
    )

    for cls in (
        EmbeddingGemma2Config,
        EmbeddingGemma2TextConfig,
        EmbeddingGemma2Processor,
        EmbeddingGemma2VideoProcessor,
        EmbeddingGemma2Model,
        EmbeddingGemma2TextModel,
    ):
        cls.__module__ = f"{__name__}.fold"

    AutoConfig.register(HF_MODEL_TYPE, EmbeddingGemma2Config, exist_ok=True)
    AutoModel.register(EmbeddingGemma2Config, EmbeddingGemma2Model, exist_ok=True)
    AutoModel.register(EmbeddingGemma2TextConfig, EmbeddingGemma2TextModel, exist_ok=True)
    AutoProcessor.register(EmbeddingGemma2Config, EmbeddingGemma2Processor, exist_ok=True)
    AutoVideoProcessor.register(EmbeddingGemma2Config, EmbeddingGemma2VideoProcessor, exist_ok=True)


def register_embedding_gemma2() -> None:
    """Register the folded model architecture and its config hook with the running vLLM (idempotent).

    E2 round 4: the engine image the recipe pins reports vLLM 0.31.0 and its registry carries no
    ``EmbeddingGemma2Model`` (its supported-architecture list equals the v0.31.0 tag's, while the
    nightly commit the recipe names carries both the model module and the registry entry), so the
    plugin registers the folded model class lazily and, when the running vLLM's ``MODELS_CONFIG_MAP``
    lacks the architecture, upstream's config hook beside it (:mod:`.config`: the Gemma4
    heterogeneous-head-dimension handling, TRITON_ATTN as the default backend, the 8192 cap).

    An engine that already carries the architecture keeps its own class and hook: the registration
    is the fold's removal condition, so nothing is shadowed. Raises :class:`RuntimeError` when the
    running vLLM is outside the validated range (:func:`~rcp_ndcg_vllm.models.version_guard.require_vllm_version`).
    """
    from vllm.model_executor.models import ModelRegistry  # noqa: PLC0415 -- the engine's registry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP  # noqa: PLC0415

    if MODEL_ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(MODEL_ARCHITECTURE, MODEL_QUALNAME)
    if MODEL_ARCHITECTURE not in MODELS_CONFIG_MAP:
        from .config import EmbeddingGemma2ModelConfig  # noqa: PLC0415 -- imports vLLM by design

        MODELS_CONFIG_MAP[MODEL_ARCHITECTURE] = EmbeddingGemma2ModelConfig
