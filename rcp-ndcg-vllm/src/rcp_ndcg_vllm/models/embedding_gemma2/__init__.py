"""The embeddinggemma-2 fold: the transformers classes the digest-pinned engine nightly lacks.

E2 r1: ``google/embeddinggemma-2`` served by the digest-pinned nightly failed at startup -- the
image's transformers does not know the checkpoint's ``model_type: embedding_gemma2``, so
``AutoConfig`` refuses ``config.json``, and vLLM's own ``embedding_gemma2`` model module (which the
nightly carries) cannot import the ``transformers.models.embedding_gemma2.*`` classes it needs. The
owner's 2026-10-10 decision: backport the architecture into the engine environment through the
plugin. This module ships transformers 5.19.0's three ``embedding_gemma2`` modules (``fold/``) and,
at engine startup (the ``vllm.general_plugins`` entry point), loads each under its upstream module
name, registers the config class with ``AutoConfig`` and the processor and video-processor classes
with ``AutoProcessor`` / ``AutoVideoProcessor``. vLLM's own registry maps ``EmbeddingGemma2Model``
to its ``embedding_gemma2`` module, so the plugin registers no model class.

Two details are deliberate:

* the fold's classes are loaded under their upstream names so their relative imports
  (``from ...configuration_utils import PreTrainedConfig``) resolve inside transformers; their
  ``__module__`` is then set to this package. transformers 5.17's ``_LazyAutoMapping.register``
  refuses to map a key whose ``__module__`` starts with ``"transformers."`` (``auto_factory.py``,
  the native-config guard), and a folded class is local code -- it ships in this wheel -- so it
  must present itself as such for ``AutoProcessor.register`` to take effect. Without the
  registration ``processor_class_from_name`` finds nothing, and ``AutoProcessor.from_pretrained``
  silently falls back to the bare tokenizer, dropping the checkpoint's processor.
* an engine image whose transformers already carries the classes is left alone: the import probe
  returns before anything is installed, so the fold is inert there (the removal condition in
  ``fold/``'s headers).
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["FOLD_MODULES", "FOLD_PACKAGE", "HF_MODEL_TYPE", "install_fold"]

#: The checkpoint's ``model_type`` (config.json), what ``AutoConfig.register`` keys on.
HF_MODEL_TYPE = "embedding_gemma2"
#: The upstream package the folded modules are installed as.
FOLD_PACKAGE = "transformers.models.embedding_gemma2"
#: The fold's own source modules, as this package ships them (the behaviour fingerprint keys exactly
#: these for a recipe that declares the ``EmbeddingGemma2Config`` registration).
FOLD_MODULES: tuple[str, ...] = (
    "rcp_ndcg_vllm.models.embedding_gemma2",
    "rcp_ndcg_vllm.models.embedding_gemma2.fold.configuration_embedding_gemma2",
    "rcp_ndcg_vllm.models.embedding_gemma2.fold.processing_embedding_gemma2",
    "rcp_ndcg_vllm.models.embedding_gemma2.fold.video_processing_embedding_gemma2",
)

#: ``(upstream module name, file in fold/)``, in import order (the processing module imports the
#: config module's classes through their upstream names).
_UPSTREAM_MODULES: tuple[tuple[str, str], ...] = (
    ("configuration_embedding_gemma2", "configuration_embedding_gemma2.py"),
    ("processing_embedding_gemma2", "processing_embedding_gemma2.py"),
    ("video_processing_embedding_gemma2", "video_processing_embedding_gemma2.py"),
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

    from transformers import AutoConfig, AutoProcessor, AutoVideoProcessor  # noqa: PLC0415

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
    ):
        cls.__module__ = f"{__name__}.fold"

    AutoConfig.register(HF_MODEL_TYPE, EmbeddingGemma2Config, exist_ok=True)
    AutoProcessor.register(EmbeddingGemma2Config, EmbeddingGemma2Processor, exist_ok=True)
    AutoVideoProcessor.register(EmbeddingGemma2Config, EmbeddingGemma2VideoProcessor, exist_ok=True)
