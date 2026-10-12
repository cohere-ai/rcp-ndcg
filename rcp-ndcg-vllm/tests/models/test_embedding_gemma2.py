"""The embeddinggemma-2 fold: the classes the engine image the recipe pins lacks.

E2 r1: the image's transformers does not know the checkpoint's ``model_type: embedding_gemma2``, so
serve failed; the plugin ships transformers 5.19.0's four ``embedding_gemma2`` modules and the
``embeddinggemma2-transformers-fold`` patch installs them. E2 round 4: the image reports vLLM 0.31.0
and its registry carries no ``EmbeddingGemma2Model`` (its supported-architecture list equals the
v0.31.0 tag's, while the nightly commit the recipe names carries both the model module and the
registry entry), so the plugin also ships that model (vLLM nightly 8cbd5d03's
``embedding_gemma2.py``) and registers it with upstream's config hook. This module pins the fold's
declaration (its modules are the ones the fingerprint keys) offline and, where transformers is
importable, the registration's effect on the real library.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import json
import logging
import pickle
import sys
import types
from pathlib import Path

import pytest
from rcp_ndcg_vllm.models import ARCHITECTURE_MODULES, LAZY_MODEL_MODULES, embedding_gemma2
from rcp_ndcg_vllm.models.embedding_gemma2 import (
    FOLD_MODULES,
    FOLD_PACKAGE,
    HF_MODEL_TYPE,
    MODEL_ARCHITECTURE,
    MODEL_QUALNAME,
    install_fold,
    register_embedding_gemma2,
)
from rcp_ndcg_vllm.patches import PATCH_MODULES, PATCH_NAMES

FOLD_DIR = Path(__file__).resolve().parents[2] / "src" / "rcp_ndcg_vllm" / "models" / "embedding_gemma2"
#: The fold package itself: importing it installs the folded modules under their upstream names, before
#: any folded module body runs (the vendored vLLM model's first imports).
FOLD_PACKAGE_MODULE = "rcp_ndcg_vllm.models.embedding_gemma2.fold"
#: The folded transformers 5.19.0 modules, installed under their upstream names.
UPSTREAM_FILES = (
    "configuration_embedding_gemma2.py",
    "modeling_embedding_gemma2.py",
    "processing_embedding_gemma2.py",
    "video_processing_embedding_gemma2.py",
)
#: The folded vLLM nightly module (the model class the image's registry lacks).
VLLM_MODEL_FILE = "embedding_gemma2.py"


class _UpstreamPackageBlocker(importlib.abc.MetaPathFinder):
    """Makes ``transformers.models.embedding_gemma2`` unimportable: the pinned image's situation.

    The engine image's transformers does not carry the package, and the fold's whole point is that the
    vendored model's imports resolve anyway; a library that does carry it (5.19+) would otherwise win the
    import and the test would exercise the library's modules instead of the folded ones.
    """

    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:  # noqa: ARG002
        if fullname == FOLD_PACKAGE or fullname.startswith(f"{FOLD_PACKAGE}."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def test_the_registration_key_carries_exactly_the_fold_modules() -> None:
    """One home: the architecture key the recipe declares names the registration module and the folded
    files, and those are what the behaviour fingerprint hashes."""
    assert ARCHITECTURE_MODULES["EmbeddingGemma2Config"] == FOLD_MODULES
    assert FOLD_MODULES[:2] == ("rcp_ndcg_vllm.models.embedding_gemma2", FOLD_PACKAGE_MODULE)
    assert FOLD_MODULES[2:] == tuple(
        f"rcp_ndcg_vllm.models.embedding_gemma2.fold.{name.removesuffix('.py')}" for name in UPSTREAM_FILES
    )
    for module in FOLD_MODULES:
        spec = importlib.util.find_spec(module)
        assert spec is not None and spec.origin is not None, module
        assert Path(spec.origin).is_file(), module


def test_the_model_key_names_the_folded_model_and_the_config_hook() -> None:
    """The architecture the plugin registers (E2 round 4) is keyed by the fingerprint too: the folded
    vLLM model, upstream's config hook and every folded module its transformers imports resolve
    through."""
    model_modules = ARCHITECTURE_MODULES[MODEL_ARCHITECTURE]
    assert model_modules[0] == "rcp_ndcg_vllm.models.embedding_gemma2.config"
    assert "rcp_ndcg_vllm.models.embedding_gemma2.fold.embedding_gemma2" in model_modules
    assert model_modules[1:3] == (
        FOLD_PACKAGE_MODULE,
        "rcp_ndcg_vllm.models.embedding_gemma2.fold.embedding_gemma2",
    )
    assert set(FOLD_MODULES[1:]) <= set(model_modules)
    for module in model_modules:
        spec = importlib.util.find_spec(module)
        assert spec is not None and spec.origin is not None, module
        assert Path(spec.origin).is_file(), module
    assert MODEL_QUALNAME == "rcp_ndcg_vllm.models.embedding_gemma2.fold.embedding_gemma2:EmbeddingGemma2Model"


def test_importing_the_fold_package_installs_the_fold_before_any_folded_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The vendored vLLM model's first statements are ``from transformers.models.embedding_gemma2...``
    (E2 r1: the digest-pinned image's transformers does not carry the package), so importing the fold
    package must install the folded modules under those upstream paths *before* any folded module body
    runs -- E2 round 5's ``ModuleNotFoundError`` was the model module imported without the fold.  The
    call is pinned with a stub: the real installation executes the folded files, which import the real
    library (the transformers-gated test below exercises those imports)."""
    calls: list[str] = []
    fake_transformers = types.ModuleType("transformers")
    # ``importlib.util.find_spec`` reads the module's spec; the fold package only probes availability.
    fake_transformers.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(embedding_gemma2, "install_fold", lambda: calls.append("install_fold"))
    monkeypatch.delitem(sys.modules, FOLD_PACKAGE_MODULE, raising=False)

    importlib.import_module(FOLD_PACKAGE_MODULE)

    assert calls == ["install_fold"], "the fold package must install the fold at its own import"


def test_the_vendored_models_upstream_imports_resolve_from_the_fold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E2 round 5: the vendored vLLM model's exact transformers imports resolve from the fold alone.

    The library's own ``transformers.models.embedding_gemma2`` is blocked (the pinned image's
    transformers does not carry the package), the modules are evicted, and only the fold package is
    imported -- then the vendored model module's own import statements must resolve, to the *folded*
    classes (their ``__module__`` is the fold package, which ``install_fold`` sets)."""
    pytest.importorskip(
        "transformers",
        reason="transformers is not importable on this CPU environment; the upstream import path the "
        "fold serves is verified in the engine image's environment (E2 wave) and in the lane's scratch venv",
    )
    from transformers.utils import is_vision_available

    if not is_vision_available():
        pytest.skip(
            "the folded processor module's class body names upstream's vision-only import (inside "
            "transformers' own ``is_vision_available`` guard), so a CPU environment without the vision "
            "stack cannot install the fold; the engine image carries it"
        )
    for name in list(sys.modules):
        if name == FOLD_PACKAGE or name.startswith(f"{FOLD_PACKAGE}."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, FOLD_PACKAGE_MODULE, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_UpstreamPackageBlocker(), *sys.meta_path])

    importlib.import_module(FOLD_PACKAGE_MODULE)

    # The vendored model module's own import statements (fold/embedding_gemma2.py).
    from transformers.models.embedding_gemma2.configuration_embedding_gemma2 import (
        EmbeddingGemma2Config,
        EmbeddingGemma2TextConfig,
    )
    from transformers.models.embedding_gemma2.processing_embedding_gemma2 import EmbeddingGemma2Processor

    assert EmbeddingGemma2Config.model_type == HF_MODEL_TYPE
    assert EmbeddingGemma2TextConfig.model_type == "embedding_gemma2_text"
    assert EmbeddingGemma2Processor.__name__ == "EmbeddingGemma2Processor"
    assert EmbeddingGemma2Config.__module__ == FOLD_PACKAGE_MODULE, "the folded class, not the library's"


def test_the_folded_modules_are_registry_lazy() -> None:
    """The fold's files import transformers/vLLM, so the no-torch harness scan must not import them."""
    for module in FOLD_MODULES[1:]:
        assert module in LAZY_MODEL_MODULES, module
    for module in ARCHITECTURE_MODULES[MODEL_ARCHITECTURE]:
        assert module in LAZY_MODEL_MODULES, module


def test_the_fold_names_its_upstream_source_and_removal_condition() -> None:
    """The fold's headers record the upstream versions and when the fold goes away."""
    for name in UPSTREAM_FILES:
        text = (FOLD_DIR / "fold" / name).read_text(encoding="utf-8")
        assert "transformers 5.19.0" in text, name
        assert "Delete the fold when" in text, name
    vllm_model = (FOLD_DIR / "fold" / VLLM_MODEL_FILE).read_text(encoding="utf-8")
    assert "vLLM commit 8cbd5d03006c33185f402249ff2b448efd594986" in vllm_model
    assert "Delete the fold when" in vllm_model
    init = (FOLD_DIR / "__init__.py").read_text(encoding="utf-8")
    assert "2026-10-10" in init and "embedding_gemma2" in init


def test_the_patch_is_declared_and_names_the_fold_module() -> None:
    """The opt-in patch exists, is in ``PATCH_NAMES``, and its module implements it."""
    assert "embeddinggemma2-transformers-fold" in PATCH_NAMES
    module = PATCH_MODULES["embeddinggemma2-transformers-fold"]
    assert module == "rcp_ndcg_vllm.patches.embeddinggemma2_fold"
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin is not None
    text = Path(spec.origin).read_text(encoding="utf-8")
    assert 'PATCH_NAME = "embeddinggemma2-transformers-fold"' in text
    assert "install_fold" in text


def test_register_embedding_gemma2_registers_the_model_and_the_config_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E2 round 4: the image's registry lacks ``EmbeddingGemma2Model`` (its supported list equals the
    v0.31.0 tag's) and its ``MODELS_CONFIG_MAP`` lacks the hook, so the plugin registers both; an
    engine that already carries them is left alone (the fold's removal condition)."""
    registered: dict[str, str] = {}
    configured: dict[str, type] = {}
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__path__ = []
    fake_logger = types.ModuleType("vllm.logger")
    fake_logger.init_logger = lambda name: logging.getLogger(name)
    fake_models = types.ModuleType("vllm.model_executor.models")
    fake_config = types.ModuleType("vllm.model_executor.models.config")
    fake_models.ModelRegistry = types.SimpleNamespace(
        get_supported_archs=lambda: set(registered),
        register_model=lambda arch, qualname: registered.setdefault(arch, qualname),
    )
    fake_config.MODELS_CONFIG_MAP = configured
    fake_config.Gemma4Config = type("Gemma4Config", (), {"verify_and_update_config": staticmethod(lambda _: None)})
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.logger", fake_logger)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", types.ModuleType("vllm.model_executor"))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", fake_models)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.config", fake_config)

    register_embedding_gemma2()
    assert registered == {MODEL_ARCHITECTURE: MODEL_QUALNAME}
    assert MODEL_ARCHITECTURE in configured

    # Idempotent, and inert where the engine already carries both.
    register_embedding_gemma2()
    assert registered == {MODEL_ARCHITECTURE: MODEL_QUALNAME}
    already = {"EmbeddingGemma2Model": "elsewhere:EmbeddingGemma2Model"}
    fake_models.ModelRegistry = types.SimpleNamespace(
        get_supported_archs=lambda: set(already),
        register_model=lambda arch, qualname: already.setdefault(arch, qualname),
    )
    configured.clear()
    configured[MODEL_ARCHITECTURE] = object
    register_embedding_gemma2()
    assert already == {"EmbeddingGemma2Model": "elsewhere:EmbeddingGemma2Model"}
    assert configured[MODEL_ARCHITECTURE] is object


def test_install_fold_registers_the_classes_with_the_real_transformers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where transformers is importable: the fold is inert when the library already carries the
    classes, and otherwise it installs the modules under their upstream names and registers the
    config with ``AutoConfig``, the processors with ``AutoProcessor``/``AutoVideoProcessor`` and the
    model classes with ``AutoModel`` (the effect the engine depends on: ``processor_class_from_name``
    is how the checkpoint's ``processor_class`` resolves, and the folded vLLM model's
    ``AutoModel.from_config`` calls build its text backbone).
    """
    pytest.importorskip(
        "transformers",
        reason="transformers is not importable on this CPU environment; the fold's registration effect "
        "is verified in the engine image's environment (E2 wave) and in the lane's scratch venv",
    )

    # The inert branch: a transformers that already carries the package is left alone.
    dummy = type(sys)(FOLD_PACKAGE)
    monkeypatch.setitem(sys.modules, FOLD_PACKAGE, dummy)
    install_fold()
    assert sys.modules[FOLD_PACKAGE] is dummy

    # The real branch: no upstream package in the library (this environment's transformers does not
    # carry it -- 5.17 does not; a 5.19+ environment takes the inert branch above and this test then
    # asserts the registration is already the library's own).
    monkeypatch.delitem(sys.modules, FOLD_PACKAGE)
    for name in UPSTREAM_FILES:
        monkeypatch.delitem(sys.modules, f"{FOLD_PACKAGE}.{name.removesuffix('.py')}", raising=False)
    install_fold()
    install_fold()  # idempotent

    from transformers import AutoConfig, AutoModel
    from transformers.models.auto.processing_auto import processor_class_from_name

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": HF_MODEL_TYPE}), encoding="utf-8")
    config = AutoConfig.from_pretrained(checkpoint)
    assert type(config).__name__ == "EmbeddingGemma2Config"
    assert config.model_type == HF_MODEL_TYPE
    assert processor_class_from_name("EmbeddingGemma2Processor") is not None
    # AutoModel resolves the folded model classes (the folded vLLM model's from_config calls).
    assert AutoModel._model_mapping[type(config)] is not None
    text_config = config.text_config
    assert AutoModel._model_mapping[type(text_config)] is not None

    # The folded classes declare the fold package as their ``__module__`` (transformers'
    # ``_LazyAutoMapping.register`` refuses a key whose module starts with ``transformers.``), so that
    # location must carry them: vLLM pickles the checkpoint's config when it spawns the engine core,
    # and a class whose declared module has no such attribute cannot be pickled (E2 round 5).
    folded = (
        type(config),
        type(text_config),
        AutoModel._model_mapping[type(config)],
        AutoModel._model_mapping[type(text_config)],
    )
    for cls in folded:
        module = importlib.import_module(cls.__module__)
        assert getattr(module, cls.__name__) is cls, cls.__name__
    assert pickle.loads(pickle.dumps(config)).model_type == HF_MODEL_TYPE
