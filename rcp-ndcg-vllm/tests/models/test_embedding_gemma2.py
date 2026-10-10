"""The embeddinggemma-2 fold: the transformers classes the digest-pinned engine nightly lacks.

E2 r1: the digest-pinned nightly's transformers does not know the checkpoint's ``model_type:
embedding_gemma2``, so serve failed. The plugin ships transformers 5.19.0's three
``embedding_gemma2`` modules and the ``embeddinggemma2-transformers-fold`` patch installs them.
This module pins the fold's declaration (its modules are the ones the fingerprint keys) offline and,
where transformers is importable, the registration's effect on the real library.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from rcp_ndcg_vllm.models import ARCHITECTURE_MODULES, LAZY_MODEL_MODULES
from rcp_ndcg_vllm.models.embedding_gemma2 import FOLD_MODULES, FOLD_PACKAGE, HF_MODEL_TYPE, install_fold
from rcp_ndcg_vllm.patches import PATCH_MODULES, PATCH_NAMES

FOLD_DIR = Path(__file__).resolve().parents[2] / "src" / "rcp_ndcg_vllm" / "models" / "embedding_gemma2"
UPSTREAM_FILES = (
    "configuration_embedding_gemma2.py",
    "processing_embedding_gemma2.py",
    "video_processing_embedding_gemma2.py",
)


def test_the_registration_key_carries_exactly_the_fold_modules() -> None:
    """One home: the architecture key the recipe declares names the registration module and the three
    folded files, and those are what the behaviour fingerprint hashes."""
    assert ARCHITECTURE_MODULES["EmbeddingGemma2Config"] == FOLD_MODULES
    assert FOLD_MODULES[0] == "rcp_ndcg_vllm.models.embedding_gemma2"
    assert FOLD_MODULES[1:] == tuple(
        f"rcp_ndcg_vllm.models.embedding_gemma2.fold.{name.removesuffix('.py')}" for name in UPSTREAM_FILES
    )
    for module in FOLD_MODULES:
        spec = importlib.util.find_spec(module)
        assert spec is not None and spec.origin is not None, module
        assert Path(spec.origin).is_file(), module


def test_the_folded_modules_are_registry_lazy() -> None:
    """The fold's files import transformers, so the no-torch harness scan must not import them."""
    for module in FOLD_MODULES[1:]:
        assert module in LAZY_MODEL_MODULES, module


def test_the_fold_names_its_upstream_source_and_removal_condition() -> None:
    """The fold's headers record the upstream version and when the fold goes away."""
    for name in UPSTREAM_FILES:
        text = (FOLD_DIR / "fold" / name).read_text(encoding="utf-8")
        assert "transformers 5.19.0" in text, name
        assert "Delete the fold when" in text, name
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


def test_install_fold_registers_the_classes_with_the_real_transformers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where transformers is importable: the fold is inert when the library already carries the
    classes, and otherwise it installs the modules under their upstream names and registers the
    config with ``AutoConfig`` and the processor with ``AutoProcessor`` (the effect the engine
    depends on; ``processor_class_from_name`` is how the checkpoint's ``processor_class`` resolves).
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
    for module in FOLD_MODULES[1:]:
        monkeypatch.delitem(sys.modules, f"{FOLD_PACKAGE}.{module.rsplit('.', 1)[1]}", raising=False)
    install_fold()
    install_fold()  # idempotent

    from transformers import AutoConfig
    from transformers.models.auto.processing_auto import processor_class_from_name

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": HF_MODEL_TYPE}), encoding="utf-8")
    config = AutoConfig.from_pretrained(checkpoint)
    assert type(config).__name__ == "EmbeddingGemma2Config"
    assert config.model_type == HF_MODEL_TYPE
    assert processor_class_from_name("EmbeddingGemma2Processor") is not None
