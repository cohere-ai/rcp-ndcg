"""The plugin's code declaration: which modules each registered architecture's engine loads.

The behaviour fingerprint (``rcp-fp/4``) keys the plugin code a recipe's engine runs by hashing exactly
these modules.  One home: the plugin package declares the mapping beside the registrations it mirrors, and
this suite pins it against those registrations and against the lazy-import contract.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from rcp_ndcg_vllm.models import ARCHITECTURE_MODULES, LAZY_MODEL_MODULES, PLUGIN_ENGINE_MODULES
from rcp_ndcg_vllm.models.pplx import LATE_ARCHITECTURE, PLUGIN_ARCHITECTURE
from rcp_ndcg_vllm.models.topk.plugin import MODEL_ARCHITECTURE
from rcp_ndcg_vllm.patches import PATCH_MODULES, PATCH_NAMES


def _module_file(module: str) -> Path:
    """The module's source file, resolved without importing it (the model modules import vLLM/torch)."""
    spec = importlib.util.find_spec(module)
    assert spec is not None and spec.origin is not None, module
    return Path(spec.origin)


def test_the_architecture_mapping_names_every_registered_architecture() -> None:
    """One key per registered model architecture (plus the config-only registration), and no invented one:
    the registration constants are the truth."""
    assert set(ARCHITECTURE_MODULES) == {PLUGIN_ARCHITECTURE, LATE_ARCHITECTURE, MODEL_ARCHITECTURE, "PplxV1Config"}


def test_every_architecture_module_is_a_real_source_file() -> None:
    for architecture, modules in ARCHITECTURE_MODULES.items():
        assert modules, architecture
        for module in modules:
            assert _module_file(module).is_file(), f"{architecture}: {module}"


def test_every_architecture_module_is_a_registry_lazy_module() -> None:
    """The architecture modules import vLLM/torch by design: they must be in the lazy list, or the
    no-torch harness scan would import them."""
    lazy = set(LAZY_MODEL_MODULES)
    for architecture, modules in ARCHITECTURE_MODULES.items():
        missing = sorted(set(modules) - lazy)
        assert not missing, f"{architecture}: not in LAZY_MODEL_MODULES: {missing}"


def test_the_shared_engine_modules_resolve_and_are_not_lazy() -> None:
    """The entry point, the guard, the registrations and the patch applier import clean (the no-torch scan
    imports them); they run for every plugin recipe."""
    assert PLUGIN_ENGINE_MODULES == (
        "rcp_ndcg_vllm.models",
        "rcp_ndcg_vllm.models.version_guard",
        "rcp_ndcg_vllm.models.pplx",
        "rcp_ndcg_vllm.models.pplx.config",
        "rcp_ndcg_vllm.models.pplx.hf_config",
        "rcp_ndcg_vllm.models.topk",
        "rcp_ndcg_vllm.models.topk.config",
        "rcp_ndcg_vllm.models.topk.plugin",
        "rcp_ndcg_vllm.patches",
    )
    for module in PLUGIN_ENGINE_MODULES:
        assert _module_file(module).is_file(), module


def test_the_patch_mapping_names_every_patch_and_resolves_it() -> None:
    """``PATCH_MODULES`` is the one home of the patch-name -> module mapping; ``PATCH_NAMES`` follows it."""
    assert PATCH_NAMES == tuple(PATCH_MODULES)
    for name, module in PATCH_MODULES.items():
        path = _module_file(module)
        assert path.is_file(), name
        assert f'PATCH_NAME = "{name}"' in path.read_text(encoding="utf-8"), name


def test_every_patch_module_declares_its_apply() -> None:
    for module in PATCH_MODULES.values():
        assert "def apply(" in _module_file(module).read_text(encoding="utf-8"), module
