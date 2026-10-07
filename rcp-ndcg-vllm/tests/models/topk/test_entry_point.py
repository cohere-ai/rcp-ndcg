"""Entry-point registration: the ``vllm.general_plugins`` declaration and, where
vLLM is importable, the actual registry effect."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def entry_points() -> Iterator[dict[str, str]]:
    """The ``vllm.general_plugins`` entry points declared by pyproject.toml."""
    with (Path(__file__).resolve().parents[3] / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    yield metadata["project"]["entry-points"]["vllm.general_plugins"]


def test_entry_point_is_declared(entry_points: dict[str, str]) -> None:
    """The distribution registers exactly one vllm.general_plugins callable."""
    assert entry_points == {"rcp-ndcg-vllm": "rcp_ndcg_vllm.models:register"}, entry_points


def test_entry_point_callable_imports_without_vllm_or_torch() -> None:
    """The entry-point module (and everything it imports) needs neither vLLM
    nor torch, so plugin loading cannot fail on an engine that lacks either
    and the CPU suite can import it.  Runs in a subprocess so an earlier
    import in this process cannot mask the check."""
    src = Path(__file__).resolve().parents[3] / "src"
    code = (
        f"import sys; sys.path.insert(0, {str(src)!r}); "
        "import rcp_ndcg_vllm.models.topk, rcp_ndcg_vllm.models.topk.plugin, "
        "rcp_ndcg_vllm.models.version_guard, rcp_ndcg_vllm.models.topk.weights; "
        "assert 'vllm' not in sys.modules, 'vllm imported'; "
        "assert 'torch' not in sys.modules, 'torch imported'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_register_registers_the_architecture() -> None:
    """Calling the entry point registers TopkEmbedModel in the model registry
    (the effect ``vllm serve`` depends on).  Skipped where vLLM cannot be
    imported (the CPU dev environment has no vLLM)."""
    pytest.importorskip(
        "vllm",
        reason=(
            "vLLM is not importable on this CPU environment; the registry "
            "effect of the entry point is verified on the engine image "
            "(GPU wave T0)"
        ),
    )
    from rcp_ndcg_vllm.models.topk.plugin import MODEL_ARCHITECTURE, register
    from vllm.model_executor.models import ModelRegistry

    register()
    assert MODEL_ARCHITECTURE in ModelRegistry.get_supported_archs()
    # The loader may run the callable once per process; a second call must be
    # tolerated (vllm/plugins/__init__.py load_general_plugins).
    register()
    assert MODEL_ARCHITECTURE in ModelRegistry.get_supported_archs()
