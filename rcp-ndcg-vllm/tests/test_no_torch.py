"""The package imports no torch or transformers outside the registry-lazy model modules (the reference runs as a
subprocess; the model modules are imported by vLLM itself, when the architecture resolves)."""

from __future__ import annotations

import sys

LAZY_MODEL_MODULES = (
    "rcp_ndcg_vllm.models.pplx.config",
    "rcp_ndcg_vllm.models.pplx.model",
    "rcp_ndcg_vllm.models.pplx.pooler",
    "rcp_ndcg_vllm.models.pplx.pooling_core",
    "rcp_ndcg_vllm.models.topk.config",
    "rcp_ndcg_vllm.models.topk.model",
    "rcp_ndcg_vllm.models.topk.pooling",
    "rcp_ndcg_vllm.models.topk.plugin",
    "rcp_ndcg_vllm.models.topk.weights",
)
"""The registry-lazy model modules: they import vLLM/torch by design, and only vLLM imports them (as the
``module:Class`` strings the entry point registers). Everything else -- including the entry-point callable and
the one version guard -- must import clean."""


_SCAN = """
import importlib, pkgutil, sys
import rcp_ndcg_vllm

skip = set({lazy!r})
for module_info in pkgutil.walk_packages(rcp_ndcg_vllm.__path__, prefix="rcp_ndcg_vllm."):
    if module_info.name in skip:
        continue
    importlib.import_module(module_info.name)
for heavy in ("torch", "transformers"):
    assert heavy not in sys.modules, f"the harness imported {{heavy}}"
print("clean")
"""


def test_no_torch_or_transformers_in_the_harness_process() -> None:
    """Import every package module outside the lazy model modules; the rest stays free of torch/transformers.

    Runs in a subprocess so an earlier import in this process (a test module importing a model shim)
    cannot mask the check.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", _SCAN.format(lazy=LAZY_MODEL_MODULES)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "clean"
