"""The harness process imports no torch or transformers (the reference runs as a subprocess)."""

from __future__ import annotations

import importlib
import pkgutil
import sys


def test_no_torch_or_transformers_in_the_harness_process() -> None:
    """Import every package module; the harness process stays free of torch and transformers."""
    import rcp_ndcg_vllm

    for module_info in pkgutil.walk_packages(rcp_ndcg_vllm.__path__, prefix="rcp_ndcg_vllm."):
        module = importlib.import_module(module_info.name)
        assert module is not None
    for heavy in ("torch", "transformers"):
        assert heavy not in sys.modules, f"the harness imported {heavy}"
