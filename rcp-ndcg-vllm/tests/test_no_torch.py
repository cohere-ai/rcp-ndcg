"""The package imports no torch or transformers outside the registry-lazy model modules (the reference runs as a
subprocess; the model modules are imported by vLLM itself, when the architecture resolves)."""

from __future__ import annotations

import sys

from rcp_ndcg_vllm.models import LAZY_MODEL_MODULES

"""The registry-lazy model modules (one home: ``rcp_ndcg_vllm.models``): they import vLLM/torch by design,
and only vLLM imports them (as the ``module:Class`` strings the entry point registers). Everything else --
including the entry-point callable and the one version guard -- must import clean."""


_SCAN = """
import importlib, pkgutil, sys
import rcp_ndcg_vllm

skip = set({lazy!r})

def walk(path, prefix):
    # ``pkgutil.walk_packages`` imports every subpackage to walk into it, which would import a
    # registry-lazy *package* (the embeddinggemma-2 fold: its import installs the folded modules under
    # their upstream names) and defeat the skip list; this walk honours it.
    for module_info in pkgutil.iter_modules(path, prefix):
        yield module_info
        if module_info.ispkg and module_info.name not in skip:
            module = importlib.import_module(module_info.name)
            yield from walk(module.__path__, module_info.name + ".")

for module_info in walk(rcp_ndcg_vllm.__path__, "rcp_ndcg_vllm."):
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
    import os
    import subprocess
    from pathlib import Path

    import rcp_ndcg_vllm

    # The child imports the same package this process imported, installed or not (a source checkout on
    # pytest's path is not inherited by a subprocess).
    source_root = str(Path(rcp_ndcg_vllm.__file__).resolve().parents[1])
    inherited = os.environ.get("PYTHONPATH")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([source_root, inherited] if inherited else [source_root])}
    result = subprocess.run(
        [sys.executable, "-c", _SCAN.format(lazy=LAZY_MODEL_MODULES)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "clean"
