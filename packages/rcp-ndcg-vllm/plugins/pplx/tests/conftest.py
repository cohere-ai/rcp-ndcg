"""Fixtures for the plugin tests: shared tiny-config tensors and the token contract.

The tests run in two regimes:

- **Without vLLM** (the CPU lane machine): the pure-torch core of
  :mod:`rcp_vllm_pplx.pooling_core` and everything that does not import vLLM
  (the version guard, the entry-point metadata, the registration against a stub
  registry, the wheel's freeze behaviour). This is the regime that proves the pooling
  fidelity on a tiny random configuration of the model's architecture.
- **With vLLM importable** (the engine image): the wired pooler against a real
  ``PoolingMetadata``, the registration against the real registry. These tests skip
  with a clear reason where vLLM cannot import on CPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

try:
    import vllm  # noqa: F401, PLC0415

    HAS_VLLM = True
except Exception:  # noqa: BLE001  (vllm can fail to import for many environment reasons)
    HAS_VLLM = False

if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def skip_without_vllm() -> None:
    """Skip a test that needs vLLM importable, naming the reason."""
    if not HAS_VLLM:
        pytest.skip(
            "vLLM is not importable in this environment (the rcp-ndcg dev venv carries "
            "torch only, by design); the wired-pooler and model-class checks run on the "
            "GPU wave, whose engine image has vLLM 0.31.0"
        )
