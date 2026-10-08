"""Shared setup for the plugin's CPU test suite.

The suite runs in three environments with different capabilities and states
its skips clearly:

- the CPU dev environment (this repo's venv: torch, numpy, no vLLM, no
  transformers) — every test except the vLLM-gated ones runs;
- the engine environment (vllm/vllm-openai image: vLLM, torch,
  transformers) — the vLLM-gated ones run too;
- the GPU wave (real weights) — the full served-vs-reference equivalence,
  outside this suite (README).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SRC = PLUGIN_ROOT / "rcp-ndcg" / "src"
PYPROJECT = PLUGIN_ROOT / "rcp-ndcg" / "pyproject.toml"

if str(SRC) not in sys.path:
    # The tests run from the checkout (CPU dev environment and engine image
    # alike) without installing the distribution; the wheel install is
    # covered separately by the freeze check.
    sys.path.insert(0, str(SRC))

#: The checkpoint's tensor names (one per line, sorted), captured from the
#: safetensors header of topk-io/topk-embed-v1-small at revision
#: e54485ebab921f2c18c4d092b3f4c40dcca26781 (618 tensors, all BF16).
CENSUS_NAMES_FILE = Path(__file__).parent / "fixtures" / "checkpoint_tensor_names.txt"


@pytest.fixture
def checkpoint_tensor_names() -> Iterator[list[str]]:
    """The 618 checkpoint tensor names (fixture, no network)."""
    names = CENSUS_NAMES_FILE.read_text(encoding="utf-8").splitlines()
    assert len(names) == 618, f"fixture drift: {len(names)} tensor names"
    yield names
