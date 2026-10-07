"""The conformance suite's shared wiring: the harness package on ``sys.path``, the tokenizer stores.

``rcp-ndcg-vllm`` is outside the uv workspace (its own distribution), so the root tests put its ``src``
on ``sys.path`` the way ``tests/docs/test_docs_recipes.py`` does. Every tokenizer the committed corpora
hash is vendored in the corpus's shared store; registering it lets the fingerprint, the emulators and the
clients load the recipes' **real** tokenizer files offline (the one resolution in
``rcp_ndcg_vllm.fingerprint``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the repo root, for tests._engines

ROOT = Path(__file__).resolve().parents[2]
ENGINES = ROOT / "tests" / "contract" / "engines"
RECIPES = ROOT / "packages" / "rcp-ndcg-vllm" / "recipes"
_VLLM_SRC = str(ROOT / "packages" / "rcp-ndcg-vllm" / "src")


@pytest.fixture(scope="session", autouse=True)
def _harness_on_path_and_tokenizers_vendored() -> None:
    """Put ``rcp_ndcg_vllm`` on the path and register the corpora's tokenizer stores, once per session
    (the one wiring home: ``tests._engines.harness``)."""
    from tests._engines import harness

    harness()


def harness_imports() -> None:
    """Import ``rcp_ndcg_vllm`` and return; tests call it after fixtures ran (a plain import at module
    level would run before the session fixture puts the package on the path)."""
    return None
