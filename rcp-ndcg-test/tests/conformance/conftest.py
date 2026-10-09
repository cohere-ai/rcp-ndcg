"""The conformance suite's shared wiring: the harness package on ``sys.path`` and the corpora's
tokenizer stores registered, once per session (the one wiring home: ``tests._engines.harness``).

``rcp-ndcg-vllm`` is outside the uv workspace (its own distribution), so the root tests put its ``src``
on ``sys.path``. Every tokenizer the committed corpora hash is vendored in the corpus's shared store;
registering it lets the fingerprint, the emulators and the clients load the recipes' **real** tokenizer
files offline (the one resolution in ``rcp_ndcg_test.fingerprint``).
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _harness_on_path_and_tokenizers_vendored() -> None:
    from tests._engines import harness

    harness()
