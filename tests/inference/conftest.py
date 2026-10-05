"""Shared fixtures for the inference tests: a saved test tokenizer for the explicit text budgets.

A self-hosted role config must declare ``tokenizer`` and ``max_tokens`` (the owner's explicit-budget
rule), and a client that declares a budget loads the tokenizer it names -- so the tests declare one of
the tokenizers from :mod:`tests._tokenizers`, saved to the session's temporary directory. The endpoint
helpers read the saved path from the module globals this fixture fills, so a test builds a served config
with the explicit budget by default and overrides it when the test needs its own.
"""

from __future__ import annotations

import pytest

from tests._tokenizers import save, word_tokenizer


@pytest.fixture(scope="session")
def tokenizer_json(tmp_path_factory: pytest.TempPathFactory) -> str:
    """The path of a saved ``tokenizer.json`` (the word-level test tokenizer), for configs that declare one."""
    return str(save(word_tokenizer(), tmp_path_factory.mktemp("inference-tokenizers")))


@pytest.fixture(scope="session", autouse=True)
def _default_budget(tokenizer_json: str) -> None:
    """Give the shared endpoint helpers their default explicit budget (a served role config declares one)."""
    from tests.inference import _budget, _embed

    _embed.DEFAULT_TOKENIZER = tokenizer_json
    _budget.DEFAULT_TOKENIZER = tokenizer_json
