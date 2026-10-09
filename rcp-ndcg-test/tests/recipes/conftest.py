"""The recipe lanes' tests: real models, real tokenizers, real engines -- network-gated.

Every test in this directory downloads a real tokenizer (or reaches a real engine) at test time, with one
deliberate exception: ``test_family_goldens.py`` is offline by construction (package data, the template
file's bytes and the pure-Python fingerprint; its tokenizer hash comes from the vendored corpora store or
the golden itself), so the conftest leaves it unmarked and it runs in every CI job.  AGENTS.md allows
network in tests only for tests marked ``pytest.mark.network`` that skip themselves unless
``RCP_NDCG_NETWORK_TESTS=1`` is set: the conftest below marks every other test collected here ``network``
and skips it (with the reason naming the variable) when the variable is unset, so an offline run stays
green.  The downloads go to ``RCP_NDCG_VLLM_TOKENIZER_CACHE`` when it is set, else to the test's
``tmp_path`` -- never into the checkout (``HF_HOME``/``TRANSFORMERS_CACHE`` are pointed there for the
duration of each test too).

``--update-goldens`` rewrites ``tests/recipes/golden/`` from the current tree (the contract snapshots'
``--update-snapshots`` pattern); the golden guard's own reproducibility test pins the writer.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

VARIABLE = "RCP_NDCG_NETWORK_TESTS"
CACHE_VARIABLE = "RCP_NDCG_VLLM_TOKENIZER_CACHE"

OFFLINE_MODULES = frozenset({"test_family_goldens.py"})
"""Test modules that never touch the network: they stay unmarked and run offline."""


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="Rewrite tests/recipes/golden/ from the current tree (review the diff, then commit).",
    )


@pytest.fixture
def update_goldens(request: pytest.FixtureRequest) -> bool:
    """Whether ``--update-goldens`` was given (the golden guard rewrites instead of comparing)."""
    return bool(request.config.getoption("--update-goldens"))


def pytest_collection_modifyitems(config: object, items: list) -> None:  # noqa: ARG001 - pytest API
    """Mark every test collected under ``tests/recipes/`` as ``network`` and skip it when the variable is
    unset -- except the offline golden guard, which runs everywhere."""
    offline = VARIABLE not in os.environ
    here = Path(__file__).parent
    for item in items:
        if here not in Path(item.fspath).parents:
            continue
        if Path(item.fspath).name in OFFLINE_MODULES:
            continue
        item.add_marker(pytest.mark.network)
        if offline:
            item.add_marker(
                pytest.mark.skip(reason=f"recipe tests download a real tokenizer; set {VARIABLE}=1 to run them")
            )


@pytest.fixture(autouse=True)
def _tokenizer_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The downloads land in the declared cache directory or the test's ``tmp_path`` -- never the checkout."""
    cache = os.environ.get(CACHE_VARIABLE)
    root = Path(cache) if cache else tmp_path / "tokenizer-cache"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HF_HOME", str(root))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(root))
