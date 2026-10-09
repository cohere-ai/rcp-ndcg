"""The recipe lanes' tests: the offline pins run everywhere, the Hub-backed checks are network-gated.

A recipe test is network-gated when it *needs* the public Hub, and only then. The rule lives here, in one
place: an item is network-gated when its requested fixtures include one of :data:`NETWORK_FIXTURES` (every
fixture whose construction downloads a tokenizer file or the checkpoint's own chat template), or when it
carries an explicit ``@pytest.mark.network``; an explicit ``@pytest.mark.offline`` opts an item out of the
rule. A network item carries the ``network`` marker and, when ``RCP_NDCG_NETWORK_TESTS`` is unset, skips with
the reason naming the variable (AGENTS.md's rule: a network test is marked and skips itself). Everything else
-- the contract pins, the mutants, the template and serve-argv pins, the family goldens, the root smoke test
-- runs in the offline CI jobs.

The one deliberate exception to "the fixture decides": a test that downloads inside its body (through a
module helper such as ``_tokenizer_file``, not through a fixture) carries ``@pytest.mark.network`` itself.
The gate test (``test_network_gate.py``) pins the rule and the marker's behaviour.

``--update-goldens`` rewrites ``tests/recipes/golden/`` from the current tree (the contract snapshots'
``--update-snapshots`` pattern); the golden guard's own reproducibility test pins the writer.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pytest

VARIABLE = "RCP_NDCG_NETWORK_TESTS"
CACHE_VARIABLE = "RCP_NDCG_VLLM_TOKENIZER_CACHE"

NETWORK_FIXTURES = frozenset(
    {
        # The tokenizer fixtures: each loads a Hub tokenizer.json (fetch_tokenizer's own download, or the
        # product's load_tokenizer through huggingface_hub) at a pinned revision.
        "tokenizer",
        "tokenizer_dir",
        "zerank_tokenizer",
        # The checkpoint's own template and the fixtures built on top of a tokenizer.
        "chat_template",
        "_seed_checkpoint_template",
        "recipe_cpu",
        "checkpoint",
        "snapshot",
        "recipe",
        "pairs_path",
    }
)
"""The fixtures whose construction downloads from the public Hub: an item requesting one is network-gated."""


def network_fixture_requested(fixturenames: Iterable[str]) -> bool:
    """Whether a test's requested fixtures include one that downloads from the public Hub."""
    return bool(set(fixturenames) & NETWORK_FIXTURES)


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
    """Gate the Hub-backed items: mark them ``network`` and skip them when the variable is unset.

    The explicit markers win over the fixture rule: ``offline`` opts an item out, ``network`` opts it in.
    """
    offline_run = VARIABLE not in os.environ
    here = Path(__file__).parent
    for item in items:
        if here not in Path(item.path).parents:
            continue
        if item.get_closest_marker("offline") is not None:
            continue
        explicit = item.get_closest_marker("network") is not None
        if not explicit and not network_fixture_requested(item.fixturenames):
            continue
        item.add_marker(pytest.mark.network)
        if offline_run:
            item.add_marker(pytest.mark.skip(reason=f"needs the public Hugging Face Hub; set {VARIABLE}=1 to run it"))


@pytest.fixture(autouse=True)
def _tokenizer_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The downloads land in the declared cache directory or the test's ``tmp_path`` -- never the checkout."""
    cache = os.environ.get(CACHE_VARIABLE)
    root = Path(cache) if cache else tmp_path / "tokenizer-cache"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HF_HOME", str(root))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(root))
