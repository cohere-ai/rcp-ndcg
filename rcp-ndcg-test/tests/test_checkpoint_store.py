"""The checkpoint store (the corpora's vendored checkpoint files, ``rcp_ndcg_test.equivalence.checkpoint``):
the file a committed replay reads resolves offline through it, a corrupt entry is refused by name, and a
registered store answers before the Hub."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest
from rcp_ndcg_test.errors import HarnessError

from tests._engines import ROOT

CORPORA_CHECKPOINTS = ROOT / "corpora" / "vllm-0.31.0" / "_checkpoints"
"""The corpora's vendored checkpoint files (registered beside the tokenizer store by ``tests._engines``)."""

REPLAY_RECIPE = "qwen3-vl-embedding-2b"
"""The committed corpus whose replay renders the checkpoint's own chat template: the recipe serves none
(``serve.chat_template: null``) and its records are chat-shaped, so the emulator reads the checkpoint's
``chat_template.jinja`` at the pinned revision."""

REPLAY_TEMPLATE = "chat_template.jinja"
REPLAY_TEMPLATE_SHA256 = "a47e6afb389f86f45be7810f17d2686fd42b2bec7ba6e6958abf85845af258c5"
"""The vendored template's SHA-256: the recipe family's pinned pin (the hash the recipe's own test pins)."""


@pytest.fixture
def _alone_in_the_store_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The test's registrations alone: the module's store list is replaced and restored by the monkeypatch,
    so a fixture-keyed entry here never answers for another test's recipe."""
    from rcp_ndcg_test.equivalence import checkpoint

    monkeypatch.setattr(checkpoint, "_CHECKPOINT_STORES", [])


def test_the_committed_replays_checkpoint_file_resolves_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no Hub cache and no Hub to ask (the offline CI job's shape), the file the committed replay reads
    resolves from the corpora's checkpoint store, and the vendored copy is the pinned template's bytes."""
    from tests._engines import load_recipe
    from tests.conftest import hub_cache

    hub_cache(tmp_path, monkeypatch, "fixtures/DenseEmbedder", "0" * 40, {})
    recipe = load_recipe(REPLAY_RECIPE)  # registers the corpora's stores (tokenizer beside checkpoint)
    from rcp_ndcg_test.equivalence.checkpoint import checkpoint_file, checkpoint_text

    found = checkpoint_file(recipe, REPLAY_TEMPLATE)
    assert found is not None
    assert found == CORPORA_CHECKPOINTS / "a47e6afb389f86f45be7810f17d2686fd42b2bec7ba6e6958abf85845af258c5.jinja"
    text = checkpoint_text(found)
    assert text != ""
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == REPLAY_TEMPLATE_SHA256


def test_a_corrupt_store_entry_is_refused_by_name(tmp_path: Path, _alone_in_the_store_registry: None) -> None:
    """A vendored copy whose bytes do not hash to the index's sha256 is refused by name: never served, and
    never silently fallen through to the Hub; so is an entry whose vendored file is missing."""
    from rcp_ndcg_test.equivalence.checkpoint import stored_checkpoint_file, use_checkpoint_store

    key = "fixtures/DenseEmbedder@0123456789abcdef0123456789abcdef01234567:chat_template.jinja"
    store = tmp_path / "store"
    store.mkdir()
    (store / "template.jinja").write_text("not the vendored bytes", encoding="utf-8")
    (store / "index.json").write_text(
        json.dumps({key: {"file": "template.jinja", "sha256": hashlib.sha256(b"the pinned bytes").hexdigest()}}),
        encoding="utf-8",
    )
    use_checkpoint_store(store)
    with pytest.raises(HarnessError) as error:
        stored_checkpoint_file("fixtures/DenseEmbedder", "0123456789abcdef0123456789abcdef01234567", REPLAY_TEMPLATE)
    assert "template.jinja" in str(error.value) and "corrupt or stale" in str(error.value)

    (store / "template.jinja").unlink()
    with pytest.raises(HarnessError) as missing:
        stored_checkpoint_file("fixtures/DenseEmbedder", "0123456789abcdef0123456789abcdef01234567", REPLAY_TEMPLATE)
    assert "template.jinja" in str(missing.value)


def test_the_store_answers_before_the_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _alone_in_the_store_registry: None
) -> None:
    """A registered store's entry answers even with an empty Hub cache: the store is consulted before
    ``huggingface_hub``, so a vendored file needs neither the cache nor a Hub request. The entry here is
    ``.gz``: the index hashes the content, and the read decompresses."""
    from rcp_ndcg_vllm import load_recipe

    from tests.conftest import RECIPES, hub_cache

    hub_cache(tmp_path, monkeypatch, "fixtures/DenseEmbedder", "0123456789abcdef0123456789abcdef01234567", {})
    template = "{%- for message in messages -%}{{ message.content }}{%- endfor -%}"
    store = tmp_path / "store"
    store.mkdir()
    (store / "template.jinja.gz").write_bytes(gzip.compress(template.encode("utf-8")))
    (store / "index.json").write_text(
        json.dumps(
            {
                "fixtures/DenseEmbedder@0123456789abcdef0123456789abcdef01234567:chat_template.jinja": {
                    "file": "template.jinja.gz",
                    "sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    from rcp_ndcg_test.equivalence.checkpoint import checkpoint_file, checkpoint_text, use_checkpoint_store

    use_checkpoint_store(store)
    recipe = load_recipe(RECIPES / "fixture-embed")
    found = checkpoint_file(recipe, REPLAY_TEMPLATE)
    assert found == store.resolve() / "template.jinja.gz"
    assert checkpoint_text(found) == template
