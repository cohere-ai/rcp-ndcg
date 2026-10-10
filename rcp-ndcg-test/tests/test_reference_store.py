"""The stored reference outputs (owner decision 35 item 3): the content-addressed key, the store's
integrity, the staleness naming and stage 2's reuse of an unchanged entry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rcp_ndcg_test.errors import HarnessError
from rcp_ndcg_test.reference_store import (
    load_stored,
    reference_key,
    save_stored,
    store_path,
)

ROOT = Path(__file__).resolve().parents[2]
RECIPES = ROOT / "rcp-ndcg-test" / "tests" / "fixtures" / "recipes"

BASE = {
    "reference_sha256": "a" * 64,
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "pairs_sha256": "b" * 64,
    "lock_sha256": "c" * 64,
    "device": "cpu",
    "dtype": "bfloat16",
    "model": "fixtures/DenseEmbedder",
    "mode": "embed",
    "reference": {"score_scale": "cosine"},
}


def test_the_key_changes_with_every_declared_input() -> None:
    """A changed pairs file, revision, lock, device, dtype, reference hash or mode is a new key; an
    unchanged one is the same key."""
    base = reference_key(**BASE)
    assert base.fingerprint == reference_key(**BASE).fingerprint
    for name, value in (
        ("pairs_sha256", "d" * 64),
        ("revision", "f" * 40),
        ("lock_sha256", "e" * 64),
        ("device", "cuda"),
        ("dtype", "float16"),
        ("reference_sha256", "1" * 64),
        ("mode", "score"),
    ):
        changed = reference_key(**{**BASE, name: value})
        assert changed.fingerprint != base.fingerprint, name


def test_store_round_trip_and_changed_inputs(tmp_path: Path) -> None:
    key = reference_key(**BASE)
    document = {"rows": [{"index": 0, "query_vectors": [[0.1, 0.2]]}]}
    directory = save_stored(tmp_path, key, document, family="fixture-embed")
    assert directory == store_path(tmp_path, key.fingerprint)
    loaded, changed = load_stored(tmp_path, key)
    assert loaded == document and changed == []
    # A different dtype: no entry, and the inputs that moved are named (the stale.json pattern).
    other = reference_key(**{**BASE, "dtype": "float16"})
    loaded, changed = load_stored(tmp_path, other)
    assert loaded is None and changed == ["dtype"]
    # A revision bump is named too (the closest entry is the same model and mode, not the same revision).
    bumped = reference_key(**{**BASE, "revision": "f" * 40})
    loaded, changed = load_stored(tmp_path, bumped)
    assert loaded is None and changed == ["revision"]
    # An immutable entry: saving the same key again never rewrites the output.
    save_stored(tmp_path, key, {"rows": []}, family="fixture-embed")
    loaded, _ = load_stored(tmp_path, key)
    assert loaded == document


def test_a_tampered_entry_is_refused(tmp_path: Path) -> None:
    key = reference_key(**BASE)
    directory = save_stored(tmp_path, key, {"rows": []}, family="fixture-embed")
    (directory / "output.json").write_text('{"rows": [{"index": 9}]}\n', encoding="utf-8")
    with pytest.raises(HarnessError, match="does not hash to its manifest"):
        load_stored(tmp_path, key)


def test_stage2_reuses_an_unchanged_stored_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The wave's whole point: the second stage 2 over the same pairs/environment does not run the
    reference subprocess again.  The first call runs it for real; the second replaces
    ``run_reference`` with one that raises if it is called at all."""
    import sys

    from rcp_ndcg_test.equivalence import stages
    from rcp_ndcg_vllm import load_recipe

    from tests.conftest import TOKENIZER, sample_pairs, start_stub

    recipe = load_recipe(RECIPES / "fixture-embed")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in sample_pairs()[:1]), encoding="utf-8")
    store = tmp_path / "references"
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        first = stages.stage2_scores(recipe, pairs, sys.executable, base_url=engine.base_url, reference_store=store)
        assert first["reference_outputs"]["state"] == "computed"
        assert first["passed"] is True
        calls = 0

        def must_not_run(*args: object, **kwargs: object) -> dict[str, object]:
            nonlocal calls
            calls += 1
            raise AssertionError("the stored reference output must be reused, not recomputed")

        monkeypatch.setattr(stages, "run_reference", must_not_run)
        second = stages.stage2_scores(recipe, pairs, sys.executable, base_url=engine.base_url, reference_store=store)
        assert second["reference_outputs"]["state"] == "reused"
        assert second["passed"] is True
        assert calls == 0
    finally:
        engine.stop()


def test_the_reference_hash_covers_vendored_siblings(tmp_path: Path) -> None:
    """The family reference hash hashes every ``*.py`` in the family directory, so a vendored sibling the
    entry imports (qwen3-vl-embedding's card script) moves the stored output's key."""
    from rcp_ndcg_test.equivalence import stages
    from rcp_ndcg_vllm import load_recipe

    recipe = load_recipe(RECIPES / "fixture-embed")
    family = tmp_path / "family"
    family.mkdir()
    (family / "reference.py").write_text("# the entry\n", encoding="utf-8")
    (family / "sibling.py").write_text("# v1\n", encoding="utf-8")
    first = stages._reference_file_sha256(recipe, family)
    (family / "sibling.py").write_text("# v2\n", encoding="utf-8")
    assert stages._reference_file_sha256(recipe, family) != first


def test_stage2_without_a_store_never_writes_one(tmp_path: Path) -> None:
    """No store: the reference runs and the state says so (the library's pure path)."""
    import sys

    from rcp_ndcg_test.equivalence import stages
    from rcp_ndcg_vllm import load_recipe

    from tests.conftest import TOKENIZER, sample_pairs, start_stub

    recipe = load_recipe(RECIPES / "fixture-embed")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(row) + "\n" for row in sample_pairs()[:1]), encoding="utf-8")
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        summary = stages.stage2_scores(recipe, pairs, sys.executable, base_url=engine.base_url)
        assert summary["reference_outputs"] == {"state": "computed"}
        assert summary["passed"] is True
        assert not (tmp_path / "references").exists()
    finally:
        engine.stop()
