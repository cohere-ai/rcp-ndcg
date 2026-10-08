"""The change handling (OBSERVATIONS-SPEC section 7): re-record-changed-only and the behaviour diff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from rcp_ndcg_test.changes import behaviour_report, changed_recipes, main
from rcp_ndcg_test.corpus import manifest_digest
from rcp_ndcg_test.fingerprint import behaviour_fingerprint, fingerprint_inputs
from rcp_ndcg_vllm.recipe import load_recipe

RECIPES = Path(__file__).resolve().parent / "fixtures" / "recipes"


def _record(embedding: list[float]) -> dict:
    """One record in the observation-corpus format (``rcp_ndcg_test.corpus``, ``RECORD_SCHEMA`` 1)."""
    request = {"model": "fixture-embed", "input": ["doc: hi [END]"]}
    reply = {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": embedding}],
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }
    return {
        "record_schema": 1,
        "exchange_id": "a" * 64,
        "sequence": 0,
        "repetition": "same_process_1",
        "request": {
            "method": "POST",
            "path": "/v1/embeddings",
            "headers": {},
            "body_raw": json.dumps(request),
            "body_parsed": request,
        },
        "response": {
            "status": 200,
            "headers": {"content-type": "application/json"},
            "body_raw": json.dumps(reply),
            "body_parsed": reply,
        },
        "inputs": {},
    }


def _corpora_root(tmp_path: Path, index: dict) -> Path:
    """A corpora root whose manifests carry ``index``'s fingerprints over the fixture recipe."""
    for fingerprint, inputs in index.items():
        directory = tmp_path / "vllm-0.31.0" / "fixture-embed" / fingerprint
        directory.mkdir(parents=True)
        manifest = {
            "schema": "rcp-ndcg.observation-corpus/1",
            "engine": {"name": "vllm", "version": "0.31.0", "image": "vllm/vllm-openai:v0.31.0"},
            "model": {"id": "fixtures/DenseEmbedder", "revision": "0123456789abcdef0123456789abcdef01234567"},
            "recipe": {"id": "fixture-embed", "behaviour_fingerprint": fingerprint, "fingerprint_inputs": inputs},
        }
        records = directory / "records.jsonl"
        records.write_text(json.dumps(_record([0.5, 0.5])) + "\n", encoding="utf-8")
        manifest["integrity"] = {
            "files": {"records.jsonl": {"sha256": hashlib.sha256(records.read_bytes()).hexdigest()}},
            "records_count": 1,
        }
        manifest["integrity"]["manifest_sha256"] = manifest_digest(manifest)
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path / "vllm-0.31.0"


def test_the_changed_selection_names_the_recipes_and_the_inputs(tmp_path: Path) -> None:
    recipe = load_recipe(RECIPES / "fixture-embed")
    inputs = dict(fingerprint_inputs(recipe))
    fingerprint = behaviour_fingerprint(recipe)

    # one corpus exactly as recorded: unchanged
    root = _corpora_root(tmp_path / "same", {fingerprint: inputs})
    states = changed_recipes(RECIPES, root)
    assert states["fixture-embed"]["state"] == "unchanged"
    assert states["fixture-embed"]["behaviour_fingerprint"] == fingerprint

    # a corpus of an older fingerprint whose template input differs: changed, with the input named
    moved = dict(inputs)
    moved["client.template"] = "{}"
    root = _corpora_root(tmp_path / "moved", {"0" * 64: moved})
    states = changed_recipes(RECIPES, root)
    assert states["fixture-embed"]["state"] == "changed"
    assert states["fixture-embed"]["changed_inputs"] == ["client.template"]

    # no corpus at all: new
    states = changed_recipes(RECIPES, tmp_path / "empty")
    assert states["fixture-embed"]["state"] == "new"
    assert states["fixture-embed"]["changed_inputs"] == []


def test_the_behaviour_diff_reports_per_input_deltas(tmp_path: Path) -> None:
    root = _corpora_root(tmp_path / "before", {"0" * 64: {}})
    before = root / "fixture-embed" / ("0" * 64)
    after = before.parent / ("1" * 64)
    after.mkdir()
    (after / "records.jsonl").write_text(json.dumps(_record([0.5, 0.75])) + "\n", encoding="utf-8")  # one moved
    manifest = json.loads((before / "manifest.json").read_text(encoding="utf-8"))
    manifest["recipe"]["behaviour_fingerprint"] = "1" * 64
    (after / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = behaviour_report(before, after)
    assert report["schema"] == "rcp-ndcg.behaviour-diff/1"
    assert report["summary"]["inputs"] == 1 and report["summary"]["changed"] == 1
    (row,) = report["inputs"]
    assert row["numeric_deltas"] == {".data[0].embedding[1]": 0.25}
    assert row["protocol_changed"] == []
    assert report["fingerprints"] == {"before": "0" * 64, "after": "1" * 64}


def test_the_changes_command_prints_json(tmp_path: Path, capsys) -> None:
    recipe = load_recipe(RECIPES / "fixture-embed")
    fingerprint = behaviour_fingerprint(recipe)
    root = _corpora_root(tmp_path, {fingerprint: dict(fingerprint_inputs(recipe))})
    assert main(["changed", "--recipes-root", str(RECIPES), "--corpora-root", str(root)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["fixture-embed"]["state"] == "unchanged"


def test_corpora_resolve_by_their_manifests_and_a_moved_fingerprint_names_its_inputs(tmp_path: Path) -> None:
    """B2: the corpus of a recipe is found by scanning manifests (a directory name is never derived from
    the recomputed fingerprint), and a stale recipe fails naming the inputs that moved."""
    import pytest
    from rcp_ndcg_test.changes import StaleCorpusError, recipe_state, resolve_corpus

    recipe = load_recipe(RECIPES / "fixture-embed")
    inputs = dict(fingerprint_inputs(recipe))
    fingerprint = behaviour_fingerprint(recipe)
    root = _corpora_root(tmp_path / "same", {fingerprint: inputs})
    renamed = root / "fixture-embed" / "any-directory-name"
    (root / "fixture-embed" / fingerprint).rename(renamed)
    assert resolve_corpus(recipe, root) == renamed

    moved = dict(inputs)
    moved["client.max_tokens"] = "1"
    root = _corpora_root(tmp_path / "moved", {"0" * 64: moved})
    state = recipe_state(recipe, root)
    assert state["state"] == "changed" and state["changed_by_corpus"] == {"0" * 64: ["client.max_tokens"]}
    with pytest.raises(StaleCorpusError) as error:
        resolve_corpus(recipe, root)
    assert "client.max_tokens" in str(error.value)
    with pytest.raises(StaleCorpusError) as error:
        resolve_corpus(recipe, tmp_path / "empty")
    assert "no committed corpus" in str(error.value)


def test_a_corpus_whose_hashes_do_not_hold_is_refused(tmp_path: Path) -> None:
    """The change handling reads every corpus through the one reader (``rcp_ndcg_test.corpus``) and its
    integrity check: a manifest edited without its digest is refused, naming the corpus -- never compared."""
    from rcp_ndcg_test.changes import recipe_state
    from rcp_ndcg_test.errors import HarnessError

    recipe = load_recipe(RECIPES / "fixture-embed")
    fingerprint = behaviour_fingerprint(recipe)
    root = _corpora_root(tmp_path, {fingerprint: dict(fingerprint_inputs(recipe))})
    assert recipe_state(recipe, root)["state"] == "unchanged"
    path = root / "fixture-embed" / fingerprint / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["recipe"]["fingerprint_inputs"]["model"] = "edited"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(HarnessError, match="manifest_sha256 mismatch"):
        recipe_state(recipe, root)
