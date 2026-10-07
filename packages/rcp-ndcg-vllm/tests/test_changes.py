"""The change handling (OBSERVATIONS-SPEC section 7): re-record-changed-only and the behaviour diff."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from rcp_ndcg_vllm.changes import behaviour_report, changed_recipes, main
from rcp_ndcg_vllm.fingerprint import behaviour_fingerprint, fingerprint_inputs
from rcp_ndcg_vllm.recipe import load_recipe

RECIPES = Path(__file__).resolve().parent / "fixtures" / "recipes"


def _corpora_root(tmp_path: Path, index: dict) -> Path:
    """A corpora root whose manifests carry ``index``'s fingerprints over the fixture recipe."""
    for fingerprint, inputs in index.items():
        directory = tmp_path / "vllm-0.31.0" / "fixture-embed" / fingerprint
        directory.mkdir(parents=True)
        manifest = {
            "schema": "rcp-ndcg.observation/1",
            "documents": "exchanges.jsonl.gz",
            "engine": {"name": "vllm", "version": "0.31.0", "image": "vllm/vllm-openai:v0.31.0"},
            "recipe": {
                "id": "fixture-embed",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "behaviour_fingerprint": fingerprint,
                "fingerprint_inputs": inputs,
            },
        }
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        rows = [
            {
                "line_schema": 1,
                "sequence": 0,
                "method": "POST",
                "path": "/v1/embeddings",
                "request_body": {"model": "fixture-embed", "input": ["doc: hi [END]"]},
                "status": 200,
                "response_headers": {"content-type": "application/json"},
                "response": {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.5, 0.5]}],
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                },
            }
        ]
        (directory / "exchanges.jsonl.gz").write_bytes(
            gzip.compress("".join(json.dumps(row) + "\n" for row in rows).encode())
        )
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
    document = json.loads(gzip.decompress((before / "exchanges.jsonl.gz").read_bytes()))
    document["response"]["data"][0]["embedding"] = [0.5, 0.75]  # one vector component moved
    rows = json.dumps(document) + "\n"
    (after / "exchanges.jsonl.gz").write_bytes(gzip.compress(rows.encode()))
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
