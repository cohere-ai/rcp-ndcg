"""The wave runner's observation-corpus step and its re-record-changed-only mode (stub engine, CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rcp_ndcg_vllm.fingerprint import behaviour_fingerprint
from rcp_ndcg_vllm.jobs.run_wave import run_wave
from rcp_ndcg_vllm.recipe import load_recipe

from tests.conftest import RECIPES, TOKENIZER, sample_pairs, write_pairs

VLLM_CMD = f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'}"
REFERENCE_PYTHON = sys.executable


def _wave(tmp_path: Path, **overrides: object) -> dict:
    pairs = tmp_path / "pairs"
    pairs.mkdir(exist_ok=True)
    write_pairs(pairs / "fixture-embed.jsonl", sample_pairs(documents=2))
    kwargs: dict = {
        "recipe_ids": ["fixture-embed"],
        "recipes_root": RECIPES,
        "gpus": 1,
        "out_dir": tmp_path / "wave",
        "pairs_dir": pairs,
        "reference_python": REFERENCE_PYTHON,
        "vllm_cmd": f"{VLLM_CMD} --tokenizer {TOKENIZER}",
        "port_base": 0,
    }
    kwargs.update(overrides)
    return run_wave(**kwargs)


def _checks(step: dict) -> dict[str, dict]:
    return {check["check"]: check for check in step["checks"]}


def test_the_wave_records_an_observation_corpus_at_its_keyed_path(tmp_path: Path) -> None:
    """``--record-corpus`` writes the corpus at ``observations/<engine>-<version>/<recipe>/<fingerprint>/<at>/``,
    keyed by the one behaviour fingerprint; the engine restarts between the in-process passes and the
    after-restart pass (a new run id); the corpus is checked against the equivalence stage's replies."""
    document = _wave(tmp_path, record_corpus=True)
    row = document["recipes"][0]
    step = row["steps"]["observation_corpus"]
    assert step["state"] == "passed", [check for check in step.get("checks", []) if not check["passed"]] or step
    fingerprint = behaviour_fingerprint(load_recipe(RECIPES / "fixture-embed"))
    assert row["behaviour_fingerprint"] == fingerprint == document["fingerprints"]["fixture-embed"]
    assert document["engine_versions"] == {"fixture-embed": "test-stub"}
    corpus = Path(step["corpus_dir"])
    assert corpus.parent == tmp_path / "wave" / "observations" / "vllm-test-stub" / "fixture-embed" / fingerprint
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    passes = manifest["collector"]["passes"]
    assert [entry["repetition"] for entry in passes] == ["same_process_1", "same_process_2", "after_restart"]
    assert passes[2]["server_run_id"] != passes[0]["server_run_id"]
    consistency = _checks(step)["equivalence_consistent"]
    assert consistency["passed"] is True and consistency["compared"] > 0, consistency


def test_changed_since_skips_the_unchanged_and_records_the_changed(tmp_path: Path) -> None:
    """--changed-since re-records only what moved (OBSERVATIONS-SPEC 7): an unchanged recipe is listed as
    skipped_unchanged and never served again; a changed fingerprint or a new engine version records again."""
    first = _wave(tmp_path, record_corpus=True, out_dir=tmp_path / "wave1")
    index = tmp_path / "wave1" / "wave.json"
    assert index.is_file() and first["fingerprints"] and first["engine_versions"]
    second = _wave(tmp_path, record_corpus=True, out_dir=tmp_path / "wave2", changed_since_index=index)
    assert second["skipped_unchanged"] == ["fixture-embed"]
    assert second["recipes"] == []
    document = json.loads(index.read_text(encoding="utf-8"))
    document["fingerprints"]["fixture-embed"] = "0" * 64
    index.write_text(json.dumps(document), encoding="utf-8")
    third = _wave(tmp_path, record_corpus=True, out_dir=tmp_path / "wave3", changed_since_index=index)
    assert third["skipped_unchanged"] == []
    assert third["changes"] == {"fixture-embed": ["behaviour_fingerprint"]}
    assert third["recipes"] and third["recipes"][0]["recipe"] == "fixture-embed"


def test_a_new_engine_version_records_every_recipe_again(tmp_path: Path) -> None:
    """The behaviour fingerprint does not include the engine: a recording under another engine version is a
    new corpus key, so an unchanged fingerprint under a new engine records again (and the protocol layer is
    due for that engine version)."""
    index = tmp_path / "old-wave.json"
    recipe = load_recipe(RECIPES / "fixture-embed")
    index.write_text(
        json.dumps(
            {
                "fingerprints": {"fixture-embed": behaviour_fingerprint(recipe)},
                "engine_versions": {"fixture-embed": "0.30.0"},
            }
        ),
        encoding="utf-8",
    )
    document = _wave(tmp_path, record_corpus=True, changed_since_index=index)
    assert document["skipped_unchanged"] == []
    assert document["changes"] == {"fixture-embed": ["engine_version"]}
    assert document["protocol_due"] == ["test-stub"]
