"""The wave runner's observation-corpus step and its re-record-changed-only mode (stub engine, CPU)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rcp_ndcg_vllm.jobs.run_wave import run_wave

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


def test_the_wave_records_an_observation_corpus_per_recipe(tmp_path: Path) -> None:
    """``--record-corpus`` writes <out>/<id>/corpus/ (records, non-determinism, hash-chained manifest),
    records the recipe's behaviour fingerprint in the wave document, and the engine restarts between
    the in-process passes and the after-restart pass."""
    document = _wave(tmp_path, record_corpus=True)
    row = document["recipes"][0]
    step = row["steps"]["observation_corpus"]
    assert step["state"] == "passed", step
    corpus = tmp_path / "wave" / "fixture-embed" / "corpus"
    assert (corpus / "records.jsonl").is_file() and (corpus / "nondeterminism.json").is_file()
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["collector"]["repetitions"] == ["same_process", "same_process", "after_restart"]
    assert document["fingerprints"]["fixture-embed"] == row["behaviour_fingerprint"]
    assert row["state"] in ("verified", "failed"), row  # the corpus step itself passed either way


def test_changed_since_skips_the_unchanged_and_records_the_changed(tmp_path: Path) -> None:
    """--changed-since re-records only what moved (OBSERVATIONS-SPEC 7): an unchanged recipe is listed
    as skipped_unchanged and never served again; a changed fingerprint records again."""
    first = _wave(tmp_path, record_corpus=True, out_dir=tmp_path / "wave1")
    index = tmp_path / "wave1" / "wave.json"
    assert index.is_file() and first["fingerprints"]
    second = _wave(tmp_path, record_corpus=True, out_dir=tmp_path / "wave2", changed_since_index=index)
    assert second["skipped_unchanged"] == ["fixture-embed"]
    assert second["recipes"] == []
    # A stale fingerprint in the index means the recipe moved and records again.
    document = json.loads(index.read_text(encoding="utf-8"))
    document["fingerprints"]["fixture-embed"] = "0" * 64
    index.write_text(json.dumps(document), encoding="utf-8")
    third = _wave(tmp_path, record_corpus=True, out_dir=tmp_path / "wave3", changed_since_index=index)
    assert third["skipped_unchanged"] == []
    assert third["recipes"] and third["recipes"][0]["recipe"] == "fixture-embed"
