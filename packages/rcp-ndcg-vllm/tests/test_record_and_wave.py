"""The recorder and the wave runner: the product's wire path observed through the recording transport."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from rcp_ndcg_vllm import load_recipe
from rcp_ndcg_vllm.equivalence.metrics import stage3_metrics
from rcp_ndcg_vllm.jobs.run_wave import run_wave
from rcp_ndcg_vllm.record import record

from tests.conftest import RECIPES, TOKENIZER, sample_pairs, start_stub

REFERENCE_PYTHON = sys.executable
VLLM_CMD = f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'}"
REFERENCE_PY = sys.executable


def test_record_writes_exchanges_per_route(tmp_path: Path) -> None:
    """The product's adapter sends the fixed request set through the recording transport."""

    recipe = load_recipe(RECIPES / "fixture-embed")
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        written = record(recipe, engine.base_url, tmp_path)
    finally:
        engine.stop()
    assert len(written) >= 3  # models, embeddings, over-length, unknown-field
    for path in written:
        document = json.loads(path.read_text(encoding="utf-8"))
        assert "http://engine" in document["route"]
        assert "127.0.0.1" not in json.dumps(document)


def test_record_exchanges_carry_the_product_shape(tmp_path: Path) -> None:
    """The recorded requests match the product's adapter (the wire path, not a second path)."""

    recipe = load_recipe(RECIPES / "fixture-embed")
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        written = record(recipe, engine.base_url, tmp_path)
    finally:
        engine.stop()
    by_name = {path.name: json.loads(path.read_text(encoding="utf-8")) for path in written}
    embeddings = [document for name, document in by_name.items() if "embeddings" in name]
    assert embeddings
    body = embeddings[0]["body"]
    assert "embedding" in body["data"][0]


def test_stage3_metrics_compares_served_against_reference(tmp_path: Path) -> None:
    """Stage 3 shells out to `rcp-ndcg eval score`; identical rankings give delta 0 and a pass."""
    pytest.importorskip("rcp_ndcg")
    from rcp_ndcg_vllm.equivalence.gates import ResolvedGates

    from rcp_ndcg.data import Rankings

    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    for system in ("served", "reference"):
        Rankings.from_orders({"q1": ["a", "b", "c"], "q2": ["b", "a", "c"]}, system=system).save(
            rankings_dir / f"toy.{system}.jsonl"
        )
    dataset = [
        {
            "query_id": "q1",
            "query": "q1",
            "doc_ids": ["a", "b", "c"],
            "docs": ["A", "B", "C"],
            "qrels": {"a": 1.0, "b": 0.5, "c": 0.1},
        },  # fmt: skip
        {
            "query_id": "q2",
            "query": "q2",
            "doc_ids": ["a", "b", "c"],
            "docs": ["A", "B", "C"],
            "qrels": {"a": 0.2, "b": 0.9, "c": 0.0},
        },  # fmt: skip
    ]
    (rankings_dir / "toy.dataset.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dataset), encoding="utf-8"
    )
    gates = ResolvedGates(
        prob_p99_abs=0.02, prob_max_abs=0.05, logit_rel_abs=0.05, cos_max_abs=0.01,
        vec_min_cosine=0.999, tau_min=0.98, metrics_max_abs=2e-3, embed_dtype="float16",
    )  # fmt: skip
    document = stage3_metrics(rankings_dir, gates)
    assert document["passed"] is True


def test_wave_runs_a_recipe_end_to_end(tmp_path: Path) -> None:
    """One recipe on one slot with the stub engine: serve, smoke, equivalence."""
    out = tmp_path / "wave"
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    (pairs_dir / "fixture-embed.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in sample_pairs()[:1]), encoding="utf-8"
    )
    document = run_wave(
        ["fixture-embed"],
        RECIPES,
        gpus=1,
        out_dir=out,
        pairs_dir=pairs_dir,
        reference_python=REFERENCE_PYTHON,
        vllm_cmd=f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'} --tokenizer {TOKENIZER}",
        port_base=0,
    )
    by_id = {row["recipe"]: row for row in document["recipes"]}
    assert by_id["fixture-embed"]["state"] == "verified"
    assert (out / "fixture-embed" / "serve.log").is_file()
    assert (out / "fixture-embed" / "status.json").is_file()
    assert (out / "wave.json").is_file()


def test_wave_recipe_cannot_start_fails_only_itself(tmp_path: Path) -> None:
    """An engine that cannot start fails that recipe only."""
    out = tmp_path / "wave"
    document = run_wave(["fixture-embed"], RECIPES, gpus=1, out_dir=out,
                        reference_python=REFERENCE_PYTHON, vllm_cmd="/nonexistent/binary", port_base=0)  # fmt: skip
    row = document["recipes"][0]
    assert row["state"] == "failed"
    assert "cannot start the engine" in (row.get("error") or "")
