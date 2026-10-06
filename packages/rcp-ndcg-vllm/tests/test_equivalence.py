"""The equivalence harness against the product's fit and the stub engine.

Stage 1 runs the product's fit; the anchor audit reads fit's output; the engine's /tokenize must agree (R29);
the reference runs as a subprocess in its own environment; the harness process never imports torch or
transformers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from rcp_ndcg_vllm import load_recipe
from rcp_ndcg_vllm.equivalence import run, stage1_prompts, stage2_scores

from tests.conftest import RECIPES, TOKENIZER, sample_pairs, start_stub, write_pairs


def load(recipe_id: str):
    return load_recipe(RECIPES / recipe_id)


def reference_python() -> str:
    return sys.executable


@pytest.mark.parametrize(
    "recipe_id",
    [
        "fixture-embed",
        "fixture-embed-cls",
        "fixture-embed-edge",
        "fixture-embed-marker",
        "fixture-multi-vector",
        "fixture-rerank-pointwise",
    ],
)
def test_stage1_passes_for_every_anchor_kind_with_the_reference_render(tmp_path: Path, recipe_id: str) -> None:
    """fit's renders, the anchor audit, the reference subprocess render and the template check all agree."""
    recipe = load(recipe_id)
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    document = stage1_prompts(recipe, pairs, reference_python(), over_length_per_shape=3)
    assert document["fit"], recipe_id
    assert document["anchor_check"]["passed"] is True, (recipe_id, document["anchor_check"]["failures"][:1])
    assert document["render_check"]["passed"] is True, (recipe_id, document["render_check"]["failures"][:1])
    if recipe.serve.chat_template is not None:
        assert document["template_render_check"]["passed"] is True, recipe_id


def test_engine_tokenize_check_runs_against_the_stub_and_fails_on_drift(tmp_path: Path) -> None:
    """The engine's /tokenize must agree with fit's ids and counts (R29); without an engine: not_run."""
    recipe = load("fixture-embed")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage1_prompts(recipe, pairs, None, base_url=engine.base_url, over_length_per_shape=2)
        assert document["engine_tokenize_check"]["status"] == "run"
        assert document["engine_tokenize_check"]["passed"] is True
        assert document["engine_tokenize_check"]["checked"] > 0
    finally:
        engine.stop()
    # Without an engine the check is reported not_run, never passed.
    document = stage1_prompts(recipe, pairs, None, over_length_per_shape=1)
    assert document["engine_tokenize_check"]["status"] == "not_run"
    assert document["engine_tokenize_check"]["passed"] is None  # not_run is neutral, never passed


def test_stage1_without_a_reference_python_reports_not_run(tmp_path: Path) -> None:
    recipe = load("fixture-embed")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    document = stage1_prompts(recipe, pairs, None, over_length_per_shape=2)
    assert document["render_check"]["status"] == "not_run"
    assert document["anchor_check"]["passed"] is True


def test_stage2_rerank_via_the_product_client_and_the_reference_subprocess(tmp_path: Path) -> None:
    """Stage 2 sends through the product's RerankClient and compares against the reference subprocess."""
    recipe = load("fixture-rerank-pointwise")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(
            recipe, pairs, reference_python(), base_url=engine.base_url, served_model_name=recipe.id
        )
        assert document["passed"] is True, document["gates"]
        assert document["abs_delta_max"] == 0.0
        assert document["kendall_tau_median"] == 1.0
        assert document["over_cap"]["n_pairs"] == 0
    finally:
        engine.stop()


def test_stage2_embed_via_the_product_client(tmp_path: Path) -> None:
    recipe = load("fixture-embed")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(
            recipe, pairs, reference_python(), base_url=engine.base_url, served_model_name=recipe.id
        )
        assert document["passed"] is True, document["gates"]
    finally:
        engine.stop()


def test_stage2_multi_vector_runs(tmp_path: Path) -> None:
    """The multi_vector stage 2 runs; the per-token comparison needs clients-final's budget wiring."""
    recipe = load("fixture-multi-vector")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(
            recipe, pairs, reference_python(), base_url=engine.base_url, served_model_name=recipe.id
        )
        assert "per_vector" in document
        assert "gates" in document
    finally:
        engine.stop()


def test_stage2_gates_only_under_cap_pairs_when_the_deviation_is_declared(tmp_path: Path) -> None:
    """With anchor_drop_over_cap, over-cap pairs are reported non-gating; the under-cap pairs still gate."""
    recipe = load("fixture-rerank-pointwise")
    deviating = recipe.model_copy(
        update={"reference": recipe.reference.model_copy(update={"known_deviations": ["anchor_drop_over_cap"]})}
    )
    pairs = write_pairs(
        tmp_path / "pairs.jsonl",
        [
            {"query": "over the cap", "documents": ["long document tokens " * 400]},
            *sample_pairs(2)[:1],
        ],
    )
    engine = start_stub("--noise", "0.2", "--tokenizer", str(TOKENIZER))
    try:
        document = stage2_scores(
            deviating, pairs, reference_python(), base_url=engine.base_url, served_model_name=recipe.id
        )
        assert document["over_cap"]["n_pairs"] == 1
        assert document["over_cap"]["gating"] is False
        assert document["passed"] is False  # the under-cap pair's gates still decide
    finally:
        engine.stop()


def test_stage2_requires_the_reference_python(tmp_path: Path) -> None:
    from rcp_ndcg_vllm.errors import HarnessError

    recipe = load("fixture-rerank-pointwise")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        with pytest.raises(HarnessError, match="reference runs in its own environment"):
            stage2_scores(recipe, pairs, "", base_url=engine.base_url, served_model_name=recipe.id)
    finally:
        engine.stop()


def test_full_run_writes_the_report(tmp_path: Path) -> None:
    recipe = load("fixture-embed")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs()[:1])
    engine = start_stub("--tokenizer", str(TOKENIZER))
    try:
        document = run(
            recipe, base_url=engine.base_url, pairs_path=str(pairs), out_dir=str(tmp_path), stages=[1, 2],
            reference_python=reference_python(),
        )  # fmt: skip
    finally:
        engine.stop()
    assert document["passed"] is True
    equivalence = json.loads((tmp_path / "equivalence.json").read_text(encoding="utf-8"))
    assert equivalence["recipe"] == "fixture-embed"
    assert "PASS" in (tmp_path / "EQUIVALENCE.md").read_text(encoding="utf-8")
