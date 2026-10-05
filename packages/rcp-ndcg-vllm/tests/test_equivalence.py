"""The equivalence harness against the stub engine: stage 1 on CPU, stage 2 clean and noisy, stage 3 offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from rcp_ndcg_vllm import Recipe, load_recipe
from rcp_ndcg_vllm.equivalence import load_pairs, load_reference, run, stage1_prompts, stage2_scores
from rcp_ndcg_vllm.errors import HarnessError

from tests.conftest import RECIPES, sample_pairs, start_stub, write_pairs

RECIPE_DIRS = RECIPES


def load(recipe_id: str) -> Recipe:
    return load_recipe(RECIPE_DIRS / recipe_id)


def test_stage1_passes_for_the_rerank_template() -> None:
    """The template renders (jinja, engine settings) to exactly the reference's token ids."""
    recipe = load("fixture-rerank-pointwise")
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    tokenizer = reference._module.tokenizer()
    report = stage1_prompts(recipe, sample_pairs(), reference, tokenizer)
    assert report["passed"] is True
    assert report["checked"] == report["pairs"] * 4
    assert report["mismatch"] is None


def test_stage1_passes_for_the_field_instruction_template() -> None:
    recipe = load("fixture-rerank-listwise")
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    report = stage1_prompts(recipe, sample_pairs(), reference, reference._module.tokenizer())
    assert report["passed"] is True


def test_stage1_passes_for_prompted_embedding_roles() -> None:
    for recipe_id in ("fixture-embed", "fixture-multi-vector"):
        recipe = load(recipe_id)
        reference = load_reference(str(RECIPE_DIRS / recipe_id), recipe.reference.entry)
        report = stage1_prompts(recipe, sample_pairs(), reference, reference._module.tokenizer())
        assert report["passed"] is True, recipe_id


def test_stage1_reports_the_first_mismatch_with_token_strings(tmp_path: Path) -> None:
    """A template drift is reported with both sides' token strings, not just a boolean."""
    recipe = load("fixture-rerank-pointwise")
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    tokenizer = reference._module.tokenizer()
    template = RECIPE_DIRS / "fixture-rerank-pointwise" / "template.jinja"
    original = template.read_text(encoding="utf-8")
    template.write_text(original.replace("Query: ", "Question: "), encoding="utf-8")
    try:
        report = stage1_prompts(recipe, sample_pairs()[:1], reference, tokenizer)
    finally:
        template.write_text(original, encoding="utf-8")
    assert report["passed"] is False
    mismatch = report["mismatch"]
    assert mismatch["served_tokens"] != mismatch["reference_tokens"]
    assert mismatch["served_ids"] != mismatch["reference_ids"]
    assert mismatch["query_index"] == 0


def test_stage2_rerank_passes_exactly_against_the_clean_stub(stub: Any, tmp_path: Path) -> None:
    """A clean stub engine equals its reference bit for bit; every gate holds with zero slack."""
    recipe = load("fixture-rerank-pointwise")
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    document = stage2_scores(recipe, stub.base_url, sample_pairs(), reference)
    assert document["passed"] is True
    assert document["abs_delta_max"] == 0.0
    assert document["kendall_tau_median"] == 1.0
    assert all(row["within"] for row in document["per_document"])


def test_stage2_rerank_fails_when_the_stub_adds_noise(tmp_path: Path) -> None:
    """Noise above the probability gates fails the recipe; the report names the offending documents."""
    engine = start_stub("--noise", "0.2")
    try:
        recipe = load("fixture-rerank-pointwise")
        reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
        document = stage2_scores(recipe, engine.base_url, sample_pairs(), reference)
        assert document["passed"] is False
        assert document["abs_delta_max"] > 0.05
        assert any(not row["within"] for row in document["per_document"])
    finally:
        engine.stop()


def test_stage2_embed_passes_and_fails_with_noise(tmp_path: Path) -> None:
    recipe = load("fixture-embed")
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    engine = start_stub()
    try:
        clean = stage2_scores(recipe, engine.base_url, sample_pairs(), reference)
        assert clean["passed"] is True
        assert clean["cosine_min"] == pytest.approx(1.0)
    finally:
        engine.stop()
    noisy = start_stub("--noise", "0.5")
    try:
        document = stage2_scores(recipe, noisy.base_url, sample_pairs(), reference)
        assert document["passed"] is False
        assert document["cosine_min"] < 1.0 - 1e-3
    finally:
        noisy.stop()


def test_stage2_multi_vector_per_token_after_float16_cast(tmp_path: Path) -> None:
    recipe = load("fixture-multi-vector")
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    engine = start_stub()
    try:
        document = stage2_scores(recipe, engine.base_url, sample_pairs(), reference)
        assert document["passed"] is True
        assert document["multi_vector"] is True
        assert document["embed_dtype"] == "float16"
        assert len(document["per_vector"]) >= len(sample_pairs()) * 5  # every token of every text
    finally:
        engine.stop()
    noisy = start_stub("--noise", "0.5")
    try:
        document = stage2_scores(recipe, noisy.base_url, sample_pairs(), reference)
        assert document["passed"] is False
    finally:
        noisy.stop()


def test_gates_override_the_defaults(tmp_path: Path) -> None:
    """A recipe's gates section loosens the gate the noise hits, where the published default would fail."""
    recipe = load("fixture-embed")
    loose = recipe.model_copy(update={"gates": recipe.gates.model_copy(update={"vec_min_cosine": 0.5})})
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    engine = start_stub("--noise", "0.2")
    try:
        document = stage2_scores(loose, engine.base_url, sample_pairs(), reference)
        assert document["passed"] is True
        assert document["gates"][0]["bound"] == 0.5
    finally:
        engine.stop()


def test_logit_gate_is_relative_to_the_reference_score() -> None:
    from rcp_ndcg_vllm.equivalence.gates import ResolvedGates, kendall_tau_b

    gates = ResolvedGates(
        prob_p99_abs=0.02, prob_max_abs=0.05, logit_rel_abs=0.05, cos_max_abs=0.01,
        vec_min_cosine=0.999, tau_min=0.98, metrics_max_abs=2e-3, embed_dtype="float16",
    )  # fmt: skip
    bound = gates.logit_rel_abs * (1 + abs(-8.0))
    assert bound == 0.45
    assert kendall_tau_b([1.0, 2.0], [2.0, 1.0]) == -1.0
    assert kendall_tau_b([1.0], [1.0]) is None
    assert kendall_tau_b([1.0, 1.0], [1.0, 1.0]) is None  # fully tied: no ordering to agree on


def test_full_run_writes_equivalence_json_and_markdown(tmp_path: Path) -> None:
    recipe = load("fixture-rerank-pointwise")
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    pairs = sample_pairs()
    engine = start_stub()
    try:
        document = run(
            recipe, base_url=engine.base_url, pairs=pairs, out_dir=tmp_path, stages=[1, 2],
            tokenizer=reference._module.tokenizer(), reference=reference,
        )  # fmt: skip
    finally:
        engine.stop()
    assert document["passed"] is True
    equivalence = json.loads((tmp_path / "equivalence.json").read_text(encoding="utf-8"))
    assert equivalence["recipe"] == "fixture-rerank-pointwise"
    markdown = (tmp_path / "EQUIVALENCE.md").read_text(encoding="utf-8")
    assert "PASS" in markdown
    assert "|served - reference|" in markdown  # every gate row carries its referent


def test_stage1_only_run_needs_no_engine(tmp_path: Path) -> None:
    """The CPU validation path: stage 1 alone, no base URL, no engine running."""
    recipe = load("fixture-rerank-pointwise")
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    document = run(
        recipe, base_url=None, pairs=sample_pairs(), out_dir=tmp_path, stages=[1],
        tokenizer=reference._module.tokenizer(), reference=reference,
    )  # fmt: skip
    assert document["passed"] is True


def test_stage3_scores_via_the_rcp_ndcg_subprocess(tmp_path: Path) -> None:
    """Stage 3 shells out to `rcp-ndcg eval score`; identical rankings give delta 0 and a pass."""
    pytest.importorskip("rcp_ndcg")  # the metrics extra; the root environment has it
    from rcp_ndcg.data import Rankings

    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    orders = {"q1": ["a", "b", "c"], "q2": ["b", "a", "c"]}
    for system in ("served", "reference"):
        Rankings.from_orders(orders, system=system).save(rankings_dir / f"subset.{system}.jsonl")
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
    (rankings_dir / "subset.dataset.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in dataset), encoding="utf-8"
    )
    from rcp_ndcg_vllm.equivalence.gates import ResolvedGates
    from rcp_ndcg_vllm.equivalence.metrics import stage3_metrics

    gates = ResolvedGates(
        prob_p99_abs=0.02, prob_max_abs=0.05, logit_rel_abs=0.05, cos_max_abs=0.01,
        vec_min_cosine=0.999, tau_min=0.98, metrics_max_abs=2e-3, embed_dtype="float16",
    )  # fmt: skip
    document = stage3_metrics(rankings_dir, gates)
    assert document["passed"] is True
    assert document["subsets"][0]["abs_delta"] == 0.0
    # A different ranking must move the metric and fail the 2e-3 gate.
    Rankings.from_orders({"q1": ["c", "b", "a"], "q2": ["b", "a", "c"]}, system="served").save(
        rankings_dir / "subset.served.jsonl"
    )
    moved = stage3_metrics(rankings_dir, gates)
    assert moved["passed"] is False


def test_pairs_file_validation(tmp_path: Path) -> None:
    good = write_pairs(tmp_path / "pairs.jsonl", sample_pairs(2))
    assert len(load_pairs(good)) == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"query": "q", "documents": ["a"]}\nnot json\n', encoding="utf-8")
    with pytest.raises(HarnessError, match="not a JSON object"):
        load_pairs(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(HarnessError, match="no pairs"):
        load_pairs(empty)
    wrong = tmp_path / "wrong.jsonl"
    wrong.write_text('{"query": "q"}\n', encoding="utf-8")
    with pytest.raises(HarnessError, match="documents"):
        load_pairs(wrong)


def test_cli_exit_codes(tmp_path: Path) -> None:
    """The CLI: exit 0 on pass, 1 on a failed gate, 2 on a usage error."""
    from rcp_ndcg_vllm.equivalence import main

    recipe_dir = str(RECIPE_DIRS / "fixture-rerank-pointwise")
    pairs = write_pairs(tmp_path / "pairs.jsonl", sample_pairs())
    engine = start_stub()
    try:
        clean_out = tmp_path / "clean"
        assert (
            main(
                ["--recipe", recipe_dir, "--base-url", engine.base_url, "--pairs", str(pairs), "--out", str(clean_out)]
            )
            == 0
        )
    finally:
        engine.stop()
    noisy = start_stub("--noise", "0.2")
    try:
        noisy_out = tmp_path / "noisy"
        assert (
            main(["--recipe", recipe_dir, "--base-url", noisy.base_url, "--pairs", str(pairs), "--out", str(noisy_out)])
            == 1
        )
    finally:
        noisy.stop()
    assert (
        main(
            [
                "--recipe",
                recipe_dir,
                "--base-url",
                "http://127.0.0.1:1",
                "--pairs",
                str(pairs),
                "--out",
                str(tmp_path / "x"),
            ]
        )
        == 2
    )
