"""The equivalence harness against the stub engine: stage 1 on CPU, stage 2 clean and noisy, stage 3 offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from rcp_ndcg_vllm import Recipe, load_recipe
from rcp_ndcg_vllm.equivalence import (
    load_pairs,
    load_reference,
    run,
    stage1_prompts,
    stage2_scores,
)
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
    """A template drift is reported with both sides' token strings, not just a boolean.

    The drifted recipe lives in tmp_path: the tracked fixture is never written.
    """
    import shutil

    copied = tmp_path / "recipes" / "fixture-rerank-pointwise"
    copied.mkdir(parents=True)
    for name in ("recipe.yaml", "template.jinja", "reference.py"):
        shutil.copy(RECIPE_DIRS / "fixture-rerank-pointwise" / name, copied / name)
    (copied.parent.parent / "deterministic.py").write_bytes(
        (RECIPE_DIRS.parent / "deterministic.py").read_bytes()
    )  # the fixture reference imports its number module from parents[2] of its own file
    (copied / "template.jinja").write_text(
        (copied / "template.jinja").read_text(encoding="utf-8").replace("Query: ", "Question: "),
        encoding="utf-8",
    )
    recipe = load_recipe(copied)
    reference = load_reference(str(copied), recipe.reference.entry)
    report = stage1_prompts(recipe, sample_pairs()[:1], reference, reference._module.tokenizer())
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


def test_stage2_cosine_scale_uses_the_cosine_gate(tmp_path: Path) -> None:
    """A cosine-scale reranker is gated by cos_max_abs (0.01), not by the probability gates."""
    recipe = load("fixture-rerank-pointwise")
    cosine_recipe = recipe.model_copy(
        update={"reference": recipe.reference.model_copy(update={"score_scale": "cosine"})}
    )
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    engine = start_stub("--noise", "0.012")  # deltas ~0.012: above the 0.01 cosine gate, below 0.05 and tau-safe
    try:
        document = stage2_scores(cosine_recipe, engine.base_url, sample_pairs(), reference)
        assert document["passed"] is False
        assert document["gates"][0]["gate"] == "max_abs_delta"
        assert document["gates"][0]["bound"] == 0.01
    finally:
        engine.stop()
    loose = cosine_recipe.model_copy(update={"gates": cosine_recipe.gates.model_copy(update={"cos_max_abs": 0.05})})
    engine = start_stub("--noise", "0.012")
    try:
        document = stage2_scores(loose, engine.base_url, sample_pairs(), reference)
        assert document["passed"] is True
    finally:
        engine.stop()


def test_rerank_summary_gate_rows_follow_the_scale() -> None:
    """Unit: the deciding gate rows per scale (probability p99+max; logit relative; cosine absolute)."""
    from rcp_ndcg_vllm.equivalence.gates import ResolvedGates
    from rcp_ndcg_vllm.equivalence.stages import _rerank_summary

    gates = ResolvedGates(
        prob_p99_abs=0.02, prob_max_abs=0.05, logit_rel_abs=0.05, cos_max_abs=0.01,
        vec_min_cosine=0.999, tau_min=0.98, metrics_max_abs=2e-3, embed_dtype="float16",
    )  # fmt: skip
    per_query = [{"query_index": 0, "query": "q", "documents": 2, "kendall_tau": 1.0, "within": True}]
    # cosine scale, one document at |delta| 0.015 (within the probability gates, beyond the cosine gate):
    per_document = [
        {
            "query_index": 0,
            "document_index": 0,
            "served": 0.5,
            "reference": 0.47,
            "abs_delta": 0.03,
            "bound": 0.01,
            "within": False,
        },  # fmt: skip
        {
            "query_index": 0,
            "document_index": 1,
            "served": 0.5,
            "reference": 0.495,
            "abs_delta": 0.005,
            "bound": 0.01,
            "within": True,
        },  # fmt: skip
    ]
    document = _rerank_summary(per_document, per_query, gates, "cosine")
    assert document["passed"] is False  # was wrongly True when the probability gates decided
    logit_document = [
        {
            "query_index": 0,
            "document_index": 0,
            "served": 5.0,
            "reference": 4.5,
            "abs_delta": 0.2,
            "bound": 0.05 * 5.5,
            "within": True,
        },  # fmt: skip
        {
            "query_index": 0,
            "document_index": 1,
            "served": 1.0,
            "reference": 1.0,
            "abs_delta": 0.0,
            "bound": 0.1,
            "within": True,
        },  # fmt: skip
    ]
    document = _rerank_summary(logit_document, per_query, gates, "logit")
    assert document["passed"] is True  # 0.2 <= 0.05*(1+4.5)=0.275: the relative bound decides, not 0.05
    probability_document = [
        {
            "query_index": 0,
            "document_index": 0,
            "served": 0.5,
            "reference": 0.47,
            "abs_delta": 0.03,
            "bound": 0.05,
            "within": True,
        },  # fmt: skip
        {
            "query_index": 0,
            "document_index": 1,
            "served": 0.5,
            "reference": 0.495,
            "abs_delta": 0.005,
            "bound": 0.05,
            "within": True,
        },  # fmt: skip
    ]
    document = _rerank_summary(probability_document, per_query, gates, "probability")
    assert document["passed"] is False  # p99 of two deltas is their max: 0.03 > 0.02 fails the 99%-of-documents rule
    small = [
        {
            "query_index": 0,
            "document_index": 0,
            "served": 0.5,
            "reference": 0.49,
            "abs_delta": 0.01,
            "bound": 0.05,
            "within": True,
        },  # fmt: skip
        {
            "query_index": 0,
            "document_index": 1,
            "served": 0.5,
            "reference": 0.495,
            "abs_delta": 0.005,
            "bound": 0.05,
            "within": True,
        },  # fmt: skip
    ]
    document = _rerank_summary(small, per_query, gates, "probability")
    assert document["passed"] is True  # p99 0.01 <= 0.02, max 0.01 <= 0.05


def test_vector_summary_fails_on_token_count_mismatches() -> None:
    """A late-interaction engine that returns the wrong token count fails stage 2, not just a report row."""

    from rcp_ndcg_vllm.equivalence.gates import ResolvedGates
    from rcp_ndcg_vllm.equivalence.stages import _vector_summary

    gates = ResolvedGates(
        prob_p99_abs=0.02, prob_max_abs=0.05, logit_rel_abs=0.05, cos_max_abs=0.01,
        vec_min_cosine=0.999, tau_min=0.98, metrics_max_abs=2e-3, embed_dtype="float16",
    )  # fmt: skip
    per_vector = [
        {"referent": "query 0 document 0 token 0", "cosine": 1.0, "within": True},
        {
            "referent": "query 0 document 0 token count",
            "cosine": None,
            "within": False,
            "note": "engine returned 9 tokens, reference 10",
        },  # fmt: skip
    ]
    document = _vector_summary(load("fixture-multi-vector"), per_vector, gates, multi=True)
    assert document["passed"] is False  # a perfect cosine is not a pass when the token count differs


def test_p99_gate_is_the_fraction_of_documents_within_the_bound() -> None:
    """The probability gate means |delta| <= 0.02 for 99% of documents, as the design states."""
    from rcp_ndcg_vllm.equivalence.gates import ResolvedGates
    from rcp_ndcg_vllm.equivalence.stages import _rerank_summary

    gates = ResolvedGates(
        prob_p99_abs=0.02, prob_max_abs=0.05, logit_rel_abs=0.05, cos_max_abs=0.01,
        vec_min_cosine=0.999, tau_min=0.98, metrics_max_abs=2e-3, embed_dtype="float16",
    )  # fmt: skip
    per_query = [{"query_index": 0, "query": "q", "documents": 200, "kendall_tau": 1.0, "within": True}]
    # 198 of 200 documents (99%) within 0.02, the rest at 0.021: the stated rule passes, the p99 rule refused it.
    per_document = [
        {
            "query_index": 0,
            "document_index": i,
            "served": 0.5,
            "reference": 0.5 - 0.021,
            "abs_delta": 0.02 if i < 198 else 0.021,
            "bound": 0.05,
            "within": True,
        }  # fmt: skip
        for i in range(200)
    ]
    document = _rerank_summary(per_document, per_query, gates, "probability")
    assert document["within_p99_fraction"] == 0.99
    p99_row = next(row for row in document["gates"] if row["gate"] == "p99_documents_within")
    assert p99_row["value"] == 0.99
    assert document["passed"] is True


def test_stage2_only_run_needs_no_tokenizer(tmp_path: Path) -> None:
    """A stage-2-only run with a hook-less reference works without transformers (the metrics-only install)."""
    import shutil

    recipe = load("fixture-rerank-pointwise")
    hookless = tmp_path / "recipes" / "fixture-rerank-pointwise"
    hookless.mkdir(parents=True)
    for name in ("recipe.yaml", "template.jinja"):
        shutil.copy(RECIPE_DIRS / "fixture-rerank-pointwise" / name, hookless / name)
    (hookless.parent.parent / "deterministic.py").write_bytes((RECIPE_DIRS.parent / "deterministic.py").read_bytes())
    reference_text = (RECIPE_DIRS / "fixture-rerank-pointwise" / "reference.py").read_text(encoding="utf-8")
    # Strip the tokenizer() hook: a production reference omits it.
    start = reference_text.index("def tokenizer()")
    reference_text = reference_text[:start]  # the hook is the module's last definition
    (hookless / "reference.py").write_text(reference_text, encoding="utf-8")
    recipe = load_recipe(hookless)
    reference = load_reference(str(hookless), recipe.reference.entry)
    assert getattr(reference._module, "tokenizer", None) is None  # the hook is really gone
    engine = start_stub()
    try:
        document = run(
            recipe, base_url=engine.base_url, pairs=sample_pairs(), out_dir=tmp_path, stages=[2], reference=reference
        )  # fmt: skip
    finally:
        engine.stop()
    assert document["passed"] is True  # stage 2 never needed the tokenizer


def test_stage3_runs_for_a_stored_scores_recipe(tmp_path: Path) -> None:
    """--stages 3 on a stored_scores recipe runs (the refusal must not fire from a tokenizer lookup)."""
    pytest.importorskip("rcp_ndcg")
    from rcp_ndcg.data import Rankings

    recipe = load("fixture-rerank-pointwise")
    stored = recipe.model_copy(update={"reference": recipe.reference.model_copy(update={"kind": "stored_scores"})})
    rankings_dir = tmp_path / "rankings"
    rankings_dir.mkdir()
    for system in ("served", "reference"):
        Rankings.from_orders({"q1": ["a", "b", "c"]}, system=system).save(rankings_dir / f"toy.{system}.jsonl")
    (rankings_dir / "toy.dataset.jsonl").write_text(
        json.dumps(
            {"query_id": "q1", "query": "q", "doc_ids": ["a", "b", "c"], "docs": ["A", "B", "C"], "qrels": {"a": 1.0}}
        )
        + "\n",  # fmt: skip
        encoding="utf-8",
    )
    document = run(
        stored, base_url=None, pairs=[], out_dir=tmp_path, stages=[3], reference=None, rankings_dir=rankings_dir
    )  # fmt: skip
    assert document["passed"] is True
