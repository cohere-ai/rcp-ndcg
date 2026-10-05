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
    stage1_anchor_check,
    stage1_prompts,
    stage2_scores,
)
from rcp_ndcg_vllm.errors import HarnessError

from tests.conftest import RECIPES, sample_pairs, start_stub, write_pairs
from tests.fixtures.deterministic import token_id, tokens

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
    """A template drift is reported: the served template file and the declared shapes must render identically.

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
    check = report["template_render_check"]
    assert check is not None and check["passed"] is False
    # The drift rewrites "Query: " to "Question: ": the id sequences diverge even at the same length.
    assert check["shape_ids_head"] != check["template_ids_head"]


def test_stage2_rerank_passes_exactly_against_the_clean_stub(stub: Any, tmp_path: Path) -> None:
    """A clean stub engine equals its reference bit for bit; every gate holds with zero slack."""
    recipe = load("fixture-rerank-pointwise")
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    tokenizer = reference._module.tokenizer()
    document = stage2_scores(recipe, stub.base_url, sample_pairs(), reference, tokenizer=tokenizer)
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
        tokenizer = reference._module.tokenizer()
        document = stage2_scores(recipe, engine.base_url, sample_pairs(), reference, tokenizer=tokenizer)
        assert document["passed"] is False
        assert document["abs_delta_max"] > 0.05
        assert any(not row["within"] for row in document["per_document"])
    finally:
        engine.stop()


def test_stage2_embed_passes_and_fails_with_noise(tmp_path: Path) -> None:
    recipe = load("fixture-embed")
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    tokenizer = reference._module.tokenizer()
    engine = start_stub()
    try:
        clean = stage2_scores(recipe, engine.base_url, sample_pairs()[:1], reference, tokenizer=tokenizer)
        assert clean["passed"] is True
        assert clean["cosine_min"] == pytest.approx(1.0)
    finally:
        engine.stop()
    noisy = start_stub("--noise", "0.5")
    try:
        document = stage2_scores(recipe, noisy.base_url, sample_pairs()[:2], reference, tokenizer=tokenizer)
        assert document["passed"] is False
        assert document["cosine_min"] < 1.0 - 1e-3
    finally:
        noisy.stop()


def test_stage2_multi_vector_per_token_after_float16_cast(tmp_path: Path) -> None:
    recipe = load("fixture-multi-vector")
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    tokenizer = reference._module.tokenizer()
    engine = start_stub()
    try:
        document = stage2_scores(recipe, engine.base_url, sample_pairs()[:1], reference, tokenizer=tokenizer)
        assert document["passed"] is True
        assert document["multi_vector"] is True
        assert document["embed_dtype"] == "float16"
        assert len(document["per_vector"]) >= len(sample_pairs()) * 5  # every token of every text
    finally:
        engine.stop()
    noisy = start_stub("--noise", "0.5")
    try:
        document = stage2_scores(recipe, noisy.base_url, sample_pairs()[:1], reference, tokenizer=tokenizer)
        assert document["passed"] is False
    finally:
        noisy.stop()


def test_gates_override_the_defaults(tmp_path: Path) -> None:
    """A recipe's gates section loosens the gate the noise hits, where the published default would fail."""
    recipe = load("fixture-embed")
    loose = recipe.model_copy(update={"gates": recipe.gates.model_copy(update={"vec_min_cosine": 0.5})})
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    tokenizer = reference._module.tokenizer()
    engine = start_stub("--noise", "0.2")
    try:
        document = stage2_scores(loose, engine.base_url, sample_pairs()[:1], reference, tokenizer=tokenizer)
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
    tokenizer = reference._module.tokenizer()
    engine = start_stub("--noise", "0.012")  # deltas ~0.012: above the 0.01 cosine gate, below 0.05 and tau-safe
    try:
        document = stage2_scores(cosine_recipe, engine.base_url, sample_pairs(), reference, tokenizer=tokenizer)
        assert document["passed"] is False
        assert document["gates"][0]["gate"] == "max_abs_delta"
        assert document["gates"][0]["bound"] == 0.01
    finally:
        engine.stop()
    loose = cosine_recipe.model_copy(update={"gates": cosine_recipe.gates.model_copy(update={"cos_max_abs": 0.05})})
    engine = start_stub("--noise", "0.012")
    try:
        document = stage2_scores(loose, engine.base_url, sample_pairs(), reference, tokenizer=tokenizer)
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


# ---------------------------------------------------------------------------
# Anchors: one fixture per anchor kind, over-length survival, the naive-cut mutation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("recipe_id", "anchor"),
    [
        ("fixture-embed", "last"),
        ("fixture-embed-cls", "first"),
        ("fixture-embed-marker", "marker"),
        ("fixture-multi-vector", "mean"),
        ("fixture-rerank-pointwise", "last"),
    ],
)
def test_every_anchor_kind_declares_its_anchor_and_passes_the_audit(recipe_id: str, anchor: str) -> None:
    """Every anchor-kind fixture loads with its anchor AND passes the per-shape over-length audit.

    This is the regression pin for the per-shape audit: a revert to auditing the default shape's render
    against every shape's rule fails here the moment a recipe's shapes disagree (the marker fixture's query
    and document shapes both carry sep, but their heads differ; the cls fixture's head is special on both).
    """
    recipe = load(recipe_id)
    assert recipe.client.template is not None
    assert recipe.client.template.anchor == anchor
    reference = load_reference(str(RECIPES / recipe_id), recipe.reference.entry)
    report = stage1_prompts(
        recipe, sample_pairs()[:1], reference, reference._module.tokenizer(), over_length_per_shape=3
    )
    assert report["passed"] is True, (
        recipe_id,
        (report["anchor_check"] or {}).get("failures"),
    )


def test_every_fixture_recipe_is_covered_by_the_anchor_kind_table() -> None:
    assert {
        recipe.id
        for recipe in (
            load(name)
            for name in (
                "fixture-embed",
                "fixture-embed-cls",
                "fixture-embed-marker",
                "fixture-multi-vector",
                "fixture-rerank-pointwise",
            )
        )
    } == {
        "fixture-embed",
        "fixture-embed-cls",
        "fixture-embed-marker",
        "fixture-multi-vector",
        "fixture-rerank-pointwise",
    }  # the noisy fixture is a copy of the pointwise one (same anchors)


def test_anchor_check_catches_a_whole_prompt_right_cut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation: a naive whole-prompt right cut (the ZeRank/ctxl defect class) loses the tail anchor.

    The recipe's renderer is replaced with the naive cut and the anchor audit must fail on the over-length
    samples; with the real renderer the same inputs pass.
    """
    import rcp_ndcg_vllm.equivalence.prompt as prompt_module

    recipe = load("fixture-embed")
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    tokenizer = reference._module.tokenizer()

    real_segment_text = prompt_module._segment_text

    def naive_assemble(recipe: Recipe, tokenizer: Any, shape: str, contents: dict[str, str]) -> str:
        # Render every segment UNCUT, join, then truncate the whole prompt at the budget: the defect class.
        parts = [
            contents[segment.content]
            if segment.content is not None
            else real_segment_text(recipe, tokenizer, segment, shape, 0)
            for segment in (recipe.client.template and getattr(recipe.client.template, shape)) or []
        ]
        return tokenizer.truncate("".join(parts), recipe.client.max_tokens)

    monkeypatch.setattr(prompt_module, "assemble_shape_text", naive_assemble)
    audit = stage1_prompts(recipe, sample_pairs()[:1], reference, tokenizer, over_length_per_shape=5)
    assert audit["anchor_check"]["passed"] is False
    assert audit["anchor_check"]["failures"]
    assert audit["anchor_check"]["failures"][0]["side"] == "served"


def test_stage2_gates_only_under_cap_pairs_when_the_deviation_is_declared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With anchor_drop_over_cap, over-cap pairs are reported non-gating; the gates use the under-cap pairs."""
    recipe = load("fixture-rerank-pointwise")
    reference = load_reference(str(RECIPE_DIRS / recipe.id), recipe.reference.entry)
    tokenizer = reference._module.tokenizer()
    deviating = recipe.model_copy(
        update={"reference": recipe.reference.model_copy(update={"known_deviations": ["anchor_drop_over_cap"]})}
    )
    over_cap_pairs = [
        {
            "query": "over the cap",
            "documents": ["long document " * 400],  # uncut render far beyond max_tokens
        },
        *sample_pairs(2)[:1],
    ]
    engine = start_stub("--noise", "0.2")  # above every gate: the under-cap pair must still fail
    try:
        document = stage2_scores(deviating, engine.base_url, over_cap_pairs, reference, tokenizer=tokenizer)
        assert document["over_cap"]["n_pairs"] == 1
        assert document["over_cap"]["gating"] is False
        assert document["over_cap"]["pairs"][0]["over_cap"] is True
        assert document["passed"] is False  # the under-cap pair's gates still decide
        # A mixed query (over-cap + under-cap documents) contributes its under-cap pairs to the tau gate:
        # a query whose under-cap docs are inverted fails the tau gate under the deviation.
        clean_pairs = sample_pairs(2)
        mixed = [
            {"query": "mixed query", "documents": ["long document " * 400, *clean_pairs[0]["documents"]]},
        ]
        document = stage2_scores(deviating, engine.base_url, mixed, reference, tokenizer=tokenizer)
        assert document["over_cap"]["n_pairs"] == 1
        assert document["per_query"], "the mixed query must contribute a tau row over its under-cap pairs"
        assert document["per_query"][0]["documents"] == 2  # the tau covers exactly the under-cap pairs
        assert document["passed"] is False  # the stub's noise inverts the under-cap ranking
        # Without the declared deviation the over-cap pair gates like any other (and fails on the noise).
        document = stage2_scores(recipe, engine.base_url, over_cap_pairs, reference, tokenizer=tokenizer)
        assert document["over_cap"]["n_pairs"] == 1
        assert document["passed"] is False
    finally:
        engine.stop()


def test_content_final_anchor_last_with_a_post_processor_end_token(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The declared-end-token escape, end to end: the audit asserts the tokenizer's block at the tail.

    The recipe's document shape is content-final; ``add_special_tokens: true`` makes the post-processor's
    appended block the anchor. A tokenizer that appends exactly that block passes; one that also prepends
    (BOS) fails the audit loudly instead of silently.
    """
    import shutil

    copied = tmp_path_factory.mktemp("escape") / "fixture-embed"
    copied.mkdir(parents=True)
    for name in ("recipe.yaml", "reference.py"):
        shutil.copy(RECIPES / "fixture-embed" / name, copied / name)
    (copied.parent / "deterministic.py").write_bytes((RECIPE_DIRS.parent / "deterministic.py").read_bytes())
    recipe = load_recipe(copied)
    content_final = recipe.model_copy(
        update={
            "client": recipe.client.model_copy(
                update={
                    "add_special_tokens": True,
                    "template": recipe.client.template.model_copy(update={"document": None}),
                }
            ),
        },
    )
    from rcp_ndcg_vllm import TemplateSegment

    content_final = content_final.model_copy(
        update={
            "client": content_final.client.model_copy(
                update={
                    "template": content_final.client.template.model_copy(
                        update={"document": [TemplateSegment(text="doc: "), TemplateSegment(content="document")]}
                    ),
                }
            ),
        }
    )

    class AppendOnly:
        """A tokenizer whose post-processor appends one end token (id 49997) with add_special_tokens."""

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            ids = [token_id(word) for word in tokens(text)]
            return ids + [49997] if add_special_tokens else ids

        def id_to_token(self, token_id_value: int) -> str:
            return {49997: "<<end>>"}.get(token_id_value, f"tok:{token_id_value}")

        def special_tokens(self) -> dict[str, int]:
            return {"end": 49997}

        def decode(self, token_ids: list[int]) -> str:
            return " ".join(self.id_to_token(value) for value in token_ids)

        def truncate(self, text: str, max_tokens: int) -> str:
            words = tokens(text)
            return text if len(words) <= max_tokens else " ".join(words[:max_tokens])

    class BosAndAppends(AppendOnly):
        """A post-processor that also prepends a BOS token: the tail assertion fails loudly."""

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            ids = super().encode(text, add_special_tokens=add_special_tokens)
            return [49996] + ids if add_special_tokens else ids

    reference = load_reference(str(RECIPES / "fixture-embed"), recipe.reference.entry)
    report = stage1_anchor_check(content_final, reference, AppendOnly(), sample_pairs()[:1], over_length_per_shape=3)
    # The reference's own render is the un-pinned fixture render; only the served side is asserted here.
    served_failures = [failure for failure in report["failures"] if failure["side"] == "served"]
    assert report["passed"] is False  # the reference render carries no appended specials: the audit says so
    assert served_failures == []
    loud = stage1_anchor_check(content_final, reference, BosAndAppends(), sample_pairs()[:1], over_length_per_shape=2)
    assert loud["passed"] is False
    assert loud["failures"][0]["side"] == "served"
