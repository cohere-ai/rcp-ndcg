"""The T3 quality stage and the negative controls: tables, gates, golden replay and the mutation proof."""

from __future__ import annotations

import json
from pathlib import Path

from rcp_ndcg_vllm.observe.controls import CONTROLS, control_variants, controls_summary
from rcp_ndcg_vllm.quality import (
    QUALITY_TOLERANCE,
    TASK_MATRIX,
    comparison_rows,
    golden_replay_selection,
    quality_md,
    reference_command,
    served_commands,
    write_golden_replay,
)
from rcp_ndcg_vllm.recipe import default_recipes_root, load_recipe

from tests.conftest import RECIPES


def test_the_task_matrix_covers_every_recipe() -> None:
    """GPU-VALIDATION.md's T3 matrix as data: every recipe directory is named once as a model."""
    listed = [model for row in TASK_MATRIX for model in row["models"]]
    recipe_ids = sorted(path.name for path in default_recipes_root().iterdir() if (path / "recipe.yaml").is_file())
    for recipe_id in recipe_ids:
        assert recipe_id in listed, f"{recipe_id} is missing from TASK_MATRIX"
    assert len(listed) == len(set(listed)) + 1, "topk-embed-v1-small appears in two families (documented)"
    assert QUALITY_TOLERANCE == 0.005


def test_comparison_rows_gate_the_served_delta_and_note_deviations() -> None:
    """Served vs reference within 0.5 points passes; over the gate fails; a published mismatch needs
    its explanation (the deviation-note column)."""
    ok = comparison_rows(
        {"NanoNQ": 0.4120},
        {"NanoNQ": 0.4100},
        paper={"NanoNQ": 0.4095},
        published={"NanoNQ": 0.45},
        published_source="model card",  # fmt: skip
    )
    assert ok["passed"] is True
    row = ok["rows"][0]
    assert abs(row["delta_vs_reference"] - 0.0020) < 1e-9
    assert "unexplained deviation" in row["deviation_note"]

    explained = comparison_rows(
        {"NanoNQ": 0.4120},
        {"NanoNQ": 0.4100},
        published={"NanoNQ": 0.45},
        deviation_note={"NanoNQ": "pool: paper dropped anchors over cap"},
    )
    assert explained["rows"][0]["deviation_note"] == "pool: paper dropped anchors over cap"

    failing = comparison_rows({"a": 0.40}, {"a": 0.41})
    assert failing["passed"] is False
    assert failing["rows"][0]["within_tolerance"] is False


def test_quality_md_table_and_verdict() -> None:
    comparison = comparison_rows({"NanoNQ": 0.4120}, {"NanoNQ": 0.4100})
    text = quality_md("fixture-embed", comparison)
    assert "| NanoNQ | 0.4120 | 0.4100 | 0.0020 |" in text
    assert "Verdict: **PASS**" in text


def test_served_commands_use_the_product_cli() -> None:
    retrieval = served_commands(
        view="retrieval", model="m", engine_url="http://engine", dataset="suite:nanobeir", out_dir="/tmp/x"
    )
    assert retrieval[0][:3] == ["rcp-ndcg", "retrieval", "index"]
    assert retrieval[1][:3] == ["rcp-ndcg", "retrieval", "search"]
    assert retrieval[2][:3] == ["rcp-ndcg", "eval", "score"]
    rerank = served_commands(
        view="reranking", model="m", engine_url="http://engine", dataset="suite:nanobeir", out_dir="/tmp/x"
    )
    assert rerank[0][:3] == ["rcp-ndcg", "retrieval", "rerank"]
    assert reference_command(model="org/model", task="NanoNQRetrieval", out_dir="/tmp/x")[:2] == ["mteb", "run"]


def test_the_golden_replay_keeps_the_full_subsets_only(tmp_path: Path) -> None:
    """One NanoBEIR and one ViDoRe subset's FULL served exchanges (probes excluded)."""

    def record(subset: str, probe: str | None = None) -> dict:
        return {
            "inputs": {
                "source": {"subset": subset},
                "probe": probe or "ok",
            },
            "response": {"status": 200},
        }

    records = [
        record("NanoNQRetrieval"),
        record("NanoNQRetrieval"),
        record("hr__english"),
        record("NanoFiQA2018Retrieval"),
        record("NanoNQRetrieval", probe="tokenize"),
    ]
    selection = golden_replay_selection(records, subsets=("NanoNQRetrieval", "hr__english"))
    assert selection["count"] == 3
    assert {row["inputs"]["source"]["subset"] for row in selection["records"]} == {
        "NanoNQRetrieval",
        "hr__english",
    }
    path = write_golden_replay(selection, tmp_path)
    assert path.is_file() and (tmp_path / "index.json").is_file()
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_control_variants_break_exactly_the_declared_field() -> None:
    """(a)-(d) five text controls over the fixture recipe; each variant's breakage is the declared one."""
    recipe = load_recipe(RECIPES / "fixture-embed")
    variants = control_variants(recipe)
    by_name = {entry["name"]: entry["recipe"] for entry in variants}
    assert "--truncate-prompt-tokens" in by_name["right-cut"].serve.extra_args
    assert by_name["wrong-pooling"].serve.pooler_config["pooling_type"] == "MEAN"
    assert "template-removed" not in by_name, "chat_template is already None here: the control cannot apply"
    for entry in variants:
        assert entry["recipe"].id == f"fixture-embed.{entry['name']}"

    # A recipe WITH a chat template gets the (a) control: the served frame is dropped.
    template_recipe = load_recipe(RECIPES / "fixture-rerank-pointwise")
    templated = {entry["name"]: entry["recipe"] for entry in control_variants(template_recipe)}
    assert templated["template-removed"].serve.chat_template is None
    assert [spec.letter for spec in CONTROLS] == ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]


def test_the_no_op_gate_mutation_is_flagged_as_a_control_blocker() -> None:
    """The mutation test (GPU-VALIDATION 5): with every gate live the controls are all caught; make a
    control's gate a no-op (its broken variant now PASSES) and the report flags it as a blocker."""
    caught_rows = [
        {"control": "(a)", "name": "template-removed", "equivalence": {"passed": False}},
        {"control": "(b)", "name": "right-cut", "equivalence": {"passed": False}},
    ]
    healthy = controls_summary(caught_rows)
    assert healthy["passed"] is True and healthy["blockers"] == []

    # The mutation: (b)'s gate is made a no-op -- the broken variant's report now reads "passed".
    rows = [
        {"control": "(a)", "name": "template-removed", "equivalence": {"passed": False}},
        {"control": "(b)", "name": "right-cut", "equivalence": {"passed": True}},
    ]
    report = controls_summary(rows)
    assert report["passed"] is False
    assert [blocker["control"] for blocker in report["blockers"]] == ["(b)"]
    assert "no-op or blind gate" in report["blockers"][0]["reason"]
    assert json.dumps(report)  # the wave report renders it as data (WAVE.md prints the blockers)"
