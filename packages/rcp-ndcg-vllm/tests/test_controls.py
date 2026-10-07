"""The negative controls (a)-(f): the broken variants and the wave report that flags a control that passes."""

from __future__ import annotations

import json

from rcp_ndcg_vllm.observe.controls import CONTROLS, control_variants, controls_summary
from rcp_ndcg_vllm.recipe import load_recipe

from tests.conftest import RECIPES


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
