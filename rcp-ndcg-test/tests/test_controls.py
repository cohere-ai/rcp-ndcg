"""The negative controls (a)-(f): each provably fails the gates, through the wave, on the stub engine.

GPU-VALIDATION.md item 5: every GPU check must prove it can fail -- each wave serves deliberately broken variants
and requires the gates to fail them; a control that passes is a blocker.  Here every control runs through
``run_wave --controls`` against the stub engine, which breaks exactly as vLLM v0.31.0 breaks (it honours
``truncate_prompt_tokens``/``truncation_side``, ``use_activation`` and the requested ``embed_dtype``; and, as
properties of the emulated checkpoint, ``--model-pooling`` and ``--model-needs-template``).  One test per
control asserts its row was CAUGHT (the gates failed it) while the recipe's own gates passed; the mutation test
makes a gate a no-op and shows the wave flag the control as a blocker; (f) is caught by the media stage's engine
count on a vision embedder whose pixel pin lies below the family's stock floor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from rcp_ndcg_test.jobs.run_wave import run_wave
from rcp_ndcg_test.observe.controls import CONTROLS, control_variants, controls_summary, right_cut_tokens
from rcp_ndcg_vllm.recipe import load_recipe

from tests.conftest import RECIPES, TOKENIZER, sample_pairs, write_pairs

STUB = f"{sys.executable} {Path(__file__).resolve().parent / 'stub_engine.py'} --tokenizer {TOKENIZER}"


def _wave(tmp_path: Path, recipe_id: str, *, model: str = "") -> dict[str, Any]:
    pairs = tmp_path / "pairs"
    pairs.mkdir(exist_ok=True)
    write_pairs(pairs / f"{recipe_id}.jsonl", sample_pairs(documents=3))
    return run_wave(
        [recipe_id],
        RECIPES,
        gpus=1,
        out_dir=tmp_path / "wave",
        pairs_dir=pairs,
        reference_python=sys.executable,
        vllm_cmd=f"{STUB} {model}".strip(),
        port_base=0,
        controls=True,
    )


def _controls(document: dict[str, Any]) -> dict[str, Any]:
    row = document["recipes"][0]
    assert row["steps"]["equivalence"]["passed"] is True, "the recipe's own gates must pass first"
    return row["steps"]["controls"]


def _caught(step: dict[str, Any], letter: str) -> dict[str, Any]:
    rows = {row["control"]: row for row in step["rows"]}
    assert letter in rows, f"control {letter} was not served: {step}"
    return rows[letter]


@pytest.fixture(scope="module")
def rerank_wave(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One wave of the pointwise reranker fixture, its checkpoint pooled with LAST and needing its template."""
    return _wave(
        tmp_path_factory.mktemp("rerank"),
        "fixture-rerank-pointwise",
        model="--model-pooling LAST --model-needs-template",
    )


@pytest.fixture(scope="module")
def embed_wave(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _wave(tmp_path_factory.mktemp("embed"), "fixture-embed", model="--model-pooling LAST")


@pytest.fixture(scope="module")
def pooling_wave(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _wave(tmp_path_factory.mktemp("pooling"), "fixture-multi-vector")


def test_control_a_the_removed_template_fails_the_gates(rerank_wave: dict[str, Any]) -> None:
    row = _caught(_controls(rerank_wave), "(a)")
    assert row["caught"] is True and row["gates_passed"] is False


def test_control_b_an_engine_side_right_cut_fails_the_gates(
    rerank_wave: dict[str, Any], embed_wave: dict[str, Any], pooling_wave: dict[str, Any]
) -> None:
    """The wire carries ``truncate_prompt_tokens`` (a request field in vLLM v0.31.0; the earlier serve flag
    ``--truncate-prompt-tokens`` does not exist and would only have kept the engine from starting)."""
    for wave in (rerank_wave, embed_wave, pooling_wave):
        assert _caught(_controls(wave), "(b)")["caught"] is True


def test_control_c_a_flipped_use_activation_fails_the_gates(rerank_wave: dict[str, Any]) -> None:
    assert _caught(_controls(rerank_wave), "(c)")["caught"] is True


def test_control_d_the_wrong_pooling_fails_the_gates(rerank_wave: dict[str, Any], embed_wave: dict[str, Any]) -> None:
    """fixture-embed declares ``seq_pooling_type: LAST``: the variant flips THAT key (adding ``pooling_type``
    beside it, as before, makes vLLM refuse the config: "Cannot set both")."""
    variant = next(v for v in control_variants(load_recipe(RECIPES / "fixture-embed")) if v["control"] == "(d)")
    assert variant["recipe"].serve.pooler_config == {"seq_pooling_type": "MEAN"}
    for wave in (rerank_wave, embed_wave):
        assert _caught(_controls(wave), "(d)")["caught"] is True


def test_control_e_float32_read_as_float16_fails_the_gates(pooling_wave: dict[str, Any]) -> None:
    """The /pooling request asks for float32 while the client decodes float16 (flipping the dtype on BOTH
    sides, as before, only lowers the precision -- the cosine gate passes that)."""
    assert _caught(_controls(pooling_wave), "(e)")["caught"] is True
    step = _controls(pooling_wave)
    assert step["passed"] is True and step["blockers"] == []


def test_control_f_an_unpinned_pixel_budget_fails_the_media_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(f) unpins the nested ``images_kwargs`` pin of a vision embedder whose budget lies below the family's
    stock floor: the variant's engine re-resizes the prepared images under that floor, the media stage's engine
    count differs from the client's, and the control is caught -- while the recipe's own gates pass."""
    from tests.conftest import hub_cache
    from tests.test_media import media_pairs

    # The emulated checkpoint's own budget: Qwen2-VL's stock 3136..12845056 px, which the pinned 784 px floor leaves.
    stock = {"min_pixels": 3136, "max_pixels": 12845056}
    revision = "0123456789abcdef0123456789abcdef01234567"
    hub_cache(
        tmp_path, monkeypatch, "fixtures/VisionEmbedder", revision, {"preprocessor_config.json": json.dumps(stock)}
    )
    pairs = tmp_path / "pairs"
    pairs.mkdir()
    media_pairs(pairs / "fixture-vl-embed.jsonl")
    document = run_wave(
        ["fixture-vl-embed"],
        RECIPES,
        gpus=1,
        out_dir=tmp_path / "wave",
        pairs_dir=pairs,
        reference_python=sys.executable,
        vllm_cmd=f"{STUB} --model-pooling LAST",
        port_base=0,
        controls=True,
    )
    step = _controls(document)
    row = _caught(step, "(f)")
    assert row["caught"] is True and row["gates_passed"] is False
    variant = next(v for v in control_variants(load_recipe(RECIPES / "fixture-vl-embed")) if v["control"] == "(f)")
    assert variant["recipe"].serve.mm_processor_kwargs == {}
    assert step["passed"] is True and step["blockers"] == [], step["blockers"]


VL_RERANKER = ("Qwen/Qwen3-VL-Reranker-2B", "4bd860ac4f15ad1897a214615cccc700f8f71818")
VL_EMBEDDING = ("Qwen/Qwen3-VL-Embedding-2B", "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda")
TOPK = ("topk-io/topk-embed-v1-small", "e54485ebab921f2c18c4d092b3f4c40dcca26781")


def _shipped(recipe_id: str) -> Any:
    from rcp_ndcg_vllm.recipe import iter_recipes

    return next(recipe for recipe in iter_recipes() if recipe.id == recipe_id)


def _control_f(recipe: Any) -> dict[str, Any]:
    (row,) = [v for v in control_variants(recipe) if v["control"] == "(f)"]
    return row


def test_control_f_applies_only_where_the_pin_leaves_the_checkpoints_own_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unpinning hands the engine the checkpoint's own budget (its preprocessor config at the pinned revision),
    and the engine keeps every image the client prepared inside it: (f) is a breakage only where the pin
    leaves that budget. qwen3-vl-reranker-2b pins 4096..1310720 inside its checkpoint's 4095..1310720 and
    topk-embed-v1-small 65536..1310720 inside 65536..16777216 -- not applicable, said why; qwen3-vl-embedding-2b
    pins 1843200 px over its checkpoint's 1310720 -- served (the media stage's engine count catches it)."""
    from tests.conftest import hub_cache

    reranker = {"min_pixels": 4095, "max_pixels": 1310720, "size": {"shortest_edge": 65536, "longest_edge": 16777216}}
    hub_cache(tmp_path / "r", monkeypatch, *VL_RERANKER, {"preprocessor_config.json": json.dumps(reranker)})
    row = _control_f(_shipped("qwen3-vl-reranker-2b"))
    assert row["kind"] is None and "4095-1310720" in row["reason"] and "preprocessor_config.json" in row["reason"]
    embedding = {"min_pixels": 4096, "max_pixels": 1310720}
    hub_cache(tmp_path / "e", monkeypatch, *VL_EMBEDDING, {"preprocessor_config.json": json.dumps(embedding)})
    assert _control_f(_shipped("qwen3-vl-embedding-2b"))["kind"] == "recipe"
    topk = {"image_processor": {"size": {"shortest_edge": 65536, "longest_edge": 16777216}}}
    hub_cache(
        tmp_path / "t", monkeypatch, *TOPK, {"processor_config.json": json.dumps(topk)}, ("preprocessor_config.json",)
    )
    row = _control_f(_shipped("topk-embed-v1-small"))
    assert row["kind"] is None and "processor_config.json" in row["reason"]


def test_control_f_with_an_unreadable_checkpoint_budget_is_a_blocker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the checkpoint's own budget the control cannot say whether unpinning changes anything: it is
    never declared inapplicable on a guess -- it is unresolved, and the wave's summary counts it a blocker."""
    from tests.conftest import hub_cache

    hub_cache(tmp_path / "unknown", monkeypatch, *VL_RERANKER, {})
    unknown = _control_f(_shipped("qwen3-vl-reranker-2b"))
    assert unknown["kind"] == "unresolved" and "preprocessor_config.json" in unknown["reason"], unknown
    hub_cache(tmp_path, monkeypatch, *VL_RERANKER, {}, ("preprocessor_config.json", "processor_config.json"))
    row = _control_f(_shipped("qwen3-vl-reranker-2b"))
    assert row["kind"] == "unresolved" and "no pixel budget resolves" in row["reason"]
    summary = controls_summary([{"control": "(f)", "name": row["name"], "equivalence": {"passed": None}}])
    assert summary["passed"] is False and summary["blockers"][0]["control"] == "(f)"


def test_every_control_is_served_or_says_why_not(embed_wave: dict[str, Any]) -> None:
    """A control that does not apply is listed with its reason, never dropped."""
    step = _controls(embed_wave)
    served = {row["control"] for row in step["rows"]}
    skipped = {row["control"]: row["reason"] for row in step["inapplicable"]}
    assert served | set(skipped) == {spec.letter for spec in CONTROLS}
    assert all(skipped.values())
    assert served == {"(b)", "(d)"}


def test_a_variant_is_served_and_asked_under_one_model_name() -> None:
    """The variant's served name is its id, and so is its client's ``model`` (the earlier variants asked for
    the base recipe's name, which the variant's engine does not serve)."""
    for variant in control_variants(load_recipe(RECIPES / "fixture-rerank-pointwise")):
        if variant["kind"] == "recipe":
            assert (
                variant["recipe"].client.get("model")
                == variant["recipe"].id
                == f"fixture-rerank-pointwise.{variant['name']}"
            )
    assert right_cut_tokens(load_recipe(RECIPES / "fixture-rerank-pointwise")) == 40


def test_a_no_op_gate_is_flagged_as_a_control_blocker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mutation: stage 2's rerank gates made a no-op (every summary passes).  The recipe's own gates still
    pass, the flipped-activation control now passes too -- and the wave flags it as a blocker and fails."""
    from rcp_ndcg_test.equivalence import stages

    real = stages._rerank_summary

    def no_op(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {**real(*args, **kwargs), "passed": True}

    monkeypatch.setattr(stages, "_rerank_summary", no_op)
    document = _wave(tmp_path, "fixture-rerank-pointwise", model="--model-pooling LAST --model-needs-template")
    step = _controls(document)
    assert step["passed"] is False and "(c)" in {blocker["control"] for blocker in step["blockers"]}
    assert document["recipes"][0]["state"] == "failed" and document["passed"] is False
    assert "(c)" in {blocker["control"] for blocker in document["control_blockers"]["fixture-rerank-pointwise"]}
    assert "BLOCKER fixture-rerank-pointwise control (c)" in (tmp_path / "wave" / "WAVE.md").read_text(encoding="utf-8")
