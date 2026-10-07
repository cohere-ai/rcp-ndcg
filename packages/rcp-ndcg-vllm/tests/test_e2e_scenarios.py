"""The T4 scenarios: the schema, the run configs they materialize, and the phased script they render.

The scenarios drive the product (R30): a scenario materializes a product ``RunConfig`` (its validators
decide), and the phased job script is the product's SLURM rendering.  These tests pin the materialization
and the rendering -- the latter as a golden file (GPU-VALIDATION.md, T4's offline counterparts).
Everything here runs on CPU with no engine and no network beyond loading the recipe files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rcp_ndcg_vllm.e2e import (
    Scenario,
    build_run_config,
    compare_runs,
    default_scenarios_root,
    iter_scenarios,
    load_scenario,
    render_phased_script,
    scenario_json_schema,
)
from rcp_ndcg_vllm.recipe import load_recipe

RECIPES = Path(__file__).resolve().parents[1] / "recipes"
SCENARIOS = default_scenarios_root()
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden"
INSTALL = {"wheelhouse": "/stage/wheelhouse", "constraints": "/stage/requirements-constraints.txt", "version": "0.0.1"}

#: Fixed paths for the golden rendering (a rendered script names its workdir, log dir and run directory).
FIXED_RUNS = Path("/e2e/runs/rcp-text-four-phases")
FIXED_OPTIONS = {
    "log_dir": "/e2e/out/logs",
    "workdir": str(FIXED_RUNS),
    "env": {"PYTHONPATH": "/e2e/out/probe-site", "RCP_E2E_PROBE_JSONL": "/e2e/out/client-probe.jsonl"},
}


def _names() -> list[str]:
    return sorted(path.stem for path in SCENARIOS.glob("*.yaml"))


def _prepared(name: str):
    """The product's prepared pipeline of one scenario (nothing written beside the run directory)."""
    from rcp_ndcg.runs.run import prepare

    scenario = load_scenario(SCENARIOS / f"{name}.yaml")
    config = build_run_config(scenario, recipes_root=[RECIPES])
    return scenario, prepare(config, runs_dir="/e2e/runs", label=scenario.id)


def test_every_scenario_validates() -> None:
    """Every shipped scenario loads (frozen schema, unknown keys refused) and is named by its file."""
    scenarios = iter_scenarios(SCENARIOS)
    assert [scenario.id for scenario in scenarios] == _names()
    for scenario in scenarios:
        assert scenario.description.strip()


@pytest.mark.parametrize("name", _names())
def test_every_scenario_recipes_load(name: str) -> None:
    """Each scenario's recipe loads against the current product (R30: the recipe's ``client`` block is the
    product's endpoint config, constructed at load with the product's messages).

    This is the failing-test-first of the recipe fixes this lane made: a scenario whose recipe does not
    load materializes no run.
    """
    scenario = load_scenario(SCENARIOS / f"{name}.yaml")
    for recipe_id in (scenario.encoder_recipe, scenario.rerank_recipe):
        if recipe_id is not None:
            load_recipe(RECIPES / recipe_id)


@pytest.mark.parametrize("name", _names())
def test_every_scenario_materializes_its_run_config(name: str) -> None:
    """The product's ``RunConfig`` of each scenario validates, with the phases the scenarios are named for."""
    scenario = load_scenario(SCENARIOS / f"{name}.yaml")
    config = build_run_config(scenario, recipes_root=[RECIPES])
    assert config.ordered_steps == [step for step in scenario.steps]
    assert config.judge_config().base_url
    if scenario.encoder_recipe is not None:
        retrieval = config.candidates.retrieval
        assert retrieval is not None and retrieval.encoder.base_url is None  # the engine sets it at runtime
    if scenario.rerank_recipe is not None:
        assert config.candidates.rerank is not None
    assert config.serve is not None
    serve = config.serve
    if scenario.mode == "outage":
        assert serve.judge is None  # the driver owns the judge; the script must not watch it
    else:
        assert serve.judge is not None
    for role in ("encoder", "reranker"):
        engine = getattr(serve, role)
        if engine is not None:
            assert engine.image is None  # container_runtime: none: the command runs on the node
            assert engine.env["CUDA_VISIBLE_DEVICES"]  # node-runtime item 7: a device slice, never all


def test_the_identity_rerun_moves_no_identity_field() -> None:
    """Ports move between the identity scenario's runs and no identity follows them (nothing recomputes)."""
    from rcp_ndcg.runs.pipeline import Pipeline
    from rcp_ndcg.runs.run import prepare
    from rcp_ndcg.support.identity import hash_payload

    scenario = load_scenario(SCENARIOS / "identity.yaml")
    base = build_run_config(scenario, recipes_root=[RECIPES], port_offset=0)
    moved = build_run_config(scenario, recipes_root=[RECIPES], port_offset=scenario.identity_port_offset)
    assert base.judge_config().base_url != moved.judge_config().base_url  # runtime: the URL differs
    left = Pipeline(prepare(base, runs_dir="/e2e/runs", label=scenario.id).config)
    right = Pipeline(prepare(moved, runs_dir="/e2e/runs", label=scenario.id).config)
    for stage in ("tournament", "rubric"):
        # The product's own identity builder, exercised from outside exactly as the run does.
        assert hash_payload(left._identity(stage)) == hash_payload(right._identity(stage))  # noqa: SLF001


def test_the_four_phase_script_is_the_golden_file() -> None:
    """The rendered phased script of scenario 1, byte for byte (the in-pod job.sh and the golden copy)."""
    scenario, pipeline = _prepared("text-four-phases")
    script = render_phased_script(pipeline, options=FIXED_OPTIONS, run_dir=FIXED_RUNS, **INSTALL)
    script = script.replace(f"rcp-{pipeline.layout.run_id}", "rcp-RUN-ID")  # the run id names the job
    golden = GOLDEN / "job-text-four-phases.sh"
    if not golden.exists():  # the first write is deliberate; the test pins the file afterwards
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(script, encoding="utf-8")
        pytest.fail(f"wrote {golden} for the first time; review and commit it")
    assert script == golden.read_text(encoding="utf-8")


def test_the_script_installs_every_coordinator_from_the_staged_wheelhouse() -> None:
    """Node-runtime items 1-2: each phase's coordinator runs through ``uvx`` from the staged wheelhouse
    (``--find-links`` + ``--no-index`` + the staged constraints), and the engine commands run as-is."""
    _, pipeline = _prepared("text-four-phases")
    script = render_phased_script(pipeline, options=FIXED_OPTIONS, run_dir=FIXED_RUNS, **INSTALL)
    coordinators = [line for line in script.splitlines() if line.startswith("exec ") and " rcp-ndcg run resume" in line]
    assert coordinators  # one per phase
    for line in coordinators:
        assert "--find-links /stage/wheelhouse" in line
        assert "--no-index" in line
        assert "--constraints /stage/requirements-constraints.txt" in line
        assert "rcp-ndcg[calibrate,hf,s3,azure]==0.0.1" in line  # the client mechanism's own spec
    assert any("vllm serve" in line for line in script.splitlines())  # the engines run as-is
    assert "PYTHONPATH=/e2e/out/probe-site" in script  # the probe rides the phase workers, not the engines


def test_the_identity_scenario_rerender_points_at_the_same_run() -> None:
    """The identity rerun renders the same run directory on new ports (``run_dir`` re-points the phases)."""
    _, pipeline = _prepared("identity")
    rerun = render_phased_script(pipeline, options=FIXED_OPTIONS, run_dir="/e2e/runs/the-first-run", **INSTALL)
    assert "--run /e2e/runs/the-first-run" in rerun
    assert "8220" not in rerun  # port offsets move the engine commands and URLs only at their owner's render


def test_the_two_identical_runs_comparison_states_what_it_compares() -> None:
    """The comparison's contract (judgements may differ at temperature > 0): what is compared is
    identities, the deterministic steps' outputs byte for byte, and the windows' counts and families."""
    from rcp_ndcg_vllm.e2e import run_facts

    facts = {"families": ["f"], "windows": {"tournament": 2, "rubric": 3}, "steps": [{"name": "retrieve"}]}
    assert compare_runs(facts, dict(facts))["ok"]
    changed = {**facts, "steps": [{"name": "retrieve", "identity_hash": "x"}]}
    assert not compare_runs(facts, changed)["ok"]
    documented = run_facts.__doc__
    for phrase in ("temperature", "compared", "byte for byte"):
        assert documented is not None and phrase in documented


def test_the_scenario_schema_is_exported_and_current() -> None:
    """``schema/scenario.schema.json`` is the frozen :class:`Scenario` schema, kept current."""
    schema = json.loads((Path(__file__).resolve().parents[1] / "schema" / "scenario.schema.json").read_text("utf-8"))
    assert schema == scenario_json_schema()


def test_an_unknown_scenario_key_is_refused() -> None:
    """The schema is closed: a typo fails at load, not at GPU time."""
    with pytest.raises(ValueError, match="typo"):
        Scenario.model_validate({**_valid(), "typo": 1})


def _valid() -> dict:
    return {
        "id": "x",
        "description": "d",
        "dataset": {"uri": "jsonl:rows.jsonl"},
        "steps": ["tournament", "rubric", "calibrate", "evaluate"],
        "judge": {
            "command": ["vllm", "serve", "m", "--port", "{port}"],
            "candidate": {"model": "org/m", "revision": "0" * 40},
            "config": {"base_url": "http://127.0.0.1:1/v1"},
            "slot": {"port": 8120, "cuda_visible_devices": "0"},
        },
    }
