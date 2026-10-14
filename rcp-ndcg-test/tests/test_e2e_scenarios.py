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
from rcp_ndcg_test.e2e import (
    Scenario,
    build_run_config,
    compare_runs,
    default_scenarios_root,
    find_recipe,
    iter_scenarios,
    load_scenario,
    render_phased_script,
    scenario_json_schema,
)
from rcp_ndcg_test.errors import RecipeError
from rcp_ndcg_vllm.patches import PATCHES_ENV
from rcp_ndcg_vllm.recipe import default_recipes_root, load_recipe

RECIPES = default_recipes_root()
SCENARIOS = default_scenarios_root()
GOLDEN = Path(__file__).resolve().parent / "fixtures" / "golden"
INSTALL = {"wheelhouse": "/stage/wheelhouse", "constraints": "/stage/requirements-constraints.txt", "version": "0.0.1"}

#: Fixed paths for the golden rendering (a rendered script names its workdir, log dir and run directory).
FIXED_RUNS = Path("/e2e/runs/rcp-text-four-phases")
FIXED_OUT = Path("/e2e/out")
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
            load_recipe(recipe_id, root=RECIPES)  # the scenario names a variant id (decision 34)


def test_find_recipe_reports_a_broken_family_not_an_unknown_id(tmp_path: Path) -> None:
    """A root whose family is broken reports its load error, not "no recipe under [...]".

    The regression: the first-root lookup swallowed every ``RecipeError``, so a broken family read as an
    unknown id and the real diagnostic (the file and the validation error) was lost.
    """
    broken = tmp_path / "broken"
    (broken / "broken-family").mkdir(parents=True)
    (broken / "broken-family" / "family.yaml").write_text("id: broken-family\nvariants: []\n", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RecipeError) as raised:
        find_recipe([broken, empty], "broken-family")
    message = str(raised.value)
    assert "family.yaml" in message
    assert "no recipe" not in message


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
            # The shipped scenarios' recipes opt into no patch: the engine env carries the explicit empty
            # value, never an inherited one (the fingerprint keys the patches, so the process must match).
            assert engine.env[PATCHES_ENV] == ""


def test_the_serve_config_renders_the_recipes_patches(tmp_path: Path) -> None:
    """The e2e driver's engine config carries the recipe's declared patches (empty when none), the same
    rendering the serve console and the wave runner use: the engine runs the code the fingerprint keys."""
    from rcp_ndcg_test import e2e as e2e_module

    slot = e2e_module.EngineSlot(port=8100, vllm_port=8200, cuda_visible_devices="0", gpus=1)
    patched = e2e_module._serve_config(["vllm", "serve", "m"], slot, tmp_path, patches=("pooling-full-context",))
    assert patched.env[PATCHES_ENV] == "pooling-full-context"
    none = e2e_module._serve_config(["vllm", "serve", "m"], slot, tmp_path, patches=())
    assert none.env[PATCHES_ENV] == ""


def test_the_identity_rerun_moves_no_identity_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ports move between the identity scenario's runs and no identity follows them (nothing recomputes).

    Every role config's identity carries its tokenizer's content digest; offline, the digest is a stand-in
    keyed on the tokenizer spec (both runs name the same tokenizers, so only the ports differ).
    """
    import hashlib

    import rcp_ndcg.data.tokenizer
    from rcp_ndcg.runs.pipeline import Pipeline
    from rcp_ndcg.runs.run import prepare
    from rcp_ndcg.support.identity import hash_payload

    monkeypatch.setattr(
        rcp_ndcg.data.tokenizer,
        "tokenizer_identity",
        lambda spec: {"tokenizer_sha256": hashlib.sha256(spec.encode()).hexdigest()},
    )

    scenario = load_scenario(SCENARIOS / "identity.yaml")
    base = build_run_config(scenario, recipes_root=[RECIPES], port_offset=0)
    moved = build_run_config(scenario, recipes_root=[RECIPES], port_offset=scenario.identity_port_offset)
    assert base.judge_config().base_url != moved.judge_config().base_url  # runtime: the URL differs
    left = Pipeline(prepare(base, runs_dir="/e2e/runs", label=scenario.id).config)
    right = Pipeline(prepare(moved, runs_dir="/e2e/runs", label=scenario.id).config)
    for stage in ("tournament", "rubric"):
        # The product's own identity builder, exercised from outside exactly as the run does.
        assert hash_payload(left._identity(stage)) == hash_payload(right._identity(stage))  # noqa: SLF001


def test_the_four_phase_script_is_the_golden_file(tmp_path: Path) -> None:
    """The rendered phased script of scenario 1, byte for byte (the in-pod job.sh and the golden copy).

    The options are the driver's own (:func:`rcp_ndcg_test.e2e.render_options` -- the golden pins the
    exact byte stream `_prepare_run` executes).  Two paths are normalized out of the comparison (they name
    where the render ran, not what it is): the run id and the recipes root's absolute path (``serve_argv``
    renders ``--chat-template`` absolutely, from wherever the recipes live).
    """
    from rcp_ndcg_test.e2e import render_options

    scenario, pipeline = _prepared("text-four-phases")
    script = render_phased_script(
        pipeline, options=render_options(FIXED_OUT, FIXED_RUNS, pipeline=pipeline), run_dir=FIXED_RUNS, **INSTALL
    )
    script = script.replace(f"rcp-{pipeline.layout.run_id}", "rcp-RUN-ID")  # the run id names the job
    script = script.replace(str(RECIPES), "/e2e/recipes")  # the recipes' absolute root (this checkout's)
    assert "mkdir -p /tmp/rcp-e2e-text-four-phases/tmp-8100 " in script  # each slot's TMPDIR, created by setup
    assert "if ! command -v uvx" in script  # the driver's setup bootstrap: the golden pins what runs
    golden = GOLDEN / "job-text-four-phases.sh"
    if not golden.exists():  # never write the checkout from a test: the candidate lands in tmp_path
        candidate = tmp_path / "job-text-four-phases.sh"
        candidate.write_text(script, encoding="utf-8")
        pytest.fail(f"no golden file; copy {candidate} to {golden} after reviewing it, and commit it")
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


def test_the_identity_rerun_moves_every_port_together() -> None:
    """The identity rerun on new ports is internally consistent: the engine commands listen on the ports
    the script waits for and hands the coordinators (command, ``ServeConfig.port``, the engine script's
    ``VLLM_PORT``/``TMPDIR``, the ``RCP_NDCG_ENGINES`` overlay and the judge's URL move together), and
    the phases still point at the first run's directory."""
    scenario = load_scenario(SCENARIOS / "identity.yaml")
    offset = scenario.identity_port_offset
    config = build_run_config(scenario, recipes_root=[RECIPES], port_offset=offset)
    serve = config.serve
    assert serve is not None
    slots = {"encoder": scenario.slots["encoder"], "reranker": scenario.slots["reranker"], "judge": scenario.judge.slot}
    for role in ("encoder", "reranker", "judge"):
        engine = getattr(serve, role)
        if engine is None:
            continue
        slot = slots[role]
        assert engine.port == slot.port + offset, role  # the overlay, the wait and the URL's port
        joined = " ".join(engine.command)
        assert f"--port {engine.port}" in joined, role  # the command listens where the script points
        assert engine.env["VLLM_PORT"] == str(slot.vllm_port + offset), role
        assert f"tmp-{engine.port}" in engine.env["TMPDIR"], role
    assert config.judge_config().base_url == f"http://127.0.0.1:{serve.judge.port}/v1"

    from rcp_ndcg.runs.run import prepare as prepare_run

    moved = build_run_config(scenario, recipes_root=[RECIPES], port_offset=offset)
    rerun_pipeline = prepare_run(moved, runs_dir="/e2e/runs", label=scenario.id)
    rerun = render_phased_script(rerun_pipeline, options=FIXED_OPTIONS, run_dir="/e2e/runs/the-first-run", **INSTALL)
    assert "--run /e2e/runs/the-first-run" in rerun  # the rerun drives the first run's directory
    moved_serve = moved.serve
    assert moved_serve is not None and moved_serve.judge is not None
    port = moved_serve.judge.port
    assert "rcp_ndcg_wait_ready RCP_NDCG_ENGINE_PID" in rerun  # the phases wait for their engines
    assert f"{port} /v1/models" in rerun  # ... on the port the command binds
    assert f'"judge": {{"urls": ["http://127.0.0.1:{port}/v1"]' in rerun  # coordinators point at it too


def test_the_outage_scenario_window_covers_the_restart_boot() -> None:
    """The shipped outage scenario's ``wait_on_outage_s`` covers the recovers sub-run's whole down span.

    The judge is down for the outage sleep **and** for the restarted engine's boot.  The rc0 outage run
    (waves-rc0-202610141005) measured this node's judge at 77 s per warm restart and 185 s for the T0
    smoke's cold boot; against the scenario's 90 s window the 15 s outage ran ~2 s past the window and
    the sub-run failed the way its expiry twin is meant to.  The scenario's numbers must pass
    :func:`rcp_ndcg_test.e2e.recovers_outage_s` for boots of the size this node has shown.
    """
    from rcp_ndcg_test.e2e import recovers_outage_s

    scenario = load_scenario(SCENARIOS / "outage.yaml")
    wait_s = float(scenario.judge.config["wait_on_outage_s"])
    for boot_s in (77.0, 185.0):  # the warm restart and the cold boot the rc0 run measured
        outage_s = recovers_outage_s(wait_s, boot_s)  # raises when the numbers cannot recover
        assert outage_s + boot_s < wait_s  # the judge is back before the run's window expires


def test_the_two_identical_runs_comparison_states_what_it_compares() -> None:
    """The comparison's contract (judgements may differ at temperature > 0): what is compared is
    identities, the deterministic steps' ranking tables (their scores within the served engine's numeric
    noise), and the windows' counts and families."""
    from rcp_ndcg_test.e2e import run_facts

    facts = {"families": ["f"], "windows": {"tournament": 2, "rubric": 3}, "steps": [{"name": "retrieve"}]}
    assert compare_runs(facts, dict(facts))["ok"]
    changed = {**facts, "steps": [{"name": "retrieve", "identity_hash": "x"}]}
    assert not compare_runs(facts, changed)["ok"]
    documented = run_facts.__doc__
    for phrase in ("temperature", "compared", "numeric noise"):
        assert documented is not None and phrase in documented


def _ranking_run(root: Path, table: dict[str, dict[str, float]]) -> Path:
    """A run directory whose retrieve and rerank steps recorded ``table`` as their rankings artifacts."""
    from rcp_ndcg.data import Rankings
    from rcp_ndcg.runs.layout import RunLayout
    from rcp_ndcg.runs.manifest import RunManifest, RunStatus, StepRecord, StepStatus
    from rcp_ndcg.storage.artifacts import artifact_ref

    root.mkdir()
    layout = RunLayout.at(root)
    first = Path(layout.path("work", "first_stage.parquet"))
    first.parent.mkdir(parents=True)
    rankings = Rankings.from_scores(table, system="candidates")
    rankings.save(first, format="parquet")
    rankings.save(layout.candidates, format="parquet")
    judgements = Path(layout.path("judgements", "tournament.jsonl"))
    judgements.parent.mkdir(parents=True)
    judgements.write_text("window 1\nwindow 2\n", encoding="utf-8")
    manifest = RunManifest.new(root.name, config={})
    manifest.status = RunStatus.COMPLETED
    manifest.steps = [
        StepRecord(
            name="retrieve",
            status=StepStatus.COMPLETED,
            identity={"step": "retrieve"},
            identity_hash="r",
            outputs=[artifact_ref(first, layout=layout)],
        ),
        StepRecord(
            name="rerank",
            status=StepStatus.COMPLETED,
            identity={"step": "rerank"},
            identity_hash="k",
            inputs=[artifact_ref(first, layout=layout)],
            outputs=[artifact_ref(layout.candidates, layout=layout)],
        ),
        StepRecord(
            name="tournament",
            status=StepStatus.COMPLETED,
            identity={"step": "tournament"},
            identity_hash="t",
            outputs=[artifact_ref(judgements, layout=layout)],
        ),
    ]
    manifest.save(layout)
    return root


def test_the_pair_comparison_reads_the_ranking_tables_not_their_bytes(tmp_path: Path) -> None:
    """Two identical runs whose retrieve table differs only by the engine's numeric noise compare equal.

    A served engine is not bit-deterministic: the rc0 pair's ``work/first_stage.parquet`` held 3 of 1500
    scores moved by at most 3e-8 absolute / 9.5e-8 relative (the engine's reduction order moves within
    its batching strata), so the parquet's bytes differ between two identical runs while the table does
    not.  The comparison reads the table -- keys exact, scores within the noise bound -- and still fails
    a table whose scores really moved.
    """
    from rcp_ndcg_test.e2e import run_facts

    table = {"PLAIN-1": {"MED-1": 0.9, "MED-2": 0.5}, "PLAIN-2": {"MED-1": 0.25, "MED-2": 0.75}}
    noisy = {
        "PLAIN-1": {"MED-1": 0.9 + 3e-8, "MED-2": 0.5},
        "PLAIN-2": {"MED-1": 0.25, "MED-2": 0.75 - 3e-8},
    }
    left = _ranking_run(tmp_path / "left", table)
    right = _ranking_run(tmp_path / "right", noisy)
    # The premise of the check: the two runs' artifacts are not byte-identical ...
    assert (left / "work" / "first_stage.parquet").read_bytes() != (right / "work" / "first_stage.parquet").read_bytes()
    # ... while the tables are equal within the engine's numeric noise.
    check = compare_runs(run_facts(left), run_facts(right))
    assert check["ok"], check["detail"]

    moved = _ranking_run(
        tmp_path / "moved",
        {"PLAIN-1": {"MED-1": 0.9, "MED-2": 0.5}, "PLAIN-2": {"MED-1": 0.25, "MED-2": 0.7}},
    )
    check = compare_runs(run_facts(left), run_facts(moved))
    assert not check["ok"]
    assert "retrieve" in check["detail"] and "noise" in check["detail"]


def test_the_scenario_schema_is_exported_and_current() -> None:
    """``schema/scenario.schema.json`` is the frozen :class:`Scenario` schema, kept current."""
    schema = json.loads((Path(__file__).resolve().parents[1] / "schema" / "scenario.schema.json").read_text("utf-8"))
    assert schema == scenario_json_schema()


def test_the_outage_judge_keeps_its_devices_to_itself() -> None:
    """The outage scenario's judge is the driver's run-scoped engine: it stays up while the phases run, so
    its device slice must not overlap the encoder's or the reranker's (the judge's vLLM pre-allocates the
    device it sees).

    The T4's second run found this as the encoder phase's ``CUDA out of memory`` (the driver's 27B judge
    held GPU 0 when the phase's engine tried to allocate there), and the resume that followed could not
    finish the run.  The shipped scenario obeys the rule, and a scenario that does not is refused by name.
    """
    scenario = load_scenario(SCENARIOS / "outage.yaml")
    judge_devices = _devices(scenario.judge.slot.cuda_visible_devices)
    for role, slot in scenario.slots.items():
        assert not judge_devices & _devices(slot.cuda_visible_devices), role
    data = _valid()
    data["mode"] = "outage"
    data["steps"] = ["retrieve", "tournament"]
    data["encoder_recipe"] = "some-encoder"
    data["slots"] = {"encoder": {"port": 8100, "vllm_port": 9100, "cuda_visible_devices": "0"}}
    data["judge"] = {**data["judge"], "slot": {"port": 8120, "vllm_port": 9120, "cuda_visible_devices": "0,1"}}
    with pytest.raises(ValueError, match="shares device"):
        Scenario.model_validate(data)


def _devices(value: str) -> set[str]:
    """The device ids a slot's ``cuda_visible_devices`` names."""
    return {device.strip() for device in value.split(",") if device.strip()}


def test_slot_ports_never_collide_across_roles() -> None:
    """One engine per phase, but the roles share a node: every ``port`` and every ``vllm_port`` is
    unique (an unset ``vllm_port`` defaults to 8100 and would collide loudly here)."""
    data = _valid()
    data["steps"] = ["retrieve", "calibrate", "evaluate"]
    data["encoder_recipe"] = "some-encoder"
    data["slots"] = {"encoder": {"port": 8100, "cuda_visible_devices": "0"}}
    with pytest.raises(ValueError, match="vllm_port"):
        Scenario.model_validate(
            {**data, "judge": {**data["judge"], "slot": {"port": 8120, "vllm_port": 8100, "cuda_visible_devices": "0"}}}
        )
    with pytest.raises(ValueError, match="port"):
        Scenario.model_validate(
            {**data, "judge": {**data["judge"], "slot": {"port": 8100, "vllm_port": 9100, "cuda_visible_devices": "0"}}}
        )


def test_a_missing_scenario_list_is_a_bad_request_not_a_crash() -> None:
    """``main``'s documented contract: 2 for a bad request, never a traceback."""
    from rcp_ndcg_test.e2e import main

    status = main(
        [
            "--scenarios",
            "@does-not-exist.txt",
            "--recipes-root",
            str(RECIPES),
            "--out",
            "/e2e/unused-out",
            "--wheelhouse",
            "/stage/wheelhouse",
        ]
    )
    assert status == 2


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
            "recipe": "gpt-oss-120b",
            "config": {"wait_on_outage_s": 300},
            "slot": {"port": 8120, "cuda_visible_devices": "0"},
        },
    }


def test_a_judge_slot_that_disagrees_with_the_recipe_is_refused() -> None:
    """The recipe renders --tensor-parallel-size from its resources.gpus and the slot sizes --gres around
    it: two numbers here is one incoherent engine, refused by name."""
    from rcp_ndcg_test.errors import HarnessError

    data = _valid()
    data["judge"] = {**data["judge"], "slot": {"port": 8120, "cuda_visible_devices": "0,1", "gpus": 2}}
    scenario = Scenario.model_validate(data)
    with pytest.raises(HarnessError, match="gpus 2 but the recipe serves on 1"):
        build_run_config(scenario, recipes_root=[RECIPES])


def test_a_content_override_in_the_scenario_judge_config_is_refused() -> None:
    """The scenario's judge.config may carry runtime fields only (decision 17): the recipe is the source of
    the client block, and a CONTENT field there is refused against the product's own role declaration."""
    from rcp_ndcg_test.errors import HarnessError

    data = _valid()
    data["judge"] = {**data["judge"], "config": {"temperature": 0.5}}
    scenario = Scenario.model_validate(data)
    with pytest.raises(HarnessError, match="CONTENT field"):
        build_run_config(scenario, recipes_root=[RECIPES])
