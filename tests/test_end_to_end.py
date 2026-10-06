"""Offline end to end: a tiny world judged by the fake judge, through the CLI, a job runner and the facade.

Every ``--json`` document is checked against the model its command declares (the model the exported JSON Schema is
generated from) and must carry that schema's id.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

import rcp_ndcg as rcp
from rcp_ndcg.cli.introspect import command_specs
from rcp_ndcg.cli.main import cli
from rcp_ndcg.cli.output import CliEnvelope
from rcp_ndcg.testing import TINY_RUBRIC, TINY_TOURNAMENT, tiny_rows


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The tiny world's rows as a JSONL dataset, and a run config over it judged by the fake judge."""
    root = tmp_path_factory.mktemp("e2e")
    rows, _ = tiny_rows()
    data = root / "rows.jsonl"
    data.write_text(
        "".join(
            json.dumps({"query_id": r.id, "query": r.query, "doc_ids": r.doc_ids, "docs": r.docs, "qrels": r.qrels})
            + "\n"
            for r in rows
        ),
        encoding="utf-8",
    )
    config = {
        "label": "e2e",
        "dataset": f"jsonl:{data}",
        "judge": "fake",
        "steps": ["tournament", "rubric", "calibrate", "evaluate"],
        "tournament": TINY_TOURNAMENT.model_dump(),
        "rubric": TINY_RUBRIC.model_dump(),
    }
    path = root / "run.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return {"root": root, "data": data, "config": path}


def _json(*args: str) -> dict:
    """Run a command with ``--json``; check the envelope and the data against the command's declared model."""
    result = CliRunner().invoke(cli, [*args, "--json"])
    document = json.loads(result.stdout)
    CliEnvelope.model_validate(document)
    assert result.exit_code == 0, document
    spec = command_specs()[document["command"]]
    assert document["data"]["schema"] == spec.output_schema
    if spec.result is not None:
        spec.result.model_validate(document["data"])
    return document["data"]


def test_the_cli_runs_judges_calibrates_and_evaluates(world: dict[str, Path]) -> None:
    started = _json("run", "start", str(world["config"]), "--runs-dir", str(world["root"] / "runs"))
    run = started["run_dir"]

    status = _json("run", "status", "--run", run)
    shown = _json("run", "show", "--run", run)
    scored = _json(
        "eval", "score", "--rankings", str(Path(run) / "candidates.parquet"), "--dataset", f"jsonl:{world['data']}",
        "--calibration", run, "--k", "3",
    )  # fmt: skip
    calibration = _json("calibration", "show", "--calibration", run)
    compared = _json("eval", "compare", "--run", run, "--bootstrap", "50", "--include-reference")
    explained = _json("eval", "explain", "--run", run, "--query-id", "q1", "--k", "3", "--include-text")
    logged = _json("run", "logs", "--run", run, "--tail", "5")

    assert started["mode"] == "ran"
    assert [(step["name"], step["status"]) for step in status["steps"]] == [
        ("tournament", "completed"), ("rubric", "completed"), ("calibrate", "completed"), ("evaluate", "completed"),
    ]  # fmt: skip
    assert status["metrics"]["candidates/rcp_ndcg@10"] > 0
    assert {name for name, exists in shown["artifacts"].items() if not exists} == {"jobs"}  # it ran in-process
    assert {row["metric"] for row in scored["summary"]} == {"rcp_ndcg", "qrel_ndcg"}
    assert scored["gains_source"] == "calibration"
    assert calibration["mode"] == "tournament" and calibration["queries"] == 2
    assert [(pair["system_a"], pair["system_b"]) for pair in compared["pairs"]] == [("candidates", "judge")]
    assert explained["query"] == "tiny query q1" and len(explained["systems"][0]["top"]) == 3
    assert "[run]" in logged["text"]


def test_a_job_runner_runs_the_same_pipeline_started_over_mcp(
    world: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """MCP ``run_start`` hands the run to the local job runner in the background (one job, ``rcp-ndcg run resume``
    in a subprocess) and returns its directory at once; ``run_status`` follows it and ``run logs`` reads its log."""
    from rcp_ndcg import mcp

    monkeypatch.setenv("PATH", f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}")
    judge = {"base_url": "fake://seed/0", "model": "fake"}
    arguments = {
        "config": str(world["config"]),
        "label": "job",
        "runs_dir": str(world["root"] / "runs"),
        "set": [f"judge={json.dumps(judge)}"],
    }
    started = mcp.call_tool("run_start", arguments)
    assert not started["isError"], started
    data = started["structuredContent"]
    assert data["mode"] == "submitted" and data["state"]["status"] in ("submitted", "running")

    deadline = time.monotonic() + 120
    while True:
        status = mcp.call_tool("run_status", {"run": data["run_dir"]})["structuredContent"]
        if status["jobs"][0]["status"] in ("completed", "failed", "cancelled"):
            break
        assert time.monotonic() < deadline, status
        time.sleep(0.2)
    logged = _json("run", "logs", "--run", data["run_dir"])

    assert status["jobs"][0]["status"] == "completed" and status["runner"] == "local"
    assert status["status"] == "completed"
    assert "[run] evaluate: starting" in logged["text"]
    assert json.loads(Path(data["run_dir"], "logs", "jobs.json").read_text())["jobs"][0]["handle"]


def test_the_facade_runs_the_same_flow(world: dict[str, Path], tmp_path: Path) -> None:
    dataset = rcp.load_dataset(f"jsonl:{world['data']}")
    judge = rcp.JudgeConfig.fake(0)
    store = tmp_path / "judgements"

    projected = rcp.estimate(dataset, None, judge, stages=("tournament",))
    rcp.judge(dataset, None, judge, stage="tournament", out=store, schedule=TINY_TOURNAMENT)
    judged = rcp.judge(dataset, None, judge, stage="rubric", out=store, schedule=TINY_RUBRIC)
    from rcp_ndcg.calibration import read_judgements

    calibration = rcp.calibrate(read_judgements(store))
    candidates = rcp.Rankings.from_orders(dataset.candidates, system="candidates")
    report = rcp.evaluate(candidates, dataset=dataset, gains=calibration, k=3)
    run = rcp.run(rcp.RunConfig.load(world["config"]), runs_dir=str(tmp_path / "runs"))
    estimate = rcp.run(rcp.RunConfig.load(world["config"]), estimate=True)

    assert projected.calls > 0 and projected.input_tokens > 0
    assert isinstance(judged, rcp.JudgementSet) and len(judged) > 0
    assert isinstance(calibration, rcp.Calibration) and calibration.mode == "tournament"
    assert isinstance(report, rcp.EvalReport) and report.value("candidates", "rcp_ndcg", 3) > 0
    assert isinstance(run, rcp.Run) and run.status().status == "completed"
    assert isinstance(estimate, rcp.CostEstimate) and estimate.calls > 0
    assert {"candidates", "calibration", "report", "log"} <= set(run.artifacts())
