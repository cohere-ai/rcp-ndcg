"""The skill's insertion recipe, command by command: plan the windows, judge exactly those into the store, insert."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from rcp_ndcg.cli.main import cli
from rcp_ndcg.data import Rankings
from rcp_ndcg.examples import tiny

ROWS = tiny() / "rows.jsonl"
SCHEDULE = [
    "--set", "schedule.window=4", "--set", "schedule.random_windows=4", "--set", "schedule.stratified_windows=2",
    "--set", "schedule.adaptive_window=4", "--set", "schedule.adaptive_batches=1",
    "--set", "schedule.adaptive_windows_per_batch=2",
]  # fmt: skip


def _json(*args: str) -> dict:
    result = CliRunner().invoke(cli, [*args, "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["data"]


def test_a_new_document_is_inserted_by_judging_exactly_the_planned_windows_into_the_store(tmp_path: Path) -> None:
    rows = [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines()]
    pools = tmp_path / "pools.jsonl"
    Rankings.from_orders({r["query_id"]: [d for d in r["doc_ids"] if d != "q1-d1"] for r in rows}, system="p").save(
        pools
    )
    store, calibration = str(tmp_path / "store"), str(tmp_path / "calibration")
    dataset = ["--dataset", f"jsonl:{ROWS}", "--judge", "fake"]
    _json("judge", "tournament", *dataset, "--candidates", str(pools), *SCHEDULE, "--out", store)
    rubric = [
        "--set",
        "schedule.window=4",
        "--set",
        "schedule.windows_per_query=8",
        "--set",
        "schedule.random_windows=4",
    ]
    _json("judge", "rubric", *dataset, "--candidates", str(pools), *rubric, "--out", store)
    _json("calibration", "fit", "--judgements", store, "--out", calibration)

    # The window size comes from the store's schedule; the plan says how many calls judging it takes.
    plan_file = tmp_path / "plan.json"
    plan = _json("calibration", "insert", "--calibration", calibration, "--judgements", store, "--plan",
                 "--query", "q1", "--doc", "q1-d1", "--n", "6", "--out", str(plan_file))["plan"]  # fmt: skip
    assert [len(window) for window in plan["windows"]] == [4, 4]
    assert all(window[0] == "q1-d1" for window in plan["windows"])
    assert plan["calls"] == 4  # two windows, each mirrored by the store's schedule

    estimate = _json("judge", "tournament", *dataset, "--plan", str(plan_file), "--out", store, "--estimate")
    assert estimate["estimate"]["calls"] == plan["calls"]
    judged = _json("judge", "tournament", *dataset, "--plan", str(plan_file), "--out", store)
    assert judged["usage"]["requests"] == plan["calls"]
    # Six opponents in one pass give little evidence: accept a standard error of 1 logit (the default asks 0.5).
    inserted = _json("calibration", "insert", "--calibration", calibration, "--judgements", store,
                     "--se-target", "1.0", "--out", str(tmp_path / "extended"))  # fmt: skip

    assert inserted["extension"]["anchor_report"]["ok"]
    assert [record["doc_id"] for record in inserted["extension"]["records"]] == ["q1-d1"]
