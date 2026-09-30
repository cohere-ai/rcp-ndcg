"""Read-only questions about runs, as the CLI and the MCP server ask them."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from rcp_ndcg.cli.main import cli
from rcp_ndcg.errors import MissingInputError
from rcp_ndcg.runs import Pipeline
from rcp_ndcg.runs import inspect as inspect_runs
from tests.runs.conftest import tiny_config


class TestListAndShow:
    def test_list_runs_summarises_each_run_and_tolerates_a_broken_one(self, finished: Path, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        shutil.copytree(finished, runs_dir / finished.name)
        broken = runs_dir / "20200101-000000-broken"
        broken.mkdir()
        (broken / "manifest.json").write_text("{not json", encoding="utf-8")
        rows = inspect_runs.list_runs(runs_dir)
        assert [row["run_id"] for row in rows] == [finished.name, broken.name]
        assert rows[0]["status"] == "completed"
        assert rows[0]["judges"] == ["fake"]
        assert rows[0]["steps"] == dict.fromkeys(["tournament", "rubric", "calibrate", "evaluate"], "completed")
        assert rows[1]["status"] == "unreadable"
        assert inspect_runs.list_runs(tmp_path / "nothing") == []

    def test_get_run_reports_which_artifacts_exist(self, finished: Path) -> None:
        """run show once had its own artifact map ('metrics' where Run.artifacts says 'report', and no config,
        log or jobs)."""
        from rcp_ndcg.runs.run import Run

        payload = inspect_runs.get_run(finished)
        assert payload["manifest"]["run_id"] == finished.name
        present = Run(finished).artifacts()
        assert payload["artifacts"] == {name: name in present for name in payload["artifacts"]}
        # The pipeline ran directly (no job, no run log); the rest, the evaluation report among them, is there.
        assert {name for name, exists in payload["artifacts"].items() if not exists} == {"jobs", "log"}

    def test_pointing_at_the_runs_root_is_a_clear_error(self, finished: Path) -> None:
        with pytest.raises(MissingInputError, match="no manifest at") as caught:
            inspect_runs.get_run(finished.parent)
        assert "not the runs root" in (caught.value.hint or "")


class TestCompareAndExplain:
    def test_they_recompute_from_the_runs_artifacts(self, finished: Path) -> None:
        args = ["eval", "compare", "--run", str(finished), "--bootstrap", "20", "--json"]
        result = CliRunner().invoke(cli, [*args, "--include-reference"])
        assert result.exit_code == 0, result.output
        pairs = json.loads(result.stdout)["data"]["pairs"]
        assert [(pair["system_a"], pair["system_b"]) for pair in pairs] == [("candidates", "judge")]
        explanation, dataset = inspect_runs.explain_query(finished, "q1")
        assert explanation.query_id == "q1" and dataset.queries["q1"].text
        assert {system.system for system in explanation.systems} == {"candidates", "judge"}

    def test_compare_leaves_the_reference_systems_out_unless_asked(self, finished: Path) -> None:
        """A run's candidates and judge orders are references, not systems to compare (the judge scores 1)."""
        result = CliRunner().invoke(cli, ["eval", "compare", "--run", str(finished), "--bootstrap", "20", "--json"])
        error = json.loads(result.stdout)["error"]
        assert result.exit_code == 12, result.output
        assert "--include-reference" in error["hint"]
        baseline = ["--baseline", "candidates"]
        result = CliRunner().invoke(cli, ["eval", "compare", "--run", str(finished), *baseline, "--json"])
        assert result.exit_code == 12, "an explicit baseline alone still leaves one system"

    def test_explain_carries_the_calibrated_criteria_per_document(self, finished: Path) -> None:
        result = CliRunner().invoke(cli, ["eval", "explain", "--run", str(finished), "--query-id", "q1", "--json"])
        assert result.exit_code == 0, result.output
        documents = json.loads(result.stdout)["data"]["systems"][0]["top"]
        assert all(document["criteria"] and document["theta"] is not None for document in documents)

    def test_before_calibration_says_what_is_missing(self, data: Path, tmp_path: Path) -> None:
        pipeline = Pipeline(tiny_config(data, steps=["tournament"]), runs_dir=str(tmp_path / "runs"))
        pipeline.run()
        with pytest.raises(MissingInputError, match="calibration/items.json") as caught:
            inspect_runs.evaluation_report(pipeline.layout.root)
        assert "calibrate step" in (caught.value.hint or "")


class TestStatus:
    def test_every_planned_step_is_listed_with_judge_progress(self, finished: Path, tmp_path: Path) -> None:
        """A poller sees 1/4 steps, not 1/1: steps not started are pending, and judging steps count windows."""
        from rcp_ndcg.runs import RunManifest, RunStatus
        from rcp_ndcg.runs.layout import RunLayout
        from rcp_ndcg.runs.run import Run

        run_dir = tmp_path / finished.name
        shutil.copytree(finished, run_dir)
        complete = Run(run_dir).status()
        assert complete.done is True and complete.status == "completed"
        tournament = next(step for step in complete.steps if step.name == "tournament")
        assert tournament.progress is not None and 0 < tournament.progress.done <= tournament.progress.planned

        manifest = RunManifest.load(run_dir)
        manifest.steps = [record for record in manifest.steps if record.name == "tournament"]
        manifest.status = RunStatus.RUNNING
        manifest.save(RunLayout.at(run_dir))
        result = CliRunner().invoke(cli, ["run", "status", "--run", str(run_dir), "--json"])

        state = json.loads(result.stdout)["data"]
        assert result.exit_code == 0, result.output
        assert state["done"] is False
        assert [(step["name"], step["status"]) for step in state["steps"]] == [
            ("tournament", "completed"),
            ("rubric", "pending"),
            ("calibrate", "pending"),
            ("evaluate", "pending"),
        ]
        rubric = state["steps"][1]["progress"]
        assert rubric["planned"] > 0 and rubric["done"] > 0, "the rubric store is there from the copied run"
