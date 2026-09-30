"""The pipeline, offline: every step on a tiny dataset judged by the fake judge, then resumed."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from rcp_ndcg.data import load_rankings
from rcp_ndcg.errors import ConfigError, DataError, IdentityError, MissingInputError, RcpNdcgWarning
from rcp_ndcg.llm import TournamentSchedule
from rcp_ndcg.runs import Pipeline, RunManifest, RunStatus, StepStatus
from rcp_ndcg.testing import TINY_TOURNAMENT, tiny_rows
from tests.runs.conftest import STEPS, tiny_config


@pytest.fixture
def run_dir(finished: Path, tmp_path: Path) -> Path:
    return Path(shutil.copytree(finished, tmp_path / finished.name))


def _statuses(manifest: RunManifest) -> dict[str, str]:
    return {record.name: record.status.value for record in manifest.steps}


def _order(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda doc: (scores[doc], doc), reverse=True)


def _lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


class TestARun:
    def test_every_step_writes_its_part_of_the_layout(self, finished: Path) -> None:
        manifest = RunManifest.load(finished)
        assert manifest.status is RunStatus.COMPLETED
        assert _statuses(manifest) == dict.fromkeys(STEPS, "completed")
        for name in ("run.yaml", "candidates.parquet", "judgements/tournament.jsonl", "judgements/rubric.jsonl"):
            assert (finished / name).is_file(), name
        for name in ("items.json", "queries.parquet", "thetas.parquet"):
            assert (finished / "calibration" / name).is_file(), name
        assert load_rankings(finished / "candidates.parquet").systems == ["candidates"]
        assert sorted(load_rankings(finished / "candidates.parquet").queries()) == ["q1", "q2"]
        report = json.loads((finished / "metrics" / "report.json").read_text(encoding="utf-8"))
        assert report["schema"] == "rcp-ndcg.eval-report.v1"
        candidates = {
            (row["metric"], row["k"]): row["value"] for row in report["summary"] if row["system"] == "candidates"
        }
        assert {row["system"] for row in report["summary"]} == {"candidates", "judge"}
        # The manifest names the system of every number, and leaves out the judge's own order (1 by construction).
        assert manifest.metrics["candidates/rcp_ndcg@10"] == candidates[("rcp_ndcg", 10)]
        assert not any(key.startswith("judge/") or "/" not in key for key in manifest.metrics)
        comparison = json.loads((finished / "metrics" / "comparison.json").read_text(encoding="utf-8"))
        assert [(pair["system_a"], pair["system_b"]) for pair in comparison["pairs"]] == [("candidates", "judge")]

    def test_the_manifest_records_families_usage_and_identity(self, finished: Path) -> None:
        manifest = RunManifest.load(finished)
        assert {family.stage for family in manifest.families.values()} == {"tournament", "rubric"}
        assert {family.judge_model for family in manifest.families.values()} == {"fake"}
        assert manifest.usage.requests > 0 and manifest.usage.input_tokens > 0
        assert manifest.dataset is not None and manifest.dataset.name == "rows"
        tournament = manifest.step("tournament")
        assert tournament.identity["judge"]["model"] == "fake"
        assert tournament.usage is not None and tournament.usage.requests > 0
        assert [ref.path for ref in tournament.outputs] == ["judgements/tournament.jsonl"]


def test_a_judging_step_records_what_the_judges_endpoint_serves(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rcp_ndcg.llm import JudgeClient
    from tests.llm.test_judging import _SchemaEndpoint

    endpoint = _SchemaEndpoint("3.1")
    monkeypatch.setattr(JudgeClient, "from_config", staticmethod(lambda config: endpoint.client()))
    judge = {"base_url": "http://judge.test/v1", "model": "m"}
    pipeline = Pipeline(tiny_config(data, judge=judge, steps=["tournament"]), runs_dir=str(tmp_path))
    manifest = pipeline.run()
    (engine,) = manifest.step("tournament").engines
    assert (engine.model, engine.max_model_len, engine.system_fingerprint) == ("m", 32768, "engine-3.1")
    assert "engines" not in json.dumps(manifest.step("tournament").identity)


class TestResume:
    def test_current_steps_are_skipped_and_nothing_is_judged_again(self, run_dir: Path) -> None:
        judged = _lines(run_dir / "judgements" / "rubric.jsonl")
        before = RunManifest.load(run_dir).steps
        pipeline = Pipeline.resume(run_dir)
        assert pipeline.plan() == [{"step": step, "status": "would skip"} for step in STEPS]
        manifest = pipeline.run()
        assert _statuses(manifest) == dict.fromkeys(STEPS, "completed"), "a no-op resume shows them completed"
        assert manifest.steps == before, "and leaves their records (times, usage) as they were"
        assert _lines(run_dir / "judgements" / "rubric.jsonl") == judged

    def test_a_changed_calibration_redoes_the_steps_after_the_judging(self, run_dir: Path) -> None:
        manifest = Pipeline.resume(run_dir, overrides=["calibration.priors.sigma_tau=2.0"]).run()
        assert _statuses(manifest) == dict.fromkeys(STEPS, "completed")
        assert manifest.step("calibrate").ended_at > manifest.step("rubric").ended_at

    def test_a_missing_output_redoes_its_step(self, run_dir: Path) -> None:
        (run_dir / "metrics" / "report.json").unlink()
        manifest = Pipeline.resume(run_dir).run()
        assert _statuses(manifest)["evaluate"] == "completed"
        assert manifest.step("calibrate").ended_at < manifest.step("evaluate").started_at
        assert (run_dir / "metrics" / "report.json").is_file()

    def test_without_resume_every_step_runs_again(self, run_dir: Path) -> None:
        manifest = Pipeline.resume(run_dir).run(resume=False)
        assert _statuses(manifest) == dict.fromkeys(STEPS, "completed")

    def test_a_changed_schedule_is_refused_and_leaves_the_run_resumable(self, run_dir: Path) -> None:
        # The store holds judgements of the old schedule; mixing the two would change what the fit reads.
        recorded = RunManifest.load(run_dir)
        run_yaml = (run_dir / "run.yaml").read_text(encoding="utf-8")
        pipeline = Pipeline.resume(run_dir, overrides=["rubric.seed=7"])
        assert pipeline.plan()[1] == {"step": "rubric", "status": "would run"}
        with pytest.raises(IdentityError) as refused:
            pipeline.run()
        assert "--force" not in (refused.value.hint or "") and "new run" in refused.value.hint

        # The refusal changed nothing: the recorded config, the run's status and the step records stand.
        manifest = RunManifest.load(run_dir)
        assert manifest.config == recorded.config
        assert (run_dir / "run.yaml").read_text(encoding="utf-8") == run_yaml
        assert manifest.status is RunStatus.COMPLETED
        assert _statuses(manifest) == dict.fromkeys(STEPS, "completed")
        assert _statuses(Pipeline.resume(run_dir).run()) == dict.fromkeys(STEPS, "completed")


class TestFailure:
    def test_a_failing_step_fails_the_run(self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(*args, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr("rcp_ndcg.calibration.calibrate", broken)
        pipeline = Pipeline(tiny_config(data), runs_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="boom"):
            pipeline.run()
        manifest = RunManifest.load(pipeline.layout.root)
        assert manifest.status is RunStatus.FAILED
        assert manifest.step("calibrate").status is StepStatus.FAILED
        assert manifest.step("calibrate").error == "RuntimeError: boom"


class TestAFailedChange:
    """A resume that changes the config and then fails leaves the run as it was: run.yaml, config and status."""

    def test_a_failed_override_leaves_the_run_as_it_was(self, run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken(*args, **kwargs):
            raise RuntimeError("boom")

        recorded = RunManifest.load(run_dir)
        run_yaml = (run_dir / "run.yaml").read_text(encoding="utf-8")
        monkeypatch.setattr("rcp_ndcg.calibration.calibrate", broken)
        with pytest.raises(RuntimeError, match="boom"):
            Pipeline.resume(run_dir, overrides=["calibration.priors.sigma_tau=3.0"]).run()

        manifest = RunManifest.load(run_dir)
        assert manifest.config == recorded.config
        assert (run_dir / "run.yaml").read_text(encoding="utf-8") == run_yaml
        assert manifest.status is RunStatus.COMPLETED
        assert _statuses(manifest) == dict.fromkeys(STEPS, "completed")
        monkeypatch.undo()
        assert _statuses(Pipeline.resume(run_dir).run()) == dict.fromkeys(STEPS, "completed")

    def test_a_step_rerun_before_the_failure_is_redone_under_the_recorded_config(
        self, run_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # calibrate reruns under the new priors, then evaluate fails: the config goes back, so calibrate is redone.
        def broken(*args, **kwargs):
            raise RuntimeError("boom")

        recorded = RunManifest.load(run_dir)
        monkeypatch.setattr("rcp_ndcg.eval.evaluate", broken)
        with pytest.raises(RuntimeError, match="boom"):
            Pipeline.resume(run_dir, overrides=["calibration.priors.sigma_tau=3.0"]).run()
        assert RunManifest.load(run_dir).config == recorded.config
        monkeypatch.undo()
        plan = {row["step"]: row["status"] for row in Pipeline.resume(run_dir).plan()}
        assert plan["calibrate"] == "would run"

    def test_a_failure_after_judging_under_the_new_config_keeps_it(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The store now holds judgements of the new config: going back would make every later resume refuse.
        pipeline = Pipeline(tiny_config(data, steps=["tournament"]), runs_dir=str(tmp_path))
        pipeline.run()

        def broken(self, request, *args, **kwargs):
            raise RuntimeError("endpoint gone")

        monkeypatch.setattr("rcp_ndcg.llm._fake.FakeJudge._send", broken)
        with pytest.raises(RuntimeError, match="endpoint gone"):
            Pipeline.resume(pipeline.layout.root, overrides=["steps=[tournament, rubric]"]).run()
        manifest = RunManifest.load(pipeline.layout.root)
        assert manifest.config["steps"] == ["tournament", "rubric"]
        assert manifest.status is RunStatus.FAILED and manifest.step("rubric").status is StepStatus.FAILED
        monkeypatch.undo()
        assert _statuses(Pipeline.resume(pipeline.layout.root).run()) == {
            "tournament": "completed",
            "rubric": "completed",
        }


class TestOnly:
    def test_only_runs_the_named_steps_and_never_rewrites_the_recorded_steps(self, run_dir: Path) -> None:
        recorded = RunManifest.load(run_dir)
        run_yaml = (run_dir / "run.yaml").read_text(encoding="utf-8")
        pipeline = Pipeline.resume(run_dir, only=["evaluate"])
        assert pipeline.plan(resume=False) == [{"step": "evaluate", "status": "would run"}]
        manifest = pipeline.run(resume=False)

        assert manifest.config == recorded.config
        assert (run_dir / "run.yaml").read_text(encoding="utf-8") == run_yaml
        assert manifest.status is RunStatus.COMPLETED
        assert [row["step"] for row in Pipeline.resume(run_dir).plan()] == STEPS

    def test_only_a_step_the_run_does_not_have_is_refused(self, run_dir: Path) -> None:
        with pytest.raises(ConfigError, match="retrieve"):
            Pipeline.resume(run_dir, only=["retrieve"])


class TestTheBradleyTerryPenalty:
    def test_the_calibrate_step_refits_with_the_configured_l2(self, data: Path, tmp_path: Path) -> None:
        calibration = {"priors": {"bt_l2": 0.05}}
        pipeline = Pipeline(tiny_config(data, calibration=calibration, steps=STEPS[:3]), runs_dir=str(tmp_path))

        with pytest.warns(RcpNdcgWarning, match="L2 0.05"):
            pipeline.run()

        from rcp_ndcg.calibration import Calibration

        calibration = Calibration.load(Path(pipeline.layout.root) / "calibration")
        assert calibration.identity.priors.bt_l2 == 0.05
        assert [w["code"] for w in calibration.warnings] == ["BT_L2_MISMATCH"]


class TestEstimateAndRetrieve:
    def test_an_estimate_calls_no_judge_and_creates_no_run(self, data: Path, tmp_path: Path) -> None:
        estimate = Pipeline(tiny_config(data), runs_dir=str(tmp_path / "runs")).estimate()
        assert set(estimate.stages) == {"tournament", "rubric"}
        assert estimate.calls > 0 and estimate.input_tokens > 0
        endpoint = {"base_url": "http://127.0.0.1:9/v1", "model": "my-judge"}  # nothing listens: never called
        assert Pipeline(tiny_config(data, judge=endpoint), runs_dir=str(tmp_path / "runs")).estimate().calls > 0
        assert not (tmp_path / "runs").exists()

    def test_a_retrieval_sourced_run_is_estimated_at_its_depth_before_it_retrieves(
        self, data: Path, tmp_path: Path
    ) -> None:
        candidates = {"from": "retrieval", "retrieval": {"kind": "bm25"}, "depth": 3}
        config = tiny_config(data, candidates=candidates)
        runs = str(tmp_path / "runs")

        before = Pipeline(config, runs_dir=runs).estimate()
        assert not (tmp_path / "runs").exists()
        assert "assumed to be 3 documents" in before.assumptions[-1]

        retrieved = Pipeline(config.model_copy(update={"steps": ["retrieve"]}), runs_dir=runs)
        retrieved.run()
        after = Pipeline(config, layout=retrieved.layout).estimate()
        assert after.calls > 0
        assert {s: e.calls for s, e in before.stages.items()} == {s: e.calls for s, e in after.stages.items()}
        assert not any("retrieval has not run" in note for note in after.assumptions)

    def test_supplied_rankings_become_the_candidates(self, data: Path, tmp_path: Path) -> None:
        rankings = tmp_path / "rankings.jsonl"
        rankings.write_text(json.dumps({"query_id": "q1", "query": "q", "doc_ids": ["a", "b"]}) + "\n")
        config = tiny_config(data, candidates={"from": "rankings", "rankings": str(rankings)}, steps=["retrieve"])
        pipeline = Pipeline(config, runs_dir=str(tmp_path / "runs"))
        manifest = pipeline.run()
        written = load_rankings(pipeline.layout.candidates)
        assert written.systems == ["candidates"]
        assert sorted(written.for_query("q1"), key=written.for_query("q1").get, reverse=True) == ["a", "b"]
        assert [ref.path for ref in manifest.step("retrieve").inputs] == [str(rankings)]
        rankings.write_text(json.dumps({"query_id": "q1", "query": "q", "doc_ids": ["b", "a"]}) + "\n")
        assert Pipeline.resume(pipeline.layout.root).plan() == [{"step": "retrieve", "status": "would run"}]

    def test_a_rankings_sourced_run_is_estimated_from_its_rankings_file_before_it_retrieves(
        self, data: Path, tmp_path: Path
    ) -> None:
        from rcp_ndcg.data import Rankings

        rankings = tmp_path / "systems.jsonl"
        rows, _ = tiny_rows()
        Rankings.from_orders({row.id: row.doc_ids[:4] for row in rows}, system="bm25").save(rankings)
        candidates = {"from": "rankings", "rankings": str(rankings), "system": "bm25"}
        config = tiny_config(data, candidates=candidates, steps=["retrieve", "tournament", "rubric"])

        before = Pipeline(config, runs_dir=str(tmp_path / "runs")).estimate()
        assert before.calls > 0 and not (tmp_path / "runs").exists()
        retrieved = Pipeline(config.model_copy(update={"steps": ["retrieve"]}), runs_dir=str(tmp_path / "runs"))
        retrieved.run()
        after = Pipeline(config, layout=retrieved.layout).estimate()
        assert {s: e.calls for s, e in before.stages.items()} == {s: e.calls for s, e in after.stages.items()}

    def test_retrieval_writes_the_candidates(self, data: Path, tmp_path: Path) -> None:
        config = tiny_config(
            data,
            candidates={"from": "retrieval", "retrieval": {"kind": "bm25"}, "depth": 3},
            steps=["retrieve"],
        )
        pipeline = Pipeline(config, runs_dir=str(tmp_path / "runs"))
        pipeline.run()
        written = load_rankings(pipeline.layout.candidates)
        assert written.systems == ["candidates"]
        assert {query: len(docs) for query, docs in written.queries().items()} == {"q1": 3, "q2": 3}

    def test_a_rerank_step_reorders_the_pools_the_judge_reads(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def reverse(examples, cfg, **kwargs):
            # The reranker scores in the pool's order; here it prefers the pool's last documents.
            return [e.model_copy(update={"scores": [float(i) for i in range(len(e.doc_ids))]}) for e in examples]

        monkeypatch.setattr("rcp_ndcg.retrieval.cross_encoder.rerank_examples", reverse)
        config = tiny_config(
            data,
            candidates={
                "rerank": {"provider": "openai_compatible", "model": "stub", "base_url": "http://stub:8000"},
                "depth": 4,
            },
            steps=["rerank", "tournament"],
            tournament=TINY_TOURNAMENT.model_copy(update={"window": 4}).model_dump(),
        )
        pipeline = Pipeline(config, runs_dir=str(tmp_path / "runs"))
        pipeline.run()
        root = Path(pipeline.layout.root)
        pools = {q: _order(docs) for q, docs in load_rankings(root / "candidates.parquet").queries().items()}
        first = {q: _order(docs) for q, docs in load_rankings(root / "work/first_stage.parquet").queries().items()}
        rows, _ = tiny_rows()
        for row in rows:
            assert first[row.id] == row.doc_ids
            assert pools[row.id] == row.doc_ids[::-1]
        judged = {
            p["doc_id"] for r in map(json.loads, (root / "judgements/tournament.jsonl").open()) for p in r["placements"]
        }
        assert judged == {doc for row in rows for doc in row.doc_ids[::-1][:4]}
        assert Pipeline.resume(root).plan() == [
            {"step": "rerank", "status": "would skip"},
            {"step": "tournament", "status": "would skip"},
        ]


class TestTheEvaluateIdentity:
    def test_a_changed_seed_redoes_the_evaluation_whose_bootstrap_it_draws(self, run_dir: Path) -> None:
        calibrated = RunManifest.load(run_dir).step("calibrate")
        manifest = Pipeline.resume(run_dir, overrides=["seed=7", "steps=[calibrate, evaluate]"]).run()
        assert manifest.step("calibrate") == calibrated, "the calibration does not depend on the seed"
        assert manifest.step("evaluate").ended_at > calibrated.ended_at
        assert manifest.step("evaluate").identity["seed"] == 7


class TestTheRunsSeed:
    def test_it_defaults_to_the_schedules_seed(self) -> None:
        from rcp_ndcg.llm import RubricSchedule, TournamentSchedule
        from rcp_ndcg.runs import RunConfig

        assert (
            RunConfig(dataset="jsonl:x", steps=["evaluate"]).seed == TournamentSchedule().seed == RubricSchedule().seed
        )

    def test_it_is_the_seed_of_a_schedule_that_sets_none(self, data: Path, tmp_path: Path) -> None:
        unseeded = {key: value for key, value in TINY_TOURNAMENT.model_dump().items() if key != "seed"}
        pipeline = Pipeline(
            tiny_config(data, seed=7, tournament=unseeded, steps=["tournament"]), runs_dir=str(tmp_path)
        )
        pipeline.run()
        identity = json.loads((Path(pipeline.layout.judgements) / "identity.json").read_text(encoding="utf-8"))
        assert identity["stages"]["tournament"]["identity"]["schedule"]["seed"] == 7
        assert Pipeline.resume(pipeline.layout.root).plan() == [{"step": "tournament", "status": "would skip"}]
        # A schedule's own seed wins; without a schedule, the paper's is seeded with the run's seed.
        assert tiny_config(data, seed=7, tournament={**unseeded, "seed": 3}).tournament.seed == 3
        paper = Pipeline(tiny_config(data, seed=7, tournament=None)).schedule("tournament")
        assert paper.seed == 7 and paper.window == TournamentSchedule().window


class TestEvaluationSystems:
    def test_a_multi_system_file_and_a_picked_system_are_scored(self, run_dir: Path, tmp_path: Path) -> None:
        """Every system of a file is scored under its own name; `<file>#<system>` picks one under the key."""
        from rcp_ndcg.data import Rankings

        rows, _ = tiny_rows()
        orders = {row.id: row.doc_ids for row in rows}
        both = tmp_path / "systems.jsonl"
        Rankings.concat(
            [
                Rankings.from_orders(orders, system="forward"),
                Rankings.from_orders({q: d[::-1] for q, d in orders.items()}, system="reverse"),
            ]
        ).save(both)
        overrides = [f"evaluation.systems={{mine: {both}, picked: '{both}#reverse'}}"]

        manifest = Pipeline.resume(run_dir, overrides=overrides).run()

        systems = {key.split("/")[0] for key in manifest.metrics}
        assert systems == {"candidates", "forward", "reverse", "picked"}
        assert manifest.metrics["picked/rcp_ndcg@10"] == manifest.metrics["reverse/rcp_ndcg@10"]
        with pytest.raises(DataError, match="holds no system 'other'"):
            Pipeline.resume(run_dir, overrides=[f"evaluation.systems={{x: '{both}#other'}}"]).run()

    def test_a_file_whose_rows_name_the_runs_dataset_is_scored(self, run_dir: Path, tmp_path: Path) -> None:
        from rcp_ndcg.data import Rankings

        rows, _ = tiny_rows()
        orders = {row.id: row.doc_ids for row in rows}
        name = Pipeline.resume(run_dir).dataset.name
        plain, tagged = tmp_path / "plain.parquet", tmp_path / "tagged.parquet"
        Rankings.from_orders(orders, system="bm25").save(plain)
        Rankings.from_orders(orders, system="bm25", dataset=name).save(tagged)

        manifest = Pipeline.resume(run_dir, overrides=[f"evaluation.systems={{a: {plain}, b: {tagged}}}"]).run()

        assert manifest.metrics["b/rcp_ndcg@10"] == manifest.metrics["a/rcp_ndcg@10"]


class TestFailures:
    def test_a_jobs_runtime_overrides_are_no_config_change_so_its_failure_is_recorded(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job that starts its engine passes the replica URLs and the outage wait; its failure rolled the run back
        to 'submitted' with no error, as if a --set change had failed."""
        import yaml
        from click.testing import CliRunner

        from rcp_ndcg.cli.main import cli

        config = tiny_config(
            data,
            judge={"base_url": "http://engine:8000/v1", "model": "m"},
            candidates={"from": "rankings", "rankings": str(tmp_path / "missing.parquet")},
            steps=["retrieve", *STEPS],
        )
        pipeline = Pipeline(config, runs_dir=str(tmp_path / "runs"))
        pipeline.layout.ensure()
        pipeline._write_config()
        pipeline.manifest.status = RunStatus.SUBMITTED
        pipeline.manifest.save(pipeline.layout)
        monkeypatch.setenv("RCP_NDCG_JUDGE_URLS", "http://node1:8000/v1,http://node2:8000/v1")
        args = ["run", "resume", "--run", pipeline.layout.root, "--set", "judge.wait_on_outage_s=900", "--json"]
        assert CliRunner().invoke(cli, args).exit_code == 4

        manifest = RunManifest.load(pipeline.layout)
        assert manifest.status is RunStatus.FAILED
        assert (
            manifest.step("retrieve").status is StepStatus.FAILED
            and "missing.parquet" in manifest.step("retrieve").error
        )
        assert yaml.safe_load(Path(pipeline.layout.config).read_text())["judge"]["base_url"] == [
            "http://node1:8000/v1",
            "http://node2:8000/v1",
        ]

    def test_a_failed_change_leaves_a_run_that_never_ran_failed_not_submitted(self, data: Path, tmp_path: Path) -> None:
        pipeline = Pipeline(tiny_config(data, steps=["tournament"]), runs_dir=str(tmp_path / "runs"))
        pipeline.layout.ensure()
        pipeline._write_config()
        pipeline.manifest.status = RunStatus.SUBMITTED
        pipeline.manifest.save(pipeline.layout)
        changed = Pipeline.resume(
            pipeline.layout.root, overrides=["candidates.from=rankings", "candidates.rankings=nope"]
        )
        with pytest.raises(MissingInputError):  # the rankings were never retrieved
            changed.run()
        manifest = RunManifest.load(pipeline.layout)
        assert manifest.status is RunStatus.FAILED and manifest.config == pipeline.manifest.config
        assert manifest.step("tournament") is None or manifest.step("tournament").error

    def test_a_judging_step_that_fails_keeps_its_requests(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The requests of a failed judging step were lost from the step and the run's usage."""
        from rcp_ndcg.llm.client import BackendUnavailableError
        from rcp_ndcg.testing import FakeJudge

        answer = FakeJudge._send
        answered = []

        async def down_after_three(self, request, replica=None):
            if len(answered) >= 3:
                raise BackendUnavailableError("the endpoint went away")
            answered.append(request)
            return await answer(self, request, replica)

        monkeypatch.setattr(FakeJudge, "_send", down_after_three)
        pipeline = Pipeline(tiny_config(data, steps=["tournament"]), runs_dir=str(tmp_path / "runs"))
        with pytest.raises(BackendUnavailableError):
            pipeline.run()
        manifest = RunManifest.load(pipeline.layout)
        assert manifest.step("tournament").usage.requests >= 3 and manifest.usage.requests >= 3

    @pytest.mark.parametrize("cancelled", [False, True])
    def test_an_interrupted_run_is_recorded_as_one_that_mirrors(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool
    ) -> None:
        """SIGTERM without a mirror (KeyboardInterrupt) left the run 'running' forever; with one it was 'failed'.
        A job that `run cancel` stopped keeps the 'cancelled' the cancel recorded."""
        pipeline = Pipeline(tiny_config(data, steps=["tournament"]), runs_dir=str(tmp_path / "runs"))

        def interrupted(self):
            if cancelled:
                on_disk = RunManifest.load(self.layout)
                on_disk.status = RunStatus.CANCELLED
                on_disk.save(self.layout)
            raise KeyboardInterrupt

        monkeypatch.setattr(Pipeline, "_step_tournament", interrupted)
        with pytest.raises(KeyboardInterrupt):
            pipeline.run()
        manifest = RunManifest.load(pipeline.layout)
        record = manifest.step("tournament")
        if cancelled:
            assert (manifest.status, record.status) == (RunStatus.CANCELLED, StepStatus.CANCELLED)
        else:
            assert (manifest.status, record.status) == (RunStatus.FAILED, StepStatus.FAILED)
        assert record.error.startswith("Interrupted")
