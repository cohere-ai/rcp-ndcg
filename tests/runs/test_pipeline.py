"""The pipeline, offline: every step on a tiny dataset judged by the fake judge, then resumed."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from rcp_ndcg.data import load_rankings
from rcp_ndcg.errors import ConfigError, DataError, IdentityError, MissingInputError, RcpNdcgWarning
from rcp_ndcg.judging import TournamentSchedule
from rcp_ndcg.runs import Pipeline, RunManifest, RunStatus, StepStatus
from rcp_ndcg.support.identity import check_declarations, identity_payload
from rcp_ndcg.testing import TINY_RUBRIC, TINY_TOURNAMENT, tiny_rows
from tests._tokenizers import byte_bpe_tokenizer
from tests.conftest import SESSION_TOKENIZER
from tests.runs.conftest import STEPS, tiny_config

_SERVED_BUDGET: dict[str, Any] = {"tokenizer": str(SESSION_TOKENIZER), "max_tokens": 8192}
"""The explicit budget every served config declares (a self-hosted role config names its tokenizer and cap)."""

_SERVED_RERANK_BUDGET: dict[str, Any] = {**_SERVED_BUDGET, "use_activation": False}
"""The served rerankers' budget plus the explicit ``use_activation`` (F10: a served rerank config sets it)."""


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
    from rcp_ndcg.judging import JudgeClient
    from tests.judging.test_judging import _SchemaEndpoint

    endpoint = _SchemaEndpoint("3.1")
    monkeypatch.setattr(JudgeClient, "from_config", staticmethod(lambda config: endpoint.client()))
    judge = {"base_url": "http://judge.test/v1", "model": "m"}
    pipeline = Pipeline(tiny_config(data, judge=judge, steps=["tournament"]), runs_dir=str(tmp_path))
    manifest = pipeline.run()
    (engine,) = manifest.step("tournament").engines
    assert (engine.model, engine.max_model_len, engine.system_fingerprint) == ("m", 32768, "engine-3.1")
    # Engine facts live in the step's ``engines``, never in its content identity. Checked over the
    # identity's KEYS: a serialized identity embeds paths (a workspace directory can spell "engines").

    def _keys(value: object):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from _keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from _keys(item)

    identity_keys = list(_keys(manifest.step("tournament").identity))
    assert not [key for key in identity_keys if "fingerprint" in key or "engine" in key], identity_keys


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


class TestStepBudget:
    """``step_budget_s``: a per-step wall-clock budget stops the step typed, and the store resumes."""

    def test_a_step_over_its_budget_stops_typed_and_the_store_resumes(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        from rcp_ndcg_core.irt._bradley_terry import BradleyTerryEstimator  # noqa: F401 - warm the lazy import

        from rcp_ndcg.errors import StepBudgetExceededError
        from rcp_ndcg.judging.client import JudgeClient

        complete = JudgeClient.complete

        async def slow(self, request):
            reply = await complete(self, request)  # the request lands and is stored...
            await asyncio.sleep(2.0)  # ...and the phase then outlives the budget
            return reply

        monkeypatch.setattr(JudgeClient, "complete", slow)
        pipeline = Pipeline(tiny_config(data, steps=["tournament"], step_budget_s=1.0), runs_dir=str(tmp_path))
        with pytest.raises(StepBudgetExceededError) as stopped:
            pipeline.run()
        assert "step_budget_s" in (stopped.value.hint or "")
        assert pipeline.manifest.status is RunStatus.FAILED
        assert pipeline.manifest.step("tournament").status is StepStatus.FAILED
        assert pipeline.manifest.step("tournament").error.startswith("StepBudgetExceededError")

        monkeypatch.setattr(JudgeClient, "complete", complete)
        store = Path(pipeline.layout.root) / "judgements" / "tournament.jsonl"
        judged = _lines(store) if store.exists() else 0
        manifest = Pipeline.resume(pipeline.layout.root, overrides=["step_budget_s=null"]).run()
        assert manifest.status is RunStatus.COMPLETED
        assert judged > 0 and _lines(store) > judged  # the resumed pass carried on from what it wrote

    def test_the_budget_is_runtime_only(self, run_dir: Path) -> None:
        # Changing the budget alone is not a config change: the completed steps stay current (no re-keying).
        pipeline = Pipeline.resume(run_dir, overrides=["step_budget_s=60"])
        assert pipeline.plan() == [{"step": step, "status": "would skip"} for step in STEPS]
        assert pipeline.run().status is RunStatus.COMPLETED

    def test_the_judging_seam_stops_a_pass_that_never_reaches_the_transport(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The judging pass checks the budget before a phase's windows: a client that answers in process (the
        fake judge's transport seam is never reached) still stops typed."""
        from rcp_ndcg.errors import StepBudgetExceededError
        from rcp_ndcg.judging.client import JudgeClient, RequestRejectedError

        async def rejecting(self, request):
            raise RequestRejectedError("refused without a transport")

        monkeypatch.setattr(JudgeClient, "complete", rejecting)
        pipeline = Pipeline(tiny_config(data, steps=["tournament"], step_budget_s=1e-9), runs_dir=str(tmp_path))
        with pytest.raises(StepBudgetExceededError):
            pipeline.run()
        assert pipeline.manifest.step("tournament").status is StepStatus.FAILED


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

        def broken(self, request):
            raise RuntimeError("endpoint gone")

        monkeypatch.setattr("rcp_ndcg.judging.client.JudgeClient.complete", broken)
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
        def score_by_position(self, examples, *, checkpoint=None):
            # The stub client scores each document by its pool position: it prefers the pool's last documents.
            for example in examples:
                scores = tuple(float(i) for i in range(len(example.doc_ids)))
                if checkpoint is not None:
                    checkpoint(str(example.id), scores)
            return []

        monkeypatch.setattr("rcp_ndcg.retrieval._api.RerankClient.rerank_many", score_by_position)
        config = tiny_config(
            data,
            candidates={
                "rerank": {
                    "api": "rerank",
                    "model": "stub",
                    "base_url": "http://stub:8000",
                    **_SERVED_RERANK_BUDGET,
                },
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


class TestTheRetrieveAndRerankIdentities:
    """RFC-0001 5.4: the retrieve and rerank step identities hold the content payload of the candidates config
    (``identity_payload``, like the judge steps), never its runtime fields: a served encoder's or reranker's URL,
    its key variable, its concurrency, timeouts and retries move the work, not the numbers.
    """

    @staticmethod
    def _dense(**encoder: Any) -> dict[str, Any]:
        served = {"model": "embedder", "base_url": "http://engine.test/v1", **_SERVED_BUDGET, **encoder}
        return {
            "from": "retrieval",
            "retrieval": {"kind": "dense", "encoder": {"api": "openai_embeddings", **served, **_SERVED_BUDGET}},
        }

    @staticmethod
    def _cohere(**encoder: Any) -> dict[str, Any]:
        hosted = {"model": "embed-v4.0", **encoder}
        return {"from": "retrieval", "retrieval": {"kind": "dense", "encoder": {"api": "cohere", **hosted}}}

    @staticmethod
    def _rerank(**reranker: Any) -> dict[str, Any]:
        served = {"model": "reranker", "base_url": "http://engine.test:8000", **_SERVED_RERANK_BUDGET, **reranker}
        return {"rerank": {"api": "rerank", **served, **_SERVED_RERANK_BUDGET}}

    def _two(self, data: Path, tmp_path: Path, candidates_a: dict, candidates_b: dict, step: str):
        """Two pipelines on one run directory whose candidates sections differ as given."""
        first = Pipeline(tiny_config(data, candidates=candidates_a, steps=[step]), runs_dir=str(tmp_path / "runs"))
        second = Pipeline(tiny_config(data, candidates=candidates_b, steps=[step]), layout=first.layout)
        return first._identity(step), second._identity(step)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("base_url", "http://elsewhere.test:9000/v1"),
            ("api_key_env", "OTHER_KEY"),
            ("timeout_s", 5.0),
            ("max_retries", 9),
            ("connect_timeout_s", 1.0),
        ],
    )
    def test_a_served_encoders_runtime_fields_do_not_rekey_the_retrieve_step(
        self, data: Path, tmp_path: Path, field: str, value: Any
    ) -> None:
        one, two = self._two(data, tmp_path, self._dense(), self._dense(**{field: value}), "retrieve")
        assert one == two

    @pytest.mark.parametrize(
        "field, value",
        [
            ("model", "other-embedder"),
            ("revision", "20260101"),
            ("query_prompt", "Query: "),
            ("doc_prompt", "Passage: "),
        ],
    )
    def test_a_served_encoders_content_fields_rekey_the_retrieve_step(
        self, data: Path, tmp_path: Path, field: str, value: Any
    ) -> None:
        one, two = self._two(data, tmp_path, self._dense(), self._dense(**{field: value}), "retrieve")
        assert one != two

    @pytest.mark.parametrize(
        "field, value",
        [
            ("base_url", "http://proxy.test/v1"),
            ("api_key_env", "OTHER_KEY"),
            ("timeout_s", 5.0),
            ("max_retries", 9),
        ],
    )
    def test_a_hosted_encoders_runtime_fields_do_not_rekey_the_retrieve_step(
        self, data: Path, tmp_path: Path, field: str, value: Any
    ) -> None:
        one, two = self._two(data, tmp_path, self._cohere(), self._cohere(**{field: value}), "retrieve")
        assert one == two

    @pytest.mark.parametrize(
        "field, value",
        [("model", "embed-v4.0-preview"), ("revision", "20260101"), ("batch_size", 96)],
    )
    def test_a_hosted_encoders_content_fields_rekey_the_retrieve_step(
        self, data: Path, tmp_path: Path, field: str, value: Any
    ) -> None:
        one, two = self._two(data, tmp_path, self._cohere(), self._cohere(**{field: value}), "retrieve")
        assert one != two

    @pytest.mark.parametrize(
        "field, value",
        [
            ("base_url", "http://elsewhere.test:9000"),
            ("concurrency", 2),
            ("api_key_env", "OTHER_KEY"),
            ("timeout_s", 5.0),
            ("max_retries", 9),
        ],
    )
    def test_a_served_rerankers_runtime_fields_do_not_rekey_the_rerank_step(
        self, data: Path, tmp_path: Path, field: str, value: Any
    ) -> None:
        one, two = self._two(data, tmp_path, self._rerank(), self._rerank(**{field: value}), "rerank")
        assert one == two

    @pytest.mark.parametrize("field, value", [("model", "other-reranker"), ("revision", "20260101")])
    def test_a_served_rerankers_content_fields_rekey_the_rerank_step(
        self, data: Path, tmp_path: Path, field: str, value: Any
    ) -> None:
        one, two = self._two(data, tmp_path, self._rerank(), self._rerank(**{field: value}), "rerank")
        assert one != two

    @pytest.mark.parametrize(
        "reranker",
        [
            {"api": "cohere", "model": "rerank-v4.0-pro"},
            {"api": "rerank", "model": "a-served-reranker", "base_url": "http://h:8000/v1", **_SERVED_RERANK_BUDGET},
        ],
        ids=["cohere", "served"],
    )
    def test_a_rerankers_batch_size_rekeys_the_rerank_step(self, data: Path, tmp_path: Path, reranker: dict) -> None:
        """Request packing is content: a bf16 batch's composition can move the scores, so a cached rerank
        result is never reused across two batch sizes."""
        one, two = self._two(data, tmp_path, {"rerank": reranker}, {"rerank": {**reranker, "batch_size": 32}}, "rerank")
        assert one != two

    @pytest.mark.parametrize(
        "reranker",
        [
            {"api": "cohere", "model": "rerank-v4.0-pro"},
            {"api": "rerank", "model": "a-served-reranker", "base_url": "http://h:8000/v1", **_SERVED_RERANK_BUDGET},
        ],
        ids=["cohere", "served"],
    )
    @pytest.mark.parametrize("field, value", [("model", "another"), ("revision", "20260101")])
    def test_a_rerankers_content_fields_rekey_the_rerank_step(
        self, data: Path, tmp_path: Path, reranker: dict, field: str, value: Any
    ) -> None:
        one, two = self._two(data, tmp_path, {"rerank": reranker}, {"rerank": {**reranker, field: value}}, "rerank")
        assert one != two

    def test_the_tokenizer_sha_splices_into_the_step_identities(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The endpoint's tokenizer digest (``identity_extra()``) is spliced in at the encoder and the reranker:
        two tokenizers with the same bytes (different names) share an identity, different bytes do not, and a
        URL change still never re-keys."""
        from tests._tokenizers import save, word_tokenizer

        def config_with(reranker: dict[str, Any]) -> dict[str, Any]:
            return {"rerank": reranker}

        for directory in ("one", "two", "other"):
            (tmp_path / directory).mkdir()
        first = save(word_tokenizer(), tmp_path / "one")
        second = save(word_tokenizer(), tmp_path / "two")  # same bytes, different path
        other = save(byte_bpe_tokenizer(), tmp_path / "other")

        def identity(reranker: dict[str, Any]) -> dict[str, Any]:
            return Pipeline(
                tiny_config(data, candidates={"rerank": reranker}, steps=["rerank"]), runs_dir=str(tmp_path / "runs")
            )._identity("rerank")

        named = {
            "api": "rerank",
            "model": "rr",
            "base_url": "http://h:8000/v1",
            "tokenizer": str(first),
            **_SERVED_RERANK_BUDGET,
        }
        same_sha = {**named, "tokenizer": str(second)}  # same bytes, different path
        moved = {**named, "base_url": "http://moved:8000/v1"}
        other_sha = {**named, "tokenizer": str(other)}

        base = identity({"api": "rerank", "model": "rr", "base_url": "http://h:8000/v1", **_SERVED_RERANK_BUDGET})
        with_digest = identity(named)

        assert "tokenizer" not in base, "the name is runtime"
        assert "tokenizer_sha256" in with_digest["rerank"], "the digest is spliced in"
        assert identity(same_sha) == with_digest, "the same tokenizer bytes (any path) share the identity"
        assert identity(moved) == with_digest, "a moved URL does not re-key"
        assert identity(other_sha) != with_digest, "different tokenizer bytes re-key"

    def test_the_prompt_content_and_the_judge_tokenizer_splice_into_the_judge_step_identity(
        self, data: Path, tmp_path: Path
    ) -> None:
        """A judge step is keyed by what its judgements answer for: the resolved prompt's content (its name or
        path is runtime -- the same text under another name is the same instrument, edited text is not) and the
        judge tokenizer's digest. Editing a prompt file or swapping the tokenizer bytes re-keys the step, so a
        resume re-judges instead of skipping with stale judgements."""
        from rcp_ndcg.judging.prompts import load_prompt, shipped_prompts_digest
        from tests._tokenizers import byte_bpe_tokenizer, save, word_tokenizer

        for directory in ("one", "two", "prompts"):
            (tmp_path / directory).mkdir()
        first = save(word_tokenizer(), tmp_path / "one")
        other = save(byte_bpe_tokenizer(), tmp_path / "two")
        named, moved, edited = (tmp_path / "prompts" / name for name in ("a.txt", "b.txt", "c.txt"))
        named.write_text(load_prompt("rubric").text, encoding="utf-8")
        moved.write_text(load_prompt("rubric").text, encoding="utf-8")  # the same text under another name
        edited.write_text(load_prompt("rubric").text + "\nA criterion notes line.", encoding="utf-8")

        def identity(prompt: Path | None, tokenizer: Path | None = None) -> dict[str, Any]:
            judge: dict[str, Any] = {"base_url": "http://judge.test/v1", "model": "m"}
            if tokenizer is not None:
                judge["tokenizer"] = str(tokenizer)
            fields = {"judge": judge, "steps": ["rubric"]}
            if prompt is not None:
                fields["rubric"] = TINY_RUBRIC.model_copy(update={"prompt": str(prompt)}).model_dump()
            return Pipeline(tiny_config(data, **fields), runs_dir=str(tmp_path / "runs"))._identity("rubric")

        base = identity(None)
        assert base["prompt_sha256"] == shipped_prompts_digest("rubric")  # the shipped set, by content
        custom = identity(named)
        assert custom["prompt_sha256"] == load_prompt("rubric").sha256  # the named prompt, by content
        assert identity(moved) == custom, "the same text under another name is the same instrument"
        assert identity(edited) != custom, "edited prompt content re-keys the step"
        with_tokenizer = identity(None, first)
        assert "tokenizer" not in with_tokenizer["judge"], "the name is runtime"
        assert with_tokenizer["judge"]["tokenizer_sha256"]
        assert identity(None, other) != with_tokenizer, "different tokenizer bytes re-key"

    def test_the_encoders_pooling_rekeys_the_retrieve_step(self, data: Path, tmp_path: Path) -> None:
        """``pooling: token`` is the late-interaction route (``/pooling``), not the one-vector one."""
        dense = {
            "from": "retrieval",
            "retrieval": {
                "kind": "dense",
                "encoder": {
                    "api": "openai_embeddings",
                    "model": "m",
                    "base_url": "http://engine.test/v1",
                    **_SERVED_BUDGET,
                },
            },
        }
        token = {
            "from": "retrieval",
            "retrieval": {
                "kind": "late_interaction",
                "encoder": {"api": "vllm_pooling", "model": "m", "base_url": "http://engine.test/v1", **_SERVED_BUDGET},
            },
        }
        one, two = self._two(data, tmp_path, dense, token, "retrieve")
        assert one != two

    def test_the_candidates_payload_keys_by_field_name_not_alias(self, data: Path, tmp_path: Path) -> None:
        """``identity_payload`` keys by field name: ``from:`` in YAML is ``source`` in the identity."""
        from rcp_ndcg.retrieval import BM25Config
        from rcp_ndcg.runs.config import CandidatesConfig

        pipeline = Pipeline(
            tiny_config(data, candidates={"from": "retrieval", "retrieval": {"kind": "bm25"}}, steps=["retrieve"]),
            runs_dir=str(tmp_path / "runs"),
        )
        candidates = pipeline._identity("retrieve")["candidates"]
        assert "source" in candidates and "from" not in candidates
        aliased = CandidatesConfig.model_validate({"from": "retrieval", "retrieval": {"kind": "bm25"}})
        named = CandidatesConfig(source="retrieval", retrieval=BM25Config())
        assert identity_payload(aliased) == identity_payload(named) == candidates

    @pytest.mark.parametrize(
        "candidates",
        [
            {"from": "retrieval", "retrieval": {"kind": "bm25"}},
            {
                "from": "retrieval",
                "retrieval": {"kind": "dense", "encoder": {"api": "cohere", "model": "embed-v4.0"}},
            },
            {
                "from": "retrieval",
                "retrieval": {"kind": "dense", "encoder": {"api": "voyage", "model": "voyage-3-large"}},
            },
            {
                "from": "retrieval",
                "retrieval": {"kind": "dense", "encoder": {"api": "gemini", "model": "gemini-embedding-001"}},
            },
            {
                "from": "retrieval",
                "retrieval": {
                    "kind": "dense",
                    "encoder": {
                        "api": "openai_embeddings",
                        "model": "m",
                        "base_url": "http://engine.test/v1",
                        **_SERVED_BUDGET,
                    },
                },
            },
            {
                "from": "retrieval",
                "retrieval": {
                    "kind": "late_interaction",
                    "encoder": {
                        "api": "vllm_pooling",
                        "model": "m",
                        "base_url": "http://engine.test/v1",
                        **_SERVED_BUDGET,
                    },
                },
            },
            {"from": "rankings", "rankings": "rankings.jsonl", "system": "bm25"},
            {"rerank": {"api": "rerank", "model": "m", "base_url": "http://engine.test:8000", **_SERVED_RERANK_BUDGET}},
            {"rerank": {"api": "cohere", "model": "rerank-v4.0-pro"}},
            {"rerank": {"api": "voyage", "model": "rerank-2.5"}},
        ],
    )
    def test_every_model_the_candidates_payload_reaches_declares_its_roles(self, candidates: dict) -> None:
        from rcp_ndcg.runs.config import CandidatesConfig

        check_declarations(CandidatesConfig)
        identity_payload(CandidatesConfig.model_validate(candidates))

    @pytest.mark.parametrize(
        "klass",
        [
            "BM25Config",
            "DenseConfig",
            "LateInteractionConfig",
            "ServedEmbedding",
            "ServedPooling",
            "ServedReranker",
            "CohereEmbedding",
            "VoyageEmbedding",
            "GeminiEmbedding",
            "CohereReranker",
            "VoyageReranker",
        ],
    )
    def test_every_retrieval_config_class_declares_its_roles(self, klass: str) -> None:
        """Every class of the retrieval configs -- the abstract bases included -- declares exactly its own fields.

        A role for a field the class does not define (say ``provider``, declared on a base whose subclasses own
        the field) is stale on that class: the declaration lives next to the field, and the MRO merge hands it to
        the leaves.
        """
        import rcp_ndcg.retrieval.config as retrieval_config

        check_declarations(getattr(retrieval_config, klass))

    def test_a_changed_reranker_url_skips_a_completed_rerank_step(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def score_by_position(self, examples, *, checkpoint=None):
            for example in examples:
                scores = tuple(float(i) for i in range(len(example.doc_ids)))
                if checkpoint is not None:
                    checkpoint(str(example.id), scores)
            return []

        monkeypatch.setattr("rcp_ndcg.retrieval._api.RerankClient.rerank_many", score_by_position)
        candidates = {
            "rerank": {"api": "rerank", "model": "stub", "base_url": "http://stub.test:8000", **_SERVED_RERANK_BUDGET},
            "depth": 4,
        }
        pipeline = Pipeline(tiny_config(data, candidates=candidates, steps=["rerank"]), runs_dir=str(tmp_path / "runs"))
        pipeline.run()
        before = RunManifest.load(pipeline.layout.root).step("rerank")

        manifest = Pipeline.resume(
            pipeline.layout.root, overrides=["candidates.rerank.base_url=http://moved.test:8000"]
        ).run()

        assert manifest.step("rerank") == before, "the moved URL is runtime: the completed step is skipped"
        assert manifest.status is RunStatus.COMPLETED


class TestTheEnginesOverlay:
    """``RCP_NDCG_ENGINES`` is the runtime channel between a phase's engines and the steps that call them: the
    coordinator applies each role's URLs and outage wait to the role config in memory, never to ``run.yaml`` and
    never to an identity, so a run is byte-identical with and without the variable."""

    @staticmethod
    def _dense(**encoder: Any) -> dict[str, Any]:
        served = {"model": "embedder", "base_url": "http://engine.test/v1", **_SERVED_BUDGET, **encoder}
        return {
            "from": "retrieval",
            "retrieval": {"kind": "dense", "encoder": {"api": "openai_embeddings", **served, **_SERVED_BUDGET}},
        }

    def test_the_encoder_overlay_reaches_the_step_and_never_the_config(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tiny_config(data, candidates=self._dense(), steps=["retrieve"])
        plain = Pipeline(config, runs_dir=str(tmp_path / "runs"))
        assert plain._overlaid_retrieval().encoder.base_url == "http://engine.test/v1"
        monkeypatch.setenv(
            "RCP_NDCG_ENGINES", json.dumps({"encoder": {"urls": ["http://node1:8000/v1"], "wait_on_outage_s": 900}})
        )
        overlaid = Pipeline(config, runs_dir=str(tmp_path / "runs"))
        encoder = overlaid._overlaid_retrieval().encoder
        assert (encoder.base_url, encoder.wait_on_outage_s) == ("http://node1:8000/v1", 900.0)
        # The recorded config is untouched: run.yaml and every identity are what they would be without the var.
        assert plain.config.resolved() == overlaid.config.resolved()
        for step in plain.config.ordered_steps:
            assert plain._identity(step) == overlaid._identity(step)

    def test_the_reranker_overlay_reaches_the_client_and_never_the_config_or_an_identity(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rerank step builds its client from the overlaid config: the engine's URL is in the client's
        config, the recorded config keeps the placeholder-free shape, and the identity never moves."""
        candidates = {"rerank": {"api": "rerank", "model": "stub-reranker", **_SERVED_RERANK_BUDGET}, "depth": 4}
        plain = Pipeline(tiny_config(data, candidates=candidates, steps=["rerank"]), runs_dir=str(tmp_path / "runs"))
        assert plain._overlaid_reranker().base_url is None
        identity_before = plain._identity("rerank")

        seen: dict[str, Any] = {}

        def score_by_position(self, examples, *, checkpoint=None):
            seen["base_url"] = self.config.base_url
            seen["wait_on_outage_s"] = self.config.wait_on_outage_s
            for example in examples:
                checkpoint(str(example.id), tuple(float(i) for i in range(len(example.doc_ids))))
            return []

        monkeypatch.setattr("rcp_ndcg.retrieval._api.RerankClient.rerank_many", score_by_position)
        monkeypatch.setenv(
            "RCP_NDCG_ENGINES", json.dumps({"reranker": {"urls": ["http://node:8000/v1"], "wait_on_outage_s": 60}})
        )
        overlaid = Pipeline(tiny_config(data, candidates=candidates, steps=["rerank"]), runs_dir=str(tmp_path / "runs"))
        assert overlaid._overlaid_reranker().base_url == "http://node:8000/v1"

        overlaid.run()

        assert seen == {"base_url": "http://node:8000/v1", "wait_on_outage_s": 60.0}, "the overlay reached the client"
        assert "base_url" not in overlaid.config.resolved()["candidates"]["rerank"], "the config is untouched"
        assert overlaid._identity("rerank") == identity_before, "the identity is untouched"

    def test_the_judge_overlay_reaches_the_client_and_never_the_config(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config = tiny_config(data, judge={"base_url": "http://judge.test/v1", "model": "m"})
        monkeypatch.setenv(
            "RCP_NDCG_ENGINES",
            json.dumps({"judge": {"urls": ["http://n1:8000/v1", "http://n2:8000/v1"], "wait_on_outage_s": 120}}),
        )
        pipeline = Pipeline(config, runs_dir=str(tmp_path / "runs"))
        overlaid = pipeline._judge_client_config()
        assert (overlaid.urls, overlaid.wait_on_outage_s) == (("http://n1:8000/v1", "http://n2:8000/v1"), 120.0)
        assert pipeline.config.judge_config().urls == ("http://judge.test/v1",)
        for step in config.ordered_steps:
            assert Pipeline(config, runs_dir=str(tmp_path / "runs"))._identity(step) == pipeline._identity(step)

    def test_the_overlay_is_validated_like_a_configured_config(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The overlay rebuilds the config through its model, so an overlaid URL is normalised as a configured
        one is, and a value no configured endpoint accepts (a fake:// replica list) is refused with the typed
        error where the overlay is applied."""
        from rcp_ndcg.judging import JudgeConfig
        from rcp_ndcg.support.serve import ENGINES_ENV

        config = tiny_config(data, judge={"base_url": "http://judge.test/v1/", "model": "m"})
        configured = JudgeConfig.model_validate({"base_url": "http://n1:8000/v1", "model": "m"})
        monkeypatch.setenv(ENGINES_ENV, json.dumps({"judge": {"urls": ["http://n1:8000/v1/"]}}))
        pipeline = Pipeline(config, runs_dir=str(tmp_path / "runs"))
        assert pipeline._judge_client_config().base_url == configured.base_url

        monkeypatch.setenv(
            ENGINES_ENV, json.dumps({"judge": {"urls": ["fake://seed/0", "fake://seed/1"], "wait_on_outage_s": 9}})
        )
        overlaid = Pipeline(config, runs_dir=str(tmp_path / "runs"))
        with pytest.raises(ConfigError, match="not a replica list"):
            overlaid._judge_client_config()

    def test_run_yaml_and_identities_are_byte_identical_with_and_without_the_variable(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A full run under the overlay writes what the same run writes without it."""
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        def embeddings_server() -> ThreadingHTTPServer:
            class Embeddings(BaseHTTPRequestHandler):
                def do_POST(self) -> None:
                    body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    payload = json.dumps({"data": [{"embedding": [1.0, 0.5, 0.25]} for _ in body["input"]]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

                def log_message(self, *args) -> None:
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Embeddings)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            return server

        first, second = embeddings_server(), embeddings_server()
        config = tiny_config(
            data, candidates=self._dense(base_url=f"http://127.0.0.1:{first.server_port}/v1"), steps=["retrieve"]
        )
        one = Pipeline(config, runs_dir=str(tmp_path / "one"))
        one.run()
        monkeypatch.setenv(
            "RCP_NDCG_ENGINES", json.dumps({"encoder": {"urls": [f"http://127.0.0.1:{second.server_port}/v1"]}})
        )
        two = Pipeline(config, runs_dir=str(tmp_path / "two"))
        two.run()

        one_yaml = Path(one.layout.config).read_text(encoding="utf-8")
        two_yaml = Path(two.layout.config).read_text(encoding="utf-8")
        assert one_yaml == two_yaml, "the overlay must not reach run.yaml"
        assert one.manifest.config == two.manifest.config
        for record in one.manifest.steps:
            assert record.identity_hash == two.manifest.step(record.name).identity_hash

    def test_the_overlay_refuses_roles_this_run_cannot_point_at_an_engine(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rcp_ndcg.support.serve import ENGINES_ENV

        def refused(variable: str, match: str) -> None:
            monkeypatch.setenv(ENGINES_ENV, json.dumps(variable))
            with pytest.raises(ConfigError, match=match):
                Pipeline(tiny_config(data), runs_dir=str(tmp_path / "runs"))

        refused({"judge": {"urls": ["http://n1:8000/v1"]}}, "this run's judge is the offline fake")  # judge: fake
        refused({"encoder": {"urls": ["http://n1:8000/v1"]}}, "no served encoder")
        refused({"reranker": {"urls": ["http://n1:8000/v1"]}}, "no served reranker")
        refused({"judge": {"urls": []}}, "at least 1 item after validation")

    def test_a_retrieval_role_takes_one_replica_url(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from rcp_ndcg.support.serve import ENGINES_ENV

        monkeypatch.setenv(
            ENGINES_ENV,
            json.dumps({"encoder": {"urls": ["http://n1:8000/v1", "http://n2:8000/v1"]}}),
        )
        with pytest.raises(ConfigError, match="addresses one URL"):
            Pipeline(tiny_config(data, candidates=self._dense(), steps=["retrieve"]), runs_dir=str(tmp_path / "runs"))

    def test_a_substance_of_the_candidates_ignores_their_runtime_fields(self, data: Path) -> None:
        """An injected engine URL (a --set or an overlay) is no config change: a failed resume keeps it."""
        from rcp_ndcg.runs.pipeline import _substance

        base = self._rerank()
        one = tiny_config(data, candidates=base, steps=["rerank", "tournament", "rubric", "calibrate", "evaluate"])
        two = tiny_config(data, candidates=self._rerank(base_url="http://elsewhere.test:9000"), steps=one.steps)
        assert _substance(one) == _substance(two)
        assert _substance(one) != _substance(tiny_config(data, candidates=self._rerank(model="other"), steps=one.steps))

    @staticmethod
    def _rerank(**reranker: Any) -> dict[str, Any]:
        served = {"model": "reranker", "base_url": "http://engine.test:8000", **_SERVED_RERANK_BUDGET, **reranker}
        return {"rerank": {"api": "rerank", **served, **_SERVED_RERANK_BUDGET}}


class TestTheEvaluateIdentity:
    def test_a_changed_seed_redoes_the_evaluation_whose_bootstrap_it_draws(self, run_dir: Path) -> None:
        calibrated = RunManifest.load(run_dir).step("calibrate")
        manifest = Pipeline.resume(run_dir, overrides=["seed=7", "steps=[calibrate, evaluate]"]).run()
        assert manifest.step("calibrate") == calibrated, "the calibration does not depend on the seed"
        assert manifest.step("evaluate").ended_at > calibrated.ended_at
        assert manifest.step("evaluate").identity["seed"] == 7


class TestTheRunsSeed:
    def test_it_defaults_to_the_schedules_seed(self) -> None:
        from rcp_ndcg.judging import RubricSchedule, TournamentSchedule
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
    def test_a_jobs_engines_overlay_is_no_config_change_so_its_failure_is_recorded(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job that starts engines hands them to its steps through RCP_NDCG_ENGINES; its failure keeps the run's
        recorded config (the overlay is runtime only) and is recorded failed."""
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
        run_yaml = Path(pipeline.layout.config).read_text(encoding="utf-8")
        pipeline.manifest.status = RunStatus.SUBMITTED
        pipeline.manifest.save(pipeline.layout)
        monkeypatch.setenv(
            "RCP_NDCG_ENGINES",
            json.dumps({"judge": {"urls": ["http://node1:8000/v1", "http://node2:8000/v1"], "wait_on_outage_s": 900}}),
        )
        args = ["run", "resume", "--run", pipeline.layout.root, "--json"]
        assert CliRunner().invoke(cli, args).exit_code == 4

        manifest = RunManifest.load(pipeline.layout)
        assert manifest.status is RunStatus.FAILED
        assert (
            manifest.step("retrieve").status is StepStatus.FAILED
            and "missing.parquet" in manifest.step("retrieve").error
        )
        # The overlay never reaches run.yaml: the recorded config still names its own judge URL.
        assert Path(pipeline.layout.config).read_text(encoding="utf-8") == run_yaml
        assert yaml.safe_load(run_yaml)["judge"]["base_url"] == "http://engine:8000/v1"

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
        from rcp_ndcg.judging import JudgeClient
        from rcp_ndcg.judging.client import BackendUnavailableError

        complete = JudgeClient.complete
        answered = []

        async def down_after_three(self, request):
            if len(answered) >= 3:
                raise BackendUnavailableError("the endpoint went away")
            answered.append(request)
            return await complete(self, request)

        monkeypatch.setattr(JudgeClient, "complete", down_after_three)
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


def test_a_resume_whose_judge_config_is_gone_raises_the_typed_error(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resume of a finished run whose judge config file is gone fails with the typed MissingInputError from
    the identity check -- not the AttributeError of the failure handler touching an unset usage, which used to
    mask it and skip finish_step."""
    from rcp_ndcg.judging import JudgeClient
    from tests.judging.test_judging import _SchemaEndpoint

    judge_yaml = tmp_path / "judge.yaml"
    judge_yaml.write_text("base_url: http://judge.test/v1\nmodel: m\n", encoding="utf-8")
    monkeypatch.setattr(JudgeClient, "from_config", staticmethod(lambda config: _SchemaEndpoint("3.1").client()))
    pipeline = Pipeline(tiny_config(data, judge=str(judge_yaml), steps=["tournament"]), runs_dir=str(tmp_path / "runs"))
    pipeline.run()
    assert RunManifest.load(pipeline.layout.root).step("tournament").succeeded
    judge_yaml.unlink()
    with pytest.raises(MissingInputError, match="judge config"):
        Pipeline.resume(pipeline.layout.root).run()
    manifest = RunManifest.load(pipeline.layout.root)
    assert manifest.status is RunStatus.FAILED
    assert manifest.step("tournament").error and "judge.yaml" in manifest.step("tournament").error


def test_a_rankings_sourced_rerank_run_without_a_retrieve_step_reranks_the_supplied_pools(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`from: rankings` + a rerank step, with no retrieve step: the rankings file IS the first stage, so the
    run works -- the preamble writes the first stage from the supplied pools, the reranker rescores it, and
    the judging steps read its candidates (the combination used to fail mid-run on an internal scratch path)."""
    from rcp_ndcg.judging import JudgeClient
    from tests.judging.test_judging import _SchemaEndpoint

    rankings = tmp_path / "rankings.jsonl"
    rows, _ = tiny_rows()
    rankings.write_text(
        "".join(json.dumps({"query_id": row.id, "system": "bm25", "doc_ids": row.doc_ids[:6]}) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(JudgeClient, "from_config", staticmethod(lambda config: _SchemaEndpoint("3.1").client()))

    def score_by_position(self, examples, *, checkpoint=None, **kwargs):
        """The stub client scores each document by its pool position: it prefers the pool's last documents."""
        for example in examples:
            checkpoint(str(example.id), tuple(float(i) for i in range(len(example.doc_ids))))
        return []

    monkeypatch.setattr("rcp_ndcg.retrieval._api.RerankClient.rerank_many", score_by_position)
    config = tiny_config(
        data,
        candidates={
            "from": "rankings",
            "rankings": str(rankings),
            "system": "bm25",
            "rerank": {"api": "rerank", "model": "stub", "base_url": "http://stub:8000", **_SERVED_RERANK_BUDGET},
            "depth": 3,
        },
        steps=["rerank", "tournament", "rubric", "calibrate", "evaluate"],
    )
    pipeline = Pipeline(config, runs_dir=str(tmp_path / "runs"))
    manifest = pipeline.run()
    assert manifest.status is RunStatus.COMPLETED
    assert manifest.step("rerank").succeeded
    written = load_rankings(pipeline.layout.candidates)
    assert written.systems == ["candidates"], "the reranker's order is what the judging steps read"
    # The stub scores by pool position: the supplied pools' last documents now rank first -- the reranker
    # rescored the rankings file's own pools, not a retrieval's. And the JUDGES judged the reranked order's
    # top documents, not the raw rankings' (the judged set is the reranker's best depth).
    rows, _ = tiny_rows()
    judged = {
        placement["doc_id"]
        for record in map(json.loads, (Path(pipeline.layout.root) / "judgements/tournament.jsonl").open())
        for placement in record["placements"]
    }
    for row in rows:
        assert _order(written.for_query(row.id)) == row.doc_ids[:6][::-1]
        assert judged == {doc for row2 in rows for doc in row2.doc_ids[:6][::-1][:3]}, (
            "the judges judged the reranked order's top depth, not the raw rankings'"
        )


def test_a_rankings_run_without_retrieve_judges_its_supplied_pools(data: Path, tmp_path: Path) -> None:
    rows, _ = tiny_rows()
    rankings = tmp_path / "rankings.jsonl"
    rankings.write_text(
        "".join(json.dumps({"query_id": row.id, "system": "bm25", "doc_ids": row.doc_ids[:6]}) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = tiny_config(
        data, candidates={"from": "rankings", "rankings": str(rankings)}, steps=["tournament", "rubric"]
    )
    manifest = Pipeline(config, runs_dir=str(tmp_path / "runs")).run()
    assert manifest.step("tournament").succeeded, "the rankings file is the first stage; no retrieve step needed"


def test_a_retrieval_sourced_rerank_run_without_a_retrieve_step_is_refused(data: Path) -> None:
    """`from: retrieval` really has no first stage until the retrieve step runs it: the config is refused."""
    with pytest.raises(Exception, match="retrieve step"):
        tiny_config(
            data,
            candidates={
                "from": "retrieval",
                "retrieval": {"kind": "bm25"},
                "rerank": {"api": "cohere", "model": "rerank-v4.0"},
            },
            steps=["rerank"],
        )


def test_a_rankings_run_without_retrieve_pins_the_rankings_file_on_resume(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `from: rankings` run without a retrieve step reads its pools straight from the rankings file: the
    file is the rerank and judging steps' input, so a resume after editing it re-runs the steps (a stale
    candidates file would otherwise be kept silently -- the inputs were empty and the identity unchanged)."""
    from rcp_ndcg.judging import JudgeClient
    from tests.judging.test_judging import _SchemaEndpoint

    rankings = tmp_path / "rankings.jsonl"
    rows, _ = tiny_rows()
    rankings.write_text(
        "".join(json.dumps({"query_id": row.id, "system": "bm25", "doc_ids": row.doc_ids[:6]}) + "\n" for row in rows),
        encoding="utf-8",
    )
    monkeypatch.setattr(JudgeClient, "from_config", staticmethod(lambda config: _SchemaEndpoint("3.1").client()))
    config = tiny_config(data, candidates={"from": "rankings", "rankings": str(rankings)}, steps=["tournament"])
    pipeline = Pipeline(config, runs_dir=str(tmp_path / "runs"))
    pipeline.run()
    record = RunManifest.load(pipeline.layout.root).step("tournament")
    assert [ref.path for ref in record.inputs] == [str(rankings)], "the rankings file is pinned as the input"

    rankings.write_text(
        "".join(
            json.dumps({"query_id": row.id, "system": "bm25", "doc_ids": list(reversed(row.doc_ids[:6]))}) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    plan = {row["step"]: row["status"] for row in Pipeline.resume(pipeline.layout.root).plan()}
    assert plan["tournament"] == "would run", "the edited rankings file makes the step stale"


def test_naming_the_judges_default_wire_does_not_rekey_the_judge_step(data: Path, tmp_path: Path) -> None:
    """`api: openai_chat` names the default wire: the same instrument, so the judge STEP identity is the unset
    case's (the family and the store gate already normalize; the pipeline does too)."""
    from tests.judging.test_judging import _SchemaEndpoint  # noqa: F401  (import keeps the fake route registered)

    def identity(**judge: Any) -> dict[str, Any]:
        return Pipeline(
            tiny_config(data, judge={"base_url": "http://judge.test/v1", "model": "m", **judge}, steps=["rubric"]),
            runs_dir=str(tmp_path / "runs"),
        )._identity("rubric")

    assert identity() == identity(api="openai_chat"), "a spelling of the default wire is the same instrument"
    assert identity(api="other") != identity(), "another wire is a different instrument"
