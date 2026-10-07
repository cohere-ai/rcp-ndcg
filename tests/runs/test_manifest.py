"""The run layout and manifest: fixed names, step records, usage, persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rcp_ndcg.errors import ConfigError, DataError
from rcp_ndcg.llm import Usage
from rcp_ndcg.runs import RunLayout, RunManifest, RunStatus, StepStatus, discover_runs, new_run_id
from rcp_ndcg.storage.artifacts import artifact_ref
from rcp_ndcg.support.identity import hash_payload


@pytest.fixture
def layout(tmp_path: Path) -> RunLayout:
    return RunLayout.at(tmp_path / "runs" / "20260724-000000-test").ensure()


def _manifest(run_id: str = "r1", **kwargs) -> RunManifest:
    return RunManifest.new(run_id, config=kwargs.pop("config", {}))


class TestRunIds:
    def test_ids_sort_chronologically_and_carry_the_label(self) -> None:
        assert new_run_id() > "20260724-090000-abc123"
        assert "nanobeir-gpt-5" in new_run_id("NanoBEIR / GPT-5")

    def test_ids_are_unique_within_a_second(self) -> None:
        assert new_run_id("x") != new_run_id("x")


class TestLayout:
    def test_artifacts_have_fixed_names(self, layout: RunLayout) -> None:
        assert layout.manifest.endswith("/manifest.json")
        assert layout.config.endswith("/run.yaml")
        assert layout.candidates.endswith("/candidates.parquet")
        assert layout.judgements.endswith("/judgements")
        assert layout.calibration.endswith("/calibration")
        assert layout.metrics.endswith("/metrics/report.json")
        assert layout.comparison.endswith("/metrics/comparison.json")
        for directory in (layout.judgements, layout.calibration, layout.metrics_dir, layout.logs_dir):
            assert Path(directory).is_dir()

    def test_paths_inside_the_run_are_recorded_relative(self, layout: RunLayout) -> None:
        # So a run directory can be moved or copied to a bucket unedited.
        assert layout.relative(layout.candidates) == "candidates.parquet"
        assert layout.relative("/elsewhere/corpus.jsonl") == "/elsewhere/corpus.jsonl"
        assert layout.resolve("candidates.parquet") == layout.candidates
        assert layout.resolve("gs://bucket/x.jsonl") == "gs://bucket/x.jsonl"

    def test_only_directories_with_a_manifest_are_runs(self, tmp_path: Path) -> None:
        runs = tmp_path / "runs"
        for run_id in ("20260101-000000-a", "20260102-000000-b"):
            _manifest(run_id).save(RunLayout.at(runs / run_id).ensure())
        RunLayout.at(runs / "20260103-000000-interrupted").ensure()
        assert discover_runs(runs) == ["20260102-000000-b", "20260101-000000-a"]
        assert discover_runs(tmp_path / "absent") == []


class TestSteps:
    def test_a_step_records_its_identity_and_timing(self) -> None:
        manifest = _manifest()
        manifest.start_step("retrieve", identity={"depth": 100})
        manifest.finish_step("retrieve")
        record = manifest.step("retrieve")
        assert record.status is StepStatus.COMPLETED
        assert record.identity_hash == hash_payload({"depth": 100})
        assert record.duration_s is not None and record.duration_s >= 0
        assert record.succeeded

    def test_the_status_state_models_carry_the_enums(self) -> None:
        """RunState/StepState/JobState status fields are the three vocabularies themselves (one enum each),
        `StepStatus` covers `pending` (which `run status` emits for a step not started), and a finished-OK job
        reads `completed` like a finished-OK step and run — one word per meaning."""
        from rcp_ndcg.runners.base import JobStatus
        from rcp_ndcg.runs.run import JobState, RunState, StepState

        assert StepState.model_fields["status"].annotation is StepStatus
        assert RunState.model_fields["status"].annotation is RunStatus
        assert JobState.model_fields["status"].annotation is JobStatus
        assert [status.value for status in StepStatus] == ["pending", "running", "completed", "failed", "cancelled"]
        assert [status.value for status in JobStatus] == [
            "pending", "running", "completed", "failed", "cancelled", "unknown",
        ]  # fmt: skip
        assert [status.value for status in RunStatus] == [
            "submitted", "running", "completed", "failed", "partial", "cancelled",
        ]  # fmt: skip

    def test_a_retry_reuses_one_record(self) -> None:
        manifest = _manifest()
        manifest.start_step("retrieve", identity={})
        manifest.finish_step("retrieve", status=StepStatus.FAILED, error="boom")
        assert not manifest.step("retrieve").succeeded
        manifest.start_step("retrieve", identity={})
        manifest.finish_step("retrieve")
        assert [record.name for record in manifest.steps] == ["retrieve"]
        assert manifest.step("retrieve").error is None

    def test_completed_counts_as_done_and_failed_does_not(self) -> None:
        manifest = _manifest()
        manifest.finish_step("retrieve")
        manifest.finish_step("tournament", status=StepStatus.FAILED)
        assert manifest.step("retrieve").succeeded
        assert not manifest.step("tournament").succeeded

    def test_usage_rolls_up(self) -> None:
        manifest = _manifest()
        manifest.finish_step("tournament", usage=Usage(requests=10, input_tokens=100))
        manifest.finish_step("rubric", usage=Usage(requests=5, input_tokens=50))
        assert manifest.usage.requests == 15
        assert manifest.usage.input_tokens == 150

    def test_outputs_are_recorded_run_relative_with_their_hash(self, layout: RunLayout) -> None:
        store = Path(layout.judgements) / "tournament.jsonl"
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text('{"a":1}\n{"a":2}\n', encoding="utf-8")
        ref = artifact_ref(str(store), layout=layout)
        assert (ref.path, len(ref.sha256)) == ("judgements/tournament.jsonl", 64)
        manifest = _manifest()
        manifest.finish_step("tournament", outputs=[ref])
        assert manifest.step("tournament").outputs == [ref]


class TestPersistence:
    def test_round_trips_through_disk(self, layout: RunLayout) -> None:
        manifest = _manifest(layout.run_id, config={"dataset": "nano"})
        manifest.start_step("tournament", identity={"k": 1})
        manifest.finish_step("tournament", usage=Usage(requests=3, output_tokens=25))
        manifest.save(layout)
        loaded = RunManifest.load(layout.root)
        assert loaded == RunManifest.load(layout)
        assert loaded.config == {"dataset": "nano"}
        assert (loaded.usage.requests, loaded.usage.output_tokens) == (3, 25)
        assert loaded.step("tournament").identity == {"k": 1}
        assert loaded.status is RunStatus.RUNNING
        assert loaded.code.package_version
        payload = json.loads(Path(layout.manifest).read_text(encoding="utf-8"))
        assert (payload["schema"], payload["layout"]) == ("rcp-ndcg.run-manifest.v1", "rcp-ndcg.run-layout.v1")
        assert not list(Path(layout.root).glob("*.tmp"))

    def test_a_remote_run_directory_is_refused(self) -> None:
        # A preempted job must not lose what a bucket did not receive yet: runs are local and mirrored.
        with pytest.raises(ConfigError, match="local") as refused:
            RunLayout.at("memory://runs/r1")
        assert "--mirror" in (refused.value.hint or "")

    def test_another_schema_is_refused(self, layout: RunLayout) -> None:
        # Silently misreading a future manifest is worse than failing.
        _manifest(layout.run_id).save(layout)
        payload = json.loads(Path(layout.manifest).read_text(encoding="utf-8"))
        payload["schema"] = "rcp-ndcg.run-manifest.v99"
        Path(layout.manifest).write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(DataError) as caught:
            RunManifest.load(layout)

        assert "rcp-ndcg.run-manifest.v1" in caught.value.message
        assert caught.value.hint and "run" in caught.value.hint


class TestAFailedRerun:
    def test_a_failed_rerun_leaves_no_outputs_of_the_attempt_it_did_not_run(self) -> None:
        """A step re-run that fails records what THAT attempt did: the previous attempt's inputs, outputs,
        usage and engines describe work this attempt did not do (a record saying ``failed`` while listing
        outputs it never wrote is a lie a reader cannot tell from the truth)."""
        manifest = _manifest()
        manifest.start_step("evaluate", identity={"a": 1})
        manifest.finish_step("evaluate", inputs=["in"], outputs=["y.parquet"])
        record = manifest.step("evaluate")
        assert record.outputs == ["y.parquet"] and record.inputs == ["in"]

        manifest.start_step("evaluate", identity={"a": 2})
        manifest.finish_step("evaluate", status=StepStatus.FAILED, error="boom")
        record = manifest.step("evaluate")
        assert record.status is StepStatus.FAILED and record.error == "boom"
        assert record.outputs == [] and record.inputs == [], "the failed attempt wrote nothing"
