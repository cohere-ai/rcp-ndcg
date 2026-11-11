"""The run layout and manifest: fixed names, step records, usage, persistence."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field
from rcp_core.schemas import JudgementFamily

from rcp.errors import ConfigError, DataError
from rcp.judging import Usage
from rcp.runs import (
    DatasetRef,
    RunLayout,
    RunManifest,
    RunStatus,
    StepRecord,
    StepStatus,
    discover_runs,
    new_run_id,
)
from rcp.storage.artifacts import CodeVersion, artifact_ref
from rcp.support.identity import hash_payload


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
        from rcp.runners.base import JobStatus
        from rcp.runs.run import JobState, RunState, StepState

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
        assert (payload["schema"], payload["layout"]) == ("rcp.run-manifest.v1", "rcp-ndcg.run-layout.v1")
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

        assert "rcp.run-manifest.v1" in caught.value.message
        assert caught.value.hint and "run" in caught.value.hint


class _FirstV1Reader(BaseModel):
    """The first ``run-manifest.v1``'s thirteen fields, and only those, under the additive rule.

    A reader that ignores the fields it does not know reads a later ``v1`` record: the additions are optional,
    so it does not need them. Pinning the field set here means a change that made a new field required, or
    changed an old field's type, fails this reader instead of passing unnoticed. The shipped reader is not
    this one: ``RunManifest`` is a closed record and refuses a field it does not know, which
    ``test_a_first_v1_reader_reads_a_later_v1_manifest`` pins beside this reader.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # The artifact identity moved from `rcp-ndcg.` to `rcp.` while the fields stayed the same, so a reader
    # written against the first v1 accepts both spellings: that is what the move promised, and `accepted_schema_ids`
    # is where the product states it.
    schema_name: Literal["rcp.run-manifest.v1", "rcp-ndcg.run-manifest.v1"] = Field(alias="schema")
    layout: Literal["rcp-ndcg.run-layout.v1"] = "rcp-ndcg.run-layout.v1"
    run_id: str
    created_at: datetime
    updated_at: datetime
    status: RunStatus = RunStatus.RUNNING
    code: CodeVersion
    config: dict[str, Any] = Field(default_factory=dict)
    dataset: DatasetRef | None = None
    families: dict[str, JudgementFamily] = Field(default_factory=dict)
    steps: list[StepRecord] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)


class TestTheV1Additions:
    """G4: five optional fields were added to ``run-manifest.v1``, and no existing field moved."""

    @staticmethod
    def _first_v1() -> dict[str, Any]:
        """A manifest exactly as the first ``v1`` wrote it: thirteen properties, no additions."""
        return {
            "schema": "rcp-ndcg.run-manifest.v1",
            "layout": "rcp-ndcg.run-layout.v1",
            "run_id": "20260102-030405-first-v1",
            "created_at": "2026-01-02T03:04:05Z",
            "updated_at": "2026-01-02T03:04:05Z",
            "status": "completed",
            "code": {"package_version": "0.0.1", "python_version": "3.12", "platform": "linux"},
            "config": {"dataset": "tiny"},
            "dataset": {"name": "tiny", "subset": "test", "split": "test"},
            "families": {},
            "steps": [],
            "metrics": {"mine/qrel_ndcg@10": 0.5},
            "usage": {"requests": 1, "input_tokens": 10},
        }

    @staticmethod
    def _later_v1(layout: RunLayout) -> dict[str, Any]:
        """A later ``v1`` manifest: the same fields, plus the five additions populated."""
        return {
            "run_id": layout.run_id,
            "code": {"package_version": "0.0.1", "python_version": "3.12", "platform": "linux"},
            "datasets": [
                {
                    "name": "tiny",
                    "subset": "test",
                    "split": "test",
                    "internal": True,
                    "args": {"qrels_uri": "qrels.jsonl"},
                }
            ],
            "suites": [{"name": "nanobeir", "entries": ["NanoNFCorpusRetrieval"]}],
            "rows": {"schema": "example.rows.v1"},
        }

    def test_a_first_v1_manifest_still_validates(self) -> None:
        manifest = RunManifest.model_validate(self._first_v1())

        assert manifest.run_id == "20260102-030405-first-v1"
        assert manifest.metrics == {"mine/qrel_ndcg@10": 0.5}
        assert manifest.dataset is not None and manifest.dataset.subset == "test"
        # The additions are optional: an old record gets their defaults and no existing value moves.
        assert manifest.path is None
        assert manifest.datasets == []
        assert manifest.suites == []
        assert manifest.rows is None

    def test_a_later_v1_manifest_round_trips_the_added_fields(self, layout: RunLayout) -> None:
        manifest = RunManifest.model_validate(self._later_v1(layout))
        manifest.save(layout)

        loaded = RunManifest.load(layout)
        assert loaded == manifest
        assert loaded.path == layout.root
        assert loaded.datasets[0].name == "tiny"
        assert loaded.datasets[0].internal is True
        assert loaded.datasets[0].args == {"qrels_uri": "qrels.jsonl"}
        assert loaded.suites[0].name == "nanobeir"
        assert loaded.suites[0].entries == ["NanoNFCorpusRetrieval"]
        assert loaded.rows is not None and loaded.rows.schema_name == "example.rows.v1"

        payload = json.loads(Path(layout.manifest).read_text(encoding="utf-8"))
        assert payload["path"] == layout.root
        assert payload["datasets"][0]["internal"] is True
        assert payload["datasets"][0]["args"] == {"qrels_uri": "qrels.jsonl"}
        assert payload["suites"][0]["entries"] == ["NanoNFCorpusRetrieval"]
        assert payload["rows"] == {"schema": "example.rows.v1"}

    def test_a_first_v1_reader_reads_a_later_v1_manifest(self, layout: RunLayout) -> None:
        manifest = RunManifest.model_validate(self._later_v1(layout))
        manifest.save(layout)

        first = _FirstV1Reader.model_validate_json(Path(layout.manifest).read_text(encoding="utf-8"))
        assert first.run_id == manifest.run_id
        assert first.config == manifest.config
        assert first.metrics == manifest.metrics
        assert "path" not in first.model_dump()

        # The shipped reader is a closed record: it reads a later record only while it knows every field, and
        # it refuses a field it does not know instead of dropping it. A field added after a release therefore
        # needs a reader at least as new as the writer -- a refusal, never a misread -- and the additive rule
        # is what keeps that the only breakage.
        later = json.loads(Path(layout.manifest).read_text(encoding="utf-8")) | {"added_later": 1}
        Path(layout.manifest).write_text(json.dumps(later), encoding="utf-8")
        with pytest.raises(DataError) as caught:
            RunManifest.load(layout)
        assert "added_later" in caught.value.message
        assert _FirstV1Reader.model_validate(later).run_id == manifest.run_id

    def test_the_added_fields_are_optional_and_the_manifest_stays_closed(self) -> None:
        schema = RunManifest.model_json_schema(mode="serialization")

        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == {"run_id", "code"}
        added = {"path", "datasets", "suites", "rows"}
        assert added <= set(schema["properties"])
        assert added.isdisjoint(schema["required"])

        definitions = schema["$defs"]
        assert definitions["DatasetEntry"]["properties"]["internal"]["default"] is False
        assert "args" not in definitions["DatasetEntry"]["required"]
        assert "internal" not in definitions["DatasetEntry"]["required"]
        assert "entries" not in definitions["SuiteEntry"]["required"]
        assert definitions["Rows"]["properties"]["schema"]["default"] is None
        # The existing ``dataset`` field keeps its closed shape: the additions live on the new array's entries.
        assert {"internal", "args"}.isdisjoint(definitions["DatasetRef"]["properties"])


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
