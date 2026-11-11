"""The run's event stream: the vocabulary, the emitter and the order a run publishes it in.

A subscriber is a callable installed in the ``rcp.events`` entry-point group. These tests install one by
pointing the module's ``entry_points`` lookup at a fake entry point (the way the result-sink tests do), so the
real ``importlib.metadata`` path is what runs, with the group's content under the test's control.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from rcp.errors import IdentityError
from rcp.runs import Pipeline, RunManifest, RunStatus, StepStatus
from rcp.support import events
from rcp.support.events import Emitter, RunEvent
from tests.runs.conftest import STEPS, tiny_config


@dataclass(frozen=True)
class FakeEntryPoint:
    """An entry point with the fields the emitter reads (``load`` returns the target directly)."""

    name: str
    target: Any
    group: str = "rcp.events"
    failure: Exception | None = None

    def load(self) -> Any:
        if self.failure is not None:
            raise self.failure
        return self.target


def install(monkeypatch: pytest.MonkeyPatch, *entry_points: FakeEntryPoint) -> None:
    """Point the ``rcp.events`` lookup at *entry_points* (other groups stay empty)."""
    monkeypatch.setattr(
        events,
        "entry_points",
        lambda group: [entry for entry in entry_points if entry.group == group],
    )


@pytest.fixture
def received(monkeypatch: pytest.MonkeyPatch) -> list[RunEvent]:
    """A recording subscriber, installed in the group, and the events it received."""
    log: list[RunEvent] = []
    install(monkeypatch, FakeEntryPoint("recorder", log.append))
    return log


def _recorded(payload: dict[str, Any]) -> dict[str, Any]:
    """The manifest without what two runs of one config cannot share: ids, clocks, durations, and the hashes of
    the judgement store files, whose every record carries the wall clock it was written at. Every path, status,
    identity, usage and metric stays, so the comparison is of what the run recorded, not of when."""
    # `path` is the run's own directory: two runs of one config in different roots cannot share it, and
    # it is the only L4 addition that is about where the run lives rather than what it recorded.
    volatile = ("run_id", "created_at", "updated_at", "path")
    stable = {key: value for key, value in payload.items() if key not in volatile}
    stable["steps"] = [
        {
            **{key: value for key, value in step.items() if key not in ("started_at", "ended_at", "duration_s")},
            **{
                kind: [{k: v for k, v in ref.items() if k != "sha256"} for ref in step[kind]]
                for kind in ("inputs", "outputs")
            },
        }
        for step in payload["steps"]
    ]
    return stable


class TestTheSequence:
    def test_a_run_publishes_one_started_and_one_finished_per_step_in_run_order(
        self, data: Path, tmp_path: Path, received: list[RunEvent]
    ) -> None:
        """The documented sequence: a step's ``step_started``, its ``step_finished``, its ``usage`` when the step
        reported any; then the next step. Every event names the run and the step it belongs to."""
        pipeline = Pipeline(tiny_config(data), runs_dir=str(tmp_path / "runs"))
        manifest = pipeline.run()

        assert [(event.phase, event.step) for event in received] == [
            (phase, step)
            for step in STEPS
            for phase in ("step_started", "step_finished", *(["usage"] if step in ("tournament", "rubric") else []))
        ]
        assert {event.run_id for event in received} == {manifest.run_id}
        assert all(event.at.tzinfo is not None for event in received)
        assert [event.at for event in received] == sorted(event.at for event in received)

    def test_a_finished_event_carries_the_record_and_a_usage_event_carries_the_counts(
        self, data: Path, tmp_path: Path, received: list[RunEvent]
    ) -> None:
        pipeline = Pipeline(tiny_config(data), runs_dir=str(tmp_path / "runs"))
        manifest = pipeline.run()

        finished = next(event for event in received if event.phase == "step_finished")
        assert finished.step == "tournament"
        assert finished.detail["status"] == "completed"
        assert finished.detail["duration_s"] >= 0
        assert finished.usage is None

        usage = next(event for event in received if event.phase == "usage")
        assert usage.step == "tournament"
        record = manifest.step("tournament")
        assert record is not None and record.usage is not None
        assert usage.usage == record.usage.model_dump()

    def test_a_subscriber_sees_the_manifest_the_step_already_wrote(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The event is emitted after the record is final, not before it.

        A subscriber that reads ``manifest.json`` while it is called must find the step it is being told about
        already written. Emitting before the write would still pass every assertion about the event's own
        fields, so the ordering is pinned here, from the subscriber's side.
        """
        runs = tmp_path / "runs"
        seen: list[tuple[str, str | None]] = []

        def subscribe(event: RunEvent) -> None:
            manifest_path = next(runs.glob("*/manifest.json"), None)
            recorded = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path else {}
            steps = {step["name"]: step["status"] for step in recorded.get("steps", [])}
            seen.append((event.phase, steps.get(event.step)))

        monkeypatch.setattr("rcp.runs.pipeline.Emitter.installed", classmethod(lambda cls: Emitter([subscribe])))
        Pipeline(tiny_config(data), runs_dir=str(runs)).run()

        finished = [status for phase, status in seen if phase == "step_finished"]
        assert finished, seen
        assert all(status == "completed" for status in finished), seen
        # `step_started` is emitted after the step is recorded as running, so its status is already there.
        started = [status for phase, status in seen if phase == "step_started"]
        assert started and all(status == "running" for status in started), seen

    def test_a_failed_step_emits_step_failed_and_the_run_stops(
        self, data: Path, tmp_path: Path, received: list[RunEvent]
    ) -> None:
        """A step that raises closes with ``step_failed`` (never ``step_finished``), carrying the manifest's
        redacted error, and no later step starts."""
        pipeline = Pipeline(tiny_config(data, steps=["evaluate"]), runs_dir=str(tmp_path / "runs"))
        with pytest.raises(Exception):  # noqa: B017 - the calibration the evaluate step reads is not there
            pipeline.run()

        assert [(event.phase, event.step) for event in received] == [
            ("step_started", "evaluate"),
            ("step_failed", "evaluate"),
        ]
        assert received[-1].detail["error"].startswith("MissingInputError: ")
        assert received[-1].detail["status"] == "failed"
        assert pipeline.manifest.status is RunStatus.FAILED
        record = pipeline.manifest.step("evaluate")
        assert record is not None and record.status is StepStatus.FAILED

    def test_a_cancelled_judging_step_says_cancelled_and_still_reports_its_usage(
        self, data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, received: list[RunEvent]
    ) -> None:
        """``run cancel`` during a judging step: the closing event says ``cancelled``, not ``failed``, and the
        usage event still reports what the step asked the judge before it stopped."""
        from rcp.judging import JudgeClient
        from rcp.judging.client import BackendUnavailableError

        pipeline = Pipeline(tiny_config(data, steps=["tournament"]), runs_dir=str(tmp_path / "runs"))
        complete = JudgeClient.complete
        answered = []

        async def stopped_midway(self, request):
            if len(answered) >= 3:
                on_disk = RunManifest.load(pipeline.layout)
                on_disk.status = RunStatus.CANCELLED
                on_disk.save(pipeline.layout)
                raise BackendUnavailableError("the job was stopped")
            answered.append(request)
            return await complete(self, request)

        monkeypatch.setattr(JudgeClient, "complete", stopped_midway)
        with pytest.raises(BackendUnavailableError):
            pipeline.run()

        assert [(event.phase, event.step) for event in received] == [
            ("step_started", "tournament"),
            ("step_failed", "tournament"),
            ("usage", "tournament"),
        ]
        assert received[1].detail["status"] == "cancelled"
        assert received[2].usage is not None and received[2].usage["requests"] >= 3

    def test_a_refused_store_identity_emits_step_failed(
        self, finished: Path, tmp_path: Path, received: list[RunEvent]
    ) -> None:
        """A judging step that refuses another identity closes with ``step_failed``: the refusal reaches a
        subscriber, and the run keeps the records it had."""
        import shutil

        run_dir = Path(shutil.copytree(finished, tmp_path / finished.name))
        with pytest.raises(IdentityError):
            Pipeline.resume(run_dir, overrides=["rubric.seed=7"]).run()

        assert [(event.phase, event.step) for event in received] == [
            ("step_started", "rubric"),
            ("step_failed", "rubric"),
            ("usage", "rubric"),
        ]
        assert received[1].detail["error"].startswith("IdentityError: ")
        record = RunManifest.load(run_dir).step("rubric")
        assert record is not None and record.status is StepStatus.COMPLETED

    def test_a_step_a_resume_skips_emits_nothing(
        self, finished: Path, tmp_path: Path, received: list[RunEvent]
    ) -> None:
        import shutil

        run_dir = Path(shutil.copytree(finished, tmp_path / finished.name))
        Pipeline.resume(run_dir).run()
        assert received == []


class TestASubscriberThatRaises:
    def test_the_run_completes_and_the_failure_is_logged_once(
        self,
        data: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        def broken(event: RunEvent) -> None:
            raise RuntimeError("the dashboard is down")

        install(monkeypatch, FakeEntryPoint("broken", broken))
        pipeline = Pipeline(tiny_config(data), runs_dir=str(tmp_path / "runs"))
        with caplog.at_level(logging.WARNING, logger="rcp"):
            manifest = pipeline.run()

        assert manifest.status is RunStatus.COMPLETED
        warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert len(warnings) == 1, [record.getMessage() for record in warnings]
        assert "broken" in warnings[0].getMessage()
        assert "the dashboard is down" in warnings[0].getMessage()

    def test_the_other_subscribers_still_receive_every_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[RunEvent] = []

        def broken(event: RunEvent) -> None:
            raise RuntimeError("no")

        install(monkeypatch, FakeEntryPoint("broken", broken), FakeEntryPoint("recorder", seen.append))
        emitter = Emitter.installed()
        for step in ("retrieve", "rerank"):
            emitter.emit(RunEvent(run_id="r-1", step=step, phase="step_started"))
        assert [event.step for event in seen] == ["retrieve", "rerank"]

    def test_a_subscriber_that_does_not_load_is_logged_and_the_others_receive(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        seen: list[RunEvent] = []
        install(
            monkeypatch,
            FakeEntryPoint("gone", None, failure=ImportError("no such module")),
            FakeEntryPoint("recorder", seen.append),
        )
        with caplog.at_level(logging.WARNING, logger="rcp"):
            Emitter.installed().emit(RunEvent(run_id="r-1", step="retrieve", phase="step_started"))
        assert [event.step for event in seen] == ["retrieve"]
        assert "gone" in caplog.text

    def test_a_target_that_is_not_callable_is_logged_and_skipped(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        seen: list[RunEvent] = []
        install(monkeypatch, FakeEntryPoint("not-a-subscriber", 42), FakeEntryPoint("recorder", seen.append))
        with caplog.at_level(logging.WARNING, logger="rcp"):
            Emitter.installed().emit(RunEvent(run_id="r-1", step="retrieve", phase="step_started"))
        assert [event.step for event in seen] == ["retrieve"]
        assert "not-a-subscriber" in caplog.text


class TestNoSubscriber:
    def test_a_run_with_no_subscriber_emits_nothing_and_records_the_same_manifest(
        self,
        data: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        install(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="rcp"):
            silent = Pipeline(tiny_config(data), runs_dir=str(tmp_path / "silent")).run()
        assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []

        seen: list[RunEvent] = []
        install(monkeypatch, FakeEntryPoint("recorder", seen.append))
        subscribed = Pipeline(tiny_config(data), runs_dir=str(tmp_path / "subscribed")).run()

        assert seen, "the subscriber of the second run received its events"
        assert _recorded(silent.model_dump(mode="json")) == _recorded(subscribed.model_dump(mode="json"))


class TestTheEmitter:
    def test_every_subscriber_receives_the_event_in_the_order_it_was_emitted(self) -> None:
        first: list[RunEvent] = []
        second: list[RunEvent] = []
        emitter = Emitter([first.append, second.append])
        for phase in ("step_started", "step_finished", "usage"):
            emitter.emit(RunEvent(run_id="r-1", step="retrieve", phase=phase))  # type: ignore[arg-type]
        assert [event.phase for event in first] == ["step_started", "step_finished", "usage"]
        assert first == second

    def test_an_empty_emitter_does_nothing(self) -> None:
        Emitter().emit(RunEvent(run_id="r-1", step="retrieve", phase="step_started"))

    def test_the_group_is_read_in_entry_point_name_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        order: list[str] = []
        install(
            monkeypatch,
            FakeEntryPoint("b", lambda event: order.append("b")),
            FakeEntryPoint("a", lambda event: order.append("a")),
        )
        Emitter.installed().emit(RunEvent(run_id="r-1", step="retrieve", phase="step_started"))
        assert order == ["a", "b"]


class TestTheModel:
    def test_the_event_carries_the_documented_fields(self) -> None:
        event = RunEvent(run_id="20260724-151233-tiny-a1b2c3", step="retrieve", phase="step_started")
        payload = event.model_dump(mode="json")
        assert set(payload) == {"run_id", "step", "phase", "at", "usage", "detail"}
        assert json.loads(json.dumps(payload))["phase"] == "step_started"
        assert payload["usage"] is None and payload["detail"] == {}

    def test_the_phases_are_the_documented_four(self) -> None:
        assert events.PHASES == ("step_started", "step_finished", "step_failed", "usage")
        with pytest.raises(ValueError, match="phase"):
            RunEvent(run_id="r-1", step="retrieve", phase="step_cancelled")  # type: ignore[arg-type]

    def test_the_group_name_is_the_documented_one(self) -> None:
        assert events.EVENTS_GROUP == "rcp.events"
