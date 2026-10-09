"""``rcp_ndcg.testing.results_conformance``: the sink contract as one check a sink's own tests call.

A sink consumes a versioned record and nothing else: its name is a registered entry point, its constructor takes
``uri`` first, it accepts a valid record without mutating it, ``flush`` is idempotent, and it refuses a record
whose schema it does not know. The built-in sinks run through the same call a plugin's test suite runs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.results import (
    RESULT_SCHEMA,
    JsonlResultSink,
    NullResultSink,
    ParquetResultSink,
    ResultDataset,
    ResultMetric,
    ResultRecord,
    ResultsSink,
    ResultSubject,
    record_identity,
)
from rcp_ndcg.testing import results_conformance
from tests.results._sinks import inject_sinks


def _record() -> ResultRecord:
    subject = ResultSubject(kind="system", name="mine", system="mine")
    dataset = ResultDataset(name="toy")
    metrics = [ResultMetric(metric="rcp_ndcg", k=10, value=0.5, num_queries=2)]
    return ResultRecord(
        record_id=record_identity(subject, dataset, metrics),
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        subject=subject,
        dataset=dataset,
        metrics=metrics,
    )


def test_the_builtin_sinks_pass_the_conformance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)

    results_conformance(JsonlResultSink(uri=str(tmp_path / "records.jsonl")))
    results_conformance(ParquetResultSink(uri=str(tmp_path / "records.parquet")))
    results_conformance(NullResultSink())


def test_a_sink_without_a_name_fails_the_conformance(monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)

    class Nameless(NullResultSink):
        name = ""

    with pytest.raises(AssertionError, match="name is not a non-empty string"):
        results_conformance(Nameless())


def test_an_unregistered_sink_fails_the_conformance(monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)

    class Unregistered(NullResultSink):
        name = "not-installed"

    with pytest.raises(AssertionError, match="not registered"):
        results_conformance(Unregistered())
    results_conformance(Unregistered(), registered=False)


def test_a_sink_that_does_not_take_uri_first_fails_the_conformance(monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)

    class WrongConstructor(ResultsSink):
        name = "wrong"

        def __init__(self, out: str | None = None, options: object = None) -> None:  # noqa: ARG002
            self.out = out

        def emit(self, record: ResultRecord) -> None:
            self._known(record)

    with pytest.raises(AssertionError, match="'uri'"):
        results_conformance(WrongConstructor(), registered=False)


def test_a_sink_that_swallows_a_foreign_schema_fails_the_conformance(monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)

    class Sloppy(NullResultSink):
        name = "sloppy"

        def emit(self, record: ResultRecord) -> None:
            self.emitted += 1

    with pytest.raises(AssertionError, match="unknown schema"):
        results_conformance(Sloppy(), registered=False)


def test_a_sink_that_mutates_the_record_fails_the_conformance(monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)

    class Mutating(ResultsSink):
        name = "mutating"

        def emit(self, record: ResultRecord) -> None:
            object.__setattr__(record, "record_id", "changed")

    with pytest.raises(AssertionError, match="mutat"):
        results_conformance(Mutating(), registered=False)


def test_a_sink_that_raises_on_a_valid_record_fails_the_conformance(monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)

    class Broken(NullResultSink):
        name = "broken"

        def emit(self, record: ResultRecord) -> None:
            raise ConfigError("no")

    with pytest.raises(AssertionError, match="emit raised"):
        results_conformance(Broken(), registered=False)


def test_the_conformance_names_the_schema_it_emits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sink that only accepts the v1 id passes; the check never invents a second schema."""
    inject_sinks(monkeypatch)

    class Strict(ResultsSink):
        name = "strict"

        def emit(self, record: ResultRecord) -> None:
            self._known(record)

    results_conformance(Strict(), registered=False)
    assert RESULT_SCHEMA == "rcp-ndcg.result-record.v1"
