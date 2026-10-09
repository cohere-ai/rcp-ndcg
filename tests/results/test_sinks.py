"""The built-in result sinks and the ``rcp_ndcg.results`` registry.

A sink consumes one :class:`~rcp_ndcg.results.ResultRecord` at a time: the ``jsonl`` sink writes one record per
line, the ``parquet`` sink one row per (system, dataset, metric, cutoff), and ``null`` discards (for a dry run
and for the conformance suite). The registry resolves a name through the entry-point group, and every failure
names the installed sinks.
"""

from __future__ import annotations

import json
import tomllib
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest
from pydantic import ValidationError
from rcp_ndcg_core.protocol import Protocol

from rcp_ndcg.errors import ConfigError
from rcp_ndcg.results import (
    RESULT_SCHEMA,
    JsonlResultSink,
    NullResultSink,
    ParquetResultSink,
    ResultDataset,
    ResultMetric,
    ResultRecord,
    ResultSubject,
    record_identity,
    registered_result_sinks,
    result_sink_class,
)
from tests.contract.surface import REPO
from tests.results._sinks import DECLARED_SINKS, inject_sinks


def _record(metrics: list[ResultMetric] | None = None) -> ResultRecord:
    from datetime import UTC, datetime

    subject = ResultSubject(kind="system", name="mine", system="mine", run_id="20260101-000000-tiny-ab12cd")
    dataset = ResultDataset(name="toy", protocol="plain", protocol_spec=Protocol(name="plain", ties="group_mean"))
    if metrics is None:
        metrics = [ResultMetric(metric="rcp_ndcg", k=10, value=0.75, num_queries=2)]
    return ResultRecord(
        record_id=record_identity(subject, dataset, metrics),
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        subject=subject,
        dataset=dataset,
        metrics=metrics,
    )


def test_the_manifest_declares_the_builtin_sinks() -> None:
    data = tomllib.loads((REPO / "rcp-ndcg" / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["entry-points"]["rcp_ndcg.results"] == DECLARED_SINKS


def test_the_registry_lists_the_installed_sinks_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)

    assert registered_result_sinks() == ("jsonl", "null", "parquet")


def test_the_registry_loads_a_sink_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)

    assert result_sink_class("jsonl") is JsonlResultSink
    assert result_sink_class("parquet") is ParquetResultSink
    assert result_sink_class("null") is NullResultSink


def test_an_unknown_sink_is_a_config_error_naming_the_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)

    with pytest.raises(ConfigError, match="unknown result sink 'nope'") as caught:
        result_sink_class("nope")
    assert "jsonl" in str(caught.value)
    assert "rcp_ndcg.results" in (caught.value.hint or "")


def test_a_sink_that_fails_to_import_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch, EntryPoint(name="broken", value="rcp_ndcg.results:NoSuchSink", group="rcp_ndcg.results"))

    with pytest.raises(ConfigError, match="'broken' entry point .* could not be loaded"):
        result_sink_class("broken")


def test_an_entry_point_that_is_not_a_sink_is_a_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch, EntryPoint(name="wrong", value="rcp_ndcg.results:ResultRecord", group="rcp_ndcg.results"))

    with pytest.raises(ConfigError, match="not a ResultsSink"):
        result_sink_class("wrong")


def test_the_jsonl_sink_writes_one_record_per_line(tmp_path: Path) -> None:
    out = tmp_path / "records.jsonl"
    sink = JsonlResultSink(uri=str(out))

    first, second = _record(), _record(metrics=[ResultMetric(metric="qrel_ndcg", k=5, value=0.25, num_queries=2)])
    sink.emit(first)
    sink.emit(second)
    sink.flush()

    lines = out.read_text(encoding="utf-8").splitlines()
    assert [ResultRecord.model_validate_json(line) for line in lines] == [first, second]
    assert json.loads(lines[0])["schema"] == RESULT_SCHEMA


def test_the_jsonl_sink_creates_the_parent_directory(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "records.jsonl"
    sink = JsonlResultSink(uri=str(out))
    sink.emit(_record())
    sink.flush()

    assert out.is_file()


def test_the_jsonl_sink_needs_a_uri() -> None:
    with pytest.raises(ConfigError, match="jsonl"):
        JsonlResultSink(uri=None)


def test_the_jsonl_sink_replaces_the_file_by_default(tmp_path: Path) -> None:
    out = tmp_path / "records.jsonl"
    for _ in range(2):
        sink = JsonlResultSink(uri=str(out))
        sink.emit(_record())
        sink.flush()

    assert len(out.read_text(encoding="utf-8").splitlines()) == 1


def test_the_jsonl_sink_appends_with_the_option(tmp_path: Path) -> None:
    out = tmp_path / "records.jsonl"
    for _ in range(2):
        sink = JsonlResultSink(uri=str(out), options={"append": "1"})
        sink.emit(_record())
        sink.flush()

    assert len(out.read_text(encoding="utf-8").splitlines()) == 2


def test_the_parquet_sink_serialises_a_datetime_provenance(tmp_path: Path) -> None:
    """The JSONL sink serialises a datetime in provenance; the parquet sink must not crash on it."""
    import pyarrow.parquet as pq

    out = tmp_path / "records.parquet"
    sink = ParquetResultSink(uri=str(out))
    sink.emit(_record().model_copy(update={"provenance": {"when": _record().created_at}}))
    sink.flush()

    row = pq.read_table(out).to_pylist()[0]
    assert json.loads(row["provenance"])["when"].startswith("2026-01-02")


def test_the_parquet_sink_writes_one_row_per_metric(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    out = tmp_path / "records.parquet"
    sink = ParquetResultSink(uri=str(out))
    sink.emit(
        _record(
            metrics=[
                ResultMetric(metric="rcp_ndcg", k=10, value=0.75, num_queries=2),
                ResultMetric(metric="qrel_ndcg", k=5, value=0.25, num_queries=2, dataset="toy"),
            ]
        )
    )
    sink.flush()

    table = pq.read_table(out)
    assert table.num_rows == 2
    rows = table.to_pylist()
    assert [row["metric"] for row in rows] == ["rcp_ndcg", "qrel_ndcg"]
    assert [row["k"] for row in rows] == [10, 5]
    assert [row["dataset_name"] for row in rows] == ["toy", "toy"]
    assert [row["subject_name"] for row in rows] == ["mine", "mine"]
    spec = json.loads(rows[0]["dataset_protocol_spec"])
    assert spec["name"] == "plain"
    assert spec["ties"] == "group_mean"
    assert spec["qrel_gain"] == "linear"
    assert spec["round_digits"] is None


def test_the_parquet_sink_writes_a_record_without_metrics_as_one_row(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    out = tmp_path / "records.parquet"
    sink = ParquetResultSink(uri=str(out))
    sink.emit(_record(metrics=[]))
    sink.flush()

    table = pq.read_table(out)
    assert table.num_rows == 1
    assert table.to_pylist()[0]["metric"] is None


def test_the_parquet_sink_needs_a_uri() -> None:
    with pytest.raises(ConfigError, match="parquet"):
        ParquetResultSink(uri=None)


def test_the_null_sink_counts_and_discards() -> None:
    sink = NullResultSink()
    sink.emit(_record())
    sink.emit(_record())
    sink.flush()

    assert sink.emitted == 2


def test_a_sink_refuses_a_record_of_another_schema(tmp_path: Path) -> None:
    foreign = ResultRecord.model_construct(
        record_id="x",
        created_at=_record().created_at,
        subject=_record().subject,
        schema_name="rcp-ndcg.result-record.v2",
    )
    for sink in (JsonlResultSink(uri=str(tmp_path / "a.jsonl")), ParquetResultSink(uri=str(tmp_path / "a.parquet"))):
        with pytest.raises(ConfigError, match="schema"):
            sink.emit(foreign)


def test_a_sink_refuses_a_value_that_is_not_a_record() -> None:
    sink = NullResultSink()
    with pytest.raises(ConfigError, match="ResultRecord"):
        sink.emit({"schema": RESULT_SCHEMA})  # type: ignore[arg-type]


def test_a_metric_needs_a_non_negative_cutoff() -> None:
    with pytest.raises(ValidationError):
        ResultMetric(metric="rcp_ndcg", k=0, value=0.5)
