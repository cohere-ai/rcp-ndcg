"""``rcp-ndcg results``: list the registered sinks and export records to one.

``results export`` builds one ``rcp-ndcg.result-record.v1`` record per (system, dataset, metric, cutoff) from a
run directory or a report file and hands it to the named sink. The registry's entry points are injected here
(the lane's environment predates the group; the gate rebuilds it and the packaging snapshot pins the
declaration).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from rcp_ndcg_core.protocol import Protocol

from rcp_ndcg.cli.results import results_group
from rcp_ndcg.data import Dataset, Rankings
from rcp_ndcg.eval import ReportInputs, evaluate
from rcp_ndcg.results import ResultRecord
from tests.results._runs import build_run
from tests.results._sinks import inject_sinks

QUERIES = ("q1", "q2", "q3")
GAINS = {query: {"a": 0.9, "b": 0.4, "c": 0.1} for query in QUERIES}
QRELS = {query: {"a": 1.0, "b": 1.0, "c": 0.0} for query in QUERIES}


def _invoke(*args: str) -> dict:
    result = CliRunner().invoke(results_group, [*args, "--json"])
    return {"exit_code": result.exit_code, **json.loads(result.stdout)}


@pytest.fixture(scope="module")
def report_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dataset = Dataset(name="toy", qrels=QRELS, candidates={query: ["a", "b", "c"] for query in QUERIES})
    rankings = Rankings.concat(
        [
            Rankings.from_scores({q: {"a": 3.0, "b": 2.0, "c": 1.0} for q in QUERIES}, system="good"),
            Rankings.from_scores({q: {"b": 3.0, "a": 2.0, "c": 1.0} for q in QUERIES}, system="bad"),
        ]
    )
    report = evaluate(
        rankings,
        dataset=dataset,
        gains=GAINS,
        protocol=Protocol(name="bright", ties="doc_id_desc", qrel_gain="linear", round_digits=5),
        k=5,
        bootstrap=0,
    )
    report = report.model_copy(
        update={
            "inputs": ReportInputs(
                rankings="rows.parquet",
                suite="nanobeir",
                subset="toy",
                revision="a" * 40,
                split="validation",
                task="MyTask",
            )
        }
    )
    path = tmp_path_factory.mktemp("results-report") / "report.json"
    path.write_text(report.to_json(indent=2), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_run(tmp_path_factory.mktemp("results-cli-run"))


def test_sinks_lists_the_registered_sinks(monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)

    document = _invoke("sinks")

    assert document["ok"] is True
    assert document["data"]["sinks"] == ["jsonl", "null", "parquet"]


def test_export_writes_jsonl_from_a_report(report_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)
    out = tmp_path / "records.jsonl"

    document = _invoke("export", "--report", str(report_path), "--sink", "jsonl", "--out", str(out))

    assert document["ok"] is True
    assert document["data"]["records"] > 0
    records = [ResultRecord.model_validate_json(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == document["data"]["records"]
    assert {record.subject.name for record in records} == {"good", "bad"}
    assert {record.dataset.name for record in records} == {"nanobeir"}
    assert {record.dataset.subset for record in records} == {"toy"}
    assert {record.dataset.revision for record in records} == {"a" * 40}
    assert {record.dataset.split for record in records} == {"validation"}
    assert {record.dataset.task for record in records} == {"MyTask"}
    assert {record.artifacts[0].uri for record in records} == {str(report_path)}
    assert all(record.dataset.protocol == "bright" for record in records)
    assert all(record.dataset.protocol_spec.ties == "doc_id_desc" for record in records)
    assert all(record.artifacts[0].schema_name == "rcp-ndcg.eval-report.v1" for record in records)


def test_export_defaults_to_the_jsonl_sink(report_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)
    out = tmp_path / "records.jsonl"

    document = _invoke("export", "--report", str(report_path), "--out", str(out))

    assert document["ok"] is True
    assert document["data"]["sink"] == "jsonl"
    assert out.is_file()


def test_export_writes_parquet_from_a_report(
    report_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow.parquet as pq

    inject_sinks(monkeypatch)
    out = tmp_path / "records.parquet"

    document = _invoke("export", "--report", str(report_path), "--sink", "parquet", "--out", str(out))

    assert document["ok"] is True
    table = pq.read_table(out)
    assert table.num_rows == document["data"]["records"]
    assert set(table.column_names) >= {"record_id", "subject_name", "dataset_name", "metric", "k", "value"}


def test_export_from_a_run_excludes_the_reference_systems(
    run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inject_sinks(monkeypatch)
    out = tmp_path / "records.jsonl"

    document = _invoke("export", "--run", str(run_dir), "--out", str(out))

    assert document["ok"] is True
    records = [ResultRecord.model_validate_json(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {record.subject.name for record in records} == {"mine"}
    assert {record.subject.run_id for record in records} == {run_dir.name}
    assert all(record.dataset.name == "rows" for record in records)


def test_export_from_a_run_can_include_the_reference_systems(
    run_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inject_sinks(monkeypatch)
    out = tmp_path / "records.jsonl"

    document = _invoke("export", "--run", str(run_dir), "--include-reference", "--out", str(out))

    assert document["ok"] is True
    records = [ResultRecord.model_validate_json(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {record.subject.name for record in records} == {"candidates", "judge", "mine"}


def test_export_without_a_source_is_a_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)

    document = _invoke("export", "--sink", "jsonl", "--out", "records.jsonl")

    assert document["ok"] is False
    assert document["error"]["code"] == "USAGE"
    assert document["exit_code"] == 2


def test_export_with_an_unknown_sink_names_the_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)

    document = _invoke("export", "--report", "report.json", "--sink", "nope", "--out", "records.jsonl")

    assert document["ok"] is False
    assert document["error"]["code"] == "CONFIG"
    assert "jsonl" in document["error"]["message"]


def test_export_without_an_out_for_the_jsonl_sink_is_a_config_error(
    report_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inject_sinks(monkeypatch)

    document = _invoke("export", "--report", str(report_path), "--sink", "jsonl")

    assert document["ok"] is False
    assert document["error"]["code"] == "CONFIG"
    assert "--out" in (document["error"]["hint"] or "")


def test_export_from_a_run_that_is_not_a_run_is_a_missing_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inject_sinks(monkeypatch)

    document = _invoke("export", "--run", str(tmp_path / "nope"), "--out", str(tmp_path / "records.jsonl"))

    assert document["ok"] is False
    assert document["error"]["code"] == "MISSING_INPUT"


def test_export_of_a_report_without_inputs_is_a_data_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A report with no recorded inputs cannot say which dataset it scored: refused, never a default name."""
    inject_sinks(monkeypatch)
    dataset = Dataset(name="toy", qrels=QRELS, candidates={query: ["a", "b", "c"] for query in QUERIES})
    rankings = Rankings.from_scores({q: {"a": 3.0, "b": 2.0, "c": 1.0} for q in QUERIES}, system="good")
    report = evaluate(rankings, dataset=dataset, gains=GAINS, protocol="plain", k=5, bootstrap=0)
    path = tmp_path / "report.json"
    path.write_text(report.to_json(indent=2), encoding="utf-8")

    document = _invoke("export", "--report", str(path), "--out", str(tmp_path / "records.jsonl"))

    assert document["ok"] is False
    assert document["error"]["code"] == "DATA"
    assert "--run" in (document["error"]["hint"] or "")


def test_export_uses_the_null_sink_without_an_out(report_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inject_sinks(monkeypatch)

    document = _invoke("export", "--report", str(report_path), "--sink", "null")

    assert document["ok"] is True
    assert document["data"]["records"] > 0
