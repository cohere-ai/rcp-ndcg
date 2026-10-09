"""Building records from rcp-ndcg's own outputs: an ``EvalReport`` and a run directory.

One record per (system, dataset, metric, cutoff) row. From a report, the subject is derived per system and the
dataset's protocol is the report's. From a run, the subject carries the run id, the dataset carries the
resolved revision and the artifacts name what the row was computed from.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from rcp_ndcg_core.protocol import Protocol

from rcp_ndcg.data import Dataset, Rankings
from rcp_ndcg.errors import ConfigError, MissingInputError
from rcp_ndcg.eval import EvalReport, evaluate
from rcp_ndcg.results import (
    ResultDataset,
    ResultSubject,
    records_from_report,
    records_from_run,
)
from tests.results._runs import build_run

CREATED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
QUERIES = ("q1", "q2", "q3")
GAINS = {query: {"a": 0.9, "b": 0.4, "c": 0.1} for query in QUERIES}
QRELS = {query: {"a": 1.0, "b": 1.0, "c": 0.0} for query in QUERIES}


def _report(protocol: str | Protocol = "plain") -> EvalReport:
    dataset = Dataset(name="toy", qrels=QRELS, candidates={query: ["a", "b", "c"] for query in QUERIES})
    rankings = Rankings.concat(
        [
            Rankings.from_scores({q: {"a": 3.0, "b": 2.0, "c": 1.0} for q in QUERIES}, system="good"),
            Rankings.from_scores({q: {"b": 3.0, "a": 2.0, "c": 1.0} for q in QUERIES}, system="bad"),
        ]
    )
    return evaluate(rankings, dataset=dataset, gains=GAINS, protocol=protocol, k=5, bootstrap=0)


@pytest.fixture(scope="module")
def report() -> EvalReport:
    return _report()


def test_one_record_per_row(report: EvalReport) -> None:
    records = records_from_report(report, created_at=CREATED)

    assert len(records) == len(report.per_dataset) + len(report.summary)
    assert all(len(record.metrics) == 1 for record in records)
    assert all(record.created_at == CREATED for record in records)


def test_the_per_dataset_rows_name_their_dataset_and_the_summary_does_not(report: EvalReport) -> None:
    records = records_from_report(report, created_at=CREATED)
    per_dataset = [record for record in records if record.metrics[0].dataset is not None]
    summary = [record for record in records if record.metrics[0].dataset is None]

    assert per_dataset and summary
    assert {record.metrics[0].dataset for record in per_dataset} == {"toy"}
    assert {record.dataset.name for record in per_dataset} == {"toy"}
    assert {record.dataset.name for record in summary} == {"toy"}
    assert all(record.dataset.subset == "default" for record in records)


def test_the_subject_is_derived_per_system(report: EvalReport) -> None:
    records = records_from_report(report, created_at=CREATED)

    assert {(r.subject.kind, r.subject.name, r.subject.system) for r in records} == {
        ("system", "good", "good"),
        ("system", "bad", "bad"),
    }


def test_the_report_protocol_is_carried_in_full(report: EvalReport) -> None:
    records = records_from_report(report, created_at=CREATED)

    for record in records:
        assert record.dataset.protocol == "plain"
        assert record.dataset.protocol_spec == Protocol(name="plain", ties="group_mean")
        assert record.dataset.gains_source == "gains"


def test_two_reports_differing_only_in_protocol_never_compare_equal() -> None:
    linear = records_from_report(
        _report(Protocol(name="bright", ties="doc_id_desc", qrel_gain="linear", round_digits=5)), created_at=CREATED
    )
    exponential = records_from_report(
        _report(Protocol(name="bright", ties="doc_id_desc", qrel_gain="exponential", round_digits=5)),
        created_at=CREATED,
    )

    assert linear and exponential
    assert all(left != right for left, right in zip(linear, exponential, strict=True))
    assert all(left.record_id != right.record_id for left, right in zip(linear, exponential, strict=True))


def test_a_given_dataset_is_used_unchanged(report: EvalReport) -> None:
    given = ResultDataset(name="suite", subset="sub", revision="c" * 40)
    records = records_from_report(report, dataset=given, created_at=CREATED)

    assert {record.dataset.name for record in records} == {"suite"}
    assert {record.dataset.subset for record in records} == {"sub"}
    assert {record.dataset.revision for record in records} == {"c" * 40}


def test_a_single_subject_for_several_systems_is_refused(report: EvalReport) -> None:
    subject = ResultSubject(kind="run", name="my-run")
    with pytest.raises(ConfigError, match="one subject"):
        records_from_report(report, subject=subject, created_at=CREATED)


def test_a_callable_subject_is_called_per_system(report: EvalReport) -> None:
    records = records_from_report(
        report,
        subject=lambda system: ResultSubject(kind="system", name=f"run-{system}", system=system),
        created_at=CREATED,
    )

    assert {record.subject.name for record in records} == {"run-good", "run-bad"}


def test_artifacts_and_provenance_travel_unchanged(report: EvalReport) -> None:
    from rcp_ndcg.results import ResultArtifact

    artifact = ResultArtifact(role="report", uri="report.json", schema_name="rcp-ndcg.eval-report.v1")
    records = records_from_report(report, artifacts=[artifact], provenance={"code": "x"}, created_at=CREATED)

    assert all(record.artifacts == [artifact] for record in records)
    assert all(record.provenance == {"code": "x"} for record in records)


def test_an_empty_report_yields_no_records() -> None:
    dataset = Dataset(name="toy", qrels={"q1": {"a": 1.0}}, candidates={"q1": ["a"]})
    rankings = Rankings.from_scores({"q1": {"a": 1.0}}, system="only")
    report = evaluate(rankings, dataset=dataset, metrics=("qrel_ndcg",), k=5, bootstrap=0)

    assert records_from_report(report, created_at=CREATED) != []
    empty = report.model_copy(update={"per_dataset": [], "summary": []})
    assert records_from_report(empty, created_at=CREATED) == []


# ---------------------------------------------------------------------------
# records_from_run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_run(tmp_path_factory.mktemp("results-run"))


def test_records_from_run_excludes_the_reference_systems_by_default(run_dir: Path) -> None:
    records = records_from_run(run_dir)

    assert {record.subject.name for record in records} == {"mine"}
    assert all(record.subject.kind == "system" for record in records)
    assert all(record.subject.system == record.subject.name for record in records)
    assert all(record.subject.run_id == run_dir.name for record in records)
    assert all(record.dataset.name == "rows" for record in records)
    assert all(record.dataset.protocol == "plain" for record in records)
    assert all(record.metrics for record in records)


def test_records_from_run_can_include_the_reference_systems(run_dir: Path) -> None:
    records = records_from_run(run_dir, include_reference=True)

    assert {record.subject.name for record in records} == {"candidates", "judge", "mine"}


def test_records_from_run_carries_the_run_provenance_and_artifacts(run_dir: Path) -> None:
    records = records_from_run(run_dir)

    for record in records:
        assert len(record.provenance["run_config_sha256"]) == 64
        assert record.provenance["code"]["package_version"]
        roles = {artifact.role for artifact in record.artifacts}
        assert {"manifest", "report"} <= roles
        assert all(not artifact.uri.startswith("/") for artifact in record.artifacts)
        assert any(artifact.uri == "manifest.json" for artifact in record.artifacts)
        assert record.subject.identity is not None
        assert len(record.subject.identity) == 64


def test_records_from_run_refuses_a_report_with_only_reference_systems(tmp_path: Path) -> None:
    """A run with no systems of its own exports nothing by default: the error names the way out."""
    run_dir = build_run(tmp_path / "run", systems={})

    with pytest.raises(ConfigError, match="reference systems"):
        records_from_run(run_dir)
    assert {record.subject.name for record in records_from_run(run_dir, include_reference=True)} == {
        "candidates",
        "judge",
    }


def test_records_from_run_names_an_unknown_system(run_dir: Path) -> None:
    with pytest.raises(ConfigError, match="nope"):
        records_from_run(run_dir, systems=["nope"])


def test_records_from_run_reads_an_explicit_report_file(run_dir: Path, tmp_path: Path) -> None:
    report = EvalReport.model_validate_json((run_dir / "metrics" / "report.json").read_text(encoding="utf-8"))
    other = tmp_path / "report.json"
    other.write_text(report.to_json(indent=2), encoding="utf-8")

    assert records_from_run(run_dir, report=other, created_at=CREATED) == records_from_run(run_dir, created_at=CREATED)


def test_records_from_run_needs_a_report_or_a_finished_evaluate_step(tmp_path: Path) -> None:
    with pytest.raises(MissingInputError):
        records_from_run(tmp_path / "nope")


def test_records_from_run_uses_the_resolved_dataset_revision(tmp_path: Path) -> None:
    """A run over a local ``jsonl:`` source resolves no Hub commit: the revision stays unknown, never invented."""
    records = records_from_run(build_run(tmp_path / "run", systems={}), include_reference=True)

    assert records and all(record.dataset.revision is None for record in records)
