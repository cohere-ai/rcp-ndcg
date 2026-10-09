"""The ``rcp-ndcg.result-record.v1`` record: its shape, its identity and its protocol field.

The record is a public contract (the compatibility page), so its field names, defaults and the digest that
identifies it are pinned here. The protocol must be part of the identity: two records scored under different
conventions (qrel gain, tie rule) are different records even when every number matches.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError
from rcp_ndcg_core.protocol import Protocol

from rcp_ndcg.results import (
    RESULT_SCHEMA,
    ResultArtifact,
    ResultDataset,
    ResultMetric,
    ResultRecord,
    ResultSubject,
    record_identity,
)

CREATED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def subject(**fields: Any) -> ResultSubject:
    return ResultSubject(**{"kind": "system", "name": "mine", "system": "mine", **fields})


def metric(**fields: Any) -> ResultMetric:
    return ResultMetric(**{"metric": "qrel_ndcg", "k": 10, "value": 0.5, "num_queries": 4, **fields})


def record(**fields: Any) -> ResultRecord:
    subject_value = fields.pop("subject", None) or subject()
    dataset = fields.pop("dataset", None)
    if dataset is None:
        dataset = ResultDataset(
            name="NanoArguAnaRetrieval",
            subset="NanoArguAnaRetrieval",
            revision="abc123",
            protocol="bright",
            protocol_spec=Protocol(name="bright", ties="doc_id_desc", round_digits=5),
            gains_source="dataset",
        )
    metrics = fields.pop("metrics", None)
    if metrics is None:
        metrics = [metric()]
    created_at = fields.pop("created_at", CREATED)
    return ResultRecord(
        record_id=record_identity(subject_value, dataset, metrics),
        created_at=created_at,
        subject=subject_value,
        dataset=dataset,
        metrics=metrics,
        **fields,
    )


def test_the_schema_id_is_the_declared_constant() -> None:
    assert RESULT_SCHEMA == "rcp-ndcg.result-record.v1"
    assert record().schema_name == RESULT_SCHEMA


def test_a_record_serialises_its_schema_field_and_round_trips() -> None:
    original = record(
        artifacts=[ResultArtifact(role="rankings", uri="candidates.parquet", sha256="a" * 64)],
        provenance={"code": {"package_version": "0.0.1"}},
    )

    payload = json.loads(original.model_dump_json())
    assert payload["schema"] == RESULT_SCHEMA
    assert "schema_name" not in payload
    assert ResultRecord.model_validate(payload) == original


def test_the_record_id_is_deterministic() -> None:
    assert record().record_id == record().record_id
    assert record().record_id == record_identity(subject(), record().dataset, [metric()])
    assert len(record().record_id) == 64


def test_two_records_differing_only_in_protocol_are_different_records() -> None:
    """The brief's case: linear integer-qrel nDCG with ``doc_id_desc`` ties versus group-mean ties."""
    linear = record(
        dataset=ResultDataset(
            name="NanoArguAnaRetrieval",
            protocol="bright",
            protocol_spec=Protocol(name="bright", ties="doc_id_desc", qrel_gain="linear", round_digits=5),
            gains_source="dataset",
        )
    )
    group_mean = record(
        dataset=ResultDataset(
            name="NanoArguAnaRetrieval",
            protocol="bright",
            protocol_spec=Protocol(name="bright", ties="group_mean", qrel_gain="linear", round_digits=5),
            gains_source="dataset",
        )
    )

    assert linear != group_mean
    assert linear.record_id != group_mean.record_id


def test_the_qrel_gain_is_part_of_the_identity() -> None:
    linear = record(
        dataset=ResultDataset(
            name="d", protocol="plain", protocol_spec=Protocol(name="plain", ties="group_mean", qrel_gain="linear")
        )
    )
    exponential = record(
        dataset=ResultDataset(
            name="d",
            protocol="plain",
            protocol_spec=Protocol(name="plain", ties="group_mean", qrel_gain="exponential"),
        )
    )

    assert linear.record_id != exponential.record_id


def test_the_dataset_revision_is_part_of_the_identity() -> None:
    first = record(dataset=ResultDataset(name="d", revision="a" * 40))
    second = record(dataset=ResultDataset(name="d", revision="b" * 40))

    assert first.record_id != second.record_id


def test_the_value_is_part_of_the_identity() -> None:
    assert record(metrics=[metric(value=0.5)]).record_id != record(metrics=[metric(value=0.4)]).record_id


def test_a_naive_created_at_is_refused() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        record(created_at=datetime(2026, 1, 2, 3, 4, 5))


def test_the_record_is_frozen() -> None:
    value = record()
    with pytest.raises(ValidationError):
        value.subject = subject(name="other")  # type: ignore[misc]


def test_an_unknown_field_is_refused() -> None:
    with pytest.raises(ValidationError):
        ResultSubject(kind="system", name="mine", extra="x")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ResultDataset(name="d", nope=1)  # type: ignore[call-arg]


def test_a_subject_kind_is_closed() -> None:
    with pytest.raises(ValidationError):
        ResultSubject(kind="model", name="mine")  # type: ignore[arg-type]


def test_an_artifact_role_is_closed() -> None:
    with pytest.raises(ValidationError):
        ResultArtifact(role="scores", uri="x")  # type: ignore[arg-type]


def test_a_record_with_no_metrics_is_allowed() -> None:
    value = record(metrics=[])
    assert value.metrics == []
    assert value.record_id == record_identity(value.subject, value.dataset, [])


def test_a_non_finite_metric_value_is_refused() -> None:
    """A NaN cannot be hashed into a record id and is not valid JSON: refused at the model, never written."""
    with pytest.raises(ValidationError):
        metric(value=float("nan"))
    with pytest.raises(ValidationError):
        metric(ci_low=float("inf"))
    with pytest.raises(ValidationError):
        metric(ci_high=float("-inf"))
