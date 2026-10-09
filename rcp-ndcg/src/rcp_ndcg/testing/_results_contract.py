"""The results-sink conformance suite: the sink contract as one check a sink's own tests call.

Every sink -- built-in or plugin, registered through the ``rcp_ndcg.results`` entry-point group -- is held to
the same invariants: it names itself, its constructor takes ``uri`` first (what ``--out`` passes), it accepts a
valid ``rcp-ndcg.result-record.v1`` record without mutating it, ``flush`` is idempotent, and it refuses a record
whose schema it does not know (a v2 record written as v1 would corrupt the export). A plugin's test suite runs::

    from rcp_ndcg.testing import results_conformance

    def test_my_sink_passes_the_conformance(tmp_path):
        results_conformance(MySink(uri=str(tmp_path / "records.jsonl")))

and every built-in sink runs through the same call in ``tests/results/test_conformance.py`` -- one definition
of what a sink must do.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

from rcp_ndcg.results import (
    RESULT_SCHEMA,
    ResultDataset,
    ResultMetric,
    ResultRecord,
    ResultsSink,
    ResultSubject,
    record_identity,
    registered_result_sinks,
)

__all__ = ["results_conformance"]


def results_conformance(sink: ResultsSink, *, registered: bool = True) -> None:
    """The sink contract, as one check; raises :class:`AssertionError` naming every failed invariant.

    Checks, collected as a list rather than a first failure:

    * ``name`` is a non-empty string and -- with ``registered`` -- a name of the sink table (the
      ``rcp_ndcg.results`` entry-point group);
    * the constructor's first parameter after ``self`` is named ``uri`` (what the registry's callers pass by
      keyword; a sink that names it otherwise is unreachable through ``--out``);
    * ``emit`` and ``flush`` are callable;
    * ``emit`` accepts a valid record, does not mutate it, and ``flush`` is idempotent;
    * ``emit`` refuses a record whose ``schema`` is not :data:`~rcp_ndcg.results.RESULT_SCHEMA`.

    Args:
        sink: The sink instance (the class's ``name`` and constructor are read from it).
        registered: Whether ``name`` must be in the sink table; ``False`` for a sink under development, whose
            package is not installed yet.

    Raises:
        AssertionError: naming every failed check, so a third party's test suite fails with the list -- never
            somewhere inside the export.
    """
    failures: list[str] = []
    cls = type(sink)
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        failures.append(f"{cls.__name__}.name is not a non-empty string: {name!r}")
    elif registered and name not in registered_result_sinks():
        failures.append(f"{cls.__name__}.name {name!r} is not registered (a sink is a 'rcp_ndcg.results' entry point)")
    parameters = list(inspect.signature(cls.__init__).parameters)
    if len(parameters) < 2 or parameters[1] != "uri":
        first = parameters[1] if len(parameters) > 1 else "<none>"
        failures.append(f"{cls.__name__} takes {first!r} rather than 'uri' as its first parameter")
    for method in ("emit", "flush"):
        if not callable(getattr(sink, method, None)):
            failures.append(f"{cls.__name__} has no callable {method}()")
    failures.extend(_emit_invariants(sink) if not failures else [])
    if failures:
        raise AssertionError("the results conformance failed:\n  - " + "\n  - ".join(failures))


def _emit_invariants(sink: ResultsSink) -> list[str]:
    """The emit/flush invariants: a valid record in, no mutation, an unknown schema refused."""
    failures: list[str] = []
    record = _sample_record()
    before = record.model_dump()
    try:
        sink.emit(record)
    except Exception as exc:  # noqa: BLE001 - any refusal of a valid record is a contract failure
        failures.append(f"emit raised {type(exc).__name__}: {exc} on a valid {RESULT_SCHEMA} record")
        return failures
    if record.model_dump() != before:
        failures.append("emit mutated the record (a sink consumes it; it never rewrites it)")
    try:
        sink.flush()
        sink.flush()
    except Exception as exc:  # noqa: BLE001 - flush must be idempotent
        failures.append(f"flush raised {type(exc).__name__}: {exc}")
    foreign = ResultRecord.model_construct(
        record_id="foreign",
        created_at=record.created_at,
        subject=record.subject,
        dataset=record.dataset,
        metrics=record.metrics,
        schema_name="rcp-ndcg.result-record.v2",
    )
    try:
        sink.emit(foreign)
    except Exception:  # noqa: BLE001 - the refusal is the contract
        pass
    else:
        failures.append("emit accepted a record of an unknown schema (rcp-ndcg.result-record.v2)")
    return failures


def _sample_record() -> ResultRecord:
    """A small valid record (one system, one dataset, one metric) the conformance emits."""
    subject = ResultSubject(kind="system", name="conformance", system="conformance")
    dataset = ResultDataset(name="conformance")
    metric = ResultMetric(metric="rcp_ndcg", k=10, value=0.5, num_queries=1)
    return ResultRecord(
        record_id=record_identity(subject, dataset, [metric]),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        subject=subject,
        dataset=dataset,
        metrics=[metric],
    )
