"""``rcp-ndcg results``: export evaluation results as versioned records to a sink.

* ``sinks`` -- list the registered sinks (the ``rcp_ndcg.results`` entry-point group).
* ``export`` -- build one ``rcp-ndcg.result-record.v1`` record per (system, dataset, metric, cutoff) from a run
  directory (``--run``) or a report file (``--report``) and hand them to ``--sink`` (``--out`` is the sink's
  URI). A run's reference systems (``candidates``, ``judge``) are left out unless ``--include-reference`` (or
  named with ``--system``). A report file must record its inputs (``eval score --out`` does), because the
  record names the dataset it was scored on.

The record is the public compatibility contract (:mod:`rcp_ndcg.results`); a sink is a plugin seam, exactly
like the dataset readers and job runners.
"""

from __future__ import annotations

from pathlib import Path

import click
from pydantic import BaseModel, Field

from rcp_ndcg.cli.command import command
from rcp_ndcg.errors import DataError, MissingInputError, UsageError
from rcp_ndcg.results import (
    ResultArtifact,
    ResultDataset,
    ResultRecord,
    records_from_report,
    records_from_run,
    registered_result_sinks,
    result_sink_class,
)


class ResultsSinksRequest(BaseModel):
    """No inputs."""


class ResultsSinks(BaseModel):
    """The registered result sinks."""

    sinks: list[str] = Field(description="The sink names, sorted (the rcp_ndcg.results entry-point group).")


@command("results sinks", request=ResultsSinksRequest, result=ResultsSinks)
def results_sinks(request: ResultsSinksRequest) -> ResultsSinks:
    """List the registered result sinks."""
    return ResultsSinks(sinks=list(registered_result_sinks()))


class ResultsExportRequest(BaseModel):
    run: str | None = Field(
        default=None, description="A run directory whose report and artifacts are exported (`metrics/report.json`)."
    )
    report: str | None = Field(
        default=None,
        description="A report file from `eval score --out`. With --run it overrides the run's report file; alone "
        "it must record its inputs (which dataset it scored).",
    )
    sink: str = Field(default="jsonl", description="The sink to write to; see `rcp-ndcg results sinks`.")
    out: str | None = Field(default=None, description="The sink's URI, e.g. records.jsonl or records.parquet.")
    system: list[str] = Field(
        default_factory=list,
        description="Export only these systems (repeatable; default every system, minus a run's reference "
        "systems `candidates` and `judge`).",
    )
    include_reference: bool = Field(
        default=False, description="With --run, also export the run's reference systems `candidates` and `judge`."
    )


class ResultsExport(BaseModel):
    """What ``results export`` wrote."""

    sink: str
    out: str | None = None
    records: int = Field(description="The number of records emitted (one per system, dataset, metric and cutoff).")


def _report_records(path: str, systems: list[str]) -> list[ResultRecord]:
    """Records from a report file: the dataset comes from the report's recorded inputs, never a default."""
    from rcp_ndcg.eval import EvalReport

    report_path = Path(path)
    if not report_path.is_file():
        raise MissingInputError(f"no report at {report_path}", hint="write one with `rcp-ndcg eval score --out`")
    report = EvalReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    if report.inputs is None:
        raise DataError(
            "the report records no inputs: it does not say which dataset it scored",
            hint="export a run directory with --run, or write the report with `rcp-ndcg eval score --out`",
        )
    # The per-dataset rows name the dataset (a suite's rows name its subsets); the suite/dataset URI is the
    # fallback for a summary over several datasets, where no single row name describes the whole.
    names = list(dict.fromkeys(row.dataset for row in report.per_dataset))
    name = names[0] if len(names) == 1 else (report.inputs.suite or report.inputs.dataset)
    if not name:
        raise DataError(
            "the report records neither a suite nor a dataset",
            hint="write the report with `rcp-ndcg eval score --out` (--suite or --dataset)",
        )
    dataset = ResultDataset(
        name=name,
        subset=report.inputs.subset or "default",
        split=report.inputs.split or "test",
        task=report.inputs.task,
        revision=report.inputs.revision,
    )
    from rcp_ndcg.storage.artifacts import artifact_ref

    artifacts = [
        ResultArtifact(
            role="report",
            uri=path,
            sha256=artifact_ref(report_path).sha256,
            schema_name="rcp-ndcg.eval-report.v1",
        )
    ]
    return records_from_report(report, dataset=dataset, systems=systems or None, artifacts=artifacts)


@command(
    "results export",
    request=ResultsExportRequest,
    result=ResultsExport,
    read_only=False,
)
def results_export(request: ResultsExportRequest) -> ResultsExport:
    """Write one record per (system, dataset, metric, cutoff) to a sink.

    Records come from a run directory (--run; its manifest, report and artifacts) or from a report file
    (--report; its recorded inputs name the dataset). --out is the sink's URI.
    """
    if request.run is None and request.report is None:
        raise UsageError(
            "pass --run DIR or --report FILE",
            hint="export a run directory with --run, or a report from `eval score --out` with --report",
        )
    sink_class = result_sink_class(request.sink)
    if request.run is None:
        assert request.report is not None
        records = _report_records(request.report, request.system)
    else:
        records = records_from_run(
            request.run,
            report=request.report,
            systems=request.system or None,
            include_reference=request.include_reference,
        )
    sink = sink_class(uri=request.out)
    for record in records:
        sink.emit(record)
    sink.flush()
    return ResultsExport(sink=request.sink, out=request.out, records=len(records))


@click.group(name="results", help="Export evaluation results as versioned records to a sink.")
def results_group() -> None:
    """``rcp-ndcg results``."""


for _command in (results_sinks, results_export):
    results_group.add_command(_command)


__all__ = ["ResultsExport", "ResultsExportRequest", "ResultsSinks", "ResultsSinksRequest", "results_group"]
