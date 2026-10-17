# Export evaluation results

An evaluation ends as numbers inside a run directory or a report file. The results-export seam turns those
numbers into a versioned, self-contained record per **system × dataset × metric × cutoff** and hands them to a
sink that writes them where a consumer reads them. The record is the compatibility contract between the
package and whoever consumes its results ([the results record](../reference/results-record.md)); the sink is a
plugin seam, exactly like the dataset readers and the job runners.

## From a run

A run's manifest, report and artifacts are the export's source:

```bash
rcp-ndcg results export --run runs/20260102-030405-tiny-ab12cd --sink jsonl --out records.jsonl
```

Every record carries the run id, the resolved dataset revision, the scoring protocol and the artifact hashes.
The run's reference systems (`candidates`, the pool order, and `judge`, the judge's own abilities) are left
out; `--include-reference` adds them, and `--system NAME` (repeatable) picks the systems to export.

`rcp-ndcg results sinks` lists the registered sinks (`jsonl`, `parquet`, `null`).

## From a report

A report written by `eval score --out` records its inputs, so it can be exported on its own:

```bash
rcp-ndcg eval score --rankings run.parquet --dataset hf://YOUR-ORG/your-dataset --out report.json
rcp-ndcg results export --report report.json --sink parquet --out records.parquet
```

A report without recorded inputs is refused: the record names the dataset it was scored on, and the export
never invents one. `--run` and `--report` may be combined: the report file overrides the run's
`metrics/report.json`, and the run supplies the identity and the artifacts.

## In Python

`records_from_report` and `records_from_run` are the builders the command line uses:

```python
from rcp_ndcg.data import Dataset, Rankings
from rcp_ndcg.eval import evaluate
from rcp_ndcg.results import JsonlResultSink, records_from_report

dataset = Dataset(
    name="toy",
    qrels={"q1": {"d1": 1.0, "d2": 0.0}},
    candidates={"q1": ["d1", "d2"]},
)
rankings = Rankings.from_scores({"q1": {"d1": 0.9, "d2": 0.1}}, system="mine")
report = evaluate(rankings, dataset=dataset, metrics=["qrel_ndcg"], k=[10])

sink = JsonlResultSink(uri="records.jsonl")
for record in records_from_report(report):
    sink.emit(record)
sink.flush()
```

The `jsonl` sink writes one record per line; the `parquet` sink writes one row per metric row (a record with
several metrics becomes several rows); `null` discards. All three accept `uri` and an `options` mapping, and a
sink is always closed with `flush()`.

## What a record holds

A record (`rcp-ndcg.result-record.v1`) is frozen and one-way:

| Field | Meaning |
|---|---|
| `schema` | `rcp-ndcg.result-record.v1` |
| `record_id` | a digest of the subject, the dataset and the metrics: deterministic, and a protocol difference is part of it |
| `created_at` | when the record was built (timezone-aware) |
| `subject` | `kind` (`system`, `run` or `calibration`), `name`, the `Rankings` `system`, the `run_id`, an optional content `identity`, free-form `labels` |
| `dataset` | `name`, `subset`, `split`, `task`, `revision`, `protocol` (the preset name), `protocol_spec` (qrel gain, tie rule, pool restriction, rounding), `gains_source` |
| `metrics` | one row per record: `metric`, `k`, `value`, `num_queries`, `num_datasets`, `ci_low`, `ci_high`, `dataset` (the per-dataset mean's dataset, or `null` for the summary row) |
| `artifacts` | `role`, `uri`, `sha256`, `schema_name`, `media_type` of the files the row was computed from |
| `provenance` | free-form exporter provenance: the code version, the run's resolved-config digest |

Numbers compare only when the conventions match. `dataset.protocol_spec` states the whole convention, and
`record_id` covers it: two records that differ only in the qrel gain (`linear` versus `exponential`) or the
tie rule (`doc_id_desc` versus `group_mean`) are different records, never the same number under two labels.

## Write your own sink

A sink is one class in a package that publishes it under the `rcp_ndcg.results` entry-point group:

```toml
[project.entry-points."rcp_ndcg.results"]
mine = "my_package.results:MySink"
```

```python
from rcp_ndcg.results import ResultRecord, ResultsSink


class MySink(ResultsSink):
    """Keep the records in memory (a test double, or a bridge to a service)."""

    name = "mine"

    def __init__(self, uri: str | None = None, options: dict | None = None) -> None:
        super().__init__(uri, options)
        self.records: list[ResultRecord] = []

    def emit(self, record: ResultRecord) -> None:
        self._known(record)  # refuses a record of a schema this sink does not know
        self.records.append(record)
```

The constructor's first parameter after `self` is named `uri` (what `--out` passes), `emit` refuses a record
whose `schema` is not `rcp-ndcg.result-record.v1`, and `flush` is called once after the last record. Run the
shared contract check from your own tests:

```python
from rcp_ndcg.testing import results_conformance

sink = MySink()
sink.emit(record)
results_conformance(sink, registered=False)
```

`registered=False` skips the entry-point check while the package is under development; once it is installed,
`results_conformance(sink)` also verifies that the name resolves through the group.
