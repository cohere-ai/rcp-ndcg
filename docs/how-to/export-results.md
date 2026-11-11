# Export evaluation results

An evaluation ends as numbers in a run directory or a report file. The results-export extension point turns those
numbers into a versioned record. There is one self-contained record per **system × dataset × metric × cutoff**.
The extension point hands each record to a sink. A sink is a plugin that writes records where a consumer reads
them. The record is the compatibility contract between the package and the consumer of its results ([the results
record](../reference/results-record.md)). The sink is an extension point, like the dataset readers and the job
runners.

## From a run

A run's manifest, report and artifacts are the source of the export:

```bash
rcp results export --run runs/20260102-030405-tiny-ab12cd --sink jsonl --out records.jsonl
```

Every record carries the run id, the resolved dataset revision, the scoring protocol and the artifact hashes. The
export leaves out the run's reference systems: `candidates`, the pool order, and `judge`, the judge's own
abilities. `--include-reference` adds them. `--system NAME` (repeatable) selects the systems to export.

`rcp results sinks` lists the registered sinks (`jsonl`, `parquet`, `null`).

## From a report

A report from `eval score --out` records its inputs. Thus you can export the report alone:

```bash
rcp eval score --rankings run.parquet --dataset hf://YOUR-ORG/your-dataset --out report.json
rcp results export --report report.json --sink parquet --out records.parquet
```

The export refuses a report without recorded inputs. The record names the dataset that the report was scored on.
The export never invents a dataset. You can combine `--run` and `--report`. The report file overrides the run's `metrics/report.json`, and the run supplies the identity and the artifacts.

## In Python

`records_from_report` and `records_from_run` are the builders that the command line uses:

```python
from rcp.data import Dataset, Rankings
from rcp.eval import evaluate
from rcp.results import JsonlResultSink, records_from_report

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

The `jsonl` sink writes one record per line. The `parquet` sink writes one row per metric row: a record with
several metrics becomes several rows. The `null` sink discards records. All three accept `uri` and an `options`
mapping. Always close a sink with `flush()`.

## What a record holds

A record (`rcp.result-record.v1`) is frozen. It has no update operation:

| Field | Meaning |
|---|---|
| `schema` | `rcp.result-record.v1` |
| `record_id` | a digest of the subject, the dataset and the metrics; it is deterministic, and it includes a protocol difference |
| `created_at` | when the record was built (timezone-aware) |
| `subject` | `kind` (`system`, `run` or `calibration`), `name`, the `Rankings` `system`, the `run_id`, an optional content `identity`, free-form `labels` |
| `dataset` | `name`, `subset`, `split`, `task`, `revision`, `protocol` (the preset name), `protocol_spec` (qrel gain, tie rule, pool restriction, rounding), `gains_source` |
| `metrics` | one row per record: `metric`, `k`, `value`, `num_queries`, `num_datasets`, `ci_low`, `ci_high`, `dataset` (the per-dataset mean's dataset, or `null` for the summary row) |
| `artifacts` | `role`, `uri`, `sha256`, `schema_name`, `media_type` of the files that the row was computed from |
| `provenance` | free-form exporter provenance: the code version, the run's resolved-config digest |

Compare numbers only when the conventions match. `dataset.protocol_spec` states the whole convention. `record_id`
covers that convention. Two records can differ only in the qrel gain (`linear` versus `exponential`) or in the
tie rule (`doc_id_desc` versus `group_mean`). Such records are different records. They are never the same number
under two labels.

## Write your own sink

A sink is one class in a package. The package publishes the class under the `rcp.results` entry-point group.
An entry-point group is a named list of plugin entry points in the package metadata:

```toml
[project.entry-points."rcp.results"]
mine = "my_package.results:MySink"
```

```python
from rcp.results import ResultRecord, ResultsSink


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

The constructor's first parameter after `self` is named `uri`. `--out` passes this value. `emit` refuses a record
whose `schema` is not `rcp.result-record.v1`. The package calls `flush` once after the last record. Run the
shared contract check from your own tests:

```python
from rcp.testing import results_conformance

sink = MySink()
sink.emit(record)
results_conformance(sink, registered=False)
```

`registered=False` skips the entry-point check while the package is under development. After installation,
`results_conformance(sink)` also verifies that the name resolves through the group.
