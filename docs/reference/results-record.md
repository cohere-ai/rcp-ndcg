# The results record: a compatibility contract

`rcp-ndcg.result-record.v1` is the public, versioned record of one evaluation row. There is one record per **system ×
dataset × metric × cutoff**. A consumer reads this record instead of the internal layout of the package. The record is
pinned like every other public surface. `schemas/result-record.v1.json` is exported from the pydantic model
(`rcp_ndcg.results.ResultRecord`), and `tests/contract/snapshots/python_api.json` pins the fields of the model. A
change to either one is a reviewed diff with a CHANGELOG entry. The how-to is
[Export evaluation results](../how-to/export-results.md).

## Versioning

* Within `v1`, fields are only ever **added**. An added field is optional or has a default. A reader of the first `v1`
  therefore keeps reading later `v1` records.
* A change to the meaning or the type of an **existing** field is a new schema id (`rcp-ndcg.result-record.v2`). It is
  never a silent reinterpretation of `v1`.
* The `schema` field names the version of the record. A sink is a result destination, and it refuses a record whose
  schema it does not know. A v2 record written as v1 would corrupt the file instead of failing.
* `record_id` is deterministic over `(subject, dataset, metrics)`. It is not a random id. The same row exported twice
  therefore has the same id. A row with a changed protocol has a different one.

## The fields

`ResultRecord` is frozen, and it refuses an unknown field. `schema` is the serialisation alias of `schema_name`:

| Field | Type | Notes |
|---|---|---|
| `schema` | `"rcp-ndcg.result-record.v1"` | the version tag of the record |
| `record_id` | `str` | `record_identity(subject, dataset, metrics)`, a SHA-256 hex digest |
| `created_at` | `datetime` | it must carry a timezone; a naive datetime is refused |
| `subject` | `ResultSubject` | the subject of the row |
| `dataset` | `ResultDataset \| null` | the dataset and its scoring convention |
| `metrics` | `list[ResultMetric]` | the metric rows; one row for each exported row |
| `artifacts` | `list[ResultArtifact]` | the inputs that the row was computed from |
| `provenance` | `dict[str, Any]` | the exporter owns this field: the code version, the config digest and the environment |

`ResultSubject` holds `kind` (`"system"`, `"run"` or `"calibration"`), `name`, `system`, `run_id` and `identity`.
`system` is the system value of the `Rankings`. `identity` is the content digest of the subject, when it is known. The
record also holds `labels`: free-form provenance labels such as `{"model": "org/model"}`. A label is never a host or a
bucket.

`ResultDataset` holds `name`, `subset` (`"default"`), `split` (`"test"`), `task`, `revision` and `protocol`.
`protocol` is the preset name, for example `"mteb"`, `"nanobeir"` or `"plain"`. It also holds `protocol_spec` and
`gains_source` (`"gains"`, `"calibration"`, `"dataset"` or `"none"`). The manifest of a run records the subset, the
split and the task that the data was read at. A report written by `eval score --out` records them in its `inputs`, so
an export states the real provenance. The defaults apply when the source declares none.

`ResultMetric` holds `metric`, `k`, `value` (or `null` when the metric is undefined for the row), `num_queries`,
`num_datasets`, `ci_low` and `ci_high`. It also holds `dataset`: the dataset of a per-dataset mean, or `null` for the
summary row.

`ResultArtifact` holds `role` (`"rankings"`, `"report"`, `"manifest"`, `"calibration"`, `"judgements"` or `"log"`),
`uri`, `sha256`, `schema_name` and `media_type`.

## Protocol and comparability

A number has the meaning that its scoring convention gives it. `ResultDataset.protocol_spec` is the full
[`Protocol`](../concepts/protocols.md). It holds the qrel gain (`linear` or `exponential`), the tie rule
(`doc_id_desc`, `group_mean` or `input_order`), `restrict_to_candidates`, `drop_identical_ids` and `round_digits`. The
ideal DCG follows from the metric and from that protocol:

* qrel-nDCG sorts the **positive qrels** of the query, after `qrel_gain` maps a grade to a gain;
* RCP-nDCG and Count-nDCG sort **all** of the query's gains;
* `restrict_to_candidates` decides which scored documents enter the ranking (the judged pool). `drop_identical_ids`
  removes the document whose id equals the query id.

A record can therefore state a convention that the package did not score itself. An importer of a run scored as linear
integer-qrel nDCG with `doc_id_desc` ties writes
`protocol_spec=Protocol(name="bright", ties="doc_id_desc", qrel_gain="linear", round_digits=5)`. Two records that
differ only in the protocol are **different records**. The protocol covers the qrel gain, the tie rule and the pool
restriction. `record_id` covers the whole protocol, and a consumer groups rows for comparison by
`(dataset.name, dataset.revision, dataset.protocol, dataset.protocol_spec)`, not by the metric label alone.

The gains source is part of comparability too. It is `"dataset"` (the released gains), `"calibration"` (a fitted
calibration), `"gains"` (passed in) or `"none"`. The same metric under a different gains table is not the same number.
The record states which table it came from.

The report that a record is built from carries the same conventions. `EvalReport` holds the per-query, per-dataset and
summary tables for every system, metric and cutoff that it scored. It holds the protocol that the report was scored
under (`report.protocol`). It also holds the gains table that it read (`report.gains_source`, one of the four values
above). For a report written by `rcp-ndcg eval score --out`, it also holds the input files (`report.inputs`).
`value(system, metric, k)` reads one summary value. `leaderboard(metric, k)` reads the wide table. The table has a row
for each system and a column for each dataset. Its summary row is `mean`, best first. All four methods refuse a
`(metric, k)` that the report never computed. `value` raises a `DataError` that names the system, the metric and the
cutoff that it asked for. `leaderboard`, `compare` and `sensitivity` raise one that names the cutoffs that the report
has. A wrong cutoff therefore never reads as an empty table or as a message about shared queries.

Comparisons are made over the queries that two systems share. For each pair of systems, `compare` gives the difference
(B minus A), a paired t-test and a query-clustered bootstrap interval. It also gives the queries where RCP-nDCG and
qrel-nDCG disagree in sign. The paper's `sensitivity` is the per-dataset share of the system pairs. A pair counts when
its paired t-test over the shared queries is significant at `alpha`. That share is averaged over the datasets. Both
methods read a report, so a comparison cannot mix scoring conventions. The protocol is the protocol of the report.

## The sink contract

A sink consumes records, and it never reads a run. The sink is registered under the `rcp_ndcg.results` entry-point
group. Its constructor takes `uri` as its first parameter after `self`. Its `emit` refuses a record whose schema it
does not know. The caller calls `flush` one time after the last record. The shared check is
`rcp_ndcg.testing.results_conformance(sink)`. The built-in sinks (`jsonl`, `parquet`, `null`) run through it in the
tests of the package. `rcp-ndcg results sinks` lists the installed sinks.
