# The results record: a compatibility contract

`rcp-ndcg.result-record.v1` is the public, versioned record of one evaluation row: one record per **system ×
dataset × metric × cutoff**. It is what a consumer reads instead of the package's internal layout, and it is
pinned like every other public surface: `schemas/result-record.v1.json` is exported from the pydantic model
(`rcp_ndcg.results.ResultRecord`), `tests/contract/snapshots/python_api.json` pins the model's fields, and a
change to either is a reviewed diff with a CHANGELOG entry. The how-to is
[Export evaluation results](../how-to/export-results.md).

## Versioning

* Within `v1`, fields are only ever **added**; an added field is optional or has a default, so a reader of the
  first `v1` keeps reading later `v1` records.
* A change to an **existing** field's meaning or type is a new schema id (`rcp-ndcg.result-record.v2`), never a
  silent reinterpretation of `v1`.
* The `schema` field names the record's version. A sink refuses a record whose schema it does not know:
  writing a v2 record as v1 would corrupt the file rather than fail.
* `record_id` is deterministic over `(subject, dataset, metrics)`; it is not a random id, so the same row
  exported twice has the same id, and a row whose protocol changed has a different one.

## The fields

`ResultRecord` (frozen; unknown fields are refused; `schema` is the serialisation alias of `schema_name`):

| Field | Type | Notes |
|---|---|---|
| `schema` | `"rcp-ndcg.result-record.v1"` | the version tag |
| `record_id` | `str` | `record_identity(subject, dataset, metrics)`, a SHA-256 hex digest |
| `created_at` | `datetime` | timezone-aware; a naive datetime is refused |
| `subject` | `ResultSubject` | what the row is about |
| `dataset` | `ResultDataset \| null` | the dataset and the scoring convention |
| `metrics` | `list[ResultMetric]` | the metric rows (one per exported row) |
| `artifacts` | `list[ResultArtifact]` | what the row was computed from |
| `provenance` | `dict[str, Any]` | exporter-owned: code version, config digest, environment |

`ResultSubject`: `kind` (`"system"`, `"run"` or `"calibration"`), `name`, `system` (the `Rankings` system
value), `run_id`, `identity` (the subject's content digest, when known), `labels` (free-form provenance labels
such as `{"model": "org/model"}`; never a host or a bucket).

`ResultDataset`: `name`, `subset` (`"default"`), `split` (`"test"`), `task`, `revision`, `protocol` (the preset
name, e.g. `"mteb"`, `"nanobeir"`, `"plain"`), `protocol_spec`, `gains_source` (`"gains"`, `"calibration"`,
`"dataset"` or `"none"`). A run's manifest records the subset, split and task the data was read at, and a report
written by `eval score --out` records them in its `inputs`, so an export states the real provenance; the defaults
apply when the source declares none.

`ResultMetric`: `metric`, `k`, `value` (or `null` when the metric is undefined for the row), `num_queries`,
`num_datasets`, `ci_low`, `ci_high`, `dataset` (the dataset of a per-dataset mean; `null` for the summary row).

`ResultArtifact`: `role` (`"rankings"`, `"report"`, `"manifest"`, `"calibration"`, `"judgements"` or `"log"`),
`uri`, `sha256`, `schema_name`, `media_type`.

## Protocol and comparability

A number only means what its scoring convention says it means. `ResultDataset.protocol_spec` is the full
[`Protocol`](../concepts/protocols.md): the qrel gain (`linear` or `exponential`), the tie rule (`doc_id_desc`,
`group_mean` or `input_order`), `restrict_to_candidates`, `drop_identical_ids` and `round_digits`. The ideal
DCG follows from the metric and that protocol:

* qrel-nDCG sorts the query's **positive qrels** (after `qrel_gain` maps a grade to a gain);
* RCP-nDCG and Count-nDCG sort **all** of the query's gains;
* `restrict_to_candidates` decides which scored documents enter the ranking (the judged pool) and
  `drop_identical_ids` removes the document whose id equals the query id.

So a record can state a convention the package itself did not score: an importer of a run scored as linear
integer-qrel nDCG with `doc_id_desc` ties writes
`protocol_spec=Protocol(name="bright", ties="doc_id_desc", qrel_gain="linear", round_digits=5)`. Two records
that differ only in the protocol — the qrel gain, the tie rule, the pool restriction — are **different
records**: `record_id` covers the whole protocol, and a consumer groups rows for comparison by `(dataset.name,
dataset.revision, dataset.protocol, dataset.protocol_spec)` rather than by the metric label alone.

The gains source is part of comparability too: `"dataset"` (the released gains), `"calibration"` (a fitted
calibration), `"gains"` (passed in) or `"none"`. The same metric under a different gains table is not the same
number, and the record states which table it came from.

## The sink contract

A sink consumes records; it never reads a run. It is registered under the `rcp_ndcg.results` entry-point group,
its constructor's first parameter after `self` is `uri`, its `emit` refuses a record whose schema it does not
know, and `flush` is called once after the last record. The shared check is
`rcp_ndcg.testing.results_conformance(sink)`; the built-in sinks (`jsonl`, `parquet`, `null`) run through it in
the package's own tests, and `rcp-ndcg results sinks` lists what is installed.
