# `rcp_ndcg.eval`

- `evaluate(rankings, *, suite=None, dataset=None, gains=None, protocol=None, k=10, metrics=("rcp_ndcg", "qrel_ndcg"),
  count_gains=None, systems=None, bootstrap=1000, seed=0)` returns an `EvalReport`: RCP-nDCG, qrel-nDCG and
  Count-nDCG of `rcp_ndcg.data.Rankings` under a scoring protocol, per query, per dataset and as a summary with a
  query-clustered bootstrap interval. The gains come from `gains` (a `{query_id: {doc_id: gain}}` mapping or a
  `Calibration`), else from the dataset's released `gain` column; `count_gains` (required for `"count_ndcg"`)
  follows the same keying rule. For a suite, `gains` and `count_gains` keys may be `"<subset>/<query_id>"` -- and
  follows the same keying rule. For a suite, `gains` and `count_gains` keys may be `"<subset>/<query_id>"` -- and
  must be when subsets share query ids, and one style only for each subset (a mix is refused, as bare ids over
  shared ids are, with the subsets named; a prefixed key for one subset and bare ids for another are each read
  where they belong; a key naming no subset of the suite is refused, so a typo'd prefix cannot silently drop a
  subset from the aggregate). Integer qrels are never used as RCP gains: without gains, RCP-nDCG raises
  `DataError`. The protocol defaults to the suite's or the dataset's, else `plain`.
  `systems` scores only the named systems (`--system` on the command line, repeatable): one system of a
  multi-system file whose rankings match nothing of the dataset is refused (every score would be 0), and this
  scores the others; an unknown name raises `ConfigError` listing the systems the file names.
- `compare(report, *, baseline=None, metric="rcp_ndcg", k=None, alpha=0.05, bootstrap=10000, seed=0,
  systems=None)` returns a `Comparison`: per pair of systems the difference (B minus A), the paired t-test, a
  query-clustered bootstrap interval, and the queries where RCP-nDCG and qrel-nDCG disagree in sign. `systems`
  restricts the comparison to some systems of the report.
- `sensitivity(report, *, metric="rcp_ndcg", k=None, alpha=0.05)` is the paper's sensitivity: per dataset, the
  share of all system pairs whose paired t-test over their shared queries has p < `alpha`, then the mean over the
  datasets.
- `explain(report, query_id, *, calibration=None, k=10, dataset=None)` returns a `QueryExplanation`: each system's
  top k with gains, grades, abilities and per-criterion pass probabilities (the criteria's item parameters once,
  in `items`), and the gap between every system and the first at cutoff `k`, split into selection (which documents
  reach the top k) and ordering (how they are arranged). The gaps come from the query's RCP gains, or from its
  qrel grades when the query has none, and are empty when the query has no labels, no positive grade, or every
  labelled document of it is excluded.
- `explain.score_delta(order_a, order_b, gains, *, k=10)` splits one nDCG@k gap between two orders (B minus A)
  the same way, returning `(total, selection, ordering)` -- the primitive the `deltas` are built on.
- `bootstrap_interval(datasets, *, resamples, seed, alpha=0.05)` is the summary interval's primitive: the
  percentile interval of the dataset-mean-then-mean aggregate over `alpha/2` and `1 - alpha/2` quantiles of the
  draws, resampling queries within each dataset (query-clustered, stratified by dataset, fixed seed);
  `(None, None)` without resamples or values.
- `rcp_ndcg.eval.mteb.get_tasks(suite, names=None, *, mode="reranking", revision=None)` returns the public suites
  as mteb tasks with `ndcg_float_at_k` ([MTEB integration](../tutorials/mteb-integration.md)); `names` lists
  subsets (each once; empty is refused, `None` is all of them).

`EvalReport` and `Comparison` have `.to_json()` and `.to_pandas()`, and `EvalReport.value(system, metric, k)` gives
one summary value. `EvalReport.leaderboard(metric="rcp_ndcg", k=None)` is the wide table: a row per system, a
column per dataset and the summary as `mean`, best first. A `(metric, k)` the report never computed is refused
with a `DataError` naming the cutoffs it has -- by `value`, `leaderboard`, `compare` and `sensitivity` alike, so
a wrong k never reads as an empty table or a message about shared queries. A report records the protocol it was
computed under (`report.protocol`) and where its gains came from (`report.gains_source`); a report written by
`rcp-ndcg eval score --out` also records its input files (`report.inputs`).

**Inputs in memory.** `Rankings.from_records(records)` and `Dataset.from_records(name=..., queries=..., corpus=...,
qrels=..., candidates=..., excluded=...)` take plain records (dicts, or the row models `RankingRow`, `QueryRow`,
`DocumentRow`, `QrelRow` of `rcp_ndcg.data`) and validate them strictly: an unknown key, a missing field, a score
that is not a finite number, a duplicate or an id that does not join is a `DataError` naming the record. pandas is
an output format only: a frame goes in as `frame.to_dict("records")` and results come out with `to_pandas()`.

```python
import pandas as pd

from rcp_ndcg.data import Dataset, Rankings
from rcp_ndcg.eval import compare, evaluate

dataset = Dataset.from_records(
    name="toy",
    qrels=[
        {"query_id": "q1", "doc_id": "d1", "grade": 2, "gain": 0.9},
        {"query_id": "q1", "doc_id": "d2", "grade": 1, "gain": 0.6},
        {"query_id": "q1", "doc_id": "d4", "grade": 0, "gain": 0.1},
        {"query_id": "q2", "doc_id": "d3", "grade": 1, "gain": 0.8},
        {"query_id": "q2", "doc_id": "d5", "grade": 0, "gain": 0.3},
    ],
)
frame = pd.DataFrame(
    {
        "system": ["A"] * 5 + ["B"] * 5,
        "query_id": ["q1", "q1", "q1", "q2", "q2"] * 2,
        "doc_id": ["d1", "d2", "d4", "d3", "d5", "d4", "d2", "d1", "d5", "d3"],
        "score": [3.0, 2.0, 1.0, 2.0, 1.0] * 2,
    }
)
rankings = Rankings.from_records(frame.to_dict("records"))
report = evaluate(rankings, dataset=dataset, bootstrap=0)
print(report.leaderboard())
print(compare(report, baseline="A", bootstrap=200).to_pandas())
```

## Conventions of the Python objects

- **Properties and methods.** An accessor without arguments that reads what the object holds is a property
  (`Dataset.queries`, `Dataset.corpus`, `Dataset.parts`, `Rankings.systems`, `Rankings.datasets`,
  `Calibration.datasets`, `EvalReport.systems`); one that takes arguments or builds a new table is a method
  (`Rankings.queries(system=, dataset=)`, `Calibration.gains(dataset=)`, `Calibration.theta_map(...)`,
  `EvalReport.leaderboard(...)`, every `to_pandas()`).
- **Query keys.** `Calibration.gains()` and `Calibration.theta_map()` key queries by their own id when `dataset=`
  names one dataset or the calibration holds one, and by `"<dataset>||<query_id>"` when it holds several;
  `Calibration.queries` always uses the `"<dataset>||<query_id>"` form.
- **Printing.** `Dataset`, `Rankings`, `Calibration`, `EvalReport` and `Comparison` print a one-line summary; the
  full data is in `model_dump()` (the pydantic models) or `to_pandas()`.
- **pandas** is an output format: `to_pandas()` on `Rankings`, `EvalReport` (`per_query`, `per_dataset`,
  `summary`, and `leaderboard()`), `Comparison` and `Calibration` (`thetas` with a `gain` column, `queries`,
  `items`). Inputs are records (`from_records`).
