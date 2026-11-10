# Scoring protocols and tie rules

The same run can score differently under different scoring rules. A reranker can tie scores. A benchmark can
remove some documents from every ranking. The ideal ranking can come from different sets of labels. The paper
fixes these rules per suite. This package states them as a `Protocol`. A `Protocol` is the record of one
suite's scoring rules. Every number it reports carries the protocol it was computed under.

## Tie rules

A reranker can give several documents the same score. Such a tie is a tied group. A rule must give the order
within a tied group. The order of the candidate pool must never give it, because that pool is often sorted by
relevance. `rcp_ndcg_core.ndcg` takes the rule explicitly (`TieRule`):

| Rule | What a tied group gets | Used by |
|---|---|---|
| `group_mean` | every document of the tied group is credited the group's mean gain: the expected nDCG over all orders of the tie | every preset and the paper's tables; mteb's `ndcg_float_at_k`; the default of `ndcg` |
| `doc_id_desc` | ties are broken by document id, descending (the trec_eval convention) | arXiv v1's NanoBEIR and BRIGHT tables |
| `input_order` | ties keep the order in which the scores are given | arXiv v1's TREC-DL tables, which passed scores in judge-pool order |

`group_mean` depends on the scores alone. `doc_id_desc` credits a tie by the arbitrary order of document ids.
`input_order` leaks relevance into tied scores when the input is sorted by relevance. Use those two rules only
to reproduce a number computed that way (`trec_eval`, arXiv v1). Use `input_order` only with an order you
constructed yourself. A cutoff can fall inside a tied group. `group_mean` then credits each rank of the group
up to the cutoff with the group's mean gain. That gain is the expected DCG@k over the orders of the tie.

## Protocols

A `Protocol` fixes the rules of one suite:

- `ties`: the tie rule;
- `qrel_gain`: `"linear"` (the grade itself, as in the paper) or `"exponential"` ($2^{\mathrm{grade}} - 1$);
- `restrict_to_candidates`: rank only the query's judged pool, and drop other scored documents;
- `drop_identical_ids`: also remove a document whose id equals the query id (BEIR's `ignore_identical_ids`). Use it
  for data without a list of excluded documents;
- `round_digits`: round qrel-nDCG to this many decimals (BEIR rounds to 5). RCP-nDCG is never rounded.

`rcp_ndcg_core.PROTOCOLS` holds the presets:

| Preset | Ties | Candidates | qrel-nDCG rounding |
|---|---|---|---|
| `nanobeir` | `group_mean` | judged pool only | 5 decimals |
| `bright` | `group_mean` | every scored document | 5 decimals |
| `vidore` | `group_mean` | judged pool only | none |
| `trecdl` | `group_mean` | judged pool only | none |
| `mteb` | `group_mean` | judged pool only | none |
| `plain` | `group_mean` | every scored document | none |

BRIGHT is not restricted to the judged pool. Its TheoremQA reranker runs hold candidates outside the pool.
Those candidates are scored with gain 0.

To score as arXiv v1 did, copy a preset and give it its old tie rule. For example, use
`PROTOCOLS["nanobeir"].model_copy(update={"ties": "doc_id_desc"})`. Do the same for BRIGHT. For TREC-DL, use
`"input_order"` and the judged pool in pool order.

## Excluded documents and ideal rankings

Some benchmarks remove documents from every ranking. On NanoArguAna, the pool of a query can hold the query's
own argument. BEIR's evaluator drops that argument. BRIGHT lists `excluded_ids` per query. The released
datasets carry these ids in their `excluded` split. Excluded documents leave the ranking and every ideal
ranking.

The ideal DCG of RCP-nDCG and Count-nDCG sorts all gains of the query's pool. The ideal DCG of qrel-nDCG sorts
all positive qrels of the query. That includes positives outside the scored candidates, as trec_eval does.

## Queries without a positive grade

qrel-nDCG is undefined for a query whose grades are all zero, because its ideal DCG is 0. This package follows
the paper's scorer. `score_query` returns NaN for such a query, and `evaluate` reports `None`. Both leave the
query out of the mean, with a `NO_POSITIVE_QRELS` warning in the report. trec_eval instead scores such a query
0 and counts it in the mean. The paper's queries all have a positive grade. The two conventions give the same
paper numbers.

A system that did not rank a labelled query at all scores 0 for that query. The report warns with
`UNRANKED_QUERIES`. For a suite, the warning names the subsets it counts. A rankings file can match the
scored dataset not at all. Such a file is refused, with a `DataError` (exit 12 on the command line). The
refusal has two causes. The first cause is that no row names any of its subsets. The second cause is that not
one ranked document id is in the dataset's pools or labels. The file's `dataset` column must hold the exact
subset name, e.g. `hr__english`, not `hr`. The second cause has one example: ranked `486` against pool
`corpus-test-486`. Every score would be 0, and that reads as a weak system where the input is broken. A
system whose rows match some subsets, or some documents, keeps scoring. The missing subsets score 0 with the
warning. Out-of-pool documents score 0 silently. The checks are per system. A file of several systems is
therefore refused when any one of them matches nothing. Score the healthy ones with `--system NAME`
(repeatable), or `systems=` in Python. Then fix or drop the broken one.

## Aggregation

A leaderboard number is the mean over queries per dataset. It is then the unweighted mean over the datasets of
the suite (`rcp_ndcg_core.aggregate`). TREC-DL reports each year separately.

## In code

```python
from rcp_ndcg_core import PROTOCOLS, score_query

scores = {"q7": 0.95, "d1": 0.80, "d2": 0.80, "d3": 0.10}  # the system ranked the query's own argument first
gains = {"d1": 0.9, "d2": 0.2, "d3": 0.5}
pool = ["d3", "q7", "d1", "d2"]  # the judged pool, in pool order

arxiv_v1 = PROTOCOLS["nanobeir"].model_copy(update={"ties": "doc_id_desc"})
print(PROTOCOLS["nanobeir"])
print(score_query(scores, gains, protocol="nanobeir", k=10, candidates=pool, excluded=["q7"]))
print(score_query(scores, gains, protocol=arxiv_v1, k=10, candidates=pool, excluded=["q7"]))
```

The two calls differ only in the tie between `d1` and `d2`. The preset credits both documents with their mean
gain. arXiv v1's rule ranks `d2` first by its id. `rcp_ndcg.eval.evaluate` applies the protocol of a suite or
dataset automatically. Its `protocol=` argument overrides it. On the command line, the equivalent is
`rcp-ndcg eval score --suite nanobeir --protocol <name>`.

## The MTEB integration

The proposed MTEB integration (embeddings-benchmark/mteb#5516) reports `ndcg_float_at_k` with group-mean ties
on every suite. That is the rule of every preset here, so the two agree on tied scores too. See [the MTEB tutorial](../how-to/mteb-integration.md).
