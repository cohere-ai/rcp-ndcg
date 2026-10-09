# MTEB integration

The released datasets run with stock [mteb](https://github.com/embeddings-benchmark/mteb). Each task reports
mteb's usual metrics on the integer qrels plus `ndcg_float_at_k`: nDCG over the released continuous RCP gains, with
group-mean ties. The main score is `ndcg_float_at_10`. An integration of these tasks into mteb itself is proposed
in embeddings-benchmark/mteb#5516.

There are two ways to run the tasks:

- **Stock mteb.** Each dataset on the Hugging Face Hub ships a `rcp_ndcg_tasks.py` that defines its tasks for mteb
  ≥ 2.0.1 (≥ 2.10.5 for ViDoRe v3). The dataset cards show how to use it.
- **This package.** `rcp_ndcg.eval.mteb.get_tasks(suite)` returns the same tasks (the `mteb` extra). When the
  installed mteb already ships a task at the same data revision, `get_tasks` returns mteb's own (the reranking
  view; the retrieval view is always built locally, as mteb ships only the reranking names). Each subset is
  accepted under its own name and under its published task name.

<!-- snippet: network -->
```python
import mteb
from rcp_ndcg.eval.mteb import get_tasks

tasks = get_tasks("nanobeir", ["NanoFiQA2018Retrieval"])  # or get_tasks("nanobeir") for all 13 tasks
model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
results = mteb.evaluate(model, tasks=tasks)
```

The default view reranks each query's judged pool (`{subset}-top_ranked`). `get_tasks(suite, mode="retrieval")`
searches the full corpus instead and reports only the integer-qrels metrics, because the RCP gains cover the judged
pools only.

mteb credits tied scores with their group's mean gain on every suite. The paper's NanoBEIR, BRIGHT and TREC-DL
tables break ties by document id or pool order instead. On untied scores the two conventions agree; with ties, use
`rcp_ndcg.eval.evaluate` with the suite's protocol to match the paper ([scoring protocols](../concepts/protocols.md)).

## Exporting a dataset to the MTEB layout

```python
from rcp_ndcg.data import Dataset
from rcp_ndcg.data.io import get_writer

dataset = Dataset.from_records(
    name="NanoArguAnaRetrieval",
    queries=[{"query_id": "q1", "text": "what refutes this claim"}],
    corpus=[{"doc_id": "d1", "text": "a counterargument"}],
    qrels=[{"query_id": "q1", "doc_id": "d1", "grade": 1}],
    candidates={"q1": ["d1"]},
)
written = get_writer("mteb").write_dataset(dataset, "out/nanobeir", subset="NanoArguAnaRetrieval")
```

The writer (`rcp_ndcg.data.io.mteb.MtebWriter`) writes exactly what `push_dataset_to_hub` writes, with
rcp-ndcg's extras only where mteb ignores them:

| Config | Columns |
|---|---|
| `{s-}corpus` | `id`, `title`, `text` |
| `{s-}queries` | `id`, `text`, `instruction` only when a query carries one |
| `{s-}qrels` | `query-id`, `corpus-id`, `score` (int64), plus `gain`/`theta` when the data carries them |
| `{s-}top_ranked` | `query-id`, `corpus-ids`; written when the data has a pool |
| `{s-}excluded` | `query-id`, `excluded-corpus-ids`; mteb reads no such config |

Each config is a parquet file at `{config}/{split}-00000-of-00001.parquet`, and the `README.md` carries the
`configs:` front matter that `load_dataset` -- and through it mteb's `RetrievalDatasetLoader` -- reads the
directory with. Pass `card=` (a mteb `TaskMetadata` or its fields) to render the card from mteb's own template.

A grade that is not a whole number is refused: the `score` column is written as int64 and mteb's loader casts
it to int32 at load, where a fractional value fails -- refusing here is what loading one does, at write time
and with the pair named. Export integer grades
and keep the continuous signal in `gain`/`theta`, which mteb ignores. Exclusions travel in the `-excluded`
config and are also folded out of `top_ranked` (out of the corpus, when the data has no pool), because
`top_ranked` is the pool mteb does read: a model scored inside mteb never sees an excluded document.

The layout round-trips through mteb: what mteb's `RetrievalDatasetLoader` reads from the written directory is
the dataset we hold -- the qrels as integers, the queries cut to the qrels-bearing ones, the corpus, and the
pool.

<!-- snippet: skip (needs the Hub and a push) -->
```python
# the published datasets re-laid in this exact layout, validated by mteb's own loader:
#   python tools/republish_mteb.py --out /tmp/republish
# the owner pushes each written repository from there (hf upload <owner>/<repo> <out>/<repo> . --repo-type dataset)
```

## Scoring a stored run inside mteb

Every kind of output (embedder, late interaction, reranker, LLM judge) produces `Rankings`, so one route covers
them all: serve the stored rankings to `mteb.evaluate` as a `SearchProtocol`, and mteb scores them against its
tasks, writes its own `{Task}_predictions.json`, and records genuine `TaskResult` files in its `ResultCache`
layout -- ready for `ResultCache.submit_results` (the results PR) and for the leaderboard, which additionally
needs the model's `ModelMeta` merged into mteb itself.

<!-- snippet: network -->
```python
import mteb
from rcp_ndcg.data import Rankings, load_rankings
from rcp_ndcg.eval.mteb import get_tasks, model_meta, stored_rankings_model

# the run to score, as the pipeline writes it (system, dataset, query_id, doc_id, score):
Rankings.from_scores({"q1": {"d1": 0.9, "d2": 0.4}}, dataset="NanoFiQA2018Retrieval").save("run.parquet")
rankings = load_rankings("run.parquet")  # one system's scores; `system=` names one of several
meta = model_meta("org/model", revision="abc123")   # declare what you know; the rest is unknown (None)
model = stored_rankings_model(rankings, meta)
results = mteb.evaluate(
    model,
    get_tasks("nanobeir", ["NanoFiQA2018Retrieval"]),
    encode_kwargs={},
    prediction_folder="preds/",          # mteb writes {Task}_predictions.json here
    cache=mteb.cache.ResultCache("results/"),
)
```

The served scores are the queries mteb asks about, and only those (mteb raises on a result for a query that
has no qrels). A task with a `top_ranked` pool (the reranking view) receives the pool's documents only, and
every query keeps at most `top_k` documents, ties by document id descending -- mteb's own tie rule, and the
cap order of `Rankings.top`. A query the run did not rank scores 0, the same semantics our evaluator reports
an unranked labelled query with. The integer `ndcg_at_10` equals `rcp_ndcg.eval.evaluate`'s `qrel_ndcg` under
the suite's protocol (the tie rules agree; the two differ only in how they round -- mteb rounds its mean to 5
decimals, the protocol rounds per query). One deliberate divergence: a query whose qrels are all zero scores 0
in mteb's mean, while qrel-nDCG is undefined without a positive grade and drops out of our means -- every query
of the shipped suites carries a positive label, so the two agree there.

Writing the predictions file directly, without running mteb, is `Rankings.save(format="mteb")`:

```python
from rcp_ndcg.data import Dataset, Rankings, load_rankings

rankings = Rankings.from_scores({"q1": {"d1": 2.0, "d2": 1.0}})
rankings.save("run.parquet")  # a stored run: system, dataset, query_id, doc_id, score
rankings = load_rankings("run.parquet")
dataset = Dataset.from_records(
    name="NanoFiQA2018Retrieval",
    queries=[{"query_id": "q1", "text": "how long do tortoises live"}],
    corpus=[{"doc_id": "d1", "text": "over a century"}],
    qrels=[{"query_id": "q1", "doc_id": "d1", "grade": 1}],
)
rankings.save(
    "preds/",
    format="mteb",
    task="NanoFiQA2018RCPReranking",
    qrels=dataset.qrels,
    model_name="org/model",
    model_revision="abc123",
)
# -> preds/NanoFiQA2018RCPReranking_predictions.json, mteb's own predictions layout:
#    {"mteb_model_meta": {...}, subset: {split: {qid: {did: score}}}}
```

Every query with a non-empty qrels dict must be ranked (a missing one is refused, naming it); a ranked query
without qrels is dropped -- declared policy, because mteb raises on a result for a query that has no qrels --
and every query keeps at most 1,000 documents, the cap mteb itself applies. An existing `{Task}_predictions.json`
is merged the way mteb's own writer merges: the (subset, split) written replaces theirs, the file's other
splits, subsets and its `mteb_model_meta` stay.
