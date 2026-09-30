# MTEB integration

The released datasets run with stock [mteb](https://github.com/embeddings-benchmark/mteb). Each task reports
mteb's usual metrics on the integer qrels plus `ndcg_float_at_k`: nDCG over the released continuous RCP gains, with
group-mean ties. The main score is `ndcg_float_at_10`. An integration of these tasks into mteb itself is proposed
in embeddings-benchmark/mteb#5516.

There are two ways to run the tasks:

- **Stock mteb.** Each dataset on the Hugging Face Hub ships a `rcp_ndcg_tasks.py` that defines its tasks for mteb
  ≥ 2.0.1 (≥ 2.10.5 for ViDoRe v3). The dataset cards show how to use it.
- **This package.** `rcp_ndcg.eval.mteb.get_tasks(suite)` returns the same tasks (the `mteb` extra). When the
  installed mteb already ships a task at the same data revision, `get_tasks` returns mteb's own.

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
