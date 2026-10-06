# Data

All data behind the paper is public on the Hugging Face Hub:

| Dataset | Contents |
|---|---|
| [rcp-ndcg-nanobeir](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir) | 13 NanoBEIR tasks with calibrated RCP gains; runs with stock `mteb` ≥ 2.0.1 (`ndcg_float_at_10`) |
| [rcp-ndcg-bright](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright) | 12 BRIGHT tasks with calibrated RCP gains; stock `mteb` ≥ 2.0.1 |
| [rcp-ndcg-vidore-v3](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-vidore-v3) | 8 ViDoRe v3 visual-document tasks (6 languages) with calibrated RCP gains; stock `mteb` ≥ 2.10.5 |
| [rcp-ndcg-trecdl](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-trecdl) | TREC-DL 2019 and 2020 with continuous RCP gains next to the NIST judgements; stock `mteb` ≥ 2.0.1 |
| [rcp-ndcg-external-validation](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-external-validation) | The human contest study (46 annotators, 311 contests, 7,080 grades) and the external LLM judges (GLM-5.3-flash, DeepSeek-4.1-flash, Kimi-K3) |

`experiments/fetch_data.py` downloads all five at the pinned revisions used for the paper's tables. Each dataset
card states its license; the TREC-DL dataset builds on MS MARCO, whose terms apply.

## What a benchmark dataset holds

Each of the four benchmark datasets has one subset per task, with these splits:

| Split | Columns | Meaning |
|---|---|---|
| `corpus` | document id, text (or page image) | the documents |
| `queries` | query id, text | the queries |
| `qrels` | query id, document id, `score`, `gain`, `theta` | the human grade (`score`), and for every judged pool document its calibrated ability (`theta`) and RCP gain (`gain`) |
| `top_ranked` | query id, the pool's document ids | the judged candidate pool of each query, in pool order |
| `excluded` | query id, document ids | documents the benchmark removes from every ranking (NanoArguAna's own argument of a query, BRIGHT's `excluded_ids`) |

Under `provenance/` each dataset also holds how the pools were built, the stored runs of the 14 rerankers, the
judge's Stage B answers per window and its item parameters. The external-validation dataset holds the human study
(contests, documents, grades, reviews and the annotation guidelines) and the blind comparisons with the external
LLM judges, with every prompt and answer.

## Loading a dataset

`rcp_ndcg.data.load_dataset` reads the released layout from a `hf://` URI and knows the protocol of each suite. It
needs the `hf` extra (`huggingface-hub`). The example below scores a toy system, which ranks each pool in its pool
order, on one NanoBEIR task:

<!-- snippet: network -->
```python
from rcp_ndcg.data import Rankings, load_dataset
from rcp_ndcg.eval import evaluate

dataset = load_dataset("hf://fabianschmidt-cohere/rcp-ndcg-nanobeir/NanoFiQA2018Retrieval")
print(dataset.name, dataset.protocol, len(dataset.candidates), "queries")

scores = {query_id: {doc_id: -rank for rank, doc_id in enumerate(pool)}
          for query_id, pool in dataset.candidates.items()}
report = evaluate(Rankings.from_scores(scores, system="pool-order"), dataset=dataset)
print(report.value("pool-order", "rcp_ndcg"), report.value("pool-order", "qrel_ndcg"))
```

`load_dataset("suite:nanobeir")` loads every task of a suite, and `evaluate(rankings, suite="nanobeir")` scores
rankings against it with the paper's protocol ([scoring protocols](concepts/protocols.md)). A revision pins the
data: `load_dataset(uri, revision="<commit>")`. Your own rankings load from a Parquet, CSV, TREC run or JSONL file
with `rcp_ndcg.data.load_rankings(path)`.

A rankings file has the columns `query_id`, `doc_id` and `score`, optionally `system` (the ranker) and `dataset` (the
subset a row belongs to; `subset` is read as well). The subsets of BRIGHT, ViDoRe v3 and NanoBEIR share query ids,
so rankings of a whole suite need the `dataset` column: `evaluate` refuses rows without it when the subsets they
would score share ids. A TREC run cannot carry the column. It holds one subset: score it against that subset
(`--subset <name>`, or `load_dataset(uri, subset=...)`), or name it with `load_rankings(path, dataset="<name>")` and
join the files of a suite with `Rankings.concat`.

Three places take a Hugging Face address, each in its own form:

| Where | Form | Example |
|---|---|---|
| `load_dataset`, `--dataset` | `hf://<owner>/<repo>[/<subset>][@<revision>]`: a dataset in the released layout | `hf://fabianschmidt-cohere/rcp-ndcg-nanobeir/NanoFiQA2018Retrieval` |
| `load_rankings`, `--rankings`, any file path | `hf://datasets/<owner>/<repo>/<path>`: one file, through `fsspec` | `hf://datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir/provenance/runs/NanoFiQA2018Retrieval.parquet` |
| `rcp-ndcg data fetch --dataset` | a suite name, or `hf://<owner>/<repo>`: the whole repository, downloaded | `hf://fabianschmidt-cohere/rcp-ndcg-bright` |

The same datasets also run through stock `mteb`, with the `rcp_ndcg_tasks.py` file each dataset ships, or through
`rcp_ndcg.eval.mteb` ([the MTEB tutorial](tutorials/mteb-integration.md)).

## Revisions and identities

A revision such as `main` is a moving pointer, so identities record the commit it resolved to. `rcp_ndcg.data.revisions`
is the public module for that: `resolve_revision(repo_id, revision)` resolves a revision to the exact commit (the
Hub once per process, then the local cache when offline), `is_commit(revision)` says whether a revision is already
a full 40-character lowercase hex commit (so it resolves to itself), and `dataset_uri_revision(uri, revision)`
reduces a dataset URI and revision to the identity payload's `{repo, commit, verified}`. A commit the resolution
could not find is `None` with a typed `UNPINNED_REVISION` warning — never an invented value.

## Other data sources

`load_dataset` also reads a BEIR directory (`beir:<dir>`), JSONL files (`jsonl:<dir>`), and directories of page
images, video clips or pre-extracted frames. `rcp-ndcg data convert` ingests such a source, or PDFs rendered to page
images, into JSONL (a
directory with `corpus.jsonl`, `queries.jsonl` and `qrels.jsonl`, or with `--shape ranking` one file of queries with
their candidates) or a BEIR directory, which `load_dataset` reads back (`jsonl:<dir>`, `beir:<dir>`); the BEIR
round trip keeps grades exactly (`repr`, not six significant digits) and carries a query's `instruction` through.
A reader refuses what it would otherwise drop silently: a row without an id, a `(query, doc)` pair labelled twice,
a qrels grade that is not a finite number, a qrels split with no recognisable grade column, a corpus row whose keys
the record does not declare. The frames reader records each frame's number in its file name in
`frame_indices`, so a clip sampled at real frame numbers says which frames of the source it showed. How page
images and video are sized for a judge is a judging setting (`preprocessing` in a run config), not part of the data.
`rcp-ndcg data inspect` summarises a dataset.
