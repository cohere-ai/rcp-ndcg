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
print(dataset.subset, dataset.split, dataset.provenance.revision[:8])

scores = {query_id: {doc_id: -rank for rank, doc_id in enumerate(pool)}
          for query_id, pool in dataset.candidates.items()}
report = evaluate(Rankings.from_scores(scores, system="pool-order"), dataset=dataset)
print(report.value("pool-order", "rcp_ndcg"), report.value("pool-order", "qrel_ndcg"))
```

`load_dataset("suite:nanobeir")` loads every task of a suite, and `evaluate(rankings, suite="nanobeir")` scores
rankings against it with the paper's protocol ([scoring protocols](concepts/protocols.md)). A revision pins the
data: `load_dataset(uri, revision="<commit>")`. Your own rankings load from a Parquet, CSV, TREC run or JSONL file
with `rcp_ndcg.data.load_rankings(path)`.

A dataset records where its data came from: `subset` and `split` (the source's, `"default"`/`"test"` when it has
none), the optional mteb `task` the data realises, the `task_instruction` (one instruction for the whole task, as a
string or per side `{"query": ..., "document": ...}`), and the `provenance` (source URI, resolved commit, subset,
split, and the duplicates policy with its counts). Exports key on them (`Dataset.export_key`, the
`(task, subset, split)` triple). A document carries its `title` as its own field and its body in `text` — nothing
joins at read time; a query's per-query `instruction` stays a field of its own too. How a model's input combines a
title with its body, and the two instruction kinds with the text, is a formatting decision made where the text is
formatted, never in the data.

A rankings file has the columns `query_id`, `doc_id` and `score`, optionally `system` (the ranker) and `dataset` (the
subset a row belongs to; `subset` is read as well). The subsets of BRIGHT, ViDoRe v3 and NanoBEIR share query ids,
so rankings of a whole suite need the `dataset` column: `evaluate` refuses rows without it when the subsets they
would score share ids. A TREC run cannot carry the column. It holds one subset: score it against that subset
(`--subset <name>`, or `load_dataset(uri, subset=...)`), or name it with `load_rankings(path, dataset="<name>")` and
join the files of a suite with `Rankings.concat`.

Three places take a Hugging Face address, each in its own form:

| Where | Form | Example |
|---|---|---|
| `load_dataset`, `--dataset` | `hf://<owner>/<repo>[/<subset>][@<revision>]`: a dataset whose layout the dataset card declares (mteb's rules: `{s-}corpus`, `{s-}queries`, a `default`/`{s-}qrels` labels table, `{s-}top_ranked` pools, an `{s-}instruction` config, and rcp-ndcg's `{s-}excluded` and qrels `gain`/`theta` columns) | `hf://mteb/nfcorpus`, `hf://fabianschmidt-cohere/rcp-ndcg-nanobeir/NanoFiQA2018Retrieval` |
| `load_dataset`, `--dataset` | `mteb:<Task>[/<subset>][@<split>]`: one of the 113 tasks whose data only the task's own loader knows (ViDoRe v1's id prefixes, BRIGHT's exclusions); the `[mteb]` extra | `mteb:BrightBiologyRetrieval` |
| `load_rankings`, `--rankings`, any file path | `hf://datasets/<owner>/<repo>/<path>`: one file, through `fsspec` | `hf://datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir/provenance/runs/NanoFiQA2018Retrieval.parquet` |
| `rcp-ndcg data fetch --dataset` | a suite name, or `hf://<owner>/<repo>`: the whole repository, downloaded | `hf://fabianschmidt-cohere/rcp-ndcg-bright` |

A repository with exactly one subset loads it without being named; a repository whose card declares none (a raw
layout such as BRIGHT's) is refused with the `mteb:<Task>` hint. The reader follows mteb's own resolution — the
`query` config wins over `{s-}queries`, the `default`-then-`qrels` fallback for the labels, the requested split
when the config declares it else the config's only split — and needs neither `datasets` nor `mteb`. The same
datasets also run through stock `mteb`, with the `rcp_ndcg_tasks.py` file each dataset ships, or through
`rcp_ndcg.eval.mteb` ([the MTEB tutorial](how-to/mteb-integration.md)).

## Other formats, and adding one

The readers and writers are entry points of the `rcp_ndcg.readers` and `rcp_ndcg.writers` groups — the same seam
as the job runners. The built-ins (`beir`, `jsonl`, `hf`, `mteb`, `images`, `videos`, `frames`, `pdf`) are
declared there; a third-party format is one class in its own package:

```toml
[project.entry-points."rcp_ndcg.readers"]
my-format = "my_package.io:MyReader"
```

A reader implements `rcp_ndcg.data.io.SourceReader` (its first constructor parameter is named `uri`), may serve
the pools (`candidates`), exclusions (`excluded`) and released gains (`gains`/`thetas`) besides the queries,
corpus and qrels, narrows the `provenance` to what it knows, and must pass the shared conformance suite
(`rcp_ndcg.testing.io_conformance`) — the one definition of what a reader must do, the same one the built-ins
run through in the project's own tests. Its name is then the URI scheme of `load_dataset` (and the `--format` of
`rcp-ndcg data convert`), except `hf` (the Hub layout is `hf://`) and `pdf` (a PDF has no queries).

## Duplicates

Exact duplicates fold: the same id read again with the same content, the same `(query, document)` pair labelled
again with the same grade. A *conflicting* duplicate — the same key with different content — refuses, naming the
rows, unless the load passes `duplicates="last"` (mteb's own behaviour when a repository repeats a pair), which
takes the last row where a table can: the labels, the pools and the exclusions are materialised, so they do. A
corpus or a query table streams, and a row it has already yielded cannot be replaced, so a conflicting row there
still refuses — with the option's scope named, never a resolution the data does not carry. The fold counts are
recorded in the dataset's provenance (`DuplicateCounts`): every load reads the labels, the pools and the
exclusions, so those are in it; a corpus's and a query table's folds happen when they are read (they are read on
demand), and are logged as they happen. `Dataset.from_records` is stricter still: it refuses any duplicate,
exact or not (the in-memory path validates, it does not ingest).

## Revisions and identities

A revision such as `main` is a moving pointer, so identities record the commit it resolved to. `rcp_ndcg.data.revisions`
is the public module for that: `resolve_revision(repo_id, revision)` resolves a revision to the exact commit (the
Hub once per process, then the local cache when offline), `is_commit(revision)` says whether a revision is already
a full 40-character lowercase hex commit (so it resolves to itself), and `dataset_uri_revision(uri, revision)`
reduces a dataset URI and revision to the identity payload's `{repo, commit, verified}`. A commit the resolution
could not find is `None` with a typed `UNPINNED_REVISION` warning — never an invented value.

## Other data sources

`load_dataset` also reads a BEIR directory (`beir:<dir>`, plain or gzip-compressed: `corpus.jsonl[.gz]`,
`queries.jsonl[.gz]`, `qrels/<split>.tsv[.gz]`), JSONL files (`jsonl:<dir>`), and directories of page
images, video clips or pre-extracted frames. `rcp-ndcg data convert` ingests such a source, or PDFs rendered to page
images, into JSONL (a
directory with `corpus.jsonl`, `queries.jsonl` and `qrels.jsonl`, or with `--shape ranking` one file of queries with
<<<<<<< HEAD
their candidates), a BEIR directory, which `load_dataset` reads back (`jsonl:<dir>`, `beir:<dir>`; the BEIR
round trip keeps grades exactly (`repr`, not six significant digits) and carries a query's `instruction`
through), or the MTEB Hub layout (`--to mteb`: what mteb's `push_dataset_to_hub` writes, plus the `gain`/`theta`
qrels columns and the `-excluded` config where mteb ignores them; see
[MTEB integration](how-to/mteb-integration.md)).
A reader refuses what it would otherwise drop silently: a row without an id, a `(query, doc)` pair labelled twice,
a qrels grade that is not a finite number, a qrels split with no recognisable grade column, a corpus row whose keys
the record does not declare. The frames reader records each frame's number in its file name in
=======
their candidates) or a BEIR directory, which `load_dataset` reads back (`jsonl:<dir>`, `beir:<dir>`); the BEIR
round trip keeps grades exactly (`repr`, not six significant digits), carries a query's per-query `instruction`
through, and writes a document's `title` into the BEIR title column so it round-trips too.
A reader refuses what it would otherwise drop silently: a row without an id, a `(query, doc)` pair labelled twice
with different grades, a qrels grade that is not a finite number, a qrels split with no recognisable grade column, a
corpus row whose keys the record does not declare. The frames reader records each frame's number in its file name in
>>>>>>> origin/rfc-0001
`frame_indices`, so a clip sampled at real frame numbers says which frames of the source it showed. How page
images and video are sized for a judge is a judging setting (`preprocessing` in a run config), not part of the data.
`rcp-ndcg data inspect` summarises a dataset.
