# Retrieval and reranking

The first stage builds each query's candidate pool and the second re-scores it. A run's `retrieve` and `rerank`
steps do both before the judging (`candidates: {from: retrieval, retrieval: ..., rerank: ..., depth: 50}`);
outside a run, the retrieval commands write a rankings file that every scoring protocol reads directly.

## What retrieval contributes

`rcp-ndcg retrieval index` (`rcp_ndcg.retrieval.index`) builds an index of a dataset's corpus;
`rcp-ndcg retrieval search` (`rcp_ndcg.retrieve`, which indexes what an index of the same identity does not
already cover) searches it into a `Rankings`; `rcp-ndcg retrieval rerank` (`rcp_ndcg.rerank`) re-scores each
query's top `depth` candidates as another system; `rcp-ndcg retrieval fuse` (`rcp_ndcg.fuse`) merges several
rankings by reciprocal rank fusion. The commands take their config as a YAML file (`--retriever`,
`--reranker`); the exact flags are in `rcp-ndcg schema show commands --json`.

## Retriever kinds

A retriever config declares one of three kinds:

| `kind` | What it computes | Fields |
|---|---|---|
| `bm25` | sparse term scores | `stemmer`: a Snowball language of PyStemmer (`english`, `german`, ...) or `null`; corpus and queries are stemmed alike, and the stemmer is part of the index identity |
| `dense` | one vector per query and document | `encoder`: an embedding endpoint ([embedding endpoints](embeddings.md)) |
| `late_interaction` | per-token vectors, scored with MaxSim | `encoder`: a pooling endpoint ([late interaction](late-interaction.md)) |

Its second stage, `rcp-ndcg retrieval rerank` and a run's `rerank` step, takes a reranker config instead: a
cross-encoder's scores over `POST {base_url}/rerank`, a served engine or a hosted API. The wire and the client
are documented as [`rcp_ndcg.inference`](../api/inference.md).

## Indexes, checkpoints and identity

Every config declares IDENTITY_ROLES: what the model computes (the `kind`, the `stemmer`, the encoder's content
fields -- `api`, model, revision, prompts, budgets, `normalize`, `dimensions`) is content and enters the step
and the index identity; where and how fast it is asked (`base_url`, `batch_size`, `concurrency`, the timeouts)
is runtime and never does. `rcp_ndcg.retrieve` reuses an index of the same identity and rebuilds one that
differs. The rerank checkpoint keys on every content field of the reranker and on the digest of the exact
candidate texts it scored: changed texts or changed settings re-score, never silently reuse.

## On the wire

All three roles send through one inference layer ([the inference layer](inference.md), with its role clients for
the [embed](embeddings.md), [pooling](late-interaction.md) and [rerank](../api/inference.md) wires). Every request
is fitted through one text-budget mechanism ([text budgets for served roles](text-budgets.md)): over-budget
content is cut client-side at token boundaries, the template's anchors are reserved and re-attached around the
cut, and every cut is recorded -- never an engine-side truncation.
