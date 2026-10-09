# Retrieval and reranking

The first stage builds each query's candidate pool and the second re-scores it. A run's `retrieve` and `rerank`
steps do both before the judging (`candidates: {from: retrieval, retrieval: ..., rerank: ..., depth: 50}`);
outside a run, the retrieval commands write a rankings file that every scoring protocol reads directly.

## What retrieval contributes

`rcp-ndcg retrieval index` (`rcp_ndcg.retrieval.index`) builds an index of a dataset's corpus;
`rcp-ndcg retrieval search` (`rcp_ndcg.retrieve`, which reuses an index whose identity matches and rebuilds one
that differs) searches it into a `Rankings`; `rcp-ndcg retrieval rerank` (`rcp_ndcg.rerank`) re-scores each
query's top `depth` candidates as another system; `rcp-ndcg retrieval fuse` (`rcp_ndcg.fuse`) merges several
rankings by reciprocal rank fusion. The commands take their config as a YAML file (`--retriever`,
`--reranker`); the exact flags are in `rcp-ndcg schema show commands --json`.

For a Matryoshka model, `rcp-ndcg retrieval store` (`rcp_ndcg.retrieval.build_store`) encodes the corpus and
its queries once at the checkpoint's full width, and `rcp-ndcg retrieval sweep` (`rcp_ndcg.retrieval.sweep`)
applies the declared head per output dimension to the stored vectors and scores each cut -- one forward pass
for every `k` ([matryoshka heads](matryoshka.md)).

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
fields -- `api`, model, revision, prompts, budgets, `normalize`, `dimensions`, the MRL kind, set and
selection) is content and enters the step
and the index identity; where and how fast it is asked (`base_url`, `batch_size`, `concurrency`, the timeouts)
is runtime and never does. `rcp_ndcg.retrieve` reuses an index of the same identity and rebuilds one that
differs. The rerank checkpoint keys on every content field of the reranker and on the digest of the exact
candidate texts it scored: changed texts or changed settings re-score, never silently reuse.

Three more things name an index, and each of them exists because the thing it names can change without any
config field moving:

- **A behaviour version per output-producing step** (`INDEX_BEHAVIOUR_VERSION`, `RETRIEVE_BEHAVIOUR_VERSION`,
  `RERANK_BEHAVIOUR_VERSION` in `rcp_ndcg.retrieval`). A code change that moves an index's, a retrieve
  output's or a rerank output's numbers without changing a config field still re-keys it, so a resume never
  reuses numbers another build computed. The package version is deliberately not used: every release would
  invalidate every resume and judgement pool. A judgement's identity is not versioned -- it is prompt- and
  model-defined.
- **The payload, by content.** `index()` writes each payload file atomically under an exclusive lock and writes
  `index.json` last, carrying the sha256 of every payload file (`vectors.npy`, `offsets.npy`, the `bm25s/`
  model files). `search` recomputes that digest and refuses a payload the record does not describe, so a
  killed or concurrent build is never scored; a missing payload is a typed error, and `retrieve` treats it as
  a cache miss and rebuilds. A rebuild clears the payload of another build (a dense rebuild never leaves a
  late-interaction build's `offsets.npy` beside its vectors).
- **A local dataset's content, and an unhashed media item's bytes.** A local dataset source has no commit, so
  its files' content names it: a file hashes as its bytes (streamed, and cached for the process by size and
  ``mtime_ns``), a directory as its sorted listing of names, sizes and mtimes. An edited `rows.jsonl`
  therefore makes every step that read it stale on resume. What that does not detect: an edit that keeps a
  file's size *and* its ``mtime_ns`` (a same-size rewrite by a tool that restores the timestamp) is invisible
  while the cached digest lives, and a directory listing sees a file's size and mtime, never its bytes. A media reference
  with `sha256` is content by definition; one without it (`hash_media: false`, the reader default) records
  the object's size and change stamp beside its URI -- `mtime_ns` for a local file, the backend's
  etag/generation for a remote object -- in the index identity, the rerank checkpoint key and the media cache
  key. What that detects: a replaced, resized or re-encoded object at the same URI. What it does not detect: a
  same-size edit that also preserves the change stamp (an object restored with its mtime), and a `data:` URI
  (its bytes are the URI). `hash_media: true` hashes the bytes at ingest and detects everything.

`load_index(path)` reads the payload from the directory it is given: the record's own `path` field is where the
index was *built* (provenance), so a copied, moved or restored index directory is searched where it now is. An
index directory must be local or on a shared filesystem: a remote `out` is refused with a hint (write locally
and copy it), never turned into a local directory named `gs:/...`.

## One tie rule

Documents with equal scores must come back in a defined order, and the whole pipeline uses one rule: **score
descending, then the lower document id**. It decides the first stage's cut (`numpy_topk`/`maxsim_topk`, rows
laid out in document-id order), BM25's cut, the rerank depth cut (`Rankings.top`) and the order candidates
go over a reranker's wire in. The metric's own per-protocol tie rules (`group_mean`, `doc_id_desc`,
`input_order`) are a separate, declared choice made at scoring time -- see [scoring protocols and tie
rules](protocols.md). The top-k's scores are exact: a float32 matrix multiply pre-selects candidates within a
margin that bounds its own rounding error, and every candidate is rescored by a deterministic float64
reduction, so the selected set is a function of the inputs alone -- not of the BLAS thread count, the tile size
or how many queries travelled together.

A listwise reranker is the one place a budget is checked per *request*: it scores the whole candidate set in
one prompt, so the summed per-document render (the frame repeats per passage) is refused when it exceeds
`max_tokens`, with a hint to lower `depth` or `document_max_tokens`. The set is never split silently.

## On the wire

All three roles send through one inference layer ([the inference layer](inference.md), with its role clients for
the [embed](embeddings.md), [pooling](late-interaction.md) and [rerank](../api/inference.md) wires). Every request
is fitted through one text-budget mechanism ([text budgets for served roles](text-budgets.md)): over-budget
content is cut client-side at token boundaries, the template's anchors are reserved and re-attached around the
cut, and every cut is recorded -- never an engine-side truncation.
