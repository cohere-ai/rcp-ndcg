# Retrieval and reranking

The first stage builds each query's candidate pool, and the second stage re-scores it. A run's `retrieve` and
`rerank` steps do both before the judging (`candidates: {from: retrieval, retrieval: ..., rerank: ..., depth: 50}`). Outside a run, the retrieval commands write a rankings file. Every scoring protocol reads that file
directly.

A run can also fuse several systems as its first stage, with this config:

`candidates: {from: fuse, fuse: {systems: {s1: <rankings file>, s2: <file>#<system>}, rrf_k: 60}}`

The run's `fuse` step then writes the RRF of the named systems as its candidate pools. RRF is reciprocal rank
fusion, which merges rankings by rank. The step uses the same `rcp.retrieval.fuse` arithmetic as
`rcp retrieval fuse`, over rankings files rather than over documents. Those files can be
`retrieval search` outputs or another run's rankings. The fusion is therefore modality-free. A system can be
unreadable in three ways. Its file is missing. The system was not retrieved into its file. A multi-system file
has no `#<system>` selector. A fusion of fewer than two systems is also refused. Both cases are a typed
refusal, never a silent ranking by one system ([runs](runs.md#runs-and-job-runners)).

## What retrieval contributes

`rcp retrieval index` (`rcp.retrieval.index`) builds an index of a dataset's corpus.
`rcp retrieval search` (`rcp.retrieve`) searches it into a `Rankings`. It reuses an index whose
identity matches, and it rebuilds one that differs. `rcp retrieval rerank` (`rcp.rerank`) re-scores
each query's top `depth` candidates as another system. `rcp retrieval fuse` (`rcp.fuse`) merges
several rankings by reciprocal rank fusion. The commands take their config as a YAML file (`--retriever`,
`--reranker`). The exact flags are in `rcp schema show commands --json`.

A Matryoshka model has two more commands. `rcp retrieval store` (`rcp.retrieval.build_store`)
encodes the corpus and its queries once, at the checkpoint's full width. `rcp retrieval sweep`
(`rcp.retrieval.sweep`) applies the declared head per output dimension to the stored vectors, and it
scores each cut. One forward pass then serves every `k` ([matryoshka heads](matryoshka.md)).

## Retriever kinds

A retriever config declares one of three kinds:

| `kind` | What it computes | Fields |
|---|---|---|
| `bm25` | sparse term scores | `stemmer`: a Snowball language of PyStemmer (`english`, `german`, ...) or `null`; corpus and queries are stemmed alike, the stop list follows the language (`en`, `de`, ...; none where bm25s keeps no list for it), and the stemmer is part of the index identity |
| `dense` | one vector per query and document | `encoder`: an embedding endpoint ([embedding endpoints](embeddings.md)) |
| `late_interaction` | per-token vectors, scored with MaxSim | `encoder`: a pooling endpoint ([late interaction](late-interaction.md)) |

The kinds read different text, each with the rule that its own tool uses. The sparse path follows **mteb's own
BM25 in the text it indexes and searches**, byte for byte. It is not a served model, and it reads no MTEB
dataloader. A corpus row is indexed as `title + "\n" + body`, with both parts as given. A query is the
per-query instruction's append alone, never the `Task:` frame. The SCORING is `bm25s` on both sides, because
mteb's wrapper and this package both call it. For the shipped `stemmer: english` config, the tokenisation
coincides with mteb's `BM25Tokenizer` for `eng`. That tokenisation is the bm25s `en` stop list and the
English Snowball stemmer. mteb's frequency-threshold filtering applies only to languages without a named stop
list. A non-English stemmer uses that language's own bm25s stop list: `german` -> `de`, `french` -> `fr`,
and so on. A language that bm25s keeps no list for keeps no stop words. The list is recorded in the index's
`meta.json` and read back at search time. The English list is therefore never applied to another language. The
dense and late-interaction paths read **MTEB's retrieval dataloader** rule. That rule gives
`(title + " " + body).strip()`, or the body alone without a title. They place the task instruction per the
encoder config's `instruction` mode. They append the per-query instruction as mteb's dataloader does
([data](../data.md)). A document-side task instruction reaches an encoder through the template's `instruction`
span. The generic default frames the query side only. The instruction enters the index identity. A served
encoder's `base_url` must be the form that its wire speaks. The OpenAI-style embeddings wire (`api: openai_embeddings`) takes `.../v1`. The pooling wire (`api: vllm_pooling`) takes the engine's root, because
vLLM registers `/pooling` at the root. A `.../v1` base there is refused where the config is read, never as a
404 at the first request.

The second stage is `rcp retrieval rerank` and a run's `rerank` step. It takes a reranker config instead:
a cross-encoder's scores over `POST {base_url}/rerank`, a served engine or a hosted API. The wire and the
client are documented as [`rcp.inference`](../reference/api/rcp.inference.md).

## Indexes, checkpoints and identity

Every config declares IDENTITY_ROLES. What the model computes is content, and it enters the step and the index
identity. That covers the `kind`, the `stemmer`, and the encoder's content fields. The fields are `api`,
model, revision, prompts, budgets, `normalize` and `dimensions`. They also include the MRL kind, set and
selection, `batch_size`, `max_images` and `max_videos`. Where and how fast it is asked is runtime, and it
never enters them: `base_url`, `concurrency` and the timeouts. `rcp.retrieve` reuses an index of the
same identity, and it rebuilds one that differs. The rerank checkpoint keys on every content field of the
reranker. It also keys on the digest of the exact candidate texts that it scored. Changed texts or changed
settings re-score, and they are never silently reused.

Three more things name an index. Each of them exists because the thing it names can change without any config
field moving:

- **A behaviour version per output-producing step**: `INDEX_BEHAVIOUR_VERSION`, `RETRIEVE_BEHAVIOUR_VERSION` and
  `RERANK_BEHAVIOUR_VERSION` in `rcp.retrieval`. A code change can move an index's numbers, a retrieve output's
  numbers or a rerank output's numbers. It can do so without changing a config field. The version still re-keys it, so a
  resume never reuses numbers that another build computed. The package version is deliberately not used. Every release
  would invalidate every resume and judgement pool. A judgement's identity is not versioned, because it is prompt- and
  model-defined.
- **The payload, by content.** `index()` writes each payload file atomically under an exclusive lock. It writes
  `index.json` last, carrying the sha256 of every payload file: `vectors.npy`, `offsets.npy` and the `bm25s/` model
  files. `search` recomputes that digest and refuses a payload that the record does not describe. A killed or
  concurrent build is therefore never scored. A missing payload is a typed error, and `retrieve` treats it as a
  cache miss and rebuilds. A rebuild clears the payload of another build: a dense rebuild never leaves a
  late-interaction build's `offsets.npy` beside its vectors. A `search` refuses a query that it cannot score,
  rather than score it 0. There are three cases. The dataset has no queries. The encoder answered zero-width query
  vectors. Over a `bm25` index, the query has no indexable term or no matching document.
- **A local dataset's content, and an unhashed media item's bytes**. A local dataset source has no commit. Its files'
  content names it. A file hashes as its bytes, streamed, and cached for the process by size and ``mtime_ns``. A
  directory hashes as its sorted listing of names, sizes and mtimes. An edited `rows.jsonl` therefore makes every step
  that read it stale on resume. One case escapes that check. An edit that keeps a file's size *and* its ``mtime_ns`` is
  invisible while the cached digest lives. A same-size rewrite by a tool that restores the timestamp does this. A
  directory listing also sees a file's size and mtime, never its bytes. A media reference with `sha256` is content by
  definition. A reference without it (`hash_media: false`, the reader default) records the object's size and change
  stamp beside its URI. The stamp is `mtime_ns` for a local file, and the backend's etag/generation for a remote object.
  It enters the index identity, the rerank checkpoint key and the media cache key. That detects a replaced, resized or
  re-encoded object at the same URI. It does not detect a same-size edit that also preserves the change stamp. One
  example is an object restored with its mtime. It does not detect a `data:` URI either, because its bytes are the URI.
  The lookup itself is memoized per URI for the process. It makes one stat or metadata call per reference, however many
  identities and cache lookups ask. A replacement is therefore seen by the *next* process, which is the granularity that
  a run's steps already have. A long-lived process (a server, a notebook) holds the stamp for its whole life. Restart it
  after replacing media at an unchanged URI, or use `hash_media: true`. `hash_media: true` hashes the bytes at ingest,
  and it detects everything.

`load_index(path)` reads the payload from the directory that it is given. The record's own `path` field is
where the index was *built*, for provenance. A copied, moved or restored index directory is therefore searched
where it now is. An index directory must be local or on a shared filesystem. A remote `out` is refused with a
hint: write locally and copy it. It is never turned into a local directory named `gs:/...`.

## One tie rule

Documents with equal scores must come back in a defined order. The whole pipeline uses one rule: **score
descending, then the lower document id**. That rule decides the first stage's cut (`numpy_topk`/`maxsim_topk`,
rows laid out in document-id order). It decides BM25's cut and the rerank depth cut (`Rankings.top`). It
decides the order in which candidates go over a reranker's wire. The metric's own per-protocol tie rules
(`group_mean`, `doc_id_desc`, `input_order`) are a separate, declared choice, made at scoring time. See
[scoring protocols and tie rules](protocols.md). The top-k's scores are exact. A float32 matrix multiply
pre-selects candidates within a margin that bounds its own rounding error. Every candidate is rescored by a
deterministic float64 reduction. The selected set is therefore a function of the inputs alone. It does not
depend on the BLAS thread count, the tile size or how many queries travelled together.

A listwise reranker is the one place where a budget is checked per *request*. It scores the whole candidate
set in one prompt. The summed per-document render is therefore refused when it exceeds `max_tokens`, because
the frame repeats per passage. The hint says to lower `depth` or `document_max_tokens`. The set is never
split silently.

## On the wire

All three roles send through one inference layer ([the inference layer](inference.md)). That layer has role
clients for the [embed](embeddings.md), [pooling](late-interaction.md) and
[rerank](../reference/api/rcp.inference.md) wires. Every request is fitted through one text-budget
mechanism ([text budgets for served roles](text-budgets.md)). Over-budget content is cut client-side at token
boundaries. The template's anchors are reserved and re-attached around the cut, and every cut is recorded.
Truncation is never engine-side.
