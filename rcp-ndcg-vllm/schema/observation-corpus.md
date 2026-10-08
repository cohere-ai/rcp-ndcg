# The pairs files and the observation corpus (`rcp-ndcg-vllm`)

The two artifacts the GPU waves' collector produces and the verified fake engines consume. The corpus format
has one home and one reader, in the product: `rcp_ndcg.testing.corpus` (`load_corpus`, `integrity_mismatches`,
`normalise_body`, `credential_findings`, `write_subset_index`; `CORPUS_SCHEMA`, `RECORD_SCHEMA`,
`NORMALISATION_VERSION`). This package writes corpora (`rcp_ndcg_vllm.record.record_corpus`) and decides
whether one is accepted (`rcp_ndcg_vllm.observe.corpus.verify_corpus`).

## The pairs files (`pairs/<recipe>.jsonl`, one row per request)

JSONL, one object per planned request -- exactly what `rcp_ndcg_vllm.equivalence.fitting.load_pairs` reads --
written by `python -m rcp_ndcg_vllm.observe.requests` (`GENERATOR_VERSION`, `SEED` and
`PINNED_DATASET_COMMITS` pin every input):

```json
{"query": "str", "documents": ["str", "..."],
 "shape": "query|document|pair",
 "instruction": "str",
 "media": {"query": [<SourceMedia>], "documents": [[<SourceMedia>], "..."]},
 "_strata": ["source:nanobeir", "length:median", "..."],
 "_source": {"suite": "...", "subset": "...", "query_id": "...", "doc_ids": ["..."], "commit": "..."}}
```

`shape`, `instruction` and `media` are optional; the `_`-prefixed keys are provenance the reference subprocess
never sees. `SourceMedia` addresses a media item by its source (`suite`, `subset`, `doc_id`, `part`, `sha256`,
`mime`, `num_bytes`), never by a machine path. `pairs/manifest.json` (`rcp-ndcg-vllm.pairs-manifest.v1`)
records the generator's identity, the pinned commits, every file's SHA-256 and row count, every stratum present
or absent with the reason, the excluded source ids, the recipes that could not load (with their error) and what
the stage-1 validation ran. `rcp-ndcg-vllm/pairs/` is the files' one home; `jobs/rc_build.sh` stages
it.

## The corpus request plan

`rcp_ndcg_vllm.observe.requests.corpus_plan` (`CORPUS_PLAN_VERSION`) is what a recording sends beyond the pairs
rows: the length ladder one token over, 2x and 10x the budget in the recipe's own tokens, and every content kind
too long for a pairs row -- each through the product's client (its cut) and bare (uncut: the engine's own
refusal); the wire variants of the role's route (`encoding_format` x `embed_dtype` on `/pooling`, `base64` and
`dimensions` on `/v1/embeddings`, `top_n`, `use_activation` and `instruction` on `/rerank`); and the protocol
edges (an invalid `embed_dtype`, `top_n` over the documents). The collector adds its standing probes
(`/v1/models`, `/health`, an unknown field, malformed JSON, a wrong model name, an empty input, over-length),
the engine's `/tokenize` of the exact prompt the client sent for each input, and -- from the wave's restart --
one request while the engine loads. Every stratum is recorded present or absent with the reason in the
manifest's `plan.strata`. The media request set is not implemented yet (recorded absent, `BLOCKED`).

## The observation corpus (one directory per recording)

Written at `observations/vllm-<version>/<recipe>/<fingerprint>/<recorded-at>/` under the wave's output
(`observe.corpus.corpus_path`; a new recording is a new path), keyed by the engine version and the recipe's
behaviour fingerprint (`rcp_ndcg_vllm.fingerprint.behaviour_fingerprint`, GPU-VALIDATION.md item 8).

- `records.jsonl` -- one `RECORD_SCHEMA = 1` record per exchange: `exchange_id` (the SHA-256 of the canonical
  request: method, path, body), `sequence`, `repetition` (`same_process_1`, `same_process_2`,
  `after_restart`), `batch_context` (the request ids and positions sent together; a reranker's candidate set
  in its `given` and `reversed` order), `server_run_id` (the after-restart pass carries the restarted engine's
  own), `request` and `response` (method and path; the headers that matter -- content type, server, the
  version header and a `bytes` reply's `metadata` framing; the body as sent or received in `body_raw`, UTF-8
  text or `{"base64": ...}`, beside `body_parsed`; `latency_s`, never used for verification) and `inputs`
  (request id, stratum labels, layer `model` or `protocol`, probe, source ids and commit, token counts, media).
  Every request is sent twice in one engine process and once after an engine restart; an embedder's inputs
  alone and in batches of 2, 8 and 32 per side.
- `nondeterminism.json` -- per request sent more than once (`exchange_id`): the repetitions and server runs it
  spans, the maximum absolute and relative differences of every numeric leaf of the normalised bodies and of
  every decoded vector component, the minimum cosine per vector, whether statuses or bodies differ; and the
  derived tolerances under `DERIVED_TOLERANCE_RULE` (3x the measured drift, floors 1e-9 and 1e-6) -- or
  `measured: false` and no tolerance when no request was repeated. Vectors are decoded by the product's own
  adapters (`observe.corpus.derived_vectors`); the volatile reply fields (`id`, `created`) are normalised away.
- `manifest.json` -- the provenance (`observe.provenance`): `engine` (image and digest, vLLM version and commit,
  torch, CUDA, driver, GPUs, the exact serve argv, the `VLLM_*` environment without credential-named variables,
  start time, readiness wait), `model` (id, revision, weight file hashes from the Hub cache, tokenizer and
  template hashes, plugin with its wheel hash, `hf_overrides`, pooler config, `mm_processor_kwargs`, dtype),
  `recipe` (id, the recipe file's hash, the behaviour fingerprint and its named inputs, status, the declared
  vector width), `collector` (package version and commit, `GENERATOR_VERSION`, `CORPUS_PLAN_VERSION`,
  `RECORD_SCHEMA`, seed, dataset commits, wave and job ids, times, the hostname's SHA-256, the passes), the
  `plan` (request ids and strata) and `integrity` (every file's SHA-256 and size, the record count and the
  manifest's own digest). A fact that cannot be collected is `{"unavailable": "<reason>"}`, never left out.

### The acceptance checks (`verify_corpus`; `python -m rcp_ndcg_vllm.observe.corpus verify DIR`)

`manifest_hashes` (every hashed file and the manifest's digest; a subset: its index's hashes),
`completeness` (every planned request id; a subset defers to its full corpus), `statuses_as_expected` (the
measured vLLM v0.31.0 table: over-length `400`, an unknown field ignored with `200`, malformed JSON `400`),
`scores_finite_and_vector_widths` (every number finite, every vector reply decoded at the declared width),
`nondeterminism_report`, `provenance_complete`, `no_credentials` (every byte, a subset's `.gz` decompressed),
and, given the equivalence stage's exchanges of the same wave, `equivalence_consistent` (the same replies
within the measured tolerance; no common request is a failure). A corpus failing any check is not used.

### The repository subset (`python -m rcp_ndcg_vllm.observe.corpus subset DIR OUT --gcs-path URI`)

Under `tests/contract/engines/vllm-<version>/<recipe>/<fingerprint>/`: `records.jsonl.gz` (the first record of
every stratum, then corpus order until 2 MB compressed; corpus order kept), the full corpus's `manifest.json`
and `nondeterminism.json` byte for byte, and `index.json` (`rcp-ndcg.observation-subset/1`: the full corpus's
storage path and manifest hash, the recorded sampling, the subset's file hashes). The 30 MB repository total is
the conformance suite's to hold.

### Change handling

- `run_wave --changed-since <wave.json>` records again only the recipes whose corpus key moved: the behaviour
  fingerprint or the engine version (`observe.corpus.changed_since`); the wave document names why
  (`changes`) and the engine versions whose protocol layer is due (`protocol_due`).
- Readers support every `RECORD_SCHEMA` ever written (`register_record_migration`; a frozen schema-1 sample,
  `tests/observation_corpus_v1/`, pins it); a record from a newer collector is refused.
- A behaviour diff between two corpora of one recipe, the append-only verification record and the retirement
  of an old corpus are the fake-engines lane's.

## The T3 quality stage (`rcp_ndcg_vllm.quality`, `run_wave --quality`)

Per recipe and task of `TASK_MATRIX` (every subset of its suites at the pinned commits): the served path is the
product's CLI (`retrieval index`/`search` with the recipe's retriever config, or `retrieval rerank` over the
suite's released pools with its reranker config, then `eval score`); the reference is `mteb` with the HF model
over the product's task definitions, in the reference environment. RCP-nDCG@10 and qrel-nDCG@10 are gated
served vs reference (and vs the paper's stored numbers, `--paper-numbers`) within 0.5 points per subset;
published numbers are a noted column. The served exchanges of one NanoBEIR and one ViDoRe task are captured by a
recording proxy as a golden-replay corpus. Output: `quality.json` and `QUALITY.md`.

## The negative controls (`rcp_ndcg_vllm.observe.controls`, `run_wave --controls`)

(a) the served template removed, (b) an engine-side right cut (`truncate_prompt_tokens` with
`truncation_side: right` on the requests), (c) `use_activation` flipped, (d) the declared pooling swapped,
(e) `/pooling` frames requested in the other `embed_dtype` than the client decodes, (f) `max_pixels` unpinned.
After the recipe's own gates passed, each applicable control runs through the ordinary gates, which must fail
it; a control that passes is a blocker that fails the recipe and is named in `wave.json` and `WAVE.md`; an
inapplicable control is listed with its reason. (f) has no media gate in the harness yet: on a VL recipe it is a
blocker.
