# The pairs files and the observation corpus (`rcp-ndcg-test`)

The two artifacts the GPU waves' collector produces and the verified fake engines consume. The corpus format
has one home and one reader, in this package: `rcp_ndcg_test.corpus` (`load_corpus`, `integrity_mismatches`,
`normalise_body`, `credential_findings`, `write_subset_index`; `CORPUS_SCHEMA`, `RECORD_SCHEMA`,
`NORMALISATION_VERSION`). This package writes corpora (`rcp_ndcg_test.record.record_corpus`) and decides
whether one is accepted (`rcp_ndcg_test.observe.corpus.verify_corpus`).

## The pairs files (`pairs/<recipe>.jsonl`, one row per request)

JSONL, one object per planned request -- exactly what `rcp_ndcg_test.equivalence.fitting.load_pairs` reads --
written by `python -m rcp_ndcg_test.observe.requests` (`GENERATOR_VERSION`, `SEED` and
`PINNED_DATASET_COMMITS` pin every text input; the media rows carry their own `MEDIA_SET_VERSION`):

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
`mime`, `num_bytes`), never by a machine path. A media row's entries may also be inline media (`{"kind":
"image"|"video", "uri": "data:...", "sha256", "mime", "width", "height", "num_bytes", "num_frames",
"duration_s", "fps"}` — a page drawn by the media set, or one of its MJPEG AVI clips) or `text` segments
(`{"kind": "text", "text": ...}` — an interleaved row's part sequence, standing where they stand).
`pairs/manifest.json` (`rcp-ndcg-vllm.pairs-manifest.v1`)
records the generator's identity, the pinned commits, every file's SHA-256 and row count, every stratum present
or absent with the reason, the excluded source ids, the recipes that could not load (with their error) and what
the stage-1 validation ran. `rcp-ndcg-test/pairs/` is the files' one home; `jobs/rc_build.sh` stages
it.

## The corpus request plan

`rcp_ndcg_test.observe.requests.corpus_plan` (`CORPUS_PLAN_VERSION`) is what a recording sends beyond the pairs
rows: the length ladder one token over, 2x and 10x the budget in the recipe's own tokens, and every content kind
too long for a pairs row -- each through the product's client (its cut) and bare (uncut: the engine's own
refusal); the wire variants of the role's route (`encoding_format` x `embed_dtype` on `/pooling`, `base64` on
`/v1/embeddings`, `top_n`, `use_activation` and `instruction` on `/rerank`); the **MRL stratum** (one bare
`dimensions=k` probe per declared Matryoshka `k`, read from the recipe's client declaration -- every `mrl_dims`
member, or a `mrl_range`'s endpoints and the run's selection; a recipe with no declared head records the stratum
absent with the reason and keeps one undeclared-cut probe); and the protocol
edges (an invalid `embed_dtype`, `top_n` over the documents). The collector adds its standing probes
(`/v1/models`, `/health`, an unknown field, malformed JSON, a wrong model name, an empty input, over-length),
the engine's `/tokenize` of the exact prompt the client sent for each input, and -- from the wave's restart --
one request while the engine loads. Every stratum is recorded present or absent with the reason in the
manifest's `plan.strata`. The media request set (`rcp_ndcg_test.observe.media_set`, `MEDIA_SET_VERSION`) is
the pairs rows themselves for every media recipe -- an image per size bucket, a captioned page, a mixed batch,
a query image, interleaved and several-image documents where the recipe's `max_images` admits them, and an
MJPEG AVI clip per size at the recipe's declared video sampling, written on CPU and structurally pinned by a
test; its protocol edges (more images than `max_images`, an undecodable image) go bare.

## The observation corpus (one directory per recording)

Written at `observations/vllm-<version>/<recipe>/<fingerprint>/<recorded-at>/` under the wave's output
(`observe.corpus.corpus_path`; a new recording is a new path), keyed by the engine version and the recipe's
behaviour fingerprint (`rcp_ndcg_test.fingerprint.behaviour_fingerprint`, GPU-VALIDATION.md item 8).

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
  torch, CUDA, driver, GPUs, the exact serve argv, the `VLLM_*` environment and `RCP_NDCG_VLLM_PATCHES`
  without credential-named variables,
  start time, readiness wait), `model` (id, revision, weight file hashes from the Hub cache, tokenizer and
  template hashes, plugin with its wheel hash, `hf_overrides`, pooler config, `mm_processor_kwargs`, dtype),
  `recipe` (id, the recipe file's hash, the behaviour fingerprint and its named inputs, status, the declared
  vector width), `collector` (package version and commit, `GENERATOR_VERSION`, `CORPUS_PLAN_VERSION`,
  `RECORD_SCHEMA`, seed, dataset commits, wave and job ids, times, the hostname's SHA-256, the passes), the
  `plan` (request ids and strata) and `integrity` (every file's SHA-256 and size, the record count and the
  manifest's own digest). A fact that cannot be collected is `{"unavailable": "<reason>"}`, never left out.

### The acceptance checks (`verify_corpus`; `python -m rcp_ndcg_test.observe.corpus verify DIR`)

`manifest_hashes` (every hashed file and the manifest's digest; a subset: its index's hashes),
`completeness` (every planned request id; a subset defers to its full corpus), `statuses_as_expected` (the
measured vLLM v0.31.0 table: over-length `400`, an unknown field ignored with `200`, malformed JSON `400`),
`scores_finite_and_vector_widths` (every number finite, every vector reply decoded at the declared width),
`nondeterminism_report`, `provenance_complete`, `no_credentials` (every byte, a subset's `.gz` decompressed),
and, given the equivalence stage's exchanges of the same wave, `equivalence_consistent` (the same replies
within the measured tolerance; no common request is a failure). A corpus failing any check is not used.

### The repository subset (`python -m rcp_ndcg_test.observe.corpus subset DIR OUT --gcs-path URI`)

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

## The T3 quality stage (`rcp_ndcg_test.quality`, `run_wave --quality`)

Per recipe and task of `TASK_MATRIX` (every subset of its suites at the pinned commits): the served path is the
product's CLI (`retrieval index`/`search` with the recipe's retriever config, or `retrieval rerank` over the
suite's released pools with its reranker config, then `eval score`); the reference is `mteb` with the HF model
over the product's task definitions, in the reference environment. RCP-nDCG@10 and qrel-nDCG@10 are gated
served vs reference (and vs the paper's stored numbers, `--paper-numbers`) within 0.5 points per subset;
published numbers are a noted column. The served exchanges of one NanoBEIR and one ViDoRe task are captured by a
recording proxy as a golden-replay corpus. Output: `quality.json` and `QUALITY.md`.

## The negative controls (`rcp_ndcg_test.observe.controls`, `run_wave --controls`)

(a) the served template removed, (b) an engine-side right cut (`truncate_prompt_tokens` with
`truncation_side: right` on the requests), (c) `use_activation` flipped, (d) the declared pooling swapped,
(e) `/pooling` frames requested in the other `embed_dtype` than the client decodes, (f) `max_pixels` unpinned,
(g) an undeclared Matryoshka cut (`dimensions` the engine's own gate must refuse: a `k` outside its declared
set, the field `/pooling` refuses outright, or any cut on a checkpoint without the Matryoshka gate).
After the recipe's own gates passed, each applicable control runs through the ordinary gates, which must fail
it; a control that passes is a blocker that fails the recipe and is named in `wave.json` and `WAVE.md`; an
inapplicable control is listed with its reason. (f)'s media half is the media stage's engine count, which
fails an unpinned or re-resizing engine (and, under the recipe's declared video policy, an engine not pinned
to the declared frame count).
