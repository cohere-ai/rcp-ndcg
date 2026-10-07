# The observation corpus and pairs schema (`rcp-ndcg-vllm`)

One document for the two artifacts the observation lane produces and lane `fake-engines` consumes.
Both are JSONL with a versioned schema (`schema_version`), keyed by **engine + version** and the
recipe's **behaviour fingerprint** (GPU-VALIDATION.md item 8) so a corpus is never silently stale.

## The pairs files (`pairs/<recipe>.jsonl`, one row per request)

JSONL, one object per planned request — exactly what `rcp_ndcg_vllm.equivalence.fitting.load_pairs`
reads — written by `rcp_ndcg_vllm.observe.requests` (`GENERATOR_VERSION`, `SEED`, `PINNED_DATASET_COMMITS`
pin every input):

```json
{"query": "str", "documents": ["str", "..."],
 "shape": "query|document|pair",        // optional: the row's request shape (else every declared shape)
 "instruction": "str",                  // optional: where the recipe's instruction mode fills one
 "media": {"query": [<SourceMedia>], "documents": [[<SourceMedia>], ...]},  // optional (media recipes)
 "_strata": ["source:nanobeir", "length:median", "..."],   // the spec's stratum labels (stripped for the reference)
 "_source": {"suite": "...", "subset": "...", "query_id": "...", "doc_ids": [...], "commit": "..."}}
```

`SourceMedia` is a media entry by source coordinates (never a machine path): `{"suite", "subset",
"doc_id", "part", "sha256", "mime", "num_bytes"}` — the bytes resolve through the product's dataset
loader and `rcp_ndcg.data.media.MediaResolver`. `pairs/manifest.json`
(`rcp-ndcg-vllm.pairs-manifest.v1`) records the generator identity, the pinned commits, every file's
SHA-256, every stratum present-or-absent (absent only when inapplicable, said why), the excluded
source ids, the recipes that could not load (with their error) and the stage-1 validation record.
Bounded runs merge into the manifest (keyed by recipe id). Size budget per GPU-VALIDATION.md item 7:
at most 2 MB per recipe, 30 MB in total.

## The observation corpus (a directory per recording)

Keyed `<engine>-<version>/<recipe>/<fingerprint>`; written by `rcp_ndcg_vllm.record.record_corpus`,
checked by `rcp_ndcg_vllm.observe.corpus.verify_corpus` (OBSERVATIONS-SPEC sections 3-6).

- `records.jsonl` — one `RECORD_SCHEMA = 1` object per exchange. Raw first: the request and the
  response **as sent/received** (`body_raw`: UTF-8 text or `{"base64": ...}`, plus `body_parsed` when
  JSON), the headers that matter only (content type, the server's version header — authorisation and
  gateway headers are stripped before writing, and the acceptance check scans every corpus byte for
  credential shapes), `exchange_id` (SHA-256 of the canonical request — a content address),
  `sequence`, `repetition` (`same_process` twice, then `after_restart`), `batch_context` (companion
  request ids and positions), `server_run_id`, and `inputs` (source ids and commit, stratum labels,
  the token count of each content span in the recipe's tokenizer, the declared media). `_derived`
  views (parsed vectors, normalised bodies, tolerances) are never stored: they recompute on load
  under `NORMALISATION_VERSION`, which names the volatile fields it strips (`id`, `created`, the
  `/v1/models` permission rows' ids and timestamps).
- `nondeterminism.json` — the measured repeated-sending deltas per input and batch context (max
  absolute/relative differences of every numeric leaf, cosine per vector, whether statuses or bodies
  differ) and the derived verification tolerances with their rule (`DERIVED_TOLERANCE_RULE`: 3× the
  measurement with absolute/relative floors — never chosen by hand).
- `manifest.json` — the provenance (engine: image reference **and digest**, vLLM version and commit,
  torch/CUDA/driver, GPUs, the exact serve argv, the behaviour-affecting `VLLM_*` environment;
  model: id, revision, weight index hash, `tokenizer.json` hash, template hash, plugin package,
  version and wheel hash, `hf_overrides`, pooler config, `mm_processor_kwargs`, dtype; recipe: id,
  file hash, the **behaviour fingerprint** and its full input list with values; collector: version and
  commit, `GENERATOR_VERSION`, `RECORD_SCHEMA`, seed, dataset URIs and commits, wave/job ids, times,
  the node hostname hashed) and `integrity` (the SHA-256 of every corpus file and the manifest's own
  hash) — a tampered file fails the check by hash.
- The repository subset (`tests/contract/engines/<engine>-<version>/<recipe>/<fingerprint>/`):
  `records.jsonl.gz` (every stratum's rows first, then sequential, recorded sampling), the manifest
  and `index.json` naming the GCS path and hash of the full corpus. Budgets (GPU-VALIDATION item 7):
  at most 2 MB per recipe, 30 MB in total.

### The acceptance checks (no check silent, no check vacuous)

`manifest_hashes` (every file hash and the manifest's own), `completeness` (every generator-plan
request id), `statuses_as_expected` (the measured engine table — vLLM v0.31.0's, from the shake1c
shakedown recordings: over-length refused `400` with the OpenAI-style `{"error": ...}` body, an
unknown request field **ignored with `200`** on every role route, malformed JSON `400`),
`scores_finite_and_vector_widths`, `nondeterminism_report`, `no_credentials`. A corpus failing any
check is not used.

### Change handling

- `run_wave --changed-since <index>` re-records only the recipes whose behaviour fingerprint differs
  from the previous `wave.json` or corpus index (`changed_since`), listing the rest as
  `skipped_unchanged`; a new engine version re-records the protocol layer once.
- A behaviour diff between two corpora of one recipe (lane `fake-engines`) is reviewed before the new
  corpus replaces the old in the repository.
- Readers support every `RECORD_SCHEMA` they ever wrote (migrations in code, tested on frozen
  samples); a schema bump never invalidates an old corpus. An old corpus leaves the repository only
  by a deliberate commit naming it; its GCS copy stays.
- The two layers key separately: the protocol layer by engine + version (`vllm-0.31.0`), the model
  layer by the recipe's behaviour fingerprint.

## The negative controls (`rcp_ndcg_vllm.observe.controls`)

Six broken variants per family ((a) template removed, (b) `--truncate-prompt-tokens` right cut,
(c) `use_activation` flipped, (d) the wrong pooling, (e) float32 read as float16, (f) `max_pixels`
unpinned), each generated from its family's recipe and served as its own recipe — the ordinary gates
must fail every one. `controls_summary` treats a passing control as a **blocker** (its gate is blind
or a no-op), naming it.