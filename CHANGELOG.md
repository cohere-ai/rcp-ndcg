# Changelog

## Versioning

While the version is `0.x`, a change that breaks the public surface bumps the minor version and an additive change
bumps the patch version; from `1.0` on, semantic versioning applies. The public surface is what
`tests/contract/snapshots/` and `schemas/` pin:
- the Python names in the `__all__` of the public modules, which `PUBLIC_MODULES` in `tests/contract/surface.py`
  lists: the facade `rcp_ndcg`; `rcp_ndcg_core` with `rcp_ndcg_core.irt`, `.metric`, `.gain`, `.protocol` and
  `.records`; and
  `rcp_ndcg.data`, `rcp_ndcg.data.preprocess`, `rcp_ndcg.inference`, `rcp_ndcg.retrieval`, `rcp_ndcg.judging`,
  `rcp_ndcg.calibration`, `rcp_ndcg.eval`, `rcp_ndcg.eval.mteb`, `rcp_ndcg.runs`, `rcp_ndcg.runners`,
  `rcp_ndcg.errors`, `rcp_ndcg.testing` and `rcp_ndcg.examples`;
- the command tree with its flags and output schemas, the exit codes, the MCP tools, the packaging (distributions,
  extras, entry points) and the JSON Schemas in `schemas/`.

Every other module (for example `rcp_ndcg.cli`, `rcp_ndcg.storage`, `rcp_ndcg.support`, `rcp_ndcg.data.io` and the
submodules of `retrieval`, `judging`, `calibration`, `runs` and `runners`) is internal and may change without notice. A
pull request that changes `tests/contract/snapshots/` or `schemas/` must add an entry here; CI checks it.

Every artifact schema carries its own version (`rcp-ndcg.<name>.v1`), bumped only when that artifact changes
incompatibly, independently of the package version. `rcp-ndcg` pins `rcp-ndcg-core` to its own version; the two are
released together.

## Unreleased

### Public surface

- **The pipeline records are public, and the compatibility rows are gone** (owner decision 37): the records
  `Document`, `Query`, `RankingExample` (with `Text`, `Input` and `ID`) live in `rcp_ndcg_core.records` -- renamed
  from the private `_records`, `__all__` declared, no shim -- and are re-exported by the `rcp_ndcg_core` and
  `rcp_ndcg.data` facades, so a reader/writer plugin imports a public path. `Dataset.from_records` takes the
  records themselves or plain dicts with their field names and aliases, and `Dataset.queries`/`Dataset.corpus` hold
  them; the compatibility row models `QueryRow`/`DocumentRow` are deleted, with their strict rules moved into the
  records (unknown keys refused, numeric ids read as strings on every record, `RankingExample` included). The
  records are the pipeline's working objects, not frozen copies: `Dataset.queries`/`Dataset.corpus` hold them and
  a mutation of a passed-in record is visible in the dataset. The formatting rules stay where workstream 10 put
  them -- one home, `Document.model_content`/`Query.format_query`/`Query.format_content` on the records, read by
  the data layer and the role clients alike. The snapshots and schemas are regenerated; every importer in the
  repository uses the public path.
- **The IRT estimator classes the pipeline refits are readable from the package**: `rcp_ndcg_core.irt` exports
  `BradleyTerryEstimator` and `RaschEstimator` (read lazily: touching them imports torch, as the stand-alone
  fits already do, while importing the package stays torch-free). `rcp_ndcg.calibration` and `rcp_ndcg.judging`
  read them from `rcp_ndcg_core.irt` instead of its private submodules.
- **The `pplx-embed-v1` family** (perplexity-ai/pplx-embed-v1-0.6b @ `2c4d510d`, -4b @ `06456497`, MIT; the
  catalog grows to 30 recipes): dense text embedders on a diffusion-continued-pretrained Qwen3 backbone with
  bidirectional attention -- one mean-pooled float vector per text (1024 dims at 0.6B, 2560 at 4B), no
  instruction, Matryoshka-capable, and an int8/binary *storage* view the checkpoint's sentence-transformers
  pipeline applies after pooling. Served on the stock image: the checkpoint's `PPLXQwen3Model` is `Qwen3Model`
  with an all-to-all mask, so the recipe pins `hf_overrides {architectures: [Qwen3ForCausalLM], is_causal:
  false}` and the pooling runner converts it to embed; the folded pplx plugin registers the checkpoint's own
  config class (`PplxV1Config`, `model_type bidirectional_pplx_qwen3`) so `config.json` parses locally and
  `trust_remote_code` stays false; the checkpoint's own ST metadata resolves MEAN pooling and no activation;
  raw text on `/v1/embeddings` (no prompt, no chat template); the context is the model's 32768 tokens; the
  empty document is `omit_zero` (an empty render is zero tokens, and the engine's MEAN pooler would divide by
  zero); the reference is the card's own sentence-transformers path stopped before its trailing
  `FlexibleQuantizer` (the served engine returns the float mean-pooled vector; the quantiser is a storage
  format), with `over_cap_cut_differs` for the card's id cut.
- **The `pplx-embed-v2-late-9b` variant** (perplexity-ai/pplx-embed-v2-late-9b @ `0f49a997`, MIT): the same
  multimodal late-interaction family at 9B -- one 128-dim vector per kept token from a 32-layer hybrid Qwen3.5
  backbone (8 full-attention layers) with the Dense head [128, 4096]; the family's shared prompts, caps, skip
  words, chat template and pixel pin are byte-identical to the 0.6B's at their pinned revisions, so the
  variant row carries the per-size facts only (bf16 ~16.8 GB, one 80 GB-class GPU; the 32-bit check at its
  4352-token `max_model_len` is below 2^31).
- **The pplx plugin serves both late sizes**: `PplxLateMultiVectorModel` now replaces the generation-only head
  (`ParallelLMHead`/`LogitsProcessor`) with vLLM's `StageMissingLayer` before the parent builds it -- neither
  checkpoint ships `lm_head` tensors (the 0.6B ties it, the 9B declares `tie_word_embeddings: false` and ships
  none) -- so the load tracker has no uninitialised head to refuse and the unused generation-head allocation
  (about 2.0 GB at the 9B's served bf16, 0.5 GB at the 0.6B's) is gone; the Dense-head loader shape-checks
  the shipped `linear.weight` against the served projector (both sizes).
- **One join and the two instructions (workstream 10 C2/C3, owner decisions 27, 33)**: a document is read
  where a model's text is formatted, with MTEB's retrieval dataloader rule, byte for byte --
  `(title + " " + body).strip()`, the body alone (stripped) without a title
  (`rcp_ndcg_core.records.mteb_document_text`, `Document.model_content(title=...)`,
  the new `DocumentTitle`). A role config may declare `title: separate` (the title as its own leading text part,
  the body untouched) instead. The two instructions live in two fields and are placed once each: the TASK
  instruction (`Dataset.task_instruction`, plus `Dataset.task_instruction_for(side)`) is placed by the role
  config's `instruction` mode -- the generic default is the prefix `Task: <instruction>\nQuery: <text>`, a
  template's `instruction` span places it instead, `instruction: none` sends none -- and the PER-QUERY
  instruction (`Query.instruction`) is appended exactly as mteb's dataloader appends it,
  `query + " " + instruction` (`Query.format_query(task_instruction=...)`,
  `Query.format_content(task_instruction=...)`). The embed and pool role configs gain the `instruction` field
  (`fold` or `none`; leaving it unset means UNDECLARED -- a request that carries a task instruction is refused,
  naming both choices, so a recipe that declares nothing is never silently re-formatted); every role config
  (the judge's included) gains `title`; and
  `EmbeddingClient.encode`/`PoolingClient.encode` take `instruction=`, `RerankClient.rerank`/`rerank_many` take
  the task instruction and the per-query one separately (`rerank_many(examples, *, instruction=, checkpoint=)`).
  The Hub reader lifts a uniform per-query instruction to `task_instruction` (BRIGHT's per-domain instructions)
  and refuses a subset that instructs only some of its queries; a differing one stays per query. The sparse
  (BM25) path keeps its own join (mteb's BM25, not the dataloader's) and its own identity for it -- see Fixed.
- **The judge's identity records the task instruction**: an in-memory dataset (`Dataset.from_records`, no URI)
  is now named by its content in a judging pass's identity (queries, corpus, labels, pools, exclusions and the
  task instruction) instead of failing on the missing URI, and a loaded dataset's identity carries its
  `task_instruction` beside its URI and revision.
- **Deployment overrides at serve time** (owner decision 36): `rcp-ndcg-vllm serve <id> --set <path>=<value>`
  sets the engine's resource, scheduling and address knobs without touching the recipe. The recipe schema
  declares that surface once (`rcp_ndcg_vllm.recipe.FIELD_ROLES`, whose values are the `RecipeFieldRole`
  `CONTENT`/`RUNTIME`/`DEPLOYMENT` roles), and only a DEPLOYMENT path may be named: `resources.gpus`
  (`--tensor-parallel-size`), `serve.gpu_memory_utilization`, `serve.max_num_seqs`,
  `serve.max_num_batched_tokens`, `serve.host`, `serve.port` and `serve.max_model_len` -- the last refused,
  with both numbers, below the client's largest token budget (`client.max_tokens`, `query_max_tokens` or
  `document_max_tokens`), because the engine would reject admissible prompts; raising it is allowed, up to the
  checkpoint's own limit, which the engine enforces at startup. A CONTENT path (the model, the revision,
  `serve.dtype`, the pooler config, a template, the hf overrides, a patch) is refused by name with the hint
  *a different revision or content is a different variant: add a variant row*; `engine.startup_timeout_s` is
  refused as RUNTIME (the run owns it). `FIELD_ROLES` is public (with `RecipeFieldRole` and `FieldSpec`).
  `--dry-run` prints the argv, the recipe's identity and the applied
  overrides; a real serve logs the identity and the overrides; a corpus manifest records the argv each engine
  was started with verbatim (`engine.serve_argv`), so an engine started with overrides is recorded with them
  (the GPU waves serve the recipes as shipped). A value is checked against its declared kind and range (`--port`/`serve.port` 0..65535, 0 being
  the engine's own ephemeral port; a finite `serve.gpu_memory_utilization` strictly above 0), and the refusal
  names the flag the operator used.
  `rcp_ndcg_vllm.recipe` gains `RecipeFieldRole`, `FieldSpec`, `deployment_fields`,
  `parse_deployment_overrides` and `recipe_digest`, `serve_argv` gains the `deployment` keyword (its `port` is
  now optional: the deployment value, else the caller's port, applies), and the `rcp-ndcg-vllm` console gains
  `--set`.
- **User recipe files** (decision 36): `rcp-ndcg-vllm serve ./family-dir/ [--variant <id>]` and
  `recipe:./family-dir` (or `recipe:/abs/path`) in `rcp-ndcg` configs and the `--retriever`/`--reranker`
  shorthands load a family directory through the same schema, families included, with the `schema_version`
  check unchanged. A name that looks like a recipe id is the catalog's recipe first (a directory of the same
  name in the working directory does not shadow it; `./name` names the file). Such a recipe is **unshipped**:
  its `status` is forced to `unverified` in every record (the verification record belongs to a shipped
  recipe), `Recipe.shipped` says so, and its identity is the content hash of its resolved form --
  `Recipe.identity`, `unshipped:sha256:<hex>` via `recipe_digest`, the referenced chat template file's bytes
  included, computed once at load -- never a shipped id, so two runs whose files differ never share a run
  identity and the path's spelling is not part of it. A config that records that identity is read back as it
  stands (a run's `status`/resume, an index reload): the pointer is recognised, so `recipe:./dir` works end to
  end. `load_recipe` gains the `variant` keyword, the console gains `--variant`, `client_config` and
  `expand_role_recipe` put that identity in the config's `recipe` field, and the corpus provenance
  (`rcp_ndcg_test.observe.provenance.recipe_facts`) records `shipped`.
- **The recipe family `embeddinggemma-2`** (owner decision 38): `google/embeddinggemma-2` @ `914f7f89` -- one
  768-d space for text, images and video -- served on a vLLM nightly pinned by digest
  (`vllm/vllm-openai:nightly-8cbd5d03...@sha256:b25e8a04...`), because the released v0.31.0 image lacks the
  `EmbeddingGemma2Model` architecture and its transformers 5.17.0 lacks the checkpoint's config and processor
  (both need vLLM commit `02b83919aa2e`/PR #60254 and transformers >= 5.19.0; the switch-to-release note is in
  the recipe). The card's SearchQuery/Document prompts ride the client's `query_prompt`/`doc_prompt`, the
  checkpoint's mean pooling (BOS/EOS reserved) and the Gemma 4 image/video processors are reproduced
  client-side; the video sampling is pinned to 60 fps capped at 32 frames (`--media-io-kwargs`). Its pairs file
  is generated (37 rows at `MEDIA_SET_VERSION` 3, 13 media rows) and its reference is the card's
  sentence-transformers path (`transformers==5.19.0`, `sentence-transformers>=6.1.0`).
- **`ImagePolicy.max_soft_tokens` and the `gemma4` processor family**: the Gemma 4 image/video processors resize
  to a soft-token budget (280 per image, 140 per video frame) rather than a pixel range; the client reproduces
  their aspect-ratio-preserving resize, prepares its fixed point (the Gemma 4 resize is not idempotent, so the
  engine would otherwise resize the prepared image again) and counts one vision wrapper per video frame.
  `EngineSpec.min_version` accepts a setuptools-scm dev series (`0.31.1.dev0`) for a digest-pinned nightly.
- **Prompt prefixes beside a content-only template**: the one-home refusal now fires only when the template's
  own shape renders a fixed segment; a content-only template (the messages route's frame is the engine's chat
  template) may carry `query_prompt`/`doc_prompt`, which is the only way the task prefix reaches a
  content-only chat render.
- **Recipe families** (owner decision 34: one family, many sizes, every size its own tested recipe id):
  the shipped recipes are family directories -- `rcp_ndcg_vllm/recipes/<family>/family.yaml` (the shared
  blocks plus a `variants` table of per-size facts), the family's ONE `reference.py`, its one chat template
  and its `requirements-reference.txt`. Every variant resolves to a full `Recipe` (the resolved recipe's
  JSON Schema is unchanged) and every consumer takes variant ids: `rcp-ndcg-vllm serve <variant-id>`,
  `recipe: <variant-id>` in `rcp-ndcg`, the catalog, the harness's discovery, the wave lists and
  `observe.requests` (one pairs file per variant); the RC builder stages the recipes from the package-data
  path the layout move created and the pairs from their harness home, and every family reference takes its
  checkpoint from the resolved recipe it is passed. A family id is never served. `rcp_ndcg_vllm.recipe` gains
  `Family`, `Variant`, `load_family`, `resolve_recipe` and `iter_families`; `load_recipe` takes a variant id
  (or a single-variant family path), `iter_recipes` returns every variant of every family, and the family
  file format has its own exported schema `rcp-ndcg-vllm/schema/family.schema.json` beside
  `recipe.schema.json`. The reference subprocess contract gains `--recipe <resolved-recipe.json>` (the
  harness passes the resolved recipe it loaded), so one family reference runs every variant.
- The standalone `recipe.yaml` path is gone: a directory without `family.yaml` is refused with a hint, and a
  variant-level override of `client.tokenizer` (injected as `model@revision` unless the family declares one)
  is refused naming the field.
- **First-class, efficient Matryoshka support (owner decision 39)**: every embedding and multi-vector
  endpoint declares its MRL head once -- `mrl_kind` (`truncation`, `projection` or unset), the card's
  supported output dimensions as `mrl_dims` (a discrete table) or `mrl_range` (`[min, max]` prose, with the
  floor enforced client-side) and, for a projection kind, `mrl_projection` (the checkpoint's
  learned `*.safetensors` matrices, read through `rcp_ndcg.storage`) -- and a run selects `k` from that
  declaration
  (`mrl_dim` on both role configs, client-side; the engine-side `dimensions` stays dense-only and
  truncation-kind-only). Every refusal names the field and the fix: a `k` outside the declaration,
  `mrl_dims` beside `mrl_range`, `dimensions`
  beside `mrl_dim`, `dimensions` on another kind, a declared kind without a declaration, and a projection kind
  without its source (or with a range, which names no chain). The one head home is `rcp_ndcg.data.mrl`
  (`MrlHead`, `mrl_cut`, `MrlProjection`): the
  truncation cut moves there from `rcp_ndcg.data.postprocess`, and the projection head loads the declared
  chain in float32 and renormalises. Every row the head changed carries a `ProcessingRecord` with the new
  `mrl_cut` mechanism and its kind, `k` and full width (`mrl_cut` joins `CHANGE_MECHANISMS`). The
  full-width `EmbeddingStore` (`rcp_ndcg.data.EmbeddingStore`, `StoredVectors`, `load_embedding_store`)
  holds corpus and query vectors, ragged offsets for late interaction, and a `store.json` with the schema
  and provenance (model, revision, recipe, prompt digest, tokenizer, budget, full width, dtype, the
  declared MRL head), content-addressed by the retrieval identity plus a full-width marker;
  `rcp_ndcg.retrieval.build_store`/`load_store`/`sweep` wire it, and the new `rcp-ndcg retrieval store` and
  `rcp-ndcg retrieval sweep` commands build it and evaluate every declared `k` from it (per-k rankings
  `<model>@<k>`, then `evaluate`/`compare`) in one forward pass.

- **The data model carries provenance** (workstream 10, owner decisions 27, 29, 33): `Document.title` is a
  field of its own -- `text` is the body, and nothing joins a title with it at read time -- and so is
  `Query.instruction`, the *per-query* instruction (mteb's InstructionRetrieval data), never merged into the
  text at load. `Dataset` gains `subset` (`"default"`), `split` (`"test"`), `task`, `task_instruction`
  (`str | {"query": ..., "document": ...}`, mteb's `TaskMetadata.prompt`) and `provenance` (source URI,
  resolved commit, subset, split, the duplicates policy with its counts), with `Dataset.export_key` the
  `(task, subset, split)` key exports use; `DocumentRow.title` and `Dataset.from_records(..., subset=, split=,
  task=, task_instruction=)` follow. How a model's input combines a title with its body, and the two
  instruction kinds with the text, is a formatting decision made where the text is formatted -- see the
  one-join entry above for what that is.
- **The reader contract widens and moves to entry points**: `SourceReader` gains optional `candidates()`
  (`top_ranked` pools), `excluded()`, `gains()`/`thetas()` (the released calibrated values), `provenance`
  (the new `Provenance`, `DuplicateCounts` and `DuplicatesPolicy` models) and `task`/`task_instruction`, and
  every `Dataset` is built from a reader through one function. The reader and writer tables are now the
  `rcp_ndcg.readers` and `rcp_ndcg.writers` entry-point groups (the same seam as the job runners and
  adapters; the built-ins are declared in `rcp-ndcg`'s manifest): a third-party format is one class in its
  own package, and the shared conformance suite is public as `rcp_ndcg.testing.io_conformance`.
- **`hf://` is mteb's layout, and `mteb:<Task>` loads a task through mteb** (owner decisions 28, 31, 32):
  the Hub reader (`rcp_ndcg.data.io.hub`, the `hf` entry point) is driven by the dataset card, following
  mteb's own resolution -- `{s-}corpus` / `{s-}queries` / `{s-}qrels` configs, the `query` config override,
  the `default`-then-`qrels` fallback, `{s-}top_ranked` pools, an `{s-}instruction` config whose rows win
  over the queries' own column, the requested split else the config's only one -- reads parquet, jsonl,
  jsonl.gz and tsv directly through `huggingface_hub` and pyarrow (no `datasets`, no `mteb`), and reads
  rcp-ndcg's extras (the qrels `gain`/`theta` columns, the `-excluded` config). `mteb:<TaskName>[/<subset>]
  [@split]` (the `[mteb]` extra) runs mteb's own loader and converts its `RetrievalSplitData`, so the 113
  custom-loaded tasks read through the same contract; it records the task name, subset, split and pinned
  revision, and carries `TaskMetadata.prompt` into `task_instruction`.
- **`load_dataset` takes `split=`** for `hf://`, `mteb:` and `beir:` (the requested split; `None` is `"test"`,
  or the source's own convention).
- **`rcp_ndcg.errors.WarningCode` gains `CARD_UNCACHED`**: an offline `hf://` load whose dataset card is not
  in the local cache falls back to the plain `{subset}/` path layout and says so (the card-declared configs
  are unavailable until one online run caches the card).
- **`rcp_ndcg.data.media` gains `media_extension(payload)`** (the format a media payload's magic numbers name)
  and `image_dimensions(payload)`; `store_media(..., mime=)` records a MIME type a suffix alone cannot state
  (a video container).
- **One processing pipeline, one postprocess home (workstream 09, decision 23)**: the split of
  `rcp_ndcg.data.preprocess` and the declared stage order of the role clients. Every public name keeps its import
  path (`rcp_ndcg.data.preprocess` is the aggregation facade), and the contract snapshot records the moved homes:
  the judge text policy, the chunk geometry and `Preprocessing` live in `rcp_ndcg.data.text_policy`; the cut
  record (`CutCause`, `TextCutRecord`) and the census (`TextTruncationCensus`) in `rcp_ndcg.data.census`; the
  served roles' `TextBudget` and `fit` in `rcp_ndcg.data.text_budget`; the census files' record I/O
  (`drop_torn_last_line`, `census_sink_lock`, `append_census_rows`, `read_census_rows`) in
  `rcp_ndcg.storage.census` (exported from `rcp_ndcg.storage`); and the postprocess of model output
  (`l2_normalize`, `max_pool_scores_by_document`, `max_pool_rubric_window_by_document` and
  `skip_keep_mask`) in `rcp_ndcg.data.postprocess` (`l2_normalize` re-exported from `rcp_ndcg.inference.types` as
  before; the Matryoshka head's `mrl_cut` moved on to `rcp_ndcg.data.mrl`, decision 39). `rcp_ndcg.inference.clients._base.STAGES` declares the one preparation pipeline every role composes
  (normalise -> empty -> media -> render -> budget -> lower), and the per-row `ProcessingRecord` is its one output.
  The facade's `__all__` grows by three names the old module carried at module level but did not export:
  `needs_tokenizer`, `require_tokenizer` and `census_sink_lock`.
- **`skip_unapplied`** joins the `ProcessingRecord` change mechanisms (`CHANGE_MECHANISMS`): a pooled document's
  declared `document_skip_token_ids` was not applied to a media item -- the image positions are exempt, the
  client keeps every returned vector, and the deviation is on the row's record, never silently unskipped.
- **The engine's video sampling (owner decision 2026-10-09)**: `VideoPolicy` gains `fps`, the vLLM v0.31.0
  `Qwen3VLVideoBackend`'s own rule, and `num_frames` becomes optional: a `wire: video_url` policy declares
  exactly one of `num_frames` (a pinned uniform count) or `fps` (the engine's rate), and `wire: frames` still
  requires `num_frames`. The new `rcp_ndcg.data.resolution.qwen3_vl_video_frame_indices` ports the backend's
  rule (`int(total_frames / original_fps * fps)`, clamped to its 30 fps ceiling and 4..768 frame bounds), and
  `content_media_tokens` gains an optional `tokenizer`: a `qwen3_vl` container under `fps` is counted from the
  clip's recorded frame count and rate, its timestamp lines exactly when the client's tokenizer is passed (the
  family's 10-token bound otherwise), and the chat template's own vision pair around the placeholder is now
  included. `approx_media_tokens` counts the fps rule's frames too; `prepare_request` and `fit_media_to_budget`
  take the caller's `tokenizer` so the media fit's gate uses the exact count. `VideoPolicy` also gains
  `engine_video_pruning` and `engine_video_pruning_method`: a nonzero engine `--video-pruning-rate` retains a
  computed subset of the per-frame tokens (the EVS or VidCom2 formula, ported for the qwen3_vl family; a
  per-frame family's flat pruned run is refused), the client counts that layout, and the recipe loader refuses
  a serve pruning flag the client has not declared (and a declaration the serve args do not carry). The fps
  rule is likewise refused beside a non-qwen3_vl processor family, and a pinned `num_frames` on the qwen3_vl
  family is refused at count time (that backend samples by fps and ignores the pin; declare `fps`). The
  shipped `qwen3-vl-embedding-2b` recipe now declares that rule (`client.video_policy.fps: 2` with
  `--media-io-kwargs '{"video": {"fps": 2}}'`), and its reference's media mode reports the same realised
  frame count from the pairs entry's own frame count and rate.
- **`PoolingEndpoint.media_head_as_system`** (a media document's fixed head as a system message, for a
  pass-through engine chat template) and **`PoolRequest.system_head`** (the field the pooling adapter renders
  it from).

- **The MTEB dataset writer** (the `mteb` writer of `WRITERS`, `data convert --to mteb`):
  `rcp_ndcg.data.io.mteb.MtebWriter` writes exactly what mteb's `push_dataset_to_hub` writes -- configs
  `{s-}corpus` (`id`, `title`, `text`), `{s-}queries` (`id`, `text`, `instruction` only when a query carries
  one), `{s-}qrels` (`query-id`, `corpus-id`, `score` as int64) and `{s-}top_ranked` -- one parquet shard per
  config at `{config}/{split}-00000-of-00001.parquet`, and a README whose `configs:` front matter is what
  `load_dataset` (and through it mteb's `RetrievalDatasetLoader`) reads the directory with; `card=` (a mteb
  `TaskMetadata` or its fields) renders the card from mteb's own template. rcp-ndcg's extras ride only where
  mteb ignores them: the calibrated `gain`/`theta` columns ride on the qrels, and the exclusions travel in the
  `{s-}excluded` config and are folded out of `top_ranked` (out of the corpus when the data has no pool). A
  grade that is not a whole number is refused (mteb's loader casts the int64 `score` column down to int32,
  where a fractional value fails), naming the pair: export integer
  grades, keep the continuous signal in `gain`/`theta`. A suite dataset writes every subset's configs into one
  directory under one README.
- **`Rankings.save(format="mteb")`**: the `{Task}_predictions.json` of mteb's `_save_task_predictions`, from
  stored rankings (`task=`, `qrels=`, `model_name=`, `model_revision=`, `split=`, `system=`). Every query with
  a non-empty qrels dict must be ranked (a missing one is refused, naming it); a ranked query without qrels is
  dropped (mteb raises on a result for a query that has no qrels); no empty dicts; at most 1,000 documents per
  query (mteb's own cap), ties by document id descending. An existing file is merged the way mteb's own writer
  merges: the (subset, split) written replaces theirs, the file's other splits, subsets and its
  `mteb_model_meta` stay.
- **Scoring stored rankings inside mteb** (`rcp_ndcg.eval.mteb`, the `mteb` extra):
  `stored_rankings_model(rankings, meta)` wraps stored `Rankings` as mteb's `SearchProtocol` -- the served
  scores are the asked queries only, restricted to the task's `top_ranked` pool when it has one, capped at
  `top_k` with ties by document id descending -- and `model_meta(name, revision, **fields)` builds mteb's
  `ModelMeta` from our model identity, the required fields the caller declares, the rest unknown. `mteb.evaluate`
  over the wrapped model writes its own predictions file and genuine `TaskResult` files in mteb's `ResultCache`
  layout (`results/{org__model}/{revision}/{Task}.json` with `model_meta.json` and `run_settings.jsonl`), ready
  for `submit_results`; the integer `ndcg_at_10` equals our `qrel_ndcg` under the suite's protocol (the tie
  rules agree).
- `tools/republish_mteb.py` re-lays the published rcp-ndcg datasets in the writer's exact layout, every subset
the published task definitions read -- all 48 ViDoRe v3 language subsets, not only the eight native-language
ones the paper scores -- each at the split its definition pins (NanoBEIR `train`, BRIGHT `standard`, ViDoRe v3
`test`; owner decision 40), with a corpus shared by several subsets written once (the card's `-corpus` entries
decide the groups: ViDoRe v3's six languages of one domain and TREC-DL's two years read the same files); it
validates each written repository with mteb's own `RetrievalDatasetLoader` (media included, a shared corpus
loaded once per group, from a uniquely named symlink view so a re-run cannot read a stale build) and refuses a
task definition whose subset or split does not match the data, in either direction, and pushes nothing (the
owner pushes, with the move to a Hugging Face organisation).
- **The MTEB writer writes mteb's media columns** (owner decision 40): a document's (or query's) `image`/`video`
  parts become mteb's own `struct<bytes, path>` cells with the parquet's `huggingface` feature metadata -- the
  shape `rcp-ndcg-vidore-v3` stores -- so `datasets.load_dataset` reads them as `datasets.Image`/`Video` and
  mteb's dataloader hands a model the decoded page image. One image and one video per row; an interleaved
  document (several images, or a video of extracted frames, container or not) is refused by name. `path` is
  null: the internal `MediaRef` is content-addressed, and mteb reads the bytes. `write_dataset(...
  corpus_group=)` writes a suite's shared corpus once, counts its rows once and refuses a repeated group whose
  rows differ.
- **The layout move**: the repository is four distribution directories (`rcp-ndcg/`, `rcp-ndcg-core/`,
  `rcp-ndcg-vllm/`, `rcp-ndcg-test/`; the root manifest is the uv workspace only). `rcp-ndcg-vllm` is the lean
  serving package (dependencies pydantic and PyYAML only; the recipes are package data read through
  `importlib.resources`; the `rcp-ndcg-vllm serve <recipe-id> [--dry-run]` console; the topk and pplx model
  plugins fold into `rcp_ndcg_vllm/models/` under one lazily registering `vllm.general_plugins` entry point and
  one version guard, and the separate plugin distributions are gone). The validation tooling -- the equivalence
  harness, the engine recorder, the wave runner and node scripts, the request generator, the T4 driver -- and
  the verified emulators and observation corpora (`rcp_ndcg_test.engines`, `rcp_ndcg_test.corpus`, `corpora/`,
  the conformance suite) move to the unpublished `rcp-ndcg-test` (owner decision 20); the product keeps
  `rcp_ndcg.inference.fake` and the offline helpers, and routes a `fake://<engine>-<version>/<recipe>` URL
  through the new `rcp_ndcg.fake_transports` entry-point group (`rcp-ndcg-test` registers its emulators;
  without one the refusal is typed, naming the package).
- **`recipe: <id>`** (docs-firstcontact Q1, owner decision 17): a role config that names a shipped serving
  recipe takes its whole client block from it -- explicit CONTENT fields must equal the recipe's or the config
  is refused naming both values, RUNTIME fields stay on the config, and `--retriever recipe:<id>` /
  `--reranker recipe:<id>` is the one-string shorthand. The shipped paper configs that agree with their recipe
  point at it (`ctxl-*`, `octen`); the ones that reproduce the paper's own path keep every explicit field and
  no pointer.
- **The recipe file format is the versioned contract** (owner decision 18): every recipe carries
  `schema_version`, the exported JSON Schema pins it, and `rcp-ndcg` checks the versions it reads
  (`rcp_ndcg.inference.recipes.RECIPE_SCHEMA_VERSIONS`) when it resolves `recipe: <id>` -- no lockstep version
  pin between rcp-ndcg and rcp-ndcg-vllm (core and rcp-ndcg keep theirs).
- `rcp_ndcg.inference` exports the recipe-resolution surface: `available_recipe_ids`, `expand_role_recipe`,
  `recipe_client_data`, `recipe_role`, `shorthand_config`, `RECIPE_SCHEMA_VERSIONS`.
- The layering charters rename `rcp_ndcg.llm` to `rcp_ndcg.judging` (owner decision 21); the module's names do
  not move.
- `rcp-ndcg-test` is never published and installs from a git subdirectory (owner decision 22); its README and
  the docs say so.
- The release workflow builds and publishes the three published distributions from their own directories in the
  order core -> rcp-ndcg -> vllm; no plugin wheels are built or published. One merged NOTICE ships
  byte-identical in all four distributions.
- **The recipe `pplx-embed-v2-late-0.6b`** (perplexity-ai/pplx-embed-v2-late-0.6b @ `8fc2de24`, MIT; 20 public
  recipes): a multimodal late-interaction retriever on a Qwen3.5 backbone -- one L2-normalized 128-dim vector per
  kept token, client-side fp32 MaxSim. The checkpoint is a native sentence-transformers export (no custom code):
  Transformer -> `1_Dense` (Linear 1024->128, no bias) -> `2_MultiVectorMask` (the 32 ASCII punctuation ids
  dropped document-side, declared as `client.document_skip_token_ids`) -> `3_Normalize`; the role prompts are the
  checkpoint's own `[Q] `/`[D] ` added tokens (the template resolves them by name, the wire stays `text`), the
  per-shape budgets are the sentence-transformers caps (query 1024, document 4096), image documents ride the
  checkpoint's chat template through the messages route (`media_sides: ["document"]`, one image per prompt, the
  R20 nested pixel pin at the shipped processor's 3136..1800964 px with `engine_pixel_pinning: true`), and the
  reference is the card's own sentence-transformers path (`reference.known_deviations: [over_cap_cut_differs]`:
  the card cuts the rendered prompt's ids at the caps, the client cuts text).
- **Recipe families (owner decision 34)**: `rcp_ndcg_vllm.recipe` exports the family loader -- `Family`,
  `Variant`, `load_family`, `iter_families`, `resolve_recipe` -- and `load_recipe` now takes a variant id (or
  a single-variant family path); a family directory's `family.yaml` carries the shared serve/client/reference
  blocks plus a `variants` table whose overrides are restricted to the declared per-size fields. The
  contract snapshot `tests/contract/snapshots/python_api.json` is regenerated for the added names.
- **The recipe family `harrier-oss-v1`** (microsoft/harrier-oss-v1-270m @ `31de22b6`, -0.6b @ `f9b9dc8d`,
  -27b @ `0c0fc62f`, MIT; 22 public recipes): one family, three sizes (owner decision 34), every size its own
  served, tested recipe id. The checkpoints share the pipeline byte for byte (modules.json:
  sentence-transformers Transformer -> Pooling(lasttoken) -> Normalize; config_sentence_transformers.json
  and mteb_v2_eval_prompts.json byte-identical at the three revisions; 1_Pooling/config.json differing only
  in word_embedding_dimension 640/1024/5376) and differ only in backbone (Gemma3TextModel for the 270m
  [all-full-attention layers] and the 27b [5:1 sliding_attention pattern, sliding_window 1024], Qwen3Model
  for the 0.6b) -- so one family.yaml carries the shared serve/client/reference blocks and the variants
  carry model, revision and their own citations. Served on stock vLLM v0.31.0 with `--runner pooling` only
  (convert auto -> embed for both backbones: the `*Model` suffix default and the sentence-transformers
  fallback in vllm/config/model.py; the unregistered Qwen3Model architecture normalizes to Qwen3ForCausalLM
  in the engine's registry, its lm_head tied), the pooler resolved from the checkpoints' own
  sentence-transformers metadata (last-token + L2 normalize, no --pooler-config), causal attention, dtype
  bfloat16 (gemma3_text refuses float16 at the tag), no plugin, no trust-remote-code, no template file. The
  query frame is the checkpoints' own `web_search_query` prompt, byte-pinned -- WITH the trailing space after
  "Query: " this checkpoint's prompt carries -- and documents are bare; the mteb_v2_eval_prompts.json
  per-task instructions are decision 33's `Dataset.task_instruction` (model-owned, per task; the field's
  product-side plumbing lands with workstream 10) and this family's target placement is the card's own
  "Instruct: <instruction>\nQuery: <text>" fold (decision 9: the card's usage wins over the generic Task:
  prefix). Budgets: client.max_tokens 32768 = the card's "Max Tokens" for every variant and its
  transformers snippet's max_length; the client cut reserves the frame and the appended post-processor anchor
  (the Gemma variants' eos id 1, the 0.6b's endoftext id 151643; the card-example query renders to 27 framed
  ids per variant). The 27b does NOT serve its 131072 max_position_embeddings: one forward of 131072 tokens
  would push the MLP activation (131072 x 21504 = 2.82e9 elements) over 2^31 -- the 32-bit element-index
  fault GPU-E1 established -- so the card's own 32768 is served. Reference: the card's own
  sentence-transformers usage (`reference.kind: sentence_transformers`; SentenceTransformer at the pinned
  revision, queries prompt_name="web_search_query", documents bare; render mode needs only huggingface-hub),
  with `over_cap_cut_differs` (the checkpoints ship no sentence_bert_config.json and no max_seq_length in
  modules.json, so sentence-transformers infers the Transformer module's max_seq_length as
  min(config.max_position_embeddings, tokenizer.model_max_length) -- 32768 for the 270m and the 0.6b,
  131072 for the 27b -- and the post-processor appends the anchor after truncation; the client cuts at
  32768 with anchors preserved).
  The request generator's `--reference-python` validation resolves a multi-variant family's recipe by its
  VARIANT id (`load_recipe` refuses a family directory that declares several variants) and reads the
  declared empty policies through the plain-dict client block, so the pairs manifest records what the recipe
  declares (`rcp_ndcg_test.observe.requests`); pairs files for the three variants are generated (24 rows
  each), and the T3 task matrix gains them under text embedders (retrieval, nanobeir/bright/trecdl). Status
  `unverified`.
- The pplx folded plugin registers a second architecture for the 19th recipe: the late checkpoint's
  `Qwen3_5Model` (absent from vLLM v0.31.0's registry) resolves to
  `rcp_ndcg_vllm.models.pplx.late.PplxLateMultiVectorModel`, a `ColQwen3_5Model` subclass that loads the
  checkpoint's separate `1_Dense/model.safetensors` head (one tensor, `linear.weight` [128, 1024]) into
  `custom_text_proj` with a shape check -- the stock weight discovery never reads a subdirectory file
  (default_loader.py globs the snapshot root) -- and marks the zero-initialised projection bias loaded; the
  checkpoint's `model_type qwen3_5` is native, so no config class registers and no remote code runs
  (`trust_remote_code: false`). The T3 task matrix (`rcp_ndcg_test.quality.TASK_MATRIX`) gains the recipe
  under visual documents (retrieval, vidore) and late interaction, text (nanobeir, bright); its pairs file
  `pairs/pplx-embed-v2-late-0.6b.jsonl` is generated (33 rows at MEDIA_SET_VERSION 3; the media rows record
  the client's skip-ids media refusal, the recipe's named no-verify path).
- `rcp_ndcg.eval.mteb.task_subsets(source)` reads a published suite file's `_SUBSETS` alias map (each subset's
  published task name, read as data; `{}` for the files that predate the task-name keys) -- the lookup
  `rcp_ndcg.eval.mteb.get_tasks` resolves its `names` through.
- **The media request set carries video and interleaved rows** (`rcp_ndcg_vllm.observe.media_set`,
  `MEDIA_SET_VERSION` 3; the text rows' sampling untouched): a recipe with video input and a declared policy
  (`client.max_videos` and `client.video_policy`) plans an MJPEG AVI clip per size (64x64 and 224x224), alone
  and with text, at the policy's declared frame count -- tiny RIFF containers written on CPU from PIL-drawn
  frames (three scenes, a moving bar; the product's `probe_video_header` reads the generated headers and the
  structural test decodes every JPEG frame back and pins the BITMAPINFOHEADER's 40 bytes and the stream
  header's rate and length), the codec/container mix the vLLM v0.31.0 default video
  backend decodes (OpenCV over bytes, `vllm/multimodal/video.py:202-249`, `video_decoders/opencv.py:70-76`).
  Every media recipe also plans a batch mixing a text-only and an image document, a query carrying an image
  where its `media_sides` allows query media and its client can encode a media query, and -- where its
  `max_images` admits them -- a text-image-text-image document (two images interleaved with text, in order)
  and a document with `max_images` images; over the capacity stays the `edge:too_many_images` bare probe.
  The pairs `media` entries gain `text` segments (a part sequence's text, standing where it stands). The
  manifest records `media:video`, `media:video:icon`/`:page`, `media:video+text`, `media:mixed_batch`,
  `media:query_image`, `media:interleaved` and `media:several_images` present or absent with the reason;
  pairs regenerated for the three media recipes (their text rows byte-identical).
- **The media gate gates video and interleaved order** (`rcp_ndcg_vllm.equivalence.media`): a video's
  declared frame count gates against the reference's `--mode media` (the sampling both sides declare), and
  its container's token count gates against the engine -- the stage probes the sent container's header (the
  product's `probe_video_header`) and counts it exactly (`content_media_tokens` under the client's declared
  policies), so an engine not pinned to the declared sampling (or whose count otherwise differs) fails the
  engine check, as the image path does. An interleaved row gates the given part order: the client's fit joins
  a side's text parts into the first one's position, and the stage compares that placement with the card's.
  The test stub engine counts a video container the way vLLM v0.31.0's video path counts it (sampled to the
  engine's declared `--media-io-kwargs` frame count, else its default 32, patchified in time under the
  emulated family's video budget, through the product's own `content_media_tokens`) and refuses over-limit
  image and video counts like the engine's per-prompt limits; the `fixture-vl-video` fixture (two images, one
  video per request, a 4-frame pinned sampling) carries the video and interleaved tests.

- **The media gate's fix round**: the recorder records a media side the role client refuses as a
  `client_refusal` record (`rcp_ndcg_vllm.record.refusal_exchange`; no status, nothing sent), so a refusal never
  ends a corpus step and loses its text rows (topk-embed-v1-small's image documents); the checkpoint's own files
  come through one reader (`rcp_ndcg_vllm.equivalence.checkpoint.checkpoint_file`, `checkpoint_pixel_budget`),
  where only an absent file (the Hub's answer, or the cache's record of it) falls through to the next source; a
  file neither cached nor askable is unknown, and that or any other read failure is unresolved; the
  media stage names a side the reference refuses (`reference_refused`, which the generator prunes) before a side
  the client did not send, and an image whose tokens either side left uncounted fails; the generator's media rows
  are `observe.media_set.planned_media_rows` (renamed from `media_rows`, which stays the pairs-file reader in
  `equivalence.media`). The test stub resizes with the product's own `smart_resize`, so on CPU only the
  reference comparison can catch a product resize bug.
- **Recipe follow-ups**: `qwen3-vl-embedding-2b` moves to the `messages` route with `add_generation_prompt: true`
  and the pinned `qwen3_vl` image policy (images and video now travel; its notes say so); `qwen3-vl-reranker-2b`
  declares the same pinned `qwen3_vl` image policy (its images were refused) and its reference reads a
  whitespace-only query as the card does (verbatim); both references, and `topk-embed-v1-small`'s, gain
  `--mode media` (the cards' own image resize; NOTICE attributes the restated `smart_resize` of qwen-vl-utils
  0.0.14 and of transformers); `jina-reranker-v3` declares `document_max_tokens: 2048` (the checkpoint's
  per-document cap); `jina-embeddings-v5-text-small`'s notes no longer describe the last_content gap lane H
  closed. The `qwen3-vl-reranker-2b` corpus is re-keyed (its new client media inputs shape no recorded
  exchange); the `qwen3-vl-embedding-2b` and `jina-reranker-v3` stale declarations name the inputs that moved.
- **The media request set is generated and recorded** (`rcp_ndcg_vllm.observe.media_set`, `MEDIA_SET_VERSION` 1,
  versioned apart from the text sampling so no recipe's text rows move): a recipe with image input plans one
  pairs row per image size bucket of OBSERVATIONS-SPEC section 1 (tiny, icon, A4 at 72/150/300 dpi, a 16:9
  slide, a tall receipt, an extreme aspect ratio; deterministic PNGs inline) and a captioned page, after its
  text rows; the corpus plan sends the edges bare (`edge:too_many_images`: `max_images + 1` images in one
  item; `edge:corrupt_image`) and records `media:video` absent with the reason (the generator writes no video
  container). The `BLOCKED` media strata are gone. The generator's validation runs the media stage offline
  (`validation.media_check` in `pairs/manifest.json`; a media failure is recorded, never pruned), stage 1's
  red rows map back to the text rows' positions, and the recorder sends each media row with its media, one
  item (a reranker's pair) per request.
- **The media stage of the equivalence harness** (`rcp_ndcg_vllm.equivalence.media.stage_media`, run by
  `equivalence.run` beside stages 1 and 2 for a recipe with image or video input): stages 1 and 2 are text-only, so
  nothing proved that a served vision-language recipe shows its model the images its reference sees. For every
  pairs row that carries `media` (inline `MediaRef` entries with their `kind`; a side's content is its media in
  order, then its text) the stage sends each media side through the recipe's role client (the product's media
  preparation and fit) and reads what crossed the wire -- the parts in order, each image's prepared geometry
  decoded from the sent bytes, each video's frames, the tokens the client counted (`content_media_tokens` under
  the client's effective policies) -- against the reference's new `--mode media` (`REFERENCE_MODES`), and with an
  engine it sends each media request again without its media: the difference of the two `usage.prompt_tokens`
  is the engine's own media count, which must equal the client's. Every image item gates exactly (count,
  placement, geometry, tokens); a container's tokens are reported. A media recipe whose pairs carry no media row
  fails the stage; stages 1 and 2 compare the text rows only (`media_rows` in stage 1's report). Negative control
  (f) unpins the nested `images_kwargs` pin (or a flat one) and is caught by the engine count; it is
  inapplicable, said why, where the pin lies inside the checkpoint's own image budget read at the pinned
  revision (`equivalence.checkpoint.checkpoint_pixel_budget`: `preprocessor_config.json`, else
  `processor_config.json`), which an unpinned engine applies and which keeps every image the client prepared
  -- `qwen3-vl-reranker-2b` (4096..1310720 inside its 4095..1310720) and `topk-embed-v1-small`; served for
  `qwen3-vl-embedding-2b` (its 1843200 px ceiling leaves its checkpoint's 1310720) -- and `unresolved`, a
  wave blocker, when that budget cannot be read. The test stub engine resizes images as the engine's processor does
  (`--model-image-factor`, `--model-image-pixels`, `--mm-processor-kwargs`), and the `fixture-vl-embed`
  fixture is a vision embedder pinned below its family's stock floor.
- **A recipe that declares a key twice is refused** (`rcp_ndcg_vllm.load_recipe`): YAML keeps the last of two equal
  keys silently, so a recipe declaring a field twice served whichever came last. The loader raises
  `RecipeError` naming the key and its line; the `fixture-rerank-noisy` fixture declared `use_activation` twice
  (both `true`) and now declares it once.
- **Change handling reads corpora through the one reader**: `rcp_ndcg_vllm.changes` (the committed-corpora
  selection and staleness) reads each manifest through `rcp_ndcg.testing.corpus.load_corpus` and refuses a
  corpus whose hashes do not hold (`HarnessError` naming the mismatches) instead of parsing `manifest.json`
  itself; the conformance wiring (`tests/_engines.py`) reads through the same reader. The two re-record
  selections state their scopes: `changes changed` compares the corpora committed in the repository, the
  wave runner's `--changed-since <wave.json>` a previous wave's index (`observe.corpus.changed_since`).
- **`rcp_ndcg.runs.execution.stage_run(pipeline)`**: the one staging of a run directory before a job runs it (the
  layout, the resolved `run.yaml`, the manifest in status `submitted`). `submit_run` calls it, and
  `rcp_ndcg_vllm.e2e.stage_run_dir` -- the GPU end-to-end driver, which runs the job itself -- now calls it
  instead of the pipeline's private `run.yaml` writer.
- **Re-keyed corpora name the manifest they were re-keyed from**: each `recipe.rekeyed` entry carries
  `from_manifest_sha256` and `from_manifest_file_sha256` beside `from_behaviour_fingerprint`, the link from the
  repository subset to the full corpus it was cut from (`qwen3-reranker-8b`, `qwen3-vl-reranker-2b`; their
  manifest digests and the corpus indexes recomputed). The conformance suite requires both on every entry.
- **The stub engine speaks vLLM's chat path, and stage 2 runs on the messages route**: `rcp-ndcg-vllm`'s test stub
  engine refused `messages` bodies, so stage 2 on the `messages` route was untested. It now frames each
  conversation as vLLM v0.31.0's chat path does (parts handed to the served chat template as their modality,
  the request's `add_generation_prompt`), resizes images as the engine's processor does under the served pixel
  pin (else the emulated checkpoint's default budget), refuses what the engine refuses (more images than
  `--limit-mm-per-prompt`, an undecodable image, a video container it cannot decode) and reports
  `usage.prompt_tokens` on every chat-shaped and `/rerank` reply; a stage-2 test passes a template that frames
  once and fails one that frames twice.
- **The harness reads the checkpoint's own chat template** (`rcp_ndcg_vllm.equivalence.stages.checkpoint_chat_template`,
  `CHECKPOINT_TEMPLATE_FILES`): a `messages` recipe without `serve.chat_template` reported its
  `template_render_check` as `not_run`, so the frame the engine renders there was never checked. The check now
  reads the checkpoint's own template at the pinned revision, in the order the engine resolves it
  (`chat_template.jinja`, `chat_template.json`, `tokenizer_config.json`; vllm/renderers/hf.py:263-300), from the
  Hub cache (a pinned revision answers without a request) or the Hub, renders it over every captured
  conversation and names the file and its SHA-256 in the report (`template`, `template_sha256`). A template
  that cannot be read fails the check (`status: unresolved`), never passes.
- **A declared generation prompt on the messages route** (`EmbeddingEndpoint.add_generation_prompt`,
  `EmbedRequest.add_generation_prompt`): vLLM v0.31.0's chat routes default `add_generation_prompt` to false
  (`vllm/entrypoints/pooling/base/protocol.py:230-237`), so a checkpoint whose frame ends with the chat
  template's assistant header (Qwen3-VL-Embedding's) rendered without it on the `messages` route. Declaring
  `add_generation_prompt: true` sends the flag with every `messages` request; it is content (it enters the
  config's identity and the recipe fingerprint as a request field), refused on any other request shape and
  on `PoolingEndpoint` (its media lowering sends no such field), and `false` -- the engine's default -- is
  stored as `None`, so no identity re-keys. The equivalence harness's messages template check renders every
  captured conversation with the flag that request carried, its parts as vLLM hands them to the template
  (`rcp_ndcg_vllm.equivalence.stages.engine_conversation`: `image_url` as `{"type": "image"}`, `video_url` as
  `{"type": "video"}`).
- **Two public accessors of `rcp-ndcg-vllm`, and the release rule for stale corpora**:
  `rcp_ndcg_vllm.fingerprint.stored_tokenizer(spec)` returns a registered tokenizer store's verified
  `tokenizer.json` bytes and SHA-256 (the golden replay materialises the recipe's tokenizer with it), and
  `rcp_ndcg_vllm.record.bare_exchange` records one bare probe as a captured exchange (the wave runner's
  readiness edge uses it); no caller imports a private helper for either. With `RCP_NDCG_RELEASE=1` the
  conformance suite requires `tests/conformance/stale.json` empty, as it requires the waiver file empty; the
  re-key of a corpus whose fingerprint moved by metadata only is a documented procedure
  (`docs/how-to/use-verified-fake-engines.md`), never the way to empty the stale list.
- **An image pixel budget the engine is pinned to** (`ImagePolicy.engine_pixel_pinning`): a budget outside the
  processor family's stock range was refused even when the engine was pinned to it (Qwen3-VL-Embedding's card
  budget, 4096..1843200 px, below `qwen3_vl`'s stock 65536 px floor), so such a client could neither resize nor
  count an image. Declaring `engine_pixel_pinning: true` admits the budget, and the policy's re-resize check
  then uses the pinned budget (the engine's own) instead of the stock range; the descriptor says `pinned`.
  `false` is stored as `None`, so no existing policy re-keys; a declared pinning is a different instrument (it
  enters the family key). A policy without a budget refuses the declaration. The media fit's shrink step
  carries the declared policy's every field (the pinning included) into its minimum policy, so a pinned budget
  below the stock floor shrinks to its own minimum; a shrink the policy cannot express is a typed `DataError`
  naming the budget. `rcp_ndcg_vllm`'s recipe
  validator checks both sides: a pinned client needs `serve.mm_processor_kwargs.images_kwargs` with both
  numbers, and every pixel number serve pins (nested or flat, `min_pixels`/`max_pixels` or the HF processor's
  `size: {shortest_edge, longest_edge}`) must equal the client's -- a serve pin beside a client that declares
  no pixel budget is refused too.
- **A per-document cap beside the pair budget** (`RerankEndpoint.document_max_tokens`,
  `TextBudget.document_max_tokens`): a reranker whose checkpoint cuts each document itself (jina-reranker-v3
  reads 2048 document tokens beside its 512-token query share) declares it, mirroring `query_max_tokens`.
  `fit` cuts every pair's document over it to it on the content span only -- also in a pair the budget would
  take whole -- re-attaches the frame (the anchors survive) and records the cut under the document's position
  with `cause: document_share` (`budget_cut` when the pair still overflowed and the budget cut it further). The
  cap is content (it enters the config's and the budget's identity; unset, both are unchanged), must be below
  `max_tokens` on the rerank config (at or over it the pair budget always binds first), and is refused
  beside `on_overflow: chunk` and on a hosted profile without a tokenizer (inert). The equivalence harness's
  rerank audit holds every captured document span to it, as it holds the query span to its share.
- **A role client records, per input row, what it changed** (`rcp_ndcg.data.preprocess.ProcessingRecord`,
  `RoleClient.processing`): every role client's preparation emits one record for each input row it changed
  before sending it -- none for a row sent as given -- naming each change by its mechanism
  (`CHANGE_MECHANISMS`: `empty_doc`, `media_resize`, `media_drop`, `document_share`, `query_share`,
  `budget_cut`) with the uncut and the kept request totals (frame, specials, content and media) and the
  shape's budget; `processing_records` builds them from the census rows the cut wrote and the media fit's
  and the empty-document policy's decisions, so nothing is measured twice. The policy's own image resize is
  the declared instrument (R20), not a change. The text census rows name the same facts:
  `TextCutRecord.cause` (`budget_cut`, `query_share` or `document_share`; `CutCause` / `CUT_CAUSES`),
  `original_request_tokens` and `kept_request_tokens`, recorded by `fit` and by the rerank client's
  shared-query settlement, also on the census sink's rows (`TextTruncationCensus.record` takes them). They are
  `None` (and absent from `as_row()`) on the judge's rows and on a vendor's budget row, so those rows are
  unchanged. `original_tokens` counts the content alone: a request whose frame pushed it over the budget has
  a content count under it, so whether a role client changed an input is read from the record. The pooling
  client's census rows now name each input's original position (an omitted empty document no longer shifts
  a later one's id), as the rerank client's do.
- **`rcp-ndcg-vllm` recipes: `reference.known_deviations` accepts `over_cap_cut_differs`** beside
  `anchor_drop_over_cap`: a reference that keeps the anchors but cuts over-cap content its own way (a joint
  `longest_first` truncation where the client settles the query at its share) declares it, and the harness
  reports those inputs outside the gates. A reference stays the paper's or the model card's; it never
  copies the client's cut to make an over-cap row pass. `schema/recipe.schema.json` carries the new value,
  and the new read-only property `ReferenceSpec.over_cap_deviation` names the declared over-cap deviation
  (or `None`), which the harness's stages read.
- **The adapter seam's contract is declared and checked** (`rcp_ndcg.inference.adapters.base`): `AdapterBase`
  carries the credential and capability ClassVars (`HOSTED`, `API_KEY_ENV`, `KEY_REQUIRED`, `AUTH_HEADER`,
  `DEFAULT_BASE_URL`, `MAX_BATCH`, `SUPPORTS_DIMENSIONS`, `ENCODING_FORMAT`, `REQUEST_SHAPES`) with declared
  defaults and the
  constructor convention (an adapter is built with its role config). `register_adapter` and the
  `rcp_ndcg.adapters` entry-point loader refuse a class without the three members (`calls`, `interpret`,
  `usage`) or the five credential facts, and a class that does not declare `HOSTED` itself (the base's
  `False` would silently serve a hosted wire as an engine) -- previously duck-typed with defaults that could
  be wrong, and a
  missing member failed only at the first request. `rcp_ndcg.testing.adapter_contract` is the contract-test
  kit the unified-inference design promised: name/role, members, facts, construction, and a recorded round's
  alignment and usage, as one listed failure set.
- `AuthProfile.homes` and `AuthProfile.applies_to(url)` (`rcp_ndcg.inference.transport`): the URLs a
  profile's key variables belong to, and whether they may authenticate a request to a replica URL -- exactly
  one of them, a trailing slash aside (see Security).
- `rcp_ndcg.support.urls.safe_url` (public): the form of a URL that may reach a log, an error or a record
  -- userinfo, query and fragment stripped, the host and path as written. The one redactor: the inference
  layer's logs, errors and engine records and the storage cache's messages (which lower-cased the bucket
  name and broke an IPv6 host) all call it. `EngineInfo.url` validates itself through it, so a run manifest
  never persists credentials embedded in a URL.
- `rcp_ndcg.data.preprocess`: `fixed_overhead` (the one home of a request frame's fixed token cost, shared by
  `fit` and the role clients' media allowances), `rendered_request` and `rendered_pair_tokens` (the one home
  of the assembled render, so a client's budget check measures what `fit` verified), and
  `TextBudget.shape_max_tokens(shape)` (the one home of a shape's budget: `query_max_tokens` on the embedding
  roles' query shape, else `max_tokens`).
- `rcp_ndcg.data.prepare`: `PreparedRequest.content_tokens` (per-content media token counts) and
  `PreparedRequest.per_content()` (each content's slice of a prepared request, in one pass -- the role
  clients prepare a request once and fit each wire request's slice, never re-inlining prepared bytes; a
  corpus encode is one request, so the slicing is linear in it); `MediaCensus.recorded()` (the public read the tests use instead of private
  state).
- `rcp_ndcg.data.media`: `data_uri` (the one builder of every inline `data:` URI the package writes) and
  `DEFAULT_IMAGE_MIME` beside it; `rcp_ndcg.data.prepare.DEFAULT_IMAGE_MIME` is re-exported from the new home.
- `rcp_ndcg.inference`: `RoleClient.usage` (the sender's accounting, as the judge's), `EmbeddingClient.probe`
  (the transport's replica probe; the embed client sends no media probe request, so it runs no engine media
  check), and `VllmPooling.media_probe_baseline` (the media probe's no-media baseline, the same `messages`
  shape).
- **The role clients expose their text budget**: `EmbeddingClient`, `PoolingClient` and `RerankClient` gain the
  read-only `text_budget` (the `TextBudget` the client fits every request to, as built from its config; `None`
  without `max_tokens`), so harnesses and case loaders read the client's budget instead of rebuilding it.
- **T4 end to end: the run scenarios the GPU validation drives inside the pod** (`rcp-ndcg-vllm`): the
  scenario configs `rcp-ndcg-vllm/scenarios/*.yaml` (schema `schema/scenario.schema.json`),
  the stage `python -m rcp_ndcg_test.e2e`, the entry `rcp-ndcg-test/src/rcp_ndcg_test/jobs/e2e.sh` and the submission
  flag `submit.sh --script e2e` (beside `bootstrap` and `wave0`; the wave list names scenario ids). One
  scenario materializes a `RunConfig`, renders its phased job script with the SLURM renderer
  (`container_runtime: none`) and runs it in the pod: the coordinator through `install_argv` from the
  staged wheelhouse, an `srun` stand-in, the process-boundary probe, and the T0 judge smoke whose
  verdict picks the scenario's judge or its `fallback` (the Flash-Next NVFP4 candidate to its FP8
  release). Shipped scenarios: `text-four-phases` (run twice; identical identities and outputs,
  judged values are never compared, window counts and families only: judgements may differ at temperature > 0), `outage` (the
  judge killed mid-tournament: parks and recovers; the `wait_on_outage_s` expiry fails with
  `BackendUnavailableError`; a resume finishes), `identity` (the same run on new ports: nothing
  recomputes) and `vidore` (ViDoRe v3 page images). Offline counterparts in the root suite: the golden
  rendered script (harness tests), the four-phase supervision re-run with the verified fake engines as
  its encoder and reranker engines (`rcp_ndcg.testing.engines` emulators of `qwen3-embedding-0.6b` and
  `qwen3-reranker-8b` served over HTTP, each coordinator's recorded request answered `replayed`; the judge
  phase keeps the supervision stub, no judge corpus being recorded;
  `tests/runners/test_supervision_replay.py`) and the observed outage behaviour as a transport
  test (`tests/inference/test_outage_observed.py`).
- **`rcp_ndcg.testing.corpus` is the whole format seam**: `ObservationCorpus.nondeterminism` exposes the
  corpus's `nondeterminism.json` (parsed; `None` when absent; an unreadable one is a `DataError`), and
  `rcp_ndcg.testing.engines.corpus_tolerance` reads it there; the append-only `verification.jsonl` lives
  here (`VERIFICATION_FILE`, `VERIFICATION_SCHEMA`, `append_verification` -- which refuses a record of
  another schema -- and `verification_records`; moved from `rcp_ndcg.testing.engines`) and no integrity
  hash covers it (`write_subset_index` and the collector's `write_corpus` leave it out); the one raw-body
  normaliser `normalise_raw` sits beside `normalise_body` (moved from `rcp_ndcg.testing.engines`); the
  section-4 provenance check is public (`PROVENANCE_KEYS`, `missing_provenance`; moved from
  `rcp_ndcg_vllm.observe.corpus`, where it was private); and `credential_findings` also names an
  `X-Api-Key` header, a cookie and a secret-valued field (`api_key`, `client_secret`, `password`,
  `access_token`, ... with a string value), never a tokenizer vocabulary's integer ids.
- **`rcp_ndcg.testing.engines`: the verified fake engines** (GPU-VALIDATION items 2-4, 7 and 8). One
  emulator per (engine, version, recipe, behaviour fingerprint), selected as
  `fake://vllm-0.31.0/<recipe>` (`rcp_ndcg.inference.fake` routes engine-version hosts there); the
  protocol is emulated (routes, request validation, error bodies, result ordering and framing, usage
  counts and token counting with the recipe's real tokenizer files) and the model outputs are
  replayed for observed inputs -- a declared deterministic surrogate for unseen ones (`surrogate_vector`
  and `surrogate_matrix` draw one SHAKE-256 stream per vector, `surrogate_scores` one draw per score;
  no value is pinned), marked
  `replayed`/`surrogate`/`mixed` in `x-rcp-ndcg-emulator-source`, recorded per reply in `answer_log`.
  The emulators read corpora through the format's one reader, `rcp_ndcg.testing.corpus`
  (`exchanges_of` views its records, `corpus_tolerance` takes the tolerance the corpus's
  `nondeterminism.json` derived from same-request repetitions -- `None` when none was measured, so a
  replay is compared exactly, never with an invented tolerance -- and each bound applies jointly);
  `find_corpora` resolves corpora by scanning manifests, `behaviour_diff` writes the per-input delta
  report, and
  the registry (`registry`, `transport_for`, `split_engine_host` -- the one engine-host pattern
  `rcp_ndcg.inference.fake.RE_ENGINE_URL` routes) resolves by (engine, version, fingerprint) with the
  `rcp_ndcg.emulators` entry-point group for out-of-tree emulators. An emulator refuses an engine
  version or recipe revision it was not verified against; the conformance suite
  (`tests/conformance/`) replays every recorded exchange and the staleness check names the changed
  fingerprint inputs (waivers: `tests/conformance/waivers.json`, empty at release).
- **`rcp_ndcg_vllm.fingerprint`: the recipe behaviour fingerprint** (GPU-VALIDATION item 8):
  `behaviour_fingerprint(recipe)` (rule `rcp-fp/3`: the SHA-256 of the checkpoint id and revision, the
  serve block, the template file's bytes, the tokenizer's SHA-256 and exactly the client fields that
  change the request bytes -- `CLIENT_FIELDS` classifies every client config field, so request packing
  (`batch_size`) and the media caps are in, client-side post-processing of the reply (`normalize`,
  `aggregation`, `dim`, `mrl_dim`, `document_skip_token_ids`, `outputs`) is out, and an unclassified
  field is refused) and `fingerprint_inputs(recipe)` (every input named, for staleness messages), with
  `fingerprint_changes` and the one tokenizer resolution (`load_recipe_tokenizer`,
  `tokenizer_sha256`, `use_tokenizer_store` -- vendored `tokenizer.json` copies whose SHA-256 is
  verified on every read).
- **`rcp_ndcg_vllm.changes`: change handling** (OBSERVATIONS-SPEC section 7): `recipe_state` (one
  recipe's fingerprint recomputed and compared with every committed corpus of it, found by scanning
  manifests with `rcp_ndcg.testing.engines.find_corpora`), `resolve_corpus` (the corpus of the current
  fingerprint, or `StaleCorpusError` naming the changed inputs per recorded corpus), `waiver_covers`
  (a dated, reasoned, unexpired staleness waiver), `changed_recipes` (the re-record-changed-only
  selection: `unchanged`/`changed`/`new`/`unloadable` per recipe, the changed inputs named) and
  `behaviour_report` (the behaviour diff of two corpora of one recipe), as functions and
  `python -m rcp_ndcg_vllm.changes` (`changed` and `diff`).
- **The provisional corpora after the families' recipe changes**: three corpora are current as recorded
  (`octen-embedding-8b`, `qwen3-embedding-0.6b`, `zembed-1-embedding`); two are re-keyed to the current
  fingerprint because only inputs that shape none of their recorded exchanges moved (`qwen3-reranker-8b`:
  `client.empty_query`, and `serve.pooler_config` restating vLLM's `use_activation` default;
  `qwen3-vl-reranker-2b`: the media inputs of a text-only corpus and the same pooler default), each
  manifest naming the move in `recipe.rekeyed`; seven are declared stale for re-recording in
  `tests/conformance/stale.json` (`jina-embeddings-v5-text-small`, `jina-reranker-v3`,
  `qwen3-reranker-0.6b`, `qwen3-reranker-4b`, `qwen3-vl-embedding-2b`, `zerank-1-small-reranker`,
  `zerank-2-reranker`): the replays skip them and each must fail the staleness gate naming exactly the
  declared inputs. The NanoBEIR golden's rerank view replays `qwen3-reranker-8b` (same recorded texts;
  the pinned metrics are unchanged).
- **The corpora under `tests/contract/engines/`**: the provisional shakedown corpus (12 recipes, 48
  exchanges) as repository subsets in the observation-corpus format (`records.jsonl.gz`, the manifest
  with a `provisional` statement -- valid to build and test the emulators, not release evidence --,
  `nondeterminism.json`, `index.json`; the manifest hashes in the repository's corpus index; a shared
  vendored tokenizer store), the append-only verification record beside each (`verification.jsonl`),
  and the
  golden replays (`tests/e2e/test_golden_replay.py`), **regression pins** labelled as such
  (`kind: regression-pin`: computed by this code from the provisional corpus, which recorded no subset
  run -- not independent GPU-run numbers; the RC0 subset corpus replaces them): a NanoBEIR-shaped mini
  through the retrieval view (two documents, the model's order reversed against the gains, and a
  moved document vector moves both metrics) and the rerank view, and the ViDoRe-shaped rerank view.
  The ViDoRe retrieval view is waived -- the corpus observes no page image -- by a tripwire on the
  corpus content. Every input is observed; the run fails on any surrogate answer, and with its
  observations deleted.
- **Conformance details**: the model layer replays an output only for the behaviour-shaping context it
  was observed under (`FIELD_CLASSES` over vLLM v0.31.0's `ROUTE_FIELDS`: the engine prompt plus
  `use_activation`, `dimensions`, `add_special_tokens` and `task`); an unobserved context answers the
  marked surrogate, a field the emulator does not model (`instruction`, `truncate_prompt_tokens`, ...)
  a 400 marked `refused-unmodelled`, an undeclared field is ignored as the engine ignores it, and a
  corpus whose one key holds different outputs is refused. `compare_exchange` checks a
  reply as the transport reads it: the status, the recorded headers that matter (content type, server,
  the bytes framing's `metadata`), the body, and its raw bytes where the corpus recorded them
  (`normalise_raw` masks the volatile ids and stamps); an undecodable body
  is a named difference, and a recorded body the reader cannot decode (a base64 token matrix: vLLM
  sends no shape) is refused naming the record. Every reply says whether a recording covers its route
  (`x-rcp-ndcg-emulator-route`, `EMULATED_ROUTES`, `VllmEmulator.unobserved_routes`, listed in the
  verification record); the unobserved `/pooling` and `/tokenize` follow vLLM v0.31.0's source
  (`PoolingResponse`, the bytes framing's `metadata`, `TokenizeResponse`; `embed_dtype` `float32`,
  `endianness` `native` by default). `RE_ENGINE_URL` is exported from `rcp_ndcg.inference.fake`
  (`rcp_ndcg.testing.engines` is pinned as public API).
- **`rcp_ndcg.testing.corpus`: the observation-corpus format and its one reader** (new module; `rcp_ndcg.testing`
  is now a package, its names unchanged). `load_corpus` reads a GPU wave's corpus directory -- the full corpus's
  `records.jsonl` or a repository subset's `records.jsonl.gz` with its `index.json` -- and migrates records of an
  older `RECORD_SCHEMA` through `register_record_migration` (a record from a newer collector, or one without a
  migration path, is a `DataError`); `integrity_mismatches` checks every hashed file and the manifest's own digest
  (`manifest_digest`), `write_subset_index` writes a subset's index, `normalise_body` strips the volatile reply
  fields (`NORMALISATION_VERSION` 1: request ids and `created` timestamps) and `credential_findings` names the
  credential shapes a text carries. The collector in `rcp-ndcg-vllm` writes this format; the verified fake
  engines read it here. A frozen schema-1 sample (`tests/observation_corpus_v1/`) pins that it stays readable.
- **The observation corpus, the T3 quality stage and the negative controls (`rcp-ndcg-vllm`)**:
  `rcp_ndcg_vllm.record.record_corpus` records a corpus in the product's format (`rcp_ndcg.testing.corpus`):
  the request plan (`observe.requests.corpus_plan`: the pairs rows, the over-length ladder and the long content
  kinds through the client and uncut, the wire variants, the protocol edges) twice in one engine process and
  once after a restart, `/tokenize` of the exact prompts, and the section-4 provenance (`observe.provenance`);
  `observe.corpus` keys it by engine version and `rcp_ndcg_vllm.fingerprint.behaviour_fingerprint`, measures the
  non-determinism over true repetitions, runs the acceptance checks and cuts the repository subset
  (`python -m rcp_ndcg_vllm.observe.corpus verify|subset`).  `quality.py` is the T3 stage: the served path
  through the product's CLI, the `mteb` reference, RCP-nDCG@10 and qrel-nDCG@10 gated vs the reference and the
  paper's numbers, `QUALITY.md`, and a recording proxy for the golden-replay corpus.  `observe.controls` derives
  the negative controls (a)-(f) as real vLLM breakages (recipe variants, or request-body patches through
  `equivalence.wire.patched_wire`).  `run_wave` gains `--record-corpus`, `--changed-since`, `--quality`
  (`--paper-numbers`) and `--controls`; a passing control fails the recipe.  The format is documented in
  `rcp-ndcg-vllm/schema/observation-corpus.md`.
- **`rcp_ndcg_vllm.observe` (the `rcp-ndcg-vllm` distribution)**: the deterministic observation request
  generator -- `GENERATOR_VERSION`, `SEED`, `PINNED_DATASET_COMMITS`, the synthetic adversarial set stored as
  text -- writing one stage-2 pairs file per recipe in the harness's pairs format plus `pairs/manifest.json`
  (per-row provenance, stratum presence records, file hashes, excluded source ids, the recipes that could not
  load with their error, and what stage-1 validation ran). `python -m rcp_ndcg_vllm.observe.requests` generates
  it under a stall watchdog (`faulthandler` to stderr every 60 s); runs merge into the manifest per recipe (a
  recipe a run touched replaces its file, skipped and pruned entries), so one bounded invocation per recipe
  composes. Stage-1 validation probes the offline fake with a `/pooling` recipe's reply width bounded (`dim` 8
  without an `mrl_dim`; every request field unchanged); a
  `/pooling` recipe whose over-length samples exceed 32768 tokens records its render check as blocked (the
  full-budget stage 1 runs against the engine); a recipe whose validation pruned every row is a skipped
  recipe with the first failure named, never an empty pairs file. The committed `pairs/` hold all 18
  recipes, generated in render mode on CPU (no model weights); the manifest records one render check as
  blocked (`pplx-embed-v2-context-9b-preview`: the offline probe bound) and, for the media recipes, the media
  check (`topk-embed-v1-small`'s failed: its client refuses image documents under its skip ids).
- **`FitDiagnostics` counts the fit's skips**: `skipped_observations` and `skipped_queries` (integers, default 0)
  are new fields, so `schemas/calibration-summary.v1.json` carries them. A tournament-mode fit counts the rubric
  placements whose document has no Bradley-Terry theta, and the queries absent from `bt_scores`, instead of
  dropping them unrecorded.
- **The plugin endpoint configs require `api`**: `PluginEmbedding`, `PluginPooling` and `PluginReranker`
  (constructed directly, a public name) no longer inherit their role's shipped `api` default -- a config
  without `api` does not build (a "plugin" was silently built around a shipped wire), and a shipped name stays
  refused in the class. The exported config schemas carry the required field.
- **Per-shape budgets for the embedding roles**: `EmbeddingEndpoint` and `PoolingEndpoint` take
  `query_max_tokens` (CONTENT) -- the `query` shape's WHOLE budget there (a late-interaction or asymmetric
  embedder caps its two sides differently, e.g. topk-embed-v1-small: query 1024, document 8192), while
  `max_tokens` keeps capping the `document` shape; `fit` honours it, and every `text_budget` census row names
  its shape's budget in the new `budget_tokens` field. `query_max_tokens` above `max_tokens` is refused (on a
  `RerankEndpoint` the field keeps its pair-share meaning, and its existing at-or-over refusal stands).
- **`TemplateSpec.anchor` gains `last_content`** (jina-embeddings-v5): the model reads the last kept content
  token -- no fixed tail exists, the shape may end on a content span (which `last` refuses), and the fixed
  segments (a head marker) stay reserved and audited.
- **Declared content normalisation**: `TemplateSpec.normalize` (CONTENT), per request shape, the ops
  `"strip"` and `"lowercase"` in order (a tuple for every declared shape, or a mapping naming every declared
  shape); `fit` applies it to the content spans before measuring (through the new `TemplateSpec.normalisers`
  and `TemplateSpec.normalize_text`), so the reference and the engine see the same text. The census rows keep
  the raw input on their original side. The new `ContentNormalizer` type is exported from
  `rcp_ndcg.data.templates`.
- **Late-interaction skip ids**: `PoolingEndpoint.document_skip_token_ids` (CONTENT, default `()`): the
  pooling client drops document vectors at the positions whose token id is listed -- the ids it sent,
  tokenised from the fitted render -- checks the returned vector count against them (a mismatch is a typed
  `ProviderError`, never a silent misalignment), keeps query vectors whole, and refuses a media batch under
  skip ids (its positions are the server's chat-template render).
- **The client-side Matryoshka cut**: `PoolingEndpoint.mrl_dim` (CONTENT, below `dim`) slices the model's
  token vectors to the MRL output size and renormalises -- cut-then-renormalise, the card's order, because
  `/pooling` refuses per-request `dimensions`.
- **Per-chunk multi-output models**: `PoolingEndpoint.outputs` (CONTENT): `"per_chunk"` declares a model that
  answers several outputs per input (one slice of chunk vectors per input), so the pooling adapter's
  one-vector-per-prompt-token usage cross-check is skipped; the `PoolRequest` it rides on carries the new
  `outputs` field.
- **Per-side media**: every role config takes `media_sides` (CONTENT, default both sides): media on a side it
  does not name is refused with a typed error naming the field, before the media is fetched; a config that
  declares media fields with no allowed side is refused.
- **`empty_query` on the rerank role** (CONTENT): `refuse` (the default) refuses an empty query with a typed
  error naming the query id; `send` keeps today's empty string. `rerank`/`arerank` take the new
  `query_id` keyword ("" names it `<unnamed>`; `arerank_many` passes each example's id).
- **`request_shape` is implemented end to end on the embedding and pooling roles**: `openai_embeddings`
  sends the chat-style embeddings input (`messages`: one user message per item, content parts, image parts
  and video parts -- sampled frames as image parts, a `video_url` container per the role's `video_policy`),
  and both clients send `token_ids` (the ids their fit tokenised; vLLM accepts token-id prompts), refused
  without a tokenizer. A shape a wire does not implement is refused at construction (the adapters declare
  their shapes), and a rerank config that declares a non-text shape is refused at the config (the rerank
  wires send rendered text today); a media item on a text or token-ids route is refused by the adapter.
  `EmbedRequest`/`PoolRequest` carry `request_shape` and `token_ids`.
- **Template specials resolve their names exactly** (whitespace included, so a token named `"[Q] "` is
  writable), and an unknown name's hint names the nearest ones before the full list.
- A prompt prefix has one home: `query_prompt`/`doc_prompt` beside a `template` is refused with a
  `ConfigError` naming the template segment to use instead (the fields stay for template-less configs).
- **The results-export seam** (owner decision 40): a versioned `rcp-ndcg.result-record.v1` record
  (`rcp_ndcg.results`: `ResultRecord`, `ResultSubject`, `ResultDataset`, `ResultMetric`, `ResultArtifact`),
  one row per system x dataset x metric x cutoff, carrying the run identity, the dataset revision, the recipe
  or model identity the run names (the judge's and the candidates') and the scoring protocol --
  `dataset.protocol` is the preset name and
  `dataset.protocol_spec` the full `Protocol` (qrel gain, tie rule, pool restriction, rounding), so an
  importer can state another convention and two records differing only in protocol never compare equal
  (`record_id` digests the protocol). The record's JSON Schema is exported as
  `schemas/result-record.v1.json`. Sinks are the `rcp_ndcg.results` entry-point group (the same seam as
  `rcp_ndcg.readers`/`writers`/`runners`), with the built-ins `jsonl` (one record per line), `parquet` (one
  row per metric row) and `null`, and the shared contract check
  `rcp_ndcg.testing.results_conformance`. `records_from_report` and `records_from_run` build records from an
  `EvalReport` or a run directory; the new `rcp-ndcg results` group lists the sinks (`results sinks`) and
  exports (`results export --run DIR [--report FILE] --sink NAME --out URI`, `--system`, `--include-reference`).
  The run manifest's `DatasetRef` records the subset, split and task the data was read at, and a report's
  `inputs` carry them too, so an exported record states the real provenance rather than the `test` convention.
  The record schema is a compatibility contract: additive fields only within `v1`, a change to an existing
  field's meaning or type a new schema id ([the compatibility page](docs/reference/results-record.md)).
- **Count-nDCG has its product path** (scoring-chain review F3): `rcp_ndcg.calibration.count_gains(judgements)`
  is the one derivation of the rubric-only gains (per window, per criterion, through `count_gain`), keyed as
  `Calibration.gains()` is; `evaluate(..., count_gains=...)` takes it, and `rcp-ndcg eval score --metrics
  count_ndcg --judgements STORE` (repeatable) is the command-line route. `ReportInputs` gains `judgements`, so
  `eval explain --report` re-scores a saved Count-nDCG report, and the `eval_score` MCP tool takes `judgements`
  too. The hint for missing count gains names the rubric windows and this derivation instead of the tournament
  store.
- **`rcp_ndcg.errors.WarningCode` gains `NO_VALID_TOURNAMENT_EVIDENCE`** (review F2): a document whose
  tournament windows are all invalid carries no comparison, and the fit says so instead of presenting the mean
  ability as judged. `CalibrationCoverage` gains `no_tournament_evidence_documents` (the
  `"<dataset>||<query_id>/<doc_id>"` list, in `coverage.json`).
- **`select_opponents(..., provisional_theta=)`** (review F5): the new document's own best guess, in logits on
  the calibration's scale (the scale of `score_documents`' EAP; the call maps it onto the query's Bradley-Terry
  scale); `None` (the default) is the query's median fitted ability, the behaviour so far.
- **`score_delta(..., scores_a=, scores_b=, ties=)`** (review F4): with the systems' score mappings and the
  protocol's tie rule the deltas are the report's metric (a `group_mean` class is credited its mean gain).
- **`Endpoint.wait_on_outage_s` defaults to 1800 s, not `None`** (review O1): every role config's outage wait
  is finite by default -- an engine restart plus a large model's load -- and a request against an endpoint whose
  replicas all stay down fails with `BackendUnavailableError` (its hint names the field) instead of parking
  forever. `wait_on_outage_s: null` stays the explicit "wait indefinitely" choice, documented as such.
- **`RunConfig.step_budget_s`** (new, default `None`): a per-step wall-clock budget in seconds. The shared
  transport checks it before each request and after every park, and the judging pass before each phase's
  windows; a step over budget stops at the next seam with the new `rcp_ndcg.errors.StepBudgetExceededError`
  (exit code 9, `INTERRUPTED`), the store keeps every judgement it wrote, and `run resume` continues from
  there. `None` leaves the steps unbudgeted.
- **`rcp_ndcg.errors.StepBudgetExceededError`** is the typed error of an exceeded `step_budget_s` (an
  `Interrupted` subclass: the state on disk is consistent and resumable).
- **`rcp-ndcg judge tournament|rubric` takes `--mirror-interval <seconds>`** (default 60, the run config's
  `mirror_interval_s`), so the standalone judging pass's mirror flushes at the interval the run config would
  use.

### Fixed

- **A one-part suite writes its subset's config names**: `MtebWriter.write_dataset` took the single-dataset
  branch for a suite with one part and used the suite's own `subset` (`"default"`), writing unprefixed
  `corpus`/`qrels`/`queries` configs that mteb cannot find for the part's subset; it now uses the part's
  `subset` and `split` (the same values for a single dataset).
- **The pplx-embed-v2-late reference's media token count** is the media item's own count: the merged patches
  plus the vision start/end wrapper (2), not the `[D] ` prompt token (which is the document's text, counted in
  the text budget; the engine's with/without-media prompt difference and the client's `content_media_tokens`
  both exclude it). The 9B's pairs validation failed every image row by one token until the count was fixed;
  the 0.6B's media rows had been refused by the pre-workstream-09 client, which is why it had not surfaced.
- **The pplx-embed-v2-late reference's embed mode loads the resolved variant's checkpoint**: it hardcoded the
  0.6B model/revision while the harness passes `--recipe` with the resolved variant, so the new 9B variant's
  stage-2 comparison would have run against the 0.6B checkpoint (a wrong oracle, not a tolerance miss); the
  reference now reads the model and revision from the recipe and cross-checks them against the tokenizer spec.
- **The request generator validates a variant of a multi-variant family and reads the client policy from the
  client dict**: `_validate_and_prune` re-reads the recipe through its family directory (decision 34), and the
  eight `getattr(recipe.client, ...)` sites now use `.get` (the `getattr` always returned the default, so
  `empty_doc: send` never planned the empty-content row and an instruction mode was never seen).
- **The formatting's own edges** (workstream 10 C2/C3, the review's M8-M11, and the two verifier rounds' minor
  findings): a template `instruction` span on a wire without an `instruction` field (a hosted rerank profile) is
  refused at construction -- the adapter's `HAS_INSTRUCTION_FIELD` fact decides, and the client never sends the
  field to a vendor body that does not declare it; `instruction: none` beside a span is refused too (the span
  would render empty); the index identity covers the resolved document-side task instruction (two builds
  differing only in it never share an index) and `retrieval.rerank` refuses a document-side instruction (a
  reranker's instruction slot is the query's); the sparse (BM25) path follows mteb's own BM25 -- a corpus row
  indexed as `title + "\n" + body`, a query as the per-query append alone, no `Task:` frame -- instead of
  borrowing the retrieval dataloader's join; the empty-query refusal is decided on the data's query, before any
  task frame is folded around it; the Hub reader's column completeness is decided over the queries mteb keeps
  (a dropped row's missing instruction no longer refuses a coherent subset); the judging identity keys a task
  instruction only when one is declared; and an embed or pool endpoint that declares no `instruction` policy
  refuses a request carrying a task instruction (naming `fold`/`none`) instead of applying the fold to a recipe
  that never chose it.
- **A torn `.mirror.json` no longer crashes `run status`** (review S1): the mirror's state file is published
  atomically (temp file + rename, the storage helper), and an unparseable state file reads as "never ran" with
  a warning, as the judgement store treats a torn identity. A reader racing a flush used to raise out of
  `Run.state`.
- **A local or shared mirror publishes whole files atomically** (review S2): `_Target.write` routes local
  targets through `storage.publish_bytes` (temp file + rename), so a concurrent `restore()` on another host can
  no longer read a partial `manifest.json`/`identity.json`; remote object stores still write each object whole
  with `pipe_file`. `storage.publish` keeps the mode a plain write would give the file (an existing target's
  mode, else `0666 & ~umask`), so a shared reader keeps its access, and names its temp `*.tmp`, which the
  mirror's walk and `restore()` skip: a SIGKILL mid-publish leaves nothing the mirror uploads or restores.
- **An opt-in engine patch ships the pooling-hang backport** (`rcp_ndcg_vllm.patches`): the
  `pooling-full-context` patch backports vllm-project/vllm#48039 (commit `e6fc81bc78`) by wrapping
  `Scheduler.__init__`, so a pooling runner stores `num_sampled_tokens_per_step = 0` and a chunked prompt of
  exactly `max_model_len` tokens schedules its last token. The engine process applies it only when its
  `RCP_NDCG_VLLM_PATCHES` names it (a comma-separated list; `rcp-ndcg-vllm serve` passes the environment
  through), logs one line when it applies, one inert line when the running vLLM already carries the fix, and
  never touches a generate runner. Delete the patch when `engine.image` moves to the first vLLM release that
  carries `e6fc81bc78`.
- **The pplx contextual plugin serves on vLLM v0.31.0**: the pooling contract's role-prefix
  validation fired on the engine's own warmup input (measured `[0, 1]`, the kernel warmup's
  `list(range(decode_query_len + 1))` at `vllm/v1/worker/gpu/warmup.py:256-257`), so the engine died at
  startup. An input whose first id is 0 is now recognised as one of the engine's dummies -- the kernel
  warmup and the all-zero pooler sizing grid -- and pools as a single span, which vLLM discards; only a
  non-zero input without a role prefix is a contract refusal.
- **The retrieval review's l10c findings (B1-B5, B8)**: every paper config that encodes or scores a query
  declares its instruction policy with the value the paper's code used (`instruction: none` for the dense
  `octen.yaml`/`cohere_embed_v4.yaml` and the hosted rerankers -- the pre-unified dense path sent the bare
  query, `external_rerankers.py`'s `_HostedRerank._payload` is `{"model", "query", "documents"}`, and the
  paper's datasets carry no per-query instruction; the BM25 config takes none by construction); the sparse
  corpus builder reads a `content`-carrying row's body (`as_content`, never the raw `text` field a media row
  leaves empty); the `messages` route refuses a template `instruction` span (it sends the content and leaves
  the frame to the engine's chat template, which cannot render the span); the run-step identities carry the
  text-formatting rule's version (`TEXT_FORMATTING_VERSION`), and the judging identity carries it beside the
  dataset's instruction, so a resume never reuses candidates or judgements built from other strings; the
  recipe loader and the harness case guard state the new instruction capability (an embed or multi-vector
  recipe's span needs `instruction: fold`; the conformance embed/pool send passes the case's instruction); the
  BM25 claim is scoped to the text mteb's BM25 indexes (the scoring is `bm25s` on both sides, with this
  package's tokenisation), a card config that only `dataset_info` lists no longer shadows the conventional
  `{subset}/{part}.parquet` path, and an index rebuild clears a stale `offsets.npy`.
- **Four new sizes for three shipped families** (decision 34): `octen-embedding-0.6b` and
  `octen-embedding-4b` (the Octen family's 0.6B and 4B checkpoints, last-token pooling and the paper's
  `"- "` document frame), `jina-embeddings-v5-text-nano` (the EuroBERT-210m encoder under the same vLLM
  `JinaEmbeddingsV5Model` dispatch as the family's Qwen3-based `-small`; its own 8192-token budget and
  Matryoshka list) and `topk-embed-v1-xsmall` (the 1024-dim sibling of the plugin-served topk retriever).
  Each is a full recipe id with its own pinned revision, per-size overrides, contract pins, stage-1 test,
  golden and pairs file; the catalog, the release checklist and the request generator's four new pairs
  files gain the rows.

### Fixed

- **The request generator validates a multi-size family's variant**: `_validate_and_prune` re-loaded
  the recipe with `load_recipe(recipe._dir)`, and `_dir` is the family directory, which the standalone
  path refuses for a family with more than one variant, so no multi-size family could regenerate its
  pairs files. It now re-reads the variant through its family directory (`load_family` +
  `load_recipes_of`).
- **The Qwen3-VL video token count**: the engine's prompt renders one timestamp line and one vision block per
  temporal group inside the chat template's own vision pair, and under the engine's fps rule the frame count
  follows the clip, not a declared `num_frames` (which the backend ignores). The count now reproduces E1's
  measured 98 and 458 tokens for the media set's two 64-frame/8 fps clips (the test loads the checkpoint's own
  vendored tokenizer); the timestamp lines are exact when the client has a tokenizer and the family's bound
  otherwise.
- **A media document's trained head can be sent as a system message**: a checkpoint whose engine chat template
  injects no frame of its own (pplx-embed-v2-late's pass-through template) otherwise renders an image-only
  document without the `[D] ` prefix its card's sentence-transformers path sends as a system message. The
  client now sends the shape's leading fixed template segments as a leading `system` message under
  `media_head_as_system: true`, keeps the user turn to the content span, and the startup media probe's
  baseline carries the same head (the media delta still cancels it).
- **`max_duration_s` no longer refuses a prepared frame set** for a duration its dropped container no longer
  carries (the source's duration was checked when it was sampled); `skip_keep_mask` raises a typed
  `rcp_ndcg.errors.DataError` with a hint instead of a bare `ValueError` from inside a client; and the engine's
  video-token pruning (`--video-pruning-rate`) is now declared, counted and cross-checked against the serve
  args instead of silently changing the prompt layout.

- **A raw-binary media column reads by its magic numbers** (mteb's Any2Any repositories store the page
  bytes directly): the Hub and `mteb:` readers sniff the format, record the dimensions the bytes state and
  refuse bytes no known format names -- a raw cell once crashed with a bare `AttributeError`. A media cell in
  the `{"bytes", "path"}` struct keeps its container's MIME type (a video cell once failed in `store_media`),
  and a decoded video object is refused by name rather than crashing; `document_parts` is a corpus setting
  (a query always reads its own media).
- **A v1-style `mteb:` task converts like mteb's own `evaluate` does**: `task.load_data()` followed by
  `convert_v1_dataset_format_to_v2` -- a task that fills `corpus`/`queries`/`relevant_docs` (BRIGHT and the
  other custom loaders) once crashed with an `AttributeError` on `task.dataset is None`.
- **topk-embed-v1-small can send images** (the MASTER open item, workstream 09): the pooling client refused every
  media document whenever `document_skip_token_ids` was declared, so the recipe's media stage failed on the node.
  The skip rule now has a rule at image positions (see `skip_unapplied` above), the media document rides the
  messages route, and its text part carries the fitted content span -- one frame on every route (the engine's
  chat template frames a media item once, exactly like the embed role's `messages` route).
- **The first GitHub CI run is green** (run 37822235213): the gated job installs `rcp-ndcg-vllm` editable (the
  recipes live beside the package in the checkout, so the non-editable install left the recipe-backed case
  validation without a recipe root; pinned by a packaging test); the MCP SDK round-trip test's expected tool
  set follows the server's registration -- the SDK server and the built-in loop list the same 13 tools through
  `rcp_ndcg.mcp.tool_manifest()`, and the test's expected set predated `eval_score` and `run_start`; and
  `rcp_ndcg.eval.mteb.get_tasks` accepts the published files' renamed tasks: the suites' current releases
  ("Rename the tasks to ...RCPReranking, add the open-corpus view") key `_TASK_METADATA` by published task
  name and ship the alias map `_SUBSETS`, so a subset name (`aops`) was refused as unknown. A subset and its
  published task name both resolve now (the same task either way; naming both is refused as a repeat), the
  retrieval view carries its published `...RCPRetrieval` name (the older files keep the `.retrieval` suffix),
  and without `names` the published default view is built (the ViDoRe files' OCR variants stay out of it).
- **A role client keeps an item's media placement through the fit**: the fitted text was put before every media
  part, so a media-first item (a page and then its caption; the vision-language cards build their inputs media
  first) went out text-first -- another input than the one given. The text now stands where the item's first
  text part stood.
- **The offline fake counts tokens as the engine would** (`rcp_ndcg.inference.fake`): `/pooling` answered one
  vector per whitespace word and drew its `prompt_token_ids`, so a pooling client with
  `document_skip_token_ids` over `fake://` refused every text whose words and tokens differ (a
  `ProviderError`: the vector count disagreed with the ids it sent) and per-token outputs had the wrong
  length. An item's count now follows the request's tokenization: a token-ids input is its ids, a text the
  ids of the tokenizer the endpoint's config declares under the request's `add_special_tokens` (default true)
  -- `fake_transport(url, model=..., tokenizer=...)` and the new `FakeEndpoint.tokenizer`; the transport and
  the equivalence harness pass the config's. Without one (and for a chat conversation) the documented
  fallback still counts whitespace words. `/embeddings` usage counts the same way.
- **`empty_doc` decides on the content, before the prompt and the template** (`EmbeddingClient`,
  `PoolingClient`): the embedding client applied the policy after the template rendered, so an empty
  document was a non-empty framed turn -- `send_text` (`NULL` on Qwen3-VL-Embedding) never fired, `omit_zero`
  never omitted, and an empty document went out as an empty framed turn; both clients also judged emptiness
  after the side's prompt, so a `doc_prompt` hid every empty document. The policy now applies to the content
  as given; the placeholder is prompted and framed like any content, and the change is the row's `empty_doc`
  processing record. The embedding client also applied no `empty_doc` at all without a budget; it does now.
- **The embed `messages` route frames once** (`EmbeddingClient`, `openai_embeddings`): the client sent the
  declared template's framed render as the user message, and vLLM v0.31.0 renders every chat-shaped
  `/embeddings` request through its chat template (vllm/entrypoints/pooling/embed/io_processor.py:302-355), so
  the frame went out twice; the content cannot ride a single client-side frame either, because the template
  places each media part inside its user turn. The route now sends each item's content (the prompt and the cut
  content span, its media parts beside it) and the engine frames it once; the request carries the declared
  `add_special_tokens` (new `EmbedRequest.add_special_tokens`; the chat route's default is false,
  base/protocol.py:248-257). Each item is its own conversation: several items went as the turns of ONE
  conversation, which the engine embeds into a single vector (embed/protocol.py:69-102); a batch now sends a
  list of conversations. The offline fake reads `messages` the same way. The equivalence harness reads a
  `messages` capture per conversation, audits the declared frame around the sent content, and its
  `template_render_check` renders the served chat template over every captured conversation against the
  declared render (`not_run` without `serve.chat_template`).
- **The anchor audit implements `anchor: last_content`** (`rcp_ndcg_vllm.equivalence`): a recipe declaring
  it (jina-embeddings-v5) fell into the `last` branch, whose edge -- the last fixed segment, not at the edge,
  plus the post-processor's tail -- is empty for a content-final shape on a tokenizer that appends nothing, so
  its audit could never pass. The branch follows the product's definition: the shape ends on its content
  span, the fixed head segments open every captured body as the engine reads them in the assembled render
  (after the post-processor's prefix), the post-processor's tail closes it when the shape declares one, and a
  content token sits between them -- the last kept content token the model pools.
- **The equivalence harness reports exactly the rows the client changed** (`rcp_ndcg_vllm.equivalence`,
  decision 9): a row was reported non-gating only when a census cut's content count exceeded `max_tokens`,
  and the census counts content only -- so a row whose framed request was over the budget while its content
  was under it, and a reranker's query settled at its share inside a pair the budget takes whole, were gated
  although the client had cut them (stage 2 skipped the shared query's settlement row altogether). Stage 1
  and stage 2 now read the client's own processing records (`RoleClient.processing`), per TEXT, never per
  row: a text is reported, under the declared over-cap deviation, exactly when the client changed it -- any
  mechanism: a budget cut counted with the frame, a per-document cap, an empty-document substitution, a media
  resize or drop; a reranker's query settlement changes every pair of its row (the reference renders an
  over-cap pair its own way, the whole pair), a document's change only that document's pair -- and the report names each change's mechanisms,
  the uncut and kept request totals and the budget. Every text the client sent uncut gates exactly, also
  beside a changed sibling in the same row. Declared normalisation (`strip`, `lowercase`) is policy both sides
  apply, never a change: the rerank settlement compares the normalised query with the settled span and
  records a cut only when content was removed; the pair fit's residual re-fit at a shorter query span records
  that settlement too. A record under no input position is attributed to every input of its call.
- **The case loader reads a recipe template's query frame as the query side's text prefix** (`rcp-ndcg-test`):
  a case's run-level `inputs.instruction` passes when the recipe's `query_prompt` or a fixed segment of its
  template's `query` shape carries it verbatim as a whole delimited unit (starting at the text's start, a
  newline or a label's colon, ending at its end or a newline: a substring such as the frame's own `query`
  label or a truncated instruction is refused; a document frame never counts). The product allows exactly one of the two per side (a
  `query_prompt` beside a template is refused), so a template recipe such as qwen3-embedding-0.6b, whose
  `Instruct: ...\nQuery:` frame sends the instruction on every query, was refused with a fix it could not
  apply. A frame without the instruction is still refused.
- **NOTICE attributes every third-party file the recipes and plugins carry**, each re-checked at its pinned
  revision (upstream SHA-256 and licence): the reference modules that port model-card or remote code
  (ctxl, jina-reranker-v3, qwen3-embedding-0.6b, the qwen3-reranker and zerank families,
  qwen3-vl-reranker-2b), the topk plugin's restated config class and the pplx plugin's pooling core and
  pooler join the templates and the vendored Qwen3-VL-Embedding script; the pplx plugin's paths are
  corrected. Two packaging tests keep it so: every repository path NOTICE names exists, and every recipe
  template, vendored recipe module and audited port is named. NOTICE's opening summary names the MIT
  licence of the Perplexity entry beside the two non-commercial ones (a third packaging test checks that
  the summary names every licence an entry names).
- **The equivalence harness's stage-1 over-length sampler is bounded** (`rcp-ndcg-vllm`): it measures the
  padding's token rate once and sizes each append from the measured deficit (at most 8 passes), instead of
  re-tokenizing the growing text at every step -- quadratic at 32768-token budgets, the network recipe tests'
  hang. A recipe tokenizer whose count never reaches twice the budget (a truncation ceiling in the file) is
  refused with a `HarnessError` instead of yielding a sample that was never over the cap.
- **The offline fake draws one seeded stream per vector**: `rcp_ndcg.inference.fake`'s `/embeddings` and
  `/pooling` vectors are one SHAKE-256 stream of the same parts each (read as `dim` uniforms), no longer one
  SHA-256 per component, so a 16k-token text at 2048 dimensions answers in seconds instead of minutes. The
  draw is bit-identical on every machine (the norm is exactly rounded, `math.fsum` of the squares: a BLAS
  `np.linalg.norm` rounded its last bit by the CPU's kernel), but every fake vector's VALUES move
  (deliberately; their shape, unit norm and per-text determinism do not); no shipped test or case pinned the
  old values, and tests now pin the new draw's values and its exact bits. `fake_uniform` (the judge's, the reranker's and the token-id draws) is unchanged. The recipe tests'
  8-wide probe copy of a 2048-wide multi-vector recipe is no longer what keeps them from hanging (the shipped
  width now costs about twice the probe's time per text, not minutes) and stays: its assertions are
  width-independent, and it keeps each probed answer small (an 8192-token text at 2048 dimensions is a
  64 MiB float32 matrix on each side of the wire).
- **The equivalence harness's stage 1 reads every wire shape** (`rcp_ndcg_vllm.equivalence`): a
  `request_shape: token_ids` body is audited on the ids it sends (it crashed the audit with a `TypeError`),
  the engine `/tokenize` check reports `not_run` for it (the engine tokenizes nothing), and the render check
  compares those ids with the reference text's ids under the shape's `add_special_tokens` flag (it compared
  the id list with the text, so a `token_ids` recipe never passed stage 1 with a reference); a `messages`
  body is audited on its messages' text parts, joined with `rcp_ndcg_core.content.TEXT_JOIN` (`"\n"`, as
  vLLM joins them), with its media parts listed as placeholders beside them (it
  extracted as no input, and the audit passed having checked nothing); an audit that read no input now
  fails. An `anchor: first` head is asserted on the assembled render, not on the head's standalone ids (a
  byte-level BPE re-tokenizes the head's join with the content -- a trailing space into `Ġdocument`, a
  Qwen-style `:` into `:Paris` -- which failed every request): a text body must start with the head's
  characters and open with its tokens lying wholly inside them; a `token_ids` body must open with the head
  tokens no content can merge away (measured on the head joined to a fixed set of probe continuations).
  An `anchor: marker` audit counts the markers in the sent content without the post-processor's tokens, so a
  post-processor that appends the same special (`add_special_tokens: true`) no longer stands in for a marker
  the client dropped.
- **The node scripts' Cloud SDK search is declared**: `wave0.sh` and `bootstrap.sh` put an SDK the auth script
  installed on `PATH` through one function (`gcs_sdk_on_path` in `jobs/gcs.sh`, where each kept its own copy of
  the loop) that searches `RCP_GCLOUD_SDK_DIRS` (colon-separated; unset, the same five install locations as
  before; set empty, none). The node-script tests set it empty: their hermetic `PATH` was undone by that
  search on a machine with an SDK in one of those locations (a GitHub runner's `/usr/lib/google-cloud-sdk`),
  which then ran the machine's `gcloud storage cp`.
- **A named recipe plugin is found in every staged wheelhouse**: `bootstrap.sh` installs a plugin named by a
  recipe (not staged as a file) with one `--find-links` per existing `<stage>/extra/<name>/wheelhouse` beside
  `<stage>/wheelhouse`, still `--no-index`: a plugin wheel staged through `rc_build.sh`'s `EXTRA_DIRS` lands
  there and was not found, so the recipes that name it failed.
- **The release publishes in install order**: `publish-vllm` waits for `publish-core` and `publish-rcp-ndcg` (it
  pins `rcp-ndcg==<version>` exactly, as `rcp-ndcg` pins the core). Every instant of the rollout installs, and a
  failed sibling can no longer strand a permanently uninstallable `rcp-ndcg-vllm` on PyPI.
- **The release's sibling pins are read as data**: one `tomllib` check requires `rcp-ndcg-core==<tag>` in
  `pyproject.toml` and, where `rcp-ndcg-vllm` depends on it, `rcp-ndcg==<tag>` exactly (PEP 508 spelling and PEP
  503 names still normalise). Every declaration of the sibling -- in any extra or scope -- must carry that same
  exact pin, and the required one must sit in the runtime dependencies. A manifest the release builds but cannot
  find is now an error -- never "nothing to check" -- and a reflowed dependencies list no longer breaks the tag
  (the `grep` of one exact TOML line is gone).
- **The constraints check is semantic, and CI runs it too**: `.github/scripts/check_constraints.py` compares the
  pins (name to exact version) of `requirements-constraints.txt` against `uv export --frozen` of the lock,
  ignoring comments, marker spelling and layout, in a new `constraints` CI job and in the release (the `sed` and
  `eval` of the file's header is gone). The header records the script's own `EXPORT_ARGV`: one home, still a
  truthful regeneration command.
- **CI opens every dependency gate in `tests/`**: the new `gated` job installs `[data]`, `[mteb]` and the MCP SDK
  (`mcp`, which no extra names) and runs the whole suite, so the pdf and datasets readers, the image-policy
  transformers parity check and the MCP SDK round trip (48 tests) run on every pull request; a cataloguing test
  fails any new `pytest.importorskip` whose gate no CI job opens. The new `vllm-plugins` job runs the model
  plugins' test suites (`rcp-ndcg-vllm/plugins/*/tests`) with `--no-deps` installs beside CPU torch and
  transformers.
- **The reproduction bounds its documented deviations**: each known deviation now names its exact population of
  cells (NanoBEIR 5 FEVER + 2 Quora + 2 NFCorpus + 1 HotpotQA cells within 0.08 nDCG points, BRIGHT 13 TheoremQA
  Theorems cells within 2.6 and 12 of the 14 qrel means within 0.25 -- 35 rows), and `experiments/checks.py`
  fails a run whose counts differ in either direction, replacing a wildcard entry that let any NanoBEIR cell
  drift within 0.08. Each population names its table position (its `label`) and may be declared once, so
  equal-valued populations (NanoQuora/NanoNFCorpus) never merge. `experiments/leaderboards.py::check_trecdl`
  refuses a ragged (query, reranker) matrix,
  naming the missing pairs (and a duplicated row, naming the pair), instead of letting NaN feed the t-test and
  the means.
- **One definition of `BRIGHT_WITH_EXCLUSIONS`** (in `experiments/fetch_data.py`); `experiments/external_judges.py`
  holds the display labels as `BRIGHT_EXCLUSION_LABELS`, with the ids/labels correspondence pinned by a test.
  The two setup snippets (`experiments/README.md`, `REPRODUCIBILITY.md`) give the same commands (installing the
  checkout's `rcp-ndcg-core` first; `pip install -e .` alone would resolve it from PyPI).
- **The experiments import fixture no longer breaks a subset run**: `tests/experiments/conftest.py` removed
  every newly imported module from `sys.modules`, including scipy and numpy's C-extension submodules, which a
  later re-import cannot load twice in one process; it now removes only `experiments/`' own modules.

- **The node runtime's harness bugs found while validating the GPU waves** (`rcp-ndcg-vllm/jobs`) — one failing
  recipe never stops the wave, end to end: `jobs.plugins collect` reports and skips a recipe that fails
  validation (never fails the job) and the wave report marks it failed with the validation message; a named
  plugin installs from the staged wheelhouse only (`--no-index --find-links <stage>/wheelhouse`) and a plugin
  found nowhere marks exactly the recipes that name it failed, with the exact name; `submit.sh` creates
  `RCP_SUBMIT_DIR` when it does not exist. The reference venv installs `--no-deps` under the image's full
  freeze as constraints (resolving the image stack fails on its unregistered dependency tree) and completes
  only its OWN distributions' missing dependencies from the wheelhouse to a fixed point (`jobs/reference_deps.py`);
  the report's `reference` block records `torch` and `torch_is_image_build`, and a CPU torch on a GPU node is
  a failed bootstrap. Every disk check measures a not-yet-created cache at its nearest existing parent, each
  engine slot's `TMPDIR` is short enough for vLLM's ZMQ IPC paths (AF_UNIX's 107 characters) whatever the
  recipe id is, and `steps.serve.state` records the serve step's own success (a clean stop is not a failure). The
  `WAVE.md` table keeps one row per recipe whatever the message wraps.
- **The T4 driver creates its output directory** before it writes the job script (`run_job_script`).
- **A tokenizer file's embedded truncation and padding no longer cap the counts** (G5): a `tokenizer.json`
  can ship `truncation: {max_length: 1024}` (topk-embed-v1-small does) or fixed-length padding, and an
  un-reset backend silently topped every count and id list at those lengths, so no budget above them could
  cut. `TextTokenizer.from_json` resets both at load -- the one construction site in the package
  (`load_tokenizer` and `from_backend` funnel through it), the load-time equivalent of transformers'
  per-call reset.
- Every `text_budget` census row names the budget that bounded it (`budget_tokens`) -- the rows `fit`
  records (the query rows a declared `query_max_tokens`, the document and pair rows `max_tokens`), the
  hosted-vendor `<budget>` row, and the rerank client's shared-query settlement row -- and the per-shape
  budget cuts the query shape against `query_max_tokens` exactly as a pair splits it. A query share EQUAL to
  `max_tokens` is a legal per-shape budget (both shapes capped the same); only a share ABOVE it is refused,
  and the rerank config keeps refusing an at-or-over pair share.
- The QA mutation survivors' boundaries are pinned: a cut that fits nothing is empty (never the whole text),
  `smart_resize` accepts an aspect ratio exactly at 200 and keeps a snapped area exactly at `max_pixels`, a
  one-token chunk sits over the cap when its token re-tokenizes longer alone; the suite runs each test under
  a per-test timeout (`RCP_NDCG_TEST_TIMEOUT`, 60s default) so a hang fails fast.
- **The exported schemas carry the types they describe**: `run-summary.v1`'s `manifest` is the
  `run-manifest.v1` model (it was `"type": "object"`), `run-list.v1`'s rows are a typed `RunListRow`, and
  `calibration-summary.v1`'s `families`, `coverage` and `diagnostics` are the `Family`, `CalibrationCoverage`
  and `Diagnostics` models instead of untyped dicts. `conversion.v1` gains `limit` (`int | null`): a
  `data convert --limit` smoke conversion records the cap, so its record is not mistaken for a complete
  small corpus. The `cli.v1` envelope schema changed description-only (`data` says which commands tag their
  data with a schema id). No payload changes shape except `run list`'s unreadable rows, which now carry the row's null
  fields explicitly; the payloads validate against the regenerated schemas.
- New served recipe `octen-embedding-8b` (`rcp-ndcg-vllm/recipes/octen-embedding-8b/`): the paper's
  first-stage retriever Octen/Octen-Embedding-8B @ `5adcfa292e712091dfc30f0e97f0b2282e6cc66c` on stock
  `vllm/vllm-openai:v0.31.0` (`--runner pooling`; last-token pooling and the normalize activation come from the
  checkpoint's own sentence-transformers configs). The client block is an `EmbeddingEndpoint` with the product's
  template as data: documents render as the one string `"- " + text` (the paper's prefix, a fixed head segment;
  never separately tokenised ids), queries as they are, the appended end-of-text anchor (added-token name
  `endoftext`, id 151643) declared via `add_special_tokens: true` and reserved by the 8192-token budget
  (`on_overflow: cut`), `max_model_len: 8192` defense in depth, `chat_template: null` (the checkpoint's ChatML
  template would change every prompt; the /v2/embed route is a trap: it auto-applies the checkpoint's ST prompts).
  `reference.py` is the paper-exact in-process path (bf16, left padding, last token, float32 L2) whose render
  mode needs no torch and no transformers; the recipe directory carries its `requirements-reference.txt`.
- New serving recipe `rcp-ndcg-vllm/recipes/qwen3-vl-embedding-2b/` (`Qwen/Qwen3-VL-Embedding-2B` at revision
  `9f2f7e71…`, role `embed`, stock `vllm/vllm-openai:v0.31.0`, no plugin): the chat frame declared as product
  `TemplateSpec` data with the model's default instruction pinned as fixed text, every item sent on the
  `messages` route and framed once by the checkpoint's own chat template (no template file ships;
  `add_generation_prompt` and `add_special_tokens` sent true), the media pixel budget pinned on both sides
  (`serve.mm_processor_kwargs` `images_kwargs` min 4096 / max 1843200; the client's `image_processor:
  qwen3_vl` and `image_policy` with `engine_pixel_pinning`, R20), its reference's `--mode media` for the
  media stage,
  explicit `client.tokenizer` + `max_tokens: 8192` with `on_overflow: cut`, `empty_doc: send_text "NULL"`
  (the card's NULL rule), and the subprocess reference running the card's `Qwen3VLEmbedder` (vendored
  verbatim, sha256-pinned; the render mode mirrors the anchor-preserving cut, and the card's whole-prompt
  right cut is the declared `anchor_drop_over_cap` deviation). The recipe's tests run stage 1 on CPU against
  the pinned tokenizer (offline: skipped with a clear reason) and prove the over-cap cut and the query-side
  frame byte-identical.

- **`rcp-ndcg-vllm` gains the release-candidate and wave scripts** (`rcp-ndcg-vllm/jobs/`, and
  `wave0.sh` with the package): `rc_build.sh <name> [<commit>]` builds an RC exactly as `release.yml`
  does — the three distributions, the version and pin checks, the constraints-file check against the
  lock, `twine check`, and a fresh-venv install smoke from the wheelhouse — and stages the six files
  with the wheelhouse (every locked dependency beside the release wheels, the CPU torch build included,
  the plugin wheels under `rcp-ndcg-vllm/plugins/*` built beside them), the recipes, the wave
  lists and any `EXTRA_DIRS` entries to `<RCP_STAGE_PREFIX>/<name>/`, with a hash manifest
  (`rcp-ndcg.rc-manifest.v1`) that names the CUDA-lock wheels (`nvidia-*`, `triton`) riding along inert
  on a CPU client — the client install refuses them. `bootstrap.sh` copies the staged wheelhouse and
  constraints from the stage prefix to a local directory on the node first (the install source reads
  only what uv reads: a local directory, `file://` or an `http(s)://` URL — a bucket scheme is refused
  at config time), executes the mounted auth script before anything else, and moves everything over
  `gcloud` or `gsutil` when either is on PATH, else the python helper (`jobs/gcs.py` over `gcsfs`,
  installed with `pip --target` into a tools directory outside the engine environment, with
  Application Default Credentials); the path that ran is recorded in the wave-0 report. `bootstrap.sh`
  replaces the superseded stub: it verifies the staged files against the manifest, installs `uv` with
  `pip --target` (the product's `bootstrap_uv` location), leaves the engine environment untouched except
  recipe plugin wheels with `--no-deps` (a `pip freeze` diff beyond exactly those wheels fails it),
  builds the client through the product's install mechanism (`uvx --find-links <wheelhouse> --no-index`
  with the staged constraints — the runners' install-source option, not a second installer) and the
  reference venv with `--system-site-packages` over the image's torch, records the install times and
  versions, and (mode `wave`) runs the wave runner with the staged recipes, wave lists and pairs.
  `submit.sh <rc-stage-uri> <out-prefix> <wave>...` submits one job per wave: `priority_class=` per
  wave, `worker.shared_memory` sized for eight engines (`RCP_SHARED_MEMORY`, default 128Gi), the HF
  token from `RCP_HF_TOKEN_FILE` as a kjobs secret expanded inside the script and never printed, at
  most `--max-jobs` jobs in flight via `depends_on`, the job CLI's output to a file with only names and
  states printed, and `--script wave0` mounting and running the node test. Wave 0 (`wave0.sh`):
  preflight assumptions, the host facts, the three environments with an unchanged engine freeze, the
  Hub (metadata with the token secret) and a gs:// round-trip through `rcp_ndcg.storage` from the
  client, the plugin canary (`fla` must not be importable in the untouched engine environment) with
  the wheelhouse path and every installed engine version recorded, two engines on two isolated slots
  at once, the product's `fit` and embedding client over 20
  texts (5 over the explicit budget) with the engine's `/tokenize` per input, the HF-cache eviction
  with the disk before/after, and the no-engine assert — fail-fast, with one JSON report
  (`rcp-ndcg.wave0-report.v1`, schema at `rcp-ndcg-vllm/schema/wave0-report.schema.json`) and
  a dry mode (`WAVE0_DRY=1`). The wave runner's wave gains per-slot `VLLM_PORT` and `TMPDIR`, the
  pre-serve disk check against the model's Hub size, the post-recipe eviction, and an upload fallback
  through the product's own `rcp_ndcg.storage` when the image has neither `gcloud` nor `gsutil`. Wave
  0's embed step runs the wired `EmbeddingClient` (the config's budget, fitted inside the client);
  every upload attempt is recorded in the report's `uploads` section with its error (the second
  durable report copy carries every attempt except its own; the stdout emit is complete), a directory
  source copies its contents under the destination on every transfer path (the caller declares the
  source's kind — `dir`, `file` or `auto` — and both gcs.py and the CLIs honour it), one retry covers
  a transient GCS error, and `submit.sh` resolves the image's digest (Docker Hub registry, then
  `gcloud container images describe`) into `env.RCP_IMAGE_DIGEST` so the report never says null.
- New package `rcp-ndcg-vllm` (`rcp-ndcg-vllm/`, outside the root uv workspace and lock; version
- The `python_api` contract snapshot's `__version__` constant is corrected from the pre-bump install
  (`0.1.0`) to the committed version (`0.0.1`), which the tree has declared all along; the stale value
  failed `test_surface_matches_snapshot[python_api]` in any venv newer than the bump. No product change.
- New package `rcp-ndcg-test` (`rcp-ndcg-test/`, a uv workspace member; version 0.0.1, depends on
  `rcp-ndcg==0.0.1` and `rcp-ndcg-vllm==0.0.1`): the reference cases and the one conformance suite for served
  recipes. **Unpublished on purpose — never on PyPI** (used by this repository's CI, the product's pytest
  suite and the GPU waves; no published package names it, and `release.yml` builds the three published
  distributions by name so it is never swept into a release). The case format is operator-defined
  (one case per `cases/<recipe-id>/<case-slug>.yaml`); validation covers the file-level rules (the
  `model_card` Hub URL, verbatim quote and 40-hex revision, media existence, the strata labels against the
  case's own inputs, the expected shapes and tolerances) and — with the recipe — the role/modality/
  template-shape rules, the strata grid coverage per recipe, the mixed-length batches' differing measured
  lengths, and the long inputs' measured token lengths against `client.max_tokens` with the product
  tokenizer (a `short` case may not measure over it either), all measured through the case's
  materialized text (a `text_ref` renders as its generated bytes, never as an empty placeholder).
  Media cases carry their files under `cases/<recipe-id>/media/`; an image media case runs on the
  pool and rerank routes (the media goes out as `image_url` content parts through the adaptation
  step). Two declared skips (recorded before every send, each with its reason): any media case on the
  embed route (its `EmbeddingClient` takes text only and refuses media before preparation) and a
  video case on every route (the product's media lowering sends images; a video container is refused
  until a frames reader lands). The runner
  sends every case through the product's role clients built from the recipe's `client` block (never raw
  HTTP, never a copy of the client), against a live engine (`target="engine"`) or a recipe-level fake
  (`target="fake"`), and returns a typed report (`CaseResult`: compared, passed, skipped with reason —
  `values: null` is a skip, never a pass). Public names: `Case`, `CaseSource`, `CaseStrata`, `CaseQuery`,
  `CaseDocument`, `CaseInputs`, `CaseTolerance`, `CaseExpected`, `CaseBundle`, `load_case`, `load_cases`,
  `TextRef`, `text_of`, `default_cases_root`, `CaseResult`, `ConformanceReport`, `Target`, `run_case`, `run_suite`, `FakeEngine`,
  `FakeReply`, `FakeEmbedEngine`, `fake_engine_for`, `fake_http_transport`, `register_fake_engine`,
  `unregister_fake_engine`, `registered_fake_engines`, `fixture_path`, `package_tokenizer_path`,
  `FIXTURE_RECIPE_ID`, `CaseRun`, `conformance_params`, `CaseError`, `ConformanceError`, and the
  generated-text / reference-text seam: `GENERATORS`, `materialize`, `GENERATOR_VERSION`
  (`rcp_ndcg_test.generators`, stdlib only), so a text-diff test outside the package resolves the same
  bytes. The `spearman_min` gate's one home: `spearman` and `average_ranks` (`rcp_ndcg_test.ranks`, tie-corrected,
  numpy alone, no SciPy). The fake-engine
  registry ships no model-level fakes yet (they are built from the GPU recordings later); one registered
  test fake for the packaged fixture recipe (`fake-embed`) and deterministic pool/rerank test fakes built
  for its `fake-pool`/`fake-rerank` fixture recipes exercise the runner end to end on CPU.
- New package `rcp-ndcg-vllm` (`rcp-ndcg-vllm/`, outside the root uv workspace; version
  0.0.1, depends on `rcp-ndcg==0.0.1` — a hard dependency, and pinned by the release workflow's version
  check): serving recipes for vLLM as data. The recipe's `client` block **is**
  the product's endpoint config (`EmbeddingEndpoint`, `PoolingEndpoint` or `RerankEndpoint`); the harness
  declares no parallel schema. Stage 1 runs the product's `fit()`; the anchor audit reads `fit`'s output; the
  engine's `/tokenize` is the tokenization truth (R29); the reference runs as a subprocess in its own
  environment (`--reference-python`, required for stage 2; the harness imports no torch). Removed from the
  earlier draft: the harness's own `TemplateSpec`, `TemplateSegment` and `BlockingSpec` (the product's
  `TemplateSpec` replaces them), `EngineClient` and `fold_instruction` (the product's transport and adapter
  replace them), and `effective_embed_dtype` (the product's `PoolingEndpoint` carries `embed_dtype`).
  Public names: `Recipe`, `ClientEndpoint`, `EngineSpec`, `Gates`, `ReferenceSpec`, `Resources`, `ServeConfig`,
  `StatusSpec`, `RecipeError`, `HarnessError`, `load_recipe`, `iter_recipes`, `serve_argv`, `client_config`,
  `recipe_json_schema`, `default_recipes_root`, `PINNED_POOLER_CONFIG_FIELDS`; the JSON Schema of `Recipe` is
  exported at `rcp-ndcg-vllm/schema/recipe.schema.json`. The recorder (`record`),
  stage 3 (`stage3_metrics`), the wave runner (`run_wave`) and the subprocess reference runner (`run_reference`)
  are public with package tests covering each; `metrics.py` shells out to `rcp-ndcg eval score` (the product is
  a dependency, so no extra is needed for stage 3).
- **The harness drives the product's role clients** (`rcp_ndcg.inference.clients.EmbeddingClient`,
  `PoolingClient`, `RerankClient`, built from `client_config(recipe)` with the recipe's real budget): stage 2
  pre-fits nothing and clears no budget field -- the client prompts, fits and settles exactly as the served
  path does (a reranker's shared query span settles once per call). Stage 1 captures the clients' request
  bodies through the product's own transport injection point (a capturing `httpx` transport handed to the
  client's `Transport` as its `httpx_transport`; the product's offline fake answers when no engine is given)
  and audits them: the anchor audit (the settle-once query included) and the engine's `/tokenize` (R29) read
  the same captured bodies, the reference's `render` compares against them, and the served template file is
  rendered against the declared template for every declared shape. Removed with the wiring: the harness's own
  `fold_query`, `_fit_pair`, `_pair_tokens` and the pre-fit-then-send path (over-cap is decided on the
  client's census; an uncut query could reach the engine before it, and the recorder hand-built bodies that
  had already drifted from the adapters').
- `record` drives the product's role client for the role route's fixture (the product's request, byte for
  byte) and records the provenance `GET /v1/models` plus the role route's over-length and unknown-field 400s
  (bare probes: the clients cut before an engine would refuse). The `/score` route is no longer recorded (no
  product client speaks it); a failed `--record` step fails the wave recipe's verdict.
- **The rerank pair fit's census rows name the documents' original positions**
  (`RoleClient._fit` takes the caller's ids; `RerankClient._fit_pair` passes them; a chunked document's rows
  carry `<original>#<chunk>`, and the pooled scores land on their document): with `empty_doc: omit_zero`, a
  later document's cut is recorded under ITS position, never the kept position an earlier omission displaced.
- Recipe rules (at load, with the product's messages): a rerank recipe speaks `api: rerank` and takes no
  `serve.convert`; an embed/multi_vector recipe's template cannot declare an `{content: instruction}` span
  (the role's clients fill no instruction); `recipe.input` declaring an image or video must declare media
  capacity on the client (`max_images`/`max_videos` > 0); `client_config()` keeps a declared `client.recipe`;
  `engine.min_version` accepts release candidates. The product's `TemplateSpec` refuses an empty fixed
  segment (a workaround marker for the anchor audit's old last-fixed-segment rule, which now falls to the
  post-processor's tail when a shape ends in content).
- Stage 1's report carries `checked` (the audited request count) for `EQUIVALENCE.md`; `tests/recipes/` is
  the recipe lanes' network-gated home (`RCP_NDCG_NETWORK_TESTS=1`; downloads land in
  `RCP_NDCG_VLLM_TOKENIZER_CACHE` or `tmp_path`, never the checkout).
- First served recipe `recipes/qwen3-reranker-4b/` (the recipe lanes' product): Qwen/Qwen3-Reranker-4B at the
  pinned revision, role `rerank`/pointwise on the unmodified `vllm/vllm-openai:v0.31.0` image — hf_overrides
  turn the checkpoint into the 1-label sequence-classification head (`classifier_from_token` [no, yes],
  `is_original_qwen3_reranker`), the paper-exact chat template ships as `template.jinja` (the stock example
  file renders one trailing newline short of the paper prompt; REVIEW-LOG R10), the budgets are the paper's
  (`max_tokens` 8192, `query_max_tokens` 4096, `on_overflow: cut`, tokenizer pinned `<repo>@<commit>`), the
  anchor (the 9-token assistant suffix) is declared `anchor: last` and reserved from every cut, and the
  reference derives unchanged from the paper's `QwenOGRerank` at bfloat16 with
  `reference.known_deviations: [anchor_drop_over_cap]` (over-cap pairs gate on under-cap pairs only).
  Stage 1 passes on CPU against the real Hub tokenizer (tokenizer files only); state `unverified` until the
  GPU waves run.
- The first serving recipe ships: `rcp-ndcg-vllm/recipes/topk-embed-v1-small/` (recipe.yaml,
  reference.py; the recipe directory is grafted into the sdist with the rest of `recipes/`), serving
  `topk-io/topk-embed-v1-small` as a multi-vector model on `vllm/vllm-openai:v0.31.0` through a
  `vllm.general_plugins` wheel (`serve.plugin: topk-embed-vllm`, built by the plugin lane). Its tests pin
  the recipe on CPU: stage 1 over the product's `fit` (tokenizer files only), the image-wrapper ids of
  the served chat template, the document keep-mask asymmetry, and anchor and frame mutations that go red.

- **`JobSpec` takes exactly one of `argv` and `phases`** (`rcp_ndcg.runners`): a job without phases runs its
  `argv`; a phased job's commands are its phases' `argv`, and it carries no `argv` of its own — both or neither
  are refused with a message naming which. `JobSpec.argv` is optional (`tuple[str, ...] | None`); phased jobs
  build one command per phase with the new `JobSpec.with_argv(argv)`. `job_for` hands a runner that renders no
  phases the whole-run command as the job's `argv` (as before), and the local runner renders a phased job's
  phases in order instead of an `argv` that covered them.
- **Every runner takes an install source** for the coordinator's release, where the job installs it from a
  staged wheelhouse: `runner.options.wheelhouse` (a directory of wheels, a `file://` URL or an `http(s)://`
  URL of one — what uv's `--find-links` reads; a bucket scheme is refused with the fix — rendered as `uvx
  --find-links <wheelhouse> --no-index`: nothing is asked of PyPI or the torch index) and
  `runner.options.constraints` (a constraints file path or URL replacing the release's). A relative local path
  is recorded absolute by the runners whose job records read paths on the submitting host (slurm); a URL is
  kept as it is. Refused where nothing installs (the local runner — the coordinator runs in this host's
  environment — and SLURM with `container_runtime: none` — the node's environment provides the release);
  rendered for the Kubernetes pod and the SLURM containers. `install_argv` takes `wheelhouse` and
  `constraints`; its default rendering is unchanged.
- **One client base for every role** (`rcp_ndcg.inference.clients.RoleClient`, R5/R14/R15): the adapter
  lookup within the client's role, the hosted profile's default base URL, the transport (built from the
  config unless a `Sender` is given), the sync bridge -- one rule: a non-transport sender must provide
  `run`, or the constructor raises `ConfigError` (no `asyncio.run` fallback) -- and the lifecycle:
  `close()` synchronous, `async aclose()` awaited, both context managers. `EmbeddingClient`,
  `RerankClient`, `PoolingClient` and `JudgeClient` derive from it (the judge's adoption is this release's;
  see Unreleased/Changed).
- **Auth in the transport** (R6): every adapter profile declares `API_KEY_ENV`, `KEY_REQUIRED` and
  `AUTH_HEADER` (the rerank profiles gain them: `cohere` `CO_API_KEY`/`COHERE_API_KEY`, `voyage`
  `VOYAGE_API_KEY`; the served wires and pooling take none), and the transport resolves the key -- the
  config's `api_key_env` first (an unset named variable is a `CredentialsError`), else the profile's
  variables in order, in the profile's header. The transport takes the facts as `AuthProfile`
  (`Transport(endpoint, auth=...)`, and `Transport.set_auth(...)` for an injected transport); every
  client-side key handling and the `api_key_env` clearing are gone. Key values never appear in logs or
  error messages.
- **No sibling left running** (R7): every client's fan-out runs in one `asyncio.TaskGroup` -- a failing
  request cancels its siblings, no rerank `checkpoint` lands after the failure, and no task is left
  pending; a group carrying exactly one failure is raised as that failure, so the typed errors surface.
- **The text budget wired into every client, and the media with it** (item 4): a config with `max_tokens`
  fits every request through the shared mechanism (`rcp_ndcg.data.preprocess.fit`), cutting only content
  spans with the template re-attached, recording every cut in the census (`client.census`, a
  `TextTruncationCensus`; the rerank client records a shared query's settlement under the doc id
  `<query>`); the interim refusal of `max_tokens` is gone. The served rerank path sends no
  `truncate_prompt_tokens`/`max_tokens_per_query`/`max_tokens_per_doc` (the client cut already). **Media is
  wired through the one preparation path** (`rcp_ndcg.data.prepare.prepare_request` -- the judge's own):
  sized exactly as the declared `image_processor` would under the role's `image_policy`, its tokens counted
  and reserved whole out of `max_tokens`, never cut; when media alone fill the budget the declared
  `on_overflow` decides (`cut` shrinks to the policy minimum then drops whole items, every drop recorded
  with `dropped=True` in `client.media_census` under its input's doc id; `fail` refuses; `chunk` is
  refused -- a vision block is atomic); a document whose every media item was dropped is empty and follows
  `empty_doc` (which every role client consumes, for an empty text document too). `max_images`/
  `max_videos` gate per wire call (the pooling wire's one media item per call; the rerank call's query
  plus that chunk's documents) before anything is sent; a role with an `image_processor` exposes
  `probe()`/`check_engine_media()` -- one prepared probe image, the engine's reported prompt tokens
  compared with the counted ones (the media block plus the probe's text tokens), a mismatch refused and a
  reply without usage recorded `not_checked` (never silent).
- **Explicit budgets for the role clients**: a self-hosted role config must declare `tokenizer` +
  `max_tokens` (already enforced at the config); a hosted profile may declare only the vendor's documented
  limit (`budget_source: vendor`, content uncut). `on_overflow: chunk` is refused for the embed and pooling
  roles (chunks pool scores by max; vectors have none to pool -- a late-interaction document is chunked at
  the corpus layer); the rerank role chunks and pools by `max_pool_scores_by_document`.
- **`dim` refused at construction** (R13): `PoolingClient` refuses a config whose `dim` is unset (the
  base64 frame of `/pooling` is flat and carries no shape), before any request runs on the GPU; a pooling
  adapter's `MAX_BATCH` cap is honoured like the embedding ones'.
- **`HOSTED` declared, not inferred** (R8): the adapters declare `HOSTED: ClassVar[bool]`; the rerank
  profiles' `use_activation` refusal keys on it, not on `DEFAULT_BASE_URL is not None`.
- **`EngineRole` meets `AdapterRole` in one written mapping** (F7): `ENGINE_ADAPTER_ROLES` and
  `check_engine_api(api, engine_role=..., where=...)` in `rcp_ndcg.inference.adapters.base`; the runners'
  engine overlay refuses a config whose `api` selects an adapter of a different engine role.
- **A served rerank config sets `use_activation` explicitly** (F10): `RerankEndpoint` with `api: rerank`
  refuses `use_activation: None` (two engines with different defaults would share an identity); hosted
  profiles keep `None` (their scale is fixed). The role client enforces the same rule keyed on the resolved
  adapter's `HOSTED` flag, so a served third-party rerank wire needs an explicit choice too.
- `Transport.aclose()` is a true async close (R15): awaited on the pool's own loop; `close()` stays the
  synchronous twin.
- **A pooling adapter's `MAX_BATCH` cap is honoured** by `PoolingClient` like the embedding ones'.
- `schemas/index.v1.json`, `schemas/judge-config.v1.json` and `schemas/run-config.v1.json`: the
  `api_key_env` and `query_max_tokens` descriptions state the wired behaviour (the transport resolves the
  key from the profile's variables; the shared query span settles once per rerank call). The Python-surface
  snapshot records the new names (`RoleClient`, `AuthProfile`, `ENGINE_ADAPTER_ROLES`, `adapter_roles_of`,
  `check_engine_api`, `Transport.set_auth`); no CLI command, flag or exit code changes.
- **The retrieval API runs on the role clients, and its configs select the wire with `api`** (the unified-inference
  design, sections 4.1, 7.1 and 7.2):
  - `rcp_ndcg.retrieval`'s configs are the inference role configs, by `api`: an encoder is the served
    `ServedEmbedding` (`api: openai_embeddings`) or the hosted `CohereEmbedding`, `VoyageEmbedding` and
    `GeminiEmbedding`; a late-interaction encoder is the served `ServedPooling` (`api: vllm_pooling`); a
    reranker is the served `ServedReranker` (`api: rerank`) or the hosted `CohereReranker` and
    `VoyageReranker`. `api` replaces `provider` as the discriminator; a hosted profile omits `base_url` (its
    adapter declares the public root), a served one names the engine's URL. Every role client —
    `EmbeddingClient`, `PoolingClient`, `RerankClient` — runs over the shared transport, so every retrieval
    role gains replica routing, outage parking, the provenance probe and typed errors.
  - `validate_retriever(data)` / `validate_reranker(data)` read a config from parsed YAML and refuse an old
    shape (`provider:`, `engine:`, an encoder `pooling`) with a `ConfigError` whose hint shows the new shape.
    The CLI's `--retriever`/`--reranker` YAML loading goes through them (and validates the retriever before
    the dataset loads).
  - The per-query rerank checkpoint moves to `rcp_ndcg.retrieval._api`: the record format (`{"q", "k", "s"}`),
    the per-record fsync, the file (`rank000.jsonl`) and the key payload are the served path's historical ones,
    so a resumed rerank reads a checkpoint an earlier release wrote.
  - **Plugin adapters reach retrieval**: a non-shipped `api` is resolved against the role's registry where the
    config is read (an unregistered or wrong-role name is refused with the registry's hint) and builds the
    role's generic endpoint config — `PluginEmbedding`, `PluginPooling` (a late-interaction encoder may also be
    one) and `PluginReranker`, exported from `rcp_ndcg.retrieval`. The adapter name is content, so a
    step (and an index) identity keys on it, as the judge's does for its third-party adapters; the retrieval
    steps run the third-party wire like a shipped one.
  - Every paper config's `recipe:` id is the checkpoint's lowercased Hub repo name, never a short Hub redirect
    (`zerank-1-reranker`, `zerank-1-small-reranker`, not `zerank-1`/`zerank-1-small`); pinned by a test over
    every config with a `recipe:`.
  - `EncoderConfig` is `ServedEmbedding | CohereEmbedding | VoyageEmbedding | GeminiEmbedding`,
    `RerankerConfig` is `ServedReranker | CohereReranker | VoyageReranker` (both plain unions, so old shapes
    reach the members' refusals); `RetrieverConfig` stays a `kind`-discriminated union.
- `rcp_ndcg.data.revisions.is_commit(revision)` is the public form of the commit-shape check (exactly 40
  lowercase hex characters, ``False`` for ``None`` or any other revision); no other module reads the private
  pattern, and the resolve paths use the same strict check: a 40-hex revision with trailing whitespace is no
  longer echoed back as a verified commit but resolved like any ref (offline, it warns `UNPINNED_REVISION`).
- `rcp_ndcg.errors.WarningCode` gains `SNAPSHOT_LISTING` (an additive change to the closed list): an offline
  corpus read whose file listing came from the local Hub snapshot instead of the Hub warns with it (the snapshot
  holds only the files a download left, and a partial cache reads as missing data). With `--json` it shows in the
  envelope's `warnings`; otherwise it prints on stderr. `schemas/cli.v1.json` and `schemas/eval-report.v1.json`
  follow.
- `rcp-ndcg eval score` gains a repeatable `--system NAME` (and `eval explain --report` one; the library call
  `rcp_ndcg.eval.evaluate` gains `systems: Sequence[str] | None = None`; the MCP tool `eval_score` takes
  `system` too): score only the named systems of the rankings file. One system whose rankings match nothing of
  the scored dataset is still refused (exit 12; every score would be 0), but it no longer stops the healthy
  systems of a multi-system file: score them with `--system NAME`. The refusal's hint names the way out (drop
  the system's rows, or score the others) with `systems=` for Python callers and `--system` on the command
  line, whenever the file holds several systems. An unknown name is refused with the systems the file names —
  a `ConfigError` (exit 3) from the library call, a `UsageError` (exit 2) on the command line, where it is a
  command-line mistake like an unknown `--fields` or `--metrics` name; `eval explain --report` re-scores the
  saved rankings for the systems the report
  scored (its own, by default; `--system` narrows them further), so one broken system of the file does not
  kill the explanation, and `--system` with `--run` there is a `UsageError` (it has no effect on a run).
- New package `rcp-ndcg-vllm-topk` (`rcp-ndcg-vllm/plugins/topk/`, outside the root uv workspace and
  lock; version 0.0.1, no dependencies): the `vllm.general_plugins` wheel that serves
  `topk-io/topk-embed-v1-small` (multimodal late interaction) on the stock `vllm/vllm-openai:v0.31.0` image
  after `pip install --no-deps`, with no `--trust-remote-code`. Two registrations: a faithful local
  configuration class (`TopkEmbedConfig`, a line-for-line restatement of the checkpoint's remote
  `TopkEmbedConfig` — whose module imports `flash-linear-attention`, absent from the engine image, so the
  remote config load would die before any weight loads) registered with transformers' `AutoConfig`, which
  takes the explicit-local-code path and never executes remote code; and the model class `TopkEmbedModel`,
  a subclass of the native `ColQwen3_5Model` that overrides only the checkpoint-name mapping (`head.` →
  `custom_text_proj.`; the Qwen3-VL naming convention restored): the checkpoint's own
  `text_config.is_causal: false` drives the stock `Qwen3NextAttention` to bidirectional ENCODER_ONLY
  attention on the six full-attention layers, so no attention code is copied. The version guard refuses vLLM
  outside `>=0.31,<0.32` with the tested range named; the pure-torch pooling chain (`token_embed_pool`) and
  the mapping table (`weights`) are importable without vLLM for the CPU tests (entry-point declaration,
  version guard, the 618-name census mapping, the tiny-config chain equivalence against the reference chain,
  and the simulated `--no-deps` freeze check, which also rejects forged wheels with a declared dependency, a
  compiled artifact or a platform tag, and is the GPU wave's script). Skips name the environment: the
  registry effects, the config-class parse and the served-class mapper cross-check need vLLM and
  transformers; the full served-vs-reference equivalence on real weights is the GPU wave's.

- A served recipe in `rcp-ndcg-vllm/recipes/`: `ctxl-rerank-v2-instruct-multilingual-2b`
  (ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b @ 6ffef5dc552583b8db58dc4a87f79f7aee78d2d9, pointwise
  rerank, paper-exact): `--runner pooling` with the `Qwen3ForSequenceClassification` conversion overrides
  (`classifier_from_token` ["!"] = the checkpoint's token id 0, `method: no_post_processing`) on the stock
  `vllm/vllm-openai:v0.31.0` image, the paper's two-line score template shipped as a dual-mode jinja file
  (the harness's check variables and the engine's `messages` render both produce the paper prompt), the
  raw-logit pooler (`use_activation: false` on the client and server side; the paper's score is the raw
  logit of vocabulary position 0 at the final position), and the paper budgets (`max_tokens` 8192 =
  `MAX_SEQ_LENGTH`, `query_max_tokens` 4096 = `MAX_QUERY_LENGTH`, `on_overflow: cut`). The paper's
  whole-prompt right truncation drops the trailing " ??" anchor over the cap, so the recipe declares
  `reference.known_deviations: [anchor_drop_over_cap]` instead of copying the drop into the served path;
  the reference subprocess derives from `experiments/paper/rerankers/reference/contextual.py`
  (paper-exact; no instruction is sent or folded -- the family's `instruction: none`). Status `unverified`
  until the GPU waves run the harness's stages 2–3.

- The `ctxl-rerank-v2-instruct-multilingual-1b`/`-2b`/`-6b` recipe family settles one policy: all three
  declare `client.instruction: none` (the paper configs' mode; the paper's in-process path never received
  an instruction), and neither a template nor a reference folds or appends one -- the 6b template and
  reference drop the vendor's inline instruction slot as 1b and 2b do, and a pairs row's instruction is
  ignored on both sides. Every recipe states the merged rerank client's settle rule (the pair fit binds its
  share on overflow only; the client settles the shared query once per call and ships it at
  `query_max_tokens` whenever it exceeds it), declares over-share queries as divergence rows (the gating
  pairs keep queries within the share), and its reference stays the paper's: `render` fills the harness's
  span format with the raw query and documents the paper's prompt builder receives, uncut, and never
  reproduces the client's cut (over-cap rows are the declared `anchor_drop_over_cap` table; under-cap rows
  gate exactly); the three references load bfloat16 weights on every device, as the paper's factory did.
  Family conventions: the template file is `template.jinja` in all three (6b's `score-template-6b.jinja`
  renamed; one trailing-newline convention), `requirements-reference.txt` ships beside every reference (1b
  gained one), `client.recipe` stays unset (`client_config` records the recipe id), and
  `engine.startup_timeout_s` is no longer restated at its 1800 default.

- The dense recipe family (`qwen3-embedding-0.6b`, `octen-embedding-8b`, `jina-embeddings-v5-text-small`,
  `zembed-1-embedding`, `jina-reranker-v3`): every reference stays the paper's or the model card's and
  never reproduces the client's cut -- `qwen3-embedding-0.6b`'s `render` now emits the card's uncut prompt
  (the card truncates ids at encode) and the recipe declares `reference.known_deviations:
  [over_cap_cut_differs]` (both sides keep the appended anchor; over-cap rows are reported, not gated), and
  `zembed-1-embedding`'s reference drops its unused copy of the client's cut search (its render needs no
  tokenizer now). `zembed-1-embedding`'s reference environment pins sentence-transformers to the 5.3 line
  (the last whose encode calls the checkpoint's remote tokenize; 5.4.0's preprocess-first pipeline drops
  the pooled suffix) with transformers >= 4.51, so the reference venv reuses the engine image's
  transformers. `query_max_tokens` is declared only where the reference caps queries (`jina-reranker-v3`:
  512, the checkpoint's `max_query_length`; the embedders' referents cut queries and documents alike, so
  none declares it); `jina-embeddings-v5-text-small` declares `anchor: last_content`; `jina-reranker-v3`
  states the merged rerank client's settle rule, its `client.recipe` identity no longer claims per-text
  engine caps (the client sends none), its `use_activation` pin is documented as inert (the model's
  pooler has no head; the score is a cosine), and its reference no longer reconfigures the product's
  cached tokenizer (a process-wide 2048-token cap). `experiments/paper/rerankers/jina_v3.yaml` states what
  the served path sends (the 8192/4096 pair cut, no per-text caps). Family conventions:
  `engine.min_version` is the verified image (0.31.0) with any feature floor in the notes,
  `requirements-reference.txt` beside every reference (`zembed-1-embedding` gained one), and every recipe
  test pins the full contract through the shared helper (two mutants red each), downloads its tokenizer
  through the one shared cache, and checks that no shipped recipe file carries an internal process
  label.

- The first served recipe in `rcp-ndcg-vllm/recipes/`: `qwen3-reranker-0.6b`
  (Qwen/Qwen3-Reranker-0.6B @ e61197ed45024b0ed8a2d74b80b4d909f1255473, pointwise rerank,
  paper-exact): `--runner pooling` with the `Qwen3ForSequenceClassification` conversion overrides
  (`classifier_from_token` no/yes, `is_original_qwen3_reranker`) on the stock
  `vllm/vllm-openai:v0.31.0` image, the shipped chat template (dual-mode: the harness's check
  variables and the engine's `messages` render both produce the paper prompt; the stock example
  template alone loses one trailing newline at the scored position), the paper budgets
  (`max_tokens` 8192 = `MAX_SEQ_LENGTH`, `query_max_tokens` 4096 = `MAX_QUERY_LENGTH`,
  `on_overflow: cut`), `use_activation: true` on the probability scale, and a reference subprocess
  derived from `experiments/paper/rerankers/reference/qwen3.py` (on branch lane/l5-packaging at this
  HEAD; probability =
  `softmax([no, yes])[yes]`, bfloat16, batch 16, no anchor ever dropped). Status `unverified`
  until the GPU waves run the harness's stages 2–3.
- `TournamentSchedule.adaptive_batches_for(n_docs)`: the adaptive batches a pool of `n_docs` runs. A pool no
  larger than `adaptive_window` runs one batch, not one per batch: every adaptive window of such a pool holds
  the whole pool, so a further batch asks the same documents again (in the refit order) and covers only what
  the first window's answers already hold. `phase_calls` and `calls_per_query` count it, so the estimates and
  the passes agree; the paper's counts at a pool of 150 are unchanged.
- `schemas/run-config.v1.json`: the `adaptive_batches` description states the one-batch rule; the Python-surface
  snapshot records the new method.
- **The adapter registry is scoped by role** (one namespace per role): `register_adapter(cls)` keys
  on `(cls.role, cls.name)`, `get_adapter(name, *, role)` resolves a config's `api` within its role's names,
  and `known_adapters(role=None)` lists one role's names (or every registered name once without a role). The
  same name registers once per role, so the embed role's `cohere`, `voyage` and the rerank role's `cohere`,
  `voyage` coexist; a wrong-role lookup fails with a hint listing that role's names (and, when the name is
  registered in another role, says so). The embedding profiles take the roles' plain names: `cohere`,
  `voyage`, `gemini` for the embed role. The
  `rcp_ndcg.adapters` entry-point group names its entries `<role>.<name>` (e.g. `embed.bedrock`); an entry
  whose class role disagrees with its prefix is refused with a `ConfigError`, as is an entry without a role
  prefix. The role clients pass their role to the registry (an unknown or wrong-role `api` is refused there),
  and each role config's default `api` is unchanged (`openai_embeddings`, `rerank`, `vllm_pooling`). The role
  list is public as `ROLES` (`rcp_ndcg.inference`, `rcp_ndcg.inference.adapters`), and an entry point's name
  must spell the class's registered name, not just its role.
- **One tokenizer-identity method for every role config**: `Endpoint.identity_extra()` (default `{}`) returns
  `{"tokenizer_sha256": <sha>}` — the SHA-256 of the config's `tokenizer.json` through the one helper
  `rcp_ndcg.data.tokenizer.tokenizer_identity` (over the judge's existing `load_tokenizer`; no second hashing
  function) — for every role config that declares a `tokenizer` (the judge's, the embedding, pooling and
  rerank configs). `RerankEndpoint.tokenizer_identity()` is removed; the judge's identity payload keeps its
  existing keys (the judgement family's tokenizer digest and the preprocessing record's `sha256`) and is
  byte-identical for every shipped judge preset, so no judgement family re-keys.
- **One text-budget mechanism for every served role** (`rcp_ndcg.data.preprocess`): a declared `TextBudget`
  (frozen, content identity: `max_tokens`, `query_max_tokens`, `template`, `on_overflow` `cut|chunk|fail`,
  `chunk` geometry, `aggregation: max`) and one function `fit(inputs, shape, budget, tokenizer) -> FitResult`.
  The template is data (`rcp_ndcg.data.templates`: `TemplateSpec`, `Segment`, `RequestShape`, `AnchorKind`,
  `ContentSpan`): per request shape (`query`, `document`, `pair`) an ordered list of `{fixed: ...}` /
  `{content: ...}` segments, special tokens written by name (`{special:<name>}`, resolved from the tokenizer's
  added tokens), the anchor the model reads its output from (`last | first | mean | marker`, with
  `anchor_markers`), and `add_special_tokens` per shape (the engine's behaviour for that route). `fit` measures
  the fixed overhead once per (template, shape), cuts only the content spans (offset-based, verified against
  the assembled render), re-attaches the template, chunks documents with the full template per chunk (ids
  `<id>#<k>`, chunk mapping), reserves declared `media_tokens` whole, and records every cut in the census under
  the new `text_budget` mechanism with `budget_source` and -- on chunked inputs -- the `max` aggregation. The
  tokenizer's SHA-256 is content (`TextBudget.identity(tokenizer)`), its name runtime; the template's canonical
  JSON is content. A hosted vendor profile without a tokenizer sends content uncut: its documented limit is
  recorded as the effective budget (`budget_source: vendor`; one row per (corpus, budget) per census, one
  warning per corpus per process).
- The role endpoint configs gain the text-budget fields (all CONTENT): `template`, `on_overflow`
  (`cut | chunk | fail`), `chunk`, `aggregation` (`max`), `empty_doc` (`send | omit_zero | send_text`, with
  `empty_doc_text`), and `request_shape` (`text | messages | token_ids`); `RerankEndpoint.instruction` gains
  `system`. A self-hosted role config (`api` in the new `rcp_ndcg.inference.SELF_HOSTED_APIS`:
  `openai_embeddings`, `vllm_pooling`, `rerank`) must declare `tokenizer` and `max_tokens` -- without them it is
  refused with a `ConfigError` whose hint shows the two fields; a hosted vendor profile may declare only
  `max_tokens` (its documented limit), a tokenizer without a number is refused for every role, and a profile
  without a tokenizer refuses everything that would be inert without one (`on_overflow` other than `cut`,
  `query_max_tokens`, `chunk`, `template`).
  `RerankEndpoint` refuses `query_max_tokens >= max_tokens` (the document's share would be non-positive), and
  `TextBudget` refuses it wherever the budget is resolved. `fit` refuses a tokenizer other than the one the
  budget declares (or none when the budget declares one), `query_max_tokens` on a non-`pair` shape,
  `media_tokens` without a tokenizer, and output ids that collide (one score would be pooled over the other);
  an input whose document splits into one piece keeps its own id.
- `rcp_ndcg.data` exports `TextBudget`, `TemplateSpec`, `Segment`, `FitResult` and `fit`;
  `rcp_ndcg.data.preprocess` additionally exports `BUDGET_DOC_ID`, `ContentParts` and
  `TextBudgetExceededError` (a `DataError`); `rcp_ndcg.inference` exports `SELF_HOSTED_APIS`.
  `TextTokenizer` gains `ids()` (token ids, optionally with the post-processor's), `count(...,
  add_special_tokens=)`, and the special-token lookup by name (`added_tokens`, `special_text`, `special_id`);
  `token_prefix` gains `add_special_tokens` (default unchanged). `ChunkPolicy` declares its field roles (all
  CONTENT) so a `TextBudget` feeds an identity.
- **Every retrieval role declares the media it sends**: `EmbeddingEndpoint`, `PoolingEndpoint` and
  `RerankEndpoint` gain the judge's `image_processor` (CONTENT), `max_images` and `max_videos` (RUNTIME, as on
  the judge: the server's per-request media limits are a gate, not a transformation) and the optional
  `image_policy` / `video_policy` (CONTENT; the judge's own `ImagePolicy` / `VideoPolicy` types, no copies),
  carried by a shared base `_MediaEndpoint`. One preparation path for every role that sends media:
  `rcp_ndcg.data.prepare.prepare_request(contents, image, video) -> PreparedRequest` (the prepared contents,
  every prepared item, and the request's exact media token counts, `MediaTokenCount`), and
  `rcp_ndcg.data.prepare.fit_media_to_budget(...)` -- the vision-block integrity rule: when media alone
  exceed a request's text budget, images shrink to the policy's minimum pixel
  budget, then whole items are dropped most-expensive-first, each with a census record (`MediaCensus.record`
  gains `dropped=`, and every media census row carries `dropped`); tokens are never cut inside a vision block.
- `rcp_ndcg.data.resolution` gains `engine_media_check(reported, counted)` and the typed
  `EngineMediaMismatch`: the pure comparison of an engine's prompt-token count for
  one prepared probe image against the counted one. `ImagePolicy` and
  `VideoPolicy` declare `IDENTITY_ROLES` (every field CONTENT: the media policy is the instrument), so a
  policy nested in an identity payload passes `check_declarations`.
- `VideoPolicy` gains `engine_video_pinning` (CONTENT, default false): whether the engine serving this corpus
  is pinned to sample exactly `num_frames` frames per container (vLLM `--media-io-kwargs`). Required for
  `wire: video_url`, refused under `wire: frames` (see below).

- `JobSpec` gains `phases` (a tuple of `JobPhase`: the engines one phase starts, by role, and the command it runs
  while they serve); exactly one of `argv` and `phases`: a job without phases runs `argv`, a phased job takes no
  `argv` of its own (its commands are its phases' `argv`).
- **`rcp_ndcg.inference` gains the embedding wire adapters and the embedding role client** (dense embeddings over
  one wire shape; no transport behaviour yet, so the client is exercised with a `Sender` a caller supplies):
  - `inference.adapters.embeddings`: four registered adapters of role `embed` — `openai_embeddings` (OpenAI
    `POST {base_url}/embeddings`: `model`, `input`, `encoding_format: "float"`, `dimensions` only when set;
    reply read from `data[].embedding` in `data[].index` order, float lists or base64 float32) and the hosted
    profiles `cohere` (v2 `POST {base_url}/embed`, `input_type` `search_query`/`search_document`, reply
    `embeddings.float`, cap 96), `voyage` (the OpenAI body with `input_type` `query`/`document`, cap 128)
    and `gemini` (`POST {base_url}/models/{model}:batchEmbedContents`, `taskType`
    `RETRIEVAL_QUERY`/`RETRIEVAL_DOCUMENT`, key in `x-goog-api-key`, reply `embeddings[].values`, cap 100).
    Names are scoped by role: the rerank role registers its own `cohere` and `voyage`, and an embed config's
    `api` resolves only among the embed role's names.
    Each profile carries its public base URL and key variables for when the config sets no `base_url`, and
    takes no `dimensions` parameter (its API fixes the output dimension): a config that sets one is refused at
    construction and the adapter refuses such a request, never silently ignored. Media raises
    `CapabilityError` naming the media type; a malformed reply (an index list that is not exactly one
    `0..n-1` per entry, an empty, scalar, non-finite or non-float32 embedding) raises `RequestRejectedError`; an
    over-length HTTP 400 ("maximum context length") maps to
    `CapabilityError` with a hint naming `max_tokens`/`batch_size`; HTTP 413 maps to `CapabilityError` naming
    `batch_size`; other 400/422 are `RequestRejectedError`; `usage()` reads the OpenAI-shaped token report
    (`usage.prompt_tokens`; the Cohere profile reads `meta.billed_units.input_tokens`), `None` when an API
    reports none.
  - `inference.clients` (new public module): `EmbeddingClient` — the role client for `EmbeddingEndpoint`. It
    applies `query_prompt`/`doc_prompt` per side through one `_prepare` seam, sends `dimensions` only when set
    (refused for the hosted profiles, which have no such parameter), L2-normalises when `normalize`, slices
    into `batch_size`-sized requests with at most `concurrency` in
    flight and reassembles in input order, resolves the API key from `api_key_env` (else the profile's
    variables) into the profile's header — an explicitly named but unset `api_key_env` variable is a
    `CredentialsError`, never a silent missing header — and refuses media, a `batch_size` over a profile's
    cap, an unknown or wrong-role `api`, and — until the text-budget mechanism is wired — any `max_tokens`
    (`ConfigError`: "max_tokens needs the text-budget mechanism, which is not wired yet"). An empty call makes
    no request and needs no key.
  - `inference.config`: `EmbeddingEndpoint.identity_extra()` returns `{"tokenizer_sha256": <sha>}` — the
    SHA-256 of the named tokenizer's `tokenizer.json` for a step identity, never its name (the name stays
    RUNTIME; the judge's rule for `JudgeConfig.tokenizer`). `check_declarations` is unchanged.
  - `rcp_ndcg.inference.__all__` gains `EmbeddingClient`.
- **`Endpoint.api_key_env` refuses an empty name** (`min_length: 1`; `schemas/index.v1.json`,
  `schemas/judge-config.v1.json`, `schemas/run-config.v1.json` regenerated): an empty variable name would
  silently send no credential header, for every role; ``None`` (unset) still sends no key.
- `rcp_ndcg.errors.WarningCode` gains `UNPINNED_REVISION` (an additive change to the closed list): a Hub dataset
  whose branch (or no revision at all) resolves to no commit — offline, or with the Hub unreachable, and no
  recorded ref in the local cache — warns with it, naming `--revision <full sha>` as the fix. With `--json` it
  shows in the envelope's `warnings`; otherwise it prints on stderr. `schemas/cli.v1.json`,
  `schemas/eval-report.v1.json` and the public-surface snapshot follow.
- **The rerank role goes on the wire** (in `rcp_ndcg.inference`, whose transport behaviour is still the
  transport work's; a role client used with an injected `Sender` runs today):
  - `inference.adapters.rerank`: the shipped wire adapters, registered at import and selectable from a config's
    `api` field -- `RerankAdapter` (`api: rerank`, the served Cohere-shaped `POST {base_url}/rerank`),
    `CohereRerankAdapter` (`api: cohere`, `https://api.cohere.com/v2/rerank`, at most 1000 documents per
    request) and `VoyageRerankAdapter` (`api: voyage`, `https://api.voyageai.com/v1/rerank`, at most 1000
    documents, requests of one query spaced half a second apart, no `top_n` -- Voyage's return-limit field is
    `top_k` and it returns every document by default), all subclasses of the new `RerankWire`.
    Requests are `model`, `query`, `documents`, `top_n`; the served engine's `instruction` and
    `use_activation` travel only when the config sets them. `interpret` parses the `results` and Voyage `data`
    answer shapes and realigns the scores by `index`; a bare list of rows (the shape SGLang and TEI answer) is
    a non-retryable `ProviderError` naming the engines with a hint to serve on vLLM, an index missing,
    duplicated or out of range is a non-retryable `ProviderError` naming the server, an over-length 400/422 a
    `CapabilityError` hinting `max_tokens`, any other refusal a `RequestRejectedError`. A candidate set above
    the cap (or a set `batch_size`) is split into requests and merged; a `listwise` config refuses to split
    (`CapabilityError`).
  - `inference.clients`: `RerankClient(config, *, sender=None)` with `rerank`/`arerank` (one query's whole
    candidate set per request, scores aligned to the input documents), `rerank_many`/`arerank_many`
    (`concurrency` queries in flight, the per-query `checkpoint(query_id, scores)` of today's served path)
    and `close`. The query text follows one rule for every path, from the config's `instruction` mode:
    `fold` (default) sends `Task: <instruction>\nQuery: <text>` exactly as the served path did, `field` sends
    the bare query plus the engine's `instruction` field (served only), `none` the bare query. Empty
    documents are sent as given (the hosted profiles' old empty-document filter is gone); an empty candidate
    set makes no request. Preparation runs through one seam (`RerankClient._prepare`); until the text-budget
    mechanism is wired the client cuts nothing and refuses a config that sets `max_tokens` with a
    `ConfigError`.
  - `Endpoint.identity_extra()`: `{"tokenizer_sha256": ...}` of the named tokenizer's `tokenizer.json`, the
    content identity a rerank step records (the name stays runtime, as the judge's already works).
  - The facade `rcp_ndcg.inference` additionally exports `RerankClient`, `RerankAdapter`,
    `CohereRerankAdapter`, `VoyageRerankAdapter` (and `rcp_ndcg.inference.adapters` re-exports them with
    `RerankWire`); `known_adapters()` now lists `cohere`, `rerank` and `voyage`.
- **`rcp_ndcg.inference` ships the first wire adapter, `vllm_pooling`, and its role client `PoolingClient`**:
  late-interaction (multi-vector) encoding over vLLM `POST {base_url}/pooling` with `task: token_embed`, the
  exact request and response field names verified against the vLLM entrypoints (the adapter's docstring cites
  them). The adapter sends `encoding_format: "base64"` with `embed_dtype` from the endpoint config and
  `endianness: "little"` (explicit, so a frame decodes the same on any server platform), one `input` batch per
  text-only request and one `messages` request per media item (the only shape in which the server applies the
  model's chat template to image placeholders). `interpret` accepts nested float lists (shape as sent), flat
  base64 frames (reshaped to `(tokens, dim)` from the declared `dim`) and the framed `bytes` encoding (per-item
  `start`/`end`/`shape` metadata from the response header; `bytes_only` has no framing and is refused, naming
  the lane that will pin it). Over-length HTTP 400 and 422 refusals raise `CapabilityError`; any other client
  error raises `RequestRejectedError`; a reply whose decoded token counts disagree with its own
  `usage.prompt_tokens` (a `token_embed` answer has one vector per prompt token) raises `ProviderError` — a
  mistyped `dim` is a loud error, never a silently mis-shaped corpus. `maxsim_topk` no longer raises
  `IndexError` when an empty item sits at the end of a ragged buffer.
- **`PoolingClient`** (`rcp_ndcg.inference.clients.pool`, exported from `rcp_ndcg.inference`): a
  `PoolingEndpoint` plus a `Sender` becomes ragged `Embeddings`. It prepends the role's prompt (`_prepare`, the
  one seam the text budget will join), splits into `batch_size`-sized requests, keeps at most `concurrency` in
  flight and reassembles in input order, L2-normalises per token when `normalize` is set (in float32, stored
  back in the transfer dtype), and refuses a config that sets `max_tokens` (`ConfigError`: the text-budget
  mechanism is not wired yet, and a budget silently ignored would change the vectors). The sync `encode` runs
  on the sender's own bridge when it has one (`Transport.run`), else on a fresh event loop.
- **`PoolingEndpoint` gains `dim`** (CONTENT): the checkpoint's token-vector width, needed to reshape the flat
  base64 frame of `/pooling` (which carries no shape); the float and bytes encodings are self-describing, and
  the `bytes` encoding makes it unnecessary. `dimensions` is never sent — vLLM's `/pooling` refuses it.
- **`PoolRequest` gains `embed_dtype`** (default `"float16"`, the owner's Q11 decision; `"float32"` opt-in)
  **and `dim`**, both copied from the endpoint config by the client and consumed by the adapter.
- **`Embeddings` keeps the transfer dtype for ragged buffers**: `ragged(per_item, *, dtype=np.float32)` and
  `empty(dim, *, multi_vector=False, dtype=np.float32)` accept the storage precision, so a multi-vector buffer
  stays float16 end to end (2 bytes per token vector, against 4 for float32); single-vector buffers are
  float32 as before. `l2_normalize` computes in float32 and returns the input's dtype (float32 in, float32
  out; float16 stays float16).
- **`maxsim_topk` accepts float16 or float32 vectors** and computes every dot product and per-query sum in
  float32, upcasting one query block and one document block at a time — never a float32 copy of the whole
  corpus (each block copy is bounded by the 64 MiB tile budget, alongside the score tile). Results for float32
  inputs are unchanged; float16 inputs match a float64 reference within 1e-3 relative on 2,000-token documents.
- `rcp_ndcg.inference.adapters` exports `VllmPooling`, and the shipped adapters register when that package is
  imported (`known_adapters()` now reports `vllm_pooling`).
- `tests/contract/snapshots/python_api.json` regenerated; it also records the already-committed additive
  `JobSpec.phases` field, which its own commit left out of the snapshot.
    `RerankWire`); `known_adapters("rerank")` now lists `cohere`, `rerank` and `voyage`.
- `schemas/run-config.v1.json`: the `CandidatesConfig` description states that the whole section is content for
  the step identities (its `IDENTITY_ROLES` declarations); no property changed.
- **New public module `rcp_ndcg.inference`**: the inference layer between `rcp_ndcg.data` and
  `rcp_ndcg.retrieval`, with the wire types, the adapter seam, the transport, the probe and the offline fakes
  the roles (the judge, the encoders, the rerankers) build on.
  - `inference.endpoint`: `Endpoint` moved here from `rcp_ndcg.support.endpoint` (that module is deleted), with
    new fields `api` (CONTENT; the wire adapter, each role config sets its default), `headers_env` (RUNTIME;
    header name -> environment variable name, values read from the environment only) and `wait_on_outage_s`
    (RUNTIME; moved up from `JudgeConfig`, which keeps it through inheritance). Every earlier field and validator
    is unchanged.
  - `inference.types`: the wire types `Call`, `Reply`, `TokenCount` and `Usage` (with `__add__`); `EngineInfo`,
    `CompletionInput` and `Completion` moved here from `rcp_ndcg.judging.client` unchanged (they stay importable
    from `rcp_ndcg.judging.client`, where the first two and the two error types remain in its `__all__`);
    `EncodeRole`, `Embeddings` and `l2_normalize` moved here from
    `rcp_ndcg.retrieval.encoder` (re-exported there and from `rcp_ndcg.retrieval`); and the new role request and
    result types `EmbedRequest`, `PoolRequest`, `RerankRequest` and `RerankResult` (whose
    `RerankResult.aligned(request, scores)` refuses a score count that does not match the request's documents).
  - `inference.adapters`: the `Adapter` protocol (generic in request and result), the `AdapterRole` literal
    (`"judge"`, `"embed"`, `"rerank"`, `"multi_vector"`) and its registry
    (`register_adapter`, `get_adapter`, `known_adapters`, constant `ADAPTER_ENTRY_POINTS =
    "rcp_ndcg.adapters"`). The `vllm_pooling` adapter ships below.
  - `inference.transport`: the `Sender` protocol and the `Transport` class -- the transport's frozen interface
    only (`send`, `probe`, `run`, `aclose`); its routing, retries, parking and status-map behaviour is the transport lane's.
  - `inference.fake`: `FAKE_SCHEME = "fake://"` and the offline fakes' contract; no implementation yet.
    "rcp_ndcg.adapters"`). No adapter is registered yet.
  - `inference.transport`: the `Sender` protocol and the `Transport`, now implemented -- the judge client's
    behaviour over `httpx` instead of the OpenAI SDK, with the same numbers: least-busy replica routing, a
    semaphore and an HTTP pool sized to `concurrency`, the within-request retries (exponential backoff, the
    server's `Retry-After` honoured, capped at 60 s), the set-aside of a failing replica (5 s doubling to 60 s),
    parking until `wait_on_outage_s` (the outage clock starts when the request holds a slot; the message states
    how long the endpoint was unavailable), the rejection rule, and the shared status map (outages retried then
    parked; 401 and 403 raise `CredentialsError`; 404 raises a non-retryable `ProviderError` naming the URL and
    the model; every other 4xx is returned as a `Reply` for the adapter to interpret). Credentials and
    `headers_env` are read from the environment at send time and never logged; JSON bodies are decoded and
    `application/octet-stream` stays `bytes`; a caller-supplied `httpx_transport` is wrapped in the transport's
    own client with the endpoint's timeouts and pool limits. The sync bridge `run` reuses one event loop and
    one pool across calls and runs on a private background thread when a loop is already running in the thread;
    `aclose` (and `close`, and the context manager) close the pool. `Transport` carries the provenance probe
    (`probe`, `engines`, `note_system_fingerprint`, over the new module `inference.probe`'s `read_replica`) and
    the usage accounting (`usage`, and `add_usage` for the tokens the adapter's `usage(reply)` reports).
  - `inference.fake`: the offline fakes, implemented. A `fake://` base URL makes the transport send through an
    in-process `httpx.MockTransport` speaking each role's wire: `GET /models`; `POST /embeddings` (OpenAI shape;
    deterministic hash-seeded unit vectors, dimension from the URL's `?dim=` query, default 64, cut to a
    request's `dimensions`); `POST /pooling` (vLLM `task: token_embed`; ragged per-token vectors, as floats or
    base64-packed in the request's `embed_dtype`, default `float16`); `POST /rerank` (Cohere shape; each
    document scored by the same hidden ability the fake judge reads, so a tiny run's rerank and judge agree).
    `register_fake_route(method, path, handler)` registers extra routes (the judge's chat completions arrive
    with the judge port; a route path must name its route, e.g. ``/chat/completions``); the shared draws
    `fake_uniform` and `hidden_ability` are the fake judge's too. The
    fakes sit below the transport, so routing, retries, parking and usage run in every offline test; the
    package exports `register_fake_route` and `FakeEndpoint`.
  - `inference.probe`: `read_replica`, one replica's best-effort `GET {url}/models` into `EngineInfo` (an
    unreadable endpoint recorded with its `error`, never raised; a server that does not list the endpoint's
    model named in a warning).
  - `inference.config`: the role endpoint configs `EmbeddingEndpoint` (`api` default `openai_embeddings`),
    `PoolingEndpoint` (default `vllm_pooling`, with `embed_dtype: float16` by default, `float32` opt-in) and
    `RerankEndpoint` (default `rerank`, `instruction: fold` default, a `batch_size` refused for a `listwise`
    model). Not wired into `rcp_ndcg.retrieval.config` yet
- **`RerankEndpoint` gains `query_max_tokens`** (CONTENT): the query's share of the pair budget
  (`max_tokens`), with the document getting the rest; `None` (the default) declares no split and leaves it to
  the adapter's recipe. The `max_tokens` docstring of every role config now states exactly what the budget
  counts: the model's whole input sequence as the engine sees it (template, special tokens, instruction and
  content), with the content cut on the client so the fixed template tokens (the anchors) always survive --
  truncation is never left to the engine.
- **`rcp_ndcg.errors` gains `BackendUnavailableError` and `RequestRejectedError`**, moved unchanged from
  `rcp_ndcg.judging.client` (still importable and exported there). Exit codes do not change: both remain
  `ProviderError` subclasses at `PROVIDER`, `RequestRejectedError` non-retryable.
- **`rcp_ndcg.errors` gains the shared status map**: `UNAVAILABLE_STATUSES` (408 and 429), `status_is_unavailable`
  and `status_error` (401/403 → `CredentialsError`; 404 → a non-retryable `ProviderError` naming the URL and the
  model; every other 4xx → `None`, a reply for the wire adapter). One table, in one place, for every role's
  transport.
- **`Endpoint.base_url` takes a replica list** (one URL, or a non-empty list of replicas of the same served
  model, without duplicates, never mixing the offline fakes with real URLs), and `Endpoint` gains the `urls`
  property; the widening moves the judge's list normalisation onto `Endpoint`, whose `JudgeConfig` keeps its own
  (required, and unchanged in behaviour and identity payloads). The retrieval layer's hosted configs
  (`rcp_ndcg.retrieval.config._Hosted`) keep `base_url` a single optional URL until the retrieval port: a
  replica list is refused there.
- **`rcp_ndcg.support.serve` gains the serve-by-role types**: `EngineRole`, `EngineConfig` (an alias of the
  unchanged `ServeConfig`), `ServeByRole`, `Phase`, `ENGINES_ENV = "RCP_NDCG_ENGINES"`, `EngineURLs`,
  `parse_engines_env`, and `plan_phases(steps, serve, uses)`, the pure phase plan.
- **`rcp_ndcg.judging.client` gains `api` and `headers_env`** through `Endpoint`; `wait_on_outage_s` moves up to
  `Endpoint` and the judge keeps declaring it only through that inheritance. A judge's identity payload is
  unchanged: `api` defaults to `None` (omitted from identities until a role config sets it), the other two are
  runtime fields.
- New layering charter (`AGENTS.md`): `data → inference → retrieval`; enforced by the new
  `tests/test_layering.py` (eager imports only; the current tree has no outward import).
- **The judge is ported onto the shared transport and the `openai_chat` wire adapter**:
  - `inference.adapters.chat` (new module, registered at import of `rcp_ndcg.inference.adapters`):
    `OpenAIChat` (role `judge`, name `openai_chat`) -- the judge's wire adapter. `calls()` builds the
    `POST {base_url}/chat/completions` body exactly as the OpenAI SDK built it (`messages`, `temperature` only
    when set, `max_completion_tokens` from `max_output_tokens`, the window's `response_format`, `extra_body`
    merged at the top level) and runs the pre-send media gate against `max_images`/`max_videos`; `interpret()`
    reads the first choice's `message.content`, the reasoning channel from `reasoning_content` or `reasoning`,
    `finish_reason` and `usage`, runs the reasoning watch under an answer schema (one warning after 8 answers
    without reasoning), and maps the refusals: a media-count text (HTTP 400/422) and a refusal of the answer
    schema are `CapabilityError`, every other returned 4xx and an answer with no choices are
    `RequestRejectedError`; `usage(reply)` reads the token report; `fingerprint(reply)` reads
    `system_fingerprint`. One wire nuance moves with the SDK: a judge without `api_key_env` no longer sends
    `Authorization: Bearer EMPTY` on its requests (the SDK always did); the transport sends credentials only
    from the environment variables the config names.
    The judge's message/media lowering (`build_messages`, `media_counts`,
    `MAX_VIDEO_BYTES`, `VIDEO_CACHE_SIZE`) moved here unchanged from the internal `rcp_ndcg.judging._payload`
    (deleted; the layering forbids `inference` importing `llm`), importable at the new home.
  - `JudgeClient` keeps its public API (`from_config`, `complete`, `probe`, `engines`, `model`, `usage`) and is
    thin: it builds the adapter from `api` within the judge role's registry (the role-scoped registry refuses a
    name of another role with that role's known names in the hint; unset resolves to `openai_chat` and stays
    out of the identity payload, so every identity is unchanged) and sends through the shared `Transport`
    (routing, retries, parking, credentials, token usage).
    parking, credentials, token usage). Its constructor takes `httpx_transport` (the transport wraps it with
    the endpoint's timeouts and pool) instead of `http_client` (which used to replace both); `config` and
    `usage` are properties now (assigning a `model_copy` of the config rebuilds the wire at the next call);
    the client-level `usage` keeps its shape (requests, failed_requests, tokens; `cached_input_tokens` stays
    at 0 -- the wire reports tokens and calls only; the old client counted the endpoint's
    `prompt_tokens_details.cached_tokens`), and a request refused while its body is built (an unprepared
    image, an oversized video container) counts as neither a request nor a failed request, where the old
    client counted it as failed -- it is refused before the transport is engaged, like the media gate.
    `is_unavailable` and the judge's private status table are gone: the shared status map in
    `rcp_ndcg.errors` is the one home.
  - `JudgeConfig.api` stays unset by default (the judge's `openai_chat` wire is resolved from it), so judge
    identity payloads are byte-identical: no shipped preset, and not `JudgeConfig.fake`, changes key.
  - The offline fake judge answers behind the transport: `fake://` endpoints answer `POST /chat/completions`
    through the route registered by `rcp_ndcg.judging._fake`, so `JudgeConfig.fake(seed)` builds a real
    `JudgeClient` over the real transport; `rcp_ndcg.testing.FakeJudge` stays importable and keeps its ability
    mapping and severity, answering through its own in-process endpoint below the transport with the same
    answer logic (its test doubles override `FakeJudge._answer`, the wire handler, where they used to override
    the client's `_send`). One numeric edge, declared: the fake reads the prompt rebuilt from the lowered
    request blocks, so a window whose clip is judged as sampled frames draws from a changed key (the prompt
    carried one marker per part, the wire carries one block per frame); text, page-image and whole-container
    windows round-trip exactly, and the tiny world's judgements are byte-identical.
- **One `Usage` for every role**: the run manifest's requests-and-tokens shape
  (`requests`, `failed_requests`, `input_tokens`, `output_tokens`, `cached_input_tokens`; frozen; merged with
  `merged_with`) is the one type, defined in `rcp_ndcg.inference.types` and re-exported from
  `rcp_ndcg.judging.client`; the transport's accumulator produces it (its former `calls`/`failed_calls`
  vocabulary is gone, renamed to `requests`/`failed_requests` with the same accounting semantics), and the
  judge client maps its own answers and refusals onto it. The run manifest's serialised usage fields are
  unchanged; no property changed.
- **`Reply` gains `url`** (default `None`): the replica base URL that answered, set by the transport -- a role
  client needs it to record a per-replica fact such as a completion's `system_fingerprint` (the judge calls
  `transport.note_system_fingerprint(reply.url, ...)` for its first completion per replica, as it did).
- **`serve:` names one engine per role, and a job runs the run in phases** (each phase starts only the engines its steps use).
  `RunConfig.serve` is a `ServeByRole` (`judge`, `encoder`, `reranker`; the old single-engine mapping is refused
  with a hint showing the new shape), and `plan_phases(steps, serve, uses)` builds the phase plan: consecutive
  steps that call the same served engines share a phase, steps that call no served engine form an engine-free
  phase, and the paper run becomes four phases. `RunConfig.engine_uses()` derives the per-step engine roles from
  the config. `runs.execution` builds the job's `JobSpec(phases=...)`: per phase the engines by role and the
  coordinator argv `rcp-ndcg run resume --run <dir> --only <steps>`; `run_argv` lost its outage argument —
  `wait_on_outage_s` travels in `RCP_NDCG_ENGINES` per role now. A runner that neither renders nor runs phases
  refuses a job that would start engines; the local runner runs the engine-free phases and refuses the ones with
  engines. `rcp_ndcg.runners` exports `JobPhase`, the per-phase engine set and coordinator command.
- **The `slurm` and `kubernetes` runners render a job's phases** (`renders_phases`), so a serving run submits
  instead of being refused. A job with phases runs them in order in one allocation: each phase that starts engines
  runs one supervision block — it starts the phase's engines once (no restart), waits until every role has a
  replica answering its readiness path, exports their URLs in `RCP_NDCG_ENGINES`, runs the phase's coordinator,
  stops and reaps its engines, and only then starts the next phase; a phase without engines runs its command
  directly. Any failure ends the job with the single-engine semantics (`ENGINE_FAILED`, fail-fast supervision,
  `SIGTERM`/`SIGKILL` cleanup, and an engine that ends non-zero before the coordinator's exit is observed fails
  the phase even where `wait -n` would miss it).
  - `runners.script`: `supervise(engines, *, coordinator, engines_env, uv)` renders one phase; the engines are
    `EngineStep(serve, role, start, hosts)` entries, and a `start` of `None` waits for replicas that are already
    running elsewhere (a Kubernetes StatefulSet). New `engines_env_value` (the phase's JSON) and, for hosts the
    script only learns when the job starts, `engines_env_spec`/`engines_env_command`; the readiness probe
    (`wait_for_replicas`, whose signature gains `pid_var`) is parameterised by the engine's pid variable.
  - SLURM: one `sbatch` asks for the maximum nodes and GPUs over the phases; each role's engines run as one
    `srun --overlap` step pinned to its slice of the allocation's nodes; a one-node allocation answers on
    `localhost`. GPUs are partitioned among the engines of a phase (below, [serving](docs/concepts/runs.md)).
  - Kubernetes: each engine phase is an init container whose engines run in one container of the (single)
    engine's image (a phase's engines share one image and, if several, need distinct ports), the last phase the
    main container; several-replica engines are StatefulSets owned by the Job as before, run-scoped, named
    `<job>-engine-<role>`. A phase that starts one engine names its role in the failure message; with several,
    the message says an engine exited.
  - The single-engine `serve:` rendering (the `RCP_NDCG_JUDGE_URLS` export) is gone, with the deprecated
    `support.serve.JUDGE_URLS_ENV` alias; a run's `serve:` reaches the job only as phases.
  - Co-located engines partition a node's or container's GPUs (RFC review R31): a node's request is the **sum** of
    what runs on it — the coordinator's own `resources.gpus` plus each engine's, per replica — and the job asks for
    the maximum of that over the phases; every co-located engine process gets a disjoint `CUDA_VISIBLE_DEVICES`
      `device_slices`), the coordinator's devices reserved first, and an engine without GPUs gets the empty slice.
    On SLURM the engine steps are pinned to disjoint node slices (one replica per node) and each step's `--gres`
    is its own engine's count, so SLURM's per-step device assignment — which `srun --overlap` may let overlap —
    never co-locates two engines; the coordinator's task claims its own `--gres` (its reservation) instead of
    srun's default all-of-the-job GRES.
- **`RCP_NDCG_ENGINES` is the runtime overlay that carries the engines' URLs to the steps.** The coordinator
  applies each role's `urls` and `wait_on_outage_s` to the role config in memory — never written into `run.yaml`,
  never in a step identity, so a run is byte-identical with and without the variable; the
  `--set judge.wait_on_outage_s=<outage_timeout_s>` job argument is gone. A failed resume keeps treating the
  injected URLs as no config change: `_substance` subtracts every declared RUNTIME field of the candidates'
  nested configs too.
- **Served-role refusals.** A role config whose engine is served must not set `base_url` (the job's URLs for it
  reach the step at runtime; setting both is refused, never silently overridden): enforced for `encoder` and
  `reranker`, whose `base_url` is now optional, omitted exactly when served; a retrieval client built without a
  URL is refused rather than silently addressing a vendor's public API. A hosted or in-process model, a BM25
  retriever, a role no step of the run calls, and more than one replica for a retrieval role are refused with a
  hint. A served encoder needs no judge (the old any-`serve:` check is gone); `serve.judge` needs a real judge
  (not `fake`) and a judging step. The judge's own `base_url` stays required (the judge client requires it) and
  is the placeholder the job's runtime URLs replace. `EngineURLs` refuses a replica listed twice.
- **`rcp-ndcg doctor --endpoint <url>`** replaces `--judge-url` and probes any role's endpoint URL
  (`GET <url>/models`).
- The `ServeConfig` fields' schema descriptions are role-neutral (the same engine shape serves the judge, the
  retrieval encoder and the reranker); no property changed.
- New plugin distribution `rcp-ndcg-vllm-pplx` (`rcp-ndcg-vllm/plugins/pplx/`, pure Python, dependency-free):
  registers the `PplxContextualModel` architecture (perplexity-ai/pplx-embed-v2-context-9b-preview,
  revision `b667039e`) with stock vLLM v0.31.x through the `vllm.general_plugins` entry point
  (`rcp_vllm_pplx:register`), so the unmodified `vllm/vllm-openai:v0.31.0` image serves it after
  `pip install --no-deps <wheel>`. Registers one lazy model class (`PplxContextualForPooling`, one embedding per
  chunk via a boundary-marker segment pooler and the checkpoint's fp32 `contextual_projection` + int8 tanh head,
  no vision tower, no lm_head) and a `MODELS_CONFIG_MAP` handler that forces the checkpoint's
  `is_causal=false` attention contract on both HF configs. At import it refuses any vLLM outside
  `>=0.31,<0.32`. The client contract (token ids with the role prefixes, per the plugin README) needs the
  product's `request_shape: token_ids` (the `vllm_pooling` wire sends it), and the
  recipe's chunker must not emit the `<|chunk_sep|>` marker as chunk content (the id wire cannot tell it from a
  boundary; declared in the plugin README).
- The served recipe `jina-reranker-v3` (`rcp-ndcg-vllm/recipes/jina-reranker-v3/`,
  jinaai/jina-reranker-v3 @ d7d7e73b6ea138ced340b83865931b5dfb6c97aa, listwise rerank, paper-exact):
  stock `vllm/vllm-openai:v0.31.0` with `--runner pooling` (the native `JinaForRanking`; no conversion,
  no plugin, and deliberately no chat-template file — the listwise prompt is built server-side by the
  engine's Python builder, which the declared pair template mirrors byte for byte), the pair budget
  `max_tokens` 3219 (the measured 147-token frame + `query_max_tokens` 512 for EACH of the query's
  two spans + the checkpoint's per-document 2048 — the checkpoint's own worst-case 1-vs-1 prompt), `instruction: none`, `use_activation: false` (raw cosine), `empty_doc:
  omit_zero`, and a reference subprocess derived from `experiments/paper/rerankers/reference/jina.py`
  (raw cosine in [-1, 1], empty documents 0.0, the checkpoint's 125-doc/2048-token blocking ported for
  the GPU waves). Stage 1 passes on CPU against the real tokenizer; the anchor mutation turns the
  audit red. Status `unverified` until the GPU waves run the harness's stages 2–3.
- New recipe `qwen3-vl-reranker-2b` (`rcp-ndcg-vllm/recipes/qwen3-vl-reranker-2b/`, shipped in the sdist):
  Qwen/Qwen3-VL-Reranker-2B as a pointwise reranker on the stock `vllm/vllm-openai:v0.31.0` pooling runner --
  three-key `hf_overrides`, the served chat template as data plus a shipped template file (rewritten to the recipe
  variable convention, byte-equal under the engine's render), the explicit 8192-token budget with a 4096 query
  share, `instruction: none` (the card's default instruction pinned as fixed frame text), `use_activation: true`
  (probability), `empty_doc: send_text "NULL"`, and `mm_processor_kwargs` min_pixels 4096 / max_pixels 1310720
  (1280 tokens/image, R20). `reference.known_deviations: [anchor_drop_over_cap]`: the card's script truncates
  over-cap pairs itself. Status `unverified` until the GPU waves run stages 2-3.
- **The embedding and pooling media allowances count from the item shape's own budget.** Under a declared
  `query_max_tokens` a query's media were fitted against `max_tokens` while the text fit measured the query
  against its share, so an image that fit `max_tokens` but not the share was kept whole and the request was
  refused (`the fixed template overhead ... plus the declared media ... already fill the budget`). The embed
  role's `messages` route now fits its media like the pooling route: one preparation sliced per item (a second
  preparation re-inlined the prepared bytes and recorded census rows against `data:` URIs), the kept media
  recorded, and the allowance reserving the shape's fixed frame.
- **A hosted rerank profile under `empty_doc: omit_zero` scores the documents after an omitted one.** The
  vendor path (a documented `max_tokens`, no tokenizer) named its fit outputs by kept position while the
  client reads original positions, so any omitted document before the last raised a bare `KeyError`.
- **The judge's engine media check is the served roles' delta check.** The judge's probe now sends the probe
  image and the same request without it, so a served chat template cancels and an honest engine passes; the
  delta counts no text, so the check no longer needs (or loads) the judge's tokenizer.
- **The rerank pair fit with media settles one query span for the whole batch and never ships a pair over the
  budget.** The settlement probe reserved only the query's media while each pair's fit re-cut the query
  against its own media-reduced cap, so a candidate set with one plain and one media document crashed on the
  fit's own consistency check (`DataError: the pair fit settled the shared query differently ... this is a
  bug`), and a lone media pair shipped the probe's un-cut query beside its media -- over `max_tokens`,
  leaving the truncation to the engine. The request is now prepared once and sliced, the media fit runs
  against an allowance that reserves the fixed template overhead (never the bare `max_tokens`, which left a
  dead zone where the media fit and the text fit disagreed) and `fit`'s own pair invariants, the query's
  media are decided once for the batch, the probe settles against the documents' maximum media count, and
  the span the pair fit verified is the span the wire carries, re-checked against the budget before send.
- **The engine media check compares the engine's media DELTA, not its whole prompt.** The old check demanded
  exact equality between the engine's `usage.prompt_tokens` and a client count that excludes the chat
  template and the post-processor specials every real route adds -- any correctly serving engine failed the
  startup gate. The probe now sends the prepared image and the same request without its media, and the DELTA
  of the two reports (template and text cancel) is compared with the counted media tokens. The probe's bare
  assert is gone (the delta needs no tokenizer) and the key-name-heuristic text walker went with it; the
  passing-path tests pin hand-verified counts, not the client's own arithmetic.
- **Retrieval-role usage is forwarded and recorded**: the three role clients fold every reply's token report
  into the transport with `add_usage` (the judge's per-reply rule) and `RoleClient.usage` exposes it -- an
  embed/pool/rerank run records the tokens its replies reported, never zeros.
- **`Transport.run()` is thread-safe**: concurrent sync callers queue on a bridge lock instead of racing two
  `run_until_complete` passes on the shared loop (the second died with `This event loop is already running`
  and its batch aborted); a `close()` from another thread waits for an in-flight sync-bridge call instead of
  raising `CancelledError` into it, and defers the pool's close to the last in-flight call on the background
  thread a `run()` inside a running loop uses (never closing the pool under it, and never waiting for a call
  that is itself awaiting the close -- code inside such a call may hand `close()` to an executor), and a rebinding concurrency gate closes the previous loop's pool best-effort.
- A server `Retry-After` of `nan` (sleeping forever, wedging the request inside the retry loop), a negative
  value (hammering the rate limiter) or any other garbage falls back to the doubling backoff; a usable value
  is clamped to the retry cap.
- URL userinfo and query are stripped from every log line, error message and engine record (`safe_url`,
  an `EngineInfo` validator, and the outage message naming the failure's own words instead of the raw httpx
  exception whose text carries the request URL).
- `MediaCensus` survives a torn last line of its shared log (the failure class the judgement store's journal
  repairs): the tail is skipped with a warning, rows are read defensively, and the dedup key names the
  outcome (`dropped`) beside the source, so a kept row never hides a later drop; records are serialised.
- `apply_media_fit` carries `frame_indices` through frame drops: a video whose frames the budget dropped no
  longer ships a part whose `frame_indices` still record the pre-drop sampling (the next count refused it as
  a corrupt corpus record).
- One home for the inline media: every `data:` URI the package writes is built by
  `rcp_ndcg.data.media.data_uri`; the chat adapter reads the container bytes through the media resolver; a
  non-base64 `data:` URI is a typed `MediaError` (not `media not found`); `hydrate` takes the inline path for
  every data URI (hash or not) and never wipes a partial dimension record; a truncated-but-box-structured
  MP4/MOV header is probed as far as it goes (`recorded unprobed`) instead of raising bare
  `struct.error`/`IndexError`.
- The `/pooling` reply's corners are refused by name: duplicated or partial `index` rows are validated like
  the embeddings parser's rule, a framing metadata header that is JSON but not an object is a
  `ProviderError` (not a bare `AttributeError`), a usage dict without `prompt_tokens` refuses the dim
  cross-check instead of silently skipping it, and an unknown `endianness` is a typed refusal (a mislabeled
  frame silently decoded little would be garbage floats).
- The hosted batch cap (`MAX_BATCH`) applies to HOSTED use only: a served `openai_embeddings` engine answers
  its own over-count batch (HTTP 413, mapped to a typed `CapabilityError` naming `batch_size`) instead of a
  stale client-side 128 refusing a batch the engine would serve. The `OpenAIEmbeddings` 128 constant is gone
  (its check could never apply to a served engine and its number is stale even for the hosted route, whose
  own over-count refusal, an HTTP 400, surfaces as a `RequestRejectedError` with the API's message).
  The rule is the client base's, one for every role: the pooling client used to refuse above any declared
  cap, a served `/pooling` wire's included.
- `TextBudget.identity` requires and verifies the budget's loaded tokenizer: the SHA-256 is the one field it
  exists to carry, and two tokenizers are not told apart by name (a budget without its loaded tokenizer used
  to produce colliding identities).
- The fake engine speaks the wire claims its docstrings make: `usage.total_tokens` equals `prompt_tokens`, a
  rerank document keyed by `id` draws that document's ability, and `/pooling` honours the request's
  `endianness`.
- The media gates' `CapabilityError`s carry hints naming the field to change, and `DocStub`-level over-claims
  are corrected (`MediaFit.tokens`'s exact-vs-bound count, the truncation census' "two mechanisms" listing
  three).
- **MaxSim no longer corrupts or drops empty items** (`rcp_ndcg.retrieval.maxsim`): the reduceat grouping
  clamped a trailing empty item's start into the last column/row, which truncated its predecessor's score by
  one token (a 2-token item next to an empty one scored over all but its last token, on either axis), and a
  block whose every item was empty was skipped whole, leaking the running `-inf`/`-1` placeholders into the
  returned `(scores, indices)` whenever `k` reached them. The reduction now runs over the non-empty items'
  starts and every block scores: an empty document carries its designed sentinel score (finite, strictly below
  every real one) with a real index, and an empty query scores 0 against every real document, so an all-empty
  query set comes back as 0.0/real indices instead of no ranking.
- **The rerank checkpoint key covers the reranker's content and the exact texts**
  (`rcp_ndcg.retrieval._api._checkpoint_key`): every CONTENT field of the config (model, revision, `api`,
  recipe, instruction mode, `use_activation`, the budgets), the tokenizer's SHA-256, the query's raw text and
  instruction, and a digest of the candidate contents -- where it named only the model, revision, historical
  budget constants and ids, so a rerun after any content change silently resumed stale scores. The record
  format is unchanged; a checkpoint written before this key existed is scored again, not resumed (nothing is
  released yet, so no checkpoint in the wild carries the old key).
- **The gains' keying rule holds for `count_gains` and for mixed styles** (`rcp_ndcg.eval.evaluate`):
  bare query ids over subsets that share them are refused for `count_gains` as they were for `gains` (they
  silently scored one subset's query with another subset's document gains); gains that mix
  `"<subset>/<query_id>"` keys with bare ids for one subset are refused (the prefixed ones won and the
  bare-keyed queries silently lost their gains and their RCP rows); a query with gains but no qrel row now
  counts as labelled for a suite's bare-key fallback (the module's labelled-query definition), so its RCP
  value is scored instead of silently dropped; and gains -- RCP or count -- that match no labelled query are
  refused, as they were for RCP gains.
- **The documented tie rule decides the top-k cut as well as the order** (`rcp_ndcg.retrieval.topk.select_topk`):
  `argpartition`'s pick among the candidates tied at the k-th score was arbitrary, so a tie class straddling
  the cut could drop a lower-index document in favour of higher-index ones; the cut now resolves that tie class
  by ascending index, as the documented "ties break toward the lower document index" says, for `numpy_topk` and
  `maxsim_topk` alike.
- **Depth is validated where it is used**: `rerank(depth=0)` raised the unrelated "rankings hold 0 systems"
  and `rerank(depth=-1)` silently kept all but the last candidate (pandas' `head(-n)`); both are refused with
  `ConfigError` like `search` does, and `Rankings.top` refuses a non-positive `depth` at the root.
- **A checkpoint record that misses a document is re-scored, not a dead loop**: resuming a record whose score
  map did not cover its example's candidates failed with a refusal whose hint -- rerun the rerank -- replayed
  the identical failure on every run, and a non-numeric score value raised a bare `ValueError`; the reader now
  drops an incomplete or unparseable record (its own record only) and the query is scored again.
- **`retrieve(out=...)` rebuilds over an `index.json` it cannot read** (an earlier release's shape, or a
  corrupt one) instead of failing there forever, which is what its own `IdentityError` hint promises.
- **A cutoff a report never computed is refused everywhere**: `leaderboard(metric, k)` silently returned an
  all-NaN table and `compare`/`sensitivity` failed with a misleading "share no scored query"; one `DataError`
  naming the cutoffs the report has now guards `leaderboard`, `compare` and `sensitivity` alike.
- **`explain`'s deltas exist without RCP gains**: on a qrel-only report `deltas` was silently `[]` although
  the docstring and the docs promise the gap between every system and the first; the gaps are now computed
  from the query's qrel grades when it has no RCP gains (the displayed `gain` values stay the RCP ones).
- **A BM25 corpus with no indexable tokens is a `DataError` with a hint** (every document empty, or only stop
  words after removal): it crashed inside bm25s with a bare `ValueError: max() iterable argument is empty`;
  and bm25s' tqdm progress bars no longer print from library code.
- **`mteb.get_tasks` refuses `names=[]`** (it meant "all subsets") **and a repeated subset name** (it built
  the task twice); `ndcg_float_scores` raises `DataError` for a query without gains, as its docstring always
  claimed (the code raised a bare `KeyError`).
- **`retrieval fuse` fuses per-subset files**: one file per (system, subset), as a per-subset fan-out writes
  them, fuses each subset from the files that name it instead of failing with "no rankings of dataset" naming
  the wrong datasets; a ranking with no rows for a subset stays out of that subset's fusion, and no rankings
  at all is refused before the loop.
- **A BM25 index directory that cannot be searched is refused by name**: one written in the earlier build's
  pickle format (loading it would run its code), one without the `meta.json` that names its stemmer, and one
  whose `meta.json` is unreadable or stemmer-less are all `MissingInputError` with a rebuild hint (the last
  two were bare file errors).
- **The storage layer's containment, `file://` handling and cache freshness**: `storage.relative` checked
  containment with a raw string prefix, so `..` escaped it
  (`relative('/base/root/../../etc/passwd', '/base/root')` returned `'../../etc/passwd'` instead of the
  documented `DataError`) — both sides are normalised first and an escape is a `DataError`. The local fast
  paths answered `file://` URIs literally (`Path('file:///x')` is a relative directory named `file:`), so
  `exists` was `False` for a live file, `info`/`get` raised `FileNotFoundError` naming the literal `file:/...`
  string, and `makedirs` grew a junk `file:` tree in the working directory while never creating the real one;
  they strip the scheme through `local_path` now. The remote-object cache re-downloads when a same-length
  remote overwrite changes a freshness field it had not listed (`created` on the in-memory backend, say) and
  never reuses an object whose backend exposes no comparable identity (with a warning) — size alone served old
  bytes forever; the payload and its identity sidecar are renamed under one exclusive lock, so two writers of
  different identities cannot interleave their renames and leave a torn pair the staleness check validates
  forever; a missing remote object is a `MissingInputError`; and the cache file name (and its error texts) no
  longer carry a presigned URL's query string.
- **The readers refuse what they would have silently dropped or last-won**: a BEIR row without an id (or with
  an empty one) vanished without a message and is now refused with its file and line; a missing corpus,
  queries or qrels file raises `MissingInputError` instead of a bare `FileNotFoundError`; a `(query, doc)` pair
  labelled twice silently took the last label in the BEIR tsv, the HF reader, the sidecar format (whose image,
  video and frame readers crashed with `AttributeError` where `jsonl` refused — one shared reader now), the
  derived qrels of a ranking-layout file (two rows for one query silently merged) and the hub loader — and is
  refused everywhere; a qrels split whose grade column is none of `score`/`relevance`/
  `label`/`grade` silently labelled every row 1.0 in `HfReader` and is refused with the column list. The hub
  loader refuses duplicate query ids (it silently kept the last row), a non-finite or out-of-range `gain`, a
  non-finite `theta`, and names a missing `top_ranked`/`excluded`/`queries` column like the qrels path does.
- A judgement record's placements carry one shape: a rubric placement with a stray `score` and a tournament
  placement with rubric `criteria` were both recorded as `valid=True` observations (a parser bug emitting both
  shapes went unnoticed); they are refused with the shape named. `cache()` returns a `file://` URI's real path
  instead of the literal `file:/...` string, which names no file.
- **The BEIR writer keeps what it wrote**: qrels labels are written with `repr` (round-trip exact) instead of
  `%g`'s six significant digits (a grade of 0.123456789 came back 0.123457); a query's `instruction` is written
  (the reader restores it) instead of dropped; a media-bearing query is refused like a media-bearing document
  instead of being written as an empty-text query; and a headerless qrels file no longer loses its first row
  when the last cell merely fails the digit check (only a row naming the columns is a header).
- **Estimator inputs are refused, not dropped or misread**: `BradleyTerryEstimator.add_comparison` discarded any
  comparison naming a document outside `doc_ids` and every self-pair with no error, count or record (an
  id-format mismatch silently weakened the fit and its standard errors) — it now refuses them, and refuses a
  non-positive/non-finite weight or a `soft_label` outside `[0, 1]`. `Criteria2PL`'s validator checked only
  `0 <= s_k <= n`, which a NaN count passes on both sides, so a NaN or fractional pass count flowed into
  `eap`/`score_document` as `theta=nan` behind a clean-looking `DocumentEstimate`; counts must be finite whole
  numbers. `Tournament2PLCalibrator.add_observation` likewise refuses a non-finite `theta_bt` (it NaN'd the
  whole fit), `calibrate_2pl_from_results` records its skips in the new `FitDiagnostics` fields, and a rubric
  observation row of length not 2 or 3 is refused instead of having its tail silently ignored.
- **The public records refuse non-finite and naive values**: `QueryParams(tau=inf)` passed `Field(gt=0)` and
  `alpha=nan` calibrated every ability to NaN; a NaN `Placement.score` made a `valid=True` tournament judgement
  whose NaN flowed into the fits; `DocumentEstimate` accepted non-finite scalars; `recorded_at` accepted a naive
  datetime although the store and `supersedes` order windows by it across hosts (mixing naive with aware
  crashed `supersedes` with a bare `TypeError`). One NaN ranking score also silently disabled the descending
  sort (a NaN comparison is always False): `descending_score_order` refuses it.
- **The metric and the gains stop returning wrong numbers at the edges**: inf/NaN gains flowed through
  `ndcg`'s sums and returned NaN with no error, and a negative gain made it return 1.163 against its own
  `[0, 1]` contract — gains and ideal gains are finite-checked like scores. `count_gain` returned 9.0 for
  passes `[9, 9]` at placements 1 and -0.5 for a negative count; pass counts must be whole numbers in
  `[0, placements]`. `qrel_gain("exponential")` escaped with a bare `OverflowError` (an OS errno string) for
  grades >= 1024 and accepted NaN grades. `discount(0)` was a bare `ZeroDivisionError`. `gain` overflowed
  `sum(gammas)` to inf on huge finite gammas and returned 0.0 instead of the weighted mean, and its scalar
  paths computed in the caller's dtype (a float32 item set returned an `np.float32` ~6e-8 off the same input
  as floats); scalars now compute in float64 and the weighted mean normalises before summing.
  `candidate_docs` deduplicates its entering ids (first occurrence) instead of returning a list `ndcg` refuses
  and `score_query` silently dedupes.
- **One concept, one field**: `RankingExample.query` duplicated the aliased `text` and could disagree with it
  in one written line (built with `text=`, `query` stayed empty and both keys were serialised); the field is
  gone, `query` is only the alias of `text`, and the readers take `example.text`. `Content.truncated` broke its
  verbatim-prefix contract when a content held an empty text part (the empty part spent the join newline's
  budget on the next part and was then dropped) and silently cut everything for a negative `max_chars` (a
  caller bug; `0` legitimately cuts to nothing); it now walks the parts against the joined text. The temp-file
  + rename publication the media cache and the PDF page render each grew their own copy of now lives in
  `storage.publish`/`publish_bytes`.
- `jsonl:`'s dataset name is the file's stem (everything before `.jsonl`), not everything before the first dot:
  `nfcorpus.v2.jsonl` loaded as `nfcorpus` and two versioned files collided under one name. Its corpus rows are
  read strictly (an unknown key is refused, not read past — a BEIR-shaped `title` silently vanished into the
  text). A rankings file of one JSON array of otherwise-valid rows raised a raw `AttributeError` past
  `load_rankings`' promised `DataError`. `frame_indices` of a frames-reader clip records the frames' own file
  numbers (which frame of the source it is) instead of their positions in the directory, which cannot say that
  once the numbering has gaps. `configure_logging("SPAM")` raises the `ConfigError` `paths.log_level()` raises,
  not a bare `ValueError` from three frames inside `logging`. `join_title` treats a non-string (e.g. NaN) title
  as no title instead of joining the literal text `nan` in front of the body. The 2PL's `_unique` TypeVar is
  bounded by the row union, not their tuple.
- The `eval_score` MCP tool no longer claims `readOnlyHint: true`: it takes `out` and overwrites that path with
  the full report, so the machine-readable contract now says the call leaves an artifact behind
  (`readOnlyHint: false`). The false claim was pinned by the test suite, the skill text and the release notes;
  all three follow the annotation.
- `rcp-ndcg mcp serve` survives a malformed `tools/call`: arguments that are not a JSON object (a string, a
  list, a number — the falsy ones included, which were silently coerced to `{}`) are answered as JSON-RPC
  invalid params (`-32602`) or a typed `USAGE` tool error instead of killing the stdio loop, a request body
  that is not an object is answered as `-32600`, and a failure raised inside the server is answered as
  `-32603` — the next request is answered either way. `call_tool()` refuses non-object arguments the same way
  for its direct (Python and SDK) callers.
- An unknown `--system` (`eval score`, `eval explain --report`) or `--baseline` (`eval compare`) value is a
  `UsageError` (exit 2), the class of every other unknown command-line value on these commands (`--fields`,
  `--metrics`), not a `ConfigError` (exit 3): there is no config file to fix. The message and the systems
  list are unchanged; the library keeps its own `ConfigError` for `systems=`/`baseline=` Python callers.
- Every refusal the command layer raises carries its `hint` (the machine-readable next step was null at 24
  raise sites of `rcp_ndcg.cli` and the MCP `call_tool`), and so do the evaluation refusals a command can
  reach (unknown `--k`/`--metrics`/data-source combinations in `evaluate`, unknown `--metric`/`--baseline`/
  one-system reports and shared-query checks in `compare`, the `--k` of a multi-cutoff report, an unknown
  `--query-id` in `explain`) and the run manifest's refusal (`run status`/`run show`,
  `eval compare/explain --run` with
  a damaged run directory). The Python wording keeps its `cli_hint` where the two differ.
- `details.errors` has one shape for every validation: the documented one (per problem the `field`, the given
  `input`, the `problem`, the `expected` type when known, a `did_you_mean` for an unknown key, and the
  `source`), built by one helper (`rcp_ndcg.support.config.validation_problems`) for config files and for the
  command layer's argument refusals alike — which used to write `{field, message}`.
- `eval explain --subset` with `--run` is refused as a `UsageError`, like `--system` there: the flag has no
  effect on a run, and it was silently ignored.
- `eval score --per-query` prints the per-query values in the text renderer too (one row per system, query,
  metric and k), not only with `--json`.
- The failure envelope's `command` field is the command path even when a global option's value precedes it
  (`--env-file f.env data inspect` no longer reports `f.env data`): the root group's value-taking options are
  skipped with their values on the paths that have no context (Ctrl-C, an unexpected failure).
- The exit-code tables and `errors.py` say "an insertion whose anchor check failed" where they said "a scale
  check" (the artifact is `data.extension.anchor_report`; no artifact named "scale check" exists).
- The output contract's wording declares its one exception (`--help`/`--version` print plain text, no
  envelope), and `CliEnvelope.data`'s description says which commands tag their data with a `schema` id.

- **A corpus is named in every judgement record id and store identity (sweep-llm B1):** `judgement_record_id`
  carries the dataset's identity key (the digest of the store identity's dataset entry: its name, URI and
  resolved revision; a row-sequence input's rows by their SHA-256), and a row-sequence input's store identity
  names its rows by that digest instead of the constant `{"name": "dataset"}`. Two corpora that share query and
  document ids no longer fuse — a second pass into the same store is refused by the gate, and across stores
  their record ids differ, so a merge keeps both corpora's windows (it used to reuse the first corpus's answers
  for the second and drop one corpus's window in a merge). Record ids deliberately moved; stores written before
  the change refuse a resumed pass until forced.
- **The family key carries the judge's declared settings (sweep-llm B2):** `Family` gains `temperature`,
  `max_output_tokens`, `context_tokens`, `extra_body` and `api`, and its digest carries each only when it
  differs from the default (the tokenizer pattern), so every family judged under the defaults keeps its key
  (the pinned `Family.key` digests are unchanged) and one judged under a declared value never pools with it,
  cross-store included. The store identity carries the fields with the family, so an old store refuses a
  resumed pass until forced.
- **A judge step's identity is keyed by what its judgements answer for (sweep-x-arch N1, sweep-runs 1):**
  `prompt_sha256` — the named prompt's content hash (the schedule's prompt name or path is runtime; when the
  schedule leaves the prompt unset, the stage's shipped prompt set by content, so an edited shipped prompt
  re-keys the step without a resume check ever reading a corpus) — and the judge's `identity_extra()` (its
  tokenizer's SHA-256, as the encoder and reranker steps splice) enter the judge-step identity; a resume after
  a prompt edit or a tokenizer swap re-judges instead of skipping with stale judgements.
- **An orphaned think-end never erases a complete answer (sweep-llm M5):** the orphaned-think-end strip applies
  only to the text before the object, so an answer followed by a stray think-end tag parses instead of being
  recorded `no_json` after every retry and dropped from every fit.
- **One torn last line of the shared `preprocessing.jsonl` no longer poisons a store (sweep-llm M6):** the
  census file's rows are read through one helper (`rcp_ndcg.data.preprocess.read_census_rows`) that skips a
  torn last row with a warning — the tolerance the judgement records have — and raises the typed `DataError`
  with a hint for a complete line that is not a census row; a killed pass's half-written census row no longer
  crashes every resumed pass, and reparse no longer copies the poison unexamined.
- **The judgement store's identity read-modify-write is serialized between processes (sweep-llm M1):** two
  passes claiming the two stages of one fresh store at the same time used to lose one stage's entry (the last
  full-file write clobbered the other, and the losing pass crashed on `read()`); `claim` and `note_engines`
  hold an advisory `flock` on the store directory around their read and their write. The multi-process stress
  test (`tests/judging/test_store_multiprocess.py`) reproduces the loss without the lock; the append side holds the
  same lock (below).
- **The store's record append holds the store's advisory lock** (sweep-llm M2): the first append's torn-tail
  repair truncates to the last complete line, and a peer's in-flight record is exactly what that truncation
  would cut — the isolated first-append window measured 4/250 lost records. The stress test now pre-claims the
  store, seeds a torn tail and aligns the two workers' first appends; a lock-scope pin catches the unlocked
  append deterministically.
- **The census writers cut a torn last row before their first append** (`rcp_ndcg.data.preprocess
  .drop_torn_last_line`, the one repair the records' append already used): a killed writer's torn row used to
  merge with the next appended row, and the merged line was refused by every later read.
- **One atomic-write helper** (`rcp_ndcg.storage.atomic_write`): the store's identity file and prompts, the run
  manifest (whose temp name was pid-only — two writers in one process shared it and lost saves) and the remote
  cache publish through it; the store's prompt texts are verified against their hash and rewritten when a torn
  write left a file whose content contradicted its filename (sweep-llm m14).
- `run status` (and `run cancel`, `run resume`) no longer fail for a job that has left the queue: a non-zero
  `squeue` — what standard Slurm answers for a finished job (`Invalid job id specified`) — means "not in queue"
  and the `sacct` fallback runs; it used to raise (sweep-runs 2).
- The uv bootstrap installs uv from the wheelhouse when one is given (`--no-index --find-links`), so an
  air-gapped node — the wheelhouse option's whole point — can start a job whose engine image has no uv
  (sweep-runs 3).
- A resume whose identity check raises before a step starts (a judge config file gone) fails with the typed
  `MissingInputError` instead of the failure handler's `AttributeError` on the not-yet-set usage, which masked
  it (sweep-runs 4).
- **`from: rankings` + a rerank step, with no retrieve step, works**: the rankings file IS the run's first
  stage — the rerank step reads its supplied pools directly, and the judging steps read the reranker's
  candidates — where the combination used to validate and fail mid-run on an internal scratch path
  (sweep-runs 5). The refusal stays where it is true: `from: retrieval` really has no first stage until the
  retrieve step runs it, and that combination is refused in the config.
- `JudgeConfig`'s copied base-URL validator and `urls` are gone: the copy had drifted to accept
  `base_url: ""` — a config that validated and could never be sent to. The `Endpoint` rule and property are the
  only ones (sweep-x-arch F1).
- An empty planned-window list (`{"q": []}`) is refused with a `ConfigError` and a hint before anything is
  asked, where it used to crash the pass mid-flight with a bare `max()` `ValueError` after other queries had
  stored answers (sweep-llm M4).
- A judging step re-run that fails keeps no outputs, inputs, usage or engines of the attempt it did not run,
  and a failed re-run of `evaluate` keeps no metrics of it (sweep-runs 7).
- `run status`, `run list` and `run show` no longer fail when they read a running job's judgement store while
  the job claims or reports a stage: the store's `identity.json` is written through a temp file and renamed (as
  the run manifest's save is) instead of rewritten in place, so a concurrent reader sees the old or the new
  file, never a truncated one. An MCP e2e poll lost this race (a `JSONDecodeError` on `identity.json`, whose
  `run_status` error result has no `jobs` key) and failed the test intermittently under `-n 4`.
- `_hub_absent` no longer crashes when huggingface_hub drops its private `_CACHED_NO_EXIST` sentinel: without
  the sentinel the cache cannot tell "absent upstream" from "not cached", so the file is treated as not cached
  (the load refuses with the offline hint and one debug line records it; an optional table is never silently
  `None`, a required one never reported as upstream-404).

- The MCP server logs the typed warnings a tool call collects (its results have no `warnings` field, so the
  server's log is where e.g. `UNPINNED_REVISION` surfaces there).
- A recorded config (run.yaml, the manifest) re-validates without refusing its own defaults: a hosted or served
  encoder's or reranker's `concurrency` equal to its default no longer fails `run start`, a resume or `run status`
  with `drop concurrency`. The one-at-a-time check compares the value against the field's default (a full dump
  cannot preserve which fields the user set); an explicitly non-default `concurrency` on a provider that sends one
  request at a time is still refused.
- **Video containers are counted as the engine's video accounting counts them**: `_container_tokens` counted
  `num_frames x per-frame` tokens at the client's declared image budget, but the container is sent unchanged,
  so the client's budget never reaches the engine: the engines patchify in time (a stock engine samples 32
  frames -- 2x under -- and a pinned one shows `ceil(num_frames / 2)` merged steps -- 2x over), and they size
  the frames by the checkpoint's own video budget, not the client's (up to 4.7x more tokens than a tight
  declared budget implies). A container now counts `ceil(num_frames / temporal_patch) x per-frame tokens`
  under the family's own video budget (`PROCESSORS`): each frame sized independently for the Qwen2-VL
  families -- stock vLLM's accounting, which passes the checkpoint's image-processor size for videos
  (8 frames of 720x1280 = 4,786 tokens) -- under one vision block; for `qwen3_vl` the whole clip is budgeted
  together (4,096..25,165,824 px, which shrinks per-frame resolution as the frame count grows: 8 frames of
  720x1280 = 3,520 patch tokens, 128 = 11,520, measured against the real `Qwen3VLVideoProcessor` with 0
  mismatches over 65 combinations) and the prompt renders one timestamp line and one vision block per
  temporal group (a declared bound of 10 tokens per group for the timestamp, measured 6 at `<0.0 seconds>`
  with the family tokenizer). Correspondingly, `wire: video_url` is refused (pydantic, at config load)
  unless the new `VideoPolicy.engine_video_pinning` declares the engine pinned to the same frame count
  (vLLM `--media-io-kwargs`), a single-frame container is refused (the declared
  instrument merges frames in time, which needs at least a temporal pair; a single frame is an image), and
  the declaration is refused under `wire: frames`, which samples on the
  client. `wire: frames` stays the default and exact. The declared count is stock vLLM's; the pinning ties
  the frame count and `engine_media_check` (above) compares the engine's actual count at run
  time. Stored judgements and the paper's tables do not move: only the window budgets and estimates of new
  judge passes over video containers change.
- **The window budget charges each media item's vision block and a declared marker reserve**: the old
  accounting charged merged patch tokens only, so a wide multi-image window could exceed the judge's context
  and fail mid-pass at the engine. The charge per document is now `content_media_tokens(...)` (every image
  and sampled frame its vision start/end + patch tokens, a container its temporal grid) plus the template's
  media marker per media part, measured with the judge's tokenizer (`media_marker_tokens`) -- a declared
  reserve, not an engine count: the payload builder replaces each marker with the media part, so the charge
  errs a few tokens high per media part, never low, and the 256-token chat-scaffold reserve covers only what
  is not media. A window whose media alone do not fit is refused (`CapabilityError`) before anything is
  spent.
- `ImagePolicy.target_size` raises the documented `DataError` (exit 12, with a hint) for an input whose aspect
  ratio exceeds 200, instead of a bare `ValueError`.
- Changing a served encoder's or reranker's URL no longer re-runs retrieval or reranking: the `retrieve` and
  `rerank` step identities hold the candidates config's content payload (`identity_payload`, as the judge steps
  already do), so its runtime fields (`base_url`, `api_key_env`, `concurrency`, the timeouts and retries,
  `batch_size`) never reach a key. The payload keys candidates by field name (`source`), not by its YAML alias
  (`from:`), and unset optional fields are omitted; step identities change once accordingly.
- `eval score`, `eval explain --report` and `evaluate()` refuse rankings that match nothing of the scored dataset
  instead of scoring every query 0 with `ok` (issue #5): a `DataError` (exit 12) when no row of a system names
  any subset of the scored dataset (the `dataset` column must hold the exact subset name), and when not one of
  its ranked document ids is in the dataset's pools or labels (the message shows one ranked id next to one
  dataset id). Partial overlap keeps scoring as before, and the `UNRANKED_QUERIES` warning names the subsets it
  counts when the scored dataset has more than one.
- An offline run over a hub cache an online run filled now works without `--revision`: the online resolution
  records the commit behind the branch (`refs/<ref>` in the cache), which a download pinned to a commit never
  wrote, and offline the repository's file listing falls back to the local snapshot (with one warning that it
  holds only the files a download left), so a run materializes its corpus from the cache. Offline with nothing
  to resolve, the failure is a non-retryable `MISSING_INPUT` whose hint says to pass `--revision <full sha>`
  and whose details name the table looked for; a Hub that cannot be reached — connection failure, timeout, or
  answering 5xx or 429 — is a retryable `PROVIDER` naming `HF_ENDPOINT`, also when the listing is what failed.
  Offline, an optional table (`excluded.parquet`, `top_ranked.parquet`) is treated as absent only when the cache
  records it so (`.no_exist`); an uncached optional table is an error with the offline hint, never a silently
  empty pool, and the pinned offline run keeps working. A corrupt cache ref is removed before resolution and
  rewritten by the next online one instead of failing it.
- **The calibration's refit is order-canonical** (scoring-chain review F1): a planned window
  (`window_seq=None`) has no schedule position, so the projections now order those by `record_id`; the same
  windows read in any store order give bit-identical Bradley-Terry abilities, standard errors, item parameters
  and fingerprint, as `docs/concepts/calibration.md` promises. Scheduled windows keep their positions, so no
  fitted number moved.
- **A document the tournament showed without a valid window is visible** (review F2): its ability stays the
  paper's (the query's mean, the ridge's standard error only when the query has other comparisons), and the fit
  lists it under
  `coverage.no_tournament_evidence_documents` and warns with `NO_VALID_TOURNAMENT_EVIDENCE`;
  `calibrate(..., strict=True)` (`calibration fit --strict`) refuses it. A missing Bradley-Terry standard error
  is written as `None` (review F7), not as 0.0 ("certain"), and `Calibration.load` validates the
  `thetas.parquet` rows (review F6): an unknown `source`, a non-finite theta or an infinite SE is a
  `DataError`; a JSON artifact that would hold a NaN names the file instead of raising a bare `ValueError`.
- **`explain` computes its gaps under the report's tie rule** (review F4): for a `group_mean` protocol the
  displayed order is document id descending, but the selection/ordering deltas now credit an equal-score class
  its mean gain, so they equal the report's per-query values; the display order is documented.
- **`RaschEstimator.add_criteria` refuses an unknown document** (review F11) as
  `BradleyTerryEstimator.add_comparison` does, instead of dropping the observation silently (the rubric
  schedule never relied on the drop).

### Changed

- **The T3 task matrix gains the pplx sizes**: `pplx-embed-v1-0.6b`/`-4b` under text embedders (nanobeir,
  bright, trecdl) and `pplx-embed-v2-late-9b` under visual documents (vidore) and late interaction, text
  (nanobeir, bright); `tests/test_quality.py`'s coverage pin moves with it.
- **A model's text is formatted where the model's text is formatted** (workstream 10 C2/C3, decisions 27,
  33): the corpus materialisation of `retrieval.index`/`search`/`retrieve`/`rerank` and of the judge reads
  each document as MTEB's dataloader does -- `(title + " " + body).strip()`, the body alone without a
  title -- instead of the body alone; the derived ranking shape (`SourceReader.examples`) carries the same
  text; and the query text of the dense, pooling, BM25, rerank and judging paths applies the two generic
  instruction defaults (the task prefix, the per-query append) once each. The paper's published runs read
  the blank line between title and body: `REPRODUCIBILITY.md` says so. The behaviour fingerprint is
  unchanged (`rcp-fp/3`): the formatting is upstream of the wire, the recorded exchanges are unchanged, and
  the run identities carry the new `title`/`instruction` fields.
- **The 18 standalone recipe directories become 13 families / 19 variants, and the new `embeddinggemma-2`
  family brings the release to 14 families / 20 variants** (decision 34): the resolved
  contracts are byte-identical to the pre-family tree except where a variant's standalone recipe declared a
  product default the family now omits (`request_shape: text`, `listwise: false`,
  `add_special_tokens: {pair: true}` -- the product's endpoint model resolves each to the same value), and
  where the family shares ONE template file whose pre-family per-size copies differed only in their jinja
  comment headers (the renders are byte-identical; the fingerprint's `template_file` input moves, declared in
  the conformance waivers). Every variant keeps its own contract test (one module per family, parametrized
  over its variants, two mutants red per family), its stage-1 network tests and its pairs file, and the
  per-variant goldens (`rcp-ndcg-test/tests/recipes/golden/`) pin the resolved contract and fingerprint in
  every CI job (offline; `--update-goldens` regenerates on purpose).
- **The float-gain metric matches mteb PR #5516 bit-for-bit, nAUC keys included**: `rcp_ndcg_core.metric`'s
  `dcg` now divides by `log2(rank + 1)` instead of multiplying by the reciprocal (`discount`), the PR's own
  operation order, so the per-query values (and through them the abstention nAUCs of
  `rcp_ndcg.eval.mteb.ndcg_float_scores`) are identical; the metric means are unchanged, and the paper's anchor
  numbers do not move (the tests and `run_all` pin them).
- **`rcp_ndcg.eval.mteb.ndcg_float_scores` follows mteb PR #5516 on a query whose gains are all null**: it
  scores 0 and stays in the mean instead of refusing the query, and it validates every gain in the table, not
  only the scored queries'. A non-finite model score is still refused (the PR ranks an infinity as usual; a
  model that emits one has a bug). The metric is pinned against the PR's own `ndcg_float_scores` in the tests,
  vendored at the PR's commit.
- **The mirror page states the sync guarantee** (review S3/S4/S6): durable is the last uploaded part; a hard
  kill loses at most one interval, re-asked on resume and never duplicated (`record_id`); one live writer per
  store, a diverged writer's flush refuses with `DataError` and the run continues unmirrored (`run status`
  shows it); parts and superseded files are never garbage-collected.
- **The Matryoshka selection is declared before it is selected**: a pooling `mrl_dim` now needs its
  `mrl_kind` and `mrl_dims`/`mrl_range` (the card's set) and a dense `mrl_dim` is new; a `k` outside the
  declaration
  is refused at load. When `mrl_dim` is set, the client normalises the full-width reply first (when
  `normalize`) and then applies the head, so a direct `k` run and the ex-post sweep over a full-width store
  compute bit-identical vectors (the head renormalises the cut, and the learned projection is linear).
- **The Hub reader reads mteb's card-driven layout** (owner decisions 28, 31, 32): the released rcp-ndcg
  repositories' tables are resolved through their cards' configs (falling back to the plain `{subset}/` path
  layout for a card that does not declare them), so `hf://mteb/nfcorpus` and the other MTEB mirrors load;
  queries are cut to those with qrels (mteb's rule); a document's `title` stays a field (nothing joins it
  with the body at read time) and its media columns become content parts, with `document_parts` and
  content-addressed persistence carried over from the retired reader. The reader needs no `datasets` and no
  `mteb`; `EXTRA_FOR_MODULE` maps `datasets` to `[mteb]`, and `[data]` drops the `datasets` dependency.
- **Duplicates fold, conflicts refuse, `duplicates="last"` takes the last row** (owner decision 30): the
  same id with the same content and the same `(query, document)` pair with the same grade fold (the one
  policy, the jsonl sidecar and the derived ranking shape included) and the counts are recorded in the
  dataset's provenance (`DuplicateCounts`): every load reads the labels, the pools and the exclusions, so
  those are counted there, and the corpus and query folds are logged as those tables are read. A conflicting
  duplicate refuses, naming the rows, unless the load passes `duplicates="last"` (mteb's own behaviour),
  which resolves it for the tables that can replace a row (the labels, the pools and the exclusions, all
  materialised) and records the count; a streamed corpus or query table refuses a conflicting row even under
  `last`, with the option's scope named -- it cannot replace a row it has already yielded. The measurement
  over the canonical repositories' labels, queries and corpora found no duplicates.
- **BEIR reads gzip-compressed files**: `corpus.jsonl.gz`, `queries.jsonl.gz` and `qrels/<split>.tsv.gz` read
  like their plain siblings (through `storage`, so remote URIs keep working), the BEIR writer writes a
  document's `title` into the title column so it round-trips, and the BEIR reader's provenance records the
  qrels split read.

- **One error shape for the role-config family**: every policy refusal raises `ConfigError` with a hint
  naming the field to change -- never a bare `ValueError` that pydantic wraps into a hintless
  `ValidationError` (the chunk geometry, `empty_doc_text`, `query_max_tokens`, `media_sides`, `mrl_dim`, a
  rerank `request_shape`, `listwise` with `batch_size`). Sibling validators of one family used to raise two
  error families.
- **Declared modes the wire cannot carry are refused at the config, never silently ignored**:
  `instruction: "system"` on a rerank config (no shipped rerank wire has a system-message slot -- the
  instruction would never reach the model; use `fold`, `field` or `none`), and `PoolingEndpoint.dimensions`
  (inherited, never sent by `/pooling`, yet re-keying every identity over full-width vectors).
- **Bare `ValueError`s across the budget and wire types raise typed errors** (AGENTS: typed errors from
  `rcp_ndcg.errors`): `fit`'s argument checks, the truncation census' mechanism check,
  `uniform_frame_indices`, `smart_resize`, `Call`, `Embeddings`' layout invariants and `concat`,
  `RerankResult.aligned` -- now `DataError` with hints. Code that caught `ValueError` for these must catch
  `rcp_ndcg.errors.DataError`.
- The role clients instantiate their adapters uniformly with the role config (`adapter_cls(config)`, the
  documented convention; an adapter that needs no config accepts and ignores it).
- The `PoolingEndpoint` docstrings state what the `/pooling` wire takes; the vector clients' `on_overflow:
  chunk` refusal is now described where the fields are declared (chunking is a rerank-only mode: an
  embedding has no score to pool, token vectors are not scores).
- **The documentation is reorganised into Concepts, How-to and Reference tiers**: `docs/tutorials/` is now
  `docs/how-to/`; the served-role budgets move from `docs/concepts/preprocessing.md` to
  `docs/concepts/text-budgets.md`; judges and runners split into `docs/concepts/judges.md` and
  `docs/concepts/runs.md`; `docs/concepts/retrieval.md`, `docs/how-to/serve-a-model.md`,
  `docs/how-to/validate-a-recipe.md`, `docs/reference/recipes.md`, `docs/reference/rcp-ndcg-test.md` and
  `SECURITY.md` (the private-reporting policy) are new. The README, the quickstart and the agent skill now
  present four paths (score, serve and score, re-judge, reproduce), state the rankings-file column contract
  with its accepted aliases, and describe `recipe: <id>`, `rcp-ndcg-vllm serve` and the judge text policy. The
  exit-code table's one home is `docs/reference/cli.md`; the skill links it.
- **The qwen3-vl recipes are one declared family** (`rcp-ndcg-vllm/recipes/qwen3-vl-embedding-2b/`,
  `qwen3-vl-reranker-2b/`): one R20 pixel-pin shape for every media recipe, `serve.mm_processor_kwargs:
  {images_kwargs: {...}}` -- at the vLLM v0.31.0 tag both shapes reach the HF image resize (measured: 1776
  vs 1240 image tokens on one page), but flat keys also re-size every video clip on vLLM's video path (a
  64-frame clip measured 672 instead of 7040 tokens); `client.image_policy` declares the same pixel numbers
  (the client cannot apply them yet: the product's `qwen3_vl` geometry refuses a budget below its 65536 px
  floor, so media stay a typed refusal); one video sampling policy on every side (64 uniformly spaced
  frames per clip: `client.video_policy` with `--media-io-kwargs` engine pinning). qwen3-vl-embedding-2b
  declares a `query` shape after the model card (the card frames queries exactly like documents, with its
  default instruction) and serves no template file (the completion route applies none; its test renders the
  checkpoint's own `chat_template.jinja` instead). Both references render the card's own over-cap cut, never
  the client's: the embedding's `truncation=True` right cut (`anchor_drop_over_cap`), the reranker's
  `truncate_tokens_optimized` cut, which keeps the anchor (`over_cap_cut_differs`, no longer the mislabelled
  `anchor_drop_over_cap`). Each contract test pins every resolved field (two mutants per recipe red).
- **The late-interaction recipes and their plugins** (`rcp-ndcg-vllm/recipes/topk-embed-v1-small/`,
  `pplx-embed-v2-context-9b-preview/`, `rcp-ndcg-vllm/plugins/{topk,pplx}`): topk serves through
  `serve.plugin: rcp-ndcg-vllm-topk` with the product's fields (`query_max_tokens: 1024`, the checkpoint's 41
  `scoring_skip_ids` as `document_skip_token_ids`, `media_sides: [document]`, `normalize: [strip]`) and the
  nested R20 shape; its reference renders the card's own 1024/8192 cut (by character offsets), declared
  `over_cap_cut_differs` (an id cut inside a multi-token character reads one id no text carries); its plugin marks the bias-less checkpoint's zero projection bias as loaded (vLLM
  v0.31.0's weight tracker refused `custom_text_proj.bias`). Neither recipe passes `--trust-remote-code`:
  each plugin registers the checkpoint's configuration class with transformers, so a `config.json` that
  names remote code resolves locally with the flag off. pplx serves `dtype: bfloat16` (vLLM v0.31.0's GDN
  kernels refuse float32; the reference keeps the card's float32, a declared deviation). The pplx NOTICE
  entry for the restated config class is added; the plugin suites run together in one process.
- **The qwen3-reranker recipes are one declared family** (`rcp-ndcg-vllm/recipes/qwen3-reranker-0.6b/`,
  `qwen3-reranker-4b/`, `qwen3-reranker-8b/`): one over-cap policy -- every reference is the paper's
  `QwenOGRerank` cut (the pair string right-cut at 8144 tokens, both anchors re-attached) and never the
  client's, so all three declare `reference.known_deviations: [over_cap_cut_differs]` (4b's
  `anchor_drop_over_cap` was mislabelled; 8b's reference stops porting the client's settle rule) and
  under-cap rows gate exactly; the references write their render as the harness's content spans, located at
  raw character offsets (never a `decode(encode())` round trip, which NFC-maps non-NFC input); one template
  file, `template.jinja`, byte-identical across the three (0.6b's `qwen3_reranker.jinja` is renamed and no
  longer reads a request instruction); `serve.pooler_config.use_activation: true` beside
  `client.use_activation`; `min_version` 0.31.0 (the verified image) with the 0.30.1 feature floor in the
  notes; every recipe ships `requirements-reference.txt`; the notes state the merged rerank client's settle
  rule (the pair fit binds on overflow; the shared query settles at its 4096-token share whenever it exceeds
  it, so an over-share query is a declared divergence row). Each contract test pins every resolved
  `serve`/`client`/`reference` field through `tests/recipes/_contract.assert_recipe_contract` (two mutants
  per recipe red). The served prompts are unchanged.
- **The zerank recipes are one declared family** (`rcp-ndcg-vllm/recipes/zerank-1-reranker/`,
  `zerank-1-small-reranker/`, `zerank-2-reranker/`): the paper's `query.strip()`/`doc.strip()` is the
  declared content normalisation `client.template normalize: [strip]` (zerank-1's template file loses its
  jinja `| trim`; the declared whitespace divergence of zerank-1-small/zerank-2 closes as declared policy),
  the recipes state the merged rerank client's settle rule (the pair fit binds on overflow; the shared query
  settles at its `query_max_tokens` share whenever it exceeds it -- over-share queries are declared
  divergence rows), all three pin `pooler_config.use_activation` server-side beside `client.use_activation`
  and declare `client.recipe`, and zerank-1 stops passing `serve.convert` (the shakedown row: a rerank
  recipe declares the checkpoint's scorer through `serve.hf_overrides`). The three references' `--mode
  render` write the paper's own cut (the whole rendered prompt right-cut at 8192 tokens, located at raw
  character offsets) as the harness's content spans -- never the client's cut; over-cap rows are the
  declared `anchor_drop_over_cap` and under-cap rows gate exactly. The template files are named
  `template.jinja` in all three (zerank-1-small's `zerank_score_template.jinja` and zerank-2's
  `zerank2_score_template.jinja` are renamed). Each recipe's contract test pins every resolved
  `serve`/`client`/`reference` field through `tests/recipes/_contract.assert_recipe_contract` (two mutants
  per recipe red). The served prompts are unchanged on whitespace-clean inputs.
- **`rcp-ndcg-vllm/jobs/rc_build.sh` stages the pairs files from `rcp-ndcg-vllm/pairs/`**
  (their one home, where the request generator writes them): a stray `<checkout-root>/pairs/` is refused with
  the home named instead of being silently staged, and a checkout without pairs stages none.
- **The BM25 index is persisted in bm25s' own format, never a pickle** (`rcp_ndcg.retrieval.sparse`): the
  index directory's model is stored with `BM25.save(..., allow_pickle=False)` (npz arrays + JSON parameters)
  and loaded with `allow_pickle=False` -- the index directory comes from ordinary user paths (`retrieval index
  --out`, `retrieval search --index`), and unpickling one somebody else wrote would run their code. A directory
  holding only the earlier build's pickle file is refused with a rebuild hint; indexes built by this build
  must be rebuilt.
- **The public retrieval and eval functions refuse their bad inputs with typed errors**: `numpy_topk` and
  `maxsim_topk` raise `ConfigError` (a non-positive `k`) and `DataError` (a non-multi-vector or mismatched
  input, a 3-D array into `numpy_topk`) instead of bare `ValueError`; `mteb._hub_text` raises
  `MissingInputError` instead of `FileNotFoundError`; the mteb task's cross-encoder and `skip_first_result`
  refusals are `CapabilityError`.
- **The run/step/job status vocabularies are typed, one enum each, and the exported schema pins them**:
  `RunState.status` is `RunStatus`, `StepState.status` is `StepStatus` and `JobState.status` is `JobStatus`
  (they were bare `str` with the closed sets only in prose). `StepStatus` gains `PENDING` — `run status` lists
  a step the run has not started as `pending`, which was a bare literal outside the enum. A finished-OK job now
  reads `completed` like a finished-OK step and run (`JobStatus.SUCCEEDED` is renamed: `run status` used to say
  `succeeded` for a job and `completed` for its steps in adjacent fields). `run list`'s rows are typed
  (`RunListRow`), so `run-list.v1.json` pins the row shape, and a manifest that does not parse is listed with
  `status: "unreadable"` — the one value outside `RunStatus`, validated and documented instead of invented per
  call. `schemas/run-status.v1.json`, `run-list.v1.json`, `run-start.v1.json` and `run-manifest.v1.json`
  (whose `StepStatus` enum gains `pending`) regenerated; nothing that reads a status by name changes value
  except a finished-OK job: `succeeded` → `completed`.

- **`--plan` means one thing in the CLI**: the plan file `judge tournament` asks exactly the windows of
  (`judge tournament --plan PLAN.json`). The boolean on `calibration insert` — plan the opponent windows,
  insert nothing — is now `--dry-run`, the no-side-effects switch every other command uses, so a script can
  chain `calibration insert --dry-run --out PLAN.json` into `judge tournament --plan PLAN.json` without the
  first `--plan` parsing as a flag. Everything that read the boolean follows: the request field, the flag
  help, `InsertResult`'s schema description, the skill's insertion recipe and the primitives page.

- **The judge's document text policy defaults to 32,768 tokens (2^15) for `truncate` and `fail` without a
  declared cap** (owner decision; was 20,000). A store judged under the earlier default keeps its recorded
  policy: an unset cap that re-judges after this change resolves differently and the store refuses the mixed
  instrument, so pin `max_tokens: 20000` explicitly when a config must keep the old cap's identity. No shipped
  preset or paper config relies on the old default.
- **`JudgeClient` derives from the shared `RoleClient`** (`rcp_ndcg.inference.clients`): the role-scoped
  adapter lookup (refused at construction like every other role's; an unset `api` still means the
  `openai_chat` wire and stays out of the identity), the resolved base URL, the transport/Sender bridge, the
  auth profile (R6) and the close/`aclose`/`gather` lifecycle are the shared base's. `JudgeConfig.urls` is the
  base's property; `probe()` runs the shared engine media check when the pass's effective preprocessing
  declares an image policy. Judgement identities are unchanged.
- One criterion-label derivation (`rcp_ndcg.judging.prompts.criterion_labels_in`), read by `Prompt.criteria` and
  the fake judge; a step re-run clears its record's previous attempt (inputs, outputs, usage, engines, and a
  failed `evaluate`'s metrics); `records_stored` (`rcp_ndcg.judging.store`) is the one count of a stage file's
  lines (the estimate's note and `run status`'s progress both read it); a plugin whose constructor rejects the
  options raises the typed `ConfigError` the built-ins raise; naming the judge's default wire (`api:
  openai_chat`) keys like the unset default (one instrument); reparse re-serializes the census rows it copies.
- `tests/contract` snapshots and the exported schemas regenerated: the `Family` fields (in
  `calibration.v1.json`, `judgement-store.v1.json`, `run-manifest.v1.json`), `JudgeConfig.urls` gone from the
  collected surface (`JudgeClient` carries its `RoleClient` base), and the text-policy default's literal.
- **One lock for a served-only package**: with the `[local]` and `[vllm]` extras gone, `uv.lock` holds one torch
  (2.14.0, the version the coordinator's extras already resolved, CPU-index compatible) instead of the
  conflict-fork pair 2.9.1/2.14.0, and drops 114 packages only the in-process stack needed (`vllm` and its engine
  dependencies `openai`, `httpx2`, `anthropic`, `mcp`, `xgrammar`, `jiter`, `flash-attn`, `accelerate`, and the
  4.57.6 transformers fork among them; transformers stays only through the `mteb` extra, at 5.17.0).
  `requirements-constraints.txt` is regenerated with the command in its header; `oauthlib` moves 3.3.1 → 4.0.0 (the
  first patched version of its open advisory). No other version the coordinator's extras install moved, and the
  paper reproduction (`experiments/run_all.py`) is unchanged: 0 failed, 35 known deviations, same summary.
- The engines overlay (`RCP_NDCG_ENGINES`, `run resume --engine`) rebuilds the role config through its model, so
  an overlaid config passes every validator a configured one does (a URL is normalised as a configured one is);
  an invalid overlay (e.g. a `fake://` replica list) is refused with the typed `ConfigError` where it is applied,
  never half-applied.
- `tests/contract/snapshots/python_api.json` regenerated for `__version__`: the committed version is 0.0.1
  (the root `pyproject.toml`), but the snapshot still pinned the pre-bump install (0.1.0), so every
  environment whose venv postdates the bump failed `test_surface_matches_snapshot[python_api]`. No product
  change: the value now records the version that ships.

- `tests/contract` snapshots and the exported schemas (`schemas/index.v1.json`, `schemas/judge-config.v1.json`,
  `schemas/run-config.v1.json`) regenerated for the moved and new fields; `tests/test_errors.py` now requires
  one *root* class per exit code, since the moved outage and refusal types are `ProviderError` subclasses and
  exit codes do not change.
- `tests/contract` snapshots and the exported schemas regenerated for the embedding adapters, the
  embedding role client and the `api_key_env` refusal (the
  `EmbeddingClient` export, its constructor and `EmbeddingEndpoint.identity_extra()`).
  exit codes do not change. The `python_api` snapshot regeneration also records `JobSpec.phases`, which its
  commit (the job-phase shape) had left out.
- The fake judge's deterministic draws (`_uniform`, `_hidden_ability` in `rcp_ndcg.judging._fake`) now come from
  `rcp_ndcg.inference.fake` (`fake_uniform`, `hidden_ability`): one home for the mechanism the fakes share;
  identical values, and both names stay importable from `rcp_ndcg.judging._fake`.
- The offline fakes' `POST /pooling` payload key is `data`, the wire's real shape (vLLM's
  `PoolingResponseData`): the fake answered `embedding`, which no client reads. Its `/embeddings` route keeps
  OpenAI's `embedding` key.
- `tests/contract` snapshots and the exported schemas (`schemas/index.v1.json`, `schemas/run-config.v1.json`)
  regenerated for the transport, the fakes and the status map (new names and members, `Endpoint.base_url`
  widened, and the retrieval configs' `base_url` described per its type: one URL, required for the served
  ones, optional for the hosted ones); `tests/test_errors.py` now requires one *root* class per exit code,
  since the moved outage and refusal types are `ProviderError` subclasses and exit codes do not change.
- `tests/contract` snapshots and the exported schemas (`schemas/index.v1.json`, `schemas/run-config.v1.json`)
  regenerated for the retrieval rewiring: the configs by `api`, the deleted `provider:` variants gone, and the
  `retrieve`/`rerank` step identities spliced with the endpoint's `identity_extra()` (the tokenizer's SHA-256).
  The paper's reranker configs are served (`recipe:`, `tokenizer:`, the paper's budgets, `instruction: none`,
  `listwise` for Jina v3) and their hosted siblings omit `base_url`.
- **`select_opponents` refuses a query with no opponents** (review F5) with a typed `DataError` naming the
  query and the documents the calibration holds, instead of returning `[[doc_id]]` (not a window: `judge`
  refused it later).
- **The Bradley-Terry refit is documented as a cold refit** (review F10): it fits the live tournament's own
  observations from zero, so it agrees with the live fit to convergence tolerance, not bit for bit. The
  diagonal standard-error approximation is stated where `theta_se` is documented (review F8).

### Removed

- **Every explicit SGLang path** (workstream 08 A, owner decision 14): the release serves every role on vLLM
  v0.31.0. The rerank adapter no longer reads SGLang's (and TEI's) bare list of `{"index", "score"}` rows: that
  shape is refused by name with a hint to serve the model on vLLM, and a row must carry `relevance_score` (the
  `score` key TEI names the relevance by went with it; TEI's rerank shape was never documented for this role,
  whose served wires are vLLM, Infinity, Cohere and Voyage). The chat adapter no longer maps SGLang's
  media-limit wording ("Image count 12 exceeds limit 10 per request.") onto a `CapabilityError`; only vLLM's
  wording is a per-request media limit, and any other refusal is that request's. The Qwen2-VL image budget is
  the checkpoint's own 3,136-12,845,056 px, which vLLM applies, so a policy in the range SGLang's 1,003,520 px
  override used to refuse is accepted. Removed with the paths: the SGLang oracle in
  `tests/data/_media_reference.py`, its NOTICE rows, `experiments/paper/serve/*.sglang.sh`, the engine-script
  test, and every SGLang passage describing a live path; the exported schemas are regenerated. `REPRODUCIBILITY.md`
  records the paper's judges as SGLang history (the paper's submission code is the record; this release serves
  them on vLLM v0.31.0).
- **The column-heuristics `hf` reader** (`rcp_ndcg.data.io.hf.HfReader`, owner decision 32): the Hub contract
  is mteb's layout, and other data is converted once. Its `document_parts` option and content-addressed
  image persistence move into the Hub reader; the `hf` extra still provides `huggingface_hub`, and
  `datasets` is no longer needed by any product path that the `[data]` extra names.

- **Every in-process model path** (the unified-inference design's paths 3–9; the owner's option 1): the package
  carries no model that loads weights. Deleted from `rcp_ndcg.retrieval`: the `local` provider and its variants
  (`Local`, `LocalEncoder`, the `engine` and `pooling` fields, `TorchDenseEncoder`, `VllmEncoder`, the
  `hf` and in-process `vllm` engines), the HTTP path (`_http.py`, `api_dense.py`, `vllm_http.py`, `encoders/`),
  the in-process and hosted rerankers (`external_rerankers.py` and its `RerankSettings`, `hf_dense.py`,
  `accel.py` and the multi-GPU `AccelState` sharding), and the `Encoder` ABC with them
  (`retrieval/encoder.py`; `Embeddings`, `EncodeRole` and `l2_normalize` are re-exported from
  `rcp_ndcg.inference.types`). The hidden budgets (`MAX_SEQ_LENGTH`, `MAX_QUERY_LENGTH`) leave with the
  module: budgets are config fields (`max_tokens`, `query_max_tokens`) that the clients refuse until the
  text-budget mechanism wires the client-side cut. Indexes built by an earlier release (whose `index.json`
  names `provider:` variants) must be rebuilt. The paper's in-process implementations move unchanged to
  `experiments/paper/rerankers/reference/` (one module per family, plus `octen.py`), importable on their own
  with a pinned `requirements.txt`; nothing in the package imports them. The `[local]` and `[vllm]` extras
  themselves left `pyproject.toml` in this release (below); nothing under `src/` imports from them any more
  (`tests/test_no_inprocess_models.py` pins it).
- **The judge's retry delays are the transport's** (the one visible change of the port): within-request
  retries back off 1 s doubling capped at 60 s, or the server's `Retry-After`, where the OpenAI SDK used its
  own delays; the set-aside and parking numbers (5 s doubling to 60 s) are unchanged. `requirements-constraints.txt`
  regenerated without `openai` (and without `httpx2`, its transport, and `jiter`): the judge sends over `httpx`
  through the shared transport. `openai` and `httpx2` remained in `uv.lock` only as the `[vllm]` extra's engine
  package's own dependency (vLLM's server speaks the OpenAI protocol with its own client), and left the lock with
  that extra (below); the `rcp-ndcg` package itself declares and resolves neither.
- `tests/contract` snapshots and the exported schemas (`schemas/judge-config.v1.json`,
  `schemas/run-config.v1.json`) regenerated for the judge port: `OpenAIChat` exported from
  `rcp_ndcg.inference`, `Reply.url`, `JudgeClient`'s `httpx_transport` keyword and its `config`/`usage`
  properties, and the `api`/`extra_body` field descriptions.

- **The OpenAI SDK dependency** (`openai` left `pyproject.toml`'s dependencies; the accepted design of the
  unified inference layer): the judge's chat completions go over `httpx` through the shared transport and the
  `openai_chat` wire adapter, so the package ships one HTTP stack, one retry policy and one error mapping.
  What the judge sent and read is unchanged (the request body, the reasoning channel, the refusals and the
  usage), the judgement family keys do not move, and the only visible difference is the retry delays, which
  now follow the transport's policy. One reading edge, declared: the adapter reads an answer's **first**
  choice, where the SDK era read the last; the judge never sends a `n` above 1, so no shipped answer moves. `requirements-constraints.txt` no longer carries `openai`, `httpx2` or
  `jiter`; in `uv.lock` the two remained only as the `[vllm]` extra's engine package's own dependency, until the
  extras left with the served-only package (above).
- **The `mcp tools` command** (owner decision): the shell fallback for calling one MCP tool without an MCP client
  is gone; the command, its `McpToolsRequest` model and the `rcp-ndcg.mcp-manifest.v1` output-schema id
  (`schemas/mcp-manifest.v1.json` deleted) leave with it, and the MCP surface is `rcp-ndcg mcp serve` alone.
  The tool list and a tool call stay reachable in Python as `rcp_ndcg.mcp.tool_manifest()` and
  `rcp_ndcg.mcp.call_tool()` (what the server itself answers through); tests that drove the CLI command use
  them directly. The MCP tool surface remains the deliberate subset of the command line it always was
  (`rcp_ndcg.mcp.TOOLS`); a plan (`--dry-run`) is CLI-only.
- The release workflow publishes three packages, one GitHub environment each: the build job builds `rcp-ndcg`,
  `rcp-ndcg-core` and `rcp-ndcg-vllm` (the last from its own directory, outside the uv workspace), checks each
  version against the tag, `rcp-ndcg`'s exact `rcp-ndcg-core` pin and the constraints file against the lock, runs
  `twine check` on every file, and uploads one artifact per package; `publish-core` (environment `pypi-core`),
  `publish-rcp-ndcg` (`pypi`, after the core it pins exactly) and `publish-vllm` (`pypi-vllm`) publish by trusted
  publishing, and the GitHub release still attaches the constraints file. The one-time PyPI trusted-publisher
  registration for the three environments (`pypi`, `pypi-core`, `pypi-vllm`) is done; `AGENTS.md` "Releasing" and
  the workflow header describe it.
- **`run resume --judge-urls` and its environment variable** (`RCP_NDCG_JUDGE_URLS` as the
  coordinator's input): pass `run resume --engine judge=url[,url]` instead (repeatable; the runtime overlay of
  the engines you started yourself). The single-engine `serve:` mapping (`serve: {image: ...}`) on a run
  config: `serve:` now maps roles to engines (`serve: {judge: {...}}`). The doctor's `--judge-url` flag is
  `--endpoint <url>`, which probes any role's endpoint.
- **The `[local]` and `[vllm]` extras** (RFC L5, the served-only package): with every in-process model path gone
  (above), the extras and their machinery left `pyproject.toml` — the `local` extra (torch 2.9.1, transformers,
  accelerate, flash-attn 2.8.3), the `vllm` extra, the `[tool.uv] conflicts` pair that kept the two in separate
  environments, and `[tool.uv.extra-build-dependencies]` (flash-attn's build-time torch). `EXTRA_FOR_MODULE`
  (and with it `rcp-ndcg doctor`) no longer names `accelerate`, `transformers` or `vllm`; `torch` maps to
  `[calibrate]`, the one torch requirement in the manifest (the core's `[irt]` extra still carries its own, for
  standalone core installs). A test pins the one-home rule: the extras `EXTRA_FOR_MODULE` names are exactly the
  runtime extras `pyproject.toml` declares.
- Every `uses:` in `.github/workflows/*.yml` is pinned to a full 40-hex commit SHA (the action's own repository,
  resolved through its tags), with the release tag in a trailing comment; a contract test refuses any `uses:`
  that is not (R22).
- The release workflow's build job additionally refuses a `rcp-ndcg-vllm` manifest that depends on `rcp-ndcg`
  without pinning it exactly `==<tag version>` (the check passes without the dependency and without the package).

### Security

- **A profile's default key variables travel only to the profile's own default host** (`OPENAI_API_KEY`,
  `CO_API_KEY`, `VOYAGE_API_KEY`, `GEMINI_API_KEY`, ...). The transport decides it per replica: a request to
  any other URL (a self-hosted engine, a gateway, a third party, a transport injected on another endpoint, a
  judge config swapped to another URL, a stranger in a replica list) carries a key only through the
  config's explicit `api_key_env`. A variable set for one vendor used to authenticate any `base_url`.
- **A config refusal never prints a URL's credentials**: `Endpoint`/`JudgeConfig.base_url` (a replica listed
  twice, the offline fakes mixed into a replica list) and `EngineURLs.urls` (a replica listed twice) raise a
  typed `ConfigError` naming the URLs through `safe_url`, instead of a `ValueError` whose message -- and
  pydantic's rendered `input_value` -- repeated them with userinfo and query. `safe_url` and `redact_urls`
  live in `rcp_ndcg.support.urls` (the support layer, so the serve specs below storage can use them).
- **An explicitly named `api_key_env` travels only to the naming config's own URLs** -- its `base_url`
  replicas, or the profile's default host when it names none -- never to an injected transport aimed at
  another URL (fail closed); an injected transport on the config's own URLs keeps receiving it. A home is the
  exact URL: a host or path that only begins like it, or a query, fragment or userinfo on it, is not home.
- **Credentials embedded in a URL never reach a record, a log or a traceback** (`https://user:pw@host/v1?key=...`):
  an engine record's `error` (persisted in the run manifest and the judgement store) carries the failure's
  type and HTTP status instead of httpx's text, which names the full request URL; the transport chains a
  redacted stand-in under `BackendUnavailableError` and `RequestRejectedError` instead of the httpx
  exception; every rerank error names its server through `safe_url`, in the message and in `details`.
  `rcp_ndcg.support.urls.redact_urls` redacts every URL in free text, beside `safe_url`.
- Dependabot alerts on the default branch's lock (operator snapshot): every alert the lock could carry is
  closed in this one. The `vllm` alerts (27 open when read, the operator's snapshot counted 11, highs among
  them) and its engine-only dependencies (`xgrammar`, `diskcache`) leave the lock with the extras;
  `transformers` stays only through the `mteb` extra at 5.17.0 (≥ the high advisory's first patched 5.10.0);
  `torch` 2.14.0 and `setuptools` 84.0.0 are already at or past their first patched versions (2.13.0, 83.0.0);
  `oauthlib` moves to 4.0.0. No pyproject floor was raised to hold any of them. The constraints file attached
  to the release carries no alerted high advisory.

## 0.1.0

The first public release, accompanying the paper
[Rubric-Calibrated Preferences: Cross-Query Calibration of LLM Judgments via Item Response Theory](https://arxiv.org/abs/2609.35739).

### Overview

RCP-nDCG ships as two distributions. `rcp-ndcg-core` is the metric, the gains, the scoring protocols, the public
records and the IRT estimators, on numpy and pydantic alone. `rcp-ndcg` is the pipeline and the `rcp-ndcg` command
built on it: datasets and rankings, retrieval, judging, calibration, evaluation, runs and their runners.

- **Public modules.** Eighteen modules are public, the ones `PUBLIC_MODULES` in `tests/contract/surface.py` pins:
  `rcp_ndcg_core` with `.metric`, `.gain`, `.protocol` and `.irt`; the facade `rcp_ndcg`; and `rcp_ndcg.data`,
  `rcp_ndcg.data.preprocess`, `rcp_ndcg.retrieval`, `rcp_ndcg.judging`, `rcp_ndcg.calibration`, `rcp_ndcg.eval`,
  `rcp_ndcg.eval.mteb`, `rcp_ndcg.runs`, `rcp_ndcg.runners`, `rcp_ndcg.examples`, `rcp_ndcg.testing` and
  `rcp_ndcg.errors`. Every other module is internal.
- **A judge is one URL.** A judge is an OpenAI-compatible endpoint: one base URL, or a list of replica URLs of the
  same model, and the served model's name. The package starts, builds and configures no engine: vLLM, SGLang, a
  gateway or a hosted API all work unchanged. The client prepares every image and video frame itself, sized as the
  judge's image processor would, so a stock engine needs no media flags. What an endpoint reports about itself is
  recorded beside the judgements and never enters an identity.
- **`Endpoint`.** One model holds what every served model shares: `base_url`, `model`, `revision`, `api_key_env`,
  `concurrency`, the timeouts and `max_retries`. `JudgeConfig` and the hosted retrieval providers extend it, so
  their configs are flat. `model` and `revision` decide what is computed and enter identities; the rest is how the
  endpoint is reached. Keys are read from the environment variable `api_key_env` names, never from a config.
- **The mirror.** A run, or a judgement store, is written to local disk and can be mirrored while it grows to any
  fsspec URI: `s3://`, `gs://`, `az://`, `memory://`, or a filesystem a package registers with fsspec. The mirror
  uses three operations only (write, read and list), uploads the append-only judgement files in immutable parts,
  and restores a missing or shorter local copy on resume, so a preempted job continues on another node. `hf://`
  works but warns, since every write to the Hub is a commit.
- **Runners and `serve:`.** A run executes in the calling process or as one job of a runner: `local`, `slurm`
  (sbatch), `kubernetes` (a Job), or a runner another package registers under the `rcp_ndcg.runners` entry-point
  group. With a `serve:` section, the job also starts the judge's engine replicas from your image and your
  verbatim command, beside the coordinator, and fails when an engine exits or never answers.

### Public surface

**Distributions.** `rcp-ndcg-core` (numpy and pydantic; the `irt` extra adds torch and scipy) and `rcp-ndcg` (the
pipeline and the `rcp-ndcg` command), Python 3.12, published on PyPI (`pip install rcp-ndcg`, `uvx rcp-ndcg`) by
the release workflow on each `v*` tag, with the release's `requirements-constraints.txt` (its lock, exported) beside
it. Extras of `rcp-ndcg`: `hf`, `calibrate`, `mteb`, `local`, `data`, `s3`, `azure`, `http`, `vllm`, `dev`,
`docs`. `hf` holds `huggingface-hub` and `tokenizers` (the judge's tokenizer, without torch). BM25 (bm25s, with
PyStemmer's stemmers) needs no extra. Both distributions carry `LICENSE` (Apache-2.0) and `NOTICE`, which attributes
the third-party code the package adapts.

**Python modules.** One entry per public module; a name that a module re-exports is listed under each module
that exports it.

- `rcp_ndcg_core`: the core in one import. The metric `ndcg` and `dcg` with `TieRule`; the gains `gain`,
  `pass_probabilities`, `count_gain` and `qrel_gain` with the type `Gains`; the scoring protocols `Protocol`,
  `PROTOCOLS`, `score_query` and `aggregate`, with `MetricName` (`rcp_ndcg`, `qrel_ndcg`, `count_ndcg`). The public
  records: `ItemParams` (2PL `gamma` and `beta` per criterion), `QueryParams` (`tau`, `alpha`), `Judgement` (one
  judge call's parsed observation), `JudgementSet` (judgements with their families; `JudgementSet.merge` keeps one
  judgement per window, `record_id`: its latest valid one), `Placement` (one document shown in one window) and
  `Family` (the poolability token of a set of judgements; its `tokenizer` is the SHA-256 of the judge's
  `tokenizer.json`, part of `key` and left out of `rubric_key`). The content parts of a query or a document:
  `Content`, `TextPart`, `ImagePart`, `VideoPart`, their union `Part`, `MediaRef` and `Modality`.
- `rcp_ndcg_core.metric`: `ndcg(scores, gains, *, k=10, ties="group_mean", ideal=None)`, `dcg`, `ideal_dcg` and
  `discount` (`1 / log2(rank + 1)`), with the tie rules `TieRule = "group_mean" | "doc_id_desc" | "input_order"`.
  RCP-, qrel- and Count-nDCG differ only in their gains. `rank_by_score` orders documents with a deterministic
  tie-break, and `tie_groups(scores)` gives the equal-score classes that `group_mean` credits, for custom metrics.
- `rcp_ndcg_core.gain`: `gain(theta, items)` (the RCP gain), `pass_probabilities` (per criterion, for one ability
  or an array), `count_gain(passes, placements)` (the Count-nDCG baseline), `qrel_gain(grade, scheme="linear")`,
  the type `Gains` (query id to document id to a gain in [0, 1]), `item_arrays`, which reads and checks item
  parameters of any shape, and `sigmoid`, the one logistic function of both packages.
- `rcp_ndcg_core.protocol`: `Protocol` and its presets `PROTOCOLS` (`nanobeir`, `bright`, `vidore`, `trecdl`,
  `mteb`, `plain`), `resolve_protocol` (a preset name to its `Protocol`), `candidate_docs` (the documents of one
  query that enter the ranking once the protocol has removed the excluded ids, the document that shares the
  query's id, and what lies outside the judged pool), `score_query` and `aggregate` (the mean over queries per
  dataset, then the unweighted mean over datasets), with `MetricName`. A query without a positive grade has an
  undefined qrel-nDCG and is left out of the mean.
- `rcp_ndcg_core.irt`: `fit_bradley_terry`, `fit_calibration` (the 2PL fit, returning a `CalibrationFit` with its
  `FitDiagnostics`), `score_document` (a document's ability from its own rubric answers, the items frozen) and
  `insert_document` (a new document into a tournament fit, the existing abilities frozen), with `Priors`,
  `AbilityPrior`, `DEFAULT_SE_TARGET` and `MIN_OPPONENTS`, and the refusals `UnidentifiableInsertion`,
  `ScaleMovementError` and `JudgeOverlapError`. torch is imported on first use only.
- `rcp_ndcg` (`import rcp_ndcg as rcp`): the facade. Fifteen functions, `load_dataset`, `load_rankings`,
  `evaluate`, `compare`, `retrieve`, `rerank`, `fuse`, `estimate`, `judge`, `calibrate`, `score_documents`,
  `insert_documents`, `run`, `ndcg` and `gain`; the types `Dataset`, `Rankings`, `Gains`, `JudgeConfig`,
  `Endpoint`, `RetrieverConfig`, `RerankerConfig`, `TournamentSchedule`, `RubricSchedule`, `Preprocessing`,
  `JudgementSet`, `Calibration`, `Extension`, `CostEstimate`, `EvalReport`, `Comparison`, `Protocol`, `RunConfig`
  and `Run`; and `__version__`. `rcp.run(..., estimate=True)` returns a `CostEstimate`, otherwise a `Run`.
- `rcp_ndcg.data`: `Dataset` and `load_dataset` over URIs (`hf://`, `suite:`, `beir:`, `jsonl:`, `images:`, `videos:`,
  `frames:`), the public suites `SUITES` of `Suite` records (a suite's name is also its protocol preset;
  `VIDORE_NATIVE_LANGUAGE` gives the language each ViDoRe v3 domain is scored in), `Rankings` (one table: `system`,
  `dataset`, `query_id`, `doc_id`, `score`; `DEFAULT_SYSTEM` names a single unnamed system) and `load_rankings`
  (Parquet, CSV, TREC run, JSONL; `dataset=` names the subset of a file whose rows name none, such as a TREC run, which
  holds one subset per file). `Rankings.queries()` and `for_query()` read the one dataset the rows name, or the one
  given: the rows that name it, else the rows that name none (`Rankings.resolve_dataset`); among several they refuse to
  guess. A `Dataset` read from the Hub is read at the commit its identity records, and `Dataset.revision` is that
  commit. In-memory input goes through typed records:
  `Dataset.from_records(name=, queries=, corpus=, qrels=, candidates=, excluded=)` and `Rankings.from_records` take
  dicts or the row models `QueryRow`, `DocumentRow`, `QrelRow` and `RankingRow`, and refuse unknown keys, missing
  fields, non-finite numbers, duplicates and ids that do not join (`DataError` naming the record and the closest known
  key). pandas is an output format only (`to_pandas`); a frame goes in as `frame.to_dict("records")`.
  `validate(dataset, rankings=None)` returns a `ValidationReport` of `ValidationCheck`s (the checks behind
  `data validate`); it checks the rankings rows that rank the dataset, as `evaluate` reads them, and reports
  `NO_RANKINGS` when none do. `Dataset.parts` and `Rankings.datasets` are properties, and the objects print one-line
  summaries. Media: `MediaResolver` resolves media references through a content-addressed cache (`default_resolver()` is
  the process-wide one); a missing asset is a `MissingInputError` (exit 4), and one that cannot be read or decoded, or
  does not match its hash, a `MediaError`, which is a `DataError` (exit 12), as is a clip the video policy refuses.
  Media documents are named by their path below the dataset's root, however the root is spelled, and two files that
  would share an id are refused. The preprocessing policies `Preprocessing`, `TextPolicy`, `ChunkPolicy`, `ImagePolicy`
  and `VideoPolicy` (below), and the judge's tokenizer: `load_tokenizer` and `TextTokenizer` load a Hugging Face
  repository's `tokenizer.json` (optionally `@revision`) or a local one, once per process.
- `rcp_ndcg.data.preprocess`: what a judge sees of a document. `Preprocessing` holds the policies of one judging
  pass and is part of its identity. `TextPolicy(on_overflow, max_tokens)` with `OnOverflow` (`keep`, `truncate`,
  `chunk`, `fail`), `DEFAULT_MAX_TOKENS` and `DEFAULT_TEXT_POLICY`; `ChunkPolicy(max_tokens, overlap_tokens)`.
  Text limits count tokens of the judge's tokenizer, and every cut falls at a token boundary of the original text,
  found with the tokenizer's offset mapping, so a truncated document is a verbatim prefix (`token_prefix`) and a
  chunk a verbatim slice (`split_into_chunks`). A policy that cuts refuses to run without a tokenizer; there is no
  character fallback. `apply_text_policy` is the one place a document's text is shortened, and a `fail` policy
  raises `DocumentOverCapError`; `TextTruncationCensus` records every cut as a `TextCutRecord`. Chunking:
  `chunk_ranking_example`, chunk ids `<doc_id>` `CHUNK_ID_SEPARATOR` `<n>`, `document_id_for_chunk`,
  `document_ids_from_chunks`, and the pooling back onto documents, `max_pool_scores_by_document` (a document
  scores as its best chunk) and `max_pool_rubric_window_by_document` (a document passes a criterion in a window
  when any of its chunks does). `ImagePolicy` is a pixel budget plus the judge's image processor family
  (`processor`, resolved by `ImagePolicy.for_processor`); `VideoPolicy` chooses the frames a judge is shown.
- `rcp_ndcg.retrieval`: `RetrieverConfig`, a union of `BM25Config`, `DenseConfig` and `LateInteractionConfig`;
  the encoder and reranker configs `EncoderConfig` and `RerankerConfig`, unions by `provider`: `local` (`Local`,
  `LocalEncoder`, in process with the `local` extra), `openai_compatible` (`OpenAICompatible`,
  `OpenAICompatibleEncoder`, `OpenAICompatibleReranker`: vLLM, a gateway, or OpenAI's API), `cohere` (`Cohere`),
  `voyage` (`Voyage`) and `gemini` (`Gemini`), the hosted ones built on `Endpoint`. `index` builds an `Index`,
  `load_index` reads one, `search` searches it, `retrieve` does both, `rerank` rescores each query's top
  candidates (its checkpoint is keyed by the reranker and each query's candidates, so a resumed rerank never reuses
  another's scores), and `fuse` is reciprocal rank fusion, also of a single system, keeping the `dataset` column.
  Rankings are read per dataset (`Rankings.for_query`). A hosted encoder refuses a `batch_size` over its vendor's
  limit.
- `rcp_ndcg.judging`: judging. `JudgeConfig` is one OpenAI-compatible endpoint (an `Endpoint`) with its `decoding`,
  `max_images`, `max_videos`, `tokenizer` (a Hugging Face repository id with an optional `@revision`, or a
  `tokenizer.json` path; the shipped self-served judges name their model's repository) and `image_processor`
  (`qwen2_vl`, `qwen2_5_vl`, `qwen3_vl`); `base_url` is one URL or a list of replica URLs, `urls` the tuple, and
  `temperature` defaults to none sent. `JudgeClient` routes least-in-flight over the replicas and handles outages
  per replica; `probe` records what each endpoint serves as `EngineInfo`, in the judgement store and the run manifest,
  never in an identity; `Usage` accumulates a client's calls and tokens. `TournamentSchedule` and
  `RubricSchedule` are specified in placements per document (`random_placements`, `stratified_placements`,
  `adaptive_placements`; `placements_per_doc`, `random_share`), so a query's window counts scale with its pool and
  with a re-judged subset (`docs=`, either stage). Windows hold `min(window, n)` documents, so the placements hold
  for a pool smaller than a window too (one adaptive window per batch when the pool fits in one), and `estimate`
  counts exactly the windows judging asks. The defaults give the paper's counts at a pool of 150, and
  `for_modality` changes only the window size (the page-image and video tournament asks 7 x 16 adaptive windows at
  150). `judge` writes into an append-only `JudgementStore` (each
  window's text budget counted in the judge's tokens; `windows=` asks exactly the given windows, e.g. an insertion
  plan; the store keeps each prompt's text as `prompts/<sha256>.txt`, and `JudgementStore.schedule(stage)` reads a
  stage's schedule back); `reparse` re-reads a store's stored answers with the current parser into a new store.
  `estimate` returns a `CostEstimate` of calls, input and output tokens and wall time (no dollar figure): input
  tokens exact with the judge's tokenizer, otherwise approximated at 2 characters per token, as `input_token_count`
  says; images of a judge without an `image_processor` approximated at 1,000 tokens each, stated in the
  assumptions. `load_prompt` returns a
  `Prompt`: the tournament and the C1 to C5 rubric prompts, for text, page images and video; a prompt without
  `{query_placeholder}` and `{passages_placeholder}` is refused.
- `rcp_ndcg.calibration`: `read_judgements` (every judgement of the given stores, per window its latest valid
  one) and `calibrate` (modes `auto`, `tournament`, `rubric_only`; judges `single` or `pooled`; rubric verdicts must
  answer exactly the family's declared criteria `C1..CK`, and a window judged twice counts once), with `Priors` and
  `judged_bt_l2` (the Bradley-Terry penalty the stores' live tournament fit used). The immutable `Calibration` has a
  seven-file layout: `gains()` and `theta_map()` key queries by their own id for one dataset, else
  `<dataset>||<query_id>` (`QUERY_ID_SEP`); `to_pandas("thetas" | "queries" | "items")` gives the thetas, as
  `ThetaRow`s, with a `gain` column. `score_documents` and `insert_documents` add documents the calibration lacks
  and return an `Extension` of `ExtensionRecord`s with its `AnchorReport`; `select_opponents` plans an insertion
  (with `window=`, the plan split into windows that each hold the new document).
- `rcp_ndcg.eval`: `evaluate` returns an `EvalReport` of the `METRICS` (`rcp_ndcg`, `qrel_ndcg`, `count_ndcg`) per query
  (`QueryValue`), per dataset (`DatasetValue`, whose `num_queries` counts the queries with a defined value) and as the
  summary (`SummaryValue`, with a bootstrap interval), with typed `ReportWarning`s. Rankings over a suite whose subsets
  share query ids (BRIGHT, ViDoRe v3, NanoBEIR) need their `dataset` column, or one subset scored on its own; without it
  `evaluate` refuses them (`DataError`, naming the subsets). A calibration of one dataset gives its gains only to the
  subset of that name; `leaderboard()` is the wide system x dataset table with the summary as `mean`, and `inputs`
  (`ReportInputs`) the files a command line scored, with the resolved commit of a Hub dataset (`revision`). `compare`
  (`systems=` restricts it) returns a `Comparison` of `PairComparison`s (the paired t-test, as in the paper, and a
  bootstrap interval) and the `SignFlip`s where RCP-nDCG and qrel-nDCG prefer different systems;
  `Comparison.to_pandas()` counts them. `sensitivity` is the paper's sensitivity. `explain` returns a
  `QueryExplanation`: each system's top k (`SystemExplanation`, `RankedDocument`), the gaps between systems split into
  selection and ordering (`ScoreDelta`), and the criteria's item parameters once, as `CriterionParams` in `items`;
  `per_criterion` gives each criterion's `CriterionContribution` to a gain.
- `rcp_ndcg.eval.mteb`: `get_tasks` returns the released suites as mteb tasks scored with `ndcg_float_at_k`
  (group-mean ties) at the cutoffs `K_VALUES`; `ndcg_float_scores` computes those scores, and `task_metadata` reads
  a suite's released revision and task metadata from its `rcp_ndcg_tasks.py`.
- `rcp_ndcg.runs`: `RunConfig` (unknown keys refused; `extends:`; a relative path resolves against the file that
  declares it, before `extends:` merges the files; `RunConfig.load` also takes a packaged config's name;
  `RunConfig.local_inputs()` lists the local paths a job would need, which a Kubernetes job refuses; `RunConfig.mirror`
  and `mirror_interval_s`), and `RunConfig.serve`, a `ServeConfig`: the judge's engine, your image and verbatim command,
  started beside the run's job by the `slurm` and `kubernetes` runners; `startup_timeout_s` (default 1800) and
  `outage_timeout_s` (default 900) bound how long the job waits for its engine to answer and its judge for an engine
  that stopped answering. A job that starts its engine checks that its image has what the job needs, fails when the
  engine exits or never answers, and stops both on cancellation. A failed job is not retried unless Kubernetes'
  `backoff_limit` says so; `run resume --runner` submits the run again with its recorded runner options and `serve:`.
  One replica on Kubernetes is a single container in the engine's image running the same supervision script as SLURM.
  The run layout `rcp-ndcg.run-layout.v1` (`LAYOUT_VERSION`, `RunLayout`) holds its manifest `RunManifest`
  (`MANIFEST_SCHEMA`): the `DatasetRef`s evaluated, the `RunStatus`, and a `StepRecord` per step of `STEPS` (`StepName`:
  retrieve, rerank, tournament, rubric, calibrate, evaluate) with its `StepStatus` (`running`, `completed`, `failed`,
  `cancelled`) and the engines the judge met. A run stopped by SIGINT or SIGTERM is recorded as failed, or as cancelled
  when `run cancel` stopped it; a failed judging step keeps its usage; the judge's runtime fields (its URL, for one) and
  the mirror are not a change of config. `Pipeline` runs the steps against a run directory; `new_run_id` and
  `discover_runs` name and find runs. `Run.status()` (`RunState`) lists every planned step in run order (`pending` until
  it starts), each judging step with its `progress` in judge windows (`StepProgress`: `done`, `planned`), and a `done`
  flag; a no-op resume leaves completed steps completed. When a run's jobs have ended but its manifest is behind, or a
  newer manifest is on the mirror, `status()` says `failed` or `cancelled` with `done` set and a `note` saying why. The
  mirror: `Mirror`, `MirrorState` (what it last did, including a failed final upload; `run status` shows it), `mirrored`
  (restore, then mirror while a block runs) and `restore`. A mirror can be any fsspec URI, `file://` or a plain path
  included; one no installed filesystem serves is refused before a run directory is created, and in `--dry-run` and
  `--estimate`. Restoring replaces whole files when the local manifest is older than the mirror's.
- `rcp_ndcg.runners`: `JobSpec` (a named command, with `serve`), `JobOptions` and `Resources` (a runner's `resources`
  and `env` are its jobs' defaults, env names must be shell identifiers, and every path a job records is absolute),
  the `JobRunner` protocol (submit, status, logs, cancel over a `JobHandle`, with a `JobStatus`; a cancel that
  cannot find or stop its job raises), `get_runner` with the `local`,
  `slurm` and `kubernetes` runners (`LocalRunner`, `SlurmRunner`, `KubernetesRunner`) and their options models
  (`LocalOptions`, `SlurmOptions`, `KubernetesOptions`; a plugin runner's name cannot be a public one's),
  `RunnerError` (exit 6), `ServeConfig` (`image` is optional, `nodes_per_replica` must be 1, and a SLURM job
  without a container runtime refuses an `image`; Kubernetes StatefulSet and Service names are cut to 52
  characters, deterministically), `worker_script`
  (the bash script a job's task runs), `install_argv` and `COORDINATOR_IMAGE` (a coordinator installs the release
  into a stock uv image with uvx), and `ENTRY_POINT_GROUP`, the `rcp_ndcg.runners` entry-point group for other
  runners.
- `rcp_ndcg.examples`: the tiny example dataset (`tiny()`) and the run configs `tiny`, `rejudge_nfcorpus` and
  `nano_nfcorpus_gpt5` (`run_config_names`, `run_config_path`), shipped as package data.
- `rcp_ndcg.testing`: `FakeJudge`, a deterministic offline judge (its rubric's criterion difficulties are
  `DEFAULT_DIFFICULTIES`), and `build_tiny_world`, a small judged and calibrated world on disk (`TinyWorld`), built
  from `tiny_rows` with the small schedules `TINY_TOURNAMENT` and `TINY_RUBRIC`.
- `rcp_ndcg.errors`: typed errors with exit codes (`ExitCode`): `RcpNdcgError` and its subclasses `UsageError`,
  `ConfigError`, `MissingInputError`, `CredentialsError`, `ProviderError`, `CapabilityError`, `Interrupted`,
  `DependencyError` (`dependency_error` builds one whose hint installs the missing extra), `IdentityError` and
  `DataError`; `classify` maps any exception onto them, and `error_class(exit_code)` gives the class of an exit code (a
  caller that ran a command as a child process raises its failure as that class). `EXTRA_FOR_MODULE` maps each optional
  module to the extra that installs it, which `doctor` checks. An error's `hint` is worded for Python callers and its
  `cli_hint`, when it differs, for the command line (which shows it). Warnings are `RcpNdcgWarning`s with a
  `WarningCode` from `WARNING_CODES` (`APPROXIMATE_IMAGE_TOKENS`, `BT_L2_MISMATCH`, `INVALID_WINDOWS`,
  `UNCALIBRATED_DOCUMENTS`, `UNREADABLE_RUN`).

**Command line (`rcp-ndcg`).** `data` (fetch, inspect, validate, convert into a layout `load_dataset` reads, formats),
`retrieval` (index, search, rerank, fuse), `judge` (tournament, rubric, reparse), `calibration` (fit, score, insert,
show), `eval` (score, compare, explain), `run` (start, resume, status, logs, cancel, list, show), `schema` (list, show,
export), `mcp` (serve) and `doctor`. Every command except `mcp serve` takes `--json` and prints one
`rcp-ndcg.cli.v1` document; judging and runs take `--estimate` and `--dry-run`; `run start` takes
`--runner` and `--detach`, and its `--dry-run` prints what a runner would submit; `run resume` takes
`--judge-urls` (`RCP_NDCG_JUDGE_URLS`), the replica URLs a runner hands its job, and `--runner` to submit a failed
run again. `run start --judge` lists the shipped judge configs in its help, and `--judge-model` needs
`--judge-url`. `run cancel` fails with exit 4 when it cannot find or stop a job, and never marks a run cancelled
that it did not cancel. `run show` lists a run's artifacts: config, candidates, tournament, rubric, calibration,
report, comparison, log and jobs. Each `retrieval` command documents its own `--set` keys. `run start`, `run resume`,
`judge tournament` and `judge rubric` take `--mirror`. `--estimate` and `--dry-run` give the refusals of the real
command (a judging identity that differs from the store's) and write nothing.
`run resume --set` keeps its change only when the resume succeeds, and `run resume --only` never changes the run's
recorded steps. `calibration insert --dry-run --judgements STORE --out PLAN` plans an insertion's windows with the
store's schedule, and `judge tournament --plan PLAN` asks exactly those windows. `run start` takes a config file or
a packaged config's name (`run start tiny`); `data fetch --dataset tiny --out DIR` copies the example data.
`eval score --json` prints the summary, the per-dataset means and the warnings (`rcp-ndcg.eval-score.v1`), with
`--per-query` and `--fields` to add or select, and `--out` for the full report; `eval explain` reads a run or a
saved report (`--report`) and adds texts only with `--include-text`; `eval compare --run` leaves the reference
systems `candidates` and `judge` out unless `--include-reference`; a run's `evaluation.systems` take multi-system
files and `<file>#<system>`. `schema show commands` prints a compact command index with each flag's help (`--full`:
the click tree). A config that does not validate lists its problems in `error.details.errors` (field, input,
expected, did-you-mean, and whether `--set` or the file set it).

**Exit codes.** 0 success, 1 internal, 2 usage, 3 config, 4 missing input, 5 credentials, 6 provider, 8 capability,
9 interrupted, 10 dependency, 11 identity, 12 data; 7 is retired and never returned. A judge endpoint that refuses
the credentials (HTTP 401, 403) stops a pass with exit 5, and one without the route or model (HTTP 404) with exit 6.

**MCP tools** (`rcp-ndcg mcp serve`). Read-only: `describe` (the command index), `schema_show`, `data_inspect`,
`eval_compare`, `eval_explain`, `calibration_show`, `run_list`,
`run_show`, `run_status`, `estimate`. `eval_score` is not read-only: it overwrites `out` with the full report when
given (its `out`, `per_query`, `fields` are as on the command line). Destructive: `run_cancel`. `run_start` starts a
run and returns its directory at once. The tool list and a call are Python calls too: `rcp_ndcg.mcp.tool_manifest()` and `rcp_ndcg.mcp.call_tool()`.

**JSON Schemas** (`schemas/`, `rcp-ndcg schema export`): the configs `run-config` and `judge-config`; the artifacts
`judgement`, `judgement-store` (a store's `identity.json`), `calibration` (a calibration's `items.json`),
`calibration-coverage`, `calibration-identity`, `extension-record` (a line of a calibration's `extensions.jsonl`),
`index`, `run-manifest`, `eval-report` and `comparison`; the output
of every `--json` command (among them `cost-estimate` and `extension`), the `cli` envelope, the `commands` tree and
the `command-index`. Every artifact names its schema in its `schema` field, and every
property carries a description (a config field's is its model's documentation).

### Fixed: tournament answers the paper's code could not parse

The code behind the paper (arXiv v1) turned some unparseable tournament answers into rankings read from the
response text, and weighted them five times as heavily as a normal window. This release decodes such answers
tolerantly and never invents a ranking. The effect on the paper's tables is small:
- NanoBEIR reranker means move by 0.12 to 0.25 pp, and BRIGHT means by 0.35 to 1.14 pp;
- no system ranking changes, and no significant comparison reverses.

A minor correction to the paper is forthcoming. Details and all numbers are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md#tournament-answers-the-papers-code-could-not-parse).

### Changed from the paper's code: text limits count the judge's tokens

The paper's judging limited document text by characters, converting a window's share of the judge's context at 2.0
characters per token. This release counts the judge's own tokens and cuts at token boundaries. A document that fits
under both rules is shown in full either way, so only a long document that is cut can end at a different place.
Details are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md#3-re-judge-a-pool-with-your-own-llm-endpoint).

### Also in this release

- `experiments/`: recomputes the paper's leaderboards, the human contest study and the external-judge comparisons
  from the public datasets, and checks every value against the paper.
- `examples/`: seven examples, five of them offline, on the tiny dataset that ships with the package.
- Judge configs for the paper's judges and a hosted judge ship inside the package (`--judge NAME`).
- `experiments/paper/`: the paper's engine command for each judge with its image pinned (`serve/`), and the
  configs of its first-stage retrievers (`retrieval/`) and rerankers (`rerankers/`).
- `skills/rcp-ndcg/`, a skill for coding agents that use the package, and contributor notes for coding agents.
