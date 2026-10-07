# Changelog

## Versioning

While the version is `0.x`, a change that breaks the public surface bumps the minor version and an additive change
bumps the patch version; from `1.0` on, semantic versioning applies. The public surface is what
`tests/contract/snapshots/` and `schemas/` pin:
- the Python names in the `__all__` of the public modules, which `PUBLIC_MODULES` in `tests/contract/surface.py`
  lists: the facade `rcp_ndcg`; `rcp_ndcg_core` with `rcp_ndcg_core.irt`, `.metric`, `.gain` and `.protocol`; and
  `rcp_ndcg.data`, `rcp_ndcg.data.preprocess`, `rcp_ndcg.inference`, `rcp_ndcg.retrieval`, `rcp_ndcg.llm`,
  `rcp_ndcg.calibration`, `rcp_ndcg.eval`, `rcp_ndcg.eval.mteb`, `rcp_ndcg.runs`, `rcp_ndcg.runners`,
  `rcp_ndcg.errors`, `rcp_ndcg.testing` and `rcp_ndcg.examples`;
- the command tree with its flags and output schemas, the exit codes, the MCP tools, the packaging (distributions,
  extras, entry points) and the JSON Schemas in `schemas/`.

Every other module (for example `rcp_ndcg.cli`, `rcp_ndcg.storage`, `rcp_ndcg.support`, `rcp_ndcg.data.io` and the
submodules of `retrieval`, `llm`, `calibration`, `runs` and `runners`) is internal and may change without notice. A
pull request that changes `tests/contract/snapshots/` or `schemas/` must add an entry here; CI checks it.

Every artifact schema carries its own version (`rcp-ndcg.<name>.v1`), bumped only when that artifact changes
incompatibly, independently of the package version. `rcp-ndcg` pins `rcp-ndcg-core` to its own version; the two are
released together.

## Unreleased

### Public surface

- **T4 end to end: the run scenarios the GPU validation drives inside the pod** (`rcp-ndcg-vllm`): the
  scenario configs `packages/rcp-ndcg-vllm/scenarios/*.yaml` (schema `schema/scenario.schema.json`),
  the stage `python -m rcp_ndcg_vllm.e2e`, the entry `src/rcp_ndcg_vllm/jobs/e2e.sh` and the submission
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
  rendered script (harness tests), the four-phase supervision re-run with the fake engines as its
  engines (`tests/runners/test_supervision_replay.py`) and the observed outage behaviour as a transport
  test (`tests/inference/test_outage_observed.py`).
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

### Fixed

- **Two recipes load against the current product** (`rcp-ndcg-vllm`; the shakedown's FINDINGS.md rows
  for `qwen3-embedding-0.6b` and `qwen3-vl-embedding-2b`): the embedding recipe's template drops its
  refused empty trailing fixed marker (the shape ends in its content span with `add_special_tokens:
  true`; rendered ids unchanged), and the VL recipe declares `client.max_images`/`max_videos` (1/1,
  mirroring `serve.limit_mm_per_prompt`) as the schema requires of an input that declares media. The
  recipe lanes own the lasting content; these are the minimal fixes the T4 scenarios need to mount.
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
- New served recipe `octen-embedding-8b` (`packages/rcp-ndcg-vllm/recipes/octen-embedding-8b/`): the paper's
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
  `TemplateSpec` data with the model's default instruction pinned as fixed text, the template file shipped
  (`serve.chat_template: template.jinja`, R10), the media pixel budget pinned on both sides
  (`serve.mm_processor_kwargs` `images_kwargs` min 4096 / max 1843200, mirrored in `client.recipe`, R20),
  explicit `client.tokenizer` + `max_tokens: 8192` with `on_overflow: cut`, `empty_doc: send_text "NULL"`
  (the card's NULL rule), and the subprocess reference running the card's `Qwen3VLEmbedder` (vendored
  verbatim, sha256-pinned; the render mode mirrors the anchor-preserving cut, and the card's whole-prompt
  right cut is the declared `anchor_drop_over_cap` deviation). The recipe's tests run stage 1 on CPU against
  the pinned tokenizer (offline: skipped with a clear reason) and prove the over-cap cut and the query-side
  frame byte-identical.

- **`rcp-ndcg-vllm` gains the release-candidate and wave scripts** (`packages/rcp-ndcg-vllm/jobs/`, and
  `wave0.sh` with the package): `rc_build.sh <name> [<commit>]` builds an RC exactly as `release.yml`
  does — the three distributions, the version and pin checks, the constraints-file check against the
  lock, `twine check`, and a fresh-venv install smoke from the wheelhouse — and stages the six files
  with the wheelhouse (every locked dependency beside the release wheels, the CPU torch build included,
  the plugin wheels under `packages/rcp-ndcg-vllm/plugins/*` built beside them), the recipes, the wave
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
  (`rcp-ndcg.wave0-report.v1`, schema at `packages/rcp-ndcg-vllm/schema/wave0-report.schema.json`) and
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
- New package `rcp-ndcg-vllm` (`packages/rcp-ndcg-vllm/`, outside the root uv workspace and lock; version
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
  exported at `packages/rcp-ndcg-vllm/schema/recipe.schema.json`. The recorder (`record`),
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
- The first serving recipe ships: `packages/rcp-ndcg-vllm/recipes/topk-embed-v1-small/` (recipe.yaml,
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
  `RerankClient` and `PoolingClient` derive from it; the judge client adopts it later.
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
- New package `rcp-ndcg-vllm-topk` (`packages/rcp-ndcg-vllm/plugins/topk/`, outside the root uv workspace and
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

- A served recipe in `packages/rcp-ndcg-vllm/recipes/`: `ctxl-rerank-v2-instruct-multilingual-2b`
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
  the reference subprocess derives from `experiments/paper/rerankers/reference/contextual.py` with the
  instruction fold the served client applies. Status `unverified` until the GPU waves run the harness's
  stages 2–3.

- The first served recipe in `packages/rcp-ndcg-vllm/recipes/`: `qwen3-reranker-0.6b`
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
  is pinned to sample exactly `num_frames` frames per container (vLLM `--media-io-kwargs`, SGLang
  `--mm-process-config`). Required for `wire: video_url`, refused under `wire: frames` (see below).

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
    `use_activation` travel only when the config sets them. `interpret` parses the `results`, Voyage `data`
    and SGLang bare-list answer shapes and realigns the scores by `index`; an index missing, duplicated or out
    of range is a non-retryable `ProviderError` naming the server, an over-length 400/422 a `CapabilityError`
    hinting `max_tokens`, any other refusal a `RequestRejectedError`. A candidate set above the cap (or a set
    `batch_size`) is split into requests and merged; a `listwise` config refuses to split
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
    `CompletionInput` and `Completion` moved here from `rcp_ndcg.llm.client` unchanged (they stay importable
    from `rcp_ndcg.llm.client`, where the first two and the two error types remain in its `__all__`);
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
  `rcp_ndcg.llm.client` (still importable and exported there). Exit codes do not change: both remain
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
- **`rcp_ndcg.llm.client` gains `api` and `headers_env`** through `Endpoint`; `wait_on_outage_s` moves up to
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
    `MAX_VIDEO_BYTES`, `VIDEO_CACHE_SIZE`) moved here unchanged from the internal `rcp_ndcg.llm._payload`
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
    through the route registered by `rcp_ndcg.llm._fake`, so `JudgeConfig.fake(seed)` builds a real
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
  `rcp_ndcg.llm.client`; the transport's accumulator produces it (its former `calls`/`failed_calls`
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
    `localhost`. GPUs are partitioned among the engines of a phase (below, [serving](docs/concepts/serving.md)).
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
- New plugin distribution `rcp-ndcg-vllm-pplx` (`packages/rcp-ndcg-vllm/plugins/pplx/`, pure Python, dependency-free):
  registers the `PplxContextualModel` architecture (perplexity-ai/pplx-embed-v2-context-9b-preview,
  revision `b667039e`) with stock vLLM v0.31.x through the `vllm.general_plugins` entry point
  (`rcp_vllm_pplx:register`), so the unmodified `vllm/vllm-openai:v0.31.0` image serves it after
  `pip install --no-deps <wheel>`. Registers one lazy model class (`PplxContextualForPooling`, one embedding per
  chunk via a boundary-marker segment pooler and the checkpoint's fp32 `contextual_projection` + int8 tanh head,
  no vision tower, no lm_head) and a `MODELS_CONFIG_MAP` handler that forces the checkpoint's
  `is_causal=false` attention contract on both HF configs. At import it refuses any vLLM outside
  `>=0.31,<0.32`. The client contract (token ids with the role prefixes, per the plugin README) needs the
  product's `request_shape: token_ids`, which no adapter implements yet (product gap, recipe lane), and the
  recipe's chunker must not emit the `<|chunk_sep|>` marker as chunk content (the id wire cannot tell it from a
  boundary; declared in the plugin README).
- The served recipe `jina-reranker-v3` (`packages/rcp-ndcg-vllm/recipes/jina-reranker-v3/`,
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
- New recipe `qwen3-vl-reranker-2b` (`packages/rcp-ndcg-vllm/recipes/qwen3-vl-reranker-2b/`, shipped in the sdist):
  Qwen/Qwen3-VL-Reranker-2B as a pointwise reranker on the stock `vllm/vllm-openai:v0.31.0` pooling runner --
  three-key `hf_overrides`, the served chat template as data plus a shipped template file (rewritten to the recipe
  variable convention, byte-equal under the engine's render), the explicit 8192-token budget with a 4096 query
  share, `instruction: none` (the card's default instruction pinned as fixed frame text), `use_activation: true`
  (probability), `empty_doc: send_text "NULL"`, and `mm_processor_kwargs` min_pixels 4096 / max_pixels 1310720
  (1280 tokens/image, R20). `reference.known_deviations: [anchor_drop_over_cap]`: the card's script truncates
  over-cap pairs itself. Status `unverified` until the GPU waves run stages 2-3.

### Fixed

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
  (vLLM `--media-io-kwargs`, SGLang `--mm-process-config`), a single-frame container is refused (the declared
  instrument merges frames in time, which needs at least a temporal pair; a single frame is an image), and
  the declaration is refused under `wire: frames`, which samples on the
  client. `wire: frames` stays the default and exact. SGLang's video path caps per-frame pixels lower than
  the declared budgets (602,112 px, clip-dependent), so the declared count is stock vLLM's there; the
  pinning ties the frame count and `engine_media_check` (above) compares the engine's actual count at run
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

### Changed

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

- `tests/contract` snapshots and the exported schemas (`schemas/index.v1.json`, `schemas/judge-config.v1.json`,
  `schemas/run-config.v1.json`) regenerated for the moved and new fields; `tests/test_errors.py` now requires
  one *root* class per exit code, since the moved outage and refusal types are `ProviderError` subclasses and
  exit codes do not change.
- `tests/contract` snapshots and the exported schemas regenerated for the embedding adapters, the
  embedding role client and the `api_key_env` refusal (the
  `EmbeddingClient` export, its constructor and `EmbeddingEndpoint.identity_extra()`).
  exit codes do not change. The `python_api` snapshot regeneration also records `JobSpec.phases`, which its
  commit (the job-phase shape) had left out.
- The fake judge's deterministic draws (`_uniform`, `_hidden_ability` in `rcp_ndcg.llm._fake`) now come from
  `rcp_ndcg.inference.fake` (`fake_uniform`, `hidden_ability`): one home for the mechanism the fakes share;
  identical values, and both names stay importable from `rcp_ndcg.llm._fake`.
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

### Removed

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
  `rcp_ndcg.data.preprocess`, `rcp_ndcg.retrieval`, `rcp_ndcg.llm`, `rcp_ndcg.calibration`, `rcp_ndcg.eval`,
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
- `rcp_ndcg.llm`: judging. `JudgeConfig` is one OpenAI-compatible endpoint (an `Endpoint`) with its `decoding`,
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
