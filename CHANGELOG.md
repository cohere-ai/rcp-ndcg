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

- New package `rcp-ndcg-vllm` (`packages/rcp-ndcg-vllm/`, outside the root uv workspace and lock; version
  0.0.1, depends on `rcp-ndcg==0.0.1`): serving recipes for vLLM as data. The recipe's `client` block **is**
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
  exported at `packages/rcp-ndcg-vllm/schema/recipe.schema.json`.

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
- `JobSpec` gains `phases` (a tuple of `JobPhase`: the engines one phase starts, by role, and the command it runs
  while they serve); a job sets `phases` or `serve`, not both.
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
  - `inference.adapters`: the `Adapter` protocol (generic in request and result) and its registry
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

### Fixed

- The MCP server logs the typed warnings a tool call collects (its results have no `warnings` field, so the
  server's log is where e.g. `UNPINNED_REVISION` surfaces there).
- A recorded config (run.yaml, the manifest) re-validates without refusing its own defaults: a hosted or served
  encoder's or reranker's `concurrency` equal to its default no longer fails `run start`, a resume or `run status`
  with `drop concurrency`. The one-at-a-time check compares the value against the field's default (a full dump
  cannot preserve which fields the user set); an explicitly non-default `concurrency` on a provider that sends one
  request at a time is still refused.
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
- `tests/contract` snapshots and the exported schemas (`schemas/index.v1.json`, `schemas/run-config.v1.json`)
  regenerated for the transport, the fakes and the status map (new names and members, `Endpoint.base_url`
  widened, and the retrieval configs' `base_url` described per its type: one URL, required for the served
  ones, optional for the hosted ones); `tests/test_errors.py` now requires one *root* class per exit code,
  since the moved outage and refusal types are `ProviderError` subclasses and exit codes do not change.
- The release workflow publishes three packages, one GitHub environment each: the build job builds `rcp-ndcg`,
  `rcp-ndcg-core` and `rcp-ndcg-vllm` (the last from its own directory, outside the uv workspace), checks each
  version against the tag, `rcp-ndcg`'s exact `rcp-ndcg-core` pin and the constraints file against the lock, runs
  `twine check` on every file, and uploads one artifact per package; `publish-core` (environment `pypi-core`),
  `publish-rcp-ndcg` (`pypi`, after the core it pins exactly) and `publish-vllm` (`pypi-vllm`) publish by trusted
  publishing, and the GitHub release still attaches the constraints file. The one-time PyPI trusted-publisher
  registration for the three environments (`pypi`, `pypi-core`, `pypi-vllm`) is done; `AGENTS.md` "Releasing" and
  the workflow header describe it.

### Removed

- **`run resume --judge-urls` and its environment variable** (`RCP_NDCG_JUDGE_URLS` as the
  coordinator's input): pass `run resume --engine judge=url[,url]` instead (repeatable; the runtime overlay of
  the engines you started yourself). The single-engine `serve:` mapping (`serve: {image: ...}`) on a run
  config: `serve:` now maps roles to engines (`serve: {judge: {...}}`). The doctor's `--judge-url` flag is
  `--endpoint <url>`, which probes any role's endpoint.

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
export), `mcp` (serve, tools) and `doctor`. Every command except `mcp serve` takes `--json` and prints one
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
recorded steps. `calibration insert --plan --judgements STORE --out PLAN` plans an insertion's windows with the
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
`eval_score` (with `out`, `per_query`, `fields`), `eval_compare`, `eval_explain`, `calibration_show`, `run_list`,
`run_show`, `run_status`, `estimate`. Destructive: `run_cancel`. `run_start` starts a run and returns its directory
at once.
`rcp-ndcg mcp tools --call TOOL --args JSON` calls one tool from the shell.

**JSON Schemas** (`schemas/`, `rcp-ndcg schema export`): the configs `run-config` and `judge-config`; the artifacts
`judgement`, `judgement-store` (a store's `identity.json`), `calibration` (a calibration's `items.json`),
`calibration-coverage`, `calibration-identity`, `extension-record` (a line of a calibration's `extensions.jsonl`),
`index`, `run-manifest`, `eval-report` and `comparison`; the output
of every `--json` command (among them `cost-estimate` and `extension`), the `cli` envelope, the `commands` tree and
the `command-index`; and the `mcp-manifest`. Every artifact names its schema in its `schema` field, and every
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
