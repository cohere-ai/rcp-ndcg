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

- `schemas/run-config.v1.json`: the `CandidatesConfig` description states that the whole section is content for
  the step identities (its `IDENTITY_ROLES` declarations); no property changed.
- **New public module `rcp_ndcg.inference`**: the inference layer between `rcp_ndcg.data` and
  `rcp_ndcg.retrieval`, with the frozen interfaces the transport, the adapters and the role clients build on. No
  transport behaviour yet: every behavioural method raises `NotImplementedError`, naming the lane that owns it.
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
    "rcp_ndcg.adapters"`). No adapter is registered yet.
  - `inference.transport`: the `Sender` protocol and the `Transport` class -- the transport's frozen interface
    only (`send`, `probe`, `run`, `aclose`); its routing, retries, parking and status-map behaviour is the transport lane's.
  - `inference.fake`: `FAKE_SCHEME = "fake://"` and the offline fakes' contract; no implementation yet.
  - `inference.config`: the role endpoint configs `EmbeddingEndpoint` (`api` default `openai_embeddings`),
    `PoolingEndpoint` (default `vllm_pooling`, with `embed_dtype: float16` by default, `float32` opt-in) and
    `RerankEndpoint` (default `rerank`, `instruction: fold` default, a `batch_size` refused for a `listwise`
    model). Not wired into `rcp_ndcg.retrieval.config` yet
- **`rcp_ndcg.errors` gains `BackendUnavailableError` and `RequestRejectedError`**, moved unchanged from
  `rcp_ndcg.llm.client` (still importable and exported there). Exit codes do not change: both remain
  `ProviderError` subclasses at `PROVIDER`, `RequestRejectedError` non-retryable.
- **`rcp_ndcg.support.serve` gains the serve-by-role types**: `EngineRole`, `EngineConfig` (an alias of the
  unchanged `ServeConfig`), `ServeByRole`, `Phase`, `ENGINES_ENV = "RCP_NDCG_ENGINES"`, `EngineURLs`,
  `parse_engines_env`, and the frozen `plan_phases(steps, serve, uses)` signature (behaviour arrives with the serve-phases work).
- **`rcp_ndcg.llm.client` gains `api` and `headers_env`** through `Endpoint`; `wait_on_outage_s` moves up to
  `Endpoint` and the judge keeps declaring it only through that inheritance. A judge's identity payload is
  unchanged: `api` defaults to `None` (omitted from identities until a role config sets it), the other two are
  runtime fields.
- New layering charter (`AGENTS.md`): `data → inference → retrieval`; enforced by the new
  `tests/test_layering.py` (eager imports only; the current tree has no outward import).

### Fixed

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

### Changed

- `tests/contract` snapshots and the exported schemas (`schemas/index.v1.json`, `schemas/judge-config.v1.json`,
  `schemas/run-config.v1.json`) regenerated for the moved and new fields; `tests/test_errors.py` now requires
  one *root* class per exit code, since the moved outage and refusal types are `ProviderError` subclasses and
  exit codes do not change.

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
