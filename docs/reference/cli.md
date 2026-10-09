# Command line

The `rcp-ndcg` command is a tree of nouns and verbs: `rcp-ndcg <noun> <verb>`. Every leaf command is a thin
adapter over one library function, and `rcp-ndcg <noun> <verb> --help` lists its flags.

```text
rcp-ndcg data         fetch       download a public suite or a Hugging Face dataset; `tiny`: copy the packaged example
                      inspect     queries, candidate pools, label and gain distributions, excluded ids
                      validate    check a dataset or suite subset (and rankings) against the scoring protocol
                      convert     between dataset formats (the rcp_ndcg.readers/writers entry points), into a layout load_dataset reads
                      formats     list the readers and writers
rcp-ndcg retrieval    index       build a sparse, dense or late-interaction index of a dataset's corpus
                      search      first-stage retrieval into a rankings file
                      rerank      rescore rankings with a cross-encoder, a served /rerank endpoint or a hosted API
                      fuse        reciprocal rank fusion of rankings
rcp-ndcg judge        tournament  Stage A: listwise windows into the judgement store
                      rubric      Stage B: C1-C5 windows into the judgement store
                      reparse     read a store's stored answers again with the current parser, into a new store
rcp-ndcg calibration  fit         judgements into a calibration (with or without the tournament; one or pooled judges)
                      score       score documents a calibration lacks, the items frozen
                      insert      insert documents into a tournament calibration, with an insertion anchor report (the refit check); --dry-run picks opponents
                      show        items, coverage, per-judge severity, diagnostics and provenance
rcp-ndcg eval         score       RCP-nDCG and qrel-nDCG of rankings under a protocol
                      compare     difference (B minus A), paired t-test, bootstrap interval, sign flips
                      explain     one query of a run or a saved report: each system's top k with theta, gain and
                                  pass probabilities, and the gaps split into selection and ordering
rcp-ndcg results      sinks       list the registered result sinks
                      export      one rcp-ndcg.result-record.v1 record per system x dataset x metric x cutoff, to a sink
rcp-ndcg run          start       run a config (a YAML file or a packaged config's name) end to end, here or on a runner
                      resume      continue a run directory; unchanged steps are skipped
                      status      every planned step with its status and judge-window progress, done, usage, jobs
                      logs        a run's log
                      cancel      cancel a run's jobs
                      list        the runs under a runs directory
                      show        one run's manifest and artifacts
rcp-ndcg schema       list | show NAME | export --out DIR
rcp-ndcg mcp          serve       the commands as MCP tools over stdio
rcp-ndcg doctor       [--endpoint URL] check the environment: versions, extras, credentials present, endpoint reachable
rcp-ndcg --version
```

`rcp-ndcg schema show commands --json` prints every command with its help, its flags (type, choices, default,
whether required or repeatable, help) and its output schema (`rcp-ndcg.command-index.v1`); `--full` prints the whole
click tree instead (`rcp-ndcg.commands.v1`).

`run start` takes a run config file or the name of one that ships with the package: `tiny` (the whole pipeline on
the tiny example with the offline judge), `rejudge_nfcorpus` and `nano_nfcorpus_gpt5`. Relative paths inside a run
config (the dataset, a rankings file, `evaluation.systems`, a judge or prompt file) are relative to the config file;
paths given with `--set` are relative to the working directory. `rcp-ndcg data fetch --dataset tiny --out DIR`
copies the example data.

A run scores two reference systems besides the ones it is given: `candidates`, the order of the pools the judge was
shown, and `judge`, the documents ranked by their calibrated abilities (RCP-nDCG 1 by construction).
`evaluation.systems` adds your own (`{name: rankings file}`; `<file>#<system>` picks one system of a file that holds
several, and a file of several systems without `#` adds each under its own name). `eval compare --run` compares
your systems and leaves the reference systems out unless `--include-reference`.

`rcp-ndcg retrieval index` and `search` read a retriever config (`--retriever`, e.g. `experiments/paper/retrieval/bm25s.yaml` in the repository).
A BM25 config's `stemmer:` names a Snowball language (`english`, `german`, ...) or `null`; stemming uses PyStemmer,
which installs with the package, and the stemmer is part of the index identity.

## Flags

| Flag | Meaning |
|---|---|
| `--json` | machine output on stdout (below); human text otherwise |
| `--set KEY=VALUE` | override one field of the command's config (dotted path); `VALUE` is a YAML literal (`5`, `true`, `[a, b]`, `{k: v}`); repeatable. On `run resume` the run keeps the change only if the resume succeeds. On `data convert` it is a reader option instead (plain `KEY=VALUE`, values coerced as `int`/`float`/`bool`/`null`). `--set judge.tokenizer=ID` names the judge's tokenizer (a Hugging Face repo id, optionally `@revision`, or a `tokenizer.json` path), in whose tokens text limits and estimates are counted |
| `--out PATH` | the output file or directory; on `results export` the sink's URI (`records.jsonl`, `records.parquet`) |
| `--dataset URI`, `--subset NAME`, `--revision REV` | a dataset (`hf://`, `suite:`, `mteb:`, `beir:`, `jsonl:`, ...), one of its subsets, a Hub revision |
| `--rankings PATH`, `--judgements DIR`, `--calibration DIR`, `--run DIR` | typed inputs |
| `--suite NAME` | a public suite: its data and its protocol (`nanobeir`, `bright`, `vidore`, `trecdl`) |
| `--protocol NAME` | override the protocol (`nanobeir`, `bright`, `vidore`, `trecdl`, `mteb`, `plain`) |
| `--judge fake\|PATH\|NAME`, `--judge-url URL`, `--judge-model ID` | a judge config, or an ad-hoc OpenAI-compatible endpoint. On `judge` commands `--judge-model` overrides the model of any `--judge`; on `run start`/`run resume` it is refused without `--judge-url` |
| `--engine ROLE=URL[,URL]` | `run resume`: point one role's model (`judge`, `encoder` or `reranker`) at the engine URLs instead of its config's `base_url`; repeatable, one role each. A runtime overlay: it never changes the run's recorded config ([serving](../concepts/runs.md#starting-the-engines-with-the-run)) |
| `--docs QUERY_ID:DOC_ID` | judge only these documents (re-annotation, insertion) |
| `--plan FILE` | `judge tournament`: ask exactly the windows of an insertion plan (`calibration insert --dry-run --out FILE`), with the `--out` store's schedule |
| `--k INT` | a cutoff; repeatable on `eval score` (several), one on `eval compare` and `eval explain` — where it is also the documents shown per system |
| `--system NAME` | `eval score` (and `eval explain --report`): score only these systems of the rankings file (repeatable); an unknown name is refused (exit 2) with the systems the file names. One system whose rankings match nothing of the dataset no longer has to stop the others. On `judge tournament`/`judge rubric` and `retrieval rerank` it is a single selector for a multi-system candidates file. On `results export` it picks the systems to export |
| `--baseline NAME` | `eval compare`: compare every system against this one (the system a comparison compares against) |
| `--per-query`, `--fields NAME` | `eval score --json`: add the per-query values (the text renderer prints them too); print only the named top-level fields (repeatable). The full report goes to `--out` |
| `--include-text` | `eval explain`: add the query and document texts |
| `--include-reference` | `eval compare --run` and `results export --run`: also include the run's reference systems `candidates` and `judge` |
| `--sink NAME` | `results export`: the sink to write to (`jsonl`, `parquet`, `null`, or an installed `rcp_ndcg.results` entry point); see `rcp-ndcg results sinks` |
| `--depth INT`, `--limit INT` | candidate depth; the first N — queries on `judge tournament\|rubric`, records on `data convert`, runs on `run list` |
| `--estimate` | print calls, tokens (input tokens exact with the judge's `tokenizer`, approximate without) and wall time; call no judge |
| `--dry-run` | print the plan; no side effects; refuses what the real command would refuse |
| `--force` | judge into a store of another identity; the old records are moved aside |
| `--strict` | `calibration fit`: refuse (exit 12) a query with invalid windows (more than 5% in a stage, or an adaptive one) instead of warning |
| `--runner NAME`, `--detach` | `local`, `slurm`, `kubernetes`, or an installed runner; return at once and follow with `run status` |
| `--mirror URI` | `run start`, `run resume`, `judge tournament\|rubric`: mirror the run directory or store to a bucket while it runs, and restore what is missing from it first ([durability](../concepts/runs.md#durability-local-runs-and-a-mirror)) |
| `-v`, `-vv`, `-q`, `--log-file PATH` | verbosity on stderr, and an optional log file (before the command) |
| `--env-file PATH` | load environment variables from a file; never implicit (before the command) |

Environment variables:

| Variable | Meaning |
|---|---|
| `RCP_NDCG_CACHE_DIR` | the package cache (indexes, downloaded media); default the user cache directory |
| `RCP_NDCG_MEDIA_CACHE` | where resolved media is cached; default `media/` under the package cache |
| `RCP_NDCG_RUNS_DIR` | where runs are created when no `--runs-dir` names a directory |
| `RCP_NDCG_LOG_LEVEL` | the log level when no `-v` or `-q` is given |
| `RCP_NDCG_MAX_VIDEO_BYTES` | the largest video inlined into a judge request (default 64 MiB) |
| `RCP_NDCG_IMAGE_CACHE_SIZE`, `RCP_NDCG_VIDEO_CACHE_SIZE` | encoded images and videos kept in memory per worker while judging |
| `RCP_NDCG_ENGINES` | the engine overlay a job's phase hands its coordinator: JSON `{"<role>": {"urls": [...], "wait_on_outage_s": 900}, ...}`; applied at run time only, never recorded ([runs](../concepts/runs.md#starting-the-engines-with-the-run)) |

## Credentials

Credentials are read only under the name a config declares (`api_key_env`), and a profile's default key
variables apply **only when the request goes to the profile's own default host**. Any other `base_url` receives
a key only from an explicit `api_key_env`. When a hosted retrieval profile (`api: cohere`, `voyage`, `gemini`)
names none and the request goes to the profile's own URL, its adapter reads the vendor's usual variables:
`CO_API_KEY` or `COHERE_API_KEY`, `VOYAGE_API_KEY`, and `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

## Machine output

With `--json`, stdout carries exactly one JSON document, and logs and progress go to stderr (`--help` and
`--version` are the exception: they print plain text and exit 0):

```json
{"schema": "rcp-ndcg.cli.v1", "command": "eval score", "ok": true,
 "data": {"schema": "rcp-ndcg.eval-score.v1", "summary": [], "per_dataset": [],
          "warnings": [{"code": "UNRANKED_QUERIES", "message": "..."}]},
 "warnings": [],
 "meta": {"version": "0.0.1", "elapsed_s": 1.84, "run_dir": null}}
```

The envelope's `warnings` carry the conditions raised while the command ran, with a code from
`rcp_ndcg.errors.WarningCode` (`APPROXIMATE_IMAGE_TOKENS`, `BT_L2_MISMATCH`, `CARD_UNCACHED`, `INVALID_WINDOWS`,
`SNAPSHOT_LISTING`, `UNCALIBRATED_DOCUMENTS`, `UNPINNED_REVISION`, `UNREADABLE_RUN`); without `--json` the same
warnings print on stderr. A result can carry warnings of its own: an evaluation report's `data.warnings` also use
`UNRANKED_QUERIES` and `NO_POSITIVE_QRELS`. `UNPINNED_REVISION` is the one a Hub dataset raises when its branch
(or no revision at all) resolved to no commit -- offline, or with the Hub unreachable, and no recorded ref in the
local cache; it names `--revision <full sha>` as the fix. `SNAPSHOT_LISTING` is the one an offline corpus read
raises when the file listing came from the local snapshot instead of the Hub: the snapshot holds only the files a
download left, so a partial cache can read as missing data. `CARD_UNCACHED` is the one an offline Hub load raises
when the dataset card is not in the cache either: the reader falls back to the plain `{subset}/` path layout and
says so -- the card-declared configs, and any table they alone name, are unavailable until one online run caches
the card.

A failure has `"ok": false` and an `error` object with `code`, `exit_code`, `message`, `hint`, `retryable` and
`details`. A config that does not validate (exit 3) lists its problems in `details.errors`: per problem the `field`
(dotted path), the given `input`, the `problem`, the `expected` type or values, a `did_you_mean` for an unknown key,
and the `source`: `--set` when an override set the field, else the config file. `--set` values are YAML literals
(`--set k=10`, `--set k=null`, `--set 'k=[1, 2]'`).

`rcp-ndcg schema show commands` describes the whole command tree as JSON, and `rcp-ndcg schema list`
lists every exported schema. A schema's `$id` is an identifier naming its committed copy under `schemas/`, not a URL
to fetch; `rcp-ndcg schema show <name>` prints the schema.

## Exit codes

| Code | Name | Meaning; what an agent should do |
|---|---|---|
| 0 | `SUCCESS` | done (including a no-op resume) |
| 1 | `INTERNAL` | a bug; report it |
| 2 | `USAGE` | bad flags or arguments; fix the invocation |
| 3 | `CONFIG` | invalid config value; fix the YAML or `--set` |
| 4 | `MISSING_INPUT` | a file, run or dataset is absent; the message or hint names it |
| 5 | `CREDENTIALS` | missing or rejected credentials; the message or hint names the variable, never its value — a backend that rejects access below the transport layer may still surface as `INTERNAL`; treat a `Forbidden`/`denied` message as credentials |
| 6 | `PROVIDER` | an endpoint or a scheduler failed after its retries (unreachable, timing out, rate limiting, an empty answer); resume later when `retryable` is true (it is false for a route or model the endpoint does not have, HTTP 404) |
| 8 | `CAPABILITY` | the judge or endpoint cannot take what a request carries: an answer schema it refuses (serve with the reasoning parser, or set `decoding: free`), images or videos beyond its `max_images` / `max_videos`, a window whose media exceed its context, media for a text-only encoder; raised by the first such request |
| 9 | `INTERRUPTED` | SIGINT or SIGTERM stopped the command; the state on disk is consistent; resume |
| 10 | `DEPENDENCY` | a missing extra; the hint is the exact install command |
| 11 | `IDENTITY` | refusing to mix: resume with a changed config, judgements from another family, an insertion whose anchor check failed (`data.extension.anchor_report`); the hint names the differing fields and the way out (a new output directory or run; `--force` where the command has it) |
| 12 | `DATA` | input that would produce wrong numbers or does not parse: malformed or non-finite rankings, qrels or gains, gains that match no labelled query, ids that do not join, a document over its text cap with `on_overflow: fail`, a new document the evidence cannot identify, a query with invalid windows under `--strict`, a damaged mirror |

Code 7 is retired: no command returns it, and it is not reused.
