# Command line

The `rcp-ndcg` command is a tree of nouns and verbs: `rcp-ndcg <noun> <verb>`. Every leaf command is a small adapter
over one library function. `rcp-ndcg <noun> <verb> --help` lists the flags of the command.

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
                      store       encode a corpus's and its queries' vectors once at the checkpoint's full width
                      sweep       apply the declared MRL head per output dimension and score every cut
rcp-ndcg judge        tournament  Stage A: listwise windows into the judgement store
                      rubric      Stage B: C1-C5 windows into the judgement store
                      check       probe an endpoint: schema accepted, answer parses, reasoning separated
                      reparse     read a store's stored answers again with the current parser, into a new store
rcp-ndcg calibration  fit         judgements into a calibration (with or without the tournament; one or pooled judges)
                      score       score documents a calibration lacks, the items frozen
                      insert      insert documents into a tournament calibration, with an insertion anchor report (the refit check); --dry-run picks opponents
                      show        items, coverage, per-judge severity, diagnostics and provenance
rcp-ndcg eval         score       RCP-nDCG, qrel-nDCG and Count-nDCG of rankings under a protocol
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
rcp-ndcg docs         api         one Markdown page per public module, from the pinned surface snapshot
rcp-ndcg mcp          serve       the commands as MCP tools over stdio
rcp-ndcg doctor       [--endpoint URL] check the environment: versions, extras, credentials present, endpoint reachable
rcp-ndcg --version
```

`rcp-ndcg schema show commands --json` prints every command with its help. It prints the flags of the command (type,
choices, default, whether required or repeatable, help) and its output schema (`rcp-ndcg.command-index.v1`). `--full`
prints the whole click tree instead (`rcp-ndcg.commands.v1`).

`run start` takes a run config file, or the name of one that ships with the package. The names are `tiny` (the whole
pipeline on the tiny example with the offline judge), `rejudge_nfcorpus` and `nano_nfcorpus_gpt5`. A relative path
inside a run config is relative to the config file. Such a path can name the dataset, a rankings file,
`evaluation.systems`, or a judge or prompt file. A path given with `--set` is relative to the working directory.
`rcp-ndcg data fetch --dataset tiny --out DIR` copies the example data.

A run scores two reference systems besides the systems that it is given. `candidates` is the order of the pools that
the judge was shown. `judge` is the documents ranked by their calibrated abilities (RCP-nDCG 1 by construction).
`evaluation.systems` adds your own systems (`{name: rankings file}`). `<file>#<system>` picks one system of a file
that holds several. A file of several systems without `#` adds each system under its own name. `eval compare --run`
compares your systems, and it leaves the reference systems out unless `--include-reference`.

`rcp-ndcg retrieval index` and `search` read a retriever config (`--retriever`, e.g.
`experiments/paper/retrieval/bm25s.yaml` in the repository). A `stemmer:` of a BM25 config names a Snowball language
(`english`, `german`, ...) or `null`. Stemming uses PyStemmer, which installs with the package, and the stemmer is
part of the identity of the index. A served encoder has a `base_url`, which is the form that its wire speaks. The form
is `.../v1` for `api: openai_embeddings`, and the root of the engine for the pooling wire (`api: vllm_pooling`). The
command refuses the other form where it reads the config.

## Flags

| Flag | Meaning |
|---|---|
| `--json` | machine output on stdout (below); human text otherwise |
| `--set KEY=VALUE` | override one field of the config of the command (dotted path). `VALUE` is a YAML literal (`5`, `true`, `[a, b]`, `{k: v}`), and the flag is repeatable. On `run resume` the run keeps the change only if the resume succeeds. On `data convert` it is a reader option instead (plain `KEY=VALUE`, with values coerced as `int`/`float`/`bool`/`null`). `--set judge.tokenizer=ID` names the tokenizer of the judge: a Hugging Face repo id, optionally with `@revision`, or a `tokenizer.json` path. The text limits and the estimates are counted in its tokens |
| `--out PATH` | the output file or directory; on `results export` the sink's URI (`records.jsonl`, `records.parquet`) |
| `--dataset URI`, `--subset NAME`, `--revision REV` | a dataset (`hf://`, `suite:`, `mteb:`, `beir:`, `jsonl:`, ...), one of its subsets, a Hub revision |
| `--rankings PATH`, `--judgements DIR`, `--calibration DIR`, `--run DIR` | typed inputs. On `eval score`, `--judgements` (repeatable) is the rubric store that the Count-nDCG gains come from, and it is required with `--metrics count_ndcg` |
| `--suite NAME` | a public suite: its data and its protocol (`nanobeir`, `bright`, `vidore`, `trecdl`) |
| `--protocol NAME` | override the protocol (`nanobeir`, `bright`, `vidore`, `trecdl`, `mteb`, `plain`) |
| `--judge fake\|PATH\|RECIPE\|NAME`, `--judge-url URL`, `--judge-model ID` | a judge: a shipped judge recipe, a judge config YAML, a shipped vendor profile (`gpt5_hosted`), or `fake`. A shipped judge recipe is `recipe:<id>` or a bare recipe id. It can also be an ad-hoc OpenAI-compatible endpoint with `--judge-url`/`--judge-model`. On `judge` commands, `--judge-model` overrides the model of any `--judge`. On `run start`/`run resume` the command refuses it without `--judge-url` |
| `--engine ROLE=URL[,URL]` | `run resume`: point the model of one role at the engine URLs instead of the `base_url` of its config. The role is `judge`, `encoder` or `reranker`. The flag is repeatable, with one role each time. It is a runtime overlay: it never changes the recorded config of the run ([serving](../concepts/runs.md#starting-the-engines-with-the-run)) |
| `--docs QUERY_ID:DOC_ID` | judge only these documents (re-annotation, insertion) |
| `--seed INT` | the bootstrap seed (`eval score`, `eval compare`, `retrieval sweep`; default 0). It is also the seed of the window schedule and of the offline judge. That default is the own seed of the schedule, 42 (`judge tournament`, `judge rubric`) |
| `--plan FILE` | `judge tournament`: ask exactly the windows of an insertion plan (`calibration insert --dry-run --out FILE`), with the schedule of the `--out` store |
| `--k INT` | a cutoff. It is repeatable on `eval score` (several cutoffs), and it is one cutoff on `eval compare` and `eval explain`. On `eval explain` it is also the number of documents shown for each system |
| `--system NAME` | `eval score` (and `eval explain --report`): score only these systems of the rankings file (repeatable). The command refuses an unknown name (exit 2), and it names the systems of the file. One system whose rankings match nothing of the dataset no longer has to stop the others. On `judge tournament`/`judge rubric` and `retrieval rerank` it is a single selector for a candidates file of several systems. On `results export` it picks the systems to export |
| `--baseline NAME` | `eval compare`: compare every system against this one. This is the system that a comparison compares against |
| `--per-query`, `--fields NAME` | `eval score --json`: add the per-query values; the text renderer prints them too. Print only the named top-level fields (repeatable). The full report goes to `--out` |
| `--include-text` | `eval explain`: add the query and document texts |
| `--include-reference` | `eval compare --run` and `results export --run`: also include the run's reference systems `candidates` and `judge` |
| `--sink NAME` | `results export`: the sink to write to (`jsonl`, `parquet`, `null`, or an installed `rcp_ndcg.results` entry point); see `rcp-ndcg results sinks` |
| `--depth INT`, `--limit INT` | candidate depth: the first N. N is a count of queries on `judge tournament\|rubric`, of records on `data convert`, or of runs on `run list` |
| `--rrf-k INT` | `run start`: the RRF constant k of the fusion, in 1 / (k + rank); it sets `candidates.fuse.rrf_k`. The systems to fuse stay those of the config ([the fuse step](../concepts/runs.md#runs-and-job-runners)) |
| `--estimate` | print the calls, the tokens and the wall time (a floor). The input tokens are exact with the `tokenizer` of the judge, and approximate without it. The command calls no judge. It refuses (exit 8) a window that carries more images than the `max_images` of the judge |
| `--dry-run` | print the plan. `calibration insert --dry-run --out` writes the plan file, and `run resume --dry-run` restores from `--mirror` (or the mirror of the run) first. Otherwise the command has no side effects. It refuses what the real command would refuse |
| `--force` | judge into a store of another identity; the old records are moved aside |
| `--strict` | `calibration fit`: refuse (exit 12) a query with invalid windows (more than 5% in a stage, or an adaptive window). It also refuses a document whose tournament windows are all invalid. The command refuses instead of warning |
| `--runner NAME`, `--detach` | `local`, `slurm`, `kubernetes`, or an installed runner. The command returns at once, and `run status` follows the run |
| `--mirror URI` | `run start`, `run resume`, `judge tournament\|rubric`: mirror the run directory or the store to a bucket while it runs. Restore what is missing from the mirror first ([durability](../concepts/runs.md#durability-local-runs-and-a-mirror)) |
| `--mirror-interval SECONDS` | `judge tournament\|rubric`: seconds between two mirror uploads (default 60), the run config's `mirror_interval_s` |
| `-v`, `-vv`, `-q`, `--log-file PATH` | verbosity on stderr, and an optional log file (before the command) |
| `--env-file PATH` | load environment variables from a file; never implicit (before the command) |

Environment variables:

| Variable | Meaning |
|---|---|
| `RCP_NDCG_CACHE_DIR` | the package cache (indexes, downloaded media); default the user cache directory |
| `RCP_NDCG_MEDIA_CACHE` | where resolved media is cached; default `media/` under the package cache |
| `RCP_NDCG_RUNS_DIR` | where runs are created when no `--runs-dir` names a directory |
| `RCP_NDCG_LOG_LEVEL` | the log level when no `-v` or `-q` is given |
| `RCP_NDCG_MAX_VIDEO_BYTES` | the largest video that is inlined into a judge request, as an integer number of bytes (default 64 MiB = 67108864). A malformed value is refused with a typed error at the guard, not at import |
| `RCP_NDCG_IMAGE_CACHE_SIZE`, `RCP_NDCG_VIDEO_CACHE_SIZE` | the prepared images that are kept in memory for each worker while media is prepared. It also holds the encoded video containers while a container is inlined into a request |
| `RCP_NDCG_ENGINES` | the engine overlay that the phase of a job hands to its coordinator: JSON `{"<role>": {"urls": [...], "wait_on_outage_s": 900}, ...}`. The overlay applies at run time only, and the run never records it. A job's `runner.options.env` cannot set it, because the phase overlay owns it ([runs](../concepts/runs.md#starting-the-engines-with-the-run)) |

## Credentials

A config declares a name with `api_key_env`, and the code reads credentials only under that name. The default key
variables of a profile apply **only when the request goes to the default host of the profile**. Any other `base_url`
receives a key only from an explicit `api_key_env`. A hosted retrieval profile (`api: cohere`, `voyage`, `gemini`) can
name none. Its adapter then reads the usual variables of the vendor when the request goes to the URL of the profile.
The variables are `CO_API_KEY` or `COHERE_API_KEY`, `VOYAGE_API_KEY`, and `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

## Machine output

With `--json`, stdout carries exactly one JSON document, and the logs and the progress go to stderr. `--help` and
`--version` are the exception: they print plain text and exit 0.

```json
{"schema": "rcp-ndcg.cli.v1", "command": "eval score", "ok": true,
 "data": {"schema": "rcp-ndcg.eval-score.v1", "summary": [], "per_dataset": [],
          "warnings": [{"code": "UNRANKED_QUERIES", "message": "..."}]},
 "warnings": [],
 "meta": {"version": "0.0.1", "elapsed_s": 1.84, "run_dir": null}}
```

The `warnings` of the envelope carry the conditions that were raised while the command ran. Each warning has a code
from `rcp_ndcg.errors.WarningCode` (`APPROXIMATE_IMAGE_TOKENS`, `BT_L2_MISMATCH`, `CARD_UNCACHED`, `INVALID_WINDOWS`,
`NO_VALID_TOURNAMENT_EVIDENCE`, `SNAPSHOT_LISTING`, `UNCALIBRATED_DOCUMENTS`, `UNPINNED_REVISION`, `UNREADABLE_RUN`).
Without `--json` the same warnings print on stderr. A result can carry warnings of its own: the `data.warnings` of an
evaluation report also use `UNRANKED_QUERIES` and `NO_POSITIVE_QRELS`. `UNPINNED_REVISION` is the code that a Hub
dataset raises when its branch resolved to no commit. The same happens when the request gives no revision at all. The
causes are an offline run or an unreachable Hub, with no recorded ref in the local cache. The warning names
`--revision <full sha>` as the fix. `SNAPSHOT_LISTING` is the code that an offline corpus read raises. It applies when
the file listing came from the local snapshot instead of the Hub. The snapshot holds only the files that a download
left, so a partial cache can read as missing data. `CARD_UNCACHED` is the code that an offline Hub load raises when
the dataset card is not in the cache either. The reader then falls back to the plain `{subset}/` path layout, and it
says so. The card-declared configs, and any table that they alone name, are unavailable until one online run caches
the card.

A failure has `"ok": false` and an `error` object with `code`, `exit_code`, `message`, `hint`, `retryable` and
`details`. A config that does not validate (exit 3) lists its problems in `details.errors`. For each problem it gives
the `field` (dotted path), the given `input`, the `problem` and the `expected` type or values. It gives a
`did_you_mean` for an unknown key, and the `source`. The source is `--set` when an override set the field, else the
config file. `--set` values are YAML literals (`--set k=10`, `--set k=null`, `--set 'k=[1, 2]'`).

`rcp-ndcg schema show commands` describes the whole command tree as JSON, and `rcp-ndcg schema list` lists every
exported schema. The `$id` of a schema is an identifier that names its committed copy under `schemas/`. It is not a
URL to fetch. `rcp-ndcg schema show <name>` prints the schema.

## Exit codes

| Code | Name | Meaning; what an agent should do |
|---|---|---|
| 0 | `SUCCESS` | done, including a no-op resume |
| 1 | `INTERNAL` | a bug; report it |
| 2 | `USAGE` | bad flags or arguments; fix the invocation |
| 3 | `CONFIG` | invalid config value; fix the YAML or `--set` |
| 4 | `MISSING_INPUT` | a file, run or dataset is absent; the message or hint names it |
| 5 | `CREDENTIALS` | missing or rejected credentials; the message or hint names the variable, never its value. A backend that rejects access below the transport layer may still surface as `INTERNAL`; treat a `Forbidden`/`denied` message as credentials |
| 6 | `PROVIDER` | an endpoint or a scheduler failed after its retries (unreachable, timing out, rate limiting, an empty answer). Resume later when `retryable` is true. It is false for a route or a model that the endpoint does not have, HTTP 404 |
| 8 | `CAPABILITY` | the judge or the endpoint cannot take what a request carries. A request can carry an answer schema that it refuses (serve with the reasoning parser, or set `decoding: free`). It can carry images or videos beyond its `max_images` / `max_videos`, or a window whose media exceed its context. It can also carry media for a text-only encoder. The first such request raises it |
| 9 | `INTERRUPTED` | SIGINT or SIGTERM stopped the command, or a run step exceeded its `step_budget_s`. The state on disk is consistent; resume |
| 10 | `DEPENDENCY` | a missing extra; the hint is the exact install command |
| 11 | `IDENTITY` | refusing to mix: resume with a changed config, judgements from another family, an insertion whose anchor check failed. The report is in `error.details.anchor_report`: `ok`, the ability and gain shifts, the tolerance and `documents_checked`. A failure envelope carries no `data`. The hint names the way out (a new output directory or run; `--force` where the command has it) |
| 12 | `DATA` | input that would produce wrong numbers, or input that does not parse. The input can be malformed or non-finite rankings, qrels or gains. It can hold gains that match no labelled query, or ids that do not join. It can hold a document over its text cap with `on_overflow: fail`, or a new document that the evidence cannot identify. It can also hold a query with invalid windows under `--strict`, or a damaged mirror |

Code 7 is retired: no command returns it, and it is not reused.
