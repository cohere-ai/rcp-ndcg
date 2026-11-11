---
name: rcp
description: Evaluate retrievers, rerankers or embedders with calibrated relevance (RCP-nDCG) instead of sparse binary qrels. Re-judge a candidate pool with an LLM judge. Extend a pool without a new fit. Use this skill to measure a retrieval system. Use it to compare two systems. Use it to check nDCG numbers that look misleading, or to make relevance labels without human annotators.
---

# RCP-nDCG

RCP-nDCG is nDCG with calibrated gains. Each gain is a relevance probability from an LLM judge. A listwise tournament orders the
pool of each query. A rubric of five binary criteria (C1 to C5) anchors the scale. A two-parameter (2PL) item-response model puts
all queries on one scale. Released datasets contain these gains, so scoring a system on them needs no LLM.

Install (Python 3.12): `pip install "rcp[hf,calibrate]" --extra-index-url https://download.pytorch.org/whl/cpu`, or run it
without an install: `uvx rcp`. Every command except `mcp serve` takes `--json` and then prints exactly one JSON document on
stdout: `{"schema", "command", "ok", "data"|"error", "warnings", "meta"}`. `mcp serve` is a stdio server. `--help` and `--version`
print plain text. Read `data` and **check `warnings`**. Each `{code, message}` records a condition raised while the command ran:
invalid windows, uncalibrated documents, an unpinned revision. Recorded cuts are in the text census beside the judgement store. On
failure, read `error` (`code`, `exit_code`, `message`, `hint`, `retryable`, `details`). Do not read the human text.
`rcp schema show commands --json` lists every command with its flags (type, default, help) and its output schema. A config
error (exit 3) lists `error.details.errors` for each **field** problem (`field`, `input`, `expected` **or** `did_you_mean`,
`source`). A config error outside the field path has empty `details`. Read `error.message` too.

## Four paths

**1. Score a system on the released data (no LLM).** The rankings file holds `query_id`, `doc_id` and `score`. The
format is Parquet, CSV, a TREC run or JSONL. It can also hold `system` and `dataset`. The table formats' columns accept the common aliases
(`query_id`/`query-id`/`qid`/`query`, `doc_id`/`corpus-id`/`corpus_id`/`docid`/`docno`, `score`/`rerank_score`/`sim`,
`system`/`model`/`run`/`tag`/`run_id`, `dataset`/`subset`). A JSONL row uses the canonical keys (`query_id`, `doc_id`, `score`; or
a `scores` / `doc_ids` row). The `dataset` column names the subset of each row, when the file ranks several subsets of a suite.
Those subsets share query ids. Without `dataset`, the command refuses the file. A TREC run holds one subset. Add
`--subset <name>`. No rankings file yet? Path 2 produces one from a model.

```bash
rcp eval score --rankings my_system.parquet --suite nanobeir --json
```

Read `data.summary`: one row per system and metric (`rcp_ndcg`, `qrel_ndcg`) with `value`, `ci_low`, `ci_high`. `data.per_dataset`
holds the mean per dataset. `--metrics count_ndcg` adds the rubric-only baseline. Its gains come from the rubric windows, so pass
their store with `--judgements <store>` (repeatable). The per-query values stay out of stdout. `--per-query` adds them,
`--fields summary` keeps only the named fields, and `--out report.json` writes the full report. `eval compare --report` and
`eval explain --report` read that report. The suites are `nanobeir`, `bright`, `vidore` and `trecdl`. Python:
`rcp.evaluate(rcp.load_rankings(path), suite="nanobeir")`. The command refuses a system that matches nothing in the
dataset (exit 12). Score the others with `--system NAME` (repeatable; `systems=[...]` in Python).

**2. Serve an open model and score it (needs a served engine).** On the engine side, use the stock `vllm/vllm-openai` image.
Install `python3 -m pip install --no-deps rcp-vllm` there. That is the one change that the engine environment takes. Then run
`rcp-vllm serve <recipe-id> --port 8000`. Add `--dry-run` to print the engine argv and exit. On the client side, install
`rcp-vllm` beside `rcp`. Write a retriever (or reranker) config that names `recipe: <id>` and sets `base_url`. The
recipe supplies the client block. The run-time fields stay yours: an explicit content field that disagrees with the recipe is
refused. `rcp retrieval index|search|rerank|fuse` produce and re-score rankings files. `rcp eval score` then reads them.
`schema show commands --json` lists the flags. `--retriever recipe:<id>` and `--reranker recipe:<id>` are shorthands. Their URL
comes from `--set ...base_url=...` or a `serve:` engine. Each recipe declares its budgets. The client cuts over-budget content at
token boundaries, keeps the template anchors and records every cut. The engine never cuts. A missing `rcp_vllm` is a typed
refusal (exit 10): its hint is `pip install rcp-vllm`. See [recipes and serving models](../../docs/reference/recipes.md) for
the 24 recipe families and their 44 variants.

**3. Re-judge a pool with an OpenAI-compatible endpoint (calls the judge).** Ask the user for the model tokenizer.
It is a Hugging Face repo id, or a `tokenizer.json` path. `--set judge.tokenizer=<id>` makes the text limits and the estimates count the judge's
own tokens. Without it, the command sends documents whole. A text policy that cuts (`truncate`, `chunk`, `fail`) then exits 3. The
judge's default text policy is `keep`, so documents run whole and nothing is cut. A declared `truncate`/`fail` policy caps at 32
768 tokens by default, and the census records every such cut. `--set judge.tokenizer=...` makes the counts exact.

```bash
rcp run start rejudge_nfcorpus --judge-url http://localhost:8000/v1 --judge-model my-model --estimate --json
rcp run start rejudge_nfcorpus --judge-url http://localhost:8000/v1 --judge-model my-model --detach --json
rcp run status --run runs/<run_id> --json
```

Read `data.estimate` of the first (`calls`, `input_tokens`, `output_tokens`, `wall_s`, `assumptions`) and `data.run_dir` of the
second. Read `data.status`, `data.steps` and `data.metrics` (`<system>/<metric>@<k>`) of the third. The full report with
confidence intervals is `<run_dir>/metrics/report.json`. `rejudge_nfcorpus` is a run config that comes with the package.
`run start` takes a packaged config name or a YAML path. Paths inside a config are relative to the config file. `data.steps` lists
every planned step in run order, each `pending` until it starts. Each judging step has `progress: {done, planned}` in judge
windows. Poll `run status` until `data.done` is true; `data.status` is then `completed`, `partial`, `failed` or `cancelled`.
`run start --only` records just those steps as the plan, and the run then ends `completed`. The command refuses
`run resume --only` names for that run, because they are not steps of this run. A job that was interrupted leaves a run with more
planned steps. On such a run, `run resume --run <dir> --only <step>` runs just those steps now. The run ends `partial` with exit
0. A later `run resume --run <dir>` runs the rest.

**4. Reproduce a table of the paper** (from a checkout, no LLM): `python experiments/fetch_data.py`, then
`python experiments/run_all.py`. Exit code 0 means that every value matched the paper within its tolerance.

## Invariants

- **Estimate before judging.** Run `--estimate` first. Show the user `calls`, the request range (`requests_min`..`requests_max`,
  retries included), the tokens and `wall_s`. `wall_s` is a lower bound, not an expectation. The command refuses a window over
  `max_images`. Then run. Input counts are exact with a tokenizer and approximate otherwise (2 characters per token).
  `input_token_count` says which.
- **Never pool judgements across families.** A family is the judge model, its revision and its tokenizer. It is also
  the prompt, the criteria and the parse version. It also holds the preprocessing and the document-reading rule (title join, text-formatting version).
  `rcp calibration show --calibration DIR --json` lists them in `data.families`. Exit code 11 refuses a mix.
- **Never compare numbers from different protocols, cutoffs or tie rules.** `data.protocol` of an eval report names the protocol.
  State the suite, protocol, `k` and judge next to every number that you report.
- **Do not edit the prompts.** The rubric has exactly five criteria, C1 to C5. A changed prompt is a new family.
- **The judge's own order is not a system.** A run scores two reference systems beside yours. `candidates` is the pool order the
  judge was shown. The calibrated abilities rank `judge`. Its RCP-nDCG is 1 by construction. `eval compare --run` leaves both out
  unless you pass `--include-reference`. Add your own systems to a run with `evaluation.systems` (`{name: rankings file}`).
  `<file>#<system>` picks one system of a file that holds several.
- **Report only what was computed.** A qrel-nDCG of `null` means that the query has no positive grade. Pass it through.
- Treat exit codes 11 and 12 as refusals to fix, not as errors to retry. The hint says how to fix them (table: [the command-line
  reference](../../docs/reference/cli.md#exit-codes)).

## Reading results

- *Is A better than B?* Run `rcp eval compare --run DIR --json`, or pass `--report FILE` from `eval score --out`. Read
  `data.pairs[]`: `delta` (B minus A), `ci_low`, `ci_high`, `p_value`, `significant`, and `sign_flips` (queries where RCP-nDCG and
  qrel-nDCG disagree).
- *Can you trust this calibration?* Run `rcp calibration show --calibration DIR --json`. Read `data.coverage` (queries
  without calibration, invalid windows, degenerate documents) and `data.diagnostics` (ECE, Brier score). Read `data.warnings` too:
  `INVALID_WINDOWS` lists queries that rest on fewer windows than planned. A poor fit means that the run needs more judgements,
  not that RCP-nDCG is wrong.
- *Why?* Run `rcp eval explain --run DIR --query-id QID --json`, or pass `--report FILE`. It gives the top k of each system.
  Each document shows its theta (its calibrated relevance estimate), gain and per-criterion pass probabilities. The criteria's
  `gamma` and `beta` are in `data.items` once. `data.deltas` gives the gap to the first system at cutoff `data.k`, split into
  `selection` and `ordering`. The report leaves out texts unless you pass `--include-text`.
- *Do you need to publish the numbers?* `rcp results export --run DIR --sink jsonl --out records.jsonl` writes
  `rcp.result-record.v1` records. `results sinks` lists the sinks.

## Recipes and runs

Score documents that a calibration lacks, without a new fit. The document was added to a pool after the fit, or it has no theta
because its windows failed. The items stay frozen. The judgement store is append-only, so the command asks only the new windows. A
document that the calibration already holds keeps its theta. The command lists it as skipped. Judge only the new documents, then
score.

```bash
rcp judge rubric --dataset <uri> --judge <judge> --docs q2:d10 --docs q2:d11 --out <store> --estimate --json
rcp calibration score --calibration <calibration> --judgements <store> --out <extended> --json
```

Insert a new document into a tournament calibration. Plan its windows. The insertion anchor report is the refit check on the
published gains. Judge exactly the planned windows, then insert. If you insert anything else, the command exits 11.

```bash
rcp calibration insert --calibration <calibration> --judgements <store> --dry-run --query q1 --doc new-doc --n 36 --out plan.json --json
rcp judge tournament --dataset <uri> --judge <judge> --plan plan.json --out <store> --estimate --json
rcp calibration insert --calibration <calibration> --judgements <store> --out <extended> --json
```

A plan cannot hold more opponents than the query has other documents. `data.plan.capped` is true when `--n` asked for more.
`data.plan.opponents` says how many the plan holds. On a small pool, only a larger `--se-target` helps. A second judge that
answered the same rubric pools with one severity per judge. Run
`rcp calibration fit --judgements <store> --judgements <second-store> --judges pooled --out <pooled> --json`. The severities
are in `data.judge_severity`.

Make a long judge pass survive a preemption (a job that the scheduler stops). Mirror it to any fsspec URI. S3 and Azure work
through the `s3`/`azure` extras. GCS works as installed, because `gcsfs` comes with `rcp`. Any fsspec filesystem that you
register works too. Follow `data.mirror.lag_s` in `run status`, and resume on any node from the mirror.

```bash
rcp run start <config> --mirror s3://bucket/runs/nano --detach --json
rcp run resume --run runs/<run_id> --mirror s3://bucket/runs/nano --json
```

Serve the models on the cluster with the run. The job uses the user's image and command verbatim. Use SLURM or Kubernetes, never
the local runner. Add a `serve:` section to the run config, one engine per role:
`serve: {judge: {image, command, resources, replicas}}`. Use `encoder` for the retrieval encoder and `reranker` for its reranker.
A served `api: rerank` or `api: openai_embeddings` model without `base_url` needs no URL in the config. A `recipe:<id>` value
needs no URL either. The job's URLs for it reach the step at runtime. A served judge names the engine's own loopback URL. Any
other `base_url` is refused. With the recipes, the engine command is `rcp-vllm serve <id>`. The engine image prepares
`python3 -m pip install --no-deps rcp-vllm`. On SLURM the image needs `container_runtime: apptainer` or `pyxis`. With the
default `none`, the command runs on the node and refuses `image`. The job runs the steps in job phases. Each phase starts only the
engines that its steps use. It passes their URLs to the coordinator in `RCP_NDCG_ENGINES`. `run logs` shows both.
`run resume --runner` submits a failed job again, engines included. It asks only for the windows that its stores lack.

```bash
rcp run start <config> --runner slurm --dry-run --json
rcp run start <config> --runner slurm --json
rcp run resume --run <run_dir> --runner slurm --json
```

Score with the released gains through MTEB (the `mteb` extra). `rcp.eval.mteb.get_tasks("nanobeir")` returns mteb tasks whose
main score is `ndcg_float_at_10`. To score a stored run inside mteb instead, wrap the rankings with
`rcp.eval.mteb.stored_rankings_model(rankings, model_meta("org/model", revision))` and pass it to `mteb.evaluate`. That
writes the predictions file and the `TaskResult`s that an mteb submission needs. To write only the predictions file, use
`rankings.save(folder, format="mteb", task=..., qrels=..., model_name=..., model_revision=...)`. To publish a dataset in MTEB's
Hub layout, write it with the `mteb` writer (`rcp.data.io.mteb.MtebWriter`, `data convert --to mteb`).

## Offline practice

`--judge fake` (or `judge: fake` in a run config) is a deterministic offline judge. It answers through the real prompts and
parsers. Its numbers mean nothing about the documents. The tiny example comes with the package. The run config `tiny` runs from
any directory. `data fetch --dataset tiny --out tiny` copies its data (`rows.jsonl`, and `systems.jsonl` with the systems `bm25`
and `reranker`). Rehearse any command on it, with no model call.

```bash
rcp run start tiny --json
rcp eval score --rankings tiny/systems.jsonl --dataset jsonl:tiny/rows.jsonl --calibration runs/<run_id> --out report.json --json
rcp eval explain --report report.json --query-id q1 --k 5 --json
```

## MCP

`rcp mcp serve` exposes the commands as MCP tools over stdio. The read-only tools are `describe`, `schema_show`,
`data_inspect`, `eval_compare`, `eval_explain` and `calibration_show`. The remaining read-only tools are `run_list`, `run_show`,
`run_status` and `estimate`. The destructive tool is `run_cancel`. `eval_score` is not read-only: when you give it `out`, it
overwrites `out` with the full report. `run_start` starts a run and returns its directory at once. Then poll `run_status`. The
tool list is `rcp.mcp.tool_manifest()`. A plan (`--dry-run`) is CLI-only. Without an MCP client, call the commands with
`--json`. `rcp schema show commands --json` describes the same surface.
