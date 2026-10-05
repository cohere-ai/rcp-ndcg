---
name: rcp-ndcg
description: Evaluate retrievers, rerankers or embedders with calibrated relevance (RCP-nDCG) instead of sparse binary qrels; re-judge a candidate pool with an LLM judge; extend a pool without refitting. Use when someone asks how good a retrieval system is, wants to compare two systems, suspects their nDCG numbers are misleading, or needs relevance labels without human annotators.
---

# RCP-nDCG

RCP-nDCG is nDCG whose gains are calibrated relevance probabilities from an LLM judge: a listwise tournament orders
each query's pool, a rubric of five binary criteria (C1 to C5) anchors the scale, and a 2PL item-response model puts
all queries on one scale. Released datasets carry these gains, so scoring a system on them needs no LLM.

Install (Python 3.12): `pip install "rcp-ndcg[hf,calibrate]" --extra-index-url https://download.pytorch.org/whl/cpu`,
or run it without installing: `uvx rcp-ndcg`. Every command
but `mcp serve` (a stdio server) takes `--json` and then prints exactly one JSON document on stdout: `{"ok": true, "data": {...}}` or
`{"ok": false, "error": {"code", "exit_code", "message", "hint", "retryable", "details"}}`. Read `data`, never the
human text. `rcp-ndcg schema show commands --json` lists every command with its flags (type, default, help) and
output schema. A config error (exit 3) carries `error.details.errors`: per problem the `field`, the given `input`,
the `expected` type, a `did_you_mean` for an unknown key, and its `source` (`--set` or the config file).

## Three paths

**1. Score a system on the released data (no LLM).** The rankings file holds `query_id`, `doc_id`,
`score` (Parquet, CSV, TREC run or JSONL), and a `dataset` column naming each row's subset when it ranks several
subsets of a suite (they share query ids; without it the file is refused). A TREC run holds one subset: add
`--subset <name>`.

```bash
rcp-ndcg eval score --rankings my_system.parquet --suite nanobeir --json
```

Read `data.summary`: one row per system and metric (`rcp_ndcg`, `qrel_ndcg`) with `value`, `ci_low`, `ci_high`.
`data.per_dataset` holds the mean per dataset. The per-query values stay out of stdout: `--per-query` adds them,
`--fields summary` keeps only the named fields, and `--out report.json` writes the full report (what `eval compare
--report` and `eval explain --report` read). Suites: `nanobeir`, `bright`, `vidore`, `trecdl`. Python:
`rcp_ndcg.evaluate(rcp_ndcg.load_rankings(path), suite="nanobeir")`. A file of several systems where one system's
rankings match nothing of the dataset is refused (exit 12); score the healthy ones with `--system NAME`
(repeatable), or `systems=[...]` in Python.

**2. Re-judge a pool with an OpenAI-compatible endpoint (calls the judge).** Ask the user for the model's
tokenizer (its Hugging Face repo id, or a `tokenizer.json` path): `--set judge.tokenizer=<id>` makes text limits
and estimates count the judge's own tokens. Without it, documents are sent whole, and a text policy that cuts
(`truncate`, `chunk`, `fail`) exits 3.

```bash
rcp-ndcg run start rejudge_nfcorpus --judge-url http://localhost:8000/v1 --judge-model my-model --estimate --json
rcp-ndcg run start rejudge_nfcorpus --judge-url http://localhost:8000/v1 --judge-model my-model --detach --json
rcp-ndcg run status --run runs/<run_id> --json
```

Read `data.estimate` of the first (`calls`, `input_tokens`, `output_tokens`, `wall_s`, `assumptions`), `data.run_dir` of the
second, and `data.status`, `data.steps` and `data.metrics` (`<system>/<metric>@<k>`; the full report with
confidence intervals is `<run_dir>/metrics/report.json`) of the third. `rejudge_nfcorpus` is a run config that ships
with the package (`run start` takes a packaged config's name or a YAML path; paths inside a config are relative to
the config file). `data.steps` lists every planned step in run order (`pending` until it starts), each judging step
with `progress: {done, planned}` in judge windows. Poll `run status` until `data.done` is true: `data.status` is then
`completed`, `partial`, `failed` or `cancelled`. A run started with `--only` for some of its steps ends `partial` with
exit code 0; `run resume --run <dir>` runs the rest.

**3. Reproduce a table of the paper** (from a checkout, no LLM): `python experiments/fetch_data.py`, then
`python experiments/run_all.py`. Exit code 0 means every value matched the paper within its tolerance.

## Invariants

- **Estimate before judging.** Run `--estimate` first, show the user `calls`, the tokens and `wall_s`, and get
  agreement; then run. Input token counts
  are exact when the judge names a tokenizer and approximate otherwise (2 characters per token);
  `data.estimate.input_token_count` says which.
- **Never pool judgements across families.** A family is the judge model, revision and tokenizer, the prompt, the
  criteria, the parse version and the preprocessing. `rcp-ndcg calibration show --calibration DIR --json` lists them in
  `data.families`. Exit code 11 is a refusal to mix, not an error to retry.
- **Never compare numbers from different protocols, cutoffs or tie rules.** `data.protocol` of an eval report names
  the protocol. State the suite, protocol, `k` and judge next to every number you report.
- **Do not edit the prompts.** The rubric has exactly five criteria, C1 to C5; a changed prompt is a new family.
- **The judge's own order is not a system.** A run scores two reference systems besides yours: `candidates` (the pool
  order the judge was shown) and `judge` (ranked by the calibrated abilities, RCP-nDCG 1 by construction).
  `eval compare --run` leaves both out unless `--include-reference`; add your own systems to a run with
  `evaluation.systems` (`{name: rankings file}`; `<file>#<system>` picks one system of a file that holds several).
- **Report only what was computed.** A qrel-nDCG of `null` means the query has no positive grade; pass it through.
- Treat exit codes 11 and 12 as refusals to fix (the hint says how), not as errors to retry.

## Reading results

- *Is A better than B?* `rcp-ndcg eval compare --run DIR --json` (or `--report FILE` from `eval score --out`):
  `data.pairs[]` with `delta` (B minus A), `ci_low`, `ci_high`, `p_value`, `significant`, and `sign_flips` (queries
  where RCP-nDCG and qrel-nDCG disagree).
- *Can this calibration be trusted?* `rcp-ndcg calibration show --calibration DIR --json`: `data.coverage` (queries
  without calibration, invalid windows, degenerate documents), `data.diagnostics` (ECE, Brier score) and
  `data.warnings` (`INVALID_WINDOWS`: queries resting on fewer windows than planned). A poor fit means the run needs
  more judgements, not that RCP-nDCG is wrong.
- *Why?* `rcp-ndcg eval explain --run DIR --query-id QID --json` (or `--report FILE`): each system's top k with
  each document's theta, gain and per-criterion pass probabilities (the criteria's `gamma` and `beta` once, in
  `data.items`), and `data.deltas`: the gap to the first system at cutoff `data.k`, split into `selection` and
  `ordering`. Texts are left out unless `--include-text`.

## Exit codes

| Code | Name | Meaning; what an agent should do |
|---|---|---|
| 0 | `SUCCESS` | done (including a no-op resume) |
| 1 | `INTERNAL` | a bug; report it |
| 2 | `USAGE` | bad flags or arguments; fix the invocation |
| 3 | `CONFIG` | invalid config value; fix the YAML or `--set` |
| 4 | `MISSING_INPUT` | a file, run or dataset is absent; the hint names it |
| 5 | `CREDENTIALS` | missing or rejected credentials; the hint names the variable, never its value |
| 6 | `PROVIDER` | an endpoint or a scheduler failed after its retries (unreachable, timing out, rate limiting, an empty answer); resume later when `retryable` is true (it is false for a route or model the endpoint does not have, HTTP 404) |
| 8 | `CAPABILITY` | the judge or endpoint cannot take what a request carries: an answer schema it refuses (serve with the reasoning parser, or set `decoding: free`), images or videos beyond its `max_images` / `max_videos`, a window whose media exceed its context, media for a text-only encoder; raised by the first such request |
| 9 | `INTERRUPTED` | SIGINT or SIGTERM stopped the command; the state on disk is consistent; resume |
| 10 | `DEPENDENCY` | a missing extra; the hint is the exact install command |
| 11 | `IDENTITY` | refusing to mix: resume with a changed config, judgements from another family, a scale check failed on insertion; the hint names the differing fields and the way out (a new output directory or run; `--force` where the command has it) |
| 12 | `DATA` | input that would produce wrong numbers or does not parse: malformed or non-finite rankings, qrels or gains, gains that match no labelled query, ids that do not join, a document over its text cap with `on_overflow: fail`, a new document the evidence cannot identify, a query with invalid windows under `--strict`, a damaged mirror |

Code 7 is retired: no command returns it, and it is not reused.

## Recipes

Score documents a calibration lacks (added to a pool after the fit, or without a theta in it because their windows
failed) without refitting (items frozen); the judgement store is append-only, so only the new windows are asked. A
document the calibration already holds keeps its theta and is listed as skipped, so this does not re-score it:

```bash
rcp-ndcg judge rubric --dataset <uri> --judge <judge> --docs q2:d10 --docs q2:d11 --out <store> --estimate --json
rcp-ndcg judge rubric --dataset <uri> --judge <judge> --docs q2:d10 --docs q2:d11 --out <store>
rcp-ndcg calibration score --calibration <calibration> --judgements <store> --out <extended> --json
```

Insert a new document into a tournament calibration: plan its windows with the calibration's tournament store
(the window size is the store's schedule), judge exactly the planned windows into that store with the same judge
(`--plan`; anything else is exit 11), then insert. `data.plan.calls` is the number of judge calls. The new document
must be in the dataset's corpus. `data.extension.anchor_report.ok` must be true.

```bash
rcp-ndcg calibration insert --calibration <calibration> --judgements <store> --plan --query q1 --doc new-doc --n 36 --out plan.json --json
rcp-ndcg judge tournament --dataset <uri> --judge <judge> --plan plan.json --out <store> --estimate --json
rcp-ndcg judge tournament --dataset <uri> --judge <judge> --plan plan.json --out <store>
rcp-ndcg calibration insert --calibration <calibration> --judgements <store> --out <extended> --json
```

If the insertion refuses with exit 12 (the new document's standard error misses `--se-target`), plan more opponents
(a larger `--n`) and judge the new plan: its windows are new (their calls are `data.plan.calls`), and the windows
judged before still count as evidence. A plan cannot hold more opponents than the query has other documents:
`data.plan.capped` is true when `--n` asked for more (`data.plan.opponents` says how many it holds), and on such a
small pool a larger `--n` adds nothing, so accept a larger `--se-target` instead.

Add a second judge that answered the same rubric: one pooled fit, one severity per judge in `data.judge_severity`.

```bash
rcp-ndcg calibration fit --judgements <store> --judgements <second-store> --judges pooled --out <pooled> --json
```

Make a long judge pass survive preemption: mirror it to any fsspec URI (S3, GCS, Azure, or a filesystem your
package registers with fsspec), follow `data.mirror.lag_s` in `run status`, and resume on any node from the mirror:

```bash
rcp-ndcg run start <config> --mirror s3://bucket/runs/nano --detach --json
rcp-ndcg run status --run runs/<run_id> --json
rcp-ndcg run resume --run runs/<run_id> --mirror s3://bucket/runs/nano --json
```

Serve the judge on the cluster with the run (the user's image and command, verbatim; SLURM or Kubernetes, never the
local runner): add a `serve:` section (`image`, `command`, `resources`, `replicas`) to the run config, check what
would be submitted, then submit. On SLURM the image needs `container_runtime: apptainer` or `pyxis`; with the default
`none` the command runs on the node and `image` is refused. The job hands the engine replicas' URLs to the judge;
`run logs` shows both. A job that failed (`run status`: `failed`, with a `note` when the job ended without recording
it) is submitted again, engine included, with `run resume --runner`; it asks only for the windows its stores lack.

```bash
rcp-ndcg run start <config> --runner slurm --dry-run --json
rcp-ndcg run start <config> --runner slurm --json
rcp-ndcg run resume --run <run_dir> --runner slurm --json
```

Score with the released gains through MTEB (the `mteb` extra): `rcp_ndcg.eval.mteb.get_tasks("nanobeir")` returns
mteb tasks whose main score is `ndcg_float_at_10`.

## Offline practice

`--judge fake` (or `judge: fake` in a run config) is a deterministic offline judge that answers through the real
prompts and parsers; its numbers mean nothing about the documents. The tiny example ships with the package: the run
config `tiny` runs from any directory, and `data fetch --dataset tiny --out tiny` copies its data (`rows.jsonl`,
`systems.jsonl` with the systems `bm25` and `reranker`). Rehearse any command on it, calling no model:

```bash
rcp-ndcg run start tiny --json
rcp-ndcg data fetch --dataset tiny --out tiny --json
rcp-ndcg judge rubric --dataset jsonl:tiny/rows.jsonl --judge fake --out practice/judgements --json
rcp-ndcg eval score --rankings tiny/systems.jsonl --dataset jsonl:tiny/rows.jsonl --calibration runs/<run_id> --out report.json --json
rcp-ndcg eval explain --report report.json --query-id q1 --k 5 --json
```

## MCP

`rcp-ndcg mcp serve` exposes the commands as MCP tools over stdio. Read-only: `describe`, `schema_show`,
`data_inspect`, `eval_score` (with `out`, `per_query` and `fields` as on the command line), `eval_compare`,
`eval_explain`, `calibration_show`, `run_list`, `run_show`, `run_status`, `estimate`. Destructive: `run_cancel`.
`run_start` starts a run and returns its directory at once; then poll `run_status`. Without an MCP client,
`rcp-ndcg mcp tools --call <tool> --args '<json>' --json` calls one tool and prints its result as the server would.
