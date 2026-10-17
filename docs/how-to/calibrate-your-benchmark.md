# Calibrate your benchmark

You have a retrieval benchmark (queries, a corpus, and a candidate pool per query) and an LLM you want to use as
the judge. This tutorial produces a calibration directory for it, and with it RCP-nDCG for any system that ranks
the pools. Judging takes many calls. Every step that calls the judge is therefore estimated first.

## 1. Describe the judge

A judge config names an OpenAI-compatible endpoint and the model; everything else has a default
([serving](../concepts/judges.md)). The shipped judges are recipes (`--judge recipe:gpt-oss-120b`, or the bare
recipe id); your own is a YAML file:

```yaml
# my_judge.yaml
base_url: http://127.0.0.1:8000/v1      # any OpenAI-compatible endpoint
model: my-model                         # the served model name
revision: 0123456789abcdef              # the checkpoint commit: recorded in every judgement
max_output_tokens: 8192
context_tokens: 128000
tokenizer: my-org/my-model              # the model's Hugging Face repo (or a tokenizer.json): text limits count its tokens
concurrency: 256
# api_key_env: MY_API_KEY               # a hosted endpoint: the key is read from this environment variable
```

No temperature is set, and the server's default sampling applies. Serve an open-weight model with any engine
and image you choose, under the judge's `model` name and with its reasoning parser ([serving](../concepts/judges.md)
has the commands). For a page-image or video corpus, set the judge's `max_images` (and `image_processor`); the client
sizes every image as that processor would, so the engine needs no pixel-budget flag.

## 2. Describe the run

A run config names the dataset, where the candidates come from, the judge and the steps. The packaged run
config `nano_nfcorpus_gpt5` (`rcp-ndcg run start nano_nfcorpus_gpt5`) retrieves 50 candidates per query with BM25 and judges the first 10 queries
of NanoNFCorpus with a hosted judge:

```yaml
label: nano-nfcorpus-gpt5
dataset: hf://fabianschmidt-cohere/rcp-ndcg-nanobeir/NanoNFCorpusRetrieval
candidates:
  from: retrieval
  retrieval: {kind: bm25}
  depth: 50
judge: gpt5_hosted
steps: [retrieve, tournament, rubric, calibrate, evaluate]
limit: 10
```

Copy it, point `judge` at your judge config and `dataset` at your data, and drop `limit` once the estimate looks
right. A dataset is a URI: `hf://org/name` (a Hub dataset whose card declares its layout, mteb's rules; the
released rcp-ndcg repositories included), `mteb:<Task>` (a task mteb's own loader builds, the `[mteb]`
extra), `beir:<dir>` (`corpus.jsonl`, `queries.jsonl`, `qrels/`) or `jsonl:<file>` (one query per line with its candidates, as in the tiny example, which `rcp-ndcg data fetch --dataset tiny --out tiny` copies).
Relative paths in a run config are relative to the config file, so a config and its data move together.
`rcp-ndcg data inspect --dataset <uri>` checks that it loads. Candidates come from the dataset's own pools
(`from: dataset`), from a rankings file, or from first-stage retrieval as here.

The schedules default to the paper's ([the tournament](../concepts/tournament.md),
[the rubric](../concepts/rubric.md)); a `tournament:` or `rubric:` section overrides them. The run's
`preprocessing` policy decides what the judge reads ([preprocessing](../concepts/preprocessing.md)).

## 3. Estimate, then run

```bash
rcp-ndcg run start my_benchmark.yaml --estimate                   # calls, tokens, wall time; calls no judge
rcp-ndcg run start my_benchmark.yaml
rcp-ndcg run status --run runs/<run_id>
rcp-ndcg run resume --run runs/<run_id>                            # only what is missing
```

The input token counts of an estimate are exact when the judge config names a `tokenizer`, and an approximation
(2.0 characters per token) otherwise. A run that stops keeps every judgement it has, and resuming it asks only for
the missing windows. The tournament
and the rubric write their judgements to `runs/<run_id>/judgements/`, one append-only record per window. The
store's `identity.json` records the prompt, the judge, the schedule, the dataset and the preprocessing. Judgements
of another identity are refused rather than appended. Two rubric variants thus cannot end up in one store.

## 4. The calibration

The run's `calibrate` step fits the 2PL model and writes `runs/<run_id>/calibration/`: the item parameters
(`items.json`), the calibrated abilities (`thetas.parquet`), the per-query scale and offset (`queries.parquet`),
`coverage.json` and `diagnostics.json` ([the calibration](../concepts/calibration.md)). Check the coverage and the
diagnostics before you trust the gains:

```bash
rcp-ndcg calibration show --calibration runs/<run_id>/calibration --json
```

The same fit from Python:

<!-- snippet: skip (needs the judgements of a finished run) -->
```python
from rcp_ndcg.calibration import calibrate, read_judgements

calibration = calibrate(read_judgements("runs/<run_id>/judgements"))
calibration.save("my_calibration/")
```

## 5. Score systems

Any system that ranks the calibrated pools can now be scored with RCP-nDCG, without calling the judge again:

```bash
rcp-ndcg eval score --rankings my_system.parquet --dataset <dataset-uri> --calibration runs/<run_id>/calibration
```

From Python, `rcp_ndcg.eval.evaluate(rankings, dataset=dataset, gains=calibration)` returns the same report. A
document outside the calibrated pools has no gain and counts 0; judging it later without refitting is what the
[primitives](../concepts/primitives.md) are for.

`--json` prints the summary and the per-dataset means; add `--out report.json` for the full report, which
`rcp-ndcg eval compare --report report.json` and `rcp-ndcg eval explain --report report.json --query-id <id>` read.
From Python, rankings held in a pandas frame go in as records:
`rcp_ndcg.Rankings.from_records(frame.to_dict("records"))`.

**What a run scores.** The evaluate step of a run scores two reference systems: `candidates`, the order of the
pools the judge was shown, and `judge`, the pool documents ranked by their calibrated abilities, which scores
RCP-nDCG 1 by construction and is no system to compare. Your own systems join them through `evaluation.systems`
in the run config (`{name: rankings file}`, `<file>#<system>` for one system of a multi-system file), and
`rcp-ndcg eval compare --run runs/<run_id>` compares them without the reference systems (`--include-reference`
adds them).
