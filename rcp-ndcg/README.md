<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/cohere-ai/rcp-ndcg/v0.0.1/docs/assets/cohere-logo-dark.svg">
    <img src="https://raw.githubusercontent.com/cohere-ai/rcp-ndcg/v0.0.1/docs/assets/cohere-logo.svg" alt="Cohere" height="36">
  </picture>
</p>

# RCP-nDCG

RCP-nDCG is nDCG whose gains come from calibrated LLM judgements instead of sparse human relevance labels. An LLM
judge orders each query's candidate pool in a listwise tournament and answers five binary rubric criteria for every
document; a two-parameter item-response model then puts the tournament scores of all queries on one scale, and a
document's gain is its discrimination-weighted probability of passing the criteria. The method and its validation
are in the paper
[Rubric-Calibrated Preferences: Cross-Query Calibration of LLM Judgments via Item Response Theory](https://arxiv.org/abs/2609.35739).

<p align="center">
  <img src="https://raw.githubusercontent.com/cohere-ai/rcp-ndcg/v0.0.1/docs/assets/rcp-pipeline.png" alt="The RCP-nDCG pipeline: an LLM judge runs a listwise tournament (Stage A) and answers five binary criteria (Stage B); an item-response model calibrates every query's tournament scores onto one shared scale; each document's gain is its discrimination-weighted probability of passing the criteria." width="100%">
</p>
<p align="center"><em>The RCP-nDCG pipeline (Figure 2 of the paper).</em></p>

This repository holds three published distributions -- `rcp-ndcg` (the pipeline and the command line),
`rcp-ndcg-core` (the metric) and `rcp-ndcg-vllm` (the serving recipes) -- and the unpublished `rcp-ndcg-test`
for contributors, the scripts that reproduce the paper's tables from the public data (`experiments/`), and
runnable examples (`examples/`).

## Install

RCP-nDCG needs Python 3.12 or later; install it from PyPI:

```bash
pip install "rcp-ndcg[hf,calibrate]" --extra-index-url https://download.pytorch.org/whl/cpu
rcp-ndcg --version
```

or, without installing anything, `uvx rcp-ndcg --version`. To work from the repository, install from a
checkout instead: `uv sync --extra hf --extra calibrate` (the [quickstart](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/quickstart.md#install) has the git install and the full extras
table). The extras add the Hugging Face Hub (`hf`), torch for the calibration fit (`calibrate`; the CPU build
from the PyTorch index above is enough) and MTEB (`mteb`; the quickstart table lists all nine). Every retrieval
model is served now, so no extra carries in-process model code. `rcp-ndcg-core`, which comes with `rcp-ndcg`, is
the metric, the gains, the scoring protocols and the IRT estimators, with numpy and pydantic only (`pip install
rcp-ndcg-core`). A command that needs a missing extra exits with code 10 and prints the install line.

## Distributions

- `rcp-ndcg` -- the judging pipeline, the calibration, the evaluation and the command line.
- `rcp-ndcg-core` -- the metric, the gains, the scoring protocols and the IRT estimators (numpy and pydantic only).
- `rcp-ndcg-vllm` -- the lean serving package: the GPU-validated serving recipes and `rcp-ndcg-vllm serve
  <recipe-id>` for the stock vLLM image ([recipes and serving models](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/reference/recipes.md)).

The unpublished `rcp-ndcg-test` holds the repository's reference cases, conformance suite and GPU job tooling
([its page](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/reference/rcp-ndcg-test.md)).

## Datasets

All data behind the paper is public on the Hugging Face Hub:

| Dataset | Contents |
|---|---|
| [rcp-ndcg-nanobeir](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-nanobeir) | 13 NanoBEIR tasks with calibrated RCP gains; runs with stock `mteb` ≥ 2.0.1 (`ndcg_float_at_10`) |
| [rcp-ndcg-bright](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-bright) | 12 BRIGHT tasks with calibrated RCP gains; stock `mteb` ≥ 2.0.1 |
| [rcp-ndcg-vidore-v3](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-vidore-v3) | 8 ViDoRe v3 visual-document tasks (6 languages) with calibrated RCP gains; stock `mteb` ≥ 2.10.5 |
| [rcp-ndcg-trecdl](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-trecdl) | TREC-DL 2019 and 2020 with continuous RCP gains next to the NIST judgements; stock `mteb` ≥ 2.0.1 |
| [rcp-ndcg-external-validation](https://huggingface.co/datasets/fabianschmidt-cohere/rcp-ndcg-external-validation) | The human contest study (46 annotators, 311 contests, 7,080 grades) and the external LLM judges (GLM-5.3-flash, DeepSeek-4.1-flash, Kimi-K3) |

`experiments/fetch_data.py` downloads all five at the pinned revisions used for the paper's tables.
[docs/data.md](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/data.md) describes their layout and how to load them. Each dataset card states its license.

## Four paths

### 1. Score a system on the released data, without an LLM

The released datasets carry a calibrated gain for every judged pool document, so scoring a ranking needs no judge.
Put your system's scores in a file with `query_id`, `doc_id` and `score` columns and optional `system` and
`dataset` (Parquet, CSV, a TREC run or JSONL: the table formats' columns take the common aliases
`query_id`/`query-id`/`qid`/`query`, `doc_id`/`corpus-id`/`corpus_id`/`docid`/`docno`, `score`/`rerank_score`/`sim`,
`system`/`model`/`run`/`tag`/`run_id`, `dataset`/`subset`, and a JSONL row uses the canonical keys `query_id`,
`doc_id`, `score` -- or a `scores` / `doc_ids` row) and score it against a suite with the paper's scoring
protocol. The subsets of a suite share query ids, so a file that ranks several subsets names each row's subset in a
`dataset` column ([data](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/data.md)):
no rankings file yet? Path 2 produces one from a model.

```bash
rcp-ndcg eval score --rankings my_system.parquet --suite nanobeir
```

[`examples/01_score_released_suite.py`](https://github.com/cohere-ai/rcp-ndcg/blob/main/examples/01_score_released_suite.py) scores the paper's 14 rerankers on a
NanoBEIR task from their released runs:

<!-- snippet: example examples/01_score_released_suite.py -->
```python
import rcp_ndcg as rcp

REPO = "fabianschmidt-cohere/rcp-ndcg-nanobeir"
TASK = "NanoFiQA2018Retrieval"

dataset = rcp.load_dataset(f"hf://{REPO}/{TASK}")  # qrels with gains, judged pools, excluded ids; protocol nanobeir
rankings = rcp.load_rankings(f"hf://datasets/{REPO}/provenance/runs/{TASK}.parquet")
rerankers = [s for s in rankings.systems if "judge" not in s]  # the file also holds the judge's own orders

report = rcp.evaluate(rankings, dataset=dataset, k=10, bootstrap=0)
print(f"{TASK} under the {dataset.protocol} protocol")
print(f"{'reranker':42s} {'RCP-nDCG@10':>12s} {'qrel-nDCG@10':>13s}")
for system in sorted(rerankers, key=lambda s: -report.value(s, "rcp_ndcg")):
    print(f"{system:42s} {report.value(system, 'rcp_ndcg'):12.4f} {report.value(system, 'qrel_ndcg'):13.4f}")
```

[`examples/02_score_tiny_offline.py`](https://github.com/cohere-ai/rcp-ndcg/blob/main/examples/02_score_tiny_offline.py) does the same offline, on a three-query
dataset that ships with the package (`rcp-ndcg data fetch --dataset tiny --out tiny` copies it).

### 2. Serve an open model and score it

The retrieval commands score any served encoder and reranker. Engine side, on the stock `vllm/vllm-openai`
image: `python3 -m pip install --no-deps rcp-ndcg-vllm` (the one change the engine environment takes), then
`rcp-ndcg-vllm serve qwen3-embedding-0.6b --port 8000` (and `rcp-ndcg-vllm serve qwen3-reranker-0.6b --port
8001` for the reranker). Client side, a retriever config whose encoder names the recipe -- the recipe supplies
the client block, `base_url` and the other run-time fields stay yours:

```yaml
kind: dense
encoder:
  recipe: qwen3-embedding-0.6b
  base_url: http://127.0.0.1:8000/v1
```

```bash
rcp-ndcg retrieval search --dataset suite:nanobeir --subset NanoSciFact --retriever retriever.yaml --out rankings.parquet
rcp-ndcg eval score --rankings rankings.parquet --suite nanobeir
```

The [recipe catalog](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/reference/recipes.md) names the 24
recipe families and their 44 variants, and the [quickstart](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/quickstart.md#serve-an-open-model-and-score-it) walks through it.

### 3. Re-judge a pool with your own endpoint

Serve a judge behind any OpenAI-compatible endpoint (vLLM, a hosted API). Estimate the calls and tokens,
then run:

```bash
rcp-ndcg run start rejudge_nfcorpus --judge-url http://localhost:8000/v1 --judge-model my-model --estimate
rcp-ndcg run start rejudge_nfcorpus --judge-url http://localhost:8000/v1 --judge-model my-model
```

The run judges the released NanoNFCorpus pools on both stages, fits the calibration and scores the pools, and
writes every artifact to `runs/<run_id>/`. For long passes, add `--mirror <any fsspec URI>` (S3 and Azure via
the `s3`/`azure` extras, GCS as installed (`gcsfs` comes with `rcp-ndcg`), or any fsspec filesystem you
register) so a preempted job resumes where it stopped; see
[durability](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/concepts/runs.md#durability-local-runs-and-a-mirror).
[Judges](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/concepts/judges.md) gives the vLLM
commands for each shipped judge, and a run config's `serve:` section starts one engine per role (the judge, the
retrieval encoder, the reranker) in a SLURM or Kubernetes job -- with the recipes, the engine command is
`rcp-ndcg-vllm serve <id>` (the quickstart's job section; [runs](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/concepts/runs.md) for depth). To rehearse offline, the same pipeline runs with a
deterministic fake judge:

```bash
rcp-ndcg run start tiny
```

[Calibrate your benchmark](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/how-to/calibrate-your-benchmark.md) walks through a run on your own data, and
[primitives](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/concepts/primitives.md) shows how to re-judge some documents, insert new ones, or pool a second
judge without refitting a published calibration (examples 04 to 06).

### 4. Reproduce a table of the paper

```bash
pip install -r experiments/requirements.txt
python experiments/fetch_data.py                 # the public datasets, at pinned revisions (about 150 MB)
python experiments/run_all.py                    # every check; exits 1 if a value falls outside its tolerance
```

The scripts recompute the NanoBEIR, BRIGHT, ViDoRe v3 and TREC-DL leaderboards, the human contest study and the
external-judge comparisons from the released data, and print each paper value next to the reproduced one.
[REPRODUCIBILITY.md](https://github.com/cohere-ai/rcp-ndcg/blob/main/REPRODUCIBILITY.md) and [experiments/README.md](https://github.com/cohere-ai/rcp-ndcg/blob/main/experiments/README.md) say what is covered.

## MTEB

The released datasets run with stock [mteb](https://github.com/embeddings-benchmark/mteb) through the
`rcp_ndcg_tasks.py` each dataset ships, and through `rcp_ndcg.eval.mteb.get_tasks` (the `mteb` extra). Each task
reports `ndcg_float_at_k`, nDCG over the continuous RCP gains with group-mean ties. An integration into mteb itself
is proposed in [embeddings-benchmark/mteb#5516](https://github.com/embeddings-benchmark/mteb/pull/5516). See
[`examples/07_mteb.py`](https://github.com/cohere-ai/rcp-ndcg/blob/main/examples/07_mteb.py) and [the MTEB page](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/how-to/mteb-integration.md).

## Documentation

- [docs/](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/index.md): the metric, the scoring protocols, the tournament, the rubric, the calibration, the
  primitives, preprocessing and text budgets, the inference layer, retrieval, judges, and runs and runners, one
  page each; the data; the how-to guides (calibrate, MTEB, reproduce, serve a model, add and validate a recipe);
  the command line and the recipe catalog.
- [examples/](https://github.com/cohere-ai/rcp-ndcg/blob/main/examples/): nine short scripts, six of which run offline.
- [skills/rcp-ndcg/SKILL.md](https://github.com/cohere-ai/rcp-ndcg/blob/main/skills/rcp-ndcg/SKILL.md): instructions for a coding agent that uses RCP-nDCG from
  another project. Every command except `mcp serve` takes `--json`; `rcp-ndcg schema show commands` describes the
  command line, and `rcp-ndcg mcp serve` serves a subset of it as MCP tools (the list is
  `rcp_ndcg.mcp.tool_manifest()`; a plan, `--dry-run`, is CLI-only).
- [AGENTS.md](https://github.com/cohere-ai/rcp-ndcg/blob/main/AGENTS.md): for contributors to this repository. [CHANGELOG.md](https://github.com/cohere-ai/rcp-ndcg/blob/main/CHANGELOG.md): the public surface. [Compatibility and versioning](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/reference/versioning.md): what is public, the 0.0.x rules, the recipe `schema_version` and the artifact tags.
- [SECURITY.md](https://github.com/cohere-ai/rcp-ndcg/blob/main/SECURITY.md): how to report a vulnerability (GitHub private vulnerability reporting) and what is supported.

## Citation

If you use RCP-nDCG, please cite the paper. [CITATION.cff](https://github.com/cohere-ai/rcp-ndcg/blob/main/CITATION.cff) holds the reference, and GitHub's "Cite
this repository" turns it into BibTeX.

## License

Apache-2.0, see [LICENSE](https://github.com/cohere-ai/rcp-ndcg/blob/main/LICENSE). Copyright 2026 Cohere Inc. The datasets carry their own licenses on the Hub. Each distribution carries `LICENSE` and `NOTICE` in its wheels and sdists.
