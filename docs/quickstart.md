# Quickstart

## Install

RCP-nDCG needs Python 3.12 or later, and will be on PyPI (as `rcp-ndcg`):

```bash
pip install "rcp-ndcg[hf,calibrate]" --extra-index-url https://download.pytorch.org/whl/cpu
```

`uvx rcp-ndcg` runs the command line without installing it (`uvx --from "rcp-ndcg[hf]" rcp-ndcg ...` with extras).

`rcp-ndcg-core` is the metric, the gain, the scoring protocols and the IRT estimators. The metric, gain and
protocols need numpy and pydantic only; the IRT estimators need its `irt` extra (torch and scipy), which the
`calibrate` extra below brings.
`rcp-ndcg` adds the data loaders, the judging client, the calibration, the evaluation and the CLI. The third
distribution, `rcp-ndcg-vllm`, is the lean serving package: the GPU-validated serving recipes and
`rcp-ndcg-vllm serve <recipe-id>` for the stock vLLM image ([recipes and serving
models](reference/recipes.md)). `rcp-ndcg-test`, an unpublished contributor package (reference cases,
conformance, the GPU job tooling), is never installed by users ([its page](reference/rcp-ndcg-test.md)). The
extras:

| Extra | Adds |
|---|---|
| `hf` | downloading the released datasets from the Hugging Face Hub |
| `calibrate` | torch for the calibration fit (the CPU build is enough: `--extra-index-url https://download.pytorch.org/whl/cpu` with pip) |
| `mteb` | the MTEB tasks ([MTEB integration](how-to/mteb-integration.md)) |
| `s3` | the S3 storage backend (`s3://` URIs) |
| `azure` | the Azure Blob storage backend (`az://` URIs) |
| `http` | `http(s)://` object URLs (fsspec's aiohttp backend) |
| `data` | the `hf` and `pdf` readers of `rcp-ndcg data convert` (the Hub dataset loader, PDF rendering) |
| `dev` | the test and lint tools |
| `docs` | the local documentation preview (mkdocs-material) |

Every retrieval model is served now: the client talks HTTP to the engines (role clients over one transport) and
installs beside an engine untouched in its own venv. Serving a shipped recipe via `rcp-ndcg-vllm` adds exactly
one pure-Python wheel to the engine's environment (`pip install --no-deps`), and nothing else changes. The old
`[local]` and `[vllm]` extras are gone; the paper's in-process reference implementations live in
`experiments/paper/rerankers/reference/` with their own pinned requirements, outside the package's lock. BM25
and the hosted APIs need no extra.

From a checkout of the repository, `uv sync --extra hf --extra calibrate` sets up the same environment with
[uv](https://github.com/astral-sh/uv). From git, the release tag installs both packages (pip needs the core in the
same command, because `rcp-ndcg` pins it; with uv the `rcp-ndcg` line alone suffices):

```bash
pip install "rcp-ndcg-core @ git+https://github.com/cohere-ai/rcp-ndcg@v0.0.1#subdirectory=rcp-ndcg-core" \
            "rcp-ndcg[hf,calibrate] @ git+https://github.com/cohere-ai/rcp-ndcg@v0.0.1" \
            --extra-index-url https://download.pytorch.org/whl/cpu
```

A command that needs a missing extra exits with code 10 and prints the install line.

## Four paths

### Score a system on the released data, without an LLM

The released datasets carry calibrated gains for every judged pool document. Scoring a system thus needs no judge.
Put your system's scores in a table with `query_id`, `doc_id` and `score` columns and optional `system` and
`dataset` (Parquet, CSV, a TREC run or JSONL work: the table formats' columns take the common aliases
`query_id`/`query-id`/`qid`/`query`, `doc_id`/`corpus-id`/`corpus_id`/`docid`/`docno`, `score`/`rerank_score`/`sim`,
`system`/`model`/`run`/`tag`/`run_id`, `dataset`/`subset`, and a JSONL row uses the canonical keys `query_id`,
`doc_id`, `score` -- or a `scores` / `doc_ids` row), and score it against a suite. The subsets of a suite
share query ids, so a file that ranks several subsets names each row's subset in a `dataset` column; without it,
`eval score` refuses the file. A TREC run has no such column and holds one subset: score it with
`--subset <name>` (in Python, `load_rankings(path, dataset="<name>")`). No rankings file yet? The next path
produces one from a model.

```bash
rcp-ndcg eval score --rankings my_system.parquet --suite nanobeir --json
```

<!-- snippet: skip (needs your rankings file and the network) -->
```python
from rcp_ndcg.data import load_rankings
from rcp_ndcg.eval import evaluate

report = evaluate(load_rankings("my_system.parquet"), suite="nanobeir")
print(report.to_pandas())
```

The suite brings the paper's scoring protocol ([scoring protocols](concepts/protocols.md)). [Data](data.md) lists
the datasets and shows a complete example.

### Serve an open model and score it

Engine side, on the stock `vllm/vllm-openai` image: `python3 -m pip install --no-deps rcp-ndcg-vllm` (the one
change the engine environment takes; `pip freeze` differs by exactly that wheel), then serve a recipe:

```bash
rcp-ndcg-vllm serve qwen3-embedding-0.6b --port 8000
rcp-ndcg-vllm serve qwen3-reranker-0.6b --port 8001   # a reranker, in another terminal or engine
# `rcp-ndcg-vllm serve <id> --dry-run` prints the engine argv and exits
```

Client side, `rcp-ndcg-vllm` installed beside `rcp-ndcg`, and a retriever config whose encoder names the recipe;
`base_url` and the other run-time fields stay on the config (an explicit content field must equal the recipe's,
or the config is refused naming both values):

```yaml
# recipe-retriever.yaml
kind: dense
encoder:
  recipe: qwen3-embedding-0.6b
  base_url: http://127.0.0.1:8000/v1
```

```bash
rcp-ndcg retrieval index --dataset suite:nanobeir --subset NanoSciFact --retriever recipe-retriever.yaml --out index/
rcp-ndcg retrieval search --dataset suite:nanobeir --subset NanoSciFact --retriever recipe-retriever.yaml --out rankings.parquet
rcp-ndcg retrieval rerank --dataset suite:nanobeir --subset NanoSciFact --rankings rankings.parquet \
    --reranker recipe:qwen3-reranker-0.6b --out rankings-reranked.parquet
rcp-ndcg eval score --rankings rankings-reranked.parquet --suite nanobeir
```

(one suite subset, as here; `--retriever recipe:<id>` / `--reranker recipe:<id>` are shorthands whose URL comes from
`--set retriever.encoder.base_url=...` or a `serve:` engine). Budgets are explicit in every recipe: over-budget
content is cut client-side at token boundaries with the template's anchors preserved, and every cut is
recorded -- never engine-side ([text budgets](concepts/text-budgets.md)). The
[recipe catalog](reference/recipes.md) names the 30 recipes and their roles.

### Re-judge a pool with your own endpoint

Serve a judge behind an OpenAI-compatible endpoint, estimate the calls and tokens, then run:

```bash
rcp-ndcg run start rejudge_nfcorpus --judge-url http://localhost:8000/v1 --judge-model my-model --estimate
rcp-ndcg run start rejudge_nfcorpus --judge-url http://localhost:8000/v1 --judge-model my-model
```

Documents run whole up to the judge's text policy -- 32 768 tokens by default (documented policy, every cut
recorded); `--set judge.tokenizer=...` makes the counts exact.

[Calibrate your benchmark](how-to/calibrate-your-benchmark.md) walks through a run, and
[primitives](concepts/primitives.md) shows how to re-judge only some documents, insert new ones, or add a second
judge without refitting a published calibration.

### Reproduce a table of the paper

```bash
pip install -r experiments/requirements.txt
python experiments/fetch_data.py
python experiments/run_all.py
```

[Reproduce the paper](how-to/reproduce-the-paper.md) says what the scripts check and what they do not cover.

## Run the whole pipeline as a job

One complete run config: candidates from a served encoder and reranker (their recipes supply the client blocks),
a judge config, and one engine per role, started inside the job. The engine image prepares
`python3 -m pip install --no-deps rcp-ndcg-vllm`; with the recipes the engine command is `rcp-ndcg-vllm serve
<id>`.

```yaml
# my_run.yaml
dataset: suite:nanobeir
candidates:
  from: retrieval
  # with `rcp-ndcg-vllm` installed, `encoder: {recipe: qwen3-embedding-0.6b}` (and `rerank: {recipe:
  # qwen3-reranker-0.6b}`) take their whole client block from the recipe instead of the fields below; a
  # self-hosted role config here must still declare its text budget (tokenizer + max_tokens), and a served
  # reranker sets use_activation explicitly
  retrieval: {kind: dense, encoder: {api: openai_embeddings, model: my-encoder,
                                     tokenizer: my-encoder-tokenizer, max_tokens: 8192}}
  rerank: {api: rerank, model: my-reranker, use_activation: true,
           tokenizer: my-reranker-tokenizer, max_tokens: 8192}
  depth: 50
judge: my-judge.yaml               # your judge config (or the name of a shipped one; see the judges page)
steps: [retrieve, rerank, tournament, rubric, calibrate, evaluate]
serve:
  encoder:
    image: registry.example.com/vllm-openai:my-tag
    command: ["rcp-ndcg-vllm", "serve", "qwen3-embedding-0.6b"]
    resources: {gpus: 1}
  reranker:
    image: registry.example.com/vllm-openai:my-tag
    command: ["rcp-ndcg-vllm", "serve", "qwen3-reranker-0.6b"]
    resources: {gpus: 1}
  judge:
    image: registry.example.com/vllm-openai:my-tag
    command: ["bash", "-lc", "<your judge's engine command, one line>"]   # an argv list, verbatim
    resources: {gpus: 8}
    outage_timeout_s: 900
runner:
  name: kubernetes
  options: {namespace: eval, secrets: [hf-token]}
mirror: s3://YOUR-BUCKET/runs/my-run
```

Walk it on the cluster: `rcp-ndcg run start my_run.yaml --runner kubernetes --dry-run` prints the step plan and
the manifests it would submit and writes nothing; the same command without `--dry-run` submits (add `--detach`
to return at once); `rcp-ndcg run status --run runs/<run_id>` and `rcp-ndcg run logs --run runs/<run_id>` follow
it; `rcp-ndcg run resume --run runs/<run_id> --runner kubernetes` submits a failed job again, engines included,
asking only for the windows its stores lack. The job runs its steps in **job phases**, each starting only the
engines its steps use and handing their URLs to the step (`RCP_NDCG_ENGINES`) -- the job's GPUs are the maximum
over phases, not the sum over engines; a job whose engine stops answering for `outage_timeout_s` fails instead
of holding the allocation. [Runs, engines and runners](concepts/runs.md) and [judges](concepts/judges.md) have
the depth.

## The metric alone

With calibrated abilities and item parameters at hand, the metric is a pure function of the core package:

```python
from rcp_ndcg_core import gain, ndcg

items = {"gamma": [1.2, 1.0, 1.1, 0.9, 0.8], "beta": [-2.0, -0.5, 0.0, 0.8, 1.7]}
gains = {doc_id: gain(theta, items) for doc_id, theta in {"d1": 1.4, "d2": -0.3, "d3": 0.6}.items()}
print(ndcg({"d1": 0.91, "d2": 0.75, "d3": 0.20}, gains, k=10))
```

[The metric](concepts/metric.md) defines it.
