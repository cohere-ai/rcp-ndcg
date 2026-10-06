# Quickstart

## Install

RCP-nDCG needs Python 3.12 or later, and is on PyPI:

```bash
pip install "rcp-ndcg[hf,calibrate]" --extra-index-url https://download.pytorch.org/whl/cpu
```

`uvx rcp-ndcg` runs the command line without installing it (`uvx --from "rcp-ndcg[hf]" rcp-ndcg ...` with extras).

`rcp-ndcg-core` is the metric, the gain, the scoring protocols and the IRT estimators. The metric, gain and
protocols need numpy and pydantic only; the IRT estimators need its `irt` extra (torch and scipy), which the
`calibrate` extra below brings.
`rcp-ndcg` adds the data loaders, the judging client, the calibration, the evaluation and the CLI. The extras:

| Extra | Adds |
|---|---|
| `hf` | downloading the released datasets from the Hugging Face Hub |
| `calibrate` | torch for the calibration fit (the CPU build is enough: `--extra-index-url https://download.pytorch.org/whl/cpu` with pip) |
| `mteb` | the MTEB tasks ([MTEB integration](tutorials/mteb-integration.md)) |
| `dev` | the test and lint tools |

Every retrieval model is served now: the package talks HTTP to the engines (role clients over one transport) and
installs cleanly next to an engine image without touching it. The old `[local]` and `[vllm]` extras are gone; the
paper's in-process reference implementations live in `experiments/paper/rerankers/reference/` with their own
pinned requirements, outside the package's lock. BM25 and the hosted APIs need no extra.

From a checkout of the repository, `uv sync --extra hf --extra calibrate` sets up the same environment with
[uv](https://github.com/astral-sh/uv). From git, the release tag installs both packages (pip needs the core in the
same command, because `rcp-ndcg` pins it; with uv the `rcp-ndcg` line alone suffices):

```bash
pip install "rcp-ndcg-core @ git+https://github.com/cohere-ai/rcp-ndcg@v0.0.1#subdirectory=packages/rcp-ndcg-core" \
            "rcp-ndcg[hf,calibrate] @ git+https://github.com/cohere-ai/rcp-ndcg@v0.0.1" \
            --extra-index-url https://download.pytorch.org/whl/cpu
```

A command that needs a missing extra exits with code 10 and prints the install line.

## Three paths

### Score a system on the released data, without an LLM

The released datasets carry calibrated gains for every judged pool document. Scoring a system thus needs no judge.
Put your system's scores in a table with `query_id`, `doc_id` and `score` columns (Parquet, CSV, a TREC run or JSONL
work), and score it against a suite. The subsets of a suite share query ids, so a file that ranks several subsets
names each row's subset in a `dataset` column; without it, `eval score` refuses the file. A TREC run has no such
column and holds one subset: score it with `--subset <name>` (in Python, `load_rankings(path, dataset="<name>")`).

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

### Re-judge a pool with your own endpoint

Serve a judge behind an OpenAI-compatible endpoint, estimate the calls and tokens, then run:

```bash
rcp-ndcg run start rejudge_nfcorpus --judge-url http://localhost:8000/v1 --judge-model my-model --estimate
rcp-ndcg run start rejudge_nfcorpus --judge-url http://localhost:8000/v1 --judge-model my-model
```

[Calibrate your benchmark](tutorials/calibrate-your-benchmark.md) walks through a run, and
[primitives](concepts/primitives.md) shows how to re-judge only some documents, insert new ones, or add a second
judge without refitting a published calibration.

### Reproduce a table of the paper

```bash
pip install -r experiments/requirements.txt
python experiments/fetch_data.py
python experiments/run_all.py
```

[Reproduce the paper](tutorials/reproduce-the-paper.md) says what the scripts check and what they do not cover.

## The metric alone

With calibrated abilities and item parameters at hand, the metric is a pure function of the core package:

```python
from rcp_ndcg_core import gain, ndcg

items = {"gamma": [1.2, 1.0, 1.1, 0.9, 0.8], "beta": [-2.0, -0.5, 0.0, 0.8, 1.7]}
gains = {doc_id: gain(theta, items) for doc_id, theta in {"d1": 1.4, "d2": -0.3, "d3": 0.6}.items()}
print(ndcg({"d1": 0.91, "d2": 0.75, "d3": 0.20}, gains, k=10))
```

[The metric](concepts/metric.md) defines it.
