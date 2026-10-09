# The paper's configurations

The exact settings behind the paper's judgements and runs, kept outside the installed package: the package ships
engine-agnostic judge presets (named with `--judge`) and talks to any OpenAI-compatible URL, while
this folder records which retriever and reranker produced each result of
[arXiv:2609.35739](https://arxiv.org/abs/2609.35739). The paper's judges ran on SGLang; the paper's submission
code is the record of those engine commands, and this release serves the same checkpoints on vLLM v0.31.0
([serving](../../docs/concepts/judges.md)). Use it from a checkout of the repository.

| Folder | What it holds | Reproduces |
|---|---|---|
| `retrieval/` | The three first-stage retrievers: BM25 (`bm25s.yaml`), Octen-Embedding-8B (`octen.yaml`), Cohere Embed v4 (`cohere_embed_v4.yaml`) | The candidate pools of the NanoBEIR, BRIGHT and ViDoRe v3 leaderboards |
| `rerankers/` | The 14 rerankers of the leaderboards, one config each ([README](rerankers/README.md)) | The reranker rows of the leaderboards |

The re-judging run configs are packaged with the library as named configs (`rcp-ndcg run start
rejudge_nfcorpus`), and the tables themselves are recomputed from the public data by the scripts one level up
([experiments/README.md](../README.md)), with no LLM call.

## Serving a judge as the paper did

The paper's judges ran on SGLang, served on the GPU host at `http://127.0.0.1:8000/v1` under the name the judge
config expects; the paper's submission code is the record of those engine commands, and this repository does not
ship them. This release serves the same checkpoints on vLLM v0.31.0: the flags are in
[serving a judge](../../docs/concepts/judges.md). The re-judging command is unchanged:

```bash
rcp-ndcg judge tournament --dataset <uri> --judge qwen35_397b_nvfp4 --out store/
```

For a job runner, put the engine image and its command into a run config's `serve:` section under the judge role
([serving](../../docs/concepts/runs.md#starting-the-engines-with-the-run)).

The paper's third judge, Qwen3.6-27B in its FP8 release (TREC-DL), has no recorded engine settings beyond its
served name and reasoning parser: `vllm serve Qwen/Qwen3.6-27B-FP8 --served-model-name qwen3.6-27b-fp8
--reasoning-parser qwen3`.

## Retrievers and rerankers

```bash
rcp-ndcg retrieval index --dataset <uri> --retriever experiments/paper/retrieval/bm25s.yaml --out index/
rcp-ndcg retrieval rerank --dataset <uri> --rankings fused.parquet \
  --reranker experiments/paper/rerankers/qwen3_reranker_8b.yaml --out reranked.parquet
```
