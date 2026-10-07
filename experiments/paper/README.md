# The paper's configurations

The exact settings behind the paper's judgements and runs, kept outside the installed package: the package ships
engine-agnostic judge presets (named with `--judge`) and talks to any OpenAI-compatible URL, while
this folder records which engine, image, retriever and reranker produced each result of
[arXiv:2609.35739](https://arxiv.org/abs/2609.35739). Use it from a checkout of the repository.

| Folder | What it holds | Reproduces |
|---|---|---|
| `serve/` | The engine command the paper ran for each judge, SGLang with its image pinned (`<judge>.sglang.sh`); the equivalent vLLM flags are in [serving](../../docs/concepts/runs.md#starting-the-engines-with-the-run) | The Stage A and Stage B judgements of Qwen3.5-397B (NanoBEIR, BRIGHT, ViDoRe v3; text only) and gpt-oss-120b (NanoBEIR, BRIGHT, TREC-DL) |
| `retrieval/` | The three first-stage retrievers: BM25 (`bm25s.yaml`), Octen-Embedding-8B (`octen.yaml`), Cohere Embed v4 (`cohere_embed_v4.yaml`) | The candidate pools of the NanoBEIR, BRIGHT and ViDoRe v3 leaderboards |
| `rerankers/` | The 14 rerankers of the leaderboards, one config each ([README](rerankers/README.md)) | The reranker rows of the leaderboards |

The re-judging run configs are packaged with the library as named configs (`rcp-ndcg run start
rejudge_nfcorpus`), and the tables themselves are recomputed from the public data by the scripts one level up
([experiments/README.md](../README.md)), with no LLM call.

## Serving a judge as the paper did

Each script starts the judge's engine on the GPU host with Docker and serves it at `http://127.0.0.1:8000/v1`
under the name the judge preset expects:

```bash
experiments/paper/serve/qwen35_397b_nvfp4.sglang.sh            # the primary judge, eight Blackwell GPUs
rcp-ndcg judge tournament --dataset <uri> --judge qwen35_397b_nvfp4 --out store/
```

`IMAGE=<tag>` overrides the pinned image. For a job runner, copy the script's image and its `ENGINE` array into a
run config's `serve:` section under the judge role ([serving](../../docs/concepts/runs.md#starting-the-engines-with-the-run)):

```yaml
judge: qwen35_397b_nvfp4
serve:
  judge:
    image: lmsysorg/sglang:v0.5.17-cu129
    command: [python3, -m, sglang.launch_server, --model-path, nvidia/Qwen3.5-397B-A17B-NVFP4, ...]
    resources: {gpus: 8}
```

The paper's third judge, Qwen3.6-27B in its FP8 release (TREC-DL), has no recorded engine settings beyond its
served name and reasoning parser: `vllm serve Qwen/Qwen3.6-27B-FP8 --served-model-name qwen3.6-27b-fp8
--reasoning-parser qwen3`, or the same flags on SGLang.

## Retrievers and rerankers

```bash
rcp-ndcg retrieval index --dataset <uri> --retriever experiments/paper/retrieval/bm25s.yaml --out index/
rcp-ndcg retrieval rerank --dataset <uri> --rankings fused.parquet \
  --reranker experiments/paper/rerankers/qwen3_reranker_8b.yaml --out reranked.parquet
```
