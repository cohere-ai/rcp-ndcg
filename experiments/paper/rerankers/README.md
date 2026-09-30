# Reranker configs

Each YAML is a `rcp_ndcg.retrieval.RerankerConfig`; pass it to `rcp-ndcg retrieval rerank --reranker <file>`
and override a field with `--set KEY=VALUE`.

The `provider` says where the model runs, and each provider takes only its own fields:

| `provider` | where | fields |
| --- | --- | --- |
| `local` | in process (the `[local]` extra): Qwen3-Reranker, ZeRank, ctxl-rerank, Jina reranker v3, picked from the model id | `model`, `revision`, `batch_size` (documents per forward pass; Qwen3-Reranker and ctxl-rerank, default 8; ZeRank and Jina batch on their own and refuse it) |
| `openai_compatible` | a served `/rerank` endpoint, e.g. `vllm serve <model> --runner pooling`, also for late-interaction checkpoints | `base_url` (the server root, required), `model` (the id the server was started with), `revision` (recorded), `api_key_env`, `concurrency` (requests in flight, default 8), `timeout_s`, `connect_timeout_s`, `max_retries` |
| `cohere`, `voyage` | the vendor's public API | `model`, `api_key_env` (default `CO_API_KEY`/`COHERE_API_KEY`, `VOYAGE_API_KEY`), `base_url` (a proxy), `batch_size` (documents per request; Cohere 100, Voyage 20), `timeout_s`, `connect_timeout_s`, `max_retries` |

A field the provider does not use is refused, never ignored.

A served reranker needs no GPU on the client:

```bash
vllm serve <reranker-model> --runner pooling --max-model-len 32768 &
rcp-ndcg retrieval rerank --dataset <uri> --rankings fused.parquet --reranker my_reranker.yaml \
  --set provider=openai_compatible --set base_url=http://localhost:8000 --out reranked.parquet
```
