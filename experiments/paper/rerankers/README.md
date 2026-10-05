# Reranker configs

Each YAML is a `rcp_ndcg.retrieval.RerankerConfig`; pass it to `rcp-ndcg retrieval rerank --reranker <file>`
and override a field with `--set KEY=VALUE`.

The `api` says which wire the model speaks, and each one takes only its own fields:

| `api` | where | fields |
| --- | --- | --- |
| `rerank` | a served `/rerank` endpoint, e.g. `vllm serve <model> --runner pooling`, also for late-interaction checkpoints | `base_url` (the server root; `None` only when the run's job starts the engine with `serve.reranker`), `model` (the id the server was started with), `revision` (recorded), `recipe` (the serving recipe the engine runs), `tokenizer` (the declared tokenizer, for the text budget), `max_tokens`, `query_max_tokens`, `instruction` (`fold` | `field` | `none`), `use_activation`, `listwise`, `api_key_env`, `concurrency` (requests in flight), `timeout_s`, `connect_timeout_s`, `max_retries` |
| `cohere`, `voyage` | the vendor's public API | `model`, `api_key_env` (default `CO_API_KEY`/`COHERE_API_KEY`, `VOYAGE_API_KEY`), `base_url` (a proxy), `batch_size` (documents per request; Cohere 100, Voyage 20), `timeout_s`, `connect_timeout_s`, `max_retries` |

A field the wire does not take is refused, never ignored: a hosted reranker takes no `instruction: field` and no
`use_activation`, and a `listwise` model takes no `batch_size` (it always scores the whole candidate set in one
prompt).

A served reranker needs no GPU on the client:

```bash
vllm serve <reranker-model> --runner pooling --max-model-len 32768 &
rcp-ndcg retrieval rerank --dataset <uri> --rankings fused.parquet --reranker my_reranker.yaml \
  --out reranked.parquet
```

## The paper's models

The paper's in-process rerankers are served now (the package carries no in-process model code). Each config names
the recipe that serves it (`recipe:`; the recipes live in `packages/rcp-ndcg-vllm`, with the paper's exact scoring
kept under `reference/` for the equivalence check) — the id is the checkpoint's lowercased Hub repo name, never a
short redirect (`zerank-1-reranker`, not `zerank-1`) — its checkpoint's `tokenizer` (whose SHA-256 keys the rerun's
resumes), and the paper's budgets (`max_tokens: 8192`, `query_max_tokens: 4096`). Their `base_url` is a
placeholder: a `serve.reranker` engine replaces it at runtime through `RCP_NDCG_ENGINES`, or pass your own with
`--set base_url=...`. Until the text-budget mechanism wires the clients, a config that sets `max_tokens` is
refused where the client is built (a budget is never silently ignored); validate the configs with
`rcp-ndcg schema`-style tooling or the test suite in the meantime.
