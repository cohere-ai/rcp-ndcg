# rcp-ndcg-vllm

The serving half of RCP-nDCG: the vetted serving recipes for retrieval models, the `rcp-ndcg-vllm serve`
command that turns one into a `vllm serve` command for the stock `vllm/vllm-openai` image, and the model
plugins that make two released checkpoints serveable on it. The engine is reached over HTTP only; this
package never imports `rcp-ndcg`, torch or vLLM at import time (its dependencies are pydantic and PyYAML).
A recipe's `client` block is plain data, validated when `rcp-ndcg` reads it.

## Install

Three contexts, three lines:

- **The engine environment** (the stock vLLM image): `python3 -m pip install --no-deps rcp-ndcg-vllm` -- the
  wheel's only dependencies are pydantic and PyYAML, which the image ships; a `pip freeze` before and after
  differs by exactly this wheel.
- **The client environment**: `pip install rcp-ndcg rcp-ndcg-vllm` -- the second package resolves `recipe:<id>`
  and runs `serve`.
- **Recipe data only**: `import rcp_ndcg_vllm` reads the recipes; it never imports vLLM or torch.

## Serve a recipe

```bash
rcp-ndcg-vllm serve <recipe-id> [--port PORT] [--dry-run]
```

builds the `vllm serve` argv from the recipe's package data (the chat template file path, the media flags, the
pooler config) and runs it (`--dry-run` prints the argv and exits). A checkpoint that needs its model plugin is
refused with the exact install line. Example, on the stock image:

```bash
python3 -m pip install --no-deps rcp-ndcg-vllm
rcp-ndcg-vllm serve qwen3-embedding-0.6b --port 8000
```

The [recipe guide](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/how-to/serve-a-model.md) walks through
the client side (`recipe: <id>` in the retriever config), and [validate a recipe on
GPUs](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/how-to/validate-a-recipe.md) through the waves every
recipe passes before the release.

## The recipes

Every recipe in this table is validated end to end on GPU against its reference implementation before v0.0.1
(equivalence, quality and end-to-end waves); each `recipe.yaml` records its model revision and `sources`, and
`status.state` (`unverified`, `verified`, `failed`) records the outcome beside the engine `image`, the `date`
and the report. The states below are copied from each recipe's `status.state`; the tag ships none unverified.

| id | model | role | input | plugin | status |
|---|---|---|---|---|---|
| `qwen3-embedding-0.6b` | Qwen/Qwen3-Embedding-0.6B | embed | text | — | unverified |
| `qwen3-vl-embedding-2b` | Qwen/Qwen3-VL-Embedding-2B | embed | text, image, video | — | unverified |
| `jina-embeddings-v5-text-small` | jinaai/jina-embeddings-v5-text-small | embed | text | — | unverified |
| `octen-embedding-8b` | Octen/Octen-Embedding-8B | embed | text | — | unverified |
| `zembed-1-embedding` | zeroentropy/zembed-1-embedding | embed | text | — | unverified |
| `pplx-embed-v2-context-9b-preview` | perplexity-ai/pplx-embed-v2-context-9b-preview | multi_vector | text | the pplx model plugin | unverified |
| `topk-embed-v1-small` | topk-io/topk-embed-v1-small | multi_vector | text, image | the topk model plugin | unverified |
| `qwen3-reranker-0.6b` | Qwen/Qwen3-Reranker-0.6B | rerank | text | — | unverified |
| `qwen3-reranker-4b` | Qwen/Qwen3-Reranker-4B | rerank | text | — | unverified |
| `qwen3-reranker-8b` | Qwen/Qwen3-Reranker-8B | rerank | text | — | unverified |
| `qwen3-vl-reranker-2b` | Qwen/Qwen3-VL-Reranker-2B | rerank | text, image | — | unverified |
| `zerank-1-reranker` | zeroentropy/zerank-1-reranker | rerank | text | — | unverified |
| `zerank-1-small-reranker` | zeroentropy/zerank-1-small-reranker | rerank | text | — | unverified |
| `zerank-2-reranker` | zeroentropy/zerank-2-reranker | rerank | text | — | unverified |
| `ctxl-rerank-v2-instruct-multilingual-1b` | ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b | rerank | text | — | unverified |
| `ctxl-rerank-v2-instruct-multilingual-2b` | ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b | rerank | text | — | unverified |
| `ctxl-rerank-v2-instruct-multilingual-6b` | ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b | rerank | text | — | unverified |
| `jina-reranker-v3` | jinaai/jina-reranker-v3 | rerank | text | — | unverified |

The `id` is the lowercased canonical Hub repository name; the `role` is what `rcp-ndcg` reads through it
(`embed`, `multi_vector`, `rerank`); the `input` is what the checkpoint reads. Budgets are explicit per recipe:
every recipe declares `client.tokenizer`, `client.max_tokens` and (where the reference caps queries)
`query_max_tokens`; over-budget content is cut client-side at token boundaries with the template's anchors
preserved, and every cut is recorded.

## Use a recipe from `rcp-ndcg`

A role config that names `recipe: <id>` takes its whole client block (api, tokenizer, budgets, template, media,
instruction mode) from the recipe; `base_url` and the other run-time fields stay on the config, and an explicit
content field must equal the recipe's or the config is refused naming both values. `--retriever recipe:<id>` and
`--reranker recipe:<id>` are command-line shorthands. Recipes are resolved lazily, so `rcp-ndcg-vllm` must be
installed beside `rcp-ndcg`; without it the refusal is typed and its hint is the install line
([recipes and serving models](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/reference/recipes.md)).

## Model plugins

`topk-embed-v1-small` and `pplx-embed-v2-context-9b-preview` fold into `rcp_ndcg_vllm/models/` under one
`vllm.general_plugins` entry point. Registration is lazy (`"module:Class"` strings): importing this package
never imports vLLM or torch. A version guard pins the tested vLLM line and refuses others loudly.

## Validation

The equivalence harness, the recorder, the reference cases and the GPU job tooling live in the unpublished
`rcp-ndcg-test` package ([its page](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/reference/rcp-ndcg-test.md));
authors of new recipes start at [add a serving
recipe](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/how-to/add-a-model.md). The judging pipeline and
the metric are `rcp-ndcg` and `rcp-ndcg-core`.

## License

Apache-2.0: `LICENSE` and `NOTICE` ship in every wheel and sdist.
Security reports: [SECURITY.md](https://github.com/cohere-ai/rcp-ndcg/blob/main/SECURITY.md) (GitHub private
vulnerability reporting).
