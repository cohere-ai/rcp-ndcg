# Serve a retrieval model

The shipped retrieval models -- 18 of them, one recipe per checkpoint -- run on the stock `vllm/vllm-openai`
image. This walk-through serves one, points `rcp-ndcg` at it and scores what it retrieves. The rules of the
recipes are in [recipes and serving models](../reference/recipes.md); hosted APIs (Cohere, Voyage, Gemini)
need none of this -- [embedding endpoints](../concepts/embeddings.md) covers them.

## 1. The engine host: one wheel on the stock image

The serving recipes and the `rcp-ndcg-vllm serve` command ship in the `rcp-ndcg-vllm` distribution -- the lean
serving package (its only dependencies are pydantic and PyYAML). Installing it with `--no-deps` is the one
change the engine environment takes; a `pip freeze` before and after differs by exactly this wheel.

```bash
python3 -m pip install --no-deps rcp-ndcg-vllm==<version>
rcp-ndcg-vllm serve qwen3-embedding-0.6b --port 8000
```

`rcp-ndcg-vllm serve <recipe-id> [--port PORT] [--dry-run]` builds the `vllm serve` argv from the recipe's
package data -- the chat template file path, the media flags, the pooler config -- and runs it. `--dry-run`
prints the argv and exits, to inspect what would run. A checkpoint that needs its model plugin is refused with
the exact install line.

Two options when the image does not carry the wheel: bake it into your image (image preparation), or install
it inline where the engine command runs:

```bash
bash -lc 'python3 -m pip install --no-deps rcp-ndcg-vllm==<version> && exec rcp-ndcg-vllm serve <id> --port 8000'
```

The same two forms work in a run config's `serve:` block, whose `command` is yours verbatim ([runs and job
runners](../concepts/runs.md#starting-the-engines-with-the-run)).

## 2. The client host: `recipe:` in the retriever config

The client side needs `rcp-ndcg` with `rcp-ndcg-vllm` installed alongside it (no extra alias): a role config
naming `recipe: <id>` takes its whole client block -- api, tokenizer, budgets, template, media, instruction
mode -- from the recipe, and `base_url` (and the other run-time fields) stays on the config. Any content field
set explicitly must equal the recipe's, or the config is refused with an error naming both values.

```yaml
# retriever.yaml
kind: dense
encoder:
  recipe: qwen3-embedding-0.6b      # the recipe's client block: api, tokenizer, budgets, template, media
  base_url: http://127.0.0.1:8000/v1  # run-time fields stay here
```

Budgets are explicit in every recipe: over-budget content is cut client-side at token boundaries with the
template's anchors preserved, and every cut is recorded -- never an engine-side truncation
([text budgets](../concepts/text-budgets.md)).

```bash
rcp-ndcg retrieval index --dataset jsonl:tiny/rows.jsonl --retriever retriever.yaml --out index/
rcp-ndcg retrieval search --dataset jsonl:tiny/rows.jsonl --retriever retriever.yaml --out rankings.parquet
rcp-ndcg eval score --rankings rankings.parquet --suite nanobeir      # or --dataset jsonl:tiny/rows.jsonl
```

The command line also takes the shorthand `--retriever recipe:qwen3-embedding-0.6b` (and `--reranker
recipe:<id>`), which expands to the mapping above with the URL from `--set retriever.encoder.base_url=...` or
from a `serve:` engine. `rcp-ndcg retrieval rerank` re-scores the pool with a `recipe:` reranker the same way
([retrieval and reranking](../concepts/retrieval.md)), and a run's `retrieve`, `rerank` steps carry the same
configs.

## See also

- [Add a serving recipe](add-a-model.md) -- the recipe format, and its checks before any GPU time.
- [Validate a recipe on GPUs](validate-a-recipe.md) -- the waves a recipe passes before the release.
- [Recipe authors' reference](../reference/recipes.md) -- the catalog and every name.