# Recipes and serving models

Exact names on the serving surface. The catalog of the 13 shipped families and their 19 variants (the
canonical variant ids, the model, the role, the input, the plugin and the status of every row) is the table in
the `rcp-ndcg-vllm` README -- the distribution's PyPI page, and the one rendered copy. This page documents
what the rows and the surface mean. [Serve a retrieval model](../how-to/serve-a-model.md) walks through using
one; [add a serving recipe](../how-to/add-a-model.md) writes one.

## The `rcp-ndcg-vllm` distribution

The recipes are package data, grouped into **families** (decision 34): `recipes/<family>/family.yaml` holds the
shared blocks and the `variants` table, with the family's ONE `reference.py` (parameterised by the variant), its
one chat template where the model needs one, and its `requirements-reference.txt`. Every variant resolves to a
full `Recipe` (the unchanged recipe schema) and is served, contract-tested, stage-1-tested and GPU-validated on
its own; a family id is never served. The public names are `rcp_ndcg_vllm.recipe`'s `Family`, `Variant`,
`Recipe`, `load_family`, `load_recipe`, `resolve_recipe`, `iter_families`, `iter_recipes` and `serve_argv` (the
serve-argv builder), the `rcp-ndcg-vllm` console tree (`serve`, with `--dry-run`), and the exported schemas
(`schema/recipe.schema.json` for a resolved recipe, `schema/family.schema.json` for a family file); everything
else in the package is internal.

## The shipped families

| family | variants | role | input |
|---|---|---|---|
| `qwen3-embedding` | `qwen3-embedding-0.6b` | embed | text |
| `qwen3-vl-embedding` | `qwen3-vl-embedding-2b` | embed | text, image, video |
| `jina-embeddings-v5-text` | `jina-embeddings-v5-text-small` | embed | text |
| `octen-embedding` | `octen-embedding-8b` | embed | text |
| `zembed-1` | `zembed-1-embedding` | embed | text |
| `pplx-embed-v2-context` | `pplx-embed-v2-context-9b-preview` | multi_vector | text |
| `pplx-embed-v2-late` | `pplx-embed-v2-late-0.6b` | multi_vector | text, image |
| `topk-embed-v1` | `topk-embed-v1-small` | multi_vector | text, image |
| `qwen3-reranker` | `qwen3-reranker-0.6b`, `-4b`, `-8b` | rerank | text |
| `qwen3-vl-reranker` | `qwen3-vl-reranker-2b` | rerank | text, image |
| `zerank` | `zerank-1-reranker`, `zerank-1-small-reranker`, `zerank-2-reranker` | rerank | text |
| `ctxl-rerank-v2-instruct-multilingual` | `-1b`, `-2b`, `-6b` | rerank | text |
| `jina-reranker-v3` | `jina-reranker-v3` | rerank | text |

The README's table is the one rendered catalog copy with every variant's model, plugin and status.

`rcp-ndcg-vllm serve <recipe-id> [--port PORT] [--dry-run]` builds the `vllm serve` argv from the recipe's
package data (the chat template file path, the media flags, the pooler config) and runs it; `--dry-run` prints
the argv and exits. A checkpoint that needs its model plugin is refused with the exact install line: the
`topk-embed-v1-small` and the two pplx checkpoints fold into `rcp_ndcg_vllm/models/` under one lazy
`vllm.general_plugins` entry point (importing `rcp_ndcg_vllm` never imports torch or vLLM).

## The catalog's columns

- `id` -- the variant's recipe id: the lowercased canonical Hub repository name of the model, the
  `--served-model-name` the engine serves and what `recipe: <id>` resolves. A family's id names the directory
  and is never served.
- `model` -- the checkpoint's Hub repository, pinned by the variant's `revision` inside `family.yaml`.
- `role` -- `embed`, `multi_vector` or `rerank`: which role client reads the served model.
- `input` -- `text`, `image`, `video`: what the checkpoint reads.
- `plugin` -- the model plugin the checkpoint needs on the stock engine, when one.
- `status` -- `status.state` from the variant's own row in `family.yaml` (the family's until a variant
  declares its own): `unverified` (written, not yet checked), `verified` (the harness passed every gate) or
  `failed`, with the engine `image`, the `date` and the report recorded beside it.

## Budgets

Every recipe declares a tokenizer (injected as `model@revision` unless the family pins one),
`client.max_tokens`, `query_max_tokens` where the reference caps queries, and (a reranker)
`document_max_tokens` where the checkpoint cuts each document itself; over-budget content is cut client-side at token boundaries with the template's anchors preserved, and
every cut is recorded ([text budgets for served roles](../concepts/text-budgets.md)). No limits are repeated
here -- the recipe file is the source.

## Equivalence policy

Served recipes use anchor-preserving cuts. The paper's code cut without regard to where the model reads its
answer; a recipe whose reference drops anchors declares `reference.known_deviations: [anchor_drop_over_cap]`,
and the equivalence gates compare those models under the cap only. Only an over-cap document can end at a
different place.

## `recipe:` in `rcp-ndcg`

A role config that names `recipe: <id>` takes its whole client block (api, tokenizer, budgets, template, media,
instruction mode) from the recipe. `base_url` and the other RUNTIME fields stay on the config; any CONTENT
field set explicitly must equal the recipe's, or the config is refused with a `ConfigError` naming both values.
On the command line `--retriever recipe:<id>` and `--reranker recipe:<id>` expand to that mapping, with the URL
from `--set ...base_url=...` or a `serve:` engine. Resolution is lazy through `rcp_ndcg_vllm`, so
`rcp-ndcg-vllm` must be installed alongside `rcp-ndcg` (there is no extra alias for it); without it the refusal
is typed and its hint is the install line.
