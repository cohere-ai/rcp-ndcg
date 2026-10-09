# Recipes and serving models

Exact names on the serving surface. The catalog of the 19 shipped recipes (the canonical ids, the model, the
role, the input, the plugin and the status of every row) is the table in the `rcp-ndcg-vllm` README -- the
distribution's PyPI page, and the one rendered copy. This page documents what the rows and the surface mean.
[Serve a retrieval model](../how-to/serve-a-model.md) walks through using one; [add a serving
recipe](../how-to/add-a-model.md) writes one.

## The `rcp-ndcg-vllm` distribution

The recipes are package data (`recipes/<id>/`: the `recipe.yaml`, the chat template a model needs, its
`reference.py`), read through `importlib.resources`. The public names are `rcp_ndcg_vllm.recipe`'s `Recipe`,
`load_recipe`, `iter_recipes` and `serve_argv` (the serve-argv builder), the `rcp-ndcg-vllm` console tree
(`serve`, with `--dry-run`), and the exported recipe schema in `schema/recipe.schema.json`; everything else in
the package is internal.

`rcp-ndcg-vllm serve <recipe-id> [--port PORT] [--dry-run]` builds the `vllm serve` argv from the recipe's
package data (the chat template file path, the media flags, the pooler config) and runs it; `--dry-run` prints
the argv and exits. A checkpoint that needs its model plugin is refused with the exact install line: the
`topk-embed-v1-small` and the two pplx checkpoints fold into `rcp_ndcg_vllm/models/` under one lazy
`vllm.general_plugins` entry point (importing `rcp_ndcg_vllm` never imports torch or vLLM).

## Engine-side patches

A recipe whose admissible prompts can reach its declared `max_model_len` under chunked prefill may need an
engine-side fix the stock image predates. The engine applies such a fix only when the engine process's
`RCP_NDCG_VLLM_PATCHES` names it -- a comma-separated list read by the one `vllm.general_plugins` entry point
(`rcp-ndcg-vllm serve` passes its environment through). One patch ships:

- `pooling-full-context` -- the backport of vllm-project/vllm#48039 (commit `e6fc81bc78`): at vLLM v0.31.0 the
  scheduler reserves one sampled-token slot for pooling requests too, so a prompt of exactly `max_model_len`
  tokens under chunked prefill never schedules its last token and the request hangs. The patch stores
  `num_sampled_tokens_per_step = 0` for the pooling runner only; it logs one line when it applies and one
  inert line when the running vLLM already carries the fix. Delete the patch when `engine.image` moves to the
  first vLLM release that carries `e6fc81bc78`.

## The catalog's columns

- `id` -- the recipe's name and directory: the lowercased canonical Hub repository name of the model. It is
  also the `--served-model-name` the engine serves.
- `model` -- the checkpoint's Hub repository, pinned by the `revision` inside `recipe.yaml`.
- `role` -- `embed`, `multi_vector` or `rerank`: which role client reads the served model.
- `input` -- `text`, `image`, `video`: what the checkpoint reads.
- `plugin` -- the model plugin the checkpoint needs on the stock engine, when one.
- `status` -- `status.state` from the recipe's own `recipe.yaml`: `unverified` (written, not yet checked),
  `verified` (the harness passed every gate) or `failed`, with the engine `image`, the `date` and the report
  recorded beside it.

## Budgets

Every recipe declares `client.tokenizer`, `client.max_tokens`, `query_max_tokens` where the reference caps
queries, and (a reranker) `document_max_tokens` where the checkpoint cuts each document itself; over-budget content is cut client-side at token boundaries with the template's anchors preserved, and
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
