# Recipes and serving models

Exact names on the serving surface. The catalog of the 16 shipped families and their 30 variants (the
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
`Recipe`, `RecipeFieldRole`, `FieldSpec`, `FIELD_ROLES`, `load_family`, `load_recipe`, `resolve_recipe`,
`iter_families`, `iter_recipes`,
`serve_argv` (the serve-argv builder), `deployment_fields`, `parse_deployment_overrides` and `recipe_digest`,
the `rcp-ndcg-vllm` console tree (`serve`, with `--variant`, `--port`, `--set` and `--dry-run`), and the
exported schemas (`schema/recipe.schema.json` for a resolved recipe, `schema/family.schema.json` for a family
file); everything else in the package is internal (``RecipeError``, the typed refusal every loader raises,
lives in `rcp_ndcg_vllm.errors` and is re-exported from the package root).

## The shipped families

| family | variants | role | input |
|---|---|---|---|
| `qwen3-embedding` | `qwen3-embedding-0.6b` | embed | text |
| `qwen3-vl-embedding` | `qwen3-vl-embedding-2b` | embed | text, image, video |
| `embeddinggemma-2` | `embeddinggemma-2` | embed | text, image, video |
| `jina-embeddings-v5-text` | `jina-embeddings-v5-text-nano`, `-small` | embed | text |
| `harrier-oss-v1` | `harrier-oss-v1-270m`, `-0.6b`, `-27b` | embed | text |
| `octen-embedding` | `octen-embedding-0.6b`, `-4b`, `-8b` | embed | text |
| `zembed-1` | `zembed-1-embedding` | embed | text |
| `pplx-embed-v1` | `pplx-embed-v1-0.6b`, `pplx-embed-v1-4b` | embed | text |
| `pplx-embed-v2-context` | `pplx-embed-v2-context-9b-preview` | multi_vector | text |
| `pplx-embed-v2-late` | `pplx-embed-v2-late-0.6b`, `pplx-embed-v2-late-9b` | multi_vector | text, image |
| `topk-embed-v1` | `topk-embed-v1-xsmall`, `-small` | multi_vector | text, image |
| `qwen3-reranker` | `qwen3-reranker-0.6b`, `-4b`, `-8b` | rerank | text |
| `qwen3-vl-reranker` | `qwen3-vl-reranker-2b` | rerank | text, image |
| `zerank` | `zerank-1-reranker`, `zerank-1-small-reranker`, `zerank-2-reranker` | rerank | text |
| `ctxl-rerank-v2-instruct-multilingual` | `-1b`, `-2b`, `-6b` | rerank | text |
| `jina-reranker-v3` | `jina-reranker-v3` | rerank | text |

The README's table is the one rendered catalog copy with every variant's model, plugin and status.

`rcp-ndcg-vllm serve <recipe-id> [--variant VARIANT-ID] [--port PORT] [--set PATH=VALUE ...] [--dry-run]`
builds the `vllm serve` argv from the recipe's package data (the chat template file path, the media flags, the
pooler config) and runs it; `--dry-run` prints the argv, the recipe's identity and the applied overrides, and
exits. `--set` names a **deployment** field of the recipe -- the engine's resource, scheduling and address
knobs (`resources.gpus`, `serve.gpu_memory_utilization`, `serve.max_num_seqs`,
`serve.max_num_batched_tokens`, `serve.host`, `serve.port`, `serve.max_model_len`) -- and the schema declares
that surface once (`rcp_ndcg_vllm.recipe.FIELD_ROLES`): a content field is refused by name, and
`serve.max_model_len` is refused below the client's largest token budget, with both numbers named. A checkpoint
that needs its model plugin is refused with the exact install line: the `topk-embed-v1-small`, `topk-embed-v1-xsmall` and the
pplx checkpoints fold into `rcp_ndcg_vllm/models/` under one lazy `vllm.general_plugins` entry point (the pplx
plugin registers the pplx-embed-v1 family's local config class and serves the v2 contextual and both
late-interaction sizes; importing `rcp_ndcg_vllm` never imports torch or vLLM).

`serve` and `recipe:` also take a **family directory of the operator's own** (`./my-family/`, with
`--variant <id>` for one size of several): the same schema validates it, families included, and every record
marks it unshipped with `status: unverified`. Its identity is the content hash of its resolved form --
`unshipped:sha256:<hex>`, the referenced chat template file included -- never a shipped id, so two runs whose
files differ never share a run identity; a config that records that identity is read back as it stands (a
run's resume, an index reload).

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

- `id` -- the variant's recipe id: the lowercased canonical Hub repository name of the model, the
  `--served-model-name` the engine serves and what `recipe: <id>` resolves. A family's id names the directory
  and is never served.
- `model` -- the checkpoint's Hub repository, pinned by the variant's `revision` inside `family.yaml`.
- `role` -- `embed`, `multi_vector` or `rerank`: which role client reads the served model.
- `input` -- `text`, `image`, `video`: what the checkpoint reads.
- `mrl` -- the variant's declared Matryoshka head: `truncation` or `projection` with the model card's
  supported output dimensions (a discrete table or a prose range), or `none`. It is declared once in the
  recipe's client block (`mrl_kind` with `mrl_dims`/`mrl_range`; a projection kind also names
  `mrl_projection`, the checkpoint's learned matrices) and, where the engine's per-request `dimensions`
  path exists, mirrored in `serve.hf_overrides` (`is_matryoshka`/`matryoshka_dimensions`); the loader
  refuses a serve gate and a client declaration that disagree. The recipes ship the checkpoint's full
  width and a run selects `k` from the declared set.
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
One exception is the MRL selection (owner decision 39): the recipe declares the Matryoshka kind and the
card's set once, so a config's `mrl_dim` (the client head) or `dimensions` (the engine-side cut) is
accepted when `k` is in the declared `mrl_dims`/`mrl_range` and refused naming the set otherwise -- the
recipes ship the checkpoint's full width and nothing is selected unless it is declared.
The `recipe` pointer itself is replaced by the recipe's identity (its shipped id, or `unshipped:sha256:<hex>`
for a file of the operator's own), so a run identity follows the file's content, never the spelling of a path.
On the command line `--retriever recipe:<id-or-path>` and `--reranker recipe:<id-or-path>` expand to that
mapping, with the URL from `--set ...base_url=...` or a `serve:` engine; `recipe:./my-family/` and
`recipe:/abs/path` load the family directory through the same schema (a multi-variant directory names its
variants in the refusal). Resolution is lazy through `rcp_ndcg_vllm`, so `rcp-ndcg-vllm` must be installed
alongside `rcp-ndcg` (there is no extra alias for it); without it the refusal is typed and its hint is the
install line.
