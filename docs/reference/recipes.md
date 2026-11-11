# Recipes and serving models

This page gives the exact names of the serving surface. The catalog has 24 shipped families and 44 variants. The table
in the [`rcp-vllm` README](https://github.com/cohere-ai/rcp-ndcg/blob/main/packages/rcp-vllm/README.md) holds the
canonical variant ids. It gives the model, the role, the input and the Matryoshka head of each variant. It also gives
the plugin, the reference and the status of every row. That README is the PyPI page of the distribution and the one
rendered copy. This page documents what the rows and the surface mean.
[Serve a retrieval model](../how-to/serve-a-model.md) walks through the use of one recipe, and
[add a serving recipe](../how-to/add-a-model.md) writes one.

## The `rcp-vllm` distribution

The recipes are package data, and they are grouped into **families**. `recipes/<family>/family.yaml` holds the shared
blocks and the `variants` table. The family also has ONE `reference.py`, which the variant parameterises, and one chat
template where the model needs one. It has `reference.in` (the justified pins) and the generated `reference.lock` (one
locked, hashed reference environment for each family). Every variant resolves to a full `Recipe`, which is the
unchanged recipe schema. The release serves, contract-tests, stage-1-tests and GPU-validates each variant on its own.
A family id is never served.

The public names are these names of `rcp_vllm.recipe`: `RecipeFamily`, `Variant`, `Recipe`, `RecipeFieldRole`,
`FieldSpec`, `FIELD_ROLES`, `load_family`, `load_recipe` and `resolve_recipe`. The list also holds `iter_families`,
`iter_recipes`, `serve_argv` (the serve-argv builder), `deployment_fields`, `parse_deployment_overrides`,
`recipe_digest` and `plugin_distribution_name`. The public surface also holds the `rcp-vllm` console tree:
`serve`, with `--variant`, `--port`, `--set` and `--dry-run`. It holds the exported schemas too:
`schema/recipe.schema.json` for a resolved recipe, and `schema/family.schema.json` for a family file. Everything else
in the package is internal. `RecipeError` is the typed refusal that every loader raises; it lives in
`rcp_vllm.errors` and is re-exported from the package root.

## The shipped families

There are 24 families: 16 retrieval families (embed, multi_vector and rerank) and 8 judge families. The table in the
README names the variants of every family. It gives their model, role, input, Matryoshka head, plugin, reference and
status.

Every family pins the stock engine image in its `engine` block (`vllm/vllm-openai:v0.33.0`, `min_version` `0.33.0`).
The `status` of every variant reads `verified`, except the two owner-disposed `topk-embed-v1` variants, which read
`failed`. The engine image is one of the inputs of the behaviour fingerprint. No recorded evidence of a recipe
therefore counts for the pinned image until a wave re-records its corpus there. Every row names the wave that did. The
[GPU validation ledger](../reference/gpu-validation.md) is that record. A `status` block keeps the `image`, the `date`
and the `report` of the wave that it names. The release writes `verified` only beside the image that the recipe pins.
The two `topk-embed-v1` variants are the recorded gate failure of the catalog (the stage-2 per-token cosine floor).
Their state returns to `failed` only from a wave that re-runs that gate. They ship only when the explicit disposition
list of the owner names them.

### Known limits, per family

The evidence of these families has a recorded limit or a disposition. The evidence of a family that is not in this
table has none.

| family | limit | evidence |
|---|---|---|
| `topk-embed-v1` (small, xsmall) | `failed`: the stage-2 per-token cosine floor is not met on the available engine configurations. Both bf16 and fp16 fail. The fp32 mode cannot be served, because the GatedDeltaNet kernels of the engine refuse it | the disposition of the owner, the `gate-topk-embed-v1` wave |
| `pplx-embed-v1` (0.6b, 4b) | the engine refuses an empty input, and the reference cannot encode a zero-length sequence. The row is recorded and pruned | `declared_unencodable`, the kind of the row |
| `pplx-embed-v2-late` (0.6b, 9b) | the reference refuses a mixed text+image document (the usage note of the card). The per-token floor of the 9b is 0.9410, and that of the 0.6b is 0.99 | `declared_reference_refusal`; the derivations of the floors in the notes of the family |
| `qwen3-vl-embedding` (2b, 8b) | the reference substitutes the `NULL` of the card for an empty query. The media rows have their own floor (2b 0.9975, 8b 0.9966) | `declared_substituted`; `gates.media_vec_min_cosine` |
| `ctxl-rerank-v2-instruct-multilingual-6b` | serves `float32`, the dtype of the 1b and the 2b. On bf16 one document exceeded the delta gate of the family | the notes of the family; the `equivalence-ctxl` wave |
| `jina-embeddings-v5-text-nano` | the at_budget row (8187 of 8192 tokens) is the minimum. The floor is 0.9964 | the derivation of the floor in the notes of the family |
| `qwen3-embedding` | the MRL-32 head had no margin. The floor is 0.9984, and the 4b keeps 0.999 with 3.5e-4 of margin | the derivation of the floor in the notes of the family |
| `embeddinggemma-2` | the transformers fold is deleted, because its parity is inside the measured bf16 floor. The SDPA patch stays | `DECISION-PARITY.md`, the `sdpa-probe` wave |

The `client` block of a judge family is `rcp.judging.JudgeConfig`. That is the same config that the judging
commands take, and the product validates it. It is never a second model. The `serve` block of the family is the engine
argv, and the family has no equivalence reference. `--judge <id>` and `rcp-vllm serve <id>` take the same id
([judges](../concepts/judges.md)).

`rcp-vllm serve <recipe-id> [--variant VARIANT-ID] [--port PORT] [--set PATH=VALUE ...] [--dry-run]` builds the
`vllm serve` argv from the package data of the recipe, and it runs that argv. The data gives the path of the chat
template file, the media flags and the pooler config. `--dry-run` prints the argv, the identity of the recipe and the
applied overrides, and it exits. `--set` names a **deployment** field of the recipe. These fields are the resource,
scheduling and address knobs of the engine: `resources.gpus`, `serve.gpu_memory_utilization`, `serve.max_num_seqs`,
`serve.max_num_batched_tokens`, `serve.host`, `serve.port` and `serve.max_model_len`. The schema declares that surface
one time (`rcp_vllm.recipe.FIELD_ROLES`). It refuses a content field by name. It also refuses
`serve.max_model_len` below the largest token budget of the client, with both numbers named. The command refuses a
checkpoint that needs its model plugin, and it gives the exact install line. These checkpoints fold into
`rcp_vllm/models/` under one lazy `vllm.general_plugins` entry point. They are the `topk-embed-v1-small`/`-xsmall` (`TopkEmbedModel`) and the pplx checkpoints. The pplx checkpoints are pplx-embed-v1's config-only
`PplxV1Config`, the pplx-embed-v2 contextual and both late-interaction sizes. The last checkpoint is embeddinggemma-2
(`EmbeddingGemma2Config`/`EmbeddingGemma2Model`). Importing `rcp_vllm` never imports torch or vLLM. A recipe that
names a plugin also declares `plugin_architectures`. Those are the architectures that its engine registers, and the
list is empty for a patch-only carrier. The behaviour fingerprint keys the code of the plugin by hashing exactly those
modules (`plugin_sha256.<module>`). The modules are the shared entry modules and the modules of the architecture. The
module of every opted-in patch is also in the hash. The harness refuses a foreign plugin whose modules it cannot
resolve, at fingerprint time and by name.

`serve` and `recipe:` also take a **family directory of the operator's own** (`./my-family/`, with `--variant <id>`
for one size of several). The same schema validates it, families included. Every record marks it unshipped with
`status: unverified`. Its identity is the content hash of its resolved form, `unshipped:sha256:<hex>`. That form
includes the chat template file that it references. The identity is never a shipped id, so two runs with different
files never share a run identity. A config that records that identity is read back as it stands. Examples are a resume
of a run and a reload of an index.

## Engine-side patches

A checkpoint that the stock image cannot serve as shipped can need an engine-side fix. The fix is an architecture that
the registry of the image lacks. It can also be an attention backend that the kernels cannot run. A recipe opts into
such a fix with `serve.patches`, which names the patch (`rcp_vllm.patches.PATCH_NAMES`). Every engine-start path
renders the declared names into `RCP_NDCG_VLLM_PATCHES` of the engine process. These paths are the
`rcp-vllm serve` console, the wave runner and the e2e driver. The variable is a comma-separated list, and the one
`vllm.general_plugins` entry point reads it. The path overrides any inherited value, so the engine runs exactly what
the recipe declares. The provenance of the corpus records the value that the engine ran with. The behaviour
fingerprint hashes the module of every opted-in patch, so a patch fix moves the key of the recipe. One patch ships,
and one recipe opts in: `embeddinggemma-2` declares it, and the `serve.patches` of every other family is empty. The
chunked-prefill patch that a few families used to carry is deleted. The scheduler of the pinned engine keeps the whole
context of a pooling prefill (vllm-project/vllm#48039). The transformers fold of embeddinggemma-2 is deleted too. The
**0.33 image's transformers 5.20.2** ships the `embedding_gemma2` classes of the checkpoint. The registry of the
v0.33.0 tag ships the architecture (`vllm/model_executor/models/registry.py:259`), so the backport was inert there.
The cross-image parity measured cosine 0.999972, inside the measured bf16 floor 0.999882 of the recipe. That parity is
the 0.31 image with the fold active against the 0.33 image with the native model.

- `torch-sdpa-attention` is the backend of embeddinggemma-2. The heterogeneous head dimensions of the model (sliding
  256, full 512) leave no backend that the engine image can run. The prefill kernel of TRITON_ATTN cannot compile for
  head_dim 512, and the kernel of FA4 caps head_dim at 256. The enum value of `TORCH_SDPA` is the empty string ("this
  tag is only used for ViT"). FLEX_ATTENTION fails in its own `torch.compile`/CUDAGraph interaction. The patch
  registers `TorchSdpaAttentionBackend` of the plugin under `Backend.TORCH_SDPA`, which is the member that
  `--attention-backend TORCH_SDPA` of the recipe selects. The backend is encoder-only, it sits over
  `torch.nn.functional.scaled_dot_product_attention`, and it lives in
  `rcp_vllm/models/embedding_gemma2/attention.py`. The class has no cap on the head dimension. It runs the
  non-causal attention of the model with the window of the sliding layers as a mask. Delete the patch and the
  `--attention-backend` pin of the recipe when `engine.image` moves to a new release. Delete them when the backends of
  that release clear the head dimensions of the model.

## Deployment and the fallback

A recipe is a declared configuration of `vllm serve`. The recipe names the image, the flags, the budgets and
the plugin. The engine serves the checkpoint from the Hub.

### The generic path

Stock vLLM serves the common architectures. This command serves a checkpoint without a recipe:

```
vllm serve <hf-repo> --runner pooling
```

The topk, pplx and embeddinggemma checkpoints need the plugin. The `rcp-vllm` wheel registers it through
`vllm.general_plugins`.

### The fallback

The reference implementation is the correctness fallback. Each recipe ships a `reference.py`. It runs the
checkpoint in process with `transformers`. Every equivalence gate compares the served engine against it.

A fallback changes how the model is served. It does not change what the model computes. Two failures stay:

- the `topk-embed-v1` gate: the per-token cosine floor fails on bf16 and fp16. The engine's GatedDeltaNet
  kernels refuse fp32;
- the pplx limits: the engine refuses an empty input. The reference refuses a mixed text+image document.

A generic deployment serves the same weights with the same numerics. The gate and the limits are the same.

## Plugin-provided fields

Part of the serving surface exists only in the plugin, never in the stock engine. It is the embed chunker's
three knobs and the rerank route's span-level relevance. Every embed and rerank change is the plugin's, and
none of it waits on an upstream release. `rcp_vllm.recipe.PLUGIN_FIELDS` is the one declaration of them. Each
field names the plugin distribution that honours it, and the loader reads that table. It never keeps a list of
names of its own.

| field | what the recipe declares | rendered as |
|---|---|---|
| `serve.pooler_config.chunk_length` | the embed chunker's chunk size | inside `--pooler-config` |
| `serve.pooler_config.chunk_weighting` | how a document's chunk scores aggregate | inside `--pooler-config` |
| `serve.pooler_config.normalize_aggregated_embeddings` | whether the aggregated document vector is L2-normalised | inside `--pooler-config` |
| `relevant_spans` | the rerank route accepts span-level relevance (the plugin's `relevant_spans` wire field) | nothing |

A recipe that declares one names the plugin in `serve.plugin`. The knobs are `PoolerConfig` fields, so they
ride inside the `--pooler-config` JSON the engine already parses. The rendered command carries them. Both
exported schemas declare them as optional properties of `serve.pooler_config` (the object stays open, so a
stock `PoolerConfig` field still passes), and the loader remains the authority on which keys exist:

```yaml
# a family's shared serve block: the chunk knobs beside the plugin that provides them
serve:
  runner: pooling
  dtype: bfloat16
  max_model_len: 8192
  plugin: rcp-vllm
  pooler_config: {chunk_length: 512, chunk_weighting: linear, normalize_aggregated_embeddings: true}
```

`rcp-vllm serve <id> --dry-run` renders that block as `--pooler-config '{"chunk_length": 512,
"chunk_weighting": "linear", "normalize_aggregated_embeddings": true}'`. A key must be a field of the pinned
engine or a row of the table. The loader refuses any other key by name. The engine would reject an unknown
key. `serve.extra_args` may not carry `--pooler-config`: `serve_argv` renders it, and the engine keeps the
last occurrence.

**The loader refuses what the target cannot honour.** A recipe declares one of these fields by naming
`serve.plugin`. The loader refuses the recipe at load without that name. The same refusal applies when the
declared plugin does not provide the field. Either way the loader names the key, before any engine sees the
argv. If the plugin is declared but this environment does not carry it, the refusal names the plugin. It also
gives the exact install line: `pip install --no-deps <spec>`, the recipe's own `serve.plugin` spec.
"Installed" is judged through the `vllm.general_plugins` entry-point group that vLLM loads in every engine
process. The packaging snapshot records that group for this wheel. The capability is therefore present only
when the mechanism that finds the plugin says so. It is never present because the recipe claims it.

`relevant_spans` is the one declared field that renders nothing. It says that the deployment honours
span-level relevance in a rerank request. The engine accepts that per request, and the plugin provides that
wire. Only a rerank recipe may declare it: the spans come back scored with a document. The loader refuses it
without the plugin, exactly as it refuses a chunk knob. It is a recipe-level field rather than a `serve` one on
purpose. Every `serve` field is a behaviour-fingerprint input. This capability changes the request shape the
deployment accepts, not how the engine computes. A `serve` home would re-key every recipe's recorded corpus.

## The catalog's columns

- `id` is the recipe id of the variant. It is the lowercased canonical Hub repository name of the model, and it is the
  `--served-model-name` that the engine serves. `recipe: <id>` resolves it. The id of a family names the directory,
  and it is never served.
- `model` is the Hub repository of the checkpoint. The `revision` of the variant pins it inside `family.yaml`.
- `role` is `embed`, `multi_vector`, `rerank` or `judge`. It names the role client that reads the served model.
- `input` is `text`, `image` or `video`. It names what the checkpoint reads.
- `mrl` is the declared Matryoshka head of the variant: `truncation` or `projection`, or `none`. A truncation or a
  projection gives the output dimensions that the model card supports. The card gives them as a discrete table or as a
  prose range. The recipe declares it one time in its client block (`mrl_kind` with `mrl_dims`/`mrl_range`). A
  projection kind also names `mrl_projection`, which holds the learned matrices of the checkpoint. Where the engine
  has the per-request `dimensions` path, the recipe mirrors the head in `serve.hf_overrides`
  (`is_matryoshka`/`matryoshka_dimensions`). The loader refuses a serve gate and a client declaration that disagree.
  The recipes ship the full width of the checkpoint, and a run selects `k` from the declared set.
- `plugin` is the model plugin that the checkpoint needs on the stock engine, when it needs one.
- `reference` names the source of the equivalence reference. It is `paper`, `card` or `—`. `paper` means that the
  `reference.py` of the family is the code path of the paper, ported from `experiments/paper/`. `card` means the
  published usage of the model card, and `—` is a judge recipe, which has no equivalence reference.
- `status` is `status.state` from the row of the variant in `family.yaml`. It is the state of the family until the
  variant declares its own. It is `unverified` (written, not yet checked), `verified` (the harness passed every gate)
  or `failed`. The engine `image`, the `date` and the report are recorded beside it.

## Budgets

Every recipe declares a tokenizer, and the loader injects it as `model@revision` unless the family pins one. The
recipe also declares `client.max_tokens`, and `query_max_tokens` where the reference caps queries. A reranker declares
`document_max_tokens` where the checkpoint cuts each document itself. The client cuts over-budget content at token
boundaries, and it preserves the anchors of the template. Every cut is recorded ([text budgets for served
roles](../concepts/text-budgets.md)). This page does not repeat the limits. The recipe file is the source.

## Equivalence policy

Served recipes use cuts that preserve the anchors. The code of the paper cut without regard to the place where the
model reads its answer. A recipe whose reference drops anchors declares
`reference.known_deviations: [anchor_drop_over_cap]`, and the equivalence gates compare those models under the cap
only. Only a document over the cap can end at a different place.

## `recipe:` in `rcp`

A role config that names `recipe: <id>` takes its whole client block from the recipe. The block holds the api, the
tokenizer, the budgets, the template, the media and the instruction mode. `base_url` and the other RUNTIME fields stay
on the config. Any CONTENT field that the config sets explicitly must equal that of the recipe. Otherwise the config
is refused with a `ConfigError` that names both values. The MRL selection is one exception. The recipe declares the
Matryoshka kind and the set of the card one time. A config can set `mrl_dim` (the client head) or `dimensions` (the
engine-side cut). The config accepts it when `k` is in the declared `mrl_dims`/`mrl_range`. Otherwise the refusal
names the set. The recipes ship the full width of the checkpoint, and nothing is selected unless it is declared. The
`recipe` pointer itself is replaced by the identity of the recipe. That is its shipped id, or `unshipped:sha256:<hex>`
for a file of the operator's own. A run identity therefore follows the content of the file, never the spelling of a
path. On the command line, `--retriever recipe:<id-or-path>` and `--reranker recipe:<id-or-path>` expand to that
mapping, with the URL from `--set ...base_url=...` or a `serve:` engine. `recipe:./my-family/` and `recipe:/abs/path`
load the family directory through the same schema. A multi-variant directory names its variants in the refusal.
Resolution is lazy through `rcp_vllm`, so `rcp-vllm` must be installed alongside `rcp`; there is no
extra alias for it. Without it the refusal is typed, and its hint is the install line.
