# Recipes and serving models

Exact names on the serving surface. The catalog of the 24 shipped families and their 44 variants (the
canonical variant ids, the model, the role, the input, the Matryoshka head, the plugin, the reference and the
status of every row) is the table in the [`rcp-vllm` README](https://github.com/cohere-ai/rcp-ndcg/blob/main/rcp-vllm/README.md)
-- the distribution's PyPI page, and the one rendered copy. This page documents
what the rows and the surface mean. [Serve a retrieval model](../how-to/serve-a-model.md) walks through using
one; [add a serving recipe](../how-to/add-a-model.md) writes one.

## The `rcp-vllm` distribution

The recipes are package data, grouped into **families**: `recipes/<family>/family.yaml` holds the
shared blocks and the `variants` table, with the family's ONE `reference.py` (parameterised by the variant), its
one chat template where the model needs one, its `reference.in` (the justified pins) and the generated
`reference.lock` (one locked, hashed reference environment per family). Every variant resolves to a
full `Recipe` (the unchanged recipe schema) and is served, contract-tested, stage-1-tested and GPU-validated on
its own; a family id is never served. The public names are `rcp_vllm.recipe`'s `RecipeFamily`, `Variant`,
`Recipe`, `RecipeFieldRole`, `FieldSpec`, `FIELD_ROLES`, `load_family`, `load_recipe`, `resolve_recipe`,
`iter_families`, `iter_recipes`,
`serve_argv` (the serve-argv builder), `deployment_fields`, `parse_deployment_overrides`,
`recipe_digest` and `plugin_distribution_name`,
the `rcp-vllm` console tree (`serve`, with `--variant`, `--port`, `--set` and `--dry-run`), and the
exported schemas (`schema/recipe.schema.json` for a resolved recipe, `schema/family.schema.json` for a family
file); everything else in the package is internal (``RecipeError``, the typed refusal every loader raises,
lives in `rcp_vllm.errors` and is re-exported from the package root).

## The shipped families

24 families: 16 retrieval families (embed, multi_vector and rerank) and 8 judge families. The README's table
names every family's variants with their model, role, input, Matryoshka head, plugin, reference and status.

Every family pins the stock engine image in its `engine` block (`vllm/vllm-openai:v0.33.0`, `min_version`
`0.33.0`), and every variant's `status` reads `verified` except the two owner-disposed `topk-embed-v1`
variants, which read `failed`: the engine image is one of the behaviour fingerprint's inputs, so no recipe's
recorded evidence counts for the pinned image until a wave re-records its corpus there, and every row names
the wave that did (the [GPU validation ledger](../reference/gpu-validation.md) is that record). A
`status` block keeps the `image`, `date` and `report` of the wave it names, and `verified` is only ever
written beside the image the recipe pins. The two `topk-embed-v1` variants are the catalog's recorded gate
failure (the stage-2 per-token cosine floor): their state returns to `failed` only from a wave that re-runs
that gate, and they ship only when the owner's explicit disposition list names them.

### Known limits, per family

The families whose evidence carries a recorded limit or a disposition. A family not listed here carries none.

| family | limit | evidence |
|---|---|---|
| `topk-embed-v1` (small, xsmall) | `failed`: the stage-2 per-token cosine floor is not met on the available engine configurations (bf16 and fp16 fail; fp32 cannot be served, the engine's GatedDeltaNet kernels refuse it) | the owner's disposition, the `gate-topk-embed-v1` wave |
| `pplx-embed-v1` (0.6b, 4b) | the engine refuses an empty input, and the reference cannot encode a zero-length sequence: the row is recorded and pruned | `declared_unencodable`, the row's own kind |
| `pplx-embed-v2-late` (0.6b, 9b) | the reference refuses a mixed text+image document (the card's usage note); the 9b's per-token floor is 0.9410, the 0.6b's 0.99 | `declared_reference_refusal`; the floors' derivations in the family notes |
| `qwen3-vl-embedding` (2b, 8b) | the reference substitutes the card's `NULL` for an empty query; the media rows have their own floor (2b 0.9975, 8b 0.9966) | `declared_substituted`; `gates.media_vec_min_cosine` |
| `ctxl-rerank-v2-instruct-multilingual-6b` | serves `float32` (the 1b/2b dtype): on bf16 one document exceeded the family's delta gate | the family notes; the `equivalence-ctxl` wave |
| `jina-embeddings-v5-text-nano` | the at_budget row (8187 of 8192 tokens) is the minimum; the floor is 0.9964 | the floor's derivation in the family notes |
| `qwen3-embedding` | the MRL-32 head had no margin; the floor is 0.9984 (the 4b keeps 0.999, with 3.5e-4 of margin) | the floor's derivation in the family notes |
| `embeddinggemma-2` | the transformers fold is deleted (its parity is inside the measured bf16 floor); the SDPA patch stays | `DECISION-PARITY.md`, the `sdpa-probe` wave |

A judge family's `client` block is `rcp.judging.JudgeConfig` (the same config the judging commands take,
validated by the product, never a second model), its `serve` block is the engine argv, and it carries no
equivalence reference: `--judge <id>` and `rcp-vllm serve <id>` take the same id
([judges](../concepts/judges.md)).

`rcp-vllm serve <recipe-id> [--variant VARIANT-ID] [--port PORT] [--set PATH=VALUE ...] [--dry-run]`
builds the `vllm serve` argv from the recipe's package data (the chat template file path, the media flags, the
pooler config) and runs it; `--dry-run` prints the argv, the recipe's identity and the applied overrides, and
exits. `--set` names a **deployment** field of the recipe -- the engine's resource, scheduling and address
knobs (`resources.gpus`, `serve.gpu_memory_utilization`, `serve.max_num_seqs`,
`serve.max_num_batched_tokens`, `serve.host`, `serve.port`, `serve.max_model_len`) -- and the schema declares
that surface once (`rcp_vllm.recipe.FIELD_ROLES`): a content field is refused by name, and
`serve.max_model_len` is refused below the client's largest token budget, with both numbers named. A checkpoint
that needs its model plugin is refused with the exact install line: the `topk-embed-v1-small`/`-xsmall`
(`TopkEmbedModel`), the pplx checkpoints (pplx-embed-v1's config-only `PplxV1Config`; the pplx-embed-v2
contextual and both
late-interaction sizes) and embeddinggemma-2 (`EmbeddingGemma2Config`/`EmbeddingGemma2Model`) fold into
`rcp_vllm/models/` under one lazy `vllm.general_plugins` entry point (importing `rcp_vllm` never
imports torch or vLLM). A recipe that names a plugin
also declares `plugin_architectures` -- the architectures its engine registers (empty for a patch-only
carrier) -- because the behaviour
fingerprint keys the plugin's code by hashing exactly those modules (`plugin_sha256.<module>`: the shared
entry modules, the architecture's modules and every opted-in patch's module); a foreign plugin whose modules
the harness cannot resolve is refused at fingerprint time, by name.

`serve` and `recipe:` also take a **family directory of the operator's own** (`./my-family/`, with
`--variant <id>` for one size of several): the same schema validates it, families included, and every record
marks it unshipped with `status: unverified`. Its identity is the content hash of its resolved form --
`unshipped:sha256:<hex>`, the referenced chat template file included -- never a shipped id, so two runs whose
files differ never share a run identity; a config that records that identity is read back as it stands (a
run's resume, an index reload).

## Engine-side patches

A checkpoint the stock image cannot serve as shipped may need an engine-side fix: an architecture the
image's registry lacks, or an attention backend its kernels cannot run. A recipe opts into such a fix with
`serve.patches`, naming the patch (`rcp_vllm.patches.PATCH_NAMES`); every engine-start path renders the
declared names into the engine process's `RCP_NDCG_VLLM_PATCHES` (the `rcp-vllm serve` console, the wave
runner and the e2e driver -- a comma-separated list read by the one `vllm.general_plugins` entry point),
overriding any inherited value so
the engine runs exactly what the recipe declares. The corpus provenance records the value the engine ran
with, and the behaviour fingerprint hashes every opted-in patch's module, so a patch fix moves the recipe's
key. One patch ships, and one recipe opts in: `embeddinggemma-2` declares it, and every other family's
`serve.patches` is empty (the chunked-prefill patch a few families used to carry is deleted -- the pinned
engine's own scheduler keeps the whole context of a pooling prefill, vllm-project/vllm#48039; the
embeddinggemma-2 transformers fold is deleted too -- the **0.33 image's transformers 5.20.2** ships the
checkpoint's `embedding_gemma2` classes and the v0.33.0 tag's registry the architecture
(`vllm/model_executor/models/registry.py:259`), so the backport was inert there, and the cross-image parity
(the 0.31 image with the fold active against the 0.33 image with the native model) measured cosine
0.999972, inside the recipe's measured bf16 floor 0.999882).

- `torch-sdpa-attention` -- the embeddinggemma-2 backend: the model's heterogeneous head dimensions
  (sliding 256, full 512) leave no backend the engine image can run -- TRITON_ATTN's prefill kernel
  cannot compile for head_dim 512, FA4's kernel caps head_dim at 256, `TORCH_SDPA`'s enum value is the
  empty string ("this tag is only used for ViT") and FLEX_ATTENTION dies in its own
  `torch.compile`/CUDAGraph interaction. The patch registers the plugin's
  `TorchSdpaAttentionBackend` (an encoder-only backend over
  `torch.nn.functional.scaled_dot_product_attention`, in
  `rcp_vllm/models/embedding_gemma2/attention.py`) under `Backend.TORCH_SDPA`, which is the
  member the recipe's `--attention-backend TORCH_SDPA` selects; the class has no head-dimension cap
  and runs the model's non-causal attention with the sliding layers' window as a mask. Delete the
  patch (and the recipe's `--attention-backend` pin) when `engine.image` moves to a release whose own
  backends clear the model's head dimensions.

## Plugin-provided fields

Part of the serving surface exists only in the plugin, never in the stock engine: the embed chunker's three
knobs and the rerank route's span-level relevance (every embed and rerank change is the plugin's, and none of
it waits on an upstream release). `rcp_vllm.recipe.PLUGIN_FIELDS` is the one declaration of them, each field
with the plugin distribution that honours it; the loader reads that table, never a list of names of its own.

| field | what the recipe declares | rendered as |
|---|---|---|
| `serve.pooler_config.chunk_length` | the embed chunker's chunk size | inside `--pooler-config` |
| `serve.pooler_config.chunk_weighting` | how a document's chunk scores aggregate | inside `--pooler-config` |
| `serve.pooler_config.normalize_aggregated_embeddings` | whether the aggregated document vector is L2-normalised | inside `--pooler-config` |
| `relevant_spans` | the rerank route accepts span-level relevance (the plugin's `relevant_spans` wire field) | nothing |

A recipe that declares one names the plugin in `serve.plugin`. The knobs are `PoolerConfig` fields, so they
ride inside the `--pooler-config` JSON the engine already parses, and the rendered command carries them:

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
"chunk_weighting": "linear", "normalize_aggregated_embeddings": true}'`. A key that is neither a field of the
pinned engine nor a row of the table is refused by name (the engine rejects an unknown key), and
`serve.extra_args` may not carry `--pooler-config`: `serve_argv` renders it, and the engine keeps the last
occurrence.

**The loader refuses what the target cannot honour.** A recipe that declares one of these fields without
`serve.plugin` -- or with a plugin that does not provide it -- is refused at load, by name, before any engine
sees the argv. If the plugin is declared but this environment does not carry it, the refusal names the plugin
and its exact install line (`pip install --no-deps <spec>`, the recipe's own `serve.plugin` spec). "Installed"
is judged through the `vllm.general_plugins` entry-point group vLLM loads in every engine process -- the group
the packaging snapshot records for this wheel -- so the capability is present when the mechanism that finds the
plugin says so, never because the recipe claims it.

`relevant_spans` is the one declared field that renders nothing: it says that the deployment honours
span-level relevance in a rerank request, which the engine accepts per request, and the plugin provides that
wire. Only a rerank recipe may declare it (the spans come back scored with a document), and the loader refuses
it without the plugin, exactly as it refuses a chunk knob. It is a recipe-level field rather than a `serve`
one on purpose: every `serve` field is a behaviour-fingerprint input, while this capability changes the request
shape the deployment accepts, not how the engine computes -- a `serve` home would re-key every recipe's
recorded corpus.

## The catalog's columns

- `id` -- the variant's recipe id: the lowercased canonical Hub repository name of the model, the
  `--served-model-name` the engine serves and what `recipe: <id>` resolves. A family's id names the directory
  and is never served.
- `model` -- the checkpoint's Hub repository, pinned by the variant's `revision` inside `family.yaml`.
- `role` -- `embed`, `multi_vector`, `rerank` or `judge`: which role client reads the served model.
- `input` -- `text`, `image`, `video`: what the checkpoint reads.
- `mrl` -- the variant's declared Matryoshka head: `truncation` or `projection` with the model card's
  supported output dimensions (a discrete table or a prose range), or `none`. It is declared once in the
  recipe's client block (`mrl_kind` with `mrl_dims`/`mrl_range`; a projection kind also names
  `mrl_projection`, the checkpoint's learned matrices) and, where the engine's per-request `dimensions`
  path exists, mirrored in `serve.hf_overrides` (`is_matryoshka`/`matryoshka_dimensions`); the loader
  refuses a serve gate and a client declaration that disagree. The recipes ship the checkpoint's full
  width and a run selects `k` from the declared set.
- `plugin` -- the model plugin the checkpoint needs on the stock engine, when one.
- `reference` -- where the equivalence reference comes from: `paper` (the family's `reference.py` is the
  paper's own code path, ported from `experiments/paper/`), `card` (the model card's published usage), or `—`
  for a judge recipe, which carries no equivalence reference.
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

## `recipe:` in `rcp`

A role config that names `recipe: <id>` takes its whole client block (api, tokenizer, budgets, template, media,
instruction mode) from the recipe. `base_url` and the other RUNTIME fields stay on the config; any CONTENT
field set explicitly must equal the recipe's, or the config is refused with a `ConfigError` naming both values.
One exception is the MRL selection: the recipe declares the Matryoshka kind and the
card's set once, so a config's `mrl_dim` (the client head) or `dimensions` (the engine-side cut) is
accepted when `k` is in the declared `mrl_dims`/`mrl_range` and refused naming the set otherwise -- the
recipes ship the checkpoint's full width and nothing is selected unless it is declared.
The `recipe` pointer itself is replaced by the recipe's identity (its shipped id, or `unshipped:sha256:<hex>`
for a file of the operator's own), so a run identity follows the file's content, never the spelling of a path.
On the command line `--retriever recipe:<id-or-path>` and `--reranker recipe:<id-or-path>` expand to that
mapping, with the URL from `--set ...base_url=...` or a `serve:` engine; `recipe:./my-family/` and
`recipe:/abs/path` load the family directory through the same schema (a multi-variant directory names its
variants in the refusal). Resolution is lazy through `rcp_vllm`, so `rcp-vllm` must be installed
alongside `rcp` (there is no extra alias for it); without it the refusal is typed and its hint is the
install line.
