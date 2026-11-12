# Serve a retrieval model

The package ships 34 retrieval models. There is one served recipe id per checkpoint. A family's sizes share a
directory. The models run on the stock `vllm/vllm-openai` image that their recipe pins. The pin is `vllm/vllm-openai:v0.33.0` for every one of them. A checkpoint that the image cannot serve as shipped carries
its model plugin instead (`embeddinggemma-2`). The pin moved every recipe to `status: unverified`. A recipe returns to `verified` only from a wave that re-recorded its corpus on the pinned image. 32 of these 34 are
verified there today. The two `topk-embed-v1` variants are the owner-disposed `failed` rows. The count is 42 verified today across all 44 recipes, and none are `unverified`. The [GPU validation
ledger](../reference/gpu-validation.md) states the rule and records every row. This procedure serves one
model, points `rcp` at it and scores what it retrieves. The rules of the recipes are in [recipes and
serving models](../reference/recipes.md). Hosted APIs (Cohere, Voyage, Gemini) need none of this.
[embedding endpoints](../concepts/embeddings.md) covers them.

## 1. The engine host: one wheel on the stock image

The serving recipes and the `rcp-vllm serve` command ship in the `cohere-rcp-vllm` distribution. This is
the lean serving package; its only dependencies are pydantic and PyYAML. Install it with `--no-deps`. This is the one change that the engine environment takes. A `pip freeze` before and after differs by exactly this
wheel.

```bash
python3 -m pip install --no-deps cohere-rcp-vllm==<version>
rcp-vllm serve qwen3-embedding-0.6b --port 8000
```

`rcp-vllm serve <recipe-id> [--variant VARIANT-ID] [--port PORT] [--set PATH=VALUE ...] [--dry-run]` builds the `vllm serve` argv from the recipe's package data. The data gives the chat template file path, the media flags and the pooler config. Then it runs the command. `--dry-run` prints the argv (one shell-quoted line), the recipe's identity and the applied overrides. It also prints the engine environment's patch opt-in (`RCP_NDCG_VLLM_PATCHES`, empty for a recipe that declares none). Then it exits, so you can inspect what would run. The argv alone is not the
engine command for a patched recipe. The patches are inert unless the engine process carries the variable.
Thus run `rcp-vllm serve` itself (or export the printed value). Do not paste the argv alone. A checkpoint
that needs its model plugin is refused with the exact install line. The check runs in the interpreter that the
`vllm` executable's shebang names. Thus the check judges a `vllm` from another environment by its own
environment. The wheel must be installed there too.

Two options exist when the image does not carry the wheel. Bake it into your image (image preparation), or
install it inline where the engine command runs:

```bash
bash -lc 'python3 -m pip install --no-deps cohere-rcp-vllm==<version> && exec rcp-vllm serve <id> --port 8000'
```

The same two forms work in a run config's `serve:` block. Its `command` is yours verbatim ([runs and job
runners](../concepts/runs.md#starting-the-engines-with-the-run)).

## 2. Deployment overrides: `--set`, for the node the recipe runs on

A recipe declares the model. The node decides how many GPUs it gets, how much of their memory it may use and
how it batches. Those are the recipe's **deployment** fields. `--set <path>=<value>` sets them at serve time
without a change to the recipe:

```bash
rcp-vllm serve qwen3-reranker-0.6b --port 8000 \
  --set resources.gpus=2 \
  --set serve.gpu_memory_utilization=0.85 \
  --set serve.max_num_seqs=64 \
  --set serve.max_num_batched_tokens=4096 \
  --set serve.host=0.0.0.0 --set serve.port=8000
```

| `--set` path | what it renders | notes |
|---|---|---|
| `resources.gpus` | `--tensor-parallel-size` | the recipe's per-variant GPU count, overridden | | `serve.gpu_memory_utilization` | `--gpu-memory-utilization` | a finite fraction above 0 and at most 1 (the engine's own default until set) |
| `serve.max_num_seqs` | `--max-num-seqs` | | | `serve.max_num_batched_tokens` | `--max-num-batched-tokens` | rendered at `serve.max_model_len` when the client's largest budget reaches it; a smaller explicit value is warned about and replaced, see below |
| `serve.host` | `--host` | the interface the engine listens on (`0.0.0.0` by default) | | `serve.port` | `--port` | takes precedence over the console's `--port`; 0-65535 (0: the engine binds an ephemeral port) | | `serve.max_model_len` | `--max-model-len` | at or above the client's largest token budget, see below |

The recipe schema declares this surface once (`rcp_vllm.recipe.FIELD_ROLES`). Thus the list above is the
schema's, not the command's. The command refuses a path that is not declared DEPLOYMENT. A **content** field is refused by name. The hint is *a different revision or content is a different variant: add a variant row*.
Content fields are the model, the revision, `serve.dtype`, the pooler config, a template, the hf overrides and
a patch. To serve another dtype or another checkpoint, add a new recipe row. It is never a flag. A run's own
knob (`engine.startup_timeout_s`) is refused as RUNTIME. The run owns it.

The command refuses `serve.max_model_len`, with both numbers, when it falls below the client's largest token budget (`client.max_tokens`, `query_max_tokens` or `document_max_tokens`). The engine would reject prompts
that the client is allowed to send. Raising it is allowed, up to the checkpoint's own context limit. The
engine reads that limit from the model config at startup. The recipe's declared value stays the verified one.
`--dry-run` prints the applied overrides beside the argv. A real serve logs them. A corpus manifest records
the argv that each engine was started with, verbatim (`engine.serve_argv`). Thus an engine started with
overrides is recorded with them. The GPU waves serve the recipes as shipped, with no overrides. `--port` (the run's own spelling of `serve.port`) and `--set serve.port` are checked the same way. `--set serve.port` takes
precedence when both are given.

`serve.max_num_batched_tokens` is **rendered at** `serve.max_model_len` when the client's largest token budget reaches `serve.max_model_len` exactly. The engine's own default chunk depends on the GPU class. It is 2048 on A100 and GPUs under 70 GiB, below these recipes' budgets. Thus an exactly-full prompt would be split into more than
one prefill step. The recipe renders the safe value instead of relying on that default. Nothing is left to the
engine, and the at-budget prompt is one prefill step. No recipe disables the floor. The one patch that
declared a recipe safe without it is deleted. That patch is `pooling-full-context`, a backport of the
scheduler fix that a vLLM 0.31 engine needed. The pinned engine's own scheduler keeps the whole context of a
pooling prefill (vllm-project/vllm#48039, vLLM v0.32.0). Thus the floor is the declared shape of the
deployment, no longer a workaround. A recipe whose client budget stays below `serve.max_model_len` never sends such a prompt and renders no derived value. The `qwen3-reranker` family is an example: 8192 against 10000. An explicit `--set serve.max_num_batched_tokens` below the floor is **warned about and replaced by** `serve.max_model_len`. The message names the value asked for and the value rendered. The rule warns rather
than refuses. But rendering the smaller value would split the prompt that the floor exists to keep whole. The floor fires for the sixteen affected recipes. They are the pooling recipes whose largest client budget reaches `serve.max_model_len`. The list starts with `ctxl-rerank-v2-instruct-multilingual-2b`, `embeddinggemma-2`, the three `harrier-oss-v1` sizes and the two `jina-embeddings-v5-text` sizes. It continues with the three `octen-embedding` sizes, the two `pplx-embed-v1` sizes, the two `qwen3-vl-embedding` sizes, `zembed-1-embedding` and `zerank-2-reranker`.

## 3. Your own recipe file: a path instead of an id

`serve` and `recipe:` also take a **family directory of your own** (or its `family.yaml`). The same schema
validates it, families included. The file never needs to live in the package.

```bash
rcp-vllm serve ./my-family/ --variant my-reranker-0.6b --port 8000
```

`--variant` names the size when the directory declares more than one. A directory with exactly one variant
needs no flag. A shipped id refuses `--variant`: the id already names one. A name that looks like a recipe id
is the catalog's recipe first. A directory of the same name in the working directory does not shadow it. Name
the file with `./` to mean the file. Such a recipe is **unshipped**. Every record of it says so. Its `status` is `unverified` whatever the file claims: the verification record belongs to a shipped recipe. Its identity is
the content hash of its resolved form: `unshipped:sha256:<hex>`, the referenced chat template file included.
It is never a shipped id. Thus two runs whose files differ never share a run identity. A run that records that
identity can be read back (its `run status`, a resume, an index reload). The pointer is recognised and the
expanded block beside it is used as it stands. That pointer is a recorded identity, not a claim to re-check.
The file is not consulted again; it may not exist on the machine that reads the config. Thus a config carrying
one is a trusted snapshot. Editing its block is editing the config.

```yaml
# retriever.yaml -- the same file, from the client side
kind: dense
encoder:
  recipe: ./my-family/            # or /abs/path/to/my-family, or ../my-family/family.yaml
  base_url: http://127.0.0.1:8000/v1
```

`recipe:./my-family/` (and `recipe:/abs/path`) resolves through the same loader. The shorthand `--retriever recipe:./my-family/` / `--reranker recipe:./my-family/` expands to it. A multi-variant family directory names
its variants in the refusal. Point `recipe:` at a single-variant directory, or serve the variant with `--variant` and use the served URL. The `schema_version` check applies to a file of your own exactly as it
does to a shipped one. The command refuses a version that this `rcp` does not read, and names both.

## 4. The client host: `recipe:` in the retriever config

The client side needs `cohere-rcp` with `cohere-rcp-vllm` installed alongside it (no extra alias). A role config that names `recipe: <id>` takes its whole client block from the recipe. The block gives the api, tokenizer, budgets, template, media and instruction mode. `base_url` (and the other run-time fields) stays on the config. Any content field set
explicitly must equal the recipe's. Otherwise the config is refused with an error that names both values.

```yaml
# retriever.yaml
kind: dense
encoder:
  recipe: qwen3-embedding-0.6b      # the recipe's client block: api, tokenizer, budgets, template, media
  base_url: http://127.0.0.1:8000/v1  # run-time fields stay here
```

Budgets are explicit in every recipe. Over-budget content is cut client-side at token boundaries, with the
template's anchors preserved. Every cut is recorded. There is never an engine-side truncation ([text
budgets](../concepts/text-budgets.md)).

```bash
rcp retrieval index --dataset suite:nanobeir --subset NanoSciFactRetrieval --retriever retriever.yaml --out index/
rcp retrieval search --dataset suite:nanobeir --subset NanoSciFactRetrieval --retriever retriever.yaml --out rankings.parquet
rcp eval score --rankings rankings.parquet --suite nanobeir --subset NanoSciFactRetrieval
```

The command line also takes the shorthand `--retriever recipe:qwen3-embedding-0.6b` (and `--reranker recipe:<id>`). It expands to the mapping above, with the URL from `--set retriever.encoder.base_url=...` or from a `serve:` engine. `rcp retrieval rerank` re-scores the pool with a `recipe:` reranker the same way
([retrieval and reranking](../concepts/retrieval.md)). A run's `retrieve`, `rerank` steps carry the same
configs.

## See also

- [Add a serving recipe](add-a-model.md) -- the recipe format, and its checks before any GPU time.
- [Validate a recipe on GPUs](validate-a-recipe.md) -- the waves a recipe passes before the release.
- [Recipe authors' reference](../reference/recipes.md) -- the catalog and every name.
