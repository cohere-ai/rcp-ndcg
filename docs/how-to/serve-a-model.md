# Serve a retrieval model

The shipped retrieval models -- 27 of them (one served recipe id per checkpoint; a family's sizes share a
directory) -- run on the stock `vllm/vllm-openai`
image, except `embeddinggemma-2`, which pins a vLLM nightly by digest (owner decision 38; the released image
lacks the architecture). This walk-through serves one, points `rcp-ndcg` at it and scores what it retrieves. The rules of the
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

`rcp-ndcg-vllm serve <recipe-id> [--variant VARIANT-ID] [--port PORT] [--set PATH=VALUE ...] [--dry-run]`
builds the `vllm serve`
argv from the recipe's package data -- the chat template file path, the media flags, the pooler config -- and
runs it. `--dry-run` prints the argv (one shell-quoted line), the recipe's identity and the applied overrides,
and exits, to inspect what would run. A checkpoint that needs its model plugin is refused with the exact
install line.

Two options when the image does not carry the wheel: bake it into your image (image preparation), or install
it inline where the engine command runs:

```bash
bash -lc 'python3 -m pip install --no-deps rcp-ndcg-vllm==<version> && exec rcp-ndcg-vllm serve <id> --port 8000'
```

The same two forms work in a run config's `serve:` block, whose `command` is yours verbatim ([runs and job
runners](../concepts/runs.md#starting-the-engines-with-the-run)).

## 2. Deployment overrides: `--set`, for the node the recipe runs on

A recipe declares the model; the node decides how many GPUs it gets, how much of their memory it may use and
how it batches. Those are the recipe's **deployment** fields, and `--set <path>=<value>` sets them at serve
time without touching the recipe:

```bash
rcp-ndcg-vllm serve qwen3-reranker-0.6b --port 8000 \
  --set resources.gpus=2 \
  --set serve.gpu_memory_utilization=0.85 \
  --set serve.max_num_seqs=64 \
  --set serve.max_num_batched_tokens=4096 \
  --set serve.host=0.0.0.0 --set serve.port=8000
```

| `--set` path | what it renders | notes |
|---|---|---|
| `resources.gpus` | `--tensor-parallel-size` | the recipe's per-variant GPU count, overridden |
| `serve.gpu_memory_utilization` | `--gpu-memory-utilization` | a finite fraction above 0 and at most 1 (the engine's own default until set) |
| `serve.max_num_seqs` | `--max-num-seqs` | |
| `serve.max_num_batched_tokens` | `--max-num-batched-tokens` | |
| `serve.host` | `--host` | the interface the engine listens on (`0.0.0.0` by default) |
| `serve.port` | `--port` | wins over the console's `--port`; 0-65535 (0: the engine binds an ephemeral port) |
| `serve.max_model_len` | `--max-model-len` | at or above the client's largest token budget, see below |

The recipe schema declares this surface once (`rcp_ndcg_vllm.recipe.FIELD_ROLES`), so the list above is the
schema's, not the command's: a path that is not declared DEPLOYMENT is refused. A **content** field -- the
model, the revision, `serve.dtype`, the pooler config, a template, the hf overrides, a patch -- is refused by
name with the hint *a different revision or content is a different variant: add a variant row*: serving
another dtype or another checkpoint is a new recipe row, never a flag. A run's own knob
(`engine.startup_timeout_s`) is refused as RUNTIME: the run owns it.

`serve.max_model_len` is refused, with both numbers, when it falls below the client's largest token budget
(`client.max_tokens`, `query_max_tokens` or `document_max_tokens`): the engine would reject prompts the client
is allowed to send. Raising it is allowed -- up to the checkpoint's own context limit, which the engine reads
from the model config at startup; the recipe's declared value stays the verified one. `--dry-run` prints the
applied overrides beside the argv, and a real serve logs them. A corpus manifest records the argv each engine
was started with verbatim (`engine.serve_argv`), so an engine started with overrides is recorded with them;
the GPU waves serve the recipes as shipped, with no overrides. `--port` (the run's own spelling of
`serve.port`) and `--set serve.port` are checked the same way; `--set serve.port` wins when both are given.

## 3. Your own recipe file: a path instead of an id

`serve` and `recipe:` also take a **family directory of your own** (or its `family.yaml`): the same schema
validates it, families included, and the file never needs to live in the package.

```bash
rcp-ndcg-vllm serve ./my-family/ --variant my-reranker-0.6b --port 8000
```

`--variant` names the size when the directory declares more than one; a directory with exactly one variant
needs no flag (and a shipped id refuses `--variant`: the id already names one). A name that looks like a recipe
id is the catalog's recipe first: a directory of the same name in the working directory does not shadow it --
name the file with `./` to mean the file. Such a recipe is **unshipped**:
every record of it says so, its `status` is `unverified` whatever the file claims (the verification record
belongs to a shipped recipe), and its identity is the content hash of its resolved form --
`unshipped:sha256:<hex>`, the referenced chat template file included -- never a shipped id, so two runs whose
files differ never share a run identity, and a run that records that identity can be read back (its `run
status`, a resume, an index reload): the pointer is recognised and the expanded block beside it is used as it
stands. That pointer is a recorded identity, not a claim to re-check: the file is not consulted again (it may
not exist on the machine reading the config), so a config carrying one is a trusted snapshot -- editing its
block is editing the config.

```yaml
# retriever.yaml -- the same file, from the client side
kind: dense
encoder:
  recipe: ./my-family/            # or /abs/path/to/my-family, or ../my-family/family.yaml
  base_url: http://127.0.0.1:8000/v1
```

`recipe:./my-family/` (and `recipe:/abs/path`) resolves through the same loader, and the shorthand
`--retriever recipe:./my-family/` / `--reranker recipe:./my-family/` expands to it. A multi-variant family
directory names its variants in the refusal: point `recipe:` at a single-variant directory, or serve the
variant with `--variant` and use the served URL. The `schema_version` check applies to a file of your own
exactly as it does to a shipped one: a version this `rcp-ndcg` does not read is refused, naming both.

## 4. The client host: `recipe:` in the retriever config

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
rcp-ndcg retrieval index --dataset suite:nanobeir --subset NanoSciFact --retriever retriever.yaml --out index/
rcp-ndcg retrieval search --dataset suite:nanobeir --subset NanoSciFact --retriever retriever.yaml --out rankings.parquet
rcp-ndcg eval score --rankings rankings.parquet --suite nanobeir
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
