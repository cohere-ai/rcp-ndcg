# rcp-ndcg-vllm

The serving half of RCP-nDCG: the vetted serving recipes for retrieval models and judges, the `rcp-ndcg-vllm
serve` command that turns one into a `vllm serve` command for the stock `vllm/vllm-openai` image (or a
digest-pinned nightly when a recipe needs a commit the release lacks), and the model plugins that make the
topk and pplx checkpoints serveable on it. The engine is reached over HTTP only; this
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
rcp-ndcg-vllm serve <recipe-id> [--variant VARIANT-ID] [--port PORT] [--set PATH=VALUE ...] [--dry-run]
```

builds the `vllm serve` argv from the recipe's package data (the chat template file path, the media flags, the
pooler config) and runs it (`--dry-run` prints the argv, the recipe's identity and the applied overrides, and
exits). `--set` names a **deployment** field of the recipe -- `resources.gpus`,
`serve.gpu_memory_utilization`, `serve.max_num_seqs`, `serve.max_num_batched_tokens`, `serve.host`,
`serve.port`, `serve.max_model_len` -- and the schema declares that surface once
(`rcp_ndcg_vllm.recipe.FIELD_ROLES`): a content field (the model, the revision, the dtype, a template, ...) is
refused by name, and `serve.max_model_len` is refused below the client's largest token budget. A checkpoint
that needs its model plugin is refused with the exact install line. Example, on the stock image:

```bash
python3 -m pip install --no-deps rcp-ndcg-vllm
rcp-ndcg-vllm serve qwen3-embedding-0.6b --port 8000
rcp-ndcg-vllm serve ./my-family/ --variant my-reranker-0.6b --set serve.max_num_seqs=64
```

The last line is a **recipe file of your own**: a family directory loaded through the same schema, marked
unshipped with `status: unverified` in every record and identified by the content hash of its resolved form
(`unshipped:sha256:<hex>`), never by a shipped id.

The [recipe guide](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/how-to/serve-a-model.md) walks through
the client side (`recipe: <id>`, or `recipe:./my-family/` for a file of your own) and the deployment overrides,
and [validate a recipe on
GPUs](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/how-to/validate-a-recipe.md) through the waves every
recipe passes before the release.

## The recipes

The shipped recipes are grouped into **families**: one directory per model family,
`recipes/<family>/family.yaml`, holds the shared serving contract and a `variants` table with only the
per-size facts, and the family's ONE `reference.py` (parameterised by the variant) runs every size's
equivalence check. Every variant is a full recipe id -- served by `rcp-ndcg-vllm serve <variant-id>`,
resolvable by `recipe: <variant-id>` in `rcp-ndcg`, contract-tested and GPU-validated on its own; a family id
is never served.

The catalog is **24 families and 44 variants**: 34 retrieval variants (16 families: embed, multi_vector and
rerank) and 10 judge variants (8 families, `role: judge`). Every variant must pass its end-to-end GPU
validation against its reference implementation before the tag (equivalence, quality and end-to-end waves);
`family.yaml` records
each variant's model revision and `sources`, and `status.state` (`unverified`, `verified`, `failed`) records the
outcome beside the engine `image`, the `date` and the report. The `status` column below is copied from each
variant's own `status.state` (the family's until a variant declares its own), so it reads `unverified` for every
variant the waves have not verified yet; the tag ships none unverified.

| family | id | model | role | input | MRL | plugin | reference | status |
|---|---|---|---|---|---|---|---|---|
| `ctxl-rerank-v2-instruct-multilingual` | `ctxl-rerank-v2-instruct-multilingual-1b` | ContextualAI/ctxl-rerank-v2-instruct-multilingual-1b | rerank | text | — | — | paper | unverified |
| `ctxl-rerank-v2-instruct-multilingual` | `ctxl-rerank-v2-instruct-multilingual-2b` | ContextualAI/ctxl-rerank-v2-instruct-multilingual-2b | rerank | text | — | — | paper | unverified |
| `ctxl-rerank-v2-instruct-multilingual` | `ctxl-rerank-v2-instruct-multilingual-6b` | ContextualAI/ctxl-rerank-v2-instruct-multilingual-6b | rerank | text | — | — | paper | unverified |
| `embeddinggemma-2` | `embeddinggemma-2` | google/embeddinggemma-2 | embed | text, image, video | truncation 128/256/512/768 | the rcp-ndcg-vllm plugin (the embeddinggemma2-transformers-fold patch carrier) | card | unverified |
| `gemma-4-12b` | `gemma-4-12b-it` | google/gemma-4-12B-it | judge | text, image | — | — | — | unverified |
| `gemma-4-26b-a4b` | `gemma-4-26b-a4b-it` | google/gemma-4-26B-A4B-it | judge | text, image | — | — | — | unverified |
| `gemma-4-26b-a4b` | `gemma-4-26b-a4b-nvfp4` | nvidia/Gemma-4-26B-A4B-NVFP4 | judge | text, image | — | — | — | unverified |
| `gemma-4-31b` | `gemma-4-31b-it-nvfp4` | nvidia/Gemma-4-31B-IT-NVFP4 | judge | text, image | — | — | — | unverified |
| `gpt-oss-120b` | `gpt-oss-120b` | openai/gpt-oss-120b | judge | text | — | — | — | unverified |
| `harrier-oss-v1` | `harrier-oss-v1-270m` | microsoft/harrier-oss-v1-270m | embed | text | none | the rcp-ndcg-vllm plugin (the pooling-full-context patch carrier) | card | unverified |
| `harrier-oss-v1` | `harrier-oss-v1-0.6b` | microsoft/harrier-oss-v1-0.6b | embed | text | none | the rcp-ndcg-vllm plugin (the pooling-full-context patch carrier) | card | unverified |
| `harrier-oss-v1` | `harrier-oss-v1-27b` | microsoft/harrier-oss-v1-27b | embed | text | none | the rcp-ndcg-vllm plugin (the pooling-full-context patch carrier) | card | unverified |
| `jina-embeddings-v5-text` | `jina-embeddings-v5-text-nano` | jinaai/jina-embeddings-v5-text-nano | embed | text | truncation 32/64/128/256/512/768 | the rcp-ndcg-vllm plugin | card | unverified |
| `jina-embeddings-v5-text` | `jina-embeddings-v5-text-small` | jinaai/jina-embeddings-v5-text-small | embed | text | truncation 32/64/128/256/512/768/1024 | the rcp-ndcg-vllm plugin (the pooling-full-context patch carrier) | card | unverified |
| `jina-reranker-v3` | `jina-reranker-v3` | jinaai/jina-reranker-v3 | rerank | text | — | — | paper | unverified |
| `octen-embedding` | `octen-embedding-0.6b` | Octen/Octen-Embedding-0.6B | embed | text | none | — | paper | unverified |
| `octen-embedding` | `octen-embedding-4b` | Octen/Octen-Embedding-4B | embed | text | none | — | paper | unverified |
| `octen-embedding` | `octen-embedding-8b` | Octen/Octen-Embedding-8B | embed | text | none | — | paper | unverified |
| `pplx-embed-v1` | `pplx-embed-v1-0.6b` | perplexity-ai/pplx-embed-v1-0.6b | embed | text | none | the pplx model plugin | card | unverified |
| `pplx-embed-v1` | `pplx-embed-v1-4b` | perplexity-ai/pplx-embed-v1-4b | embed | text | none | the pplx model plugin | card | unverified |
| `pplx-embed-v2-context` | `pplx-embed-v2-context-9b-preview` | perplexity-ai/pplx-embed-v2-context-9b-preview | multi_vector | text | truncation 1024/2048 | the pplx model plugin | card | unverified |
| `pplx-embed-v2-late` | `pplx-embed-v2-late-0.6b` | perplexity-ai/pplx-embed-v2-late-0.6b | multi_vector | text, image | none | the pplx model plugin | card | unverified |
| `pplx-embed-v2-late` | `pplx-embed-v2-late-9b` | perplexity-ai/pplx-embed-v2-late-9b | multi_vector | text, image | none | the pplx model plugin | card | unverified |
| `qwen3-embedding` | `qwen3-embedding-0.6b` | Qwen/Qwen3-Embedding-0.6B | embed | text | truncation 32-1024 | — | card | unverified |
| `qwen3-embedding` | `qwen3-embedding-4b` | Qwen/Qwen3-Embedding-4B | embed | text | truncation 32-2560 | — | card | unverified |
| `qwen3-embedding` | `qwen3-embedding-8b` | Qwen/Qwen3-Embedding-8B | embed | text | truncation 32-4096 | — | card | unverified |
| `qwen3-reranker` | `qwen3-reranker-0.6b` | Qwen/Qwen3-Reranker-0.6B | rerank | text | — | — | paper | unverified |
| `qwen3-reranker` | `qwen3-reranker-4b` | Qwen/Qwen3-Reranker-4B | rerank | text | — | — | paper | unverified |
| `qwen3-reranker` | `qwen3-reranker-8b` | Qwen/Qwen3-Reranker-8B | rerank | text | — | — | paper | unverified |
| `qwen3-vl-embedding` | `qwen3-vl-embedding-2b` | Qwen/Qwen3-VL-Embedding-2B | embed | text, image, video | truncation 64-2048 | — | card | unverified |
| `qwen3-vl-embedding` | `qwen3-vl-embedding-8b` | Qwen/Qwen3-VL-Embedding-8B | embed | text, image, video | truncation 64-4096 | — | card | unverified |
| `qwen3-vl-reranker` | `qwen3-vl-reranker-2b` | Qwen/Qwen3-VL-Reranker-2B | rerank | text, image | — | — | card | unverified |
| `qwen3-vl-reranker` | `qwen3-vl-reranker-8b` | Qwen/Qwen3-VL-Reranker-8B | rerank | text, image | — | — | card | unverified |
| `qwen3.5-397b` | `qwen3.5-397b-a17b-nvfp4` | nvidia/Qwen3.5-397B-A17B-NVFP4 | judge | text, image | — | — | — | unverified |
| `qwen3.6-27b` | `qwen3.6-27b-fp8` | Qwen/Qwen3.6-27B-FP8 | judge | text, image | — | — | — | unverified |
| `qwen3.8-27b` | `qwen3.8-27b-fp8` | Qwen/Qwen3.8-27B-FP8 | judge | text, image | — | — | — | unverified |
| `qwen3.8-flash-next` | `qwen3.8-flash-next-nvfp4` | nvidia/Qwen3.8-Flash-Next-NVFP4 | judge | text, image | — | — | — | unverified |
| `qwen3.8-flash-next` | `qwen3.8-flash-next-fp8` | Qwen/Qwen3.8-Flash-Next-FP8 | judge | text, image | — | — | — | unverified |
| `topk-embed-v1` | `topk-embed-v1-xsmall` | topk-io/topk-embed-v1-xsmall | multi_vector | text, image | truncation 64/128/256/512/1024 | the topk model plugin | card | unverified |
| `topk-embed-v1` | `topk-embed-v1-small` | topk-io/topk-embed-v1-small | multi_vector | text, image | truncation 64/128/256/512/1024/2048 | the topk model plugin | card | unverified |
| `zembed-1` | `zembed-1-embedding` | zeroentropy/zembed-1-embedding | embed | text | projection 1280/640/320/160/80/40 | the rcp-ndcg-vllm plugin (the pooling-full-context patch carrier) | card | unverified |
| `zerank` | `zerank-1-reranker` | zeroentropy/zerank-1-reranker | rerank | text | — | — | paper | unverified |
| `zerank` | `zerank-1-small-reranker` | zeroentropy/zerank-1-small-reranker | rerank | text | — | — | paper | unverified |
| `zerank` | `zerank-2-reranker` | zeroentropy/zerank-2-reranker | rerank | text | — | — | paper | unverified |

The `id` is the variant's lowercased canonical Hub repository name; the `role` is what `rcp-ndcg` reads through
it (`embed`, `multi_vector`, `rerank`, `judge`); the `input` is what the checkpoint reads; the `reference` column
names where the equivalence reference comes from -- `paper` for a family whose `reference.py` is the paper's own
code path (ported from `experiments/paper/`), `card` for the model card's published usage, and `—` for a judge
recipe, which carries no equivalence reference; the `MRL` column is the
variant's declared Matryoshka head -- `truncation` or `projection` with the model card's supported output
dimensions, or `none` -- declared once in the client block (`mrl_kind` with `mrl_dims`/`mrl_range`; a
projection kind also names `mrl_projection`) and, where the engine's per-request `dimensions` path exists,
mirrored in `serve.hf_overrides` (`is_matryoshka`/`matryoshka_dimensions`), with the loader refusing a
serve gate and a client declaration that disagree. The recipes ship the checkpoint's full width; a run
selects `k` from the declared set. Budgets are explicit per
recipe: every recipe declares a tokenizer (injected as `model@revision` unless the family pins one),
`client.max_tokens` and (where the reference caps queries) `query_max_tokens`; over-budget content is cut
client-side at token boundaries with the template's anchors preserved, and every cut is recorded.

## Use a recipe from `rcp-ndcg`

A role config that names `recipe: <id>` takes its whole client block (api, tokenizer, budgets, template, media,
instruction mode) from the recipe; `base_url` and the other run-time fields stay on the config, and an explicit
content field must equal the recipe's or the config is refused naming both values. One exception: an MRL
selection -- `mrl_dim` or the engine-side `dimensions` -- is accepted when `k` is in the recipe's declared
`mrl_dims`/`mrl_range` (the recipe declares the kind and the set, the run selects) and refused naming the set
otherwise. `--retriever recipe:<id>` and
`--reranker recipe:<id>` are command-line shorthands. Recipes are resolved lazily, so `rcp-ndcg-vllm` must be
installed beside `rcp-ndcg`; without it the refusal is typed and its hint is the install line
([recipes and serving models](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/reference/recipes.md)).

## Model plugins

`topk-embed-v1-small`, `topk-embed-v1-xsmall` and the pplx checkpoints fold into
`rcp_ndcg_vllm/models/` under one
`vllm.general_plugins` entry point (the pplx wheel serves the contextual 9B, the late-interaction 0.6B and
9B, and registers the pplx-embed-v1 family's local config class).
Registration is lazy (`"module:Class"` strings): importing this package never imports vLLM or torch. A version
guard pins the tested vLLM line and refuses others loudly.

Engine-side patches (`rcp_ndcg_vllm.patches`) are opted in per recipe with `serve.patches`; `rcp-ndcg-vllm
serve` renders the declared names into the engine process's comma-separated `RCP_NDCG_VLLM_PATCHES`, and the
same entry point applies them (overriding an inherited value, so the engine runs exactly what the recipe
declares). The `pooling-full-context` patch backports
vllm-project/vllm#48039 (commit `e6fc81bc78`) for a pooling prompt of exactly `max_model_len` tokens under
chunked prefill, and retires itself with one inert log line once the engine image carries the fix. A recipe
that names a plugin also declares `plugin_architectures`, the architectures its engine registers (empty for
a patch-only carrier); the
behaviour fingerprint hashes exactly those modules' source beside the opted-in patches'.

The late-interaction pooler applies the recipe's declared keep-rules engine-side: a recipe that sets
`client.document_skip_engine_side` renders the same ids into `serve.hf_overrides.document_skip_token_ids`
together with the document role gate `document_skip_prefix_token_id` (the leading token id a document prompt
opens with -- the rule is document-side, so a query prompt keeps every position), and a recipe that sets
`client.media_keep_token_ids` (a media allowlist, e.g. topk-embed-v1's image-patch token) renders the same ids
into `serve.hf_overrides.document_keep_token_ids` (the allowlist is its own gate: a row carrying one of its ids
is a media document). The loader cross-checks the halves, the plugin's pooler drops the excluded positions from
the token ids it sees, and the wire carries only kept vectors -- the client then checks the reply's declared
kept count instead of slicing (vLLM v0.31.0's pooling route cannot return the engine's per-position token ids,
which is why the rules' home is the engine).

## Validation

The equivalence harness, the recorder, the reference cases and the GPU job tooling live in the unpublished
`rcp-ndcg-test` package ([its page](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/reference/rcp-ndcg-test.md));
authors of new recipes start at [add a serving
recipe](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/how-to/add-a-model.md). The judging pipeline and
the metric are `rcp-ndcg` and `rcp-ndcg-core`.

The T4 end-to-end run scenarios (`scenarios/<id>.yaml`, schema `schema/scenario.schema.json`) and their in-pod
driver (`python -m rcp_ndcg_test.e2e`) are described in [release candidates and the GPU
waves](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/how-to/release-candidates.md).

The stage-2 pairs files (`pairs/<id>.jsonl` and their `manifest.json`) come from the deterministic request
generator `python -m rcp_ndcg_test.observe.requests`; the wave's observation-corpus step, the T3 quality stage
and the negative controls are `run_wave.py`'s `--record-corpus`, `--quality` and `--controls`. The corpus
format is documented in `schema/observation-corpus.md`; its one reader is `rcp_ndcg_test.corpus`. Every
corpus is keyed by the recipe behaviour fingerprint (`rcp_ndcg_test.fingerprint`); `python -m rcp_ndcg_test.changes`
selects the recipes to re-record and diffs two corpora of one recipe. The verified fake engines that replay the
corpora are `rcp_ndcg_test.engines` ([use the verified fake
engines](https://github.com/cohere-ai/rcp-ndcg/blob/main/docs/how-to/use-verified-fake-engines.md)).

## License

Apache-2.0: `LICENSE` and `NOTICE` ship in every wheel and sdist.
Security reports: [SECURITY.md](https://github.com/cohere-ai/rcp-ndcg/blob/main/SECURITY.md) (GitHub private
vulnerability reporting).
