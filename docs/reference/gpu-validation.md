# The GPU validation ledger

Every shipped recipe is validated on GPUs before a tag: served on its declared engine image, its replies
compared against its own reference implementation (the model card's published path or the paper's code),
and its observation corpus recorded. Every *retrieval* variant is compared against its reference that way;
a judge recipe carries no equivalence reference, so a judge variant is validated by `rcp-ndcg judge check`
and the T4 scenarios instead. The corpora whose replay needs them are committed under
`rcp-ndcg-test/corpora/`. This page is the release's record of that work — one row per variant:
the release candidate and the wave whose evidence the row records, the date that wave finished, and the
behaviour fingerprint the tree computes today (the key a wave's evidence must carry).

The machine-readable record is each family's `status` block in
`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/<family>/family.yaml` (the `state`, the engine `image`, the `date`
and the `report` pointer), and the [recipes and serving models](recipes.md) page explains the surface. A
recipe's behaviour fingerprint keys everything that can change what the model returns — the model revision,
the engine image and its version floor, the `serve` block, the engine-side plugin code, the tokenizer, the
template and the request-shaping client fields. A wave's evidence counts for a recipe only when the wave
ran against the fingerprint the tree computes today.

## The ledger

| recipe | role | engine image | verified at | date | fingerprint | state |
|---|---|---|---|---|---|---|
| `ctxl-rerank-v2-instruct-multilingual-1b` | rerank | `vllm/vllm-openai:v0.31.0` | `rc0-202610131335` | 2026-10-13 | `9f143b140d85…` | verified |
| `ctxl-rerank-v2-instruct-multilingual-2b` | rerank | `vllm/vllm-openai:v0.31.0` | `rc0-202610131335` | 2026-10-13 | `00934b2089bb…` | verified |
| `ctxl-rerank-v2-instruct-multilingual-6b` | rerank | `vllm/vllm-openai:v0.31.0` | `rc0-202610131335` | 2026-10-13 | `b9c99ae7e998…` | verified |
| `embeddinggemma-2` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `cc5e5c3c0766…` | verified |
| `gemma-4-12b-it` | judge | `vllm/vllm-openai:v0.31.0` | `rc0-202610121840` | 2026-10-12 | `315c94e5d8ea…` | verified |
| `gemma-4-26b-a4b-it` | judge | `vllm/vllm-openai:v0.31.0` | `rc0-202610121840` | 2026-10-12 | `ae68e4fd1c72…` | verified |
| `gemma-4-26b-a4b-nvfp4` | judge | `vllm/vllm-openai:v0.31.0` | `rc0-202610121840` | 2026-10-12 | `3359e28b3ace…` | verified |
| `gemma-4-31b-it-nvfp4` | judge | `vllm/vllm-openai:v0.31.0` | `rc0-202610121840` | 2026-10-12 | `60f770f4a3fb…` | verified |
| `gpt-oss-120b` | judge | `vllm/vllm-openai:v0.31.0` | `rc0-202610121840` | 2026-10-12 | `b947cc9ca257…` | verified |
| `harrier-oss-v1-270m` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `3885d394dee2…` | verified |
| `harrier-oss-v1-0.6b` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `302a95fbef04…` | verified |
| `harrier-oss-v1-27b` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `a9e0e1729f87…` | verified |
| `jina-embeddings-v5-text-nano` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `ad7dac20488f…` | verified |
| `jina-embeddings-v5-text-small` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `3ad7d8d01be3…` | verified |
| `jina-reranker-v3` | rerank | `vllm/vllm-openai:v0.31.0` | `rc0-202610130615` | 2026-10-13 | `f00e098f01c5…` | verified |
| `octen-embedding-0.6b` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610130615` | 2026-10-13 | `3461b2db6192…` | verified |
| `octen-embedding-4b` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610130615` | 2026-10-13 | `4023e1b3bcec…` | verified |
| `octen-embedding-8b` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610130615` | 2026-10-13 | `0e1397bf030f…` | verified |
| `pplx-embed-v1-0.6b` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `b0b22535962e…` | verified |
| `pplx-embed-v1-4b` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `b103fd481e89…` | verified |
| `pplx-embed-v2-context-9b-preview` | multi_vector | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `6ba637080c9e…` | verified |
| `pplx-embed-v2-late-0.6b` | multi_vector | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `b33c2371961b…` | verified |
| `pplx-embed-v2-late-9b` | multi_vector | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `2e41cae5f305…` | verified |
| `qwen3-embedding-0.6b` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610120530` | 2026-10-12 | `3d0c1f1f3c0f…` | verified |
| `qwen3-embedding-4b` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610120530` | 2026-10-12 | `a027098bfd13…` | verified |
| `qwen3-embedding-8b` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610120530` | 2026-10-12 | `002e4ab3450d…` | verified |
| `qwen3-reranker-0.6b` | rerank | `vllm/vllm-openai:v0.31.0` | `rc0-202610120530` | 2026-10-12 | `fb5cef35d44a…` | verified |
| `qwen3-reranker-4b` | rerank | `vllm/vllm-openai:v0.31.0` | `rc0-202610120530` | 2026-10-12 | `80bba2ca3db2…` | verified |
| `qwen3-reranker-8b` | rerank | `vllm/vllm-openai:v0.31.0` | `rc0-202610120530` | 2026-10-12 | `5c21684d7e38…` | verified |
| `qwen3-vl-embedding-2b` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610130615` | 2026-10-13 | `db8ecc32e057…` | verified |
| `qwen3-vl-embedding-8b` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610130615` | 2026-10-13 | `909c048f2308…` | verified |
| `qwen3-vl-reranker-2b` | rerank | `vllm/vllm-openai:v0.31.0` | `rc0-202610130615` | 2026-10-13 | `07c5db0a959a…` | verified |
| `qwen3-vl-reranker-8b` | rerank | `vllm/vllm-openai:v0.31.0` | `rc0-202610130615` | 2026-10-13 | `f230864643fb…` | verified |
| `qwen3.5-397b-a17b-nvfp4` | judge | `vllm/vllm-openai:v0.31.0` | `rc0-202610121840` | 2026-10-12 | `f507510505b7…` | verified |
| `qwen3.6-27b-fp8` | judge | `vllm/vllm-openai:v0.31.0` | `rc0-202610121840` | 2026-10-12 | `eeca19d80805…` | verified |
| `qwen3.8-27b-fp8` | judge | `vllm/vllm-openai:v0.31.0` | `rc0-202610121840` | 2026-10-12 | `70ac81a6c78e…` | verified |
| `qwen3.8-flash-next-nvfp4` | judge | `vllm/vllm-openai:v0.31.0` | `rc0-202610121840` | 2026-10-12 | `5c4bf3df96d4…` | verified |
| `qwen3.8-flash-next-fp8` | judge | `vllm/vllm-openai:v0.31.0` | `rc0-202610121840` | 2026-10-12 | `553fc2bf43fc…` | verified |
| `topk-embed-v1-xsmall` | multi_vector | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `01c4e57b0343…` | failed |
| `topk-embed-v1-small` | multi_vector | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `1ac72016965a…` | failed |
| `zembed-1-embedding` | embed | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `21bb1fd31dc8…` | verified |
| `zerank-1-reranker` | rerank | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `f4ca536b33ae…` | verified |
| `zerank-1-small-reranker` | rerank | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `e13348f43de3…` | verified |
| `zerank-2-reranker` | rerank | `vllm/vllm-openai:v0.31.0` | `rc0-202610140620` | 2026-10-14 | `460fbc15e393…` | verified |

## Dispositions

- **`topk-embed-v1-small` and `topk-embed-v1-xsmall` are `failed`.** Their end-to-end validation does not
  meet the stage-2 per-token cosine floor on the available engine configurations: bf16 and fp16 both fail
  (and the failure is dtype-independent at the text model's first layer), and fp32 — which clears the
  vision tower — cannot be served, because the engine's GatedDeltaNet kernels refuse fp32. The engine was
  also run on its own vendored flash-linear-attention prefill path (`--gdn-prefill-backend triton`), which
  diverges identically, so the residual is not a selectable kernel. The full measurements — the per-row
  per-token minima, the per-layer cosines, the three backends, and the text/image split — are in the
  family's notes. The evidence is keyed honestly: the wave that first measured the failure
  (`rc0-202610130615`, 2026-10-13) ran at a superseded behaviour fingerprint (`1fdf2847…`, before the
  plugin-module keying moved it), so the gate was re-run at the current key on 2026-10-14 in the
  `rc0-202610140620` wave's `plugin-carriers-vllm` sub-wave and failed again with the same signature —
  stage 1 passes, stage 2's per-token cosine floor fails. The two rows above carry that 2026-10-14 wave.
  Nothing is deferred past the tag: the state stays `failed`, and the gate's disposition stays the owner's
  explicit decision, recorded in the release notes; a recipe in this state ships only under that decision,
  never silently.
- **The `embeddinggemma-2` row's engine image is stock `vllm/vllm-openai:v0.31.0`** (vllm-7): the verifying
  waves ran that image with the recipe's plugin backport, not the nightly
  digest the row used to name. The correction moves the recipe's behaviour fingerprint (the engine image
  is a key), so the row's fingerprint cell names the value the tree computes today (`cc5e5c3c0766…`);
  the `rc0-202610140620` wave ran at the superseded nightly-keyed `d0e1e968…`, and the release's
  re-recording waves re-run this row at the current key before the tag.
- **One reference case remains `pending_gpu`**: `rcp-ndcg-test/cases/topk-embed-v1-small/query-cap-1024.yaml`,
  in the family whose gate the owner dispositioned. Its query renders past the recipe's declared
  `client.query_max_tokens: 1024`, so the client's fit cuts it, and the case's `expected.kind` is `none`,
  which carries no values by design: it observes that at-cap query rather than comparing scores, which is
  also why the case-fill skips it. The video case this bullet used to name,
  `rcp-ndcg-test/cases/qwen3-vl-embedding-2b/video-short-single.yaml`, was filled from the reference
  implementation on cuda on 2026-10-14 and reads `status: reproduced`. These are validation fixtures, not
  recipe gates.
- **The ViDoRe retrieval view carries a documented waiver** (`rcp-ndcg-test/tests/e2e/test_golden_replay.py`):
  the golden's retrieval view needs ViDoRe page-image embeddings, and no committed corpus of the VL
  embedder observes a ViDoRe-sourced image yet. The tripwire fails the moment one does, naming the work.

## What this page is not

It is not the recipes' contract — that is `rcp-ndcg-vllm/schema/` and the goldens under
`rcp-ndcg-test/tests/recipes/golden/`. It is not the observation corpora — those are committed under
`rcp-ndcg-test/corpora/` and keyed by the same fingerprints. And it names no infrastructure: the wave
outputs live in the release's own evidence store, referenced by the family `status` blocks.
