# The GPU validation ledger

The release validates every shipped recipe on GPUs before a tag. Each recipe is served on its declared engine image,
and its replies are compared against its own reference implementation. The reference implementation is the published
path of the model card or the code of the paper. The release also records the observation corpus of the recipe. Every
*retrieval* variant is compared against its reference in that way. A judge recipe has no equivalence reference, so
`rcp judge check` and the T4 scenarios validate a judge variant instead. The corpora that a replay needs are
committed under `rcp-test/corpora/`. This page is the record of that work for the release. It has one row per
variant. A row gives the release candidate and the wave whose evidence the row records. It also gives the date that
the wave finished, and the behaviour fingerprint that the tree computes today. That fingerprint is the key that the
evidence of a wave must carry.

The machine-readable record is the `status` block of each family in
`rcp-vllm/src/rcp_vllm/recipes/<family>/family.yaml`. The block holds the `state`, the engine `image`, the
`date` and the `report` pointer. The [recipes and serving models](recipes.md) page explains the surface. A recipe's
behaviour fingerprint is a key over everything that can change what the model returns. The key covers the model
revision, the engine image and its version floor. It also covers the `serve` block and the engine-side plugin code,
the tokenizer, the template and the request-shaping client fields. The evidence of a wave counts for a recipe only
when it ran against the current fingerprint of the tree.

## The v0.33.0 pin: every row is the wave that ran on it

The pin moved the `engine.image` of every recipe to `vllm/vllm-openai:v0.33.0`. The engine image is one of the inputs
of the fingerprint, so every key moved. **Every corpus is therefore recorded again on the 0.33 image. Every judge
recipe is probed again. No recipe returns to `verified` before that work.** The rows below are the record of that
re-record. 42 are the 0.33 waves' own (rc2 through rc10). 2 are the owner-disposed `topk-embed-v1` variants' gate
wave; no row reads `unverified`. A row never claims a wave that did not run. The ledger regenerates the row from the
wave that ran. It is never relabelled with a key that the wave did not record. A recipe is `verified` only beside the
image that it pins.

The key of a row is derived at the **wave's RC commit**. That is the commit that every wave summary records. The
derivation checks that commit's tree out of this repository, and it runs the behaviour fingerprint of that tree. The
ledger writes the row only when that key is the key that the tree computes today. A wave that measured another
behaviour is therefore refused by name, not recorded. The waves that record an observation corpus also write the key
themselves, and those recorded keys cross-check the derivation. The ledger and the `status` blocks come from
`python -m rcp_test.jobs.ledger --evidence <mirror> [--write]`. Never write them manually.

The owner holds one exception: the two `topk-embed-v1` variants. Their gate failure below is recorded evidence, not a
pending row. Their state returns to `failed` only from a wave that re-runs that gate. They ship only when the explicit
disposition list of the owner names them. The pin does not change that disposition.

## The ledger

| recipe | role | engine image | verified at | date | fingerprint | state |
|---|---|---|---|---|---|---|
| `ctxl-rerank-v2-instruct-multilingual-1b` | rerank | `vllm/vllm-openai:v0.33.0` | `rc3` | 2026-11-09 | `7b66b6690d12…` | verified |
| `ctxl-rerank-v2-instruct-multilingual-2b` | rerank | `vllm/vllm-openai:v0.33.0` | `rc3` | 2026-11-09 | `ad0702c8d30c…` | verified |
| `ctxl-rerank-v2-instruct-multilingual-6b` | rerank | `vllm/vllm-openai:v0.33.0` | `rc3` | 2026-11-09 | `1f0bbace8647…` | verified |
| `embeddinggemma-2` | embed | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `66f50262a43b…` | verified |
| `gemma-4-12b-it` | judge | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `90fb5794acfb…` | verified |
| `gemma-4-26b-a4b-it` | judge | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `bf335350445d…` | verified |
| `gemma-4-26b-a4b-nvfp4` | judge | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `dbe26de2badf…` | verified |
| `gemma-4-31b-it-nvfp4` | judge | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `ff3aed331b79…` | verified |
| `gpt-oss-120b` | judge | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `2d665a553de9…` | verified |
| `harrier-oss-v1-270m` | embed | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `4bddf11cc828…` | verified |
| `harrier-oss-v1-0.6b` | embed | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `63fe92dfa36d…` | verified |
| `harrier-oss-v1-27b` | embed | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `1770d38c8ccb…` | verified |
| `jina-embeddings-v5-text-nano` | embed | `vllm/vllm-openai:v0.33.0` | `rc6` | 2026-11-09 | `d7a5ffc910b3…` | verified |
| `jina-embeddings-v5-text-small` | embed | `vllm/vllm-openai:v0.33.0` | `rc3` | 2026-11-09 | `ce3a71ba2a6d…` | verified |
| `jina-reranker-v3` | rerank | `vllm/vllm-openai:v0.33.0` | `rc4` | 2026-11-09 | `02a5ddbe3c2e…` | verified |
| `octen-embedding-0.6b` | embed | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `44f36e3a7e5f…` | verified |
| `octen-embedding-4b` | embed | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `d6798d317a36…` | verified |
| `octen-embedding-8b` | embed | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `03027d8b6e67…` | verified |
| `pplx-embed-v1-0.6b` | embed | `vllm/vllm-openai:v0.33.0` | `rc8` | 2026-11-10 | `e488235f2240…` | verified |
| `pplx-embed-v1-4b` | embed | `vllm/vllm-openai:v0.33.0` | `rc8` | 2026-11-10 | `a20543637c05…` | verified |
| `pplx-embed-v2-context-9b-preview` | multi_vector | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `a00142136e43…` | verified |
| `pplx-embed-v2-late-0.6b` | multi_vector | `vllm/vllm-openai:v0.33.0` | `rc10` | 2026-11-10 | `14280b0d0a29…` | verified |
| `pplx-embed-v2-late-9b` | multi_vector | `vllm/vllm-openai:v0.33.0` | `rc10` | 2026-11-10 | `fef737c045cb…` | verified |
| `qwen3-embedding-0.6b` | embed | `vllm/vllm-openai:v0.33.0` | `rc5` | 2026-11-09 | `6063d7cd725e…` | verified |
| `qwen3-embedding-4b` | embed | `vllm/vllm-openai:v0.33.0` | `rc6` | 2026-11-09 | `a511a15e7b8d…` | verified |
| `qwen3-embedding-8b` | embed | `vllm/vllm-openai:v0.33.0` | `rc6` | 2026-11-09 | `09e160550790…` | verified |
| `qwen3-reranker-0.6b` | rerank | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `9b165ceb168f…` | verified |
| `qwen3-reranker-4b` | rerank | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `2ce04f46bfcd…` | verified |
| `qwen3-reranker-8b` | rerank | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `f864df1ba669…` | verified |
| `qwen3-vl-embedding-2b` | embed | `vllm/vllm-openai:v0.33.0` | `rc5` | 2026-11-09 | `2a708ab3ba3f…` | verified |
| `qwen3-vl-embedding-8b` | embed | `vllm/vllm-openai:v0.33.0` | `rc6` | 2026-11-09 | `336f3b423f7b…` | verified |
| `qwen3-vl-reranker-2b` | rerank | `vllm/vllm-openai:v0.33.0` | `rc3` | 2026-11-09 | `970db80eafdd…` | verified |
| `qwen3-vl-reranker-8b` | rerank | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `34178ed7e8f8…` | verified |
| `qwen3.5-397b-a17b-nvfp4` | judge | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `a74c53fda840…` | verified |
| `qwen3.6-27b-fp8` | judge | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `e4809d6512e7…` | verified |
| `qwen3.8-27b-fp8` | judge | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `85b5285984ee…` | verified |
| `qwen3.8-flash-next-nvfp4` | judge | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `275ab047c935…` | verified |
| `qwen3.8-flash-next-fp8` | judge | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `aefe310360ac…` | verified |
| `topk-embed-v1-xsmall` | multi_vector | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `fa00268b7a95…` | failed |
| `topk-embed-v1-small` | multi_vector | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `54fadf9869a1…` | failed |
| `zembed-1-embedding` | embed | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `1909add9dbf7…` | verified |
| `zerank-1-reranker` | rerank | `vllm/vllm-openai:v0.33.0` | `rc6` | 2026-11-09 | `d2df9d554c4b…` | verified |
| `zerank-1-small-reranker` | rerank | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `837099ca3304…` | verified |
| `zerank-2-reranker` | rerank | `vllm/vllm-openai:v0.33.0` | `rc2` | 2026-11-09 | `1bb045f2f8f1…` | verified |

## Dispositions

- **`topk-embed-v1-small` and `topk-embed-v1-xsmall` are the gate failure that the owner dispositioned**. Their
  end-to-end validation does not meet the stage-2 per-token cosine floor on the available engine configurations. Both
  bf16 and fp16 fail. The dtype does not change the failure at the first layer of the text model. The fp32 mode clears
  the vision tower, but it cannot be served: the GatedDeltaNet kernels of the engine refuse fp32. The engine also ran on
  its own vendored flash-linear-attention prefill path (`--gdn-prefill-backend triton`), and that path diverges in the
  same way. The residual is therefore not a selectable kernel. The notes of the family hold the full measurements. They
  give the per-row per-token minima, the per-layer cosines, the three backends and the text/image split. The evidence
  has the key of the wave that measured it. The wave that first measured the failure (`rc0-202610130615`, 2026-10-13)
  ran at a superseded behaviour fingerprint (`1fdf2847…`). The plugin-module keying moved it later. The gate was
  therefore re-run at the current key on 2026-10-14. The re-run was in the `plugin-carriers-vllm` sub-wave of the
  `rc0-202610140620` wave. It failed again with the same signature: stage 1 passes, and the stage-2 per-token cosine
  floor fails. The two rows above record that 2026-10-14 wave. Nothing is deferred past the tag. The disposition of the
  gate stays the explicit decision of the owner, and the release notes record it. A recipe in this state ships only
  under that decision, never silently. On the v0.33.0 pin the `gate-topk-embed-v1` wave re-ran the gate on the pinned
  image (rc2) and failed it again. The two rows therefore read `failed` beside the date and the report of that wave.
  That is the state that the schema has for a gate that failed, and never `verified` without a green re-run.
- **The `embeddinggemma-2` row belongs to the rc2 equivalence wave**. That wave ran on the stock
  `vllm/vllm-openai:v0.33.0` image with the plugin and patches of the recipe. It ran the gate at `66f50262a43b…`, which
  is the key that the tree computes today. The 0.0.1 record was different: the waves ran the stock 0.31 image with the
  backport of the fold. The cell of the old page held the key that the tree computed at `fcb87a51`. It did not hold the
  key that the wave recorded. The image is a key, so the pin superseded that record, and the re-record replaces it. The
  fold is gone (the parity decision), and the recipe is a patch-only carrier. See [recipes and serving
  models](recipes.md#engine-side-patches).
- **One reference case remains `pending_gpu`**: `rcp-test/cases/topk-embed-v1-small/query-cap-1024.yaml`, in the
  family whose gate the owner dispositioned. Its query renders past the declared `client.query_max_tokens: 1024` of
  the recipe, so the fit of the client cuts it. The `expected.kind` of the case is `none`, which carries no values by
  design. The case therefore observes that at-cap query instead of comparing scores, and the case-fill skips it for
  the same reason. The video case that this bullet used to name,
  `rcp-test/cases/qwen3-vl-embedding-2b/video-short-single.yaml`, was filled from the reference implementation on
  cuda on 2026-10-14. It reads `status: reproduced`. These are validation fixtures, not recipe gates.
- **The ViDoRe retrieval view has a documented waiver** (`rcp-test/tests/e2e/test_golden_replay.py`). The
  retrieval view of the golden needs ViDoRe page-image embeddings. No committed corpus of the VL embedder observes a
  ViDoRe-sourced image yet. The tripwire fails as soon as one does, and it names the work.

## What this page is not

This page is not the contract of the recipes. That contract is `rcp-vllm/schema/` and the goldens under
`rcp-test/tests/recipes/golden/`. This page is not the observation corpora. Those are committed under
`rcp-test/corpora/`, and the same fingerprints key them. This page names no infrastructure. The wave outputs live
in the evidence store of the release, and the family `status` blocks reference them.
