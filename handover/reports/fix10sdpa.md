# Lane fix10sdpa — embeddinggemma-2's attention backend (the last recipe)

**Lane:** `lane/fix10sdpa`. **Base:** `rfc-0001` at `58847318` — fetched again before the gate and still the
base (an ancestor of every commit here), so no merge commit exists in this lane.
**Gate:** `timeout 7200 bin/gate lane/fix10sdpa` on `93ae1908` (slot 3) — **GATE: PASS**, every step exit 0.
No GPU work in this lane; nothing pushed; `rfc-0001` was not merged into.

## 1. Status

Done, path 1 of the brief: the embeddinggemma-2 recipe's engine now runs on **a plugin-provided attention
backend registered as `Backend.TORCH_SDPA`** — the mechanism the engine's own error names — and pins
`--attention-backend TORCH_SDPA`. The class is an encoder-only backend over
`torch.nn.functional.scaled_dot_product_attention`: no compile, no block mask, no KV cache, no
head-dimension cap, one SDPA call per packed request, the sliding layers' window as a mask. It is
registered from the new `torch-sdpa-attention` patch, which the recipe opts into.

The GPU wave's T0 model load on the next RC is the confirmation; this lane runs no GPU. The fallback (a
plugin patch of the flex backend's compile mode) was not needed and is not implemented: it would
monkeypatch torch/vLLM internals where the preferred path registers a documented backend class.

## 2. Commits

| commit | what |
|---|---|
| `db77caf6` | the plugin registers an encoder-only TORCH_SDPA attention backend (the backend module, the patch module, the patch/models wiring, the plugin CPU tests) |
| `015f5f33` | the embeddinggemma-2 engine runs on the plugin's registered TORCH_SDPA backend (family.yaml, the family test, the 14 regenerated plugin goldens, `docs/reference/recipes.md`, the package README's catalog row) |
| `93ae1908` | the fix10sdpa CHANGELOG entry (the two FLEX_ATTENTION bullets become the shipped behaviour) |

No co-author lines, no attribution, nothing pushed.

## 3. The path chosen

**The rejection chain (rounds 7-10, unchanged from r8/r9 except the last step).** The model's head
dimensions are heterogeneous — sliding layers 256 with 2 KV heads, full layers 512 with 1
(`transformers_utils/configs/gemma4.py` reads `per_layer_config` for a heterogeneous Gemma4 config).
TRITON_ATTN cannot compile its prefill kernel for head_dim 512 (`triton.runtime.errors.OutOfResources:
out of resource: shared memory, Required: 266240, Hardware limit: 232448`); FA4's kernel rejects it
(`FlashAttention forward only supports head dimension at most 256`); `TORCH_SDPA`'s enum value is the
empty string ("this tag is only used for ViT"), so the manual-selection path raises
`ValueError: Backend TORCH_SDPA must be registered before use. Use register_backend(Backend.TORCH_SDPA,
'your.module.YourClass')`; FLEX_ATTENTION was the one image backend that cleared the dimensions and dies
in its own `torch.compile`/CUDAGraph interaction
(`torch._dynamo.exc.InternalTorchDynamoError: RuntimeError: Error: accessing tensor output of CUDAGraphs
that has been overwritten by a subsequent run` at `torch/nn/attention/flex_attention.py:2369`), which
`--enforce-eager` does not stop because the backend compiles independently of the engine's cudagraph mode
(the round-10 wave still failed with it).

**The shipped path.** The engine's error message is the interface: register a class under
`Backend.TORCH_SDPA`. The plugin now ships `rcp_ndcg_vllm.models.embedding_gemma2.attention`:

- `TorchSdpaAttentionBackend`: `get_name() == "TORCH_SDPA"`, `supported_dtypes` fp16/bf16/fp32, no
  head-size cap (`get_supported_head_sizes() == []`), `supports_attn_type` = ENCODER_ONLY,
  `supports_sliding_window`/`supports_non_causal` true, `forward_includes_kv_cache_update` false;
- `TorchSdpaAttentionMetadata`/`Builder`: the batch's `query_start_loc`, `num_reqs` and
  `num_actual_tokens` copied out of the engine's `CommonAttentionMetadata` (encoder-only attention keeps
  no KV cache and no block table, so nothing else exists to copy);
- `TorchSdpaAttentionImpl`: slices the packed q/k/v to `num_actual_tokens`, walks the requests by the
  boundary tensor, and calls `scaled_dot_product_attention` once per request (`enable_gqa` for the
  grouped-query layers, `scale` as the model passes it — 1.0, matching the checkpoint's own
  `self.scaling = 1.0`); a sliding layer gets a boolean `|i - j| <= W - 1` mask (vLLM's W is the
  checkpoint's `sliding_window` + 1 = 513, so the band is 512, exactly HF's
  `create_bidirectional_sliding_window_mask` with `config.sliding_window` = 512); a profiling run
  (`attn_metadata is None`) zeroes the output; `do_kv_cache_update` is a no-op; fused output
  quantization, ALiBi, a logits soft cap and KV sharing raise loudly.

The patch module `rcp_ndcg_vllm.patches.torch_sdpa_attention` (`PATCH_NAME =
"torch-sdpa-attention"`) imports the class inside `apply()` and calls
`register_torch_sdpa_backend()`, which registers the class path built from the module's own `__name__`
and the class name (one home) — so importing `rcp_ndcg_vllm.patches` stays free of torch/vLLM
(`tests/test_no_torch.py`), and only an engine process that opted the patch in overrides the enum
member. The class's module is keyed in `ARCHITECTURE_MODULES["EmbeddingGemma2Model"]` and
`LAZY_MODEL_MODULES`, so a change to it moves the recipe's behaviour fingerprint.

**The recipe.** `serve.patches: [embeddinggemma2-transformers-fold, torch-sdpa-attention]` and
`serve.extra_args: [..., --attention-backend, TORCH_SDPA, --enforce-eager]`. `--enforce-eager` is kept
as the round-9 safety pin (the SDPA backend compiles nothing, so it is inert for this backend and keeps
the engine's compilation and cudagraph modes NONE).

## 4. The interface read (the pinned nightly)

The image is the recipe's digest
(`vllm/vllm-openai:nightly-8cbd5d03006c33185f402249ff2b448efd594986@sha256:b25e8a04...`); its own python
reports **vLLM 0.31.1rc1.dev173+g8cbd5d030, torch 2.13.0+cu130, transformers 5.19.0**. The interface
was read from the image's source, not from the PR:

| file | what it fixes |
|---|---|
| `vllm/v1/attention/backend.py` | `AttentionBackend` (`get_name`/`get_impl_cls`/`get_builder_cls`/`supports_attn_type`/`supports_sliding_window`/`get_supported_head_sizes`), `AttentionMetadataBuilder` (the `(kv_cache_spec, layer_names, vllm_config, device)` constructor and `build`), `AttentionImpl` (`__init__` signature, `forward(layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale)`), `AttentionType`, `CommonAttentionMetadata` |
| `vllm/v1/attention/backends/registry.py` | `AttentionBackendEnum.TORCH_SDPA = ""`; `register_backend(backend, class_path)` writes `_ATTN_OVERRIDES`; `get_path()` raises the exact `must be registered before use` error; `get_class()` resolves the path |
| `vllm/platforms/cuda.py` | `get_attn_backend_cls`: a selected backend resolves through `_get_attn_backend_class` and is accepted only if `validate_configuration(...)` returns no reasons |
| `vllm/v1/worker/gpu/model_states/encoder_only.py` | the V2 encoder-only model state: groups ENCODER_ONLY layers by `(backend, num_heads, num_kv_heads, head_size)` (two groups here), builds one metadata builder per group, binds `torch.empty(0)` as the layer's KV cache, and builds the metadata with a dummy block table/slot mapping |
| `vllm/v1/attention/backends/cpu_attn.py` | the encoder-only reference (no cache read, the packed tokens attended per request) |
| `vllm/v1/attention/backends/flex_attention.py` | the failing backend; its `create_block_mask` compiled with `mode="reduce-overhead"` is what the round-9/10 error names |
| `vllm/transformers_utils/configs/gemma4.py` | `gemma4_layer_config` resolves the per-layer config (full layers 512/1, sliding 256/2), which is why the class must have no head-size cap |
| `vllm/transformers_utils/model_arch_config_convertor.py` | `is_mm_prefix_lm` is False for `embedding_gemma2` (no attribute, not in the fallback list), so `use_mm_prefix` does not reach `validate_configuration` |

**The real-interface check (CPU, in the image, plugin `src` on `PYTHONPATH`).** A scratch script ran the
class against the image's own vLLM: `register_torch_sdpa_backend()` then
`AttentionBackendEnum.TORCH_SDPA.get_class()` returned the plugin's class;
`validate_configuration` returned no reasons for both layer shapes (head 256 / 2 KV heads / sliding 513
and head 512 / 1 KV head / no window, bf16, attn_type encoder_only); the builder built from a real
`CommonAttentionMetadata`; and the impl's forward matched a manual non-causal GQA reference exactly on a
packed two-request batch, request 1 was untouched by rewritten request-0 tokens, and a window of 4 masked
`|i - j| > 3` exactly. Output: `ALL CHECKS PASSED`.

**The same read corrected one recipe claim.** The pinned digest's transformers *is* 5.19.0 and
`transformers/models/embedding_gemma2/` *is* present, and the image's `ModelRegistry.get_supported_archs()`
carries `EmbeddingGemma2Model` with `MODELS_CONFIG_MAP` carrying its hook: `install_fold()` returns on its
import probe and `register_embedding_gemma2()` finds the architecture and hook present, so both plugin
registrations are inert on that digest (the fold stays as the carrier for an engine image that lacks
them). The notes and the header comment now record this as the round-10 read; the r1-r5 history is kept
as the wave record it is.

## 5. Red-first evidence

1. **The plugin tests** (new file, `rcp-ndcg-vllm/tests/models/test_embedding_gemma2_attention.py`, run
   with the gate's models venv — real torch, a fake `vllm` surface): with the backend module removed,
   every test errored at setup:
   `ModuleNotFoundError: No module named 'rcp_ndcg_vllm.models.embedding_gemma2.attention'`
   (`8 errors`). With the module in place: `110 passed, 8 skipped` for the whole models suite (the
   pre-existing count plus the 8 new tests).
2. **The family test** (the new expectations against the old recipe, `family.yaml` temporarily restored
   from `HEAD`): `4 failed, 11 passed, 4 skipped` —
   `AssertionError: serve.extra_args == ['--media-io-kwargs', '{"video": {"fps": 60, "max_frames": 32}}',
   '--attention-backend', 'FLEX_ATTENTION', '--enforce-eager'], expected [... 'TORCH_SDPA' ...]` for
   `test_recipe_contract_pins_every_field`, plus `test_two_contract_mutants_are_red`,
   `test_serve_argv_carries_the_pinned_flags` and
   `test_the_engine_uses_the_plugins_registered_torch_sdpa_backend` (the argv, the patch opt-in and the
   notes evidence). After the recipe change: `15 passed, 4 skipped` (network-gated) / `64 passed` with
   the goldens.

## 6. Verification

| command | result |
|---|---|
| `ruff format --check .` / `ruff check .` | 630 files formatted / all checks passed |
| `basedpyright` | 0 errors, 0 warnings, 0 notes |
| `heavy uv run --no-sync pytest tests/ -q -n 8` | 4145 passed, 155 skipped |
| `pytest tests/contract tests/docs -q` | 413 passed, 107 skipped |
| `pytest rcp-ndcg-test/tests -q` (faulthandler, 300 s per-test guard) | 1180 passed, 229 skipped |
| the plugin package suite (`rcp-ndcg-vllm[test]`, no torch: `--ignore=tests/models`) | 50 passed |
| the models suite (torch CPU, no vLLM: `tests/models tests/test_no_torch.py`) | 110 passed, 8 skipped |
| the recipe family tests + goldens (`RCP_NDCG_VLLM_TOKENIZER_CACHE` set) | 64 passed, 4 skipped |
| `mkdocs build --strict` | built |
| the image's real-interface check (`docker run` the pinned digest, plugin `src` mounted) | `ALL CHECKS PASSED` (registration, validation for 256/512, the manual forward reference, the boundaries, the sliding band) |
| `timeout 7200 bin/gate lane/fix10sdpa` on `93ae1908` (slot 3) | **GATE: PASS** — ruff-check, ruff-format, basedpyright, pytest, contract-docs, mkdocs, test-pkg (1180 passed), recipes (0 outside the baseline), vllm-pkg (50 passed), vllm-models (110 passed), run_all (0 failed), public-names clean, clean |

## 7. Docs updated

- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/embeddinggemma-2/family.yaml` — the serve block (`patches`,
  the `--attention-backend TORCH_SDPA` comment, the `--enforce-eager` comment), the client's
  `recipe:` comment, the header comment and the notes ("Engine": the round-10 digest read; "Engine
  flags": rounds 8-10, the chosen path and the class).
- `docs/reference/recipes.md` — the patch list: "Three patches ship" and the `torch-sdpa-attention`
  entry.
- `rcp-ndcg-vllm/README.md` — the embeddinggemma-2 catalog row names both patch carriers.
- `CHANGELOG.md` — the two FLEX_ATTENTION bullets rewritten to the shipped behaviour, plus the
  overview's "the one engine patch" -> "the engine patches".
- Greps run: `git grep -n "FLEX_ATTENTION|register_backend" -- docs README.md REPRODUCIBILITY.md skills
  examples experiments mkdocs.yml rcp-ndcg-vllm/README.md` (no live page named the pin);
  `git grep -n "embeddinggemma" -- docs README.md skills examples experiments mkdocs.yml`;
  `git grep -n "one engine patch|Two patches|three patches" -- CHANGELOG.md docs rcp-ndcg-vllm/README.md
  README.md`.

## CHANGELOG entry

Inside the one `0.0.1` section, `### Fixed` — the two round-8/9 FLEX bullets replaced by one bullet
(verbatim):

- **The embeddinggemma-2 engine runs on the plugin's registered `TORCH_SDPA` backend** (the E2 rounds
  7-10 backend search): the model's heterogeneous head dimensions (sliding 256, full 512) leave no
  image backend -- TRITON_ATTN's prefill kernel needs 266240 B of shared memory against the 232448 B
  limit, FA4's kernel rejects head_dim 512 (`FlashAttention forward only supports head dimension at
  most 256`), and TORCH_SDPA's enum value is the empty string ("this tag is only used for ViT"), so
  the manual-selection path raises `Backend TORCH_SDPA must be registered before use. Use
  register_backend(Backend.TORCH_SDPA, 'your.module.YourClass')`. FLEX_ATTENTION was the one image
  backend that cleared the dimensions, and rounds 9-10 show it dying in its own
  `torch.compile`/CUDAGraph interaction: `torch._dynamo.exc.InternalTorchDynamoError: RuntimeError:
  Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run` at
  `torch/nn/attention/flex_attention.py:2369` -- its compiled `create_block_mask`
  (`mode="reduce-overhead"`) output is overwritten before the compiled `flex_attention` reads it, and
  `--enforce-eager` does not stop it (that compile is the backend's own). The shipped path is the
  mechanism the engine's own error names: the plugin's new `torch-sdpa-attention` patch registers
  `TorchSdpaAttentionBackend` -- a small encoder-only backend over
  `torch.nn.functional.scaled_dot_product_attention` (non-causal, no KV cache, no head-dimension cap,
  the sliding layers' window as a mask, one SDPA call per packed request) -- under
  `Backend.TORCH_SDPA`, and the recipe pins `--attention-backend TORCH_SDPA` with `--enforce-eager`
  beside it. The recipe notes record the whole rejection chain and each rejected backend's error; a
  CPU test pins the argv, the patch opt-in and the notes evidence, the plugin suite pins the
  registration and the forward against a manual reference, and the lane's image read runs the class
  against the pinned nightly's own vLLM. The GPU wave's T0 model load is the confirmation. (The
  plugin's shared engine modules moved every plugin recipe's behaviour fingerprint, so the wave
  re-records those recipes' corpora.)

## Public surface changes

None. No `__all__`, CLI command/flag, exit code, contract snapshot or `schemas/` file changed
(`tests/contract` and `tests/docs` pass unchanged; the recipe JSON schema's `patches` field is a free
string array and its description is static). The new patch name is recipe data validated against
`PATCH_NAMES`; the changed files are the plugin package's internal modules, the recipe package data,
the unpublished test package's tests and goldens, docs and the CHANGELOG.

## Files outside scope

None. The touched files are the plugin's backend/patch modules and their wiring, its CPU test module,
the embeddinggemma-2 recipe and its family test, the 14 plugin goldens the shared-module fingerprint
move regenerated, `docs/reference/recipes.md`, `rcp-ndcg-vllm/README.md` and `CHANGELOG.md`.

## For the next lanes

- **The GPU wave is the confirmation.** The next RC's T0 must load `embeddinggemma-2` with
  `--attention-backend TORCH_SDPA` + `--enforce-eager`. Every step up to and including engine init is
  CPU-verified against the image's own vLLM; the model load, the served vectors and stage 2 are the
  GPU's.
- **The wave re-records every plugin recipe.** Adding a module to `rcp_ndcg_vllm.models` and a name to
  `rcp_ndcg_vllm.patches` moves those two shared engine modules, which every plugin recipe's
  fingerprint keys: all 14 plugin goldens were regenerated, and the wave's staleness check will
  re-record those recipes' corpora. That is expected, not a defect.
- **The residual risks to watch in the wave.** (1) head_dim 512 runs on SDPA's math kernel: correct,
  but the long rows (the pairs' `length:at_budget` row, 8188 tokens) materialize the score matrix
  per request, so the wave's time per long row is the thing to look at; (2) the sliding band is
  `|i - j| <= 512` (vLLM W = 513 = the checkpoint's 512 + 1) — if the wave's stage 2 shows a small
  cosine shift on long rows, the window boundary is the first suspect; (3) the backend works under
  the V2 encoder-only model state (the metadata builder interface); under the V1 runner this model
  cannot serve at all (its one-encoder-only-spec assertion predates any backend), which is not this
  lane's concern.
- **The fold's status.** The round-10 read shows the pinned digest carries the classes and the
  architecture, so both plugin registrations are inert there; they stay as the carrier for an engine
  image that lacks them. When the recipe moves to a release image, delete the fold, the
  `torch-sdpa-attention` patch and the `--attention-backend` pin together.
