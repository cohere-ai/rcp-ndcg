# Report lane `recipe-fix`: the recipe, reference and plugin fixes from GPU-E1 (per family, after decision 34)

## 1. Status

**DONE.** Every brief item (1-28 and the independent review's additions) is fixed, declared or explicitly
open with evidence; two adversarial verifier rounds plus a final confirmation round ran. The lane then
merged the current `rfc-0001` twice more (tip `f0108f59`: ref-envs, harness-media, judge-fixes, run-integrity,
mrl-harness and runner-backends; then tip `7f3b94c1`: round 16 plus a report scrub and two CI-only test
fixes) and ported its declarations onto the merged tree; the port's own drift (topk's reference lock, the
qwen3-reranker reference-env test expectation) is fixed in `9f9dc962`. `bin/gate lane/recipe-fix` is
**GATE: PASS** on the gated tree `9f9dc962` (the report commits on top change only this file). GPU
confirmation (E2) is the operator's; every GPU-dependent number below is declared as E2's to measure.

## 2. Commits

| Commit | Subject |
|---|---|
| `a1be9a49` | The tokenizer sidecars, the blank-document policy and the pinned video budget |
| `13f59d8a` | The recipe schema declares its engine patches, the reference attention and the catalog pin |
| `fc23fd69` | The recipes: declared attention and head dtype, patch opt-ins, the video pin, the measured notes |
| `2ba2d120` | The regenerated artifacts: pairs, the five metadata-only corpus re-keys, the stale list |
| `83e7f3b7` | Docs and CHANGELOG for the recipe-fix lane |
| `bc7e0e73` | Verifier round 2 minors: the format fix, the pplx mutant docstring, the qwen3-vl-reranker recipe-based load pinned |
| `9e28c64c`, `93b2c58c`, `34d3be30` | Merge `rfc-0001` (`b18d34c4`: fp-v4, MRL recipes, judges, records, retrieval-fixes, runner-security) and the lane's port |
| `3e7a12be`, `9c2a4d73` | Merge `rfc-0001` (`6c388950`: the late-interaction keep-rules) and the lane's port |
| `0de5408b` | Merge `rfc-0001` (`f0108f59`: ref-envs, harness-media, judge-fixes, run-integrity, mrl-harness, runner-backends) + the lane's port |
| `0498055a` | Merge `rfc-0001` (`7f3b94c1`: round 16, the report scrub, the two CI-only test fixes) |
| `9f9dc962` | The topk reference lock follows its merged `reference.in`; the qwen3-reranker reference-env test expects the image's torch |
| (this report) | The lane report |

## 3. What changed (per brief item)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Attention in the reranker references | **fixed** | `reference.attn_implementation: sdpa` in the ctxl and qwen3-reranker families; both references read it from `--recipe` (`ctxl/reference.py:328`, `qwen3-reranker/reference.py:208`) and never `torch.cuda.is_available()`; the flash-attn pin is dropped from both `requirements-reference.txt`; CPU stub tests capture the `from_pretrained` kwargs (`test_ctxl_...py::test_the_reference_loads_with_the_declared_attention_implementation`, `test_qwen3_reranker.py::test_score_mode_setup_parses_and_reaches_the_model_load`). |
| 2 | Reranker score precision | **fixed / declared** | `serve.hf_overrides.head_dtype: model` on ctxl 1b/2b/6b, qwen3-reranker 0.6b/4b/8b and qwen3-vl-reranker 2b/8b; vLLM's pooling head defaults to fp32 (`config/model.py:2447-2468` at the tag). The notes carry E1's numbers (qwen3-reranker p99 97.7/97.7/90.9 %, ctxl 2b 0.0872, 1b 0.249); E2 re-measures. |
| 3 | The pooling hang | **fixed** | fp-v4's `serve.patches` semantics (the merge): a tuple validated against `PATCH_NAMES`, the plugin whose entry point applies the patches required, `RCP_NDCG_VLLM_PATCHES` rendered by the console, the wave runner and the e2e driver. Opt-ins: jina-embeddings-v5-text-small, zembed-1-embedding, harrier-oss-v1 x3, pplx-embed-v1 x2, pplx-embed-v2-context-9b-preview (its warmup); the patch-only carriers declare `serve.plugin: rcp-ndcg-vllm`. No `--max-num-batched-tokens`. |
| 4 | zembed vectors | **refuted / declared** | The served pooling path is correct (the root-cause spec: card path vs recorded engine vector 0.999884-0.999890; ST 6.1.0's dropped suffix reproduces E1's 0.08-0.4); the notes now record that and the pinned `sentence-transformers>=5.3,<5.4` requirements; the reference's suffix-literal guard stays; the recipe opts into the hang patch. E2 re-measures under the pinned env. |
| 5 | octen-embedding-8b | **refuted / declared** | Ids, pooled token and anchor 151643 are identical on both sides (root-cause spec); the notes record min cosine 0.99362/0.99372 and the refutation, and `gates.vec_min_cosine: 0.993` carries the measured bound (provisional for 0.6b/4b, E2 measures per size). |
| 6 | jina-reranker-v3 empty document | **fixed** | `empty_doc: omit_zero_blank` (the paper's `text.strip()` rule) and `empty_query: send`; the false note is corrected; the listwise-depth arithmetic (131072 window, 147 frame, 512 query twice, ~2063/doc, 150 docs cannot fit) is in the notes; the paper-config source line states it carries no `recipe:` pointer and differs on budgets. |
| 7 | ctxl-1b tokenizer | **fixed** (product) | `load_tokenizer` applies `tokenizer_config.json`, `added_tokens.json`, `special_tokens_map.json`; `TextTokenizer.sha256` extends only when the applied tokens change the effective vocabulary (the operator decision). Offline reproduction: ctxl-1b `ids("2 + 2") == [17,220,10,220,17]`, `ids("H + ion") == [39,220,10,27672]`; both identity cases tested. |
| 8 | qwen3-vl-reranker reference | **fixed** | `.to()` moves tensors only; `main()` loads `model@revision` from `--recipe` (stub test); `head_dtype: model`; the notes record the E1 crash and the p99. |
| 9 | qwen3-vl-embedding video | **fixed / declared** | `client.video_policy {fps: 2, wire: video_url, engine_video_pinning: true}` with `--media-io-kwargs '{"video": {"fps": 2}}'` (the Qwen3-VL backend ignores `num_frames`), `serve.mm_processor_kwargs.videos_kwargs {min_pixels: 4096, max_pixels: 7864320}` (the card's `total_pixels`) mirrored by `engine_video_min/max_pixels`, the loader cross-check and its test; the client counts the engine's fps rule and the pinned clip budget. E2 validates the engine count. |
| 10 | topk reference transformers | **fixed** | `_alias_qwen3_5_layer_type` aliases `layer_type` to `block_type` when only the new name exists; the requirements pin `transformers>=5.10.4,<5.18.0`; a CPU shim test. |
| 11 | pplx-embed-v2-context | **fixed / declared** | The plugin warmup rule is rf-engine's (`[0, 1]`); the recipe declares `max_model_len: 131072`, client 131070, `known_deviations: [over_cap_cut_differs]`, the 2^31 arithmetic (q 2^31 bytes; merged gate_up 87381 elements / 43690 bytes) and the patch opt-in; E2's real-request run decides any further step down. |
| 12 | pplx-embed-v2-late | **fixed** | `_fp16_lists` moves a CUDA tensor to CPU before numpy; `media_head_as_system: true`; the reference's media count is `patches + 2`; the notes record the E1 FLA/numpy failures and `reference.device: cuda`. |
| 13 | qwen3-vl network tests | **fixed** | `RCP_NDCG_NETWORK_TESTS=1`: `test_qwen3_vl_embedding.py` 24 passed, `test_qwen3_vl_reranker.py` 14 passed. |
| 14 | Notes honesty | **fixed** | zembed's "~1e-3", octen's equivalence, jina-reranker-v3's paper-rule claim, topk's no-verify wording, ctxl-2b's "bounded by the logit gate", qwen3-reranker's `min_version`, qwen3-embedding's `--pooler-config`, the stale `plugins/pplx/` and `requirements-reference.txt` and `rcp_ndcg_vllm.equivalence` paths and the `stale.json` action text are all corrected or recorded with the E1 evidence. |
| 15 | Public names in package data | **fixed** | qwen3-reranker/zerank/harrier/pplx notes and sources rewritten as public statements; ONE repo-wide guard (`test_recipe_hygiene.py`) scans every shipped retrieval recipe file and template (local paths, `research/`, `sweep`, `REVIEW-LOG`, `workstream`, `recipe-fix`, `handover/`). The judge families (merged later) cite the handover judge catalog; the guard scopes to the non-judge families and `handover/RELEASE-CHECKLIST.md` carries the cleanup. |
| 16 | pplx-late media head | **fixed** | `media_head_as_system: true` (the card's system-message path) and the reference's count reconciled to `patches + 2`; the exact keep-rule landed with the late-keep merge (`media_keep_token_ids` / engine-side pooler). |
| 17 | Video policy | **fixed** | fps pinned (the engine's own rule) through `--media-io-kwargs`; the inert `num_frames` pin dropped; the per-clip pixel budget pinned through `videos_kwargs`; the client counts the engine's formula (`qwen3_vl_video_frame_indices`). |
| 18 | Patch opt-in rule | **fixed** | The exact set per the rule (client budget can reach `max_model_len` AND `max_model_len` exceeds the engine's batch budget): jina small, zembed, harrier x3, pplx-v1 x2, pplx-context; nano and the rerankers do not. |
| 19 | Family field drift | **fixed** | ctxl: one shape (the 1b `batch_size` override dropped), `empty_query: send`; zerank: one shape, `empty_query: send`; jina-reranker-v3 `empty_query: send`; qwen3-vl-reranker keeps `refuse` and decision 25's wording now states that exception (`handover/00-MASTER.md:144`). |
| 20 | qwen3-reranker paper score scale | **fixed** | One line in the notes: the paper stored the raw yes-logit (`use_activation: false`), the recipe serves the card's probability scale; rankings unaffected. |
| 21 | Plugin census | **already fixed at base** | `test_late_model.py` reads `visual.patch_embed.proj.weight: (768, 3, 2, 16, 16)` and `late.py` cites `default_loader.py:221`. |
| 22 | Catalog pin | **fixed** | `rcp-ndcg-vllm/tests/test_catalog.py` pins every README retrieval row (id, model, role, input, MRL cell, plugin, status) to `iter_recipes()`; the judge rows are the judge lane's catalog docs. |
| 23 | `reference.device` | **fixed** | `cuda` only where a CPU reference is impossible (topk x2, pplx-late x2, pplx-context) or materially moves the gate (qwen3-embedding x3, qwen3-vl-embedding x2); every other reference stays CPU. |
| 24 | qwen3-vl-reranker `--recipe` | **fixed** | `main()` reads model/revision from the resolved recipe; the stub test pins it. |
| 25 | harrier opt-in | **fixed** | `serve.patches: [pooling-full-context]` + `serve.plugin: rcp-ndcg-vllm`. |
| 26 | Pairs manifest | **fixed** | The pairs and manifest are regenerated the generator's own way; the late-keep merge's affected rows (ctxl x3, jina-reranker-v3, pplx-context) regenerated and merged; the stale refusals are gone. |
| 27 | pplx-late `[D]` prefix | **fixed** | The client sends the media side's head as a system message; the reference counts `patches + 2`. |
| 28 | MRL | **open (undefined)** | The brief's "MRL" item says "see the operator's MRL section when it is added"; no section was added. The merged `rfc-0001` (mrl-recipes) declares every variant's MRL kind/set; this lane changed none of those declarations. |

**The catch-up merges and their port.** `f0108f59` brought ref-envs' per-family `reference.in`/`reference.lock`
and the reference store, harness-media, judge-fixes, run-integrity, mrl-harness and runner-backends; `7f3b94c1`
brought round 16 and the two CI-only test fixes. The port kept every lane's declarations in each family file
(this lane's attention, head-dtype, patch and video fields beside late-keep's keep-rules, ref-envs'
environments and mrl-recipes' MRL fields), moved the lane's changes into rfam's family test modules (the
per-recipe modules rfam deleted stay deleted), regenerated the generated files the documented way (the
recipe/family schemas, the contract snapshots, the goldens, the pairs manifest) and regenerated the family
locks whose merged `reference.in` this lane changed (`ctxl-rerank-v2-instruct-multilingual`, `qwen3-reranker`).
The merged tree's tests then surfaced two port drift items, fixed in `9f9dc962`: topk-embed-v1's lock still
pinned the checkpoint's `transformers==5.9.0` against the merged `reference.in`'s `>=5.10.4,<5.18.0`
(regenerated: `transformers==5.17.0`, the image's stack version, with the header hash following the merged
file; the lock's SHA-256 is the environment identity stored reference outputs key on, so topk's stored
outputs re-record under the regenerated lock), and `test_reference_env.py` still expected qwen3-reranker to
declare its own torch (the sdpa reference runs on the image's torch, so `own_torch` is false; the expectation
now says so).

The root-cause spec's items are covered above or by the merged lanes: the zembed reference guard (existing
suffix check + pin), the ctxl sidecar product fix (item 7), jina's blank policy (item 6), the head dtype
(item 2), the pplx warmup (rf-engine) and length (item 11), jina's listwise budget (item 6), the pplx-v1
patch (item 18), the media-rules questions 2/3 (items 16/17), the `instruction:` declarations (all
embed/multi-vector families declare `none`; the completeness test is in `rcp-ndcg-test/tests/test_recipe.py`),
and the judge recipes' paper sampling (lane l08-judges, merged in `rfc-0001`).

## 4. Verification

**Round 1** — two fresh verifiers in parallel (DeepSeek-V4.1-flash `:xhigh`; lens A correctness, lens B
regressions/hygiene), on `83e7f3b7`.

- **Lens A: findings, all fixed.** MAJOR: `serve.patches` was exported only by the console; the wave runner
  and the e2e driver did not set `RCP_NDCG_VLLM_PATCHES` (fixed with one home `patches_environment`, used by
  all three; a wave-side test). MINORs: the patches package docstrings and the harrier/pplx-v1 notes said the
  recipe opt-in was not shipped; the ctxl/qwen3-reranker stale flash-attn/float32 wording; the pplx-context
  mutant docstring; the qwen3-vl-embedding 64-frame wording; the golden's tokenizer digest fallback
  (fixed: the golden resolver now prefers the product's sidecar-aware digest); no failing test for the
  attention read or the instructed-dataset completeness (both added); the octen family-wide bound is
  provisional for 0.6b/4b (recorded in the notes); the family `INTERNAL_LABELS` guards remain beside the
  repo-wide one (documented as legacy); decision 25's wording (fixed).
- **Lens B: findings, all fixed.** MAJOR: the pplx-v1/harrier notes contradicted the declared patches and
  carried "workstream"/"recipe-fix" (fixed; the guard now catches both). MINORs: the patches docstrings; the
  CHANGELOG's extra `### Fixed` section (merged); the new staleness assertion was tautological (rewritten to
  compute the replay set independently of `stale.json`); the offline Hub sidecar behaviour is now in the
  CHANGELOG; the qwen3-vl-embedding stale 64-frame wording; a dropped test docstring (restored).

**Round 2** — one fresh confirmation verifier (lens A+B), on `da493e7a`.

- **VERDICT: FAIL**, with: BLOCKER `ruff format` on one edited test file (fixed); MAJOR the declared patches
  were inert for plugin-null recipes unless the wave installed the `rcp-ndcg-vllm` wheel (resolved by the
  `rfc-0001` catch-up: fp-v4's semantics require the plugin, and the patch-only carriers declare it, so the
  bootstrap installs the wheel); MAJOR the lane's `patches` implementation duplicated and contradicted the
  merged fp-v4 one (resolved by the merge: fp-v4's semantics and `rcp-fp/4` stay, the lane's
  `patches_environment`/`RUNTIME_SERVE_FIELDS` are dropped); MINORs the pplx-context mutant docstring, the
  qwen3-vl-reranker wording and test, and the CHANGELOG pplx-context rationale (all fixed).

**Round 3** — one fresh confirmation verifier (lens A+B) on the first merged tree (`34d3be30`).

- **VERDICT: FAIL**, with: MAJOR the tree was not caught up to the current `rfc-0001` (`6c388950`, the
  late-keep lane; the brief's late-interaction keep-rules) — fixed by merging it and porting the lane's
  declarations onto the keep-rule recipes; MINORs the CHANGELOG's pre-merge patch wording (fixed), the
  `plugin_architectures` "required" wording (fixed in the CHANGELOG/docs/README), no test for the video
  pixel cross-check (added), the judge hygiene carve-out (scoped and recorded in the release checklist), and
  the catalog cell overstating jina-nano (fixed). The verifier also confirmed the merged reconciliation:
  fp-v4's patch semantics with the patch modules keyed, `attn_implementation` and the video cross-check
  re-applied, the four metadata-only corpus re-keys under `rcp-fp/4`, the eight stale declarations exact,
  and the suites green.

## 5. Checks

Final gate on the merged tree `9f9dc962` (`bin/gate lane/recipe-fix`, slot 2):

```
ruff-check exit=0 / ruff-format exit=0 (612 files) / basedpyright exit=0
pytest exit=0 -> 3985 passed, 103 skipped
contract-docs exit=0 -> 302 passed, 55 skipped
mkdocs exit=0
test-pkg exit=0 -> 1086 passed, 227 skipped
recipes exit=0 (network) -> no failure outside the baseline
vllm-pkg exit=0 -> 50 passed / vllm-models exit=0 -> 92 passed, 7 skipped
run_all exit=0 -> 1022 checks, 987 match, 35 known deviations, 0 failed; 67/67; 82/82
public-names exit=0 (clean) / clean exit=0
GATE: PASS
```

The lane's own runs on the same tree: `pytest tests -n 8` 3985 passed/103 skipped; `pytest
rcp-ndcg-test/tests -n 4` 1086 passed/227 skipped; `pytest rcp-ndcg-vllm/tests` 133 passed/11 skipped; ruff
and basedpyright clean.

Earlier gate on the late-keep-merged tree `9c2a4d73` (`bin/gate lane/recipe-fix`):

```
ruff-check exit=0 / ruff-format exit=0 (604 files) / basedpyright exit=0
pytest exit=0 -> 3908 passed, 103 skipped
contract-docs exit=0 -> 301 passed, 55 skipped
mkdocs exit=0
test-pkg exit=0 -> 1044 passed, 227 skipped
recipes exit=0 (network) -> no failure outside the baseline
vllm-pkg exit=0 -> 50 passed / vllm-models exit=0 -> 92 passed, 7 skipped
run_all exit=0 -> 1022 checks, 987 match, 35 known deviations, 0 failed; 67/67; 82/82
public-names exit=0 (clean) / clean exit=0
GATE: PASS
```

Earlier checks: the lane's base gate on `83e7f3b7` also passed; the network recipe files were run one at a
time (`RCP_NDCG_NETWORK_TESTS=1`, the tokenizer cache) — all passed.

## 6. Open questions

- **E2's GPU confirmations** (the operator's): the eight rerankers under `head_dtype: model` (the p99 and
  max bounds), the sdpa references, octen's like-for-like diagnostic, the zembed reference under the pinned
  ST, the video engine count under the fps/`videos_kwargs` pin, pplx-context's real request at 131072, the
  `pooling-full-context` patch on a real at-budget prompt, and every recipe's stage 1/2.
- **Corpora**: eight recipes are declared stale with their exact moved inputs (jina-embeddings, jina-reranker,
  qwen3-reranker 0.6b/4b/8b, qwen3-vl-embedding, qwen3-vl-reranker, zembed) and four were re-keyed
  metadata-only under `rcp-fp/4` (octen-8b, qwen3-embedding-0.6b, zerank-1-small, zerank-2); the wave
  re-records them. The zerank re-key's strip claim was checked: the recorded queries carry no trailing space.
- **The judge families' handover citations** are the judge/docs lanes' cleanup before `handover/` is deleted
  (a release-checklist line records it); the repo-wide hygiene guard scopes to the non-judge families until
  then.
- **MRL (item 28)** has no operator section; the merged mrl-recipes lane owns the declarations. If the
  operator wants a per-variant `mrl_dim` selection or a stricter card-set validation, that is a new item.
- **The patch-only plugin relaxation**: `ServeConfig` admits a plugin with an empty `plugin_architectures`
  when the plugin is the patch carrier (the patch modules are still keyed). If the owner prefers a separate
  carrier field, that is a schema change.
- **The pooling-full-context patch retires** when `engine.image` moves to the first vLLM release that
  carries `e6fc81bc78`; the module's inert log line names the moment.
- **pplx-context's patch opt-in** is conservative (the engine's warmup is the `max_model_len`-token input;
  the client cap is 131070 plus the 2-token prefix); harmless, and E2 confirms.

## CHANGELOG entry

The lane's entries are under `## Unreleased` (`### Public surface`, `### Fixed`): the tokenizer sidecars and
the identity rule, `empty_doc: omit_zero_blank`, `VideoPolicy.engine_video_min/max_pixels`, the reference
`attn_implementation`, the recipe fixes (attention/head dtype, the patch opt-ins, the video pin, the
reference devices, the blank-document and instruction policies, the notes honesty) and the catalog pin. The
patch/module-hashing surface entry is the merged fp-v4 lane's, extended with the patch-only carrier wording.

## Public surface changes

- `rcp_ndcg.data.tokenizer.SIDECAR_FILES`; `TextTokenizer.from_json(sidecars=)`; the identity rule.
- `EmbeddingEndpoint.empty_doc`/`RerankEndpoint.empty_doc` gain `omit_zero_blank`.
- `VideoPolicy.engine_video_min_pixels`/`engine_video_max_pixels`; `ReferenceSpec.attn_implementation`
  (recipe schema, `rcp-ndcg-vllm`).
- Regenerated: `schemas/{index,run-config}.v1.json`, `tests/contract/snapshots/`, the recipe/family schemas,
  the recipe goldens, the pairs and their manifest, the corpora re-keys/verification records, the e2e goldens
  and the phased-script golden.

## Files outside scope

- The judge families (l08-judges, merged in `rfc-0001`) and the MRL declarations (mrl-recipes, merged): read,
  not changed, except the release-checklist line.
- `handover/00-MASTER.md` (decision 25's exception wording) and `handover/RELEASE-CHECKLIST.md` (the judge
  citation cleanup): handover scaffolding, deleted before the release.

## For the next lanes

- **E2** re-records the eight stale corpora and confirms every declared bound; the recipe notes name each.
- **ref-envs** owns the per-family reference environments the zembed root cause needs (the recipe pins
  `sentence-transformers>=5.3,<5.4`).
- **06/07** own the judge families' handover citations and the final catalog counts.
- **fp-v4** already merged; the lane's recipe declarations are on `rcp-fp/4`.
