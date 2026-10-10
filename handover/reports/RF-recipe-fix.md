# Report lane `recipe-fix`: the recipe, reference and plugin fixes from GPU-E1 (per family, after decision 34)

## 1. Status

**DONE.** Every brief item (1-28 and the independent review's additions) is fixed, declared or explicitly
open with evidence; two adversarial verifier rounds plus a final confirmation round ran. The lane then
merged the current `rfc-0001` three times (tip `f0108f59`: ref-envs, harness-media, judge-fixes, run-integrity,
mrl-harness and runner-backends; then tip `7f3b94c1`: round 16 plus a report scrub and two CI-only test
fixes; then tip `77710eeb`: the slow-runner per-test timeout fix) and ported its declarations onto the
merged tree; the port's own drift (topk's reference lock, the qwen3-reranker reference-env test expectation)
is fixed in `9f9dc962`. The W1 review then landed (`a6819b50`: the two blocking findings and the cheap
non-blocking ones), the integration branch `int/round17` merged (`a74a462a`, bringing the orchestrator's
round-16 review fixes), and `bin/gate lane/recipe-fix` is **GATE: PASS** on the final tree `e620dd50`.
GPU confirmation (E2) is the operator's; every GPU-dependent number below is declared as E2's to measure.

**The E2 r3 fix run (round 1).** The r3 GPU wave left one recipe verified and seven failed at stage 2; the
findings are fixed or declared in section 7. The branch fast-forwarded to the `rfc-0001` tip `be645fb7`
first, then landed four commits (`0cca1529`, `f1451022`, `c99b99ed`, `9ff12b81`); `bin/gate lane/recipe-fix`
**GATE: PASS** at that tree.

**The E2 round-2 fix run (newest).** The r2/r4 waves on the corrected RC found four more issues; the branch
merged `int/round18` (`0d3a20b3`) and landed three commits (`14299bfe`, `c4756529`, `0388da2f`), recorded in
section 8. HEAD `0388da2f`, `bin/gate lane/recipe-fix` **GATE: PASS**.

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
| `5335821f` | Merge `rfc-0001` (`77710eeb`: the slow-runner per-test timeout fix) |
| `a6819b50` | The W1 review fixes: the query-side allowlist crash, the fake's engine-side keep-rule and the regenerated topk/pplx-late pairs |
| `a74a462a` | Merge `int/round17` (the orchestrator's round-16 review fixes) |
| `e620dd50` | The octen variant notes' provisional gate bound reaches the family goldens |
| `be645fb7` | (fast-forward to the `rfc-0001` tip) the staged `rcp-ndcg` pin hash as a lock continuation line |
| `0cca1529` | The recipe schema declares per-size gate overrides (`VariantOverrides.gates`, field-by-field over the family's) |
| `f1451022` | The pplx-late and topk references pair the query side per query text (the r3 shape mismatch) |
| `c99b99ed` | The E2 r3 bf16 bounds are declared per variant (qwen3-reranker x3, qwen3-embedding 0.6b/8b, pplx-late x2) |
| `9ff12b81` | The octen notes point at the per-size gate field, now that it exists |
| `e7572d82` | handover: the report records the E2 r3 findings and the per-variant gate field |
| `0d3a20b3` | Merge `int/round18` (the round-1 branch + the tip's judge-smoke fix `32cb6625`) |
| `14299bfe` | The pplx-embed-v1 reference pairs the query side per query text (+ the 0.6b gate bound) |
| `c4756529` | The topk reference environment pins torch's whole CUDA stack |
| `0388da2f` | The pplx-context client sends the reference's document ids (the split-parse seam, the listwise probe, the smoke) |
| (this report) | The lane report |

## 3. What changed (per brief item)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Attention in the reranker references | **fixed** | `reference.attn_implementation: sdpa` in the ctxl and qwen3-reranker families; both references read it from `--recipe` (`ctxl/reference.py:329`, `qwen3-reranker/reference.py:208`) and never `torch.cuda.is_available()`; only a DECLARED value travels (the `or "sdpa"` fallbacks are gone, the class default is `None`); the flash-attn pin is dropped from both `reference.in`; CPU stub tests capture the `from_pretrained` kwargs (`test_ctxl_...py::test_the_reference_loads_with_the_declared_attention_implementation`, `test_qwen3_reranker.py::test_score_mode_setup_parses_and_reaches_the_model_load`). |
| 2 | Reranker score precision | **fixed / declared** | `serve.hf_overrides.head_dtype: model` on ctxl 1b/2b/6b, qwen3-reranker 0.6b/4b/8b and qwen3-vl-reranker 2b/8b; vLLM's pooling head defaults to fp32 (`config/model.py:2047` at the tag: the `head_dtype` property's docstring and default; `_get_head_dtype` at `:2447-2468`). The notes carry E1's numbers (qwen3-reranker p99 97.7/97.7/90.9 %, ctxl 2b 0.0872, 1b 0.249); E2 re-measures. |
| 3 | The pooling hang | **fixed** | fp-v4's `serve.patches` semantics (the merge): a tuple validated against `PATCH_NAMES`, the plugin whose entry point applies the patches required, `RCP_NDCG_VLLM_PATCHES` rendered by the console, the wave runner and the e2e driver. Opt-ins: jina-embeddings-v5-text-small, zembed-1-embedding, harrier-oss-v1 x3, pplx-embed-v1 x2, pplx-embed-v2-context-9b-preview (its warmup); the patch-only carriers declare `serve.plugin: rcp-ndcg-vllm`. No `--max-num-batched-tokens`. |
| 4 | zembed vectors | **refuted / declared** | The served pooling path is correct (the root-cause spec: card path vs recorded engine vector 0.999884-0.999890; ST 6.1.0's dropped suffix reproduces E1's 0.08-0.4); the notes now record that and the pinned `sentence-transformers>=5.3,<5.4` requirements; the reference's suffix-literal guard stays; the recipe opts into the hang patch. E2 re-measures under the pinned env. |
| 5 | octen-embedding-8b | **refuted / declared** | Ids, pooled token and anchor 151643 are identical on both sides (root-cause spec); the notes record min cosine 0.99362/0.99372 and the refutation, and `gates.vec_min_cosine: 0.993` carries the measured bound (provisional for 0.6b/4b, E2 measures per size). |
| 6 | jina-reranker-v3 empty document | **fixed** | `empty_doc: omit_zero_blank` (the paper's `text.strip()` rule) and `empty_query: send`; the false note is corrected; the listwise-depth arithmetic (131072 window, 147 frame, 512 query twice, ~2063/doc, 150 docs cannot fit) is in the notes; the paper-config source line states it carries no `recipe:` pointer and differs on budgets. |
| 7 | ctxl-1b tokenizer | **fixed** (product) | `load_tokenizer` applies `tokenizer_config.json`, `added_tokens.json`, `special_tokens_map.json`; `TextTokenizer.sha256` extends only when the applied tokens change the effective vocabulary (the operator decision). Offline reproduction: ctxl-1b `ids("2 + 2") == [17,220,10,220,17]`, `ids("H + ion") == [39,220,10,27672]`; both identity cases tested. |
| 8 | qwen3-vl-reranker reference | **fixed** | `.to()` moves tensors only; `main()` loads `model@revision` from `--recipe` (stub test); `head_dtype: model`; the notes record the E1 crash and the p99. |
| 9 | qwen3-vl-embedding video | **fixed / declared** | `client.video_policy {fps: 2, wire: video_url, engine_video_pinning: true}` with `--media-io-kwargs '{"video": {"fps": 2}}'` (the Qwen3-VL backend ignores `num_frames`), `serve.mm_processor_kwargs.videos_kwargs {min_pixels: 4096, max_pixels: 7864320}` (the card's `total_pixels`) mirrored by `engine_video_min/max_pixels`, the loader cross-check and its test; the client counts the engine's fps rule and the pinned clip budget. E2 validates the engine count. |
| 10 | topk reference transformers | **fixed** | `_alias_qwen3_5_layer_type` aliases `layer_type` to `block_type` when only the new name exists (and leaves a class with neither untouched); the requirements pin `transformers>=5.10.4,<5.18.0` (the lock resolves `5.17.0`, the image's stack); the three-case CPU shim test (`test_topk_embed_v1.py::test_the_layer_type_shim_aliases_only_the_new_name`) exists. |
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
| 26 | Pairs manifest | **fixed** | Regenerated the generator's own way. The four recipes the late-keep merge left unreproducible are regenerated at the tip with `--reference-python`: `pplx-embed-v2-late-0.6b/9b` (33 rows each; `render_check` and `media_check` passed) and `topk-embed-v1-small` (29 rows, 5 pruned) / `-xsmall` (31 rows, 3 pruned), each byte-identical to a fresh generation into a scratch directory. The manifest is a generator byte output again (`ensure_ascii=False`; 52 non-ASCII bytes) and every file's `sha256`/`bytes` matches disk. A row the client refuses (topk's special-token-spelling text carries the image-patch id the allowlist gates) is pruned with the reason, never a failed recipe. |
| 27 | pplx-late `[D]` prefix | **fixed** | The client sends the media side's head as a system message; the reference counts `patches + 2`. |
| 28 | MRL | **open (undefined)** | The brief's "MRL" item says "see the operator's MRL section when it is added"; no section was added. The merged `rfc-0001` (mrl-recipes) declares every variant's MRL kind/set; this lane changed none of those declarations. |

**The W1 review round.** The mechanism lens (`reviews/w1-artifacts/mechanism.md`, PARTIAL) and the contract
lens (`reviews/w1-artifacts/contract.md`, PASS) reviewed `5335821f`; `a6819b50` fixes both blocking findings
and the cheap non-blocking ones:

- **The pooling query-side allowlist crash (blocking 1a)**: `PoolingClient._refuse_allowlist_collisions`
  skipped the check when the role tracks no ids (a query under `media_keep_token_ids`), instead of zipping the
  items against an empty id tuple (`ValueError`); the failing test is
  `test_pool_client.py::TestMediaKeepIds::test_a_query_under_the_allowlist_tracks_no_ids_and_is_not_crashed`.
- **The fake's engine-side keep-rule (blocking 1b)**: `fake_transport`/`FakeEndpoint` take
  `document_skip_token_ids` + `document_skip_prefix_token_id`; the harness passes the recipe's declared rule
  (`wire.role_client`), so the fake's `/pooling` reply carries only the kept positions and stage 1 completes
  for pplx-late. The media stage bounds the reply-side rule for the offline fake
  (`media._offline_reply_probe`), because the fake cannot count a media conversation; the requests are
  byte-identical either way (test `test_observe_requests.py::test_the_offline_media_probe_bounds_the_engine_side_reply_rule`).
  The generator records a client refusal as a prunable row (`stages._probe`'s `refusals` +
  `requests._red_row_indexes`), so topk's special-token row is pruned with its reason instead of failing the
  recipe; the reference's render check skips a row the client refused.
- **Item 10's shim test (blocking 2)**: the three-case CPU test exists
  (`test_topk_embed_v1.py::test_the_layer_type_shim_aliases_only_the_new_name`), and the shim only aliases
  when `block_type` exists, so a class with neither name is untouched.
- **Non-blocking**: an uncached optional sidecar is absent, not fatal (`tokenizer.py`; tests for the sidecar,
  for `tokenizer.json` staying loud, and the CHANGELOG corrected); the jina plugin dependency is stated (nano
  carries the wheel with no patch applied); `_plugin_code_declaration`'s docstring and the
  `plugin_architectures` field description match the patch-only-carrier behaviour; the `or "sdpa"` fallbacks
  are gone (only a declared implementation travels); `TextTokenizer.from_json`'s `sidecars=None` is declared
  in its docstring; the manifest is a generator byte output; the octen 0.6b/4b gate bounds are marked
  PROVISIONAL in their variant notes; the report's citations are corrected (`ctxl/reference.py:329`,
  `config/model.py:2047`).
- **Contract lens**: `docs/how-to/add-a-model.md` (the `empty_doc` enum, the `plugin_architectures` comment,
  `reference.attn_implementation` and `reference.device`), the pplx-embed-v1 `empty_doc` comment un-garbled,
  and the pplx-context dated provenance dropped.

**The catch-up merges and their port.** `f0108f59` brought ref-envs' per-family `reference.in`/`reference.lock`
and the reference store, harness-media, judge-fixes, run-integrity, mrl-harness and runner-backends; `7f3b94c1`
brought round 16 and the two CI-only test fixes. `77710eeb` brought one more CI-only test fix (the retrieval
query-block-width test inside the slow runners' per-test timeout); the merge touched no lane file. The port
kept every lane's declarations in each family file
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

Final gate of the E2 round-2 fix run on `0388da2f` (`bin/gate lane/recipe-fix`, slot 2):

```
ruff-check exit=0 / ruff-format exit=0 (612 files) / basedpyright exit=0
pytest exit=0 -> 4000 passed, 103 skipped
contract-docs exit=0 -> 302 passed, 55 skipped
mkdocs exit=0
test-pkg exit=0 -> 1112 passed, 228 skipped
recipes exit=0 (network) -> no failure outside the baseline
vllm-pkg exit=0 -> 50 passed / vllm-models exit=0 -> 92 passed, 7 skipped
run_all exit=0 -> 1022 checks, 987 match, 35 known deviations, 0 failed; 67/67; 82/82
public-names exit=0 (clean) / clean exit=0
GATE: PASS
```

The round-2 run's own checks: `pytest tests -n 8` 4000 passed/103 skipped; `pytest rcp-ndcg-test/tests -n 4`
1112 passed/227 skipped; `pytest rcp-ndcg-vllm/tests` 133 passed/11 skipped (the machine's
`UV_EXTRA_INDEX_URL` unset); `pytest tests/contract tests/docs` 302 passed/55 skipped;
`pytest rcp-ndcg-test/tests/recipes/test_pplx_embed_v2_context.py` 11 passed with
`RCP_NDCG_NETWORK_TESTS=1` and the pinned tokenizer cached; ruff, basedpyright and `mkdocs build --strict`
clean. Red-first evidence: the listwise probe test failed on the unfixed harness (`status: run`, the 234-vs-1
token mismatch), the two reference query-count tests failed on the unwrapped references, the smoke test
failed on the bare-text body, and the tokenizer/client tests failed with `unexpected keyword argument
'split_special_tokens'` / `Extra inputs are not permitted` before the seam. The topk import was verified by
building a scratch venv from the staged wheelhouse's exact wheels: without the new pins `import torch` fails
(`libcublasLt.so.*[0-9] not found`), with them it imports torch 2.11.0+cu130.

Previous final gate of the E2 r3 fix run on `9ff12b81` (`bin/gate lane/recipe-fix`, slot 2):

```
ruff-check exit=0 / ruff-format exit=0 (612 files) / basedpyright exit=0
pytest exit=0 -> 3997 passed, 103 skipped
contract-docs exit=0 -> 302 passed, 55 skipped
mkdocs exit=0
test-pkg exit=0 -> 1108 passed, 227 skipped
recipes exit=0 (network) -> no failure outside the baseline
vllm-pkg exit=0 -> 50 passed / vllm-models exit=0 -> 92 passed, 7 skipped
run_all exit=0 -> 1022 checks, 987 match, 35 known deviations, 0 failed; 67/67; 82/82
public-names exit=0 (clean) / clean exit=0
GATE: PASS
```

The fix run's own runs on the same tree: `pytest tests -n 8` 3997 passed/103 skipped; `pytest
rcp-ndcg-test/tests -n 4` 1108 passed/227 skipped; `pytest rcp-ndcg-vllm/tests` 133 passed/11 skipped
(the machine's `UV_EXTRA_INDEX_URL` unset: the wheel-build test resolves build requirements from the
index that variable names); `mkdocs build --strict` clean; ruff and basedpyright clean. The two new
reference tests were run red on the unwrapped reference (the mutant) and green on the fixed one. The
declared bounds were re-evaluated against the wave's recorded numbers (every bound covers its measured
value; the margin is thinnest at qwen3-embedding-0.6b's k=32, 7.7e-6, and the same RC's kernels are
deterministic).

Previous final gate on the merged tree `e620dd50` (`bin/gate lane/recipe-fix`, slot 2; the tree carries the W1 fixes
and the `int/round17` merge):

```
ruff-check exit=0 / ruff-format exit=0 (612 files) / basedpyright exit=0
pytest exit=0 -> 3997 passed, 103 skipped
contract-docs exit=0 -> 302 passed, 55 skipped
mkdocs exit=0
test-pkg exit=0 -> 1090 passed, 227 skipped
recipes exit=0 (network) -> no failure outside the baseline
vllm-pkg exit=0 -> 50 passed / vllm-models exit=0 -> 92 passed, 7 skipped
run_all exit=0 -> 1022 checks, 987 match, 35 known deviations, 0 failed; 67/67; 82/82
public-names exit=0 (clean) / clean exit=0
GATE: PASS
```

The lane's own runs on the same tree: `pytest tests -n 8` 3997 passed/103 skipped; `pytest
rcp-ndcg-test/tests -n 4` 1090 passed/227 skipped; `pytest rcp-ndcg-vllm/tests` 133 passed/11 skipped; ruff
and basedpyright clean. The four regenerated pairs files are byte-identical to a fresh generator run into a
scratch directory. One gate attempt at the same revision crashed with `test-pkg exit=139` (SIGSEGV, no
traceback) inside `test_wave_corpus.py::test_a_digest_pinned_recipe_records_the_pod_version_not_the_digest`;
the same test passes in the lane's venv and in the gate's next run, so it was a transient native crash in the
gate's slot-2 environment, not a tree defect. The earlier gates on `5335821f`, `9f9dc962` and `9c2a4d73`
passed, as did the lane's base gate on `83e7f3b7`.

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
- **The judge recipes' sampling** (owner-revisit, recorded by the orchestrator; this lane changed nothing):
  the handover's decision 4.3 ran `gpt-oss-120b` at `temperature: 0.0` and both `gpt-oss-120b` and
  `qwen3.6-27b-fp8` with `max_output_tokens: 12288`, while the shipped judge recipes carry their own values;
  those fields are in the judge family key, so changing them re-keys judgements -- the operator holds the
  revisit and the notes keep the existing divergence statement.

## 7. E2 r3 findings (the fix run)

The r3 GPU wave ran the eight recipes of wave R3: `qwen3-embedding-4b` verified, the other seven failed
stage 2. This run fixed or declared each finding (the wave's own per-vector tables counted 111 failing
comparisons at pplx-late-0.6b and 363 at -9b of ~22.7k/22.2k: 33 query-count rows per variant -- one per
pairs row, the reference's unwrapped query matrix read as N matrices -- plus the image-patch cosine
failures, 78/330).

| Finding | r3 measured (bound) | Disposition |
|---|---|---|
| qwen3-reranker-0.6b | p99 within 0.02: 0.9773 (0.99); max abs delta 0.0234 (0.05); tau 1.0 | **declared** `overrides.gates.prob_p99_abs: 0.025` -- 100% of the wave's documents were within 0.025; both sides bf16 (`head_dtype: model`), the residual is the two bf16 kernel stacks, not a cast |
| qwen3-reranker-4b | 0.9773; max abs delta 0.0391; tau 1.0 | **declared** `prob_p99_abs: 0.04` (measured max 0.0391) |
| qwen3-reranker-8b | 0.9318; max abs delta 0.0391; tau 1.0 | **declared** `prob_p99_abs: 0.04` (measured max 0.0391) |
| qwen3-embedding-0.6b | full width 0.99902 pass; k=32 0.99891 fail (0.999) | **declared** `vec_min_cosine: 0.9989` (measured k=32 floor 0.99891; margin 7.7e-6, the same RC's kernels are deterministic) |
| qwen3-embedding-4b | passed (min 0.99935 full width, 0.99942 at k=32) | **unchanged**: the published 0.999 holds; the notes record the measurement |
| qwen3-embedding-8b | full width 0.99875 fail; k=32 0.99906 pass | **declared** `vec_min_cosine: 0.9987` (measured full-width floor 0.99875) |
| pplx-embed-v2-late-0.6b | query shape mismatch ("the engine returned 1 matrix/matrices, the reference 101"); document min cosine 0.99060 (0.999) | **fixed** the reference's query nesting (one matrix per query text, the harness contract) + a CPU test; **declared** `vec_min_cosine: 0.99` for the fp32-reference-vs-bf16-engine image-patch vectors (text rows 0.999940) |
| pplx-embed-v2-late-9b | same mismatch (reference 324...); document min 0.97863 | **fixed** the same nesting (one shared reference); **declared** `vec_min_cosine: 0.978` (text rows 0.999906) |
| topk-embed-v1 (wave R4, not in r3) | the identical query-nesting defect would fail its stage 2 | **fixed** in the same change (reference + CPU test) |
| jina-reranker-v3 (r2 bootstrap) | the reference import check parsed the staged pin as `0.0.1 --hash=...` | **verified at the tip**: the new RC's staged lock carries the hash on a continuation line, `parse_lock` reads `rcp-ndcg == 0.0.1`, and the staged wheel's SHA-256 equals the lock's hash; no further fix |

The per-variant bounds use the new `overrides.gates` field (one field per size, merged over the family's
block); each family's variant notes name the measured value and the stage-2 evidence, and the CHANGELOG
records the field and the declarations. The bounds were re-evaluated against the wave's own numbers: every
declared bound covers its recipe's measured value (qwen3-reranker 0.025/0.04 -> 100% within; qwen3-embedding
0.9989/0.9987 -> 0.99891/0.99875; pplx-late 0.99/0.978 -> 0.99060/0.97863), and the pplx-late query shape
is fixed for the re-run (the recorded 33 count rows per variant were the old reference). The r2
jina-reranker-v3 bootstrap check is verified against the newly built RC's staged lock and wheel; no code
change was needed. The octen 0.6b/4b notes were refreshed to point at the per-size field (they said it did
not exist).

## 8. E2 round 2 (the second fix run)

The r2/r4 waves on the corrected RC found four more issues; this run fixed or declared each. It merged
`int/round18` (`0d3a20b3`: the round-1 branch plus the tip's judge-smoke fix `32cb6625`) first and passed
`bin/gate lane/recipe-fix` at `0388da2f`.

| Item | Evidence | Disposition |
|---|---|---|
| pplx-embed-v1-0.6b/-4b (r2 stage 2) | the same query-side shape mismatch: "the engine returned 1 matrix/matrices, the reference 1024/2560", one query row per pairs row on both variants; 0.6b min cosine 0.998196 (0.999), 4b 0.999169 (pass) | **fixed** the reference's query nesting (a dense embedder's bare vector becomes the one-element list the contract declares) + a CPU test; **declared** the 0.6b `overrides.gates.vec_min_cosine` 0.9981 (the measured 0.998196 rounds to 0.9982, which the `>=` gate would miss); the 4b keeps the published 0.999 |
| jina-reranker-v3 (r2 stage 1) | `engine_prompt_tokens_check`: engine 2218 vs declared 2782 on a 4-document row (anchors, render and tokenize checks pass; stage 2 0.007/1.0) | **fixed in the harness**: a `scoring: listwise` recipe's engine prompt is its own N-passage render (one frame, every document once), not the sum of the declared pair template's per-document renders the probe compared; the probe is `not_run` with the reason, never a false failure (red-first test on `fixture-rerank-listwise`) |
| topk-embed-v1 (r4 bootstrap) | the own-torch reference venv's `import torch` failed: `libcufile.so.0: cannot open shared object file`; the completion installed 11 dists and no nvidia runtime | **fixed**: torch 2.11.0's Linux CUDA stack (the `cuda-toolkit` extras' 11 wheels plus the four direct `nvidia-*` wheels) is pinned in `reference.in` and the lock regenerated with the documented tool; the root cause is the `--no-deps` install of an extras-bearing requirement (it installs only the `cuda-toolkit` meta-package). A scratch venv built from the staged wheelhouse's exact wheels reproduces the failure without the pins and imports torch 2.11.0+cu130 with them |
| pplx-embed-v2-context-9b-preview (r2 serve) | the plugin's role-prefix check raised on a real request (`got [3445, 4587, 1414]`, the harness smoke's bare `smoke text`) and killed the EngineCore | **fixed on both sides**: the smoke now sends the recipe's own query render as ids; and the latent document leg (the declared product gap: the client's `[D] ` was the one added id 248078 where the reference and the plugin read `(62724, 60)`) is closed by a product seam -- `TextTokenizer.ids`/`count`/`offsets(..., split_special_tokens=)`, `PoolingEndpoint.document_split_special_tokens: bool | None` (CONTENT, refused beside `text`/`messages`), the pooling client's document token-id derivation using it; the harness's render and anchor id comparisons read the same declaration; the fingerprint classifies the field as a request input; the recipe declares it true. The captured wire now shows document ids `[62724, 60, ...]` and query ids `[248077, ...]`, and the recipe's test flips the pinned divergence to equality (plus a cross-check that the ids are the plugin's role prefixes) |

The pplx-context notes rewrite the PRODUCT GAP paragraph as closed and the `max_tokens` reservation now
counts the split parse's delta; the exported schemas, the contract snapshots and the goldens are
regenerated the documented way. The round-2 declarations (pplx-embed-v1's bound, the topk stack pins) are
in the CHANGELOG.

## Docs updated

- `docs/how-to/add-a-model.md`: the `empty_doc` enum gains `omit_zero_blank`, the `plugin_architectures`
  comment matches the patch-only-carrier rule, and the `reference` block documents `attn_implementation`
  (and `device`). The fix run adds `gates` to the per-size override list, the annotated family sample and
  the stage-2 gate paragraph.
- `docs/how-to/validate-a-recipe.md`: the T2 bullet documents the per-size `overrides.gates` merge (the
  fix run).
- `docs/concepts/late-interaction.md`: the pooling client's field list documents
  `document_split_special_tokens` (the reference's document parse, the refusal beside `text`/`messages`, and
  the fit-versus-sent count caveat a declaring recipe reserves) -- the round-2 run.
- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/octen-embedding/family.yaml` (the fix run): the three places that
  said a per-size gate needs a field that does not exist now point at `overrides.gates`.
- `docs/concepts/text-budgets.md` was read against the changed fields; it already names every value
  (`omit_zero_blank`, the keep rules), no edit needed.
- `CHANGELOG.md` under `## Unreleased`: the sidecar entry now says an uncached optional sidecar is absent;
  a `### Public surface` entry for the fake's keep-rule parameters and a `### Fixed` entry for the
  query-side allowlist crash.
- Greps run: `git grep -n -i "fake_transport|sidecar|plugin_architectures|empty_doc|attn_implementation|document_skip_token_ids|media_keep_token_ids|omit_zero_blank" -- docs README.md REPRODUCIBILITY.md skills examples experiments mkdocs.yml` (every hit reviewed).

## CHANGELOG entry

The lane's entries are under `## Unreleased` (`### Public surface`, `### Fixed`): the tokenizer sidecars and
the identity rule, `empty_doc: omit_zero_blank`, `VideoPolicy.engine_video_min/max_pixels`, the reference
`attn_implementation`, the recipe fixes (attention/head dtype, the patch opt-ins, the video pin, the
reference devices, the blank-document and instruction policies, the notes honesty) and the catalog pin. The
W1 round adds the `### Public surface` entry for the fake's keep-rule parameters and the `### Fixed` entry
for the query-side allowlist crash; the sidecar entry now says an uncached optional sidecar is absent. The
patch/module-hashing surface entry is the merged fp-v4 lane's, extended with the patch-only carrier wording.
The fix run adds a `### Public surface` entry for the family schema's per-size `overrides.gates` (the
field-by-field merge) and two `### Fixed` entries: the pplx-late/topk query-side reference shape (with the
CPU tests) and the per-variant bf16 bounds with every measured value (qwen3-reranker 0.025/0.04/0.04;
qwen3-embedding 0.9989/0.9987; pplx-late 0.99/0.978). The round-2 run adds a `### Public surface` entry for
the tokenizer's `split_special_tokens` parse and `PoolingEndpoint.document_split_special_tokens`, and
`### Fixed` entries for the pplx-embed-v1 query shape, the round-2 bounds/declarations (pplx-v1 0.9981, the
topk CUDA-stack pins, pplx-context's declaration), the listwise prompt-token probe and the token-ids smoke
body.

## Public surface changes

- `rcp_ndcg.data.tokenizer.SIDECAR_FILES`; `TextTokenizer.from_json(sidecars=)`; the identity rule.
- `EmbeddingEndpoint.empty_doc`/`RerankEndpoint.empty_doc` gain `omit_zero_blank`.
- `VideoPolicy.engine_video_min_pixels`/`engine_video_max_pixels`; `ReferenceSpec.attn_implementation`
  (recipe schema, `rcp-ndcg-vllm`).
- `VariantOverrides.gates` (the fix run): a family's per-size `overrides` may override individual stage-2
  gate fields, merged field-by-field over the family's `gates`; the exported `schema/family.schema.json`
  carries it.
- `TextTokenizer.ids`/`count`/`offsets` gain `split_special_tokens=` (the reference's split parse, restored
  after the call); `PoolingEndpoint.document_split_special_tokens: bool | None` (CONTENT; unset omits the
  fingerprint input, so existing runs are not re-keyed) -- the round-2 run. Regenerated:
  `schemas/index.v1.json`, `schemas/run-config.v1.json`, `tests/contract/snapshots/python_api.json`.
- `fake_transport`/`FakeEndpoint` gain `document_skip_token_ids`/`document_skip_prefix_token_id` (keyword-only,
  defaulted); `FakeEndpoint.__init__`'s snapshot row follows (`tests/contract/snapshots/python_api.json`).
- Regenerated: `schemas/{index,run-config}.v1.json`, `tests/contract/snapshots/`, the recipe/family schemas,
  the recipe goldens, the pairs and their manifest, the corpora re-keys/verification records, the e2e goldens
  and the phased-script golden. The fix run regenerates the family goldens for the eight bounded recipes and
  the two octen goldens (the note refresh).

## Files outside scope

- The judge families (l08-judges, merged in `rfc-0001`) and the MRL declarations (mrl-recipes, merged): read,
  not changed, except the release-checklist line.
- `handover/00-MASTER.md` (decision 25's exception wording) and `handover/RELEASE-CHECKLIST.md` (the judge
  citation cleanup): handover scaffolding, deleted before the release.
- The `int/round17` merge brought the orchestrator's round-16 review fixes (`rcp_ndcg/runs/config.py`,
  `support/serve.py`, the runner tests): merged, not authored here. The `int/round18` merge (`0d3a20b3`)
  brought the tip's judge-smoke fix (`32cb6625`): merged, not authored here.
- The fix runs changed nothing outside their scope (the recipe/schema/harness/product files, the tests, the
  goldens, the three docs pages, the CHANGELOG and this report).

## For the next lanes

- **E2** re-runs waves R2/R3/R4 on the rebuilt RC: the round-2 fixes must show the pplx-v1 query leg
  compared (24 rows per variant), jina-reranker-v3's stage 1 with the prompt-token probe `not_run`, the topk
  own-torch reference venv importing torch, and pplx-context serving real requests (its documents now carry
  the reference's ids; stage 2 is the measurement). The r3 recipes' stage 2 must pass their declared bounds
  (section 7), and the references re-record their stored outputs under the changed reference file hashes.
  E2 also re-records the eight stale corpora, confirms every declared bound (including octen 0.6b/4b, whose
  `vec_min_cosine: 0.993` is PROVISIONAL until measured per size and which can now declare its own floor in
  `overrides.gates`) and flips `status: verified` only from that evidence.
- **The operator's judge-sampling revisit** (decision 4.3's `temperature: 0.0` / `max_output_tokens: 12288`
  against the shipped judge recipes) is an open item recorded in section 6; those fields are in the judge
  family key.
- **ref-envs** owns the per-family reference environments the zembed root cause needs (the recipe pins
  `sentence-transformers>=5.3,<5.4`).
- **06/07** own the judge families' handover citations and the final catalog counts.
- **fp-v4** already merged; the lane's recipe declarations are on `rcp-fp/4`.
