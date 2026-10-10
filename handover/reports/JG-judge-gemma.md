# Report: lane `judge-gemma` — Gemma 4 judges for the judge catalog (research and spec only)

**Status: DONE.** The deliverable is section 7 ("Gemma 4 judges") of `handover/specs/judge-catalog.md`: the four
judges' Hub pins, engine support at vLLM v0.31.0, the serve blocks with the memory arithmetic, the client
blocks, the family grouping under decision 34, the E2 GPU validation plan and eight open questions. No code and
no recipe files were touched; lane `l08-judges` builds the recipes from the section.

## Commits

1. `f8ea3189` — handover: the Gemma 4 judges — four recipes in three families, pinned and checked at vLLM
   v0.31.0 (the section; 576 lines appended to `handover/specs/judge-catalog.md`, no line of the six existing
   judges changed).
2. `0ca86eb5` — handover: the Gemma 4 judges — verifier round 1 fixes (the two verifiers' minor findings: one
   unit for the KV arithmetic, the per-TP KV page sizes, the 31B's F8 scale bytes, the T0 dtype source, four
   citations, the sampling/trust-remote-code wording).
3. `9ab77ef9` — Merge branch `rfc-0001` into `lane/judge-gemma` (merged `rfc-0001` = `89e7a3b6`, lane `l10a`'s
   workstream-10 merge; no conflict, no drift: the only overlapping path is `handover/`, and `l10a` added its
   own report).
4. `eda5b5a9` — handover: the judge-gemma report (this file).
5. `ebc156ab` — Merge branch `rfc-0001` into `lane/judge-gemma` (the then-current `rfc-0001` = `e3a356f1`,
   lane `mrl-cards`' spec and report; clean, no conflict).
6. `2c15bcf4` — Merge branch `rfc-0001` into `lane/judge-gemma` (the current `rfc-0001` = `28afb3b7`, lane
   `l08-sglang`'s workstream-08 A merge, "vLLM only"; clean). The merge moved `_watch_reasoning` to
   `chat.py:413-431` and extended its warning text, so `634b5ab2` follows it (below); nothing else in the
   section cites a file the merge changed.
7. `634b5ab2` — handover: the Gemma 4 judges — the reasoning-watch citation follows the vLLM-only merge. The
   gate ran on this head.

## What changed (per brief item)

1. **Engine support at vLLM v0.31.0** (section 7.2): both architectures are registered at the tag —
   `Gemma4ForConditionalGeneration` (`registry.py:418`, module `gemma4_mm`) and
   `Gemma4UnifiedForConditionalGeneration` (`registry.py:419-421`, module `gemma4_unified`); no judge needs a
   newer image. The ModelOpt NVFP4 path is `modelopt_fp4` (`modelopt.py:745-746`, auto-detected at
   `:756-761`), W4A4 (the checkpoints' `config_groups` have 4-bit activations, so the W4A16 downgrade at
   `:775-791` does not fire); the KV dtype resolves from `kv_cache_scheme` to `fp8_e4m3`
   (`utils/torch_utils.py:67-70,448-521`); the dense GEMMs take the `--linear-backend` selection and the 26B's
   experts the `select_nvfp4_moe_backend` order (`fused_moe/oracle/nvfp4.py:182-203`), with the Marlin W4A16
   fallback on Hopper and the 26B card's TP=1 constraint recorded. The reasoning parser is `gemma4`
   (`reasoning/__init__.py:55-58`); the checkpoint has a thinking channel, off by default; with thinking off
   the template pre-closes an empty channel and the grammar constrains the first generated token, with thinking
   on the grammar starts at the channel end (`parser/gemma4.py:297-320,409-581`,
   `parser/engine/parser_engine.py:650-672`, `v1/structured_output/__init__.py:235-262`). Attention backends:
   `TRITON_FLASHINFER` on Blackwell, `TRITON_FLASH_ATTN` with FA4 on Hopper (the 512-wide full layers). Media:
   patch 16, pooling 3, 280 image soft tokens, 70 per video frame, 32 frames, a per-frame timestamp.
2. **Serve blocks** (sections 7.3 and 7.4): per family the TP size and `resources.gpus` (12B TP1; 26B bf16
   TP2; 26B NVFP4 TP1; 31B NVFP4 TP2), the KV arithmetic in GiB at `gpu_memory_utilization` 0.92 on an 80 GB
   class and a B200 (weights, full-layer KV per token, the HMA sliding-window state per sequence, the cache
   fits), `--max-model-len 131072` with the 32-bit index check (input ids/cu_seqlens/block tables, the router's
   packed sort key, the 32 KiB full-layer pages at every served shape), `--limit-mm-per-prompt '{"image": 10}'`,
   `--quantization modelopt_fp4` and `--kv-cache-dtype fp8_e4m3` on the NVFP4 variants, and the flags
   deliberately not declared (dtype, chat template, trust-remote-code, the kernel backends, gpu memory
   utilization, tool parsers, media pinning, MTP).
3. **Client blocks** (sections 7.1, 7.4 and 7.5): the `rcp_ndcg.judging.JudgeConfig` fields per family
   (`temperature: null` plus the cards' `extra_body: {top_p: 0.95, top_k: 64}`, `max_output_tokens: 16384`,
   `context_tokens: 131072`, the per-variant `tokenizer@revision`, `concurrency: 256`, `decoding: json_schema`,
   `image_processor: gemma4`, `max_images: 10`, `max_videos: 0`). Lane `rec-egemma2`'s `gemma4` geometry is
   reusable for images (patch 16, pooling 3, 280 soft tokens; the tower and unified resize functions are
   byte-identical at transformers 5.17.0) and is a hard dependency (that lane is not on `rfc-0001`); its video
   budget (140 per frame) is not the chat checkpoints' (70), so the judge's video path is `wire: frames`, which
   sizes each frame by the image policy.
4. **Family grouping** (section 7.1): `gemma-4-12b` (the unified 12B, the only audio judge),
   `gemma-4-26b-a4b` (bf16 + NVFP4, one model in two quantisations; the templates differ in file but the
   judge-shaped render is byte-identical, reproduced), `gemma-4-31b` (NVFP4 only). Evidence: identical
   processor config, byte-identical `tokenizer.json` (sha256 `cc8d3a0c…`), the render check.
5. **GPU validation plan for E2** (section 7.6): per judge the T0 serve smoke (with the NVFP4 kernel lines
   recorded), the structured-output conformance (both stages, both thinking modes), the parse conformance, the
   media probe, the MoE probe for the 26B pair, and the E2E scenarios 1/3/4; pass criteria and the `status`
   transition.

## Verification

- **Round 1, two fresh verifiers, both on DeepSeek-V4.1-flash (xhigh), run in parallel** (lens A: correctness
  against the brief; lens B: regressions and hygiene; each told the other exists).
  - **Lens A: VERDICT PASS.** 10 minor findings, all verified and all fixed: the full-layer KV page sizes were
    wrong for the 26B/31B served TPs; the 12B and 31B B200 cache-fit numbers mixed GiB and GB; the 31B weight
    composition omitted the F8 scale tensors (and the 26B's F32 scalars); "ready for commercial/non-commercial
    use" is on the NVIDIA card only; the `--trust-remote-code` explanation was wrong for the 26B card (its
    v0.20.0 target already has the architecture); two citations pointed at the wrong lines (`cu_seqlens` vs an
    `mm_prefix` comment; `extract_reasoning` vs the tool-call-as-reasoning-end docstring); the chat-template
    pickup citation pointed at the `add_special_tokens` block; and the named report did not exist yet (this
    file). The verifier reproduced the Hub pins, the byte-identical tokenizer and renders, the config facts,
    the weight headers, every vLLM citation, the guided-decoding chain, the resize port and the rec-egemma2
    dependency.
  - **Lens B: VERDICT PASS.** 10 minor findings, all verified and all fixed: the same unit inconsistencies
    (12B ~59→~57 and ~165→~160; the 26B bf16 components; the 31B B200 figures), the page sizes, the 12B TP2
    aside (only the full layers' head replicates; the sliding state halves), the 31B's F8 scales, the
    `--gpu-memory-utilization` mismatch (the B200 arithmetic now uses the same default 0.92), the
    `/v1/models` `dtype` field (v0.31.0 has none; the T0 smoke now records dtype from the engine log), the
    `registered_adapters.py` line range, and the "six judges send no temperature" wording. The verifier also
    confirmed the append-only diff (581 additions, 0 deletions), the public-names scan, the ruff/docs checks,
    all 70 `file:line` citations, the pins, the render identity, the harness terms and the commit hygiene.
- No blocker and no major finding in round 1, so no round 2 (the lane protocol's rule). The verifiers' own
  reproductions live in a scratch directory outside the repository. After the verifier round, the `rfc-0001`
  merges were checked for drift: `l10a` and `mrl-cards` touch no path the section cites; `l08-sglang`
  (workstream 08 A) moved the client's reasoning watch and extended its warning, and the section's citation
  was updated to `chat.py:413-431` with the warning's new caveat (commit `634b5ab2`).

## Checks (last runs)

- `bin/gate lane/judge-gemma` on the final merged head `634b5ab2` (`rfc-0001` = `28afb3b7`): **GATE: PASS** —
  `ruff-check` 0, `ruff-format` 538 files already formatted, `basedpyright` 0 errors, `pytest` 3269 passed /
  93 skipped, `contract-docs` 287 passed / 52 skipped, `mkdocs` built, `test-pkg` 570 passed / 225 skipped,
  `recipes` no failure outside the baseline (34 baseline failures remain), `vllm-pkg` 1 passed, `vllm-models`
  70 passed / 7 skipped, `run_all` 1022 checks / 987 match / 35 known deviations / 0 failed, human study
  67/67, external LLM judges 82/82, `public-names` clean, tree clean. The gate also passed on the earlier
  merged heads `9ab77ef9` and `ebc156ab`; the counts moved between runs only with `rfc-0001`'s own merges
  (`l10a`, `mrl-cards`, `l08-sglang`), never with this lane's changes.
- `git merge-base --is-ancestor rfc-0001 HEAD` after the final merge: exit 0 (`rfc-0001` = `28afb3b7`).

## Open questions

The section's own eight, summarised: (1) the 26B family shape (the template files differ but the judge render
is identical — grouped here, splittable); (2) thinking off (recommended) vs on; (3) the cards' `extra_body`
sampling vs the catalog's no-sampling convention; (4) `max_model_len` 131072 vs the available 262 144;
(5) video: `wire: frames` vs a per-checkpoint video budget in the product; (6) a bf16 31B sibling; (7) the 26B
NVFP4 card's producer/backend notes vs what the E2 wave observes; (8) audio on the 12B is out of scope.

## The owner's answers (lane `l08-judges`, 2026-10-09) and what shipped

The owner answered all eight for the recipe lane; the four recipes now exist as
`gemma-4-12b-it`, `gemma-4-26b-a4b-it`, `gemma-4-26b-a4b-nvfp4` and `gemma-4-31b-it-nvfp4`
(`rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/`).

1. **The two 26B variants stay one family** (`gemma-4-26b-a4b`): the judge-shaped render is byte-identical
   across the two checkpoints' own templates, and the lane re-derived it through the tokenizer's own
   `apply_chat_template` for thinking off and on (section 7 of the lane report below). They differ only in
   repo, revision, quantisation, licence and the flags the quantisation implies.
2. **Thinking OFF by default; thinking on is a declared client choice**: `--set judge.extra_body='{"chat_template_kwargs":
   {"enable_thinking": true}}'` (the `gemma4` parser reads the template kwarg, not `reasoning_effort`); with
   thinking off the client's advisory reasoning-watch warning is a declared false positive, and the recipe
   notes say so. `judge check` reports the reasoning channel per stage.
3. **The catalog's convention wins over the cards' sampling**: no `extra_body` sampling in the recipes; the
   cards' temperature 1.0 / top_p 0.95 / top_k 64 are documented as an opt-in in every recipe's notes.
4. **`--max-model-len 131072`** in all four recipes, with `client.context_tokens: 131072`.
5. **Video via `wire: frames` only** (the per-checkpoint 70-token video budget and the per-frame timestamp
   charge are later product work); `max_videos: 0` in the recipes.
6. **No bf16 31B sibling in 0.0.1**; the family carries the NVFP4 release only.
7. **Backends recorded in E2**: no linear/MoE backend pin; the wave records the chosen backends and any
   Marlin fallback warning. The 26B NVFP4 card's TP1 constraint matches the decision-41 shape.
8. **Audio out of scope** on the 12B.

Decision 41 also corrected section 7's serve blocks: all four judges declare `resources.gpus: 1` (TP1) -- the
26B bf16 and the 31B NVFP4 included -- and scale by replicas, with each recipe's notes stating the arithmetic
for one B200 and one H100. The lane verified every recipe on CPU (schema load, `JudgeConfig` validation,
`rcp-ndcg-vllm serve <id> --dry-run`, per-variant golden) and left `status: unverified` until the E2 wave.
The full account is `handover/reports/08bd-judge-recipes.md`.

## CHANGELOG entry

None. `handover/` is temporary scaffolding deleted before the release, and this lane moves no public name, CLI
command, exit code or schema.

## Public surface changes

None (one handover spec file; no code, no `__all__`, no CLI, no schema, no snapshot).

## Files outside scope

None. One file changed (`handover/specs/judge-catalog.md`), plus this report.

## Docs updated

None. No page under `docs/`, `README.md`, `REPRODUCIBILITY.md`, `skills/`, `examples/`, `experiments/` or
`mkdocs.yml` describes the Gemma 4 judges yet — they are recipes lane `l08-judges` will build, and their docs
land with those recipes. Sweep run: `git grep -n -i "gemma-4\|gemma4" -- docs README.md REPRODUCIBILITY.md
skills examples experiments mkdocs.yml` → no hits.

## For the next lanes

- **`l08-judges`**: build the four recipes from section 7. The hard dependency is lane `rec-egemma2`'s
  `gemma4` processor family in `rcp_ndcg.data.resolution` (without it `image_processor: gemma4` does not
  validate); the judge role needs the schema fields the serve blocks name (`tensor_parallel_size`,
  `reasoning_parser`, `quantization`/`kv_cache_dtype` or `extra_args`); the client block is the JudgeConfig
  shape, so `extra_body` carries the cards' sampling.
- **The E2 wave**: record the NVFP4 linear/MoE backends from the engine log (the card's backend notes are from
  an older vLLM), the media token count against the client's, the two thinking modes' structured-output
  results, and the 26B NVFP4 TP1 constraint; then move each recipe's `status` to `verified`.
- **A future video corpus**: `wire: frames` works today; `wire: video_url` needs the product's `gemma4` video
  budget to be per-checkpoint (70 here) and the timestamp charge counted.
