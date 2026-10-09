# Lane report: `l08-cat` — the judge catalog as a spec (workstream 08 C, research only)

Date: 2026-10-09. Branch `lane/l08-cat`, head `ee317278`, based on `rfc-0001` tip `511698ef` (the merge of the
current `rfc-0001` was "Already up to date": the branch is `511698ef` plus this lane's three handover commits; the
gate ran on the merged tree).

## Status

DONE. The deliverable is `handover/specs/judge-catalog.md` (research only, no code, no recipe files) plus the
required copy of the parallel plan at `handover/specs/parallel-08-09-10.md`. Two verifier rounds passed; every
round-1 major/minor and the round-2 residual minor are fixed.

## Commits

| Hash | Subject |
|---|---|
| `3c1f0944` | handover: the judge catalog as a spec (workstream 08 C) — six judges pinned at Hub revisions, vLLM v0.31.0 registry and flag citations, serve/client blocks, GPU arithmetic, decision-34 families; the parallel 08/09/10 plan copied into the specs |
| `15377a32` | handover: the judge catalog's arithmetic corrected to on-disk weights (index shard sums) and per-GPU KV (verifier round 1) |
| `ee317278` | handover: the judge catalog's round-2 confirmation — the full-length-sequence example counts the post-state 33 GB cache (4, not 5) and the Flash-Next text-config identity is stated as serving-relevant |

## What changed

- **`handover/specs/judge-catalog.md`** (new): for each of the six judges (decision 15) — Hub repo, 40-hex revision
  with the Hub API check date (the three Qwen3.8 pins carried from `rcp-ndcg-test/scenarios/*.yaml`, re-verified
  2026-10-09; qwen3.5/gpt-oss keep the paper's own pins; qwen3.6-27b-fp8 resolved fresh — its paper revision is
  unrecorded), licence and gated status, `config.json` architectures, the vLLM v0.31.0 registry line per
  architecture (file:line at tag commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`), the serve block as vLLM flags
  translated from `experiments/paper/serve/*` with every flag cited in the v0.31.0 source (one shared
  translation table plus the dropped SGLang knobs, declared), the client block from today's presets and the T4
  scenarios, `resources.gpus` with the memory arithmetic (weights from the safetensors index shard sums; KV per
  token from the full-attention layers; per-sequence linear-attention state), and the decision-34 family grouping:
  five families, six recipes — `qwen3.8-flash-next` is the only multi-variant family (the two quantisations); the
  two 27B judges stay separate (architecturally identical, but different generations and materially different chat
  templates), with the grouping evidence recorded.
- **`handover/specs/parallel-08-09-10.md`** (new): the parallel plan copied into the branch as the brief required.
- vLLM was cloned at tag v0.31.0 into a scratch directory outside the repository for the
  citations; Hub facts were resolved anonymously (public repos only) on 2026-10-09.

## Verification

- **Round 1, lens A (correctness)**: VERDICT PASS. Independently re-queried all six pins at the Hub API, re-read
  every vLLM v0.31.0 file:line citation, every repo-side citation, recomputed the arithmetic. Found 3 majors
  (F1/F2: my KV passages conflated elements and bytes — Flash-Next per-GPU and gpt-oss per-GPU KV fixed to
  12 288 / 9 216 B per token per GPU; F3: gpt-oss weights are 65.2 GB on disk, not the 119.0 GB params×dtype
  estimate — the index's packed-MXFP4 total is what the engine loads) and minors (F4: Flash-Next-NVFP4 weights are
  132.7 GB on disk incl. FP8 scales, not 125.1; F5: the parallel-plan copy is one row behind the live draft; F6:
  the repo@revision sentence overreached — only the qwen3.5 preset and the scenarios use that shape; F7: a citation
  range covered GEMM selection only, not attention backends; N1/N2 nitpicks). **All fixed** in `15377a32` after
  re-verification against the index shard sums (gpt-oss 65.25 GB, NVFP4 132.68 GB, FP8 185.52 GB, 27B pair
  30.87 GB).
- **Round 1, lens B (hygiene)**: VERDICT PASS. Scope clean (two handover files only), commits clean (no co-author
  or AI attribution, configured identity, base unchanged, 0 commits behind `origin/rfc-0001`), no private names and
  no literal chat-template special tokens, R30 pointers resolve to real product symbols, citation mutation test
  discriminative. Findings: the copied parallel plan was one row behind the live draft (reconciled at merge);
  write this report.
- **Round 2 (confirmation, lens A+B, fresh)**: VERDICT PASS. All six round-1 fixes confirmed correct with exact
  recomputation; hygiene re-confirmed. One residual minor fixed in `ee317278` (the qwen3.8-27b example's
  "≈ 5 full-length sequences" → "≈ 4" against the post-state 33 GB) and one nitpick fixed with it ("identical
  text configs" → "identical serving-relevant text configs", the three inert keys named with the vLLM default
  citation).

## Checks

The full gate (`bin/gate lane/l08-cat`) on the merged tree:

- First run on `3c1f0944`: every step green **except `test-pkg exit=139`** — a hard SIGSEGV mid-suite at ~96%
  with no test failure (the same step passed on the base revision `88f6e949` minutes earlier; the lane's diff is
  two markdown files). Diagnosed as a flaky segfault in the shared `wt-int` environment, not a tree defect.
- Re-run on the same revision: **GATE: PASS** — ruff-check 0, ruff-format 525 files ok, basedpyright 0 errors,
  pytest 3182 passed/82 skipped, contract+docs 270 passed/52 skipped, mkdocs strict ok, rcp-ndcg-test 570
  passed/225 skipped, vllm-pkg 1 passed, vllm-models 70 passed/7 skipped,
  run_all `leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed` / `human study: 67 checks, 67
  match, 0 failed` / `external LLM judges: 82 checks, 82 match, 0 failed`, tree clean.
- The two post-gate commits (`15377a32`, `ee317278`) touch only `handover/specs/judge-catalog.md` (markdown that
  no gate step collects: `handover/` is excluded from `tests/docs/_markdown.py` and absent from `mkdocs.yml`).
- Docs greps: `git grep -n "judge-catalog" -- docs/ README.md REPRODUCIBILITY.md skills/ mkdocs.yml examples/`
  → no hits (the deliverable is handover scaffolding; no product docs describe the judge catalog yet, so nothing
  became false).

## CHANGELOG entry

None. The lane changed no public surface: no code, no CLI, no schemas, no snapshots; `handover/` is temporary
scaffolding that is deleted before the release.

## Public surface changes

None.

## Files outside scope

None.

## For the next lanes

- **`l08-judges` (08 B+D)** implements from this spec: five family directories, six recipe ids
  (`qwen3.5-397b-a17b-nvfp4`, `gpt-oss-120b`, `qwen3.6-27b-fp8`, `qwen3.8-27b-fp8`, `qwen3.8-flash-next-nvfp4`,
  `qwen3.8-flash-next-fp8`); the serve blocks, client blocks and `resources.gpus` arithmetic are in section 3, the
  flag citations in section 2. The preset removal (08 D) keeps `gpt5_hosted` and drops `qwen35_397b_fp8`.
- The eight open questions at the end of the spec (spec section 6) need owner/lane decisions before or during
  implementation: qwen3.6-27b-fp8's engine settings and pin, served names, `max_images`/`concurrency` defaults,
  the qwen3.8-27b context length, KV dtype policy, the family-shape alternative, and MTP heads.
- Hub facts are point-in-time (checked 2026-10-09); five of six pins equal the Hub head today and can move ahead
  of their pins — re-verify before the GPU waves.
- The gate's `test-pkg` step segfaulted once (exit 139, no failing test) and passed cleanly on re-run: if it
  recurs on another lane, it is environment flakiness, not a tree defect.

## For the operator

The vLLM v0.31.0 clone and the Hub research artifacts (API responses, pinned `config.json`s, tokenizer configs)
are in a scratch directory outside the repository for re-checking.
