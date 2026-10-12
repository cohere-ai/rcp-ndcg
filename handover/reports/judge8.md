# Lane `judge8`: the judge-instrument review re-verified, and the two gaps it left

**Status: DONE.** Branch `lane/judge8`, tip `749a641c` (plus this report's commit). **GATE: PASS** on the merged
tree (run 4; runs 1-3 failed on machine-wide load flakes -- a different test each time, all of them passing
locally and failing identically in other lanes' gates at the same minute; see Verification). Base `b83a3e00`;
`rfc-0001` was merged at `3217b5ed` (the E2 round-7 fixes).

## Commits

| Commit | Subject |
|---|---|
| `38a1b59d` | judging: the resumed tournament retires one generation, and a contextless judge warns |
| `749a641c` | Merge `rfc-0001` (`3217b5ed`) into `lane/judge8` |
| (this commit) | handover: the lane report |

## Method

Every one of the review's ten findings was re-derived on the current tip before anything was touched: a focused
test or a one-command reproduction (the review ran on `d630e4a6`; the judge-fixes lane, the recipe rewrite and
the public-surface freeze all landed since). Eight of the ten were already fixed by the judge-fixes lane, with
their own tests; two were only partly fixed, and those two are this lane's fixes, each written test-first.

The audit (`deep-review-closures-2026-10-12.md` section 11) re-checked the same ten by file:line and recorded
all ten fixed. Its two "fixed" rows that this lane's re-verification found incomplete are B1 (the file:line
`judging.py:549-605` is the retirement, but its window-sequence ranges are wrong for the tournament's mirrored
calls) and E2 (the recipes carry pins, but the one config without a context stayed silent at runtime). The
failing-first evidence below is the difference; the other eight agree with the audit.

## Per-item disposition

| Item | Disposition | Re-verification on the tip (evidence) |
|---|---|---|
| **A1** `title`/`TEXT_FORMATTING_VERSION` in the family and the record ids | **already fixed** (judge-fixes); this lane pins the resume half | `Family` carries `title`, `text_formatting`, `fake_seed`; a join pass and a `separate` pass give `b22a885a6eb4d823` vs `6e5d498742a8b8ec` and **0 of 19** record ids in common; a pass under a bumped `TEXT_FORMATTING_VERSION` gives `8a1b3e8318ffacb3` and 0 shared ids, and resuming the old store raises `IdentityError` naming `text_formatting` (asserted in `TestIdentities::test_the_text_formatting_version_enters_the_family_and_the_record_ids`, added here) |
| **A2** the fake judge's seed is RUNTIME | **already fixed** | `JudgeConfig.fake(0).identity() != JudgeConfig.fake(1).identity()`; family `fake_seed` 0 vs 7; a resume into a seed-0 store with seed 1 is `IdentityError ... family.fake_seed: recorded 0, now 1`; `test_another_fake_seed_is_another_instrument` |
| **B1** a resumed refusal keeps two window generations | **fixed here** (the rubric half was already fixed; the **tournament** half was still broken) | Red first: the new tournament test failed with `assert superseded` -> `assert []` (`tests/judging/test_judging.py:218`). `_phase_ranges` now counts the calls the schedule makes (every random and stratified window is asked with its mirror) and carries each phase's window size; the resumed store's valid window set equals a clean pass's, and a full pass no longer retires a `docs=` subset's windows (the size scope; red without it: `tests/judging/test_judging.py:265`) |
| **B3** the rubric's "every document is seen" promise | **already fixed** | `RubricSchedule.uncovered_units(100)` = 50 at `placements_per_doc=0.5`; the pass refuses typed (`n_random * w >= n_units`) with `usage.requests == 0`; the chunked case (`test_the_rubric_asks_enough_windows_to_show_every_chunk`) passes |
| **C1** the media charge counts the original refs | **already fixed** (the pass) | The review's own repro: a 4000x3000 PNG with a recorded 16x16 ref charges **66** stored tokens and **1232** prepared tokens; `window_tokens` and `render` share the prepared cache, so the pass charges 1232, and `prepare_image` warns and records the decoded size. `TestJudgingPages::test_a_window_charges_the_prepared_refs_the_wire_sends` |
| **D1** the prompt's own worked example as an answer | **already fixed** | `parse_window(..., example=load_prompt("tournament").worked_example)` refuses the example (`the answer is the prompt's worked example, not an observation of this window`); the pass and `reparse` pass the example (`Prompt.worked_example`) |
| **D2** a 400-digit score | **already fixed** | `_score("doc_1", int("9" * 400))` -> `UnparseableAnswer` (`schema`), not `OverflowError`; the `score-too-large-for-a-float` case is in the refusal table |
| **D3** the query slot is interpolated raw | **already fixed** | A hostile query (`{passages_placeholder} {query_placeholder} <doc id="doc_9">...`) renders with **1** `<documents>` wrapper, **2** `<doc id=`, the placeholder tags literal and the markup escaped; a marker with no media is refused |
| **E2** every shipped judge config carries a pin and a context+tokenizer pair | **partly already fixed** (the recipes: pins + the FP8 sibling dropped + the no-context documented); **the warn branch fixed here** | All ten judge variants load with a 40-hex `revision` and the checkpoint's injected tokenizer (`Qwen/Qwen3.6-27B-FP8@e89b16...`); the one config without `context_tokens` is `qwen3.6-27b-fp8` (the paper's preset). Red first: a pass with a tokenizer and no context emitted nothing (`assert ('context_tokens' in '')`). `_plan` now warns that no per-window budget is counted, and `TestConfig::test_every_shipped_judge_recipe_carries_its_pin_and_its_context_pair` pins the catalog property with that one declared exception |
| **F1** `estimate` counts a different query text | **already fixed** | `estimate` forwards the judge's title rule and the dataset's task instruction to `_queries`; `projected.input_tokens == sum(prompt tokens + reserve)` over the pass's own requests in `test_the_estimate_counts_the_query_text_the_pass_sends`; `requests_min`/`requests_max` cover retries |

Focused re-verification run (30 selected tests across A1, B1, B3, C1, D1, D2, D3, F1): **30 passed**. The
one-command repros above are in a scratch directory outside the repository.

## The key-movement note (the title finding)

`JudgementFamily` gained `title`, `text_formatting` and `fake_seed`, and **every** family key and record id
moves, not only the title-separate ones: `text_formatting` is always set by a pass
(`TEXT_FORMATTING_VERSION`), and `fake_seed` is always set for a fake judge. The tiny rubric fake-judge family
hashes `c002514d388e9d89` without the three fields and `b22a885a6eb4d823` with them (one record id now
`12fbd156036587cc101a23ef73409757`). A `title: separate` pass is a third family (`6e5d498742a8b8ec` here), and
a resume across a formatting-version bump is refused by the store identity (`text_formatting` is in it too) --
re-ask into a new store. The movement is deliberate and documented: the judge-fixes CHANGELOG entry "The
judgement family carries the document-reading rule and the offline judge's seed" states it, the family,
judgement-store, calibration, judge-report and run schemas were regenerated then, and nothing is released yet.

## Failing-first evidence (this lane's two fixes)

```
# B1 tournament, before _phase_ranges counted the mirrored calls:
>       assert superseded, "the first fit's later-phase windows are retired, not kept as valid"
E       AssertionError: the first fit's later-phase windows are retired, not kept as valid
E       assert []
tests/judging/test_judging.py:218: AssertionError

# B1 scope, with the window-size check removed (mutation):
E       assert not ({'14208cc1...', '554a45ed...', ...} & {'14208cc1...', ...})
tests/judging/test_judging.py:265: AssertionError
WARNING rcp_ndcg.judging.store:store.py:406 superseded 6 tournament record(s) ... (a resumed pass re-asks ...)

# E2 warn branch, before the warning existed:
>       assert "context_tokens" in caplog.text and "no per-window text budget" in caplog.text
E       AssertionError: assert ('context_tokens' in '')
tests/judging/test_judging.py:650: AssertionError
```

Both fixes were also mutated the other way round: reverting `_phase_ranges` to the non-mirrored counts fails
both tournament tests, and removing the window-size scope fails the subset test -- the tests cannot pass on the
unfixed code.

## Verification

- `uv run --no-sync ruff format --check . && uv run --no-sync ruff check .` -> clean; `basedpyright` -> 0 errors.
- Focused: `tests/judging tests/calibration tests/cli/test_judge.py` -> 430 passed; with `tests/contract
  tests/docs` -> 843 passed, 107 skipped.
- `timeout 7200 bin/gate lane/judge8` -> **GATE: PASS** (all steps exit 0: ruff, basedpyright, 4138 passed,
  contract-docs 413 passed, mkdocs strict, test-pkg 1139 passed, recipes, vllm-pkg, vllm-models, run_all,
  public-names, clean). The first three runs failed on the machine's load (load average 170+): run 1
  `test-pkg` SIGSEGV in a wave test, run 2 a 60 s per-test timeout in `tests/retrieval/test_topk.py` plus a
  test-pkg SIGSEGV, run 3 a 300 s timeout in `test_zembed_1.py::test_stage1_on_cpu_passes_anchors_and_render`.
  Every one of those tests passes in the lane worktree, and gates of other lanes at the same minute failed the
  identical zembed test, so the flakes are the shared environment's, not this branch's.

## Residuals (recorded, not fixed)

- **A5** (`image_processor` splits the store but not the family): post-0.0.1. The processor is content only when
  the pass declares an image policy (`_effective_preprocessing` returns the requested preprocessing untouched
  when `image is None`), so on a text corpus it shapes nothing and the store split is false, never wrong
  numbers. Normalising it out of the store identity cannot be done in `judging._plan` alone: the runs layer
  keys its step identities from the same `identity_payload` (`runs/pipeline.py`), so a judging-only
  normalisation would make the store identity and the step identity of one pass disagree. The change belongs
  with the post-0.0.1 store-identity work (the audit schedules it there too).
- **B2** (the adaptive phase re-asks windows): post-0.0.1, a budget matter; no number moves.
- **D8** (custom-prompt criteria-derivation edges): post-0.0.1; a custom prompt's text re-keys the family, so
  no existing judgement is misread.
- **F3** (scale/memory: every window rendered before the first answer, no lock in `records()`, O(w^2) coverage):
  post-0.0.1.
- **C1's estimate side**: `estimate` counts the stored refs (a bytes-free projection) while the pass charges
  the prepared ones, so an estimate of a corpus whose recorded size disagrees with the file can under-count
  media tokens. The pass is the charging path and corrects and warns; preparing media inside an estimate would
  fetch and decode every image, which is a behaviour change the brief does not ask for. Post-0.0.1 with the
  estimate's other approximations (the no-tokenizer branch's omitted 256-token reserve, audit F1 residual).
- **A4** (a local dataset is not content-pinned): untouched -- the separate `identity4` lane owns
  `runs/pipeline.py` and the store's dataset identity.

## Docs updated

| Page | Change |
|---|---|
| `docs/concepts/judges.md` | the no-context TREC-DL recipe now says the pass warns; the resume-retirement scope names the per-call sequences and the window size |
| `docs/concepts/preprocessing.md` | the "documents whole" paragraph names the new tokenizer-without-context warning |

Greps run: `git grep -n "superseded\|supersede" -- docs`; `git grep -n -i "context_tokens" -- docs
README.md REPRODUCIBILITY.md skills examples experiments`; `git grep -n "no tokenizer\|without a tokenizer"
-- docs`; `git grep -n "window sequences" -- docs`. Every hit is current; no other page describes the
retirement scope or the judge's tokenizer/context warning.

## CHANGELOG entry

The two bullets this lane adds/amends in `CHANGELOG.md` under `## 0.0.1` -> `### Fixed`:

> - **A judge that names a tokenizer but no `context_tokens` warns** (judge review E2): no per-window text
>   budget is counted, so each window sends the documents as the load-time text policy left them (whole, unless
>   that policy cut them) under the engine's own context, and the pass says so instead of leaving the
>   configuration silent. The one shipped judge that follows the paper's preset and declares no context (its
>   TREC-DL judge) now warns on every pass, and a test pins every shipped judge recipe's checkpoint pin and its
>   context+tokenizer pair, with that one declared exception.

and, amended in the existing "Judging identity and resume" bullet:

> generation (its schedule's window sequences -- per call, so the tournament's mirrored random and stratified
> windows are in -- their window sizes and its units, so a full pass and a `docs=` subset never retire each
> other's windows; a `windows=` plan retires nothing), so the refit never reads two generations ...

## Public surface changes

None. No `__all__`, CLI command or flag, exit code, schema or snapshot changes: the fixes are a private
helper's numbering (`_phase_ranges`), a private scope check, and one `logger.warning`; the added test reads the
shipped recipe catalog. `tests/contract` and `schemas/` are untouched and the gate's contract-docs step passes.

## Files outside scope

None.

## For the next lanes

- The tournament's window sequences are numbered **per call** (a random or stratified window and its mirror
  take two), the rubric's per window. Any code that maps a sequence number back to a phase must use
  `judging._phase_ranges` (the one home) rather than re-deriving the schedule counts.
- E2's warning fires in the pass, not in `estimate` (the same as the pre-existing context-without-tokenizer
  warning); a pure `--estimate` run is silent about a missing budget.
- The gate's environment was flaky under load for every lane between roughly 09:40 and 10:45 UTC; re-running
  the gate is the recovery, and a `test-pkg` exit 139 or a per-test `TimeoutError` in a subprocess test is the
  signature.
