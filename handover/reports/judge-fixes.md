# Lane `judge-fixes`: the judge instrument's silent failures

**Status: DONE.** Branch `lane/judge-fixes`, tip `cc6ec91a` (plus this report's commit), GATE: PASS on the merged
tree. Base `45b66e1b` (rfc-0001 after l08-judges, integration round 8); `rfc-0001` was merged at `26da5852`.

## Commits

| Commit | Subject |
|---|---|
| `53a2ea0e` | judging: the document-reading rule and the fake seed are part of the instrument; reparse refuses to pool parse versions |
| `8a8a60da` | judging: one window generation per resumed query, a named rubric coverage precondition, and the per-modality window kept |
| `876a10dd` | judging: never silently unjudge a query |
| `787449f7` | judging: the window's media charge counts the prepared refs, and the media check records its outcome |
| `5bfab41f` | judging: the parser's silent failures |
| `8af811c1` | estimate: counts the query text the pass sends, and reports retries as a range |
| `a6f06d08` | docs, schemas and CHANGELOG for the judge-instrument fixes |
| `f457313b` | Merge branch `rfc-0001` (`26da5852`) into `lane/judge-fixes` |
| `12318906` | integration: the records module move reaches the judge family's new imports |
| `d040bc1a` | judging: round-1 verifier fixes |
| `6a8524ae` | judging: round-2 verifier fixes |
| `cc6ec91a` | judging: round-3 verifier fixes |
| (this commit) | handover: the lane report |

The lane history was rewritten once (`git rebase --rebase-merges`, before the round-1 fixes) to drop
`scratch/*.patch` blobs an earlier commit had picked up: the tip tree is byte-identical to the pre-rewrite tree
(`git diff --stat <pre-rewrite-tip> <post-rewrite-tip>` empty), no commit adds `scratch/`, and the merge's second
parent is untouched `rfc-0001`.

## What changed (per brief item)

1. **A1 — `title` and the text-formatting version enter the Family and the record ids.** `Family` gains
   `title` (the default `join` normalised to unset), `text_formatting` (`TEXT_FORMATTING_VERSION`) and
   `fake_seed`; `_plan` sets them and `JudgeConfig.identity()` normalises `title: "join"` and carries the fake
   seed. Title-joined and body-only judgements never pool; a resume across a formatting version re-asks. The
   family, judgement-store, calibration, judge-report and run schemas are regenerated.
2. **A3 — a planned window's text budget.** `run_planned` renders each window at its own size's budget, so the
   same window in any plan shows the same text and reuses its record.
3. **A2 — the fake judge's seed is content.** `JudgeConfig.fake_seed` (one parse, `inference/fake._seed_of_url`,
   before any query string; a URL that names none is the route's default 0) enters the family and the identity;
   `FakeJudge` takes the config's seed over an explicit one.
4. **A6 — `reparse` version guard.** A source at the current parse version (a same-key copy) and a source from a
   newer checkout (a downgrade) are refused with `IdentityError` before anything is written.
5. **B1 — one window generation on resume.** A resumed pass that re-asks a refused window retires the
   later-phase windows its first fit selected with an appended `superseded` tombstone, scoped to the pass's own
   generation (its schedule's sequence ranges and its units: a full pass and a `docs=` subset never retire each
   other's windows; a `windows=` plan retires nothing), so the refit reads one generation and the stage file
   stays append-only (the mirror's immutable parts hold). `records()` resolves a tombstone pair by file order,
   `supersedes` keeps the time-ordered rule for merges, the calibration's coverage skips tombstones, and
   `records_stored` counts live records.
6. **B3/B4/B5 — the rubric schedule.** `RubricSchedule.uncovered_units` names the coverage precondition
   `n_random * w >= n_units`; a pass (and its estimate) whose settings cannot show every unit is refused typed,
   before a call (a `windows=` plan is exempt: the plan decides). A dropped stratified phase is warned and
   recorded as a `phase_dropped` census row. A partial schedule keeps the shipped per-modality window fields it
   did not name, for the pass and the estimate.
7. **F4 — never silently unjudge.** `docs` naming no documents is a `ConfigError`; a Stage A query with fewer
   than two candidates and an empty query pool are `DataError`s; duplicate query ids in a row sequence are
   refused.
8. **C1/C2 — media.** The window's media charge counts the prepared refs the wire sends (the pass prepares each
   document once and both `window_tokens` and `render` read that cache); a recorded size that disagrees with the
   decoded image is warned about and replaced. A passing engine media check is recorded
   (`engine_media_check:ok`, with its deltas logged), the pass's media census exists before the probe and the
   client records into it, so `ok`/`not_checked` rows land in `preprocessing.jsonl`.
9. **D1/D2/D4/D5 — parsing.** A score beyond the double range is an `UnparseableAnswer` (`schema`), never an
   `OverflowError`. An answer equal to the prompt's own worked example is refused (the pass and `reparse` pass
   the example; `Prompt.worked_example` extracts it from the prompt's fenced JSON). The think-strip strips only
   outside the object, and an unclosed opener before the object makes the rest reasoning (a truncated answer's
   draft is never the judgement). A window answered once and refused afterwards keeps the answer's text and its
   parse failure.
10. **D3 — inert queries.** `Template.resolve` substitutes in one pass (a query's placeholder tags render
    literally) and the query slot's framing markup (`<doc`, `<documents>`, `<media`) is escaped; a marker with
    no media is refused.
11. **D7 — prompt pins.** Every shipped prompt's SHA-256 is pinned in `tests/judging/test_shipped_prompts.py`,
    so a wording edit fails CI and states that a changed prompt is a new judgement family.
12. **F1 — estimate.** `estimate` forwards the judge's title rule and the dataset's query-side task instruction
    to `_queries`, counts each planned window at its own size and its own documents (deduped as the pass dedupes
    them), and reports the request range `requests_min`..`requests_max` (one attempt per window .. every window
    retried to `MAX_ATTEMPTS`); the CLI text and the assumptions say so.

## Verification

Three verifier rounds on DeepSeek-V4.1-flash (`:xhigh`), fresh context, two lenses each, all inside the
lane's turn. Every finding below was reproduced by the verifier itself; each fix got a test and the local suite.

**Round 1** (tip `370dee6e`):
- Lens A (correctness): **PASS**; minors: `FakeJudge(seed=…, config=…)` recorded the config's seed but drew with
  the argument's; a query-bearing fake URL (`fake://seed/7?dim=8`) was not part of the seed identity; a rubric
  pass with a covering `windows=` plan was refused by the new coverage check.
- Lens B (regressions/hygiene): **FAIL**; major: `drop_records` rewrote the stage file, so `--mirror`'s
  immutable parts and `restore` broke (a resumed drop failed the final flush and a restore resurrected the first
  generation); minors: the lane history carried `scratch/*.patch`; stale "asks only what is missing" wording.
- Fixed in `d040bc1a`: retirement is an appended tombstone (append-only); the `windows=` check is skipped for
  plans; one seed parse; the config's seed wins; a mirror regression test; history rewritten to drop the scratch
  blobs; wording qualified. New tests: the mirror test, the planned-coverage test, the fake-seed tests.

**Round 2** (tip `d040bc1a`):
- Lens A: **FAIL**; blocker: the calibration's coverage counted tombstones as invalid windows, so
  `calibrate(strict=True)` refused a resumed store that a clean pass passes; majors: the B1 retirement ran for
  `docs=`/`windows=` passes that never re-ask the refused window; a `FakeJudge` explicit seed beside a seedless
  fake URL was not in the identity; minors: a tombstone could be retired twice, the estimate over-counted
  mixed-size plans, `title: "join"` split the identity, stale docstrings.
- Lens B: **PASS**; minors: `records_stored` over-counted after a resume, the tombstone rule depended on the
  wall clock, stale docstrings, the seedless-config edge, the rubric page's unconditional coverage claim, the
  missing report.
- Fixed in `6a8524ae`: coverage skips tombstones; the retirement was restricted (see round 3) ; `records()` is
  file-order for tombstone pairs; `records_stored` counts live records; the seedless config folds the seed;
  `title: "join"` normalises out of the identity; the estimate groups plans by size; docstrings and the rubric
  page updated.

**Round 3** (tip `6a8524ae`):
- Lens A: **FAIL**; majors: the `docs=` exemption left a `docs=` resume with two generations of its own
  schedule, and a full-pass resume retired `docs=` subset windows (the match was by query only); minors: the
  estimate's mean still missed heterogeneous planned windows, `records_stored` counted a parseable torn line, a
  repeated resume appended a second tombstone, a seedless fake URL split the plain client from `FakeJudge`.
- Lens B: **PASS**; minors: the torn-line count and the `_windows_stored` docstring.
- Fixed in `cc6ec91a`: the retirement is scoped to the pass's own generation (schedule sequence ranges + units),
  so a `docs=` resume retires its own first fit and a full pass never retires a subset's; the estimate counts a
  planned window's own documents and dedupes as the pass does; `records_stored` skips a torn last line and the
  CLI's `stored` reports live records; a seedless fake URL is seed 0 for the plain client and `FakeJudge` alike.
  New tests: a `docs=` resume with its own refusal, the subset/full-pass non-interference, heterogeneous planned
  estimate, the torn line.
- **Not re-verified**: the round-3 fixes landed after the third (last) verifier round; the gate ran on them
  (`cc6ec91a`, GATE: PASS) and the focused tests pass, but no fresh verifier saw them. The one accepted residual
  is open question 1.

## Checks

Last commands and their result lines (on the merged tree):

- `ruff format --check .` → `586 files already formatted`; `ruff check .` → `All checks passed!`
- `basedpyright` → `0 errors, 0 warnings, 0 notes`
- `heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider -o faulthandler_timeout=120` →
  `3714 passed, 102 skipped`
- `uv run --no-sync pytest tests/contract tests/docs -q -p no:cacheprovider` → `301 passed, 55 skipped`
- `uv run --no-sync pytest rcp-ndcg-test/tests -q -p no:cacheprovider` → `930 passed, 222 skipped`
- `uv run --no-sync mkdocs build --strict -d <scratch>/site` → built
- `bin/gate lane/judge-fixes` (rev `cc6ec91a`) → every step `exit=0`: ruff/format/basedpyright, pytest
  `3714 passed, 102 skipped`, contract-docs `301 passed, 55 skipped`, mkdocs, test-pkg `930 passed, 222 skipped`,
  recipes, vllm-pkg, vllm-models, run_all (`external_judges ok`; leaderboards `1022 checks, 987 match, 35 known
  deviations, 0 failed`; human study `67/67`; external judges `82/82`), public-names clean, clean tree;
  **GATE: PASS**. (One gate run before this segfaulted once in
  `rcp-ndcg-test/tests/test_record_and_wave.py::test_wave_records_disk_and_evicts_after_the_last_recipe`; the test
  passed in isolation and in every later run — a transient, recorded here.)
- Focused: `uv run --no-sync pytest tests/judging tests/calibration tests/runs tests/core tests/inference tests/cli`
  → `1851 passed, 8 skipped`; the paper anchors (`test_paper_schedule.py`, `test_schedule.py`) unchanged.

## Open questions

1. **Repeated resumes of a persistently refused window append a tombstone per generation** (round-3 minor,
   accepted): the numbers stay one generation per sequence and the refit is correct, but the file grows and the
   later-phase windows are re-asked each time (the refused window retries forever, the review's S6). Deferring
   the retirement until after the pass (retire only the slots the pass did not reuse) would make it idempotent;
   not done here.
2. **The retirement's pool scoping is by sequence range and units, not by a generation key.** A full pass can
   still retire a subset pass's record when their sequence ranges overlap *and* the subset's window happens to
   sit at a later-phase sequence of the full pass. The review's S4/F2 (the pool is not in the store identity or
   the record id) is the root; a pool/generation key in the record id would close it.
3. **The `estimate` library path still returns `calls=0` silently for a Stage A pool of one and for
   `docs={query: []}`**; the CLI's `--estimate` refuses via `preflight`. F4's judging path is fully fixed.
4. **The shipped rubric defaults now refuse a corpus above ~3.33 chunks per document** (`n_random * w <
   n_units`), by design per the brief; operators must raise `random_places`/`random_share` or lower `window`.
5. **The tombstone/`supersedes` split**: the store resolves tombstone pairs by file order, `JudgementSet.merge`
   by `recorded_at` (a cross-store merge of a tombstone against a record whose clock ran ahead can differ). The
   store is the fit's reader; merge is documented as time-ordered.

## CHANGELOG entry

The Unreleased entries added by this lane, verbatim:

**### Public surface**

- **The judgement family carries the document-reading rule and the offline judge's seed** (judge review
  A1/A2): `rcp_ndcg_core.schemas.Family` gains `title` (how a document's title reaches the judge: the default
  join, or `separate`), `text_formatting` (the `TEXT_FORMATTING_VERSION` the pass read the documents under)
  and `fake_seed` (the offline judge's draw seed; a fake URL that names none is the route's default 0). Each
  enters the family key and every record id when set, so title-joined and body-only judgements never pool, a
  resume across a formatting version re-asks, and two fake seeds never share a store; `JudgeConfig.identity()`
  carries the fake seed too, and naming the default `title: join` is normalized out of it as the family
  normalizes it. The family, judgement-store, calibration, judge-report and run schemas are regenerated.
- **`rcp_ndcg.judging.CostEstimate` gains `requests_min`/`requests_max`** (judge review F1): the request range
  at one attempt per window and when every window retries to `MAX_ATTEMPTS`. `estimate` counts the query text
  the pass sends (the judge's title rule and the dataset's query-side task instruction included), counts each
  planned window at its own size and its own documents (deduped as the pass dedupes them), and the CLI's
  `--estimate` text prints the range.
- **`rcp_ndcg.judging.RubricSchedule.uncovered_units`** (judge review B3): the units the balanced random phase
  cannot show (`max(0, n_units - n_random * w)`); a rubric pass (and its estimate) whose settings leave units
  unseen is refused with the precondition named.
- **`Prompt.worked_example`** (judge review D1): the JSON object of a prompt's own fenced example, which the
  parser refuses as an answer; `parse_window` gains the `example=` keyword, and
  `rcp_ndcg.judging.JudgementStore` gains `supersede_records` (an appended `superseded` tombstone retires the
  later-phase windows of a resumed pass, so the stage file stays append-only for the mirror). The
  `superseded` invalid category joins the judgement schema.

**### Fixed**

- **Judging identity and resume** (judge review A1/A2/A3/A6, B1, F4): `title`, the text-formatting version and
  the offline judge's seed enter the judgement family and the record ids; a planned window is rendered at its
  own size's text budget; `reparse` refuses a source at the current parse version (a same-key copy) and a
  newer one (a downgrade) before writing anything; a resumed pass that re-asks a refused window retires the
  later-phase windows its first fit selected with an appended `superseded` tombstone, scoped to the pass's own
  generation (its schedule's window sequences and its units, so a full pass and a `docs=` subset never retire
  each other's windows; a `windows=` plan retires nothing), so the refit never reads two generations and the
  stage file stays append-only for the mirror; the calibration's coverage skips tombstones and `records_stored`
  counts live records (a torn last line included). `docs` naming no documents, a Stage A pool of fewer than two
  documents and a duplicated query id are typed refusals instead of a silently unjudged query.
- **The rubric's coverage and per-modality windows** (judge review B3/B4/B5): a rubric pass whose
  `n_random * w < n_units` is refused before a call; a stratified phase dropped for want of a valid random
  answer is warned and recorded as a `phase_dropped` census row; a partial schedule keeps the shipped
  per-modality window fields it did not name (for the pass and the estimate).
- **The judge's media and parsing instruments** (judge review C1/C2, D1/D2/D3/D4/D5): the window's media charge
  counts the prepared refs the wire sends, a recorded size that disagrees with the decoded image is warned
  about and replaced, and a passing engine media check is recorded (`engine_media_check:ok`, with the client's
  census writing into the store's `preprocessing.jsonl`); a score beyond the double range is an
  `UnparseableAnswer`; the think-strip never rewrites the JSON object and an unclosed reasoning block is
  stripped; an answer equal to the prompt's worked example is refused; a window answered once and refused
  afterwards keeps the answer's text; the query slot is interpolated inert (one-pass substitution, framing
  markup escaped, a marker with no media refused).
- **Prompt pins** (judge review D7): every shipped prompt's SHA-256 is pinned in the suite, so a wording edit
  fails CI and states that a changed prompt is a new judgement family.

## Public surface changes

- `rcp_ndcg_core.schemas.Family`: `title`, `text_formatting`, `fake_seed` (optional; excluded from the key when
  unset); `InvalidCategory` gains `"superseded"`.
- `rcp_ndcg.judging.CostEstimate`: `requests_min`, `requests_max`; `rcp-ndcg.cost-estimate.v1` regenerated.
- `rcp_ndcg.judging.RubricSchedule.uncovered_units`; `rcp_ndcg.judging.JudgementStore.supersede_records`;
  `rcp_ndcg.judging.prompts.Prompt.worked_example`; `parse_window(..., example=)`.
- Regenerated: `schemas/{judgement,judgement-store,cost-estimate,judge-report,calibration,calibration-summary,run-manifest,run-start,run-summary}.v1.json`,
  `tests/contract/snapshots/python_api.json`.
- CLI: `rcp-ndcg judge ... --estimate` text prints the request range; `judge`'s report `stored` counts live
  windows. No new command, flag or exit code.

## Files outside scope

The brief assigns the judging package; these files were changed minimally because the fix lives there (each is
the one home of the concept):

- `rcp-ndcg-core/src/rcp_ndcg_core/schemas.py` — the Family fields, the `superseded` category, `supersedes`.
- `rcp-ndcg/src/rcp_ndcg/data/prepare.py` — the recorded-size validation/overwrite (C1).
- `rcp-ndcg/src/rcp_ndcg/inference/clients/_base.py` — the passing engine media check row (C2).
- `rcp-ndcg/src/rcp_ndcg/inference/fake.py` — the one seed parse (`_seed_of_url`).
- `rcp-ndcg/src/rcp_ndcg/cli/judge.py` — the estimate text, the `stored` count.
- `rcp-ndcg/src/rcp_ndcg/runs/pipeline.py` — the resume wording and `_windows_stored` docstring.

## Docs updated

- `docs/concepts/judges.md` — the `title` config row, the content-field list, the fake seed, refusals keeping an
  answer, record ids (planned budgets, refused/unjudged queries), the identity fields, the tombstone resume, the
  reparse refusals, the estimate's query text and request range.
- `docs/concepts/tournament.md` — the over-large score, the think-strip rules, the worked example, the
  `superseded` category, the per-modality window under a partial schedule.
- `docs/concepts/rubric.md` — the coverage precondition and its `windows=` exemption, the dropped stratified
  phase, the worked-example refusal, the per-modality window.
- `docs/concepts/preprocessing.md` — the prepared-refs media charge, the decoded-size record, the engine check's
  `ok`/`not_checked` outcome.
- `docs/concepts/inference.md`, `docs/concepts/text-budgets.md` — the passing engine media check row.
- `skills/rcp-ndcg/SKILL.md` — the estimate range and the family's document-reading rule.
- `CHANGELOG.md` — the Unreleased entries above.
- `mkdocs build --strict` passes; `tests/docs` passes (snippets and links).

Grep commands run over `docs/`, `skills/`, `README.md`, `examples/`, `experiments/` for every changed name and
claim (`title`, `TEXT_FORMATTING`, `fake_seed`, `reparse`, `input_token_count`, `not_checked`, `for_modality`,
`docs=`, `every document is seen`, `three attempts`, `judgement family (`); every hit that was now false was
updated in the same lane.

## For the next lanes

- The pool/generation key (review S4/F2) is still the root of mixed-pool stores; a record id that carries the
  pass's pool would also remove the residual retirement ambiguity (open question 2).
- The `superseded` tombstone is a new store state: readers outside the store (`run status` now counts live
  records; any future reader of the JSONL must decide how to treat it).
- The prompt pins are literal SHA-256s; a deliberate prompt edit must update
  `tests/judging/test_shipped_prompts.py` and the CHANGELOG, and is a new judgement family.
