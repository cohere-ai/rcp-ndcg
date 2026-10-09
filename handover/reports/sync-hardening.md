# Lane `sync-hardening`: bounded outages, atomic mirror state, the stated sync guarantee

**Status:** DONE.

**Base:** `rfc-0001` tip `28afb3b7` (the merge `git merge rfc-0001` reported "Already up to date" at the end;
`rfc-0001` did not advance during the lane).

## Commits

| Commit | Subject |
|---|---|
| `5f96a7c5` | runs: bound the outage wait and add a per-step wall-clock budget |
| `411f8985` | judge: --mirror-interval, matching the run config's mirror_interval_s |
| `cf7f7488` | runs: publish mirror state and local whole files atomically |
| `5ad76462` | docs: the outage default, the step budget and the mirror guarantee |
| `dabb77ff` | runs: fix the verifier round: shared-mirror file modes, publish temps, budget wording |

## What changed

- **O1a, the outage default.** `Endpoint.wait_on_outage_s` defaults to `1800.0` (an engine restart plus a large
  model's load); `wait_on_outage_s: null` remains the explicit "wait indefinitely" choice. The transport's
  `BackendUnavailableError` now carries a `hint`/`cli_hint` naming the field and the `null` choice. Updated in
  `inference/endpoint.py`, `inference/transport.py`, the three exported schemas and the docs.
- **O1b, `step_budget_s`.** New `RunConfig.step_budget_s: float | None` (default `None` = unbudgeted, `gt=0`).
  The budget is checked at the request seams: the shared transport checks it before a request is queued and
  after every park (and bounds a park's wake by the remaining budget), and the judging pass checks it before a
  phase's windows. A step over budget raises the new `rcp_ndcg.errors.StepBudgetExceededError`
  (`Interrupted`, exit code 9), the run marks the step failed, the judgement store keeps every record it wrote,
  and a resume with `step_budget_s: null` continues from there. `step_budget_s` is runtime-only
  (`Pipeline._substance` pops it, so changing it never re-keys or refuses a resume).
- **S1, atomic mirror state.** `.mirror.json` (or `logs/mirror.json`) is written with
  `storage.publish_bytes` (temp + rename); `read_state` treats an unparseable file as "never ran" with a
  `logger.warning`, as the judgement store treats a torn identity. `Run.status()` can no longer crash on a torn
  state file.
- **S2, atomic local whole files.** `_Target.write` routes a local/shared POSIX mirror through
  `storage.publish_bytes`; a remote target keeps `pipe_file`. The decision uses the protocol of the mirror URI
  (a stripped `memory://` or custom-filesystem root reads as local otherwise). `storage.publish` now keeps the
  mode a plain write would give the file (an existing target's mode, else `0666 & ~umask`) and names its temp
  `*.tmp`, which the mirror's walk and `restore()` skip.
- **O4, `--mirror-interval`.** `judge tournament|rubric` gains `--mirror-interval <seconds>` (default the one
  `DEFAULT_INTERVAL_S` constant, 60), passed as `mirrored(interval_s=...)`; it matches the run config's
  `mirror_interval_s`.
- **S3/S4/S6 docs.** `docs/concepts/runs.md` now states the guarantee: durable = the last uploaded part; a
  hard kill loses at most one interval, re-asked on resume and never duplicated (`record_id`); one live writer
  per store, a diverged writer's flush refuses with `DataError` and the run continues unmirrored, shown by
  `run status`; parts and `.superseded/` files are never garbage-collected.

## Verification

**Failing tests first (red), then green.** One command ran the 12 new/updated assertions before the fix:
`uv run --no-sync pytest -q -p no:cacheprovider -o faulthandler_timeout=120 <the 12 tests>` →
`12 failed, 4 passed`. Examples: `TestEndpoint::test_wait_on_outage_s_moved_up_from_the_judge` failed on
`assert None == 1800.0`; `TestStepBudget::test_a_step_over_its_budget_stops_typed_and_the_store_resumes` failed
on the unknown `step_budget_s` field; `test_a_torn_state_file_reads_as_never_ran_with_a_warning` failed with a
pydantic `ValidationError` escaping `read_state`; `test_a_local_mirror_publishes_whole_files_through_the_storage_helper`
failed on the missing `manifest.json` publish; `test_judge_mirror_interval_reaches_the_mirror` failed with the
`mirrored() missing interval_s` `TypeError`.

**Round 1, two fresh independent verifiers (DeepSeek-V4.1-flash, xhigh), both told the other existed:**

- **Lens A (correctness): VERDICT PASS.** Reproduced every brief claim (endpoint default/`null`, the hint, the
  typed budget stop + resume with 18→28 store lines, `--mirror-interval`, torn state via `Run.status()` and the
  CLI, `file://` vs `memory://` routing). Three minor findings: published files lost the `pipe_file` mode
  (0600 vs 0644); publish temps (`.<name>.<pid>.<rand>`) were mirrored/restored; the `step_budget_s` docstring
  overclaimed that any step over budget stops (CPU-only steps make no request).
- **Lens B (regressions/hygiene): VERDICT PASS.** Full suite 3279 passed at `-n 4`; contract/docs 287 passed;
  the unpublished test package 570 passed; ruff/basedpyright/mkdocs clean; five mutation checks (endpoint
  default, `read_state`, `_Target.write`, the budget wrapper, the seam matrix) each went red as expected; no
  scope creep, no second home, no attribution lines, clean tree. Three minor findings: the "never renames" doc
  claim is false for local targets; the mode regression; the judging-pass seam had no test of its own.

**Fixes (commit `dabb77ff`), each with a failing test first:** four new tests went red before the fix
(`test_a_published_file_keeps_the_mode_a_normal_write_gives_it`,
`test_a_published_file_keeps_an_existing_targets_mode`,
`test_the_publish_temp_ends_in_tmp_so_a_mirror_walk_skips_it`,
`test_a_stale_publish_temp_is_not_restored`); the judging-seam test was shown red by mutating the seam
(`if False and budget is not None` → `DID NOT RAISE StepBudgetExceededError`) and green after restoring it.
COMMON limits round 2 to a blocker or a major; round 1 found only minors, all fixed and re-tested, so no
second verifier round was run.

## Checks

Last commands and their result lines (all in the lane worktree, `uv run --no-sync`):

- `timeout 1800 heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider -o faulthandler_timeout=240`
  → `3284 passed, 93 skipped`.
- `timeout 900 uv run --no-sync pytest tests/contract tests/docs -q -p no:cacheprovider`
  → `287 passed, 52 skipped`.
- `timeout 1200 heavy uv run --no-sync pytest rcp-ndcg-test/tests -q -n 4 -p no:cacheprovider`
  → `570 passed, 225 skipped`.
- `uv run --no-sync ruff format --check .` → `539 files already formatted`; `uv run --no-sync ruff check .` →
  `All checks passed!`; `uv run --no-sync basedpyright` → `0 errors, 0 warnings, 0 notes`.
- `uv run --no-sync mkdocs build --strict -d <scratch>/site` → built.
- `bin/gate lane/sync-hardening` on `dabb77ff` → `GATE: PASS`: ruff/basedpyright 0; pytest
  `3284 passed, 93 skipped`; contract-docs 287 passed; mkdocs; test-pkg 570 passed; recipes
  (34 baseline failures, 0 fixed, none outside the baseline); vllm-pkg and vllm-models; `run_all`
  1022/987/35/0, 67/67, 82/82; public-names clean; tree clean.
- The first gate attempt (at `5ad76462`) failed one unrelated test
  (`tests/data/test_io_contract.py::TestReaderTable::test_every_format_is_listed_under_its_uri_scheme`,
  `available_writers() == ['beir','jsonl','mteb']`): the shared gate venv's editable `rcp-ndcg` metadata was
  stale (its `dist-info` carried no readers/writers, or an older branch's set). The second attempt hit a
  transient shared-venv sync failure (`failed to remove .../torch/__pycache__: Directory not empty`). The lane
  full suite passed at `-n 4` and `-n 8` throughout; the third attempt, after the shared env re-synced,
  passed. No code change was made for it.

## Open questions

- The step budget is **cooperative**: it stops at the next request seam, so a request already in flight
  finishes (bounded by `timeout_s`/`wait_on_outage_s`), and a CPU-only step (`calibrate`, `evaluate`, a local
  BM25 retrieve) is not interrupted. The field docstring, the module docstring and the run docs state this;
  a preemptive mechanism (thread + timeout, or a deadline inside the estimators) is left for a future lane.
- `storage.publish`'s mode change affects every published file (the store's `identity.json`, prompts, the
  media cache), not only mirror files. It restores the mode a plain write gives a file; revisit if a stricter
  mode is wanted for one of those.
- A job that starts an engine still applies its engine's `outage_timeout_s` (900 s default) through the
  `RCP_NDCG_ENGINES` overlay, which wins over the new 1800 s role-config default by design.

## CHANGELOG entry

Under `## Unreleased`:

`### Public surface`

- **`Endpoint.wait_on_outage_s` defaults to 1800 s, not `None`** (review O1): every role config's outage wait
  is finite by default -- an engine restart plus a large model's load -- and a request against an endpoint whose
  replicas all stay down fails with `BackendUnavailableError` (its hint names the field) instead of parking
  forever. `wait_on_outage_s: null` stays the explicit "wait indefinitely" choice, documented as such.
- **`RunConfig.step_budget_s`** (new, default `None`): a per-step wall-clock budget in seconds. The shared
  transport checks it before each request and after every park, and the judging pass before each phase's
  windows; a step over budget stops at the next seam with the new `rcp_ndcg.errors.StepBudgetExceededError`
  (exit code 9, `INTERRUPTED`), the store keeps every judgement it wrote, and `run resume` continues from
  there. `None` leaves the steps unbudgeted.
- **`rcp_ndcg.errors.StepBudgetExceededError`** is the typed error of an exceeded `step_budget_s` (an
  `Interrupted` subclass: the state on disk is consistent and resumable).
- **`rcp-ndcg judge tournament|rubric` takes `--mirror-interval <seconds>`** (default 60, the run config's
  `mirror_interval_s`), so the standalone judging pass's mirror flushes at the interval the run config would
  use.

`### Fixed`

- **A torn `.mirror.json` no longer crashes `run status`** (review S1): the mirror's state file is published
  atomically (temp file + rename, the storage helper), and an unparseable state file reads as "never ran" with
  a warning, as the judgement store treats a torn identity. A reader racing a flush used to raise out of
  `Run.state`.
- **A local or shared mirror publishes whole files atomically** (review S2): `_Target.write` routes local
  targets through `storage.publish_bytes` (temp file + rename), so a concurrent `restore()` on another host can
  no longer read a partial `manifest.json`/`identity.json`; remote object stores still write each object whole
  with `pipe_file`. `storage.publish` keeps the mode a plain write would give the file (an existing target's
  mode, else `0666 & ~umask`), so a shared reader keeps its access, and names its temp `*.tmp`, which the
  mirror's walk and `restore()` skip: a SIGKILL mid-publish leaves nothing the mirror uploads or restores.

`### Changed`

- **The mirror page states the sync guarantee** (review S3/S4/S6): durable is the last uploaded part; a hard
  kill loses at most one interval, re-asked on resume and never duplicated (`record_id`); one live writer per
  store, a diverged writer's flush refuses with `DataError` and the run continues unmirrored (`run status`
  shows it); parts and superseded files are never garbage-collected.

## Public surface changes

- `rcp_ndcg.inference.endpoint.Endpoint.wait_on_outage_s`: default `None` → `1800.0` (schema defaults in
  `schemas/index.v1.json`, `schemas/judge-config.v1.json`, `schemas/run-config.v1.json`).
- `rcp_ndcg.runs.config.RunConfig.step_budget_s`: new field (`float | None`, default `None`, `exclusiveMinimum: 0`).
- `rcp_ndcg.errors.StepBudgetExceededError`: new public error class, exit code 9 `INTERRUPTED`.
- `rcp-ndcg judge tournament|rubric --mirror-interval SECONDS`: new flag (default 60.0).
- Snapshots regenerated: `tests/contract/snapshots/{cli,exit_codes,python_api}.json`; schemas regenerated:
  `schemas/run-config.v1.json` (plus the `wait_on_outage_s` default/description in `index.v1.json` and
  `judge-config.v1.json`).

## Files outside scope

- `rcp-ndcg/src/rcp_ndcg/runs/pipeline.py`, `runs/config.py`, `errors.py`: the run-config field, the typed
  error and the per-step wrapper (the brief's O1b has no other home).
- `rcp-ndcg/src/rcp_ndcg/support/step_budget.py` (new): the context-local budget the request seams read.
- `rcp-ndcg/src/rcp_ndcg/inference/transport.py`, `judging/judging.py`: the two budget seams (O1b).
- `rcp-ndcg/src/rcp_ndcg/support/serve.py`: one docstring correction (`EngineURLs.wait_on_outage_s`'s `None`
  leaves the role config's own wait; it never meant wait-indefinitely).
- `rcp-ndcg/src/rcp_ndcg/storage/core.py`: `publish`'s mode preservation and `.tmp` temp suffix (verifier
  finding; the brief named this file for `publish_bytes`).

## Docs updated

- `docs/concepts/runs.md`: the `step_budget_s` paragraph, the `--mirror-interval` mention, the durability
  guarantee bullets, the local-publish/rename wording.
- `docs/concepts/inference.md`: the `wait_on_outage_s` table row, the status-map row and the outages paragraph
  (default 1800, `null`, the step-budget seam).
- `docs/concepts/judges.md`: the `wait_on_outage_s` row and the outage bullet.
- `docs/reference/cli.md`: `--mirror-interval` and the exit-code 9 row.
- Greps run: `git grep -n -i "wait_on_outage_s" -- docs README.md skills examples rcp-ndcg/src`;
  `git grep -n "pipe_file" -- docs README.md skills`; `git grep -n "waits indefinitely\|wait indefinitely" -- docs rcp-ndcg/src`;
  `git grep -n "mirror_interval\|--mirror" -- docs`; `git grep -n "step_budget" -- docs`.

## For the next lanes

- The cooperative step budget is the pattern to reuse: `rcp_ndcg.support.step_budget` (`StepBudget`,
  `step_budgeted`, `current_step_budget`) is the product seam; the unpublished test package's
  `rcp_ndcg_test.stepwatch` is a separate harness-layer mechanism (request-in-flight tracking, a different
  error class). If both land, keep the product's version authoritative for run steps.
- `storage.publish`'s mode/`.tmp` contract is now relied on by the mirror's walk and `restore()`; a new
  publisher must keep the `*.tmp` suffix.
- The gate needed a third attempt only because the shared gate worktrees' venvs carry editable entry-point
  metadata
  from whichever branch last synced; re-run the gate if it fails an entry-point-table test on a tree that did
  not touch `rcp_ndcg.data.io`.
