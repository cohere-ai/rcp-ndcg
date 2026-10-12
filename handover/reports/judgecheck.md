# Lane judgecheck — the release checklist's judge item

**Status: DONE (the job, its tests and the submit wiring; the node run is handed to the operator).**

The checklist's judge bullet asks for, per judge recipe: T0 smoke + `rcp-ndcg judge check` (both thinking
modes, the NVFP4 backends, the media token counts). The judge waves already ran T0 smoke (10/10 `verified`
on the current RC); `judge check` needs a live endpoint, so it runs inside the pod — the `casefill` pattern.

## 1. Status — per-item disposition

- **`jobs/judgecheck.sh`** — built. Serves each judge recipe one at a time (the big judges need tensor
  parallelism, so they never share the node), waits for `GET /v1/models` (`RCP_JUDGECHECK_STARTUP_S`, 3600 s;
  a dead engine's process group fails fast instead of waiting the timeout), runs `rcp-ndcg judge check`
  against `http://127.0.0.1:<port>/v1` for each documented thinking mode, runs the engine media probe, stops
  the engine's process group, records the verdicts + the engine version + the NVFP4/Marlin backend lines,
  continues past a failed judge, and uploads the JSONs and the report.
- **The thinking modes are the recipes' documented ones, never a guessed flag.** Only the Gemma 4 recipe
  documents a toggle (`recipes/gemma-4-12b/family.yaml`: thinking off by default; on is a declared client
  choice, `chat_template_kwargs: {enable_thinking: true}`), so the four Gemma judges run **two** modes and the
  other six run their **single documented mode** (the shipped client block, no `--set` at all). The `on` body
  is the recipe's own `extra_body` **merged** with the switch: `--set judge.extra_body` replaces the whole
  block, so the declared sampling travels with it (verified against the shipped recipes, not hand-copied).
- **The media token counts are covered by a real media probe** (`judgecheck_probe.py`): one prepared probe
  image beside its no-media baseline, the engine's two `usage.prompt_tokens` reports' delta against the
  client's counted media tokens, through the product's own `RoleClient.check_engine_media` — the same probe a
  judging pass runs at startup. The probe's policy is the processor family's own stock budget
  (`PROCESSORS`: `qwen3_vl` 65 536..16 777 216 px; `gemma4` 280 soft tokens), so a stock engine keeps the
  prepared size and the counted tokens describe what it sees. The report records the outcome
  (`ok`/`not_checked`/`mismatch`/`refused`) and, on a pass, both numbers. A text-only judge (`gpt-oss-120b`)
  is recorded `checked: false, reason: text-only` — never silently passed.
- **The check's invocation is the product's own form.** `--judge <recipe>` + `--judge-url` is refused by the
  CLI ("pass --judge or --judge-url, not both"); the working form is `--judge <recipe-id> --judge-model
  <served-name> --set judge.base_url=http://127.0.0.1:<N>/v1 --json` (the CLI's own hint).
- **`submit.sh --script judgecheck`** — wired: the validation list, the mounted `judgecheck.sh` +
  `judgecheck_probe.py` beside `bootstrap.sh`, and the worker command. `KJOBS=echo` renders the whole plan
  against the current RC (checked below).
- **Tests** — red first (section 3), then green: the submit argv, the catalog pin (the checklist's six first),
  the dry plan, the documented-modes table, the media probe (unit + over the wire against a stub chat
  engine), and the node path end to end with stubs (a dead engine recorded, the next judge still served,
  every process group stopped, the JSONs and the report uploaded).
- **The node run is not run by this lane.** `submit.sh`'s header reserves submission to the operator; the
  exact command is in section 4 and the per-judge coverage plan in section 5.

## 2. Commits

| Commit | What |
|---|---|
| `60f1c068` | The judge-check job: `judgecheck.sh` + `judgecheck_probe.py` |
| `734ceb30` | `submit.sh --script judgecheck`: the mounted job, its argv tests and the docs |
| `b696e847` | The helper's parser description is a literal, not `__doc__` (basedpyright) |

## 3. Red-first evidence (quoted)

The tests were written first and watched fail on the unfixed tree:

```
$ uv run --no-sync pytest rcp-ndcg-test/tests/test_jobs_scripts.py -q -k 'judgecheck or unknown_script'
9 failed, 76 deselected

test_submit_refuses_an_unknown_script
  E  AssertionError: assert 'bootstrap, wave0, e2e, casefill or judgecheck' in
     'submit.sh: --script must be bootstrap, wave0, e2e or casefill, got deploy\n'
test_submit_judgecheck_mounts_the_judgecheck_script
  E  AssertionError: submit.sh: --script must be bootstrap, wave0, e2e or casefill, got judgecheck
test_judgecheck_toggles_are_the_families_own_template_switches
  E  ModuleNotFoundError: No module named 'rcp_ndcg_test.jobs.judgecheck_probe'
test_the_judgecheck_dry_plan_names_the_ten_judges_and_both_modes
test_judgecheck_end_to_end_serves_checks_media_and_uploads
  E  AssertionError: bash: .../jobs/judgecheck.sh: No such file or directory (exit 127)
```

Two more red rounds during the build: the dry plan's media line and the mode-list shape, and the
end-to-end stub's engine-failure branch (the first version could not fail a middle recipe — the
`casefill` lane's lesson repeated).

## 4. The exact node command

Against the current RC (rebuild it first if the recipes moved since it was staged; the job mounts its own
script from the checkout, the client environment comes from the staged wheels):

```bash
export RCP_KJOBS_CONFIG=<the job CLI's config>       # required, no default
export RCP_GCS_AUTH_FILE=<the auth script>           # mounted at /etc/rcp/gcs_auth.sh; named, never read
export RCP_HF_TOKEN_FILE=<the HF token file>         # mounted into the job, read there, never in an argv
rcp-ndcg-test/src/rcp_ndcg_test/jobs/submit.sh --script judgecheck --priority dev-high \
  gs://YOUR-BUCKET/rcp-ndcg-rfc/rc0-<stamp> gs://YOUR-BUCKET/rcp-ndcg-rfc/waves-rc0-<stamp> judgecheck
```

Outputs: `gs://…/waves-rc0-<stamp>/judgecheck/<recipe>/<mode>.json`, `<recipe>/media.json`, and
`judgecheck-report.json`. `KJOBS=echo` renders the plan without submitting (done in this lane against the
current RC: one job `rcp-judgecheck`, `priority_class=dev-high`, `worker.shared_memory=128Gi`, the image
digest resolved, both files mounted).

Budget: the ten judges' engines booted in 319–1107 s each in the judge waves (T0 smoke, same RC), so the
sequential job is ~2 h; `RCP_JUDGECHECK_STARTUP_S` bounds the wait, `RCP_JUDGECHECK_CHECK_S` /
`RCP_JUDGECHECK_MEDIA_S` bound each command (900 s).

## 5. Verification

- `uv run --no-sync pytest rcp-ndcg-test/tests/test_jobs_scripts.py -q` — **85 passed** (10 judgecheck).
- `heavy uv run --no-sync pytest tests/ -q -n 4` — **4145 passed, 155 skipped**.
- `uv run --no-sync pytest tests/contract tests/docs -q` — **413 passed, 107 skipped**.
- `uv run --no-sync pytest rcp-ndcg-test/tests -q -n 4` — **1170 passed, 229 skipped**.
- `ruff format --check . && ruff check .` — clean; `basedpyright` — 0 errors.
- The media probe against a stub chat engine (CPU): an honest engine (delta 66 = the client's counted
  66) reports `ok` with both numbers; a lying one (67) is the product's typed `mismatch`.
- `shellcheck` on `judgecheck.sh` and `submit.sh` — clean.
- `timeout 7200 bin/gate lane/judgecheck` — see the gate result in the lane's final handoff.

## 6. Per-judge coverage plan (the node run)

| # | Recipe | T0 smoke (judge wave, current RC) | Thinking modes the job runs | Media probe | NVFP4 backend note | serve_s |
|---|---|---|---|---|---|---|
| 1 | `qwen3.5-397b-a17b-nvfp4` | verified (serve 1107 s) | single documented (`default`) | yes (`qwen3_vl`) | engine log: NvFp4/Marlin lines | 1107 |
| 2 | `gpt-oss-120b` | verified (559 s) | single documented (`default`) | `checked: false` (text-only) | — | 559 |
| 3 | `qwen3.6-27b-fp8` | verified (404 s) | single documented (`default`) | yes (`qwen3_vl`) | — | 404 |
| 4 | `qwen3.8-27b-fp8` | verified (319 s) | single documented (`default`) | yes (`qwen3_vl`) | — | 319 |
| 5 | `qwen3.8-flash-next-fp8` | verified (849 s) | single documented (`default`) | yes (`qwen3_vl`) | — | 849 |
| 6 | `qwen3.8-flash-next-nvfp4` | verified (942 s) | single documented (`default`) | yes (`qwen3_vl`) | engine log: NvFp4/Marlin lines | 942 |
| 7 | `gemma-4-12b-it` | verified (410 s) | **off + on** (the recipe's declared choice) | yes (`gemma4`, 280 soft tokens) | — | 410 |
| 8 | `gemma-4-26b-a4b-it` | verified (585 s) | **off + on** | yes (`gemma4`) | — | 585 |
| 9 | `gemma-4-26b-a4b-nvfp4` | verified (374 s) | **off + on** | yes (`gemma4`) | engine log: NvFp4/Marlin lines | 374 |
| 10 | `gemma-4-31b-it-nvfp4` | verified (415 s) | **off + on** | yes (`gemma4`) | engine log: NvFp4/Marlin lines | 415 |

The checklist's six (rows 1-6) run first. The judge waves' `status.json` records the gap this job fills
verbatim: `equivalence: skipped — a judge has no reference (decision 15); its conformance is
`rcp-ndcg judge check` and the T4 scenarios`.

## 7. Open questions / residual risks

- **The second thinking mode for the six non-Gemma judges is not run.** Their recipes document no toggle;
  the E2 plan (`handover/specs/judge-catalog.md` §7.6) does document a both-modes check
  (`chat_template_kwargs: {enable_thinking: true}`), and the template facts are recorded in
  `handover/reports/08bd-judge-recipes.md` (the Qwen3.8 and gpt-oss templates read `reasoning_effort`; the
  Qwen3.5/3.6 ones do not). Adding a mode is a one-line data change in `judgecheck_probe.judge_modes`, but
  it needs the owner's call, because a toggle that the recipe does not document runs another instrument.
- **The media probe declares the stock budget, not the run's.** A judging run may declare its own
  `preprocessing.image` (e.g. the ViDoRe scenario's 65 536..1 048 576 px); the job's probe uses the
  processor's own stock budget so a stock engine keeps the prepared size. Both are honest; they are not the
  same number.
- **Wall time.** ~2 h of engine boots plus the checks; the six priority judges are first, so a time-boxed
  run still covers the checklist's subset.

## Docs updated

- `docs/how-to/release-candidates.md` — the `--script` list gains `judgecheck`; a new section "The
  judge-check job (`--script judgecheck`)" (the command, the per-judge loop, the documented modes, the media
  probe, the outputs, the knobs), and the judge item named beside the casefill job.
- `CHANGELOG.md` — one bullet under `## 0.0.1` / `### Changed` (the submittable judge job).
- `grep -n 'casefill\|--script' docs/how-to/release-candidates.md` reviewed for every hit; `submit.sh`'s own
  header and usage updated; `tests/docs` green (396 passed).

## CHANGELOG entry

```
- **The release checklist's judge item is a submittable job** (the unpublished `rcp-ndcg-test`): a new
  `submit.sh --script judgecheck` job serves each of the ten shipped judge recipes one at a time
  (`rcp-ndcg-vllm serve <id> --port <N>`), waits for `GET /v1/models`, runs `rcp-ndcg judge check --judge
  <id> --judge-model <id> --set judge.base_url=http://127.0.0.1:<N>/v1 --json` in each recipe's documented
  thinking mode (the Gemma 4 judges' `chat_template_kwargs.enable_thinking` choice, merged over the
  recipe's own `extra_body` -- `--set judge.extra_body` replaces the whole block; the other recipes document
  none, so their single documented mode is the shipped client block, never a guessed flag), runs the engine
  media probe (`judgecheck_probe.py`: one prepared probe image beside its no-media baseline under the
  processor family's own stock budget, the engine's media delta against the client's counted tokens, the
  product's `RoleClient.check_engine_media`), stops each engine's process group, records the verdicts, the
  engine version and the NVFP4/Marlin backend lines, continues past a failed judge, and uploads
  `<OUT>/<recipe>/<mode>.json` plus `judgecheck-report.json`.
```

## Public surface changes

None. No `__all__`, CLI command or flag, exit code or schema moved: `judgecheck.sh` and `judgecheck_probe.py`
are node job tooling in the unpublished `rcp-ndcg-test` package, and `submit.sh`'s `--script` gains one value
(its own argv validation, documented in its header and the release-candidates page).

## Files outside scope

None.

## For the next lanes

- **The operator's node run** (section 4) fills the coverage table's verdict columns; the job's
  `judgecheck-report.json` is the artifact the checklist's tick cites.
- **A second thinking mode for the non-Gemma judges** is a data change in `judgecheck_probe.judge_modes`
  once the owner decides (section 7).
