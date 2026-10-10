# Lane `run-integrity`: runs talk to the right engines, and their records stay true

Runner review B2-B4, B7-B9, C3, D · base `d630e4a6` · final tree `1200450f` (after merging `rfc-0001` `26da5852`;
`45b66e1b` was merged first and re-merged when `rfc-0001` moved) · `bin/gate lane/run-integrity` **PASS**.

## Status

DONE. Every brief item is implemented, tested (failing test first) and documented; three adversarial verifier
rounds (round 1: two lenses, round 2 and 3: one confirmation verifier each) ended in PASS; the gate is green on
the merged tree.

## Commits

| Commit | Subject |
|---|---|
| `d446bc38` | fix(runners): the phase overlay owns RCP_NDCG_ENGINES and its engine port |
| `bb9e3860` | fix(runs): refuse a served judge's foreign base_url |
| `a98742b9` | fix(runs): the run's own state survives a mirror, a torn record and a live job |
| `e38f7dbe` | feat(runners): a runner contract kit, no duplicate entry-point shadowing, and a real local cancel |
| `fb6ecf9c` | docs: document the run-integrity fixes and the runner contract kit |
| `33fc95b5` | fix(runs): a served judge path that is not there is left to the run |
| `8e8503c3` | test: the T4 supervision stubs and golden follow the phase boundary |
| `2233118e` | fix(runs): close the round-1 verifier findings |
| `2a457bc4` | fix(runs): close the round-2 verifier findings |
| `8711977d` | fix(runs): a default Resources() instance is no job-field declaration |
| `02f1ae4c` | Merge branch 'rfc-0001' (tip `45b66e1b`) into lane/run-integrity |
| `1200450f` | Merge branch 'rfc-0001' (tip `26da5852`) into lane/run-integrity |
| (report) | The lane report: run integrity, the verifier rounds and the gate |

## What changed

1. **B2 — the phase overlay owns `RCP_NDCG_ENGINES`.** A job `env` entry of that name (a built-in runner's
   `runner.options.env`, a plugin's free-form `env`, or `--set`) is refused at config time (`runners/base.py`,
   `runs/config.py` `PluginRunnerConfig`), and `runners/script.py` `merge_phase_env` makes the phase's value win
   for the reserved name even if one reaches rendering.
2. **B3 — a phase never reaches the previous phase's engine.** `runners/script.py` `supervise` now refuses a
   phase whose engine port already answers (a foreign listener, a previous phase's engine), and after a
   coordinator exits 0 waits (`rcp_ndcg_wait_gone`) until each started engine's port stops answering before the
   next phase starts. Remote (StatefulSet) engines and engine-free phases are untouched.
3. **B4 — a served judge's `base_url`.** A judge may name no URL (a recipe's config does; the overlay sets it);
   one that names a URL must be the engine's own loopback URL (`http://127.0.0.1:<serve.judge.port>/v1`) or it is
   refused at config time, instead of the overlay silently replacing a foreign endpoint (`runs/config.py`).
4. **B7 — the run's own state survives.** `logs/jobs.json` is host-local and is never uploaded or restored
   (`runs/mirror.py` `SKIPPED_FILES`); it is published atomically and read with a typed `DataError` naming the
   file for a torn or wrong-shaped record (`runs/execution.py` `_write_record`, `runs/run.py` `jobs()`); a
   submission in flight leaves a `submitting` flag that blocks resubmission while a failed submission stays
   resubmittable; a multi-phase job never reads `done=true` mid-run (a live job, or a `partial` manifest with a
   job the runner cannot resolve, keeps `done=false`); a damaged `manifest.json` is replaced from the mirror
   instead of crashing `_behind`; any mirror client error makes `run status` fall back with a note.
5. **B8 — handle integrity.** The live-job guard refuses a null handle (unless the record names a failed
   submission), a `submitting` record, an `UNKNOWN` handle, and a runner that cannot report a job; `run cancel`
   says a handle-less record may be live. Kubernetes `submit` refuses an existing Job by name (apply is a no-op)
   and names the way out (`kubectl delete job …` or `ttl_seconds_after_finished`).
6. **B9 — the mirror is run-scoped.** `Mirror.restore` refuses a mirror whose manifest names another run (the run
   id is the local manifest's — salvaged from the damaged bytes when it does not parse — or the directory's
   name), and `run status` ignores such a manifest with a note instead of adopting the other run's id/metrics.
7. **C3 / runner contract.** A config's job `resources`/`image`/`env` are refused (not dropped) when the run is
   handed to a runner the config does not name (defaults excluded; a malformed `resources` counts too);
   `get_runner` refuses a name provided by more than one installed distribution (no plugin can shadow a
   built-in); `rcp_ndcg.testing.runner_conformance` is the seam's contract a plugin's tests call.
8. **`LocalRunner.cancel`.** SIGTERM the job's process group, SIGKILL what is left after the grace period, check
   the group is gone (typed `RunnerError` if not); a torn session file or a pid of 0/1 is never signalled, and
   the session file is published atomically.
9. **Status edges.** `run status` names an `unknown` job (an unmapped SLURM state, a deleted Job, a missing
   `sacct`) in its note, falls back with a note on any runner error (typed or not) and on any mirror client
   error, and the text output shows the note and the mirror state (with `safe_url`).
10. **Docs.** `docs/concepts/runs.md` (job fields, the served judge, the phase boundary, the Kubernetes
    resubmission refusal, the run-scoped mirror, `jobs.json`, status notes), `docs/reference/cli.md`, the skill,
    the CHANGELOG, the `python_api` snapshot and the two `RunState.done` schemas.

## Verification

**Round 1 — two fresh verifiers, DeepSeek-V4.1-flash `:xhigh`, in parallel.**

- **Lens A (correctness): FAIL.** F1 major: a `partial` manifest with an `UNKNOWN` job still read
  `done=true`. F2 major: `run status` adopted a foreign run's newer mirror manifest although `restore` refuses
  it. F3 minor: a non-`RcpNdcgError` from a runner (a plugin `OSError`, a damaged local `.session` file) aborted
  `run status` with INTERNAL. F4 minor: valid JSON of the wrong shape in `logs/jobs.json` reached callers as
  `KeyError`/`AttributeError`. F5 minor: a damaged local manifest in a differently named directory refused its
  own recovery.
- **Lens B (regressions/hygiene): PASS**, with minors: the broadened mirror-fallback catch had no regression
  test; the plugin `RCP_NDCG_ENGINES` refusal happened at render, not config load; `_set_job_fields` hardcoded
  `JOB_OPTIONS`; the `readiness_functions` docstring omitted `rcp_ndcg_wait_gone`; `__all__` was inconsistent;
  the skill lost a SLURM clause and had an over-long line; commit hygiene (a `docs:` commit also carried
  ruff-format edits).
- **Fixed:** F1–F5 and the lens-B minors in `2233118e` (and the skill/`__all__`/docstring items), each with a
  failing test first; the T4 stubs/golden were updated in `8e8503c3` (the marker now holds the engine pid, so a
  stopped engine's marker no longer answers; the golden carries the preflight and boundary lines).

**Round 2 — one fresh confirmation verifier (lens A+B): PASS.** F1–F5 all confirmed fixed (its own repros plus a
mutation of the F2 guard making its test red). Four new minors: N1 `run cancel`/`_refuse_live_jobs` still
propagated an untyped runner error as INTERNAL; N2 a damaged session file with a numeric prefix was trusted as a
pgid (`"0"` killed the cancelling process's own group, `"1"` pgid 1); N3 `RunState.done`'s schema description was
broader than the code; N4 a malformed `resources` was silently dropped by the C3 guard.
**Fixed:** N1–N4 in `2a457bc4`, each with a test (the N2 test instruments `os.killpg` and asserts zero calls).

**Round 3 — one fresh confirmation verifier (lens A+B): PASS.** N1–N4 confirmed (N1/N2 reproduced from scratch
and mutation-checked; N3 by a six-case `done` matrix against the regenerated schemas; N4 for its stated case).
Two new minors: a default `Resources()` instance was counted as a declaration, and the two new failure-mode
docstrings omitted the runner-cannot-report case. **Fixed:** `8711977d`, with a test for the default instance.

No blocker or major remained open after round 2; the round-3 minors were fixed and the final tree re-gated.

## Checks

Last commands and result lines (all on the final merged tree, `1200450f` unless noted):

- `bin/gate lane/run-integrity` → `GATE: PASS` (pytest 3713 passed, 102 skipped; contract-docs 301 passed,
  55 skipped; test-pkg 930 passed, 222 skipped; recipes, vllm-pkg, vllm-models, run_all, public-names, clean all
  exit 0; mkdocs strict clean).
- `uv run --no-sync ruff format --check .` → 588 files already formatted; `uv run --no-sync ruff check .` → all
  checks passed; `uv run --no-sync basedpyright` → 0 errors, 0 warnings, 0 notes.
- `heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider` → 3713 passed, 102 skipped.
- `uv run --no-sync pytest rcp-ndcg-test/tests -q -p no:cacheprovider` → 930 passed, 222 skipped.
- `uv run --no-sync pytest tests/contract tests/docs -q -p no:cacheprovider` → 301 passed, 55 skipped.
- `uv run --no-sync mkdocs build --strict` (the gate's mkdocs step) → documentation built.
- Failing-first: the new tests failed before their fixes (e.g. the round-1/2/3 test batches each ran red on the
  unfixed tree; the B2 import error, the B4 `DID NOT RAISE ConfigError`, the B3 linger timeout, the B7 mirror
  restore clobbering the handle, the B8 null handle not refused, the cancel group surviving); verifier B and the
  round-2/3 verifiers each mutated fixes and showed the matching test red (five mutations in round 1: config env
  refusal, mirror upload skip, `done=false` between phases, Kubernetes existing-Job refusal, local SIGKILL
  escalation; N1/N2 and the F2 guard in rounds 2/3).

## Open questions

1. **`logs/jobs.json` is host-local and never restored**, so a resubmission from a different host into a fresh
   directory has no record and the live-job guard cannot see another host's live handle. Mitigated by the
   Kubernetes same-namespace Job refusal and SLURM's always-new-job semantics; documented in
   `docs/concepts/runs.md`.
2. **Exit-code taxonomy in `cancel`/resubmission.** The new catches re-type a built-in runner's typed
   `RunnerError` (exit 6) as `MISSING_INPUT` (4) / `CONFIG` (3). That is the safe direction (the job may be
   live) and nothing pinned the old code, but a caller branching on exit 6 should be told.
3. **A `partial` manifest with no `logs/jobs.json`** (a host that only restored from the mirror) reads
   `done=true` with no note: it is indistinguishable from a manual `--only` run. A future `RunState` could carry
   a separate "a job may be alive elsewhere" bit.
4. **The real-`srun` question remains** (the review's serve F2 open question): the boundary wait is
   belt-and-braces on a cluster; the port wait and preflight are the offline-verifiable part.
5. **Out of this lane's scope, still open from the review:** K2 (a finished Kubernetes Job's engine
   StatefulSets stay without a TTL; the new refusal names it), the B1 wave-summary upload, B5 verdict integrity,
   B10-B12 injection/secrets, C1/C2 GPU partitioning. `RunManifest.load` reads the file outside its `try`
   (a non-UTF-8 manifest raises an untyped `UnicodeDecodeError`) — pre-existing, outside the lane diff.

## CHANGELOG entry

Under `## Unreleased`, grouped as the repository requires:

**### Public surface**

- **`rcp_ndcg.testing.runner_conformance(runner, job=...)`** is the `rcp_ndcg.runners` seam's contract as one
  check a plugin runner's own tests call: the four `JobRunner` methods, the answers' shapes (`submit` returns one
  handle per job, `status` a `JobStatus`, `logs` a string), and the optional `render`, `renders_phases` and
  `run_root` members when declared.
- **`run status`'s `done` means the run is done**, not only that its status is terminal: a job still running
  between phases, or one the runner cannot ask about (an unmapped state, a missing accounting CLI), keeps `done`
  false while its run's status is `partial` (`RunState.done`'s schema description).

**### Fixed**

- **The phase overlay owns `RCP_NDCG_ENGINES`**: a job env entry of that name (through `runner.options.env`)
  silently defeated every phase's engine URLs -- the worker re-exported the job's value after `supervise` exported
  the phase's -- so the config now refuses the name and `worker_script` lets the phase's value win for it.
- **A phase never reaches the previous phase's engine**: the phase boundary waited only for the `srun` client,
  not the engine, so two phases on one port could hand phase 2's coordinator phase 1's engine (and its
  judgements). The boundary now waits until the engine's port stops answering (up to the stop grace), and a phase
  refuses a port that already answers before it starts its engine.
- **A served judge's `base_url` is refused when it is not the job's engine**: the runtime overlay replaced a
  foreign value silently. A served judge may name no `base_url` (a recipe's config does); one that names a URL
  must be the engine's own loopback URL, and anything else is refused at config time.
- **A mirror restore can no longer destroy the submitting host's job handle**: `logs/jobs.json` is host-local and
  is never uploaded or restored; the record is published atomically and read with a typed error naming the file
  (a valid-but-wrong-shaped record included); a submission that never recorded its handle leaves a `submitting`
  flag that blocks resubmission, while a submission that failed before a handle is still resubmittable; and
  `run cancel` says a handle-less record may be live instead of claiming it was never submitted.
- **Kubernetes resubmission is never a silent no-op**: `kubectl apply` on an existing Job restarts nothing, so
  `submit` now refuses an existing Job by name and says how to remove it (or to set
  `ttl_seconds_after_finished`).
- **The mirror is run-scoped**: `restore` refuses a mirror whose `manifest.json` names another run, and
  `run status` ignores such a manifest with a note instead of adopting the other run's id and metrics; a damaged
  local manifest is replaced by the mirror's instead of crashing its own recovery path (its `run_id` is salvaged
  from the damaged bytes when it survives); and any mirror client error (a GCS 403 or refresh failure included)
  makes `run status` fall back to the local state with a note instead of aborting.
- **A multi-phase job never reads `done=true` mid-run**: a live job keeps `done=false` whatever the manifest
  says, and while the manifest is `partial` (a phase boundary) so does a job the runner cannot resolve (an
  unmapped state, a missing accounting CLI).
- **Status edges are reported, not silent**: a job the runner reports `unknown` (an unmapped SLURM state, a
  deleted Job, a missing `sacct`) is named in `run status`'s note, an untyped runner error (a damaged local
  session file included) falls back the same way, `run cancel` and a resubmission refuse a runner that cannot
  report a job with a typed error instead of INTERNAL, and the text output shows the note and the mirror state.
- **`LocalRunner.cancel` really stops the job**: it SIGTERMs the job's process group, SIGKILLs what is left
  after the grace period and checks the group is gone, instead of recording the run `cancelled` while a
  SIGTERM-ignoring coordinator kept running; a session file that is torn or names pid 0/1 is never signalled
  (the session file is published atomically too).

**### Changed**

- **`get_runner` refuses a name provided by more than one installed distribution** instead of silently keeping
  the last entry point, so a plugin can no longer shadow `local`, `slurm` or `kubernetes` (and receive the
  built-in's typed options).
- **A config's job `resources`, `image` and `env` are refused, not dropped, when another runner is in use**: the
  fields describe the job and are read when the config names the runner in use; handing the run to another
  runner with them set now fails with a message naming them (a lost `time_limit_s` or `env` was silent).

## Public surface changes

- `rcp_ndcg.testing.__all__` gains `runner_conformance` (`tests/contract/snapshots/python_api.json` regenerated).
- `RunState.done`'s description (schemas `run-start.v1.json`, `run-status.v1.json`) now reads "Whether the run is
  done: its status is terminal and no job of it is still running (or, between a job's phases, unresolved by its
  runner)." No CLI command, flag, exit code or other schema changed.

## Files outside scope

- `rcp-ndcg/src/rcp_ndcg/runs/run.py` — the typed `Run.jobs()` read (B7; the reader lives there).
- `rcp-ndcg/src/rcp_ndcg/cli/run.py` — `run status` text mode (brief item 9).
- `rcp-ndcg-test/tests/_emulator_server.py`, `test_e2e_driver.py`, `test_supervision_replay.py` and
  `tests/fixtures/golden/job-text-four-phases.sh` — the shared `supervise` change required the T4 stubs' marker
  to reflect a stopped engine and the golden to carry the new lines; all minimal and listed in commit `8e8503c3`.
- `skills/rcp-ndcg/SKILL.md` (docs), `docs/reference/cli.md`, `CHANGELOG.md`, `schemas/` and
  `tests/contract/snapshots/` (generated).

## For the next lanes

- The runner-security lane's redaction work should keep the new text-mode mirror line safe (it uses `safe_url`)
  and can now rely on `logs/jobs.json` never being mirrored.
- A real-cluster check of the `srun` client/remote-task semantics (the review's serve F2 open question) is still
  worth one node; the offline behaviour is pinned.
- Kubernetes cleanup (review K2): a finished Job's engine StatefulSets stay without a TTL; the new refusal names
  the deletion. A `run prune` for ended Jobs would close it.
- The `partial`-with-no-jobs.json case (Open question 3) needs a product decision on where a "job may be alive
  elsewhere" bit would live.
