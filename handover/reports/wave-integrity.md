# Lane wave-integrity: GPU waves can start, and their verdicts mean what they say

Brief: runner review A2-A5, B1, B5, C4 and item 9 (the plugin wheel vs the `rcp-fp/4` fingerprint).
Base: `rfc-0001` after harness-fix landed (`afecce00`); final tree: the current `rfc-0001` (`08f6166b`)
merged in.

## 1. Status

DONE. Gate `bin/gate lane/wave-integrity` -> **GATE: PASS** on the merged tip `a18c0780`.

## 2. Commits

| Commit | Subject |
|---|---|
| `19490c45` | Merge lane fp-v4 into lane/wave-integrity: `rcp-fp/4` (item 9's premise; the lane was not an ancestor of the given base) |
| `82d7ae68` | Merge lane fp-v4, second pass: the four goldens fp-v4 predates regenerate; the merged patches test clears the wave-closing flag |
| `b0b16380` | The client environment carries the harness (A2/A3, security F2) |
| `3a023f79` | Every wave upload is verified and retried, recorded in the recipe's status row; the summary is written before the last upload (B1) |
| `45b63cfb` | The wave's verdicts: after_restart required, pod engine version, duplicate ids refused, all-skipped SKIPPED (B5) |
| `bef580d8` | The staged plugin wheel's modules are hashed against the fingerprint and the recording refused on a mismatch (item 9) |
| `eb707544` | The by-hand build docs match the release build; the submitted wave's gate set is documented; the pipeline concurrency is pinned (A4/A5, C4) |
| `d77ee3f5` | ruff format |
| `9dccb87a` | Verifier round 1 fixes |
| `2cd8a046` | Verifier round 2 fixes |
| `bf18853f` | Verifier round 3 fixes |
| `a18c0780` | Merge rfc-0001 (`08f6166b`: retrieval-fixes, the judge recipes, l08-judges, the media-identity test fix) |

## 3. What changed (per brief item)

1. **A2 (E2 blocker)** — `rc_build.sh` builds the unpublished `rcp-ndcg-test` wheel by name into
   `<stage>/harness/`; `bootstrap.sh` installs it into the client environment
   (`--with rcp-ndcg-test==<version>`, `--find-links <stage>/harness`) and the client probe imports
   `rcp_ndcg_test` and checks its version against the manifest. A node-shaped bootstrap test runs the
   probe through the client mechanism and asserts the spec, the find-links and the imported version.
2. **A3** — `rc_build.sh` builds exactly the three published distributions (never `--all-packages`),
   keeps `dist/` at the six release files, and builds the harness wheel into its own directory; a test
   pins both build lists.
3. **B1** — every upload is verified against the destination (existence and size through the product's
   storage), retried up to three times with backoff, and recorded in the recipe's `status.json` row;
   `wave.json`/`WAVE.md` are written before the last upload so they reach the URI; a wave with a failed
   upload records `verdict: "failed"`, says `Verdict: **FAILED**` / `wave: FAILED`, and exits non-zero.
   A hung CLI is bounded by a declared 300 s timeout.
4. **B5** — `verify_corpus` refuses a corpus whose pass list lacks the `after_restart` sending; the
   corpus key and the recorded engine version come from the running pod (the engine's `/version`, then
   the engine environment's own `vllm` through `RCP_ENGINE_PYTHON`), never the declared image, and the
   corpus/quality steps fail with a named reason when neither answers; a duplicate recipe id in a wave
   list is refused before any engine starts; an all-skipped `--changed-since` wave reports
   `verdict: "skipped"`, `Verdict: **SKIPPED**`, `wave: SKIPPED` and exits non-zero; a digest-pinned
   recipe records the pod's version and keeps its measured status expectations.
5. **A5** — the docs state exactly which gates the submitted wave runs (T0 smoke, T2 equivalence and the
   recorder; T1's corpus, T3 and the controls are operator-run with the flags and packages they need),
   in `release-candidates.md` and `validate-a-recipe.md`.
6. **A4** — the by-hand build recipe in `docs/concepts/runs.md` matches the release build (three
   published packages by name plus the harness wheel in its own directory); the four stale paths the
   review found in `release-candidates.md` were already fixed on the base and re-verified.
7. **bootstrap injection (security F2)** — the downloaded manifest's version is validated as a plain
   PEP 440-shaped string before use, and the client wrapper writes every argv word through `printf %q`,
   so a manifest field can never become shell syntax.
8. **C4** — the post-serve pipeline already runs in per-recipe worker threads on the base; a regression
   test now pins that two recipes' smoke steps overlap (the pre-harness-fix runner serialized them).
9. **plugin wheel vs fingerprint** — the wave hashes the modules inside the staged wheel with the same
   `sha256:` canonicalisation as the fingerprint's `plugin_sha256.<module>` inputs and refuses the
   recording when they differ; the engine installs that same staged wheel (the bare-name pin plus a
   post-install version check), and the step document records which wheel was hashed or why it was not.

## 4. Verification

Round 1, two fresh DeepSeek-V4.1-flash (`:xhigh`) verifiers in parallel (lens A correctness, lens B
regressions/hygiene):
- Lens A: **FAIL** — (F1, major) a failed upload left `verdict: "passed"` in `wave.json`/`WAVE.md`/stdout;
  (F2, minor) `staged_plugin_wheel` could hash a different wheel than pip installed; (F3, minor) a
  corrupt wheel left the step `running`; (F4, minor) the quality step fell back to the declared image.
- Lens B: **FAIL** — (major) the `_pod_engine_version` composition was unpinned (a revert to the
  declared image survived the suite); (major) item 9's check was dormant in the documented flows (the
  T1 command omitted `--plugin-wheel`); (minors) the `validate_version` call was unpinned, the T3 docs
  omitted the harness wheel, the wheel hashing repeated the digest canonicalisation.
- Fixed all in `9dccb87a`, each with a failing test first (the failed-upload verdict, the pin, the
  corrupt wheel, the quality guard, the pod-version composition, the hostile-manifest end-to-end test,
  the shared `sha256_digest`, and the T1/T3 docs).

Round 2, one fresh confirmation verifier: **FAIL** — (minor) the zip-layer handler caught only
`BadZipFile`/`OSError` (an unsupported compression method or an encrypted member still escaped);
(minor) the item-9 pin silently overrode a version-pinned shipped-plugin spec; (major) the upload CLI
had no timeout, so a hung `gcloud` blocked the wave forever. Fixed in `2cd8a046` (every zip-layer
exception folds into the named refusal; only the exact bare name is pinned; a declared 300 s CLI
timeout). Residual (b) (the harness sdist beside the wheel) was shown harmless and (c) (committed
corpora without `collector.passes`) refuted.

Round 3, one fresh confirmation verifier: **FAIL** — (major) a versioned shipped-plugin spec
(`rcp-ndcg-vllm==0.0.2`) reopened item 9: the engine installed 0.0.2 while the wave hashed the staged
0.0.1 and recorded a verified row; (minor) the new docstring claimed a transfer bound for the python
upload fallback that does not exist. Fixed in `bf18853f`: the engine's installed version must equal the
staged wheel's (a mismatch records the plugin in `plugin-failures.txt` with its exact name, so the
recipes that name it fail), with unit tests for the refused and accepted versioned specs; the docstring
now states the pre-existing unbounded fallback. The round cap (three) was reached, so the remaining
residual is in Open questions.

## 5. Checks

Final gate on the merged tip `a18c0780`:

```
ruff-check exit=0 All checks passed!
ruff-format exit=0 589 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 3746 passed, 102 skipped
contract-docs exit=0 301 passed, 55 skipped
mkdocs exit=0 Documentation built in 1.52 seconds
test-pkg exit=0 959 passed, 222 skipped
recipes exit=0 no failure outside the baseline
vllm-pkg exit=0 49 passed
vllm-models exit=0 72 passed, 7 skipped
run_all exit=0 leaderboards 1022 checks, 987 match, 35 known deviations, 0 failed;
                human study 67/67; external LLM judges 82/82
public-names exit=0 clean
clean exit=0
GATE: PASS
```

Earlier: the same gate passed at `d77ee3f5` (before the rfc-0001 merge) on the second run; the first
run saw a one-off SIGSEGV in `test-pkg` at
`test_the_corpus_is_keyed_by_the_engine_version_the_pod_reports` (exit 139, no faulthandler dump) that
did not reproduce in the worktree or in either later gate run.

## 6. Open questions

- The python upload fallback (`_upload_storage` through the product's fsspec layer) has no transfer
  timeout: a stalled fallback transfer still stalls the wave. Pre-existing; bounding it needs a
  product-side storage timeout option (outside this lane).
- The lane merged `lane/fp-v4` first because item 9's premise (`rcp-fp/4`) was not an ancestor of the
  given base; `rfc-0001` has since integrated fp-v4 itself, so the merge is now redundant but harmless.
- `--changed-since` needs the pod's engine version before engines start; the bootstrap exports
  `RCP_ENGINE_PYTHON` so the node path resolves it, but the submitted wave does not pass
  `--changed-since` (T1 is operator-run per the docs). Wiring it (and T3) into `submit.sh`/`bootstrap.sh`
  is the remaining A5 work.
- Committed observation corpora predate `collector.passes`; they are repository subsets whose checks
  defer to their full corpus, so nothing breaks today. A full re-recording under `rcp-fp/4` will carry
  the three repetitions.

## CHANGELOG entry

Under `## Unreleased` -> `### Fixed`:

> **The GPU wave harness runs on the node and its verdicts mean what they say** (runner review A2-A5,
> B1, B5, C4 and item 9): `rc_build.sh` builds the unpublished `rcp-ndcg-test` wheel by name into
> `<stage>/harness/`, the bootstrap installs it into the client environment (`--with
> rcp-ndcg-test==<version>`), the client probe imports it and the wrapper quotes its argv from a
> validated manifest version (the old unquoted wrapper expanded a downloaded field into shell syntax);
> every wave upload is verified against the destination and retried with backoff, the outcome lands in
> the recipe's `status.json` row, `wave.json`/`WAVE.md` are written before the last upload (they used
> to be written after it and never reached the URI), and a wave with a failed upload no longer reports
> PASS; a corpus without the `after_restart` sending is refused, the corpus key and the engine version
> come from the running pod (`/version`, then the engine environment's own `vllm`) instead of the
> declared image, a duplicate recipe id in a wave list is refused, an all-skipped `--changed-since`
> wave reports `SKIPPED` (never PASS), the submitted wave's gate set is documented (T0/T2/the recorder;
> T1's corpus, T3 and the controls are operator-run), and the wave hashes the staged plugin wheel's
> modules against the behaviour fingerprint's `plugin_sha256.<module>` inputs and refuses the recording
> when they differ (the engine environment installs that same staged wheel, so pip cannot pick another
> version out of an extra wheelhouse).

## Public surface changes

None in the published surface: the wave harness lives in the unpublished `rcp-ndcg-test` package (not
in the contract snapshots), no CLI/exit-code/schema change in `rcp-ndcg`/`rcp-ndcg-core`/`rcp-ndcg-vllm`,
and `schemas/` plus `tests/contract/snapshots/` are unchanged by this lane's own commits (the fp-v4
merge content carries its own CHANGELOG entries and regenerated schemas).

## Files outside scope

- The `lane/fp-v4` merge (`19490c45`, `82d7ae68`) is the base correction item 9 required; its 138 files
  are that lane's content, not this lane's.
- `rcp-ndcg-test/tests/stub_engine.py`: malformed JSON now returns the measured vLLM 400 instead of a
  500 (needed so a measured status table can pass against the stub).

## For the next lanes

- The E2 wave can start: the client environment imports the harness, and uploads/verdicts are honest.
- Any operator-run `--record-corpus` wave should pass `--plugin-wheel
  <stage>/wheelhouse/rcp_ndcg_vllm-<version>-*.whl`; without it the corpus step records that the wheel
  was not cross-checked.
- `wave.json` gains `verdict` (`passed`/`skipped`/`failed`), `upload` and `upload_failures`; consumers
  should read `verdict` rather than infer from `passed`.
- Wiring T3 (mteb + the harness wheel in the reference environment, a `--quality`/`--record-corpus`
  passthrough in `submit.sh`/`bootstrap.sh`) is the remaining release-gate work.
