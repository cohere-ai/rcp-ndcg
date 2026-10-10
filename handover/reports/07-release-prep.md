# Workstream 07 — QA fixes and CPU-side release preparation

**Lane:** `lane/qa07` (worktree outside the operator's checkout). **Base:** `rfc-0001` at `9cd88f0c`, merged with
`a45d4f7b` (the judge-equivalence reference-device fix) as `4afe8d76` before the gate. **Gate:** see
"Gate" below. No GPU work was done; nothing was pushed; `rfc-0001` was not merged into.

## 1. Status

Done, test-first, and in the order the brief ranks:

- **HIGH — the conformance comparator's worst-cell verdict.** `_compare_abs` reduces the per-cell deltas with
  `np.max`; the only failure case perturbed every cell, where min and max agree, so `np.max` → `np.min` survived
  the whole conformance + golden-replay set. A one-cell case now pins the worst-cell reduction and the named cell
  (`rcp-ndcg-test/tests/test_conformance.py::test_one_cell_out_of_tolerance_fails_a_mostly_correct_matrix`). The
  mutation makes it red and the old case still pass.
- **HIGH — the test package had no per-test timeout.** `faulthandler_timeout` dumps a traceback and does not
  abort (a 4 s sleep passed under `-o faulthandler_timeout=1`), so a hang ran to the outer gate/CI timeout. The
  root conftest's SIGALRM guard is now its second copy (`rcp-ndcg-test/tests/conftest.py`; the two trees cannot
  share a conftest), with a test that the timer is armed; the gate's `test-pkg` step and CI's `test-pkg` and
  `recipe-tests` jobs set `RCP_NDCG_TEST_TIMEOUT=300`. Probe: a 4 s sleep under `RCP_NDCG_TEST_TIMEOUT=1` now
  fails with "the test exceeded its 1s per-test timeout".
- **MEDIUM — the stage-2 gate defaults.** Mutating `_TAU_MIN` 0.98 → 0.5 survived every gate-referencing test
  (126 passed). All seven defaults and `embed_dtype` are now pinned in one table for a recipe that declares no
  overrides, and a declared field is shown to override only itself.
- **MEDIUM — `fixed_overhead`'s unframed-instruction term.** `fit` subtracts the same term back out, so removing
  it from `fixed_overhead` left the whole root suite green (3991 passed). Three direct-caller tests pin it, and
  the rerank pair's media allowance — measured with `rendered_pair_tokens`, which carries the term — has a test
  that media riding whole without an instruction are refused once the request carries one; the mutation
  reproduces the "overhead plus the declared media already fill the budget" `ConfigError`.
- **MEDIUM — the root tree had no DNS watchdog.** `getaddrinfo` was patched only in the test package's conftest;
  in `tests/` the rule was `HF_HUB_OFFLINE` plus per-test `skipif`. The same guard now runs there (IP literals
  still resolve; `network`-marked items and `RCP_NDCG_NETWORK_TESTS=1` go without it). It caught two tests that
  really did touch the network, both fixed rather than exempted: a media-fit test fetched a `gs://` source to
  learn the shrink fails (a local undecodable file says the same, at once — it also cost ~50 s of retries), and
  the mirror-read fallback test submitted against `gs://` (the in-memory scheme carries the same read error).
- **MEDIUM — `leaderboards.py --suite <one>` always exited 1.** The checker was handed all six documented
  deviation populations and fails one that did not materialise, so `--suite vidore` (its table perfect: 252
  checks, 252 match) printed six `FAIL known-deviation population` lines, against the README's "each script also
  runs on its own". The populations are scoped to the suites that ran; `--suite vidore`/`bright`/`nanobeir` now
  exit 0 on the cached data (252/364/392 checks).
- **MEDIUM — the duplicated entry-point registry.** `data/io/registry.py` and `results.py` were byte-identical
  copies, and `runners/registry.py` implemented the same lookup with the opposite duplicate-name policy (refuse
  vs warn-and-last-wins), so a plugin publishing a built-in's name replaced it silently in two seams and was
  refused in the third. `rcp_ndcg.support.entrypoints` owns the listing, the load check and the ambiguity
  refusal; all three call it and refuse, naming the providers.
- **MEDIUM — the plugin-seam story.** `AGENTS.md` named one group; the product reads six. The doc now lists all
  six (with `packaging.json` as the authority) and `docs/how-to/use-verified-fake-engines.md` has a section on
  `rcp_ndcg.fake_transports` (the product's seam) versus `rcp_ndcg.emulators` (the test package's), which the
  how-to used to name as if the product read it.
- **MEDIUM — the one-home table's unnamed exception.** The `rcp-ndcg-vllm` references mirror the product's
  resize rule by design (they must not import the product); the table now names the exception and why the
  mirror is validated on GPU.
- **MEDIUM — the public-surface freeze.** 194 of the 376 pinned names appear on no docs page. The freeze is now
  deliberate: `docs/reference/public-surface.md` states the rule, and a contract test requires every pinned name
  to be documented on a page under `docs/` or listed in `tests/contract/undocumented_public_names.json` — in
  both directions, regenerated by `--update-snapshots`.
- **MEDIUM — redundancy A4/A5/A6.** The internal-labels guard had seven homes (its declared one home plus six
  per-family copies with six regexes, each subsumed); the six are gone. A byte-identical stage-3 test lived in
  `test_equivalence.py` and `test_record_and_wave.py`; the wave copy is gone. The two copies that stay (five
  retired-media-column refusals; the two folded plugins' zero-init tuples) are pinned by new tests instead of
  trusted — the topk copy had no pin at all.
- **LOW.** The zembed projection test now carries the `network` marker (it used to clear `HF_HUB_OFFLINE` and
  skip with a fetch-failure reason); the paper-configs test asserts its search result and that the dataset's
  instruction is nowhere in the requests (forcing `instruction: fold` turns it red); `bootstrap.sh`'s hermetic
  PATH in `test_jobs_scripts` names every tool the script reaches. The 111 orphaned `stub_engine` processes on
  the operator machine (all ~18 h old) are gone; an AST scan of both trees finds no `start_stub` caller that does
  not stop its engine in a `finally`, so no fixture change was needed.
- **Deferred §10 items fixed:** the credentialed URL in an `env` VALUE is stripped from the recorded config
  (`safe_url` beside the secret-name rule) — it reached the mirrored `run.yaml` in clear; `release.yml` now
  checks `CITATION.cff`'s `version:` against the tag (test-first, `tests/test_workflows.py`).
- **Release preparation:** the four `pyproject.toml` at `0.0.1` with `rcp-ndcg-core==0.0.1` and `CITATION.cff`
  `0.0.1`; the local `release.yml` dry-run (three distributions built at the version, the tag check, the sibling
  pins, the semantic constraints check — 107 pins agree with the lock — and `twine check` on all six files);
  `handover/RELEASE-CHECKLIST.md` completed with the CPU items that hold, the E2 wave state, the eight stale
  corpora, and the owner-only steps.

## 2. Commits

| commit | what |
|---|---|
| `c21e8992` | the mixed-delta conformance case + the test package's per-test timeout guard (gate/CI env) |
| `1ad133d1` | the stage-2 gate default table + the `fixed_overhead` direct-caller and media-allowance tests |
| `569fb2bf` | the root tree's DNS watchdog + the two tests it caught |
| `76b2f1eb` | the single-suite leaderboards run |
| `95436b2c` | the six entry-point groups, the `fake_transports` section, the one-home exception |
| `5e4a88e3` | the public-surface freeze (the page, the contract test, the list) |
| `f51d3bcb` | one home for the entry-point lookup, one duplicate-name policy |
| `9c3bcbee` | the retired duplicated guards, the two drifting copies pinned |
| `b14f4d9d` | import order in the new test |
| `eab5e39a` | the zembed network gate, the paper-search assertion, the sandbox PATH |
| `a8717436` | the env-value credential strip + the CITATION check in `release.yml` |
| `4afe8d76` | merge of `rfc-0001` (`a45d4f7b`) |
| `5d0ff8f7` | the completed release checklist |

No co-author lines, no attribution, nothing pushed.

## 3. What changed (files)

- `rcp-ndcg-test/tests/`: `test_conformance.py`, `conftest.py`, `test_timeout_guard.py` (new),
  `test_equivalence.py`, `test_record_and_wave.py`, `recipes/test_recipe_hygiene.py`, `recipes/test_zembed_1.py`,
  `recipes/test_jobs_scripts.py`, six `recipes/test_*.py` (the retired per-family guards).
- `rcp-ndcg-test/src/`: none (the conformance comparator itself is correct; the case was missing).
- `rcp-ndcg/src/rcp_ndcg/`: `support/entrypoints.py` (new), `data/io/registry.py`, `results.py`,
  `runners/registry.py`, `runs/config.py`.
- `tests/`: `conftest.py`, `data/test_text_budget.py`, `inference/test_client_budget.py`,
  `data/test_prepare.py`, `runs/test_execution.py`, `runs/test_secrets.py`, `retrieval/test_paper_configs.py`,
  `experiments/test_reproduction_rules.py`, `contract/test_public_names.py` (new),
  `contract/undocumented_public_names.json` (new), `support/test_entrypoints.py` (new), `test_workflows.py`.
- `rcp-ndcg-vllm/tests/models/topk/test_weight_mapping.py`.
- `experiments/leaderboards.py`.
- `docs/reference/public-surface.md` (new), `docs/how-to/use-verified-fake-engines.md`, `mkdocs.yml`, `AGENTS.md`,
  `CHANGELOG.md`, `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `handover/RELEASE-CHECKLIST.md`.

## 4. Verification

Every fix has a failing test first; the report's per-item sections name the mutation that turns each new test red:

| fix | mutation | result |
|---|---|---|
| conformance mixed-delta | `np.max` → `np.min` in `_compare_abs` | new test red, the old all-cells case green |
| stage-2 defaults | `_TAU_MIN` 0.98 → 0.5 | 1 failed (the new table), 129 passed |
| `fixed_overhead` term | the term dropped from `fixed_overhead` | 1 failed (the direct caller), 191 passed |
| media allowance | `instruction=""` in the rerank allowance | 1 failed with the documented `ConfigError` |
| env-value credential | the value rule dropped | 1 failed, 34 passed |
| paper-configs assertion | `instruction: fold` forced | 1 failed |
| test-package timeout | (probe) a 4 s sleep at `RCP_NDCG_TEST_TIMEOUT=1` | `TimeoutError`, the guard fires |

Runs (all in this worktree, offline):

- root suite with the DNS watchdog: `4001 passed, 103 skipped in 83.40s` (`-n 4`).
- test package with the timeout guard armed: `1114 passed, 227 skipped in 721.99s`; slowest test 22 s (setup),
  so 300 s is a real bound and not a tripwire.
- `tests/contract tests/docs`: `307 passed, 57 skipped`.
- the recipe tests after retiring the six guards: `263 passed, 226 skipped`.
- `experiments/leaderboards.py --suite vidore|bright|nanobeir`: exit 0 (252/364/392 checks).
- the release dry-run: three distributions built, the sibling pins, `107 pins agree with the lock's export`,
  `twine check` PASSED on all six files, the CITATION check passes at `0.0.1` and fails at `0.0.2`.
- `ruff check`, `ruff format --check`, `basedpyright` and `mkdocs build --strict`: the gate's steps (see below).

### Gate

`timeout 7200 bin/gate lane/qa07` on the merged tree (`403b26c4`, slot 4) — **GATE: PASS**, every step exit 0:

```
ruff-check exit=0 All checks passed!
ruff-format exit=0 616 files already formatted
basedpyright exit=0 0 errors, 0 warnings, 0 notes
pytest exit=0 4015 passed, 105 skipped in 53.25s
contract-docs exit=0 307 passed, 57 skipped in 41.09s
mkdocs exit=0 INFO    -  Documentation built in 1.44 seconds
test-pkg exit=0 ================ 1100 passed, 227 skipped in 719.74s (0:11:59) =================
recipes exit=0 recipes: no failure outside the baseline (0 baseline failures remain, 0 fixed; pytest exit 0)
vllm-pkg exit=0 50 passed in 4.44s
vllm-models exit=0 93 passed, 7 skipped in 54.02s
run_all exit=0   external_judges  ok
leaderboards: 1022 checks, 987 match, 35 known deviations, 0 failed
human study: 67 checks, 67 match, 0 known deviations, 0 failed
external LLM judges: 82 checks, 82 match, 0 known deviations, 0 failed
public-names exit=0 public-names: clean (0 baselined hits remain)
clean exit=0 clean
GATE: PASS
```

The two commits after it change only `handover/` Markdown (this report's gate line and the lane report);
`public-names` and `mkdocs --strict` were re-run on the final tip and stay clean.

## 5. Open questions

1. **`rcp_ndcg_test` is not public, so the per-test timeout default (300 s) is a judgement.** The tree's slowest
   tests are subprocess/node-script runs (22 s measured); a lower default (60 s, the root tree's) would fail a
   slow CI runner. The gate and CI set the variable explicitly, so the default only matters for a local run.
2. **The 194-name advanced list is a freeze, not documentation.** The brief allowed either "missing reference
   pages" or a deliberate freeze; this lane took the freeze (one generated list, enforced in both directions) and
   documented the rule. Writing 194 individual reference entries is a docs-06-sized task.
3. **`_redact_env`'s URL strip changes the recorded value of a non-secret env var.** A resume reads the redacted
   `run.yaml`, so an env value that was a URL with a query now loses the query (`https://proxy/v1?token=...` →
   `https://proxy/v1`). The alternative — leaving the credential in the mirrored record — is worse, and the
   secret-name rule already replaces such values outright; the report records it rather than leaving it silent.
4. **`rcp-ndcg.fake_transports` and `rcp_ndcg.emulators` were kept as two groups.** The arch review offered
   "document or fold"; this lane documented them (the product's seam and the test package's registry, with what a
   provider returns). Folding them would move the entry-point group in a shipped distribution, which is a
   decision for the owner, not a QA fix.
5. **The `pip-audit`/Dependabot half of the release preparation needs the owner.** This environment has no GitHub
   credentials and no advisory-database access; the lock's `oauthlib` is 4.0.0 and `transformers` 5.17.0, so PRs
   #1 and #3 are stale against it. Listed in the checklist rather than guessed at.

## 6. Deferred (found, not fixed; each with its reason)

- **redundancy A3 — the three over-length padding paths** in the harness (`stages._over_length`,
  `wave0_probe._inputs`, `requests._pad_to_tokens`): three target semantics and three failure modes. The
  consolidation the review sketches (one `_pad_to_tokens` with a target parameter plus the loud-refusal policy)
  is a harness refactor that changes what wave 0 and stage 1 inject; it is not a release blocker and would need
  its own wave evidence.
- **redundancy A2 — the tokenizer sidecar identity trap** (the harness's store path hashes raw `tokenizer.json`
  bytes while the product loader extends the digest with effective sidecar tokens): dormant on today's 24
  vendored specs, but the check that settles it needs the Hub (`load_tokenizer(spec).sha256 == stored sha` for
  every entry). Left to a network-gated test in a lane that can run it; recorded here so it is not silent.
- **redundancy T3 — the per-recipe `_assert_contract` machinery** in the recipe test files: boilerplate
  duplication with per-family self-containment as the cause; a shared `recipes/_contract.py` is a test-package
  refactor, not a fix.
- **ARCH-5 — the `errors → support.config` lazy cycle**: the only cross-layer cycle the charter does not
  sanction. Moving `config_error`/`validation_problems` to a module at or below `errors` is a layering change
  that touches the error-classification surface; the tree passes its layering test today.
- **The two `Family` concepts** (core IRT vs recipe): renaming one is a public-surface change in two
  distributions and the contract's `KNOWN_SECOND_HOMES` already records the pair; a rename belongs to the
  surface freeze decision, not to a QA fix (the freeze records it as an accepted second home).
- **§10 items not fixed:** the dshm `sizeLimit` (a job-spec change with its own contract test and docs page —
  worth doing deliberately, not as a drive-by), the PodInitializing documentation sentence (the
  `test_a_job_whose_pods_cannot_be_listed_is_unknown_not_running` test already pins the message the review
  thought unreachable: `note()` is called only for pending/running jobs and re-lists the pods, so a transient
  listing failure reaches it), the `run cancel`/`ENGINES_ENV` CHANGELOG entries (they belong to the lane that
  shipped those changes; this lane's CHANGELOG block covers its own), and the re-raised submission exception's
  redaction (`execution.py`: the recorded error is redacted, the re-raised one is not — a behaviour change for a
  plugin runner's error path, recorded rather than guessed).

## CHANGELOG entry

Added under `## Unreleased` (the exact text is in `CHANGELOG.md`):

- `### Changed` — the entry-point lookup's one home and one duplicate-name policy; the public-surface freeze
  (194 of 376 names as the reviewed advanced list).
- `### Fixed` — a credential that rides an environment VALUE no longer reaches the recorded config; a single
  `experiments/leaderboards.py --suite <one>` run exits 0.

## Public surface changes

- No public name, flag, exit code or schema changed: `tests/contract/snapshots/` and `schemas/` are byte-identical
  after a `--update-snapshots` run (verified: the regeneration produced no diff).
- The entry-point **policy** changed (an ambiguous name is now refused in the readers/writers and results
  registries as it already was for runners): same groups, same names, a different failure mode for a plugin that
  collides with a built-in. The CHANGELOG records it under `### Changed`.
- `CITATION.cff`'s version and `release.yml` gained a check; neither is part of the pinned surface.

## Files outside scope

- `.github/workflows/ci.yml`, `.github/workflows/release.yml` (the timeout env and the CITATION check).
- `AGENTS.md` (the six groups, the one-home exception and the new entry-point home).
- `mkdocs.yml` (the new reference page's nav entry).
- `handover/RELEASE-CHECKLIST.md`.
- The shared gate script outside the repository (`bin/gate` in the lanes root): its `test-pkg` step now exports
  `RCP_NDCG_TEST_TIMEOUT=300`. No other lane's worktree was touched.

## For the next lanes

- The conformance suite now fails a one-cell violation by construction; re-recorded corpora should be checked
  with that case in place.
- The test package's suite has a 300 s per-test bound: a legitimately slower test needs the `no_timeout` marker
  (or a higher `RCP_NDCG_TEST_TIMEOUT`), not a disabled guard.
- The public-surface freeze is enforced: a new public name fails `tests/contract` until it is documented or added
  to `undocumented_public_names.json` (regenerate with `--update-snapshots`).
- The root tree's DNS watchdog means a test that reaches a name fails the run; mark it `network` if it must.
