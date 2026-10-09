# Lane `qa-prep` — workstream 07 parts independent of the in-flight lanes

Base: `rfc-0001` tip `681a8cea`; merged `rfc-0001` eight times: `28afb3b7` (l08-sglang, mrl-cards, l10a) as
`b2e0e6cf`, `7013f28d` (rf-research, judge-gemma: handover files only) as `dcfd757e`, `b67699c0` (l10b: the
MTEB export surface) as `2fb7ae61`, `ec7719cc` (sync-hardening) as `b5d5f84e`, `c5ccc851` (scoring-fixes) as
`bb1255fa`, `75a341d9` (handover only) as `55aa1c66`, `247c3d53` (recipe families, decision 34, plus rec-overrides,
sz-misc, rec-egemma2, rec-harrier) as `c27d1c5b`, and `446765e5` (l10d: the MTEB export follows PR #5516) as
`9d68f37c`. Scratch work (shuffled id lists, logs, the release dry-run, the audits) lives in a scratch directory
outside the repository.

## Status

DONE. All five brief items are implemented or audited, the root suite (3546 passed, 101 skipped), the contract and
docs suites (304 passed, 55 skipped) and the `rcp-ndcg-test` suite (621 passed, 349 skipped) are green on the merged
tree, and the checkout guard keeps the tree clean. The release dry-run and the checklist audit are in this report.

**Gate:** `bin/gate lane/qa-prep` -> `GATE: PASS` on `a15672b4` (the report commit that follows is the only delta).

## Commits

| Hash | Subject |
|---|---|
| `abfd0d00` | tests: the SLURM submission-failure test keeps its log_dir under tmp_path |
| `e7b38f76` | tests: fail a run that leaves a new file or directory in the checkout |
| `42a57f51` | github: weekly grouped Dependabot updates for uv and GitHub Actions |
| `b2e0e6cf` | Merge branch `rfc-0001` (`28afb3b7`) into `lane/qa-prep` |
| `05cca7d5` | tests: the checkout guard takes its baseline before collection |
| `dcfd757e` | Merge branch `rfc-0001` (`7013f28d`) into `lane/qa-prep` |
| `2fb7ae61` | Merge branch `rfc-0001` (`b67699c0`) into `lane/qa-prep` |
| `b5d5f84e` | Merge branch `rfc-0001` (`ec7719cc`) into `lane/qa-prep` |
| `bb1255fa` | Merge branch `rfc-0001` (`c5ccc851`) into `lane/qa-prep` |
| `55aa1c66` | Merge branch `rfc-0001` (`75a341d9`, handover only) into `lane/qa-prep` |
| `c27d1c5b` | Merge branch `rfc-0001` (`247c3d53`, recipe families) into `lane/qa-prep` |
| `9d68f37c` | Merge branch `rfc-0001` (`446765e5`, the l10d MTEB export) into `lane/qa-prep` |
| `fbc3d5c2` | tests: the hub-cache warning assertions read only RCP-nDCG warnings |
| `3fbb2168` | tests: the guard's clean-run test removes a directory, not a file |
| (this report) | handover: the qa-prep report, the verifier rounds and the round-1 fixes |

## What changed (per brief item)

### 1. Test hygiene

- **The leak.** `tests/runs/test_execution.py::TestStatus::test_a_submission_that_fails_leaves_a_failed_run_saying_why`
  runs `run start --runner slurm` with `PATH` stripped of `sbatch`; `SlurmRunner.submit` creates
  `Path(log_dir)` before it calls `sbatch`, and the runner's default `log_dir` is `logs/slurm` relative to the
  working directory, so the test left an empty `logs/slurm/` in the checkout. Reproduced before the fix: running
  that single test created `logs/slurm/` (`find logs -type d`), and `git status` stayed silent because the directory
  is empty and `/logs/` is gitignored. The test now sets `runner.options.log_dir` under `tmp_path` (the config
  carries it; `--runner slurm` still selects the same code path), and the test file leaves no `logs/` behind.
- **The guard.** `tests/_checkout.py` holds the scan (`entries`) and the comparison (`checkout_guard`, a context
  manager that names every added path). The root conftest takes the baseline in `pytest_sessionstart` (before
  collection, so an import-time leak is caught too) and runs the session under `checkout_guard`; the
  `rcp-ndcg-test` suite carries its own copy of the module and the same fixture, because the two suites install
  independently and the root `tests/` tree is not importable from the unpublished one. Cache/environment
  directories (`.git`, `.venv`, `__pycache__`, the pytest/ruff/mypy/basedpyright caches, `node_modules`) are not
  the checkout's tracked content and are skipped; `--update-snapshots` / `RCP_NDCG_UPDATE_SNAPSHOTS=1` regenerates
  the tree on purpose and is exempt in both copies.
- **Its tests.** `tests/test_checkout_guard.py` and `rcp-ndcg-test/tests/test_checkout_guard.py` pin the scan, the
  comparison (a guard that compares nothing fails them), the baseline semantics (a file written between the
  session-start baseline and the guard entry is a finding) and, in the test package, that the module under test is
  that suite's own copy. The fixture itself was demonstrated red-first with a temporary leaking test: `1 passed,
  1 error`, exit 1, message `the tests left new files or directories in the checkout ...: stray-dir, stray-file.txt`;
  a module-level leak is caught the same way (`import-leak.txt`, exit 1).

### 2. The shuffled suite

The nightly workflow cannot be dispatched (see item 4/5), so its own procedure was replicated locally: collect the
node ids (`pytest <suite> --collect-only -q | grep '::' | sort`), shuffle them with `random.Random(<seed>)`, and run
`heavy uv run --no-sync pytest -p no:xdist -q @ids.txt` single-process. The collected ids are rootdir-relative, so
the `rcp-ndcg-test` runs were made from `rcp-ndcg-test/`. Runs, all green, zero order-dependent failures:

| Suite | Seeds (all passed) |
|---|---|
| root `tests/` | 20261009, 1, 424242, 13 (pre-merge), 2026100912, 20261009 (post-merge), 20261009 (after the round-1 fixes) |
| `rcp-ndcg-test/tests` | 20261009, 7, 20261010 (pre-merge), 20261009 (post-merge, twice), 20261009 (after the round-1 fixes) |

One incident, not order-dependent: the first post-merge `rcp-ndcg-test` shuffled run exited 139 (SIGSEGV) at 54 %,
inside pytest's own tmpdir cleanup (`_pytest/pathlib.py::maybe_delete_a_numbered_dir` -> `shutil.rmtree`), with
13 concurrent pytest processes on a 2 GB machine; the identical seed and order reran to `570 passed, 225 skipped`.
The verifier lens B independently saw the same class of crash twice more (seeds 314159 at 63 % and 161803 at 81 %,
load average ~53, 31 pytest processes) and each identical rerun was green. It is an environment flake, not an
order-dependent test failure, and it left the checkout clean.

The skipped-test delta between `-n 4` and the shuffled runs is explained and benign: two modules skip at import
(`pytest.importorskip`) and so have no collected ids for the nightly-style list (`tests/data/test_io_pdf.py` and
`tests/data/test_io_hf.py` before the merge; `test_io_pdf.py` and `test_io_mteb_task.py` after).

**One order-dependent failure was found and fixed.** On the `ec7719cc` merge, the shuffled root run with seed
`20261009` failed `tests/data/test_hub_cache.py::test_offline_corpus_materializes_from_the_snapshot` with
`AttributeError: 'ResourceWarning' object has no attribute 'code'` at line 256; the ordered `-n 4` run passed.
`pytest.warns` records every warning raised in its block, and in that order an earlier test's garbage (an unclosed
asyncio event loop, seen as `ResourceWarning` in 15 places across the suite) put itself first, so `seen[0]` was not
the `RcpNdcgWarning`. The four warning assertions in the file now filter to `RcpNdcgWarning` (a small `_rcp_warnings`
helper); the same seed and order reran green (`3394 passed, 91 skipped`). While diagnosing, the guard's own
`test_the_guard_passes_a_clean_run` turned out to be NFS-fragile (1 in 5 runs left an NFS `.nfs*` placeholder after
`unlink()`, which the guard correctly reported); it now removes a directory instead of a file.

### 3. Supply chain

- **`.github/dependabot.yml`** (new): one weekly grouped pull request per ecosystem — `uv` at `/` (all four
  workspace members' dependencies through `uv.lock`; the generated `requirements-constraints.txt` is never an update
  source) and `github-actions` at `/` (the pinned workflow actions). Validated against the schemastore
  `dependabot-2.0.json` schema with `check-jsonschema`: `ok -- validation done`.
- **`pip-audit` over the locked environment** (`uv export --frozen --no-hashes --no-emit-workspace --all-groups
  --all-extras --all-packages`, 156 pinned dependencies; `uvx pip-audit -r ... --no-deps --disable-pip`, both the
  `pypi` and `osv` services; the shipped `requirements-constraints.txt` subset gives the same result):

  | Package | Locked | Advisory | Alias | Fixed in |
  |---|---|---|---|---|
  | pyjwt | 2.14.0 | PYSEC-2026-4141 | CVE-2026-101918 (GHSA-42vr-xj54-vc7v) | 2.15.0 |
  | pyjwt | 2.14.0 | PYSEC-2026-4183 | CVE-2026-102275 (GHSA-x33g-cr3x-6449) | 2.15.0 |

  `pyjwt` enters only through `msal` 1.39.0 <- `azure-identity` 1.25.3 <- the `rcp-ndcg[azure]` extra. The bump is
  `pyjwt==2.14.0 -> 2.15.0` in `uv.lock`; `msal` requires `PyJWT[crypto]<3,>=1.0.0`, and 2.15.0 exists and
  satisfies it. Per the brief this is reported, not applied (the lock belongs to a later lane). The CHANGELOG's
  "Dependabot alerts on the default branch's lock" paragraph should name this bump when the re-lock lands.

### 4. Release dry-run review

`.github/workflows/release.yml` matches the four-distribution layout, and every step of its `build` job was run
locally into a scratch directory (no publish, no tag):

- `uv build --package rcp-ndcg-core|rcp-ndcg|rcp-ndcg-vllm --out-dir <scratch>/dist` built exactly the six files
  the workflow expects (`rcp_ndcg-0.0.1-{py3-none-any.whl,tar.gz}`, `rcp_ndcg_core-...`, `rcp_ndcg_vllm-...`);
  `rcp-ndcg-test` is never built (unpublished, decision 20).
- The workflow's own "the versions are the tag's" step (`GITHUB_REF_NAME=v0.0.1`) passed; "each package pins its
  sibling at the tag's version" passed (`rcp-ndcg/pyproject.toml` pins `rcp-ndcg-core==0.0.1`; `rcp-ndcg-vllm` names
  no sibling, decision 18); `python3 .github/scripts/check_constraints.py` passed ("107 pins agree with the lock's
  export"); `uvx twine check` on all six files: PASSED.
- Publish order (`build` -> `publish-core` -> `publish-rcp-ndcg` -> `publish-vllm` -> `github-release`) and the
  one-environment-per-package mapping match AGENTS.md; the tests in `tests/docs/test_packaging.py` pin them.

Mismatches found (reported, not edited — release prep has its own lane):

1. **`CITATION.cff` is not checked.** The workflow verifies the tag against the built artifacts and the sibling
   pins but never against `CITATION.cff`'s `version:`, although the workstream asks for the four versions to move
   together. Fix: one step comparing `CITATION.cff`'s `version:` to `${GITHUB_REF_NAME#v}` (or a `tests/docs` test
   running that step body), as the sibling-pin step already does.
2. **`gh release create --notes-from-tag` needs an annotated tag.** `--notes-from-tag` reads the tag's message; a
   lightweight `v0.0.1` makes the last job fail after all three packages are already on PyPI (immutable). Fix:
   document "annotated tag" in the release instructions or use `--generate-notes`.
3. No other mismatch: the build/publish/artifact/env/action-pin layout is consistent with the four members.

### 5. RELEASE-CHECKLIST audit

Every line checked against the merged tree (`9d68f37c`), each with its evidence or its owning lane. "OPEN" means
the item is not done at this base, not that the checklist is wrong.

**Section 1 — CPU workstreams**

| Line | State | Evidence / owning lane |
|---|---|---|
| 05 layout move | DONE | Four top-level distribution directories; `uv run --no-sync python tools/layout_move.py --check` prints "the tree is in the target layout, no old path anywhere"; `handover/reports/05-layout.md` is DONE |
| 08 vLLM only, recipes for every role, six judge recipes | PARTIAL — 08 A and 08 C done, 08 B/D open | `handover/reports/08a-vllm-only.md` DONE (the remaining SGLang mentions are prose comparisons and a judge example, not code paths); `handover/reports/08c-judge-catalog.md` DONE (the catalog spec only); decision 34's family layout landed (`handover/reports/03b-recipe-families.md`): 15 family directories hold 27 recipes (12 embed, 11 rerank, 4 multi-vector), all `status: unverified`, and no `role: judge` recipe exists yet. Owning lane: 08 B/D (`l08-judges`, in flight) |
| 09 pipeline | DONE | `handover/reports/09-processing-pipeline.md` DONE; merged as `7229113b`; `rcp_ndcg.inference.clients._base.STAGES` declares the pipeline |
| 10 data I/O and MTEB | PARTIAL — 10 A/B/C1 and 10 D done, C2/C3 open | `handover/reports/10a-data-io.md` DONE (the Hub reader in MTEB's layout, the `mteb:<Task>` reader, the retired `hf` heuristics reader, the data-model fields); `handover/reports/10b-mteb-export.md` DONE (the mteb writer, `Rankings.save(format='mteb')`, scoring inside mteb, the republishing converter); `l10c` (C2/C3) is in the parallel plan and in flight |
| 06 final docs and the CHANGELOG fold | OPEN — lane 06 | `CHANGELOG.md` still has `## Unreleased` (line 24); no `handover/reports/06-*.md`; `mkdocs.yml`'s nav has no compatibility/versioning page |
| 07 QA passes | PARTIAL — this lane is the independent QA/release items; the four `qa-arch/correct/redundancy/tests` passes have no report in the tree | `ls handover/reports/qa-*.md` -> none; the 00-MASTER section 9 QA list is still largely open (see "For the next lanes") |
| GitHub CI green on the final tip | OPEN — owner | M1–M3 were gated locally only; the GitHub API shows the default branch `main` carries `ci.yml` and `release.yml` but no run of this tip. The owner pushes and dispatches `gh workflow run ci.yml --ref rfc-0001` |
| Public surface frozen + the compatibility/versioning page | OPEN — lanes 07/06 | No freeze report or commit; no versioning page in `mkdocs.yml`; the policy text lives only in `CHANGELOG.md`'s Versioning section |
| README images with absolute tag-pinned URLs | VACUOUS today | `git grep -n '!\[' README.md docs/` finds no Markdown image; if lane 06 adds any, they must be absolute and tag-pinned |

**Section 2 — GPU waves (owner; all OPEN at this base)**

| Line | Evidence |
|---|---|
| Every retrieval recipe's T0–T4 waves | 27 recipes across 15 family directories (`iter_recipes()`: 12 embed, 11 rerank, 4 multi-vector), every `status:` `unverified`; the checklist's per-id list of 19 is stale and should be regenerated from the catalog (lane 06/08) |
| The six judge recipes' waves | The recipes do not exist yet (08 B/D) |
| Re-record the corpora declared stale | `rcp-ndcg-test/tests/conformance/stale.json` holds 7 entries, not `[]`; the release-flag test enforces the empty state |
| Replace the provisional corpora | 12 corpus `manifest.json` files carry `provisional` with `not_valid_for: release evidence` |
| Listwise replay coverage restored | Absent for `jina-reranker-v3` until re-recording (00-MASTER section 9) |
| Fill the `pending_gpu` expected values | 79 occurrences under `rcp-ndcg-test/cases/` (78 case-YAML values plus one literal in `qwen3-vl-embedding-2b/make_texts.py`) |
| Media: per-clip video pixel budget + a page-image observation | `qwen3-vl-embedding-2b`'s recipe declares the engine/client 25,165,824 px per clip against the card's 7,864,320 px; the ViDoRe retrieval view is waived until a corpus observes a page image |
| Media: per-clip video pixel budget + a page-image observation | `qwen3-vl-embedding-2b`'s recipe declares the engine/client 25,165,824 px per clip against the card's 7,864,320 px; the ViDoRe retrieval view is waived until a corpus observes a page image |
| Real SLURM and Kubernetes runs of the job shapes | 00-MASTER section 9: "Unverified on real infrastructure: SLURM `srun --kill-on-bad-exit/--wait`, Kubernetes, and the stock image's bash, python3 and pip" |
| Flip every recipe to `verified` | All 27 are `unverified` |

**Section 3 — Release (owner)**

| Line | State | Evidence |
|---|---|---|
| Dependabot PRs #1–#3 closed or superseded with a reason | SATISFIED | The public GitHub API reports #1 (oauthlib), #2 (vllm) and #3 (transformers) all `closed`, each with Dependabot's own ignore-confirmation comment; no open pull requests at audit time |
| HF datasets republished in MTEB's exact layout | OPEN — owner | Not verifiable from the tree; lane 10 D builds the converter, the owner pushes |
| Delete `handover/` and its `tests/docs` exclusion | OPEN — owner | `handover/` is still present, by design until the release commit |
| Owner's go, merge to `main`, tag `v0.0.1`, watch the publish order | OPEN — owner | — |

The amendment's required additions are all present in the checklist draft: `stale.json` empty, the corpora
re-recorded at the release fingerprints with the emulators re-verified, listwise replay coverage, the six judge
recipes' waves, and the GitHub CI run of the final tip.

## Verification

- **My own checks** (all on the merged tree `9d68f37c` unless noted): full root suite `-n 4` green; `tests/contract
  tests/docs` green; `rcp-ndcg-test/tests` green; ruff format/check and basedpyright clean; the shuffled single-process
  runs green (see item 2); the release build/check steps green; pip-audit reproduced; `check-jsonschema` validated the
  Dependabot config.
- **Round 1 verifiers** (fresh context, DeepSeek-V4.1-flash at `xhigh`, two lenses, launched in parallel, neither
  seeing the other): **both PASS**, no blocker and no major.
  - Lens A (correctness): PASS. Reproduced the guard red-first in both suites (nonzero exit, both paths named,
    including under `-n 4`), the `--update-snapshots` and env-var exemptions, the cache skips, the junitxml
    compatibility, the SLURM fix (pre-fix body reproduced `logs, logs/slurm`), the shuffled seeds, the pip-audit
    findings, the release steps, the Dependabot schema and the checklist spot checks. Three minor report-accuracy
    errors, all fixed in this report: the `pending_gpu` count (78 case values, 79 occurrences with the helper
    literal), decision 22 -> 20 for the unpublished package, and the `rcp-ndcg-vllm/tests` count (17 files, two
    nested conftests, no top-level session guard). One design observation (the test-package guard scans the whole
    workspace root) is intentional and now stated in its docstring.
  - Lens B (regressions and hygiene): PASS. Green root/contract/docs/test-pkg suites on a byte-identical copy of the
    commit, both mutation kills (dirnames, `__pycache__`), the pre-fix SLURM red, the Dependabot schema with a
    negative control, pip-audit, the merge ancestry, no public-surface change and the standing rules. Minors: (F1)
    the fixture's comparison had no automated test and a compare-nothing mutation survived; (F2) an import-time leak
    escaped the guard (baseline taken after collection); (F3) the duplicated guard had drifted (no exemption, no
    tests); (F4) the scratch shuffled script needs `cwd=rcp-ndcg-test` for that suite; (F5) intermittent SIGSEGV
    flakes in single-process test-package runs under host load, always green on identical rerun; (F6) the report was
    untracked and cited a placeholder commit.
- **Round-1 fixes** (minors, no round 2 required): F1 -- `checkout_guard` is a context manager with its own unit
  tests, and the compare-nothing mutation now fails them in both copies; F2 -- the baseline moved to
  `pytest_sessionstart` and an import-time leak is caught (reproduced: `import-leak.txt`, exit 1); F3 -- each
  distribution now has its own tested `_checkout.py`, the exemption is mirrored, and the test package pins that its
  import resolves to its own copy; F4 -- the procedure is recorded with its `cwd` (the runs were made from
  `rcp-ndcg-test`); F5 -- environment, reported; F6 -- the report is committed. After the fixes: root `3275 passed,
  93 skipped`, contract/docs `287 passed, 52 skipped`, test-package `575 passed, 225 skipped`, and both shuffled
  suites green again (root seed 20261009: `3275 passed, 91 skipped`; test package seed 20261009: `575 passed, 225
  skipped`). After the third `rfc-0001` merge (l10b) the same checks were rerun: root `3289 passed, 95 skipped`,
  contract/docs `289 passed, 51 skipped`, test package `575 passed, 225 skipped`, shuffled root `3289 passed, 90
  skipped`, shuffled test package `575 passed, 225 skipped`, and the release build/constraints/twine steps green
  again. After the fourth merge (`ec7719cc`) and the two order-dependent test fixes: root `3394 passed, 96 skipped`,
  contract/docs `294 passed, 52 skipped`, test package `575 passed, 225 skipped`, shuffled root seed 20261009
  `3394 passed, 91 skipped` (the failing seed, now green), and the release steps green again. After the fifth merge
  (`c5ccc851`, scoring-fixes): root `3426 passed, 96 skipped`, contract/docs `295 passed, 52 skipped`, test package
  `575 passed, 225 skipped`, shuffled root seed 20261009 `3426 passed, 91 skipped`, shuffled test package seed
  20261009 `575 passed, 225 skipped`. After the seventh merge (`247c3d53`, recipe families): root `3445 passed, 96
  skipped`, contract/docs `295 passed, 52 skipped`, test package `621 passed, 349 skipped`, shuffled root seed
  20261009 `3445 passed, 91 skipped`, shuffled test package seed 20261009 `621 passed, 349 skipped`. After the
  eighth merge (`446765e5`, the l10d MTEB export): root `3546 passed, 101 skipped`, contract/docs `304 passed, 55
  skipped`, test package `621 passed, 349 skipped`, shuffled root seed 20261009 `3546 passed, 96 skipped`, shuffled
  test package seed 20261009 `621 passed, 349 skipped`.

## Checks

```text
uv run --no-sync ruff format --check .          -> 602 files already formatted
uv run --no-sync ruff check .                   -> All checks passed!
uv run --no-sync basedpyright                   -> 0 errors, 0 warnings, 0 notes
heavy uv run --no-sync pytest tests/ -q -n 4 -p no:cacheprovider
                                                -> 3546 passed, 101 skipped in 95.77s
uv run --no-sync pytest tests/contract tests/docs -q -p no:cacheprovider
                                                -> 304 passed, 55 skipped in 47.52s
heavy uv run --no-sync pytest rcp-ndcg-test/tests -q -p no:cacheprovider
                                                -> 621 passed, 349 skipped in 417.39s
shuffled root seed 20261009 (pre-fix red, post-fix green)  -> FAILED hub_cache::test_offline_corpus..., then exit 0
shuffled root/test-package seeds across the eight merges   -> all exit 0
uv build --package <each of the three>          -> six artifacts, versions 0.0.1
python3 .github/scripts/check_constraints.py    -> 107 pins agree with the lock's export
uvx twine check <six files>                     -> all PASSED
uvx check-jsonschema --schemafile https://json.schemastore.org/dependabot-2.0.json .github/dependabot.yml
                                                -> ok -- validation done
uvx pip-audit -r <lock export> --no-deps --disable-pip
                                                -> 2 findings, pyjwt 2.14.0, fix 2.15.0
bin/gate lane/qa-prep                           -> GATE: PASS on a15672b4 (all steps exit 0; run_all
                                                   1022/987/35/0, 67/67, 82/82; clean)
git status --porcelain --untracked-files=all    -> clean
```

## Open questions

- The guard is duplicated on purpose: each distribution carries its own `_checkout.py` (the two suites install
  independently; the root `tests/` tree is not importable from the unpublished package), and each copy has its own
  unit tests plus a pin that the import resolves to that copy. A single home in `rcp_ndcg_test` would make the root
  suite's conftest depend on the dev-only package; the small tested copy was chosen instead.
- `--update-snapshots` / `RCP_NDCG_UPDATE_SNAPSHOTS=1` is exempt from the guard (that run writes `snapshots/` and
  `schemas/` on purpose). A snapshot update that also leaks elsewhere would not be caught; CI never uses the flag.
- The guard compares new entries only: modifications and deletions of tracked files are the CI `git status` step's
  job, and writes inside the skipped cache directories are out of scope by design.
- `rcp-ndcg-vllm/tests` (17 files: the wheel-contract and model-plugin tests in their own venv, with the two nested
  `tests/models/{pplx,topk}/conftest.py` files but no top-level session guard) has no checkout guard. Its tests were
  not observed to write into the checkout, but the guard does not cover them.
- After merging `rfc-0001`, `uv run --no-sync` alone is not enough: the merged `rcp-ndcg` registers new
  `rcp_ndcg.readers`/`rcp_ndcg.writers` entry points (l10a's `beir`/`jsonl`, l10b's `mteb`), and the existing
  editable install's metadata does not have them (`unknown dataset format 'jsonl'. Available: []`).
  `.github/scripts/cpu-env.sh dev docs` (the documented setup; `--locked`, CPU torch index) fixed it each time. The
  gate refreshes its environment only when `uv.lock`'s hash moves; the l10a merge moved it, but l10b added an entry
  point without moving the lock, so a gate slot that has not synced since l10b can run the suite against stale
  entry-point metadata. Worth making the gate's refresh depend on the workspace manifests too.
- The release `--notes-from-tag` point (annotated tag) is an owner decision, not a code defect.

**CHANGELOG entry**: none. The lane changes tests and repository tooling only; `git diff rfc-0001..HEAD` touches no
public name, CLI, exit code or schema, so `CHANGELOG.md` and the snapshots stay untouched.

**Public surface changes**: none.

**Files outside scope**: none (the lane's commits touch `tests/`, `rcp-ndcg-test/tests/` and
`.github/dependabot.yml` only).

**For the next lanes**

- The lock lane: bump `pyjwt` to 2.15.0 (PYSEC-2026-4141/CVE-2026-101918 and PYSEC-2026-4183/CVE-2026-102275) and
  name it in the CHANGELOG's Dependabot paragraph.
- The release lane: add the `CITATION.cff` version check to `release.yml` (failing test first, like the sibling-pin
  step's test) and settle the annotated-tag requirement; then the owner's CI dispatch on the final tip. The
  checklist's per-recipe GPU list should be regenerated from the catalog: decision 34's family layout turned the
  19 ids into 15 families / 27 recipes.
- The remaining 07 passes (00-MASTER section 9 QA items, still open at this base): the two over-length padding
  helpers (`rcp_ndcg_test/equivalence/stages.py::_over_length` vs `rcp_ndcg_test/observe/requests.py::_pad_to_tokens`)
  confirm-or-unify; the private `rcp_ndcg_core._records` / `irt._*` imports across `rcp-ndcg` and
  `rcp-ndcg-test`; the untested stage-directory guard in `rcp_ndcg_test/jobs/bootstrap.sh:338`; the ctxl reference
  `torch==2.9.1` pin against the image's torch; the head-edge audit being a lower bound; `find_corpora`
  (`rcp_ndcg_test/engines.py:208`) still parsing `manifest.json` itself; the offline/uncached chat-template case for
  `qwen3-vl-embedding-2b`; the pair census's separator-free `kept_tokens`.
- The freeze pass: `tests/contract/snapshots/` and `schemas/` need the deliberate keep/internal/experimental review
  before 0.0.1.
