# Workstream 07: QA of the release tree and release preparation

Read `handover/00-MASTER.md` first. Specs: `handover/specs/qa-common.md` (binding for all four), `qa-arch.md`,
`qa-correct.md`, `qa-redundancy.md`, `qa-tests.md`.

## A. Four read-only QA passes on the post-layout `rfc-0001` tip
Run each as an independent, adversarial review (a fresh reviewer per pass if you can; evidence or not a finding:
file:line, a reproduction, a measured number):
- **arch**: layering and the one-home table hold after the move; nothing public that should be internal and vice versa;
  the contract snapshots match; a module map (purpose, layer, public/internal per module).
- **correct**: the identities, budgets, cuts, anchors, scoring protocols, calibration fit and the paper's numbers;
  rerun `run_all.py`; probe edge cases (empty documents, over-length inputs, media limits, unicode normalisation).
- **redundancy**: second implementations of any one-home concept, dead code, duplicated tests, stale shims left by
  the many merges (e.g. two over-length padding paths, two corpus readers, a harness copy of product logic).
- **tests**: every test can fail (mutation-sample the critical modules: metric, gain, protocols, budget/fit, the
  harness gates, the emulator conformance); fewer, stronger tests; no test writes outside `tmp_path`; no network
  without the gate; a shuffled single-process run is green.
Write each report to `handover/reports/qa-<name>.md`; then fix every blocker and major test-first (a fix PR per
concern), and the cheap minors.

## B. Release preparation (CPU side; no tag, no publish)
1. Versions: all four `pyproject.toml` at `0.0.1`; `rcp-ndcg` pins `rcp-ndcg-core==0.0.1`; `rcp-ndcg-test` same version,
   never published; `CITATION.cff`'s `version:` in step (no workflow checks it today: add the check to `release.yml`,
   test-first, or say why not).
2. `release.yml` dry-run locally: the build and check steps (each dist built, versions vs a would-be tag, sibling pins,
   the semantic constraints check `python .github/scripts/check_constraints.py`, `twine check`).
3. Dependabot / `pip-audit` on the lock: list alerts; bump only what the lock allows without moving numbers.
4. GitHub CI green on the tip (`gh workflow run ci.yml --ref rfc-0001`), the nightly shuffled run green.
5. Write `handover/RELEASE-CHECKLIST.md`: what is ready; what needs GPU (per recipe: T0 smoke, T1 recordings, T2
   equivalence with the `/tokenize` check, T3 quality, T4 scenarios; re-record the corpora and re-verify the emulators
   at the release fingerprints; fill the `pending_gpu` reference cases; flip recipe statuses from verified evidence
   only); the owner's go; then merge to `main`, tag `v0.0.1`, watch `release.yml` publish core -> rcp-ndcg -> vllm.
Report in `handover/reports/07-release-prep.md`.

## Amendments after M3 (binding)
- Start from `00-MASTER.md` section 9 (open items register): every QA item there is in scope; the GPU items go to the
  release checklist.
- The checklist must also list: `tests/conformance/stale.json` empty (the release-flag test enforces it), every corpus
  re-recorded at the release fingerprints and every emulator re-verified, listwise replay coverage restored, the six
  judge recipes' waves, and the GitHub CI run of the final tip (the M3 milestone was gated locally only).
- `handover/RELEASE-CHECKLIST.md` has a draft from M3; complete it.
- **Freeze the public surface**: review every name in the contract snapshots; keep, make internal, or mark
  experimental, deliberately and once (0.0.1 should need minimal changes later). Anything internal that is public by
  accident is a finding.
- **Dependabot**: the public repository has open Dependabot PRs (#1 oauthlib, #2 vllm, #3 transformers 5.10.1, stale
  against the lock's 5.17); close or supersede each with a recorded reason after the final re-lock.
- **First GitHub CI run**: M1-M3 were gated locally only. Expect runner-specific differences (a real `gcloud` on the
  ubuntu image — the node-script tests read `RCP_GCLOUD_SDK_DIRS`; Python and uv versions; the macOS leg); fix them, do
  not skip them. Also run the nightly shuffled workflow once.
