# Lane relmech — release mechanics and hygiene

**Lane:** `lane/relmech` (worktree outside the operator's checkout). **Base:** `rfc-0001` at `b83a3e00`, merged
with its tip `3217b5ed` as `919de61d` before the lane's work. **Gate:** see below. No GPU work; nothing pushed;
`rfc-0001` was not merged into.

## 1. Status

Done, in the brief's four items plus the checklist housekeeping:

1. **The eight judge families' shipped notes and sources read as public statements, and the recipe hygiene
   guard covers them.** Every `handover/specs/judge-catalog.md`, lane-report, owner-answer, decision-number and
   wave-name citation in the eight judge families' notes, sources and comments is replaced with the public fact
   it stood for (the model card URL, the pinned revision, the flag checked in the vLLM v0.31.0 source, the
   paper-side file). The repo-wide guard
   (`rcp-ndcg-test/tests/recipes/test_recipe_hygiene.py::test_every_shipped_recipe_file_reads_as_a_public_statement`)
   drops its judge-family scoping and scans every shipped family; it fails red on the uncleaned tree (25 hits
   across the eight families) and passes after the cleanup. The ten judge goldens are regenerated the documented
   way (`--update-goldens`): notes and sources only, every fingerprint and every client/serve field unchanged.
2. **The judge-sampling decision is recorded; no shipped field moved.** The shipped recipes already send each
   checkpoint's own `generation_config.json` sampling at the pinned revision (the 2026-10-10 owner decision,
   from the recipe-fix lane); this lane records that decision as **`00-MASTER.md` decision 42**, replaces the
   judge-catalog spec's now-false "no temperature is sent" rule with the decided values and the reason (the
   owner's HF-default call, superseding decision 4.3's paper sampling), adds a note at the head of the spec's
   per-family section (the 2026-10-09 blocks still quote `temperature: null`; the recipes' `client` blocks are
   the authority), and ticks the release checklist's owner-revisit item with the exact values. The goldens
   confirm nothing moved: only `notes`/`sources` changed.
3. **The Dependabot and pip-audit analysis is written** (`handover/reports/dependabot-analysis.md`): 51
   alerts (40 open, 11 fixed), one row per alert with the recommendation; `pip-audit` ran against the release
   tree's constraints file. Nothing was dismissed, closed or upgraded.
4. **The two deferred runner items.** (a) Every Kubernetes pod's `/dev/shm` Memory `emptyDir` now renders a
   `sizeLimit`, the new `KubernetesOptions.dshm_size` (a Kubernetes memory quantity, `16Gi` by default), with
   the test first (red on the unfixed tree), the docs sentence, the CHANGELOG entry, the regenerated contract
   snapshot, the exported run-config schema and the generated API page. (b) A docs sentence documents that a
   multi-phase job whose init container is still initializing reads `pending` with the container's own
   `PodInitializing` reason in the note; the code path was reproduced (see Verification).

**Release checklist:** the 06-docs/CHANGELOG-fold, README-images, judge-citation, judge-sampling and
pip-audit/Dependabot items are ticked with their evidence; the GPU, GitHub-CI and owner items stay unticked.

## 2. Commits

| commit | what |
|---|---|
| `919de61d` | merge of `rfc-0001` (`3217b5ed`) before the lane's work |
| `07db730a` | the eight judge families read as public statements; the hygiene guard covers them; the ten goldens regenerated |
| `1204f9d5` | the judge-sampling decision recorded (decision 42 + the judge-catalog record) |
| `f9fb8c28` | the Dependabot/pip-audit analysis report |
| `94f042aa` | the Kubernetes dshm `sizeLimit` (`dshm_size`), test-first, with docs, CHANGELOG and snapshots |
| `21579ada` | the PodInitializing documentation sentence |
| `4dacf3f4` | the release checklist ticks |

No co-author lines, no attribution, nothing pushed.

## 3. What changed (files)

- `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/`: the eight judge `family.yaml` files (notes, sources, comments).
- `rcp-ndcg-test/tests/recipes/test_recipe_hygiene.py` (the scoping removed),
  `rcp-ndcg-test/tests/recipes/golden/` (the ten judge goldens regenerated).
- `handover/00-MASTER.md` (decision 42), `handover/specs/judge-catalog.md` (the sampling record),
  `handover/RELEASE-CHECKLIST.md`, `handover/reports/dependabot-analysis.md` (new), this report.
- `rcp-ndcg/src/rcp_ndcg/runners/kubernetes.py` (`DEFAULT_DSHM_SIZE`, `KubernetesOptions.dshm_size`,
  `_dshm_volume`), `tests/runners/test_kubernetes.py`.
- `docs/concepts/runs.md`, `docs/reference/api/rcp_ndcg.runners.md` (regenerated), `CHANGELOG.md`,
  `tests/contract/snapshots/python_api.json`, `schemas/run-config.v1.json`.

## 4. Verification

Item 1, the failing-first guard (the scoping removed before the cleanup):

```
uv run --no-sync pytest rcp-ndcg-test/tests/recipes/test_recipe_hygiene.py -q
E  gemma-4-12b/family.yaml:5: # (handover/specs/judge-catalog.md section 7, ...
E  ... (25 hits across the eight judge families) ...
E  assert not ['gemma-4-12b/family.yaml:5: ...', ...]
FAILED ...::test_every_shipped_recipe_file_reads_as_a_public_statement
```

After the cleanup: `2 passed`. The recipe suite on the changed tree:
`uv run --no-sync pytest rcp-ndcg-test/tests/recipes -q` — **295 passed, 228 skipped**. The goldens were
regenerated with `--update-goldens` (49 passed) and the diff is `notes`/`sources` only.

Item 4a, the failing-first sizeLimit:

```
uv run --no-sync pytest tests/runners/test_kubernetes.py -q -k dshm
E  assert {'medium': 'Memory'} == {'medium': 'Memory', 'sizeLimit': '16Gi'}
FAILED TestPhases::test_every_dshm_volume_is_bounded_by_the_declared_size
```

After the fix: `tests/runners/test_kubernetes.py` — **42 passed**. The test pins the default on the Job's own
pod and on a StatefulSet engine pod, the `dshm_size` override, and the refusal of a malformed quantity
(`ConfigError`).

Item 4b, the PodInitializing reproduction (the status/note path, with a Pending pod whose container waits):

```
pending
pod j-pod-0 container phase-1: PodInitializing
```

Root suite (heavy slot): `4136 passed, 155 skipped in 86.84s`. `tests/runners tests/contract tests/docs`:
`628 passed, 107 skipped`. `tests/docs tests/runners`: `611 passed, 107 skipped`. `ruff format --check`,
`ruff check` and `basedpyright` (0 errors) on the changed tree; `mkdocs build --strict` built.

The bare root venv's `rcp-ndcg-vllm/tests/models/topk/test_no_deps_install.py` wheel-build tests fail with
`401 Unauthorized` from the environment's extra package index (they need `uv build` to resolve `setuptools`);
the gate's own vllm steps run in the package's venv and are the authority. Unrelated to this lane's changes.

## 5. Open questions

1. **`dshm_size`'s default (`16Gi`) is this lane's call.** It matches the GPU harness's sizing (128Gi for eight
   engines, one engine per pod here) and is overridable per run; a phase that runs several local engines in one
   container needs a raised value, which the option and the docs say. If the owner wants another default, it is
   one field.
2. **The stale Dependabot alerts need a dismissal decision after the merge.** All 40 open ones are against
   `main`'s initial lock; the report gives the per-group dismissal reason. The one actionable finding is
   `pyjwt 2.14.0` (fixed in 2.15.0), reachable only through the optional `[azure]` extra.
3. **The recipes' `status` stays `unverified`** and the GitHub-CI checklist item stays unticked: both need the
   owner's GPU waves and a CI dispatch.

## Docs updated

- `docs/concepts/runs.md`: the Kubernetes bullet gains the bounded-`/dev/shm` sentence (and the sample
  Deployment manifest's dshm volume gains the same `sizeLimit`), and the `run status` sentence gains the
  `PodInitializing` note.
- `docs/reference/api/rcp_ndcg.runners.md`: regenerated (`rcp-ndcg docs api`) for the new field.
- Greps run before the last commit: the internal-reference grep over the eight judge family directories
  (`git grep -n -E "owner answer|open question|catalog's|E2 |R30|decision [0-9]|this lane|lane's|handover" --
  rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/<family>`, once per family);
  `git grep -n "dshm\|/dev/shm\|emptyDir" -- docs README.md skills examples experiments`; `git grep -n -i "kubernetes" -- docs`.

## CHANGELOG entry

Added inside the one `## 0.0.1` section:

- `### Public surface` — **`KubernetesOptions` gains `dshm_size`**: the `sizeLimit` of every pod's `/dev/shm`
  Memory `emptyDir` (`16Gi` by default; a Kubernetes memory quantity with a binary unit — `Ki`/`Mi`/`Gi`/`Ti` —
  refused otherwise). The rendered Job and engine-StatefulSet manifests carry it, so a runaway engine fails its
  pod instead of exhausting the node; a phase that runs several engines in one container raises it.
- `### Changed` — **Every Kubernetes pod's `/dev/shm` is bounded**: the Job's pod and every engine StatefulSet
  pod render the dshm Memory `emptyDir` with a `sizeLimit` (the declared `dshm_size`, `16Gi` by default) where
  it used to be unbounded.

No CHANGELOG entry for items 1-3: no public name, flag, exit code or schema changed (the judge goldens' notes
and sources are recipe data, not the pinned surface).

## Public surface changes

- `KubernetesOptions.dshm_size` (new field, default `16Gi`, pattern `^\d+(\.\d+)?(Ki|Mi|Gi|Ti)$`), rendered as
  the dshm volume's `sizeLimit`; `tests/contract/snapshots/python_api.json` and `schemas/run-config.v1.json`
  regenerated; `docs/reference/api/rcp_ndcg.runners.md` regenerated. No CLI, exit code or other schema changed.

## Files outside scope

- `handover/00-MASTER.md`, `handover/specs/judge-catalog.md`, `handover/RELEASE-CHECKLIST.md`,
  `handover/reports/dependabot-analysis.md` (the brief's own items 1-3 and the checklist update).
- `docs/concepts/runs.md`, `docs/reference/api/rcp_ndcg.runners.md`, `CHANGELOG.md`,
  `tests/contract/snapshots/python_api.json`, `schemas/run-config.v1.json` (item 4's public change).

## For the next lanes

- The hygiene guard has no judge scoping: any new shipped recipe file must read as a public statement, judge
  families included.
- The judge goldens moved (notes/sources only): a re-recording wave must use the new golden bytes.
- `KubernetesOptions.dshm_size` is the one knob for a pod's `/dev/shm`; a phase with several local engines
  needs it raised.
