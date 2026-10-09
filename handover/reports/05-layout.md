# Lane l05 — the layout move (workstream 05, with owner decisions 17-22)

**Status: DONE.** Branch `lane/l05`, HEAD `4e030a26`, base `rfc-0001` merged at `0db46130` (media-inputs
and ci-fix; the merge commits are `eff66062` and the follow-up `94a8afdf` + `2931ebc6`). The tree is
clean (`git status --porcelain --untracked-files=all` prints nothing), `tools/layout_move.py --check`
exits 0, and every gate of the brief and MASTER §2 passes on the merged HEAD.

## Commits (lane stack, in order)

| Commit | Subject |
|---|---|
| `a682e015` | The layout move, mechanically: four distribution directories, the workspace root manifest, every path rewritten |
| `c2a6aee4` | rcp-ndcg-vllm becomes the lean serving package (layout-move item 2) |
| `6f7e7cf5` | The topk and pplx model plugins fold into rcp_ndcg_vllm.models (layout-move item 3) |
| `99838e32` | The validation tooling and the verified emulators move to rcp-ndcg-test, unpublished (layout-move item 4, decision 20) |
| `47a4e10b` | rcp-ndcg reads recipes optionally, through the versioned recipe contract (layout-move item 5, decisions 17 and 18) |
| `baa521a4` | The recipe schema keeps the engine-neutral client block strictly apart from serve/engine (decision 19) |
| `162f5940` | rcp_ndcg.llm is renamed to rcp_ndcg.judging (owner decision 21) |
| `de2b2855` | rcp-ndcg-test is installed from a git subdirectory (owner decision 22) |
| `cdab7978` | The release workflow builds and publishes three dists in order core -> rcp-ndcg -> vllm (item 6); one merged NOTICE byte-identical in four distributions |
| `b6f34124` | tests/contract pins the public rcp_ndcg_vllm names (docs-release Q3) |
| `a0241255` | The root README is the short landing card (docs-release Q1) |
| `f9f842d9` | Regenerate the lock and the constraints file for the four-member workspace (release gate) |
| `9d9ff334` | CI jobs adapted to the new paths (the layout move's last hand edit) |
| `16718cb5`, `0cd2a589`, `c580e41a`, `a237d0b9` | Small follow-ups (snippet schema_version, an escape-sequence warning, the schema re-export, formatting) |
| `eff66062` + `94a8afdf` | Merge rfc-0001 (`0db46130`: media-inputs, ci-fix, rec-pplx-late) — rename detection carried their edits into the moved files; each merged lane's edits verified landed (media gate, video/interleaved rows, ci-fix's CI green pass, the 19th recipe pplx-embed-v2-late-0.6b and its plugin architecture) |
| `2931ebc6` | Type-checking stays at the move's former scope; the merge's reverted CI files restored |
| `4e030a26` | The verifier pair's round-1 findings (see Verification) |

## What changed (per brief item)

- **Mechanical move (item 1)**: `tools/layout_move.py` (from `wip/layout-script`, tables extended: the
  `dependency-groups` split, the gitignore build globs) performs 7 `git mv`s (root package → `rcp-ndcg/`,
  `packages/rcp-ndcg-{core,vllm,test}` → top level) and the declared path rewrites (CI, release.yml,
  AGENTS.md, docs, mkdocs, tests, snapshots, constraints, MANIFESTs, the lock); the root manifest splits
  into the package manifest + the workspace-only root with the four members; `--check` passes on HEAD
  (`test_layout_move.py` runs the real script on a fixture repo, idempotency, refusal and stale-scan).
- **Lean `rcp-ndcg-vllm` (item 2)**: deps pydantic + PyYAML only; recipes are package data read through
  `importlib.resources`; `rcp-ndcg-vllm serve <id> [--dry-run]` for any role; the recipe's `client` block
  is plain data the product validates when it reads it; the harness modules that stayed read the plain
  dict and validate templates through the product's `TemplateSpec` (R30).
- **Plugins fold (item 3)**: one `vllm.general_plugins` entry point (`rcp_ndcg_vllm.models:register`),
  one version guard (`models/version_guard.py`), lazy `module:Class` registrations; the plugin
  distributions deleted; the families' fixes carried (the topk `load_weights` override + mapper, the pplx
  plugin-registered config class `hf_config.py`, and the merged late-interaction architecture).
- **Tooling + emulators + corpora to `rcp-ndcg-test` (item 4, decision 20)**: the equivalence harness,
  the recorder, the wave jobs, the request generator, the T4 driver, the fingerprint/change tooling, the
  recipe and harness tests; `rcp_ndcg.testing.engines`/`corpus` → `rcp_ndcg_test.engines`/`corpus` with
  the corpora at `rcp-ndcg-test/corpora/` and the conformance suite in its tests; the product keeps
  `rcp_ndcg.inference.fake` + the offline helpers, routing `fake://<engine>-<version>/<recipe>` URLs
  through the new `rcp_ndcg.fake_transports` entry-point group (typed refusal naming rcp-ndcg-test
  without a provider).
- **`recipe: <id>` resolution (item 5, decisions 17-18)**: `rcp_ndcg.inference.recipes` reads
  rcp-ndcg-vllm's recipe data lazily (CONTENT equal-or-refused naming both values, RUNTIME the config's);
  `--retriever/--reranker recipe:<id>` shorthands; every recipe carries `schema_version`, the exported
  JSON Schema pins it, and rcp-ndcg refuses versions it does not read (`RECIPE_SCHEMA_VERSIONS`) — no
  lockstep pin (rcp-ndcg-vllm declares no sibling; core and rcp-ndcg keep theirs). The paper configs
  point at their recipes where the strict rule allows (ctxl-*, octen).
- **Decision 19**: the client block refuses engine-specific keys (a named list of serve/engine/resources
  keys); the shipped recipes' client blocks carry no `recipe:` key any more (the pointer owns the key).
- **Decision 21**: `git mv rcp_ndcg/llm rcp_ndcg/judging` (+ tests/llm → tests/judging); every import,
  docstring, docs page and the layering charter follow; names unchanged.
- **Decision 22**: rcp-ndcg-test's README and `docs/reference/rcp-ndcg-test.md` give the
  `git+...#subdirectory=rcp-ndcg-test` install line.
- **Release workflow (item 6)**: three dists built one `--package` per member; publish order
  core → rcp-ndcg → vllm (needs chain); no plugin wheels; one merged NOTICE byte-identical (with
  LICENSE) in all four distributions.
- **Contract pins (Q3)**: `rcp_ndcg_vllm.recipe` in PUBLIC_MODULES, `vllm_cli.json` (build_parser-driven),
  packaging.json records the vllm project's script + entry point; rcp-ndcg-test unpinned (Q2).
- **Root README (Q1)**: the landing card; the full README lives at `rcp-ndcg/README.md`.
- **Lock + constraints**: regenerated for the four-member workspace; the constraints export gains
  `--package rcp-ndcg` (one home in `check_constraints.py`).
- **CI + gate table**: the vllm-recipes job installs the members the package exercises; the plugin job
  becomes the folded-models job (`tests/models`); DEPENDENCY_GATES/test_packaging assertions follow; the
  nightly is untouched.

## Verification

Round 1 — two fresh GLM-5.3-flash (`:xhigh`) verifiers, one per lens:

- **Lens A (correctness): FAIL** — 3 blockers (stale `packages/rcp-ndcg-vllm` duplicates the merge
  resurrected; the merged vllm tests at pre-fold spellings red at HEAD; the late recipe's
  `serve.plugin` naming the deleted `rcp-ndcg-vllm-pplx` distribution, failing `serve --dry-run` from
  the installed wheel), 5 majors (test_no_torch's stale LAZY list, register() rename, AGENTS/CI prose,
  CHANGELOG's plugin-wheel entry, layout_move's unanchored helper patterns), 2 minors. All
  reproductions anchored to HEAD via `git show`.
- **Lens B (regressions/hygiene): FAIL** — the same blocker set from the hygiene side, plus: the two
  shipped refusals (schema_version compatibility, fake_transports no-provider) had **no test** (mutants
  survived the full suite — shown by scratch mutation), 7 minor docs-path findings. R30/one-home/drift
  checks green; scope creep none.

Fix round (`4e030a26` + `2931ebc6`): every agreed finding fixed — the six stale files git-rm'd, the late
recipe serves through the folded plugin (its pinning test, notes and docs follow), the merged vllm test
files fixed (guard home, `register_pplx`, `N_RECIPES = 19`, the one `LAZY_MODEL_MODULES` home),
`layout_move --check` passes again (the landing-card README modelled, the helper patterns
boundary-anchored, `handover/` scaffolding skipped, the notice-pattern file's old-spelling-as-data
exempted), AGENTS/CI/CHANGELOG prose updated, the two refusals tested (red first, then green), the
minor docs paths fixed, and the merge-reverted CI files (`check_constraints.py`'s `--package` flag, the
release.yml build lines, the folded-models job) restored.

Round 2 — one fresh confirmation verifier (lens A+B) launched over the committed HEAD; its gate
re-runs are quoted in the final checks below if it completes before this report is filed.

## Checks (commands run last, and their results)

- `bin/heavy uv run --no-sync pytest tests/ -q -n 8 -p no:cacheprovider` → **3182 passed, 82 skipped**
- `uv run --no-sync pytest tests/contract tests/docs -q -p no:cacheprovider` → **270 passed, 52 skipped**
- `uv run --no-sync pytest rcp-ndcg-test/tests -q -p no:cacheprovider` → **570 passed, 225 skipped**
- `uv run --no-sync pytest rcp-ndcg-vllm/tests -q -p no:cacheprovider` → **62 passed, 11 skipped**
- `uv run --no-sync basedpyright` → **0 errors**; `ruff format --check .` / `ruff check .` → clean
- `uv run --no-sync mkdocs build --strict -d /root/repos/rcp-ndcg-lanes/l05/scratch/site` → OK
- `python tools/layout_move.py --check` → **exit 0, "no old path anywhere"**
- Wheels: `uv build --package rcp-ndcg / -core / -vllm` (6 files); fresh-venv installs import cleanly
  (core alone, rcp-ndcg after core, vllm); `pip install --no-deps rcp-ndcg-vllm` into a pydantic+PyYAML
  venv changes `pip freeze` by exactly the wheel line; `importlib.resources` lists **19** recipes from
  the installed wheel; `serve <id> --dry-run` exit 0 for all 19 from the installed wheel
- NOTICE + LICENSE sha256-identical in all four distributions
- `python3 .github/scripts/check_constraints.py` → 107 pins agree; `uvx twine check dist/*` → PASSED;
  versions 0.0.1 in all four pyprojects; `rcp-ndcg` pins `rcp-ndcg-core==0.0.1`; vllm names no sibling
- `RCP_EXPERIMENTS_DATA=/root/repos/rcp-ndcg-lanes/data uv run --no-sync python experiments/run_all.py` →
  **1022 checks, 987 match, 35 known deviations, 0 failed; 67/67; 82/82** (unchanged)
- `git status --porcelain --untracked-files=all` → empty
- Merged `rfc-0001` at `0db46130`; `bin/gate lane/l05` NOT run by the lane (shared/serialized; the
  operator's queue runs it on merge) — the local quality bar above is the full §2 bar.

## Docs updated

- `docs/reference/recipes.md`, `docs/how-to/add-a-model.md` (recipe YAML snippet carries
  `schema_version`; media/serve prose at the moved paths), `docs/how-to/use-verified-fake-engines.md`
  (corpora at `rcp-ndcg-test/corpora/...`), `docs/reference/rcp-ndcg-test.md` (git-subdirectory
  install), `docs/quickstart.md`/`docs/index.md`/`docs/concepts/*.md` (judging, moved paths — via the
  mechanical sweep), `rcp-ndcg-vllm/README.md`, `rcp-ndcg-test/README.md` (tooling sections + the
  subdirectory line), root `README.md` (landing card), `REPRODUCIBILITY.md`, `AGENTS.md` (layout,
  layering, releasing), `mkdocs.yml` (nav). Greps run: `git grep -n -i "rcp_ndcg.llm"`,
  `"packages/rcp-ndcg"`, `"src/rcp_ndcg"`, `"rcp_ndcg.testing.engines"`, `"rcp-ndcg-vllm-pplx"`,
  `"tests/contract/engines"` over docs/, README, AGENTS.md, skills/, examples/, experiments/, mkdocs.yml
  — all clean outside CHANGELOG history and `handover/`.
- `tests/docs` (links, navigation, snippets) green.

## CHANGELOG entry

See `CHANGELOG.md` `## Unreleased` → `### Public surface` (the layout move, `recipe: <id>`, the
versioned recipe contract, the inference recipe-resolution exports, the judging rename, the
rcp-ndcg-test subdirectory, the release order + NOTICE) and the pre-existing entries for the merged
lanes.

## Public surface changes

- New: `rcp_ndcg.inference.recipes` (`available_recipe_ids`, `expand_role_recipe`, `recipe_client_data`,
  `recipe_role`, `shorthand_config`, `RECIPE_SCHEMA_VERSIONS`) re-exported from `rcp_ndcg.inference`;
  the `recipe: <id>` mapping form + CLI shorthands; `rcp_ndcg.fake_transports` entry-point group;
  `rcp_ndcg_vllm.recipe` public names (Recipe, load_recipe, iter_recipes, serve_argv), the
  `rcp-ndcg-vllm` console (`serve`, `--dry-run`), the recipe schema's `schema_version` field;
  `rcp_ndcg.judging` (renamed from `rcp_ndcg.llm`).
- Removed: `rcp_ndcg.testing.engines`/`corpus` (→ rcp-ndcg-test), the plugin distributions
  (`rcp-ndcg-vllm-topk`, `rcp-ndcg-vllm-pplx` wheels), `rcp_ndcg_vllm`'s harness/record/jobs modules
  (→ rcp-ndcg-test), rcp-ndcg-vllm's dependency on rcp-ndcg.
- Snapshots regenerated: `python_api.json` (judging paths, the vllm recipe module, the recipe schema),
  `packaging.json` (`rcp_ndcg.fake_transports`, the vllm project), new `vllm_cli.json`.

## Files outside scope

- `tools/layout_move.py`, `tests/test_layout_move.py` (the brief's own script + test — new files).
- `experiments/paper/rerankers/*.yaml`, `experiments/paper/retrieval/octen.yaml` (decision 17's paper
  configs).
- `.github/scripts/check_constraints.py` (the `--package rcp-ndcg` flag), `.github/workflows/ci.yml`
  and `release.yml`, `.gitignore`, `uv.lock`, `requirements-constraints.txt`.
- The four `NOTICE`/`LICENSE` distribution copies and `rcp-ndcg/README.md` (the moved long README).
- `handover/reports/05-layout.md` (this report; the workstream's required output).

## Open questions

1. **`bin/gate lane/l05`** is shared/serialized and not run by the lane itself; the local bar above is
   the full §2 quality bar on the merged tree. The operator should run the gate (or merge and dispatch
   CI) — the tree is ready.
2. **`rcp_ndcg.emulators` entry-point group**: the moved `rcp_ndcg_test.engines` registry still loads
   out-of-tree emulators through it; nothing in-repo registers there. Harmless, but a private repo's
   integration will exercise it first.
3. **`_probe_infeasible`/`_offline_probe` duplication** (MASTER §9 QA item "two padding helpers by
   design"): untouched, as the register says ("confirm or unify" is workstream 07's call).
4. **The staged wheelhouse sdist contents**: `rcp-ndcg-test/MANIFEST.in` grafts `pairs`/`scenarios` and
   prunes `corpora` (re-recorded per wave); if the node bootstrap should ship corpora in the sdist, that
   is a one-line change for the GPU lanes to request.
5. **The NFS-backed TMPDIR flake**: `_validate_and_prune`'s tempdir cleanup can fail with ENOTEMPTY on a
   network-backed TMPDIR (~1 in 20 locally); the rfc-0001 merge brought
   `TemporaryDirectory(ignore_cleanup_errors=True)` for the media stage, and stage-1's loop now inherits
   the same guard on this tree. Local runs with `TMPDIR=/tmp` never reproduced it.

## For the next lanes

- Workstream 08 (vLLM-only + judge recipes) builds on this tree: recipes are package data under
  `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/`; the 19th recipe (pplx-embed-v2-late-0.6b) and its plugin
  architecture are merged; judge recipes carry `role: judge` and are validated by the product's judge
  config per decision 15.
- Workstream 09 (processing pipeline) should run right after — the module paths are now frozen
  (`rcp_ndcg.judging`, `rcp_ndcg.inference.recipes`).
- The `recipe:` pointer in a client block is now reserved (decision 17's mapping form); recipe authors
  must not put a free-text note there — the shipped recipes keep their prose as a comment above
  `client:`.
- `tests/conformance` moved to `rcp-ndcg-test/tests/conformance`; the release checklist's "stale.json
  empty" requirement points at `rcp-ndcg-test/tests/conformance/stale.json`.
