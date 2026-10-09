# Report: recipe families (owner decision 34, 2026-10-08)

**Status:** DONE. The 19 standalone recipe directories become 13 family directories: one `family.yaml` (the shared
`role`/`input`/`client`/`serve`/`engine`/`reference`/`gates`/`notes` blocks plus a `variants` table of per-size facts),
ONE `reference.py`, one `template.jinja` and one `requirements-reference.txt` per family. Every variant id still
resolves to a full `Recipe` (the resolved recipe's JSON Schema is unchanged) and is served, contract-tested,
stage-1-tested and gated on its own. Family ids are never served.

## Commits

- `f3c908a3` The recipe-families spec lands in handover (owner decision 34) and the decision is recorded in the master notes
- `787e7ace` The pre-family golden contracts: every shipped recipe's resolved model dump, serve argv, client config and behaviour fingerprint, captured on the standalone-recipe tree
- `dc6c5986` The loader, consumers and fixtures move to recipe families (13 family directories resolve byte-identically, modulo the declared default drops; the harness passes the resolved recipe to the family references)
- `88a83bba` The recipe test modules become one module per family: the 19 variants resolve through their family directories, the harness's `--recipe` reaches every reference invocation
- `c6a6f45e` The per-variant golden guard: `golden/DELTAS.json` declares every accepted difference, `test_family_goldens.py` compares the resolved contract and fingerprint offline in every CI job
- `a0eb3ecf` The family conversion's surrounding surface: the wheel contract, the public-surface snapshot, the family schema export, the docs and NOTICE, the CHANGELOG
- `da344613` The merge's golden deltas (the workstream-09 note-only edits to the topk and pplx-late recipes are declared; no behaviour or fingerprint input moves)
- `3ce7357a` The golden guard's shrink-only check now fails when a declared difference is reverted; `DELTAS.json` is key-unique with corrected evidence
- `2c45386a` Round-1 verifier findings: the case backing resolves variant ids through the family index, the pairs validator and the plugin collector use the family directory, `changed_recipes` survives a broken family, every family reference reads the resolved recipe
- `ecc769be` Round-2 findings: the all-recipes wave list isolates a broken family, the pplx-late notes say the image path is sent, the three load-from-recipe references get parameterisation tests, stale citations and the dead recipes-root skip
- `1a098f07` Round-3 findings: the RC builder stages the recipes from the package-data path and the pairs from the harness home, the last two references take their checkpoint from the resolved recipe, `find_recipe` keeps a broken family's diagnostic, the job-script paths in the docs
- `5c69c2bc` The live handover specs follow the family layout (the recipe-fix root-cause research and the docs blueprints cite the family files, ids and paths)
- Merges into the lane: `26b3c24a` (rfc-0001, workstream 09), `345beab0` (rfc-0001, workstream 08 A + MRL cards), `1fb4f0e0` (rfc-0001, sync-hardening)

## What changed (per brief item)

- **One family directory per model family** — 13 families hold the 19 variant ids the tree shipped before the
  conversion (no id added, none dropped): ctxl (1b/2b/6b), jina-embeddings-v5-text (small), jina-reranker-v3, octen
  (8b), pplx-embed-v2-context (9b-preview), pplx-embed-v2-late (0.6b), qwen3-embedding (0.6b), qwen3-reranker
  (0.6b/4b/8b), qwen3-vl-embedding (2b), qwen3-vl-reranker (2b), topk-embed-v1 (small), zembed-1, zerank
  (1/1-small/2). A single-size model is a family with one variant (uniform; no second loader path).
- **`family.yaml` + a `variants` table** — the variant rows carry only per-size facts (`model`, `revision`, and the
  declared override fields: `resources`, `engine` limits, `client` lengths/dims, per-size `notes`, `status`); a
  variant override of a shape or content field is refused naming the field, and a variant-level `client.tokenizer`
  override is refused (it is injected as `model@revision`).
- **One reference per family, `--recipe` on the subprocess contract** — every reference invocation (stage 1 and
  stage 2) receives the resolved recipe JSON, so the one reference serves every variant; all 13 references now take
  their checkpoint (model, revision) from it rather than from module constants.
- **Every consumer takes variant ids** — `serve_argv` / `rcp-ndcg-vllm serve <variant-id>`, `recipe: <variant-id>`,
  the catalog (`iter_recipes` returns every variant of every family), the harness's discovery, the wave lists, the
  case backing, `observe.requests` (one pairs file per variant) and the plugins collector; `load_recipe` takes a
  variant id (or a single-variant family path) and `resolve_recipe` refuses a family id with a hint.
- **Byte-identical resolved contracts** — the per-variant goldens (`tests/recipes/golden/`) were captured on the
  pre-family tree and the conversion is compared against them offline in every CI job; `DELTAS.json` declares every
  accepted difference (the product defaults a standalone recipe spelled out -- `request_shape: text`,
  `listwise: false`, `add_special_tokens: {pair: true}` -- and the shared template file's header comment), with a
  shrink-only check so a reverted difference fails.
- **One contract module per family, parametrized over its variants**, with two mutants red per family; the
  network-gated stage-1 suite runs per variant.
- **Tests, docs, schema and CHANGELOG** — the family file format has its own exported schema
  (`rcp-ndcg-vllm/schema/family.schema.json`); the resolved `recipe.schema.json` is byte-identical to the pre-family
  one; the docs (`docs/reference/recipes.md`, `docs/how-to/add-a-model.md`, the package README, the recipes README),
  the NOTICE and the CHANGELOG entry are current.
- **The layout-move debt the family tree exposed** — the RC builder staged `rcp-ndcg-vllm/recipes` (a path that has
  not existed since the layout move) and `rcp-ndcg-vllm/pairs` (the pairs live in the harness package); both are
  fixed with a regression test, and the job-script paths in the docs follow.

## Verification

**Round 1 — two independent verifiers, fresh context, DeepSeek-V4.1-flash (:xhigh)** (lens A correctness, lens B
regressions and hygiene). Verdicts: FAIL / FAIL. Findings and actions (all fixed in `2c45386a`):

- the case backing and the pairs validator resolved a variant id through the recipes root, not the family index;
  the plugin collector used `<variant-id>/<file>`; `changed_recipes` aborted on a broken family; four references
  ignored the `--recipe` they were given — all fixed, each with a test.
- one finding was wrong as stated (the golden comparison's tolerance was claimed to be loose; the guard compares the
  resolved dump exactly and the shrink-only check was added on top instead).

**Round 2 — one fresh confirmation verifier** (lens A+B), verdict FAIL, on the merged tree:

- F1 (major): `load_wave([], root)` raised a `RecipeError` out of the all-recipes expansion instead of marking the
  broken family failed — fixed in `wavelist.py` with `test_the_all_recipes_wave_list_isolates_a_broken_family`
  (mutation-red shown by the verifier).
- F2 (major): the pplx-embed-v2-late notes still claimed the media path was an open gap; they now state the
  workstream-09 rule (media sent, vectors kept whole, `skip_unapplied`), `DELTAS.json` regenerated.
- F3-F6 (minor): the three references that read the recipe got offline parameterisation tests that fail on the
  pre-fix code; the stale path citations, the `find_recipe` diagnostic and the dead recipes-root skip were fixed.

**Round 3 — one fresh confirmation verifier** (lens A+B), verdict FAIL on the merged tree:

- Confirmed every round-1/2 fix with mutation-red evidence and confirmed the brief's items (13 families / 19
  variants, family ids never served, resolved schema byte-identical, docs and CHANGELOG current, no test weakened).
- F1/F2 (major, pre-existing layout-move debt in a file this lane had touched): `rc_build.sh` staged the recipes
  from `rcp-ndcg-vllm/recipes` (missing; under `set -euo pipefail` the build aborts) and the pairs from
  `rcp-ndcg-vllm/pairs` (missing; the pairs live in the harness package) — both fixed with a regression test that
  reads the source path out of the script and requires it to exist (red on the pre-fix script).
- F3 (minor): the job-script paths in the docs — fixed (`release-candidates.md`, `add-a-model.md`, `runs.md`,
  `changes.py`'s example).
- F4 (minor): `find_recipe`'s new first-failure branch had no test and its docstring no longer held — fixed with
  `test_find_recipe_reports_a_broken_family_not_an_unknown_id` (red on the pre-fix code) and a reworded docstring.
- F5 (minor): the collector/runner prose still said `<recipe-id>/<file>` — fixed to `<recipe-directory>/<file>`.
- F6 (minor): octen and pplx-context ignored `--recipe` for their checkpoint — both now load the resolved recipe's
  model and revision, with offline tests that fail on the pre-fix code.

The three rounds are the protocol's budget; the round-3 findings are all fixed and the final state is gated, but no
fresh verifier has seen the round-3 fixes (see Open questions).

## Checks

- `bin/gate lane/rfam` at `345beab0` — **GATE: PASS** (pytest 3269 passed; contract-docs 287; test-pkg 601;
  recipes: no failure outside the baseline; vllm-models 70).
- `bin/gate lane/rfam` at `ecc769be` — **GATE: PASS** (pytest 3269 passed; contract-docs 287; test-pkg 602;
  recipes green; vllm-models 70; leaderboards 1022 checks / 987 match / 35 known deviations / 0 failed).
- `bin/gate lane/rfam` at `1a098f07` — **GATE: PASS** (ruff, format, basedpyright 0 errors; pytest 3269 passed;
  contract-docs 287; mkdocs strict; test-pkg 604 passed; recipes green; public-names clean).
- `bin/gate lane/rfam` at `911931da` (the tip, this report included, after the sync-hardening merge) — **GATE:
  PASS** (ruff, format, basedpyright 0 errors; pytest 3388 passed, 96 skipped; contract-docs 294 passed; mkdocs
  strict; test-pkg 604 passed; recipes: no failure outside the baseline; vllm-pkg 9; vllm-models 71; leaderboards
  1022 checks / 987 match / 35 known deviations / 0 failed; public-names clean).
- The gate's synced venv is authoritative. In the lane's own worktree (an unsynced venv predating the merged lock)
  the dataset-reading tests fail with `ConfigError: unknown dataset URI scheme 'jsonl'`; the same tests pass in the
  gate. This is environmental and was not "fixed" by changing code.
- `uv run --no-sync pytest rcp-ndcg-test/tests/recipes -q -n 4` with `RCP_NDCG_NETWORK_TESTS=1` and the shared
  tokenizer cache — 256 passed, 1 skipped (the skip is the recipes-root file's network gate for a family whose
  tokenizer is not cached; it runs in the gate).
- `uv run --no-sync pytest rcp-ndcg-test/tests/recipes/test_family_goldens.py` — 24 passed.
- `uv run --no-sync ruff format --check . && uv run --no-sync ruff check .` — clean; `uv run --no-sync basedpyright`
  — 0 errors; `uv run --no-sync mkdocs build --strict` — built.
- Mutation evidence: every fix in rounds 1-3 has a test that was shown red on the pre-fix code (`git show
  <rev>:<path>` shadow copies; the verifiers reproduced several independently).

## Open questions

1. **Decision 34 is recorded twice, and the two texts differ.** The master notes carry the 2026-10-08 entry (this
   lane's spec: `family.yaml` with a `variants` table) and a 2026-10-09 entry that reads "`family.yaml` + one variant
   file per checkpoint". This lane implements the 2026-10-08 form, which its brief named. If the newer wording is the
   target, the follow-up is small and local: keep the family blocks in `family.yaml` and move each variant's
   overrides into `variants/<id>.yaml`, with `load_family` reading either form and the same deep merge and refusals.
   The owner should say which text stands.
2. **The spec's "Variants to carry" table lists sizes this lane did not add** (qwen3-embedding 4b/8b, qwen3-vl 8b,
   octen 0.6b/4b, jina nano, topk xsmall, pplx-embed-v1, pplx-embed-v2-late 9b). The lane's brief scoped the
   conversion to the ids the tree shipped; the new sizes need their own GPU validation and goldens, so they are a
   separate lane. The family layout makes each one a table row plus a test parametrisation.
3. **No fresh verifier has seen the round-3 fixes** (the protocol's three rounds are spent). The fixes are covered
   by the gate and by red-first tests, but a further confirmation round is the owner's call.
4. **The completed lanes' handover material still cites pre-family paths** (their reports and specs are historical
   records; the live blueprints and the recipe-fix root-cause research were updated to the family files, ids and
   line numbers). The recipe-fix lane should read the updated citations, not the old reports.
5. **The local worktree's venv is stale** relative to the merged lock (missing the reader/writer entry points); it
   should be re-synced before anyone runs the dataset tests locally.

## CHANGELOG entry

Under `## Unreleased` → `### Public surface` (exact text):

> - **Recipe families** (owner decision 34: one family, many sizes, every size its own tested recipe id):
>   the shipped recipes are family directories -- `rcp_ndcg_vllm/recipes/<family>/family.yaml` (the shared
>   blocks plus a `variants` table of per-size facts), the family's ONE `reference.py`, its one chat template
>   and its `requirements-reference.txt`. Every variant resolves to a full `Recipe` (the resolved recipe's
>   JSON Schema is unchanged) and every consumer takes variant ids: `rcp-ndcg-vllm serve <variant-id>`,
>   `recipe: <variant-id>` in `rcp-ndcg`, the catalog, the harness's discovery, the wave lists and
>   `observe.requests` (one pairs file per variant); the RC builder stages the recipes from the package-data
>   path the layout move created and the pairs from their harness home, and every family reference takes its
>   checkpoint from the resolved recipe it is passed. A family id is never served. `rcp_ndcg_vllm.recipe` gains
>   `Family`, `Variant`, `load_family`, `resolve_recipe` and `iter_families`; `load_recipe` takes a variant id
>   (or a single-variant family path), `iter_recipes` returns every variant of every family, and the family
>   file format has its own exported schema `rcp-ndcg-vllm/schema/family.schema.json` beside
>   `recipe.schema.json`. The reference subprocess contract gains `--recipe <resolved-recipe.json>` (the
>   harness passes the resolved recipe it loaded), so one family reference runs every variant.
> - The standalone `recipe.yaml` path is gone: a directory without `family.yaml` is refused with a hint, and a
>   variant-level override of `client.tokenizer` (injected as `model@revision` unless the family declares one)
>   is refused naming the field.

Under `### Changed`:

> - **The 18 standalone recipe directories become 13 families / 19 variants** (decision 34): the resolved
>   contracts are byte-identical to the pre-family tree except where a variant's standalone recipe declared a
>   product default the family now omits (`request_shape: text`, `listwise: false`,
>   `add_special_tokens: {pair: true}` -- the product's endpoint model resolves each to the same value), and
>   where the family shares ONE template file whose pre-family per-size copies differed only in their jinja
>   comment headers (the renders are byte-identical; the fingerprint's `template_file` input moves, declared in
>   the conformance waivers). Every variant keeps its own contract test (one module per family, parametrized
>   over its variants, two mutants red per family), its stage-1 network tests and its pairs file, and the
>   per-variant goldens (`rcp-ndcg-test/tests/recipes/golden/`) pin the resolved contract and fingerprint in
>   every CI job (offline; `--update-goldens` regenerates on purpose).

## Public surface changes

- New names in `rcp_ndcg_vllm.recipe`: `Family`, `Variant`, `load_family`, `resolve_recipe`, `iter_families`
  (`__all__` snapshot updated).
- `load_recipe` takes a variant id; a family directory without `family.yaml`, a family id passed where a variant id
  is required, a variant override of a shape/content field and a variant-level `client.tokenizer` are typed errors.
- New exported schema `rcp-ndcg-vllm/schema/family.schema.json`; `schemas/recipe.schema.json` byte-identical.
- The reference subprocess CLI contract gains `--recipe <resolved-recipe.json>` for every shipped reference.
- No CLI command, flag or exit code changed; the command tree snapshot is unchanged.

## Files outside scope

- `rcp-ndcg-test/src/rcp_ndcg_test/jobs/rc_build.sh` (the staged recipes and pairs paths; a two-line fix plus the
  comment), `jobs/plugins.py` and `jobs/run_wave.py` (prose), `jobs/wavelist.py` (the all-recipes expansion),
  `e2e.py` (`find_recipe`), `changes.py` (a docstring example), `tests/test_jobs_scripts.py`,
  `tests/test_e2e_scenarios.py`, `tests/test_plugins.py`, `tests/test_record_and_wave.py` (the tests that pin the
  above), `docs/how-to/release-candidates.md`, `docs/how-to/add-a-model.md`, `docs/concepts/runs.md`,
  `handover/specs/recipe-root-causes.md`, `handover/specs/docs-release-blueprint.md`,
  `handover/specs/docs-site-blueprint.md` (paths, ids and line numbers that the family layout invalidated).

## Docs updated

- `docs/reference/recipes.md` — the family layout, the catalog columns and the loader names.
- `docs/how-to/add-a-model.md` — the family directory walkthrough and the field-by-field guide.
- `docs/concepts/runs.md` — the RC builder's home.
- `docs/how-to/release-candidates.md` — the job-script and stage paths.
- `rcp-ndcg-vllm/README.md`, `rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/README.md` — the family tables and layout.
- `rcp-ndcg-test/README.md` — the loader id form.
- Greps run before the last commit: `git grep -n "recipe\.yaml" -- docs README.md REPRODUCIBILITY.md skills
  examples experiments mkdocs.yml`, `git grep -n "rcp-ndcg-vllm/recipes\|rcp-ndcg-vllm/jobs\|rcp-ndcg-vllm/pairs"
  -- docs README.md skills examples experiments mkdocs.yml`, `git grep -n "18 recipes\|18 canonical\|the 18 " --
  docs README.md skills examples mkdocs.yml` (all empty), plus the same greps over `handover/specs/` (the live
  blueprints fixed; the completed lanes' reports and briefs are historical records and were left as written).

## For the next lanes

- The recipe-fix lane: the root-cause research now cites the family files with current line numbers; the recipes'
  shared blocks are in `family.yaml` and a per-size change goes in the variant row (or the family block when it is
  shared), never in a per-variant copy.
- The GPU lanes: `rc_build.sh` stages the recipes from the package-data path and the pairs from the harness home;
  the wave lists, the case backing and the plugin collector all take variant ids.
- The docs lane: the catalogue is 13 families / 19 variant ids; the family file format has its own schema page.
- If decision 34's 2026-10-09 wording ("one variant file per checkpoint") is the target, the change is the
  `variants/<id>.yaml` split described in Open questions 1; nothing else in this lane's surface moves.
