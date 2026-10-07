# Workstream 05: apply the layout move (the quiet window)

Read `handover/00-MASTER.md` first. Spec: `handover/specs/layout-move.md` (items 1-6, binding), owner answers
`docs-firstcontact-answers.md` (Q1 `recipe:<id>`, Q2 `serve: command`), `docs-release-answers.md` (Q1 README placement,
Q2 rcp-ndcg-test, Q3 public `rcp_ndcg_vllm` names, Q4 one NOTICE), `docs-site-answers.md` (OQ-1, OQ-6, OQ-9).

Run this only after workstreams 01-04 have landed on `rfc-0001` (nothing else should be merging meanwhile).

## Target
```
rcp-ndcg-core/   pyproject.toml, src/rcp_ndcg_core/
rcp-ndcg/        pyproject.toml, src/rcp_ndcg/            (the full README lives here: the PyPI long description)
rcp-ndcg-vllm/   pyproject.toml, src/rcp_ndcg_vllm/{recipe, serve, recipes/ (package data), models/<name>/}
rcp-ndcg-test/   pyproject.toml, src/rcp_ndcg_test/{cases, conformance, fakes, equivalence, record, jobs}, cases/
pyproject.toml   the uv workspace only; docs/, experiments/, examples/, skills/, schemas/, .github/, a short README card
```

## What `wip/layout-script` holds (base `5359655`)
- `a0d8f9e` `tools/layout_move.py`: the mechanical part as an idempotent, rule-based transform (a declared table of
  `git mv`s and old-prefix -> new-prefix rewrites over every tracked text file; `--check` mode exits 0 only when no old
  path remains), with a test on a fixture repo, plus `a979363` (three reflowed lines).
- The hand-edit stack, one commit per item (keep them free of mechanical moves so they rebase onto a fresh script run):
  `598daca` lean `rcp-ndcg-vllm` (item 2: deps pydantic + pyyaml; recipes as package data via `importlib.resources`;
  `rcp-ndcg-vllm serve <id> [--dry-run]`), `b8d4689` (id injection order on load; the vendored card script stays
  byte-identical), `149bdf4` (recipe tests on the product where the deleted helpers stood — R30-compliant), `25300e8`
  topk and pplx plugins folded into `rcp_ndcg_vllm/models/` under one lazily registering `vllm.general_plugins` entry
  point (item 3), `cf9ad9f` validation tooling moved to `rcp-ndcg-test` (item 4), `9e63ccd` `recipe: <id>` resolved
  through `rcp-ndcg-vllm` when installed, a typed error with the install line otherwise (item 5), `cc0794e` release
  workflow: three dists in order core -> rcp-ndcg -> vllm (item 6), `1e5be6d` one NOTICE byte-identical in all four
  distributions, `acef50d` `tests/contract` pins the public `rcp_ndcg_vllm` names (Q3), `cf61314` the root README as a
  landing card (Q1), `268c494` lock + constraints regenerated, `0701de0` docs truth for release/wheelhouse prose,
  `ba7c315` `.gitignore` for the four build dirs.
- A replay rehearsal on an older tip: the script ran clean; 6 of 14 hand-edit commits cherry-picked clean, the
  conflicts concentrated in the three big moves (lean vllm, plugins fold, tooling move) and their cascade.
- Its first review round was running when it stopped: review the stack yourself before applying it.

## Procedure
1. On a fresh branch from the current `rfc-0001` tip: run `python tools/layout_move.py` (take the script from
   `wip/layout-script`; extend its tables for any path that appeared since — the `--check` mode tells you), commit the
   result as one mechanical commit.
2. Cherry-pick the hand-edit stack in order; resolve conflicts against what landed since (recipes changed by the
   families, the harness gaps, the corpus/fakes/e2e code, `rcp-ndcg-test` contents). The plugins fold must carry the
   families' plugin fixes (topk bias init, pplx config registration). The harness keeps driving the product's role
   clients (R30): never restore deleted helpers.
3. Dependencies: `rcp-ndcg-vllm` must not depend on `rcp-ndcg` (the harness that needed it moved to `rcp-ndcg-test`);
   `rcp-ndcg-test` depends on `rcp-ndcg` and `rcp-ndcg-vllm`. Fix CI's package jobs (the `vllm-recipes` job installed the
   checkout's `rcp-ndcg` beside the package only because of that pin), the plugin-suite job (plugins are now inside the
   package), `tests/docs/test_packaging.py`'s gate table, the nightly workflow.
4. Gates for this workstream (in addition to the MASTER quality bar, adapted to the new paths): a fresh-venv install of
   each built wheel; `pip install --no-deps rcp-ndcg-vllm` in a venv holding only pydantic and PyYAML changes
   `pip freeze` by exactly that wheel; `rcp-ndcg-vllm serve <id> --dry-run` for every recipe; `importlib.resources`
   lists the recipes from the installed wheel; NOTICE byte-identical in all four distributions; `run_all` unchanged;
   `release.yml`'s build/check steps pass locally (`uv build --package ...`, versions, pins, constraints, `twine check`).
5. Merge into `rfc-0001`, quality bar, push, CI. Report in `handover/reports/05-layout.md`.

## Amendments after M3 (binding)
- **Base**: start from the M3 tip of `rfc-0001` (everything integrated). `wip/layout-script` was cut from an older base:
  take `tools/layout_move.py` from it, extend its tables until `--check` passes, and re-apply its hand-edit stack.
  The plugins fold must carry the families' plugin fixes (the topk `load_weights` override and its mapper; the pplx
  plugin-registered config class `hf_config.py`).
- **Owner decisions 18-22 land in this workstream**:
  - 18: `schema_version` on every recipe; the exported recipe JSON Schema; rcp-ndcg's compatibility check on
    `recipe:<id>`; no lockstep pin between rcp-ndcg and rcp-ndcg-vllm (core and rcp-ndcg keep theirs).
  - 19: the recipe schema keeps the engine-neutral `client` block strictly apart from `serve`/`engine`.
  - 20: the verified emulators and the observation corpora move into `rcp-ndcg-test` (today `rcp_ndcg.testing.engines`,
    `rcp_ndcg.testing.corpus`, `tests/contract/engines/`, `tests/conformance/`), named "emulators" to distinguish them
    from the product's `rcp_ndcg.inference.fake`. The product keeps only what users need offline.
  - 21: rename `rcp_ndcg.llm` to `rcp_ndcg.judging` (snapshots, docs, CHANGELOG).
  - 22: `rcp-ndcg-test` is installed from a git subdirectory; say so in its README and the docs.
- `packages/rcp-ndcg-vllm/README.md` already describes the post-move state; reconcile it with the code after the move.
