# Handover of RFC-0001 (`rfc-0001`) to an external code agent

Start with `00-MASTER.md`. Then one workstream at a time:

1. `01-integrate-fix-inference.md` — finish and merge the inference-layer fixes.
2. `02-recipe-common-and-harness.md` — finish the shared recipe fixes, close three harness gaps and the fake engine's
   slow path, bring the recipe integration branch up to date.
3. `03-recipe-families.md` — the six recipe families (18 recipes, 2 plugins), references kept faithful.
4. `04-corpus-fakes-e2e.md` — the CPU side of GPU validation: corpus code, verified fake engines, T4 driver.
5. `05-layout-move.md` — the per-distribution layout, applied by script plus a hand-edit stack.
6. `06-docs-final.md` — the docs that depend on the layout; the CHANGELOG fold.
7. `07-qa-and-release-prep.md` — four QA passes, fixes, the release checklist (no tag, no publish).

`specs/` holds the sanitized specifications the workstreams cite. Write your reports to `reports/`.
