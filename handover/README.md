# Handover of RFC-0001 (`rfc-0001`) to an external code agent

Start with `00-MASTER.md` (state at M3, decisions, open items). Completed: 01-04 (reports in `reports/`). Remaining, in order:

1. `01-integrate-fix-inference.md` — finish and merge the inference-layer fixes.
2. `02-recipe-common-and-harness.md` — finish the shared recipe fixes, close three harness gaps and the fake engine's
   slow path, bring the recipe integration branch up to date.
3. `03-recipe-families.md` — the six recipe families (18 recipes, 2 plugins), references kept faithful.
4. `04-corpus-fakes-e2e.md` — the CPU side of GPU validation: corpus code, verified fake engines, T4 driver.
5. `05-layout-move.md` — the per-distribution layout, applied by script plus a hand-edit stack.
6. `06-docs-final.md` — the docs that depend on the layout; the CHANGELOG fold.
7. `07-qa-and-release-prep.md` — four QA passes, fixes, the release checklist (no tag, no publish).

`specs/` holds the sanitized specifications the workstreams cite. Write your reports to `reports/`.

Added after M3:
8. `08-vllm-only-and-judge-recipes.md` — vLLM only; recipes for every role and the six judge recipes.
9. `09-processing-pipeline.md` — one ordered processing pipeline, a postprocess home, `data/preprocess.py` split.

10. `10-data-io-and-mteb.md` — MTEB-layout ingestion (`hf://`, `mteb:<Task>`), MTEB export (predictions,
    scoring through mteb, the dataset writer), the reader contract widened; evidence in `specs/mteb-data-model.md`.

Order: 05, then 08 and 09, then 10, then 06 and 07 in parallel. `RELEASE-CHECKLIST.md` is the draft release checklist.
