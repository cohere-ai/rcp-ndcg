# Workstreams 08, 09, 10 in parallel: lanes, file ownership, interfaces, merge order (2026-10-09)

Paths are under `rcp-ndcg/src/rcp_ndcg/` unless written otherwise. A lane edits only the files it owns; anything
else it needs is a one-line request in its report. Shared append-only files (CHANGELOG.md, docs nav) are merged by
union; generated files (tests/contract/snapshots/*, schemas/*) are regenerated on merge, never hand-merged.

| Lane | Scope (workstream items) | Owns | Starts | Depends on |
|---|---|---|---|---|
| w09 (running) | 09: one stage order, ProcessingRecord, postprocess home, preprocess split | inference/clients/**, inference/config.py, inference/types.py, data/preprocess.py, data/text_policy.py, data/text_budget.py, data/postprocess.py, data/census.py, data/prepare.py (stage wiring only), storage/census.py, judging/{cost,judging,reparse,store}.py, AGENTS.md table | now | - |
| l08-cat | 08 C research: the judge catalog as a spec (no code) | handover/specs/judge-catalog.md | now | - |
| l10a | 10 A, B, C1 (data model fields only) | data/dataset.py, data/io/** (except data/io/mteb.py), data/_rows.py, data/validate.py, data/revisions.py, rcp_ndcg_core record fields (Document.title, Query.instruction), rcp_ndcg.testing io conformance, readers/writers entry points, tests/data/**, docs/concepts/datasets*, docs/how-to data pages | now | - |
| l10b | 10 D: MTEB predictions, scoring inside mteb, the MTEB dataset writer, fractional grades, republishing converter | data/rankings.py (`format="mteb"`), data/io/mteb.py (new; one line in the WRITERS table), eval/mteb/**, tools/republish_mteb.py (new), tests/eval/mteb/**, docs/how-to/mteb*.md | now | the interface below; rebases on l10a |
| l08-sglang | 08 A: vLLM only | inference/adapters/**, data/resolution.py, data/prepare.py (SGLang branches), judging/_parsing/**, judging/client.py, support/serve.py, retrieval/config.py, experiments/paper/serve/*, NOTICE rows, the SGLang docs passages | after w09 | w09 |
| l10c | 10 C2, C3: one MTEB join for title+body, task and per-query instructions, through the 09 stages | the content-normalisation and template stages in inference/clients/** (after w09), data/templates.py, judging/prompts formatting, REPRODUCIBILITY.md note, identity/fingerprint bump if formatting enters it | after w09 and l10a | w09, l10a |
| rec-harrier | new recipe family harrier-oss-v1 (270m, 0.6b, 27b; owner request) | rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipes/harrier-oss-v1/, its tests in rcp-ndcg-test/tests/recipes/, its pairs files, its catalog rows | now (based on lane/rfam) | rfam |
| l08-judges | 08 B, D: role judge, the six judge recipes (as families), presets to recipes | rcp-ndcg-vllm/src/rcp_ndcg_vllm/recipe.py (judge role), rcp_ndcg_vllm/recipes/<judge families>/, judging/judges/** (presets removed), inference/recipes (judge resolution), T4 scenarios, examples, docs/concepts/judges.md | after rfam and l08-cat | rfam, l08-cat |

## Interfaces fixed now (so l10b and l10c can code against them before l10a merges)
- `Dataset.subset: str` (default `"default"`), `Dataset.split: str` (default `"test"`), `Dataset.task: str | None`;
  exports are keyed by `(task, subset, split)`.
- `Dataset.task_instruction: str | dict[Literal["query", "document"], str] | None`.
- `Document.title: str | None` (body stays in `text`; nothing joins at read time).
- `Query.instruction: str | None` (per-query; never merged into `text` at load).
- `Dataset.provenance` carries source URI, resolved revision commit, subset, split, and the duplicates policy with
  counts.
If a lane finds a field must differ, it stops and reports; the operator changes this page and tells the other lanes.

## Merge order
w09 -> l10a -> l10b (merges rfc-0001, adapts) ; l08-cat any time ; then l08-sglang and l10c in parallel ; l08-judges
after rfam. `run_all` stays 1022/987/35/0, 67/67, 82/82 at every merge; a lane that moves a number stops and reports.
