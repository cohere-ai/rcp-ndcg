<!-- Handover copy of the operator's working note `fam-vl/BRIEF.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# Lane `fam-vl`: finish the recipes `qwen3-vl-embedding-2b`, `qwen3-vl-reranker-2b` — the sweep's items and the deep review

Read first: `<operator-notes>/COMMON.md` (binding; at most two verifier rounds), `AGENTS.md`,
`docs/how-to/add-a-model.md`, and:
- `<operator-notes>/recipe-sweep/REPORT.md` (lane recipe-common: the shared helpers, notably the contract-test
  helper you MUST use for every recipe below, and what each family still owns) — your base contains its work;
- `<operator-notes>/drafts/recipe-sweep-after-p1-tail.md` (the bullets naming your recipes);
- `<operator-notes>/research/sweep-recipes/work/report.md` with the operator's verification and DECISIONS in
  `<operator-notes>/research/sweep/TRIAGE.md` section "sweep-recipes" (binding);
- `<operator-notes>/shake/FINDINGS.md` (GPU shakedown rows for your recipes; re-read it before each round —
  rows are added during the night);
- `<operator-notes>/p1-tail/REPORT.md` (the role fields) and `<operator-notes>/harness/REPORT.md`.

Base: branch `int-recipes` at the operator's merge of recipe-common. Six family lanes run in parallel: touch ONLY your
recipes' directories `packages/rcp-ndcg-vllm/recipes/<id>/`, their tests `packages/rcp-ndcg-vllm/tests/recipes/test_<id>.py`
— nothing in `src/` of any package, no shared docs (list
needed shared-doc edits under **For the next lanes**). If a recipe needs a product change, stop and report it.

## Items
sweep-recipes #6 (ONE R20 pixel shape, decided from the vLLM v0.31.0 tag source at <repo>/.refs/vllm: which `mm_processor_kwargs` shape reaches the HF processor; cite file:line), #7 (declare the client media policy with the same numbers; qwen3-vl-embedding's 'never emits video_url' sentence is stale since p1-tail 2e), the draft's video sampling reconciliation (served fps 2 / 768 frames vs the reference's fps 1 / 64: one declared policy both sides), the minors for these two. topk's pixel shape follows your decision: write it under **For the next lanes**.
Plus for every recipe: its contract test through the shared helper pins every serve/client/reference field (show two
mutants per recipe red); stage 1 (CPU, network-gated tokenizers: `RCP_NDCG_NETWORK_TESTS=1`, downloads under your
lane's scratch `RCP_NDCG_VLLM_TOKENIZER_CACHE`; public models only; never read any token file) re-run with its verdict;
one CHANGELOG bullet for the family under `## Unreleased`.

## Verifiers: round 1 IS the deep per-recipe review (owner decision)
checks the final recipe against its model card at the pinned revision (Hub API, public), its reference implementation,
the vLLM v0.31.0 source at the tag (`<repo>/.refs/vllm`: load path, pooler, template, media processor),
the plugin if any, and the stage-1 result — evidence or not a finding (`<operator-notes>/research/SWEEP-PROTOCOL.md`);
MiMo confirmation verifier. Then report; anything open goes under **Open questions**.

## Done when
`<operator-notes>/bin/gate <your branch>` passes every step (incl. `vllm-pkg`), your recipes' network-gated
tests pass with the variable set, REPORT.md has the finding -> test -> red/green table and each recipe's stage-1 verdict.

## Operator note (08:10): started early
Your base is `lane/recipe-sweep` at `d222426` (recipe-common's committed helper, marker fix and NOTICE), not its final
state: recipe-common is still finishing (a hang in the network recipe tests). Before your final report, merge
`lane/recipe-sweep` (its tip then) AND the current `rfc-0001` into your branch and pass the gate (COMMON speed rules).
The network recipe tests have hung at 100% CPU for recipe-common: run them one file at a time, under `timeout 600` with
`-o faulthandler_timeout=120`, and fix any slow path the stack shows in your own test files.
