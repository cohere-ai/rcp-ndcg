# Workstream 03: finish the six recipe families and land them in `wip/int-recipes`

Read `handover/00-MASTER.md` first (decision 9 on references is the most violated rule in this workstream). Specs:
`handover/specs/brief-fam-*.md` (one per family: the binding item list), `recipe-sweep.md`, `triage-decisions.md`
("sweep-recipes" + "OPERATOR DECISION"), `sweep-recipes-report.md` (the evidence for each finding #1-#9), the recipe
how-to `docs/how-to/add-a-model.md`, `gpu-findings.md` (GPU shakedown rows naming recipes).

Prerequisite: workstream 02 A+B (recipe-common merged into `wip/int-recipes`, which is up to date with `rfc-0001`).
Each family branch then merges `wip/int-recipes` before review and lands back into it. Families touch only their own
recipes' directories, their tests and (fam-late only) the two plugins; shared files only via CHANGELOG bullets.

## Common items for every recipe (from the briefs)
- Its contract test through the shared `_contract.py` helper pins every `serve`, `client` and `reference` field; show
  two mutants per recipe red.
- Stage 1 on CPU with the real tokenizer (network-gated) passes; record the verdict per recipe.
- One CHANGELOG bullet per family under `## Unreleased`.
- `status` stays `unverified` (GPU validation is out of scope).
- A deep review per recipe (the owner required it for all 18 + both plugins): the final recipe against its model card
  at the pinned revision (Hub API), its reference implementation, the vLLM v0.31.0 source at the tag (load path,
  pooler, template, media processor; clone `vllm-project/vllm` at `v0.31.0`), the plugin if any, and the stage-1
  result. Evidence (file:line, measured output) or it is not a finding.

## References: the rule and the suspects
A reference reproduces the paper's code or the model card. The harness compares spans, so a rerank reference's
`render` returns `{"index", "shape": "pair", "query": str, "documents": [str, ...]}` — that FORMAT is required. What a
reference must never do is port the client's cut (the query share, the settle-once rule, a strip the client applies)
so that over-cap rows match. Where the paper's over-cap cut differs from the client's and the anchors survive, declare
`known_deviations: [over_cap_cut_differs]`; where an anchor drops, `[anchor_drop_over_cap]`; under-cap rows gate
exactly. Check every reference touched since `wip/int-recipes`; known suspects:
- `wip/recipe-sweep` `f1d1e0a` (qwen3-reranker-8b, zerank-1-small: "Port of the product's rerank client");
- `wip/fam-ctxl` `68644be` ("render mode ports the wire's spans");
- `wip/fam-zerank` `ecbf3f8`, `d79a1a8` ("mirror the wire's spans exactly (strip + settle-once)");
- `wip/fam-vl` `ae300f4` ("the wire's content spans") — may be format only; verify.

## The families

### fam-qwen3-rerank — qwen3-reranker-0.6b, -4b, -8b (`wip/fam-qwen3-rerank`)
Items: sweep-recipes #3 (one over-cap policy for the family: declare a deviation only as decision 9 says; 4b's
`anchor_drop_over_cap` is mislabelled if its reference keeps the anchor), #4 (render by character-offset cut, never
`decode`: 0.6b's method family-wide; add a non-NFC render test — 4b/8b decoded renders back to NFC), #5 (the settle-rule
wording read from the merged rerank client, `rcp-ndcg/src/rcp_ndcg/inference/clients/rerank.py`), the minors (0.6b
`min_version` 0.31.0; drop default `startup_timeout_s`; template file naming convention; `requirements-reference.txt`).
State: only a WIP snapshot (`be7ef09`, uncommitted work: 4b reference toward the offset cut, 8b reference being
de-ported). The lane's tool reported a text-replacement accident ("prefix" rewritten to "secret" in three spots) while
editing; grep the diff for corrupted words before trusting any line.

### fam-ctxl — ctxl-rerank-v2-instruct-multilingual-1b, -2b, -6b (`wip/fam-ctxl`)
Items: #1 DECIDED `instruction: none` for all three (the paper configs `experiments/paper/rerankers/ctxl_rerank_*.yaml`),
#2 no reference folds an instruction, #5 wording, minors. State: committed (`0ba6504` instruction none + settle rule +
over-share rows declared, `68644be` references (suspect), `d8c1bce`, `b11fb17` contract tests with mutants, `8f5764e`
CHANGELOG); it was about to run the quality bar; no review yet.

### fam-zerank — zerank-1-reranker, zerank-1-small-reranker, zerank-2-reranker (`wip/fam-zerank`)
Items: one declared `normalize: [strip]` rule for the family (replaces jinja `| trim` and the declared whitespace
divergence), #5 wording, zerank-1's `serve.convert` refusal fixed (a rerank recipe declares its scorer through
`engine.hf_overrides`), minors. State: committed (`a65c057` normalisation + settle rule + convert fix, `ecbf3f8` and
`d79a1a8` references (suspects), `47c067d` contract tests, `082e1d4` CHANGELOG, format fixes); its first review round
was running.

### fam-vl — qwen3-vl-embedding-2b, qwen3-vl-reranker-2b (`wip/fam-vl`)
Items: #6 one R20 pixel-pin shape decided from the vLLM v0.31.0 source (the lane chose nested
`mm_processor_kwargs: {images_kwargs: {...}}`; verify at the tag which shape reaches the HF processor, cite file:line),
#7 the client media policy declared with the same numbers (`max_images`, `image_policy`, video), the stale "the
lowering never emits video_url" sentence (multimodal embeddings send video now), the video sampling reconciliation
(served fps 2 / 768 frames vs the reference's fps 1 / 64: one declared policy both sides). State: committed (`a0b9f8c`,
`ae300f4`, `b74eb02`); first review round was running. Its pixel-shape decision also binds topk (fam-late).

### fam-late — topk-embed-v1-small (+ plugin `rcp-ndcg-vllm/plugins/topk`), pplx-embed-v2-context-9b-preview (+ plugin `plugins/pplx`) (`wip/fam-late`)
Items: topk: `serve.plugin: rcp-ndcg-vllm-topk` (the plugin's distribution name), no `--trust-remote-code`; declare
with the product fields: `query_max_tokens: 1024`, the document skip ids from the checkpoint's `scoring_skip_ids`, media
on documents only (`media_sides`), the strip rule; re-run stage 1 with over-length inputs (documents up to 8192 tokens);
the R20 pixel shape as fam-vl decided. pplx: #8 (a stale sentence claimed the role client refuses `max_tokens`);
`--trust-remote-code` replaced by a plugin-registered config class where possible; the served dtype (the GPU
shakedown: vLLM v0.31.0 refuses float32 for this model — "expected bfloat16 or float16"; decide the served dtype against
the reference and document the deviation if any). topk GPU finding: the plugin model failed to load with
"Following weights were not initialized from checkpoint: {'custom_text_proj.bias'}". State: committed `f3450b4` (topk:
mark the zero projection bias initialized), `bb5f752` (pplx: register the checkpoint's config with transformers
`AutoConfig`), `5d5ecc9` (topk product fields; pplx dtype and local config parse); WIP snapshot `66f2ee6` (a topk test
registering a fake `/pooling` route through the product's public `register_fake_route`).

### fam-dense — qwen3-embedding-0.6b, octen-embedding-8b, jina-embeddings-v5-text-small, zembed-1-embedding, jina-reranker-v3 (`wip/fam-dense`)
Items: per-shape `query_max_tokens` where the reference caps queries; jina-embeddings-v5 uses `anchor: last_content`
instead of `anchor: first`; #5 where it applies (jina-reranker-v3); minors. State: committed `da61443` (contract tests
with mutants), `4b19127` (minors); WIP snapshot `8232810` (network re-runs; one test still read a stale harness report
key `document["fit"]` that no longer exists — read the current report shape in `equivalence/stages.py`).

## Landing
Per family: merge `wip/int-recipes` in, finish, review (the deep per-recipe review above + your hygiene pass), quality
bar, network tests per file, merge into `wip/int-recipes`. When all six are in: merge `wip/int-recipes` into
`rfc-0001`, quality bar, push, CI. Report in `handover/reports/03-families.md` with each recipe's stage-1 verdict.
