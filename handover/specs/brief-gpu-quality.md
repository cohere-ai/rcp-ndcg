<!-- Handover copy of the operator's working note `gpu-quality/BRIEF.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# Lane `gpu-quality`

Read first: `<operator-notes>/COMMON.md` (binding; at most two verifier rounds: round 1 MiMo deep +
GLM hygiene, round 2 MiMo confirmation), `AGENTS.md`, then the common block below (from drafts/gpu-lanes.md).
Base: the operator's `shake1` branch content (rfc-0001 + the 18 recipes + both plugins; the RC procedure, wave runner
and wave 0 of lane rc-build are merged). The harness now drives the product's role clients (R30): consume, never copy.
No lane submits a GPU job: the operator does; say in the report exactly what to submit (wave list, flags).
A layout move (packages/* -> top-level dirs, tooling -> rcp-ndcg-test) will be applied by script later; do not
anticipate it, keep your files where the package keeps them today.

## Common (drafts/gpu-lanes.md)
Common to all four: read GPU-VALIDATION.md's section "Node runtime: isolation and resources" first (binding,
items 1-12); then read `<operator-notes>/COMMON.md`, `<operator-notes>/OBSERVATIONS-SPEC.md`
(binding for gpu-quality and fake-engines: the request generator, repetitions, record format, provenance, storage,
checks and change handling), `<operator-notes>/GPU-VALIDATION.md` (the
whole file is the specification, especially "The GPU run is also the test suite's audit"), the harness package
`rcp-ndcg-vllm/` (recipe schema, equivalence harness, recorder, wave runner, `bootstrap.sh`, `submit.sh`).
file; scripts take their paths in `RCP_GCS_AUTH_FILE` and `RCP_HF_TOKEN_FILE`.


## 0. FIRST (the operator is blocked on it): stage-2 pairs files for every recipe
The wave runner runs equivalence only with `<pairs-dir>/<id>.jsonl` (or `default.jsonl`), and none exist. Build the
request generator `rcp_ndcg_vllm.observe.requests` (OBSERVATIONS-SPEC section 1; `GENERATOR_VERSION`, seed, pinned
dataset commits, the synthetic adversarial set as text) FIRST to the point where it writes one pairs file per recipe
in the harness's pairs format (read the harness's reader; rerank rows `query` + `documents`, embed rows per shape,
media rows for the VL recipes), committed compactly under `rcp-ndcg-vllm/pairs/` (within the size budget of
GPU-VALIDATION.md item 7), validated by stage 1 on CPU for every recipe. Commit that as its own commit with the subject
starting `pairs:` as soon as it passes stage 1 for all 18; then continue with the rest of this brief.
`jobs/rc_build.sh` stages `pairs/` from the checkout root only: make it stage `rcp-ndcg-vllm/pairs/` (one
home; test it), in the same commit.

## Lane `gpu-quality` (T1-T3 harness stages, the observation corpus, controls)
Node-runtime items 4, 5, 6: the harness (FOLLOWUP-unify) implements the subprocess reference and the `/tokenize`
check; you run them on the node (reference environment path, one GPU owner at a time per slot) and fail a recipe on
a mismatch.
Goal: the T1-T3 tiers as harness stages the wave runner calls per recipe (scripts, not pytest: no test needs a GPU),
producing the observation corpus for the verified fake engines, the quality tables, and the negative controls. Scope: `src/rcp_ndcg_vllm/{record,quality}.py`, the wave runner's invocation, the observation-corpus format (shared with
lane `fake-engines`: define it here first, as a documented schema with `schema_version`, keyed by engine+version and the
recipe's behaviour fingerprint per GPU-VALIDATION.md item 8), and the re-record-changed-only mode of the wave runner.
1. The observation corpus per recipe, implemented exactly per OBSERVATIONS-SPEC.md sections 1-6 (generator
   `rcp_ndcg_vllm.observe.requests` with `GENERATOR_VERSION`, repetitions incl. after-restart, raw records with
   `RECORD_SCHEMA`, the manifest with full provenance and hashes, the GCS full corpus and the repository subset, the
   acceptance checks) and section 7's `--changed-since` mode; CPU tests with the stub engine for every part, incl. a
   test that no credential-shaped string can reach a corpus, and one that a tampered file fails the manifest check.
   Original item text for context (GPU-VALIDATION.md, "The GPU run is also the test suite's audit", item 1):
   sampled requests across shapes, lengths, languages, empty documents and media, every route and error body, recorded
   twice, volatile fields stripped, the measured non-determinism stored with it; compact (item 7).
2. T2: stage 2 replies are part of the corpus (the emulators replay them).
3. T3 quality stage: rcp-ndcg's served path (`rcp-ndcg retrieval index|search|rerank`, then `eval score`) per the task
   matrix in GPU-VALIDATION.md, plus the reference implementation through `mteb` on the same tasks; the comparison
   table (served vs reference, vs the paper's stored per-subset numbers for paper models, vs published numbers with a
   deviation note column); the full served exchanges of one NanoBEIR and one ViDoRe subset for the golden replay.
4. Negative controls (a)-(f) of GPU-VALIDATION.md as recipe variants generated from each family's recipe; the wave
   fails if a control passes.
5. CPU tests of all of the above with the harness's stub engine; mutation: make a control's gate a no-op and show the
   wave report flags it.
