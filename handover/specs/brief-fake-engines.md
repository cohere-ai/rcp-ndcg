<!-- Handover copy of the operator's working note `fake-engines/BRIEF.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# Lane `fake-engines`

Read first: `<operator-notes>/COMMON.md` (binding; at most two verifier rounds: round 1 MiMo deep + GLM
hygiene, round 2 MiMo confirmation), `AGENTS.md`, then the common block and your section below (from
drafts/gpu-lanes.md), `<operator-notes>/OBSERVATIONS-SPEC.md` and `GPU-VALIDATION.md` items 2-4, 7, 8.

Base: commit `5359655` (rfc-0001 + the 18 recipes + both plugins + the RC tooling). CPU only; no job submission.

## The data you have NOW (real vLLM v0.31.0 exchanges)
`<operator-notes>/shake/shake1c/<wave>/vllm-0.31.0/<recipe>/` holds the recorder's exchanges from the GPU
shakedown (12 recipes: qwen3-embedding-0.6b, qwen3-reranker-0.6b/4b/8b, qwen3-vl-embedding-2b, qwen3-vl-reranker-2b,
zembed-1-embedding, jina-reranker-v3, jina-embeddings-v5-text-small, octen-embedding-8b, zerank-1-small-reranker,
zerank-2-reranker), in the recorder's format (`rcp-ndcg-vllm/src/rcp_ndcg_vllm/record.py`). Some of those
recipes were served with shakedown-only patches (see `<operator-notes>/shake/FINDINGS.md`): treat the corpus as
provisional input to BUILD and VERIFY the emulators; the release corpus is re-recorded at RC0 and keyed by fingerprint.
Lane `gpu-quality` is writing the full observation corpus format (OBSERVATIONS-SPEC, `rcp_ndcg_vllm.observe`); read its
worktree `<operator-notes>/wt-gpu-quality` (read-only) and put your corpus reader behind one seam so its format
plugs in when it lands. The behaviour fingerprint function is YOURS: `rcp_ndcg_vllm.fingerprint.behaviour_fingerprint(recipe)
-> str`, defined once (GPU-VALIDATION item 8's inputs), reused by the corpus writer later.

## Common (drafts/gpu-lanes.md)
Common to all four: read GPU-VALIDATION.md's section "Node runtime: isolation and resources" first (binding,
items 1-12); then read `<operator-notes>/COMMON.md`, `<operator-notes>/OBSERVATIONS-SPEC.md`
(binding for gpu-quality and fake-engines: the request generator, repetitions, record format, provenance, storage,
checks and change handling), `<operator-notes>/GPU-VALIDATION.md` (the
whole file is the specification, especially "The GPU run is also the test suite's audit"), the harness package
`rcp-ndcg-vllm/` (recipe schema, equivalence harness, recorder, wave runner, `bootstrap.sh`, `submit.sh`).
file; scripts take their paths in `RCP_GCS_AUTH_FILE` and `RCP_HF_TOKEN_FILE`.


## Lane `fake-engines` (after wave A's recordings land; CPU only)
Goal: `rcp_ndcg.testing.engines`, verified fake engines built from the observation corpus (GPU-VALIDATION.md items
2-4, 7 and 8; OBSERVATIONS-SPEC.md, especially sections 3-7: derived views recomputed from raw records by versioned
code, tolerances derived from the measured non-determinism, the manifest's hash check, the append-only verification
record, schema migrations, the behaviour diff), their conformance suite, and the golden replays. Scope: `rcp-ndcg/src/rcp_ndcg/testing.py` -> package
`rcp-ndcg/src/rcp_ndcg/testing/` if needed, `tests/contract/engines/vllm-0.31.0/` (the compact corpus), `tests/conformance/`,
`tests/e2e/test_golden_replay.py`, `inference/fake.py` only to route `fake://vllm-0.31.0/<recipe>` to an emulator.
1. Protocol emulation per route (validation, errors, ordering, framing, usage, token counting with the recipe's real
   tokenizer files and template); model outputs replayed for observed inputs, a declared deterministic surrogate for
   unseen ones (marked in the reply).
2. Conformance: every recorded exchange replayed against its emulator, identical status and body within the recorded
   non-determinism; the emulator records the engine version and recipe revision it was verified against and refuses
   another.
3. Golden replays (NanoBEIR, ViDoRe subsets) reproduce the GPU run's metrics to 1e-9.
4. Forward compatibility (GPU-VALIDATION.md item 8): protocol layer keyed by engine+version, model layer by the recipe's
   behaviour fingerprint (define the fingerprint function once, in `rcp-ndcg-vllm`, and reuse it in the corpus writer and
   the conformance suite); staleness fails with the changed inputs named (waiver file must be empty at release);
   versioned corpus schema with a migration hook; registry by (engine, version, fingerprint) with an entry-point group
   for out-of-tree emulators; a re-record-changed-only command and a behaviour-diff report.
5. Mutations: change one emulated behaviour (e.g. the over-length threshold by one token, the result ordering) and show
   conformance red; the golden replay red when a rerank score is perturbed; edit a recipe's template and show the
   staleness check fail with the template named.
