# Release checklist for 0.0.1

State at the QA/release-prep lane (workstream 07), on the `rfc-0001` tip. Ticked items were verified in this lane
or by the gate it ran (`bin/gate`, full tree: ruff, basedpyright, the root suite, contract+docs, mkdocs --strict,
the test package's suite, the recipe tests, the vLLM packages, the paper reproduction, the public-names scan,
a clean checkout). Everything unticked needs the owner, the GPU waves, or a decision that is not a lane's to make.

## 1. CPU workstreams
- [x] 05 layout move (decisions 18-22)
- [x] 08 vLLM only, recipes for every role, the six judge recipes
- [x] 09 one ordered processing pipeline
- [x] 10 data I/O and MTEB interoperability
- [ ] 06 final docs and the CHANGELOG fold into `## 0.0.1 — <date at tag time>` (docs 06 owns the fold; this
      lane added its own entries under `## Unreleased`)
- [x] 07 QA passes and fixes: the four read-only passes (`reviews/qa-artifacts/`), the two HIGH fixes (the
      conformance mixed-delta case, the test package's per-test timeout guard), the MEDIUMs this lane fixed (the
      stage-2 default pins, the unframed-instruction overhead, the root tree's DNS watchdog, the single-suite
      leaderboards run, the one-home entry-point registry, the public-surface freeze, the plugin-seam docs), the
      cheap LOWs, and the deferred items that were cheap; the rest are recorded in `handover/reports/07-release-prep.md`
- [x] Versions: all four `pyproject.toml` at `0.0.1`, `rcp-ndcg` pins `rcp-ndcg-core==0.0.1`, `CITATION.cff`
      `version: 0.0.1`, and `release.yml` now checks the citation against the tag
- [x] Local release dry-run: the three distributions build at `0.0.1`, the tag-version check, the sibling-pin
      check, the semantic constraints check (107 pins agree with the lock) and `twine check` all pass
- [x] Public surface frozen: `docs/reference/public-surface.md` and
      `tests/contract/undocumented_public_names.json` (194 of 376 names are the advanced list; the contract suite
      keeps the freeze current in both directions)
- [ ] `pip-audit` alerts reviewed and the Dependabot PRs (#1 oauthlib, #2 vllm, #3 transformers 5.10.1) closed or
      superseded: needs the owner's `gh` access and a network advisory database (this lane could only read the
      lock: oauthlib 4.0.0, transformers 5.17.0 -- the PRs are stale against it, see the report)
- [ ] GitHub CI green on the final `rfc-0001` tip (the last run this lane saw was for `0d3a20b3`; the tip has moved
      since). The nightly shuffled run was reproduced locally on samples, not dispatched
- [ ] README images use absolute URLs pinned to the tag (06)

## 2. GPU waves (owner; per recipe, on the stock `vllm/vllm-openai:v0.31.0`, except `embeddinggemma-2` on its
digest-pinned nightly)
For every recipe: T0 smoke, T1 recordings (observation corpus at the release fingerprint), T2 equivalence including
the `/tokenize` check and the media gate for media recipes, T3 quality, and the T4 scenarios once per run shape.

- [ ] `embeddinggemma-2` on its nightly image: confirm the image's transformers is 5.19.x, the architecture loads,
      the pinned video sampling (60 fps, max_frames 32) and the Gemma 4 image geometry at the engine.
- [ ] Judge recipes (6): qwen3.5-397b-a17b-nvfp4, gpt-oss-120b, qwen3.6-27b-fp8, qwen3.8-27b-fp8,
      qwen3.8-flash-next-fp8, qwen3.8-flash-next-nvfp4 -- T0 smoke + `rcp-ndcg judge check` (both thinking modes,
      the NVFP4 backends, the media token counts), on top of the retrieval waves.

**Where the E2 wave stands (the RC0 `rc0-202610101520` state):** wave0 PASS; r3 verified `qwen3-embedding-4b` and
failed seven stage-2 gates (the bf16 bounds and the pplx-late query-vector nesting, fixed by the E2 round-1 lane
with per-variant `overrides.gates`); r2 verified the jina-v5-small and the three octen sizes and failed the rest
(jina-reranker-v3's stage-1 tokenize gap, the pplx-embed-v1 nesting, pplx-context's serve crash); r4 produced no
output (the topk own-torch venv cannot import torch: `libcufile.so.0`); r1 and r5 and the judge waves are
re-running or queued. Every recipe's `status` is still `unverified` (24 families), so no flip has happened yet.

- [ ] Finish the r1-r5 retrieval waves on the fixed RC and read each `status.json`
- [ ] Judges j1/j2 (the `KeyError: 'judge'` smoke fix is in the tip; needs the rebuilt harness on the node)
- [ ] T4 e2e scenarios
- [ ] Re-record the 8 corpora declared stale in `tests/conformance/stale.json` and re-verify every emulator at the
      release fingerprints; `stale.json` empty (the release-flag test enforces it); listwise replay coverage
      restored. The eight: jina-embeddings-v5-text-small, jina-reranker-v3, qwen3-reranker-0.6b/-4b/-8b,
      qwen3-vl-embedding-2b, qwen3-vl-reranker-2b, zembed-1-embedding
- [ ] Replace the provisional (shakedown) corpora with release recordings; manifests no longer say provisional
- [ ] Fill every `pending_gpu` expected value in the reference cases
- [ ] Media: decide the qwen3-vl-embedding per-clip video pixel budget (engine/client 25,165,824 px vs card
      7,864,320 px); record a page-image observation so the ViDoRe golden's retrieval view needs no waiver
- [ ] Real infrastructure: one SLURM and one Kubernetes run of the one-container and StatefulSet job shapes
- [ ] Flip each recipe's `status` to `verified` from its wave evidence only
- [ ] Clean the judge families' `handover/specs/judge-catalog.md` / lane-report citations from their shipped notes
      and sources before `handover/` is deleted; the repo-wide recipe hygiene guard scopes itself to the non-judge
      families until then

## 3. Release (owner)
- [ ] `pip-audit` alerts reviewed; Dependabot PRs closed or superseded with a reason
- [ ] The owner-revisit item from `ORCHESTRATION-STATE.md` section 11 is decided: the shipped judge sampling
      (`temperature: null`, `max_output_tokens` 8192/16384) diverges from handover decision 4.3 (`temperature 0.0`,
      `max_output_tokens 12288`). Changing those fields moves the judge family keys, so it is a decision, not a fix
- [ ] Hugging Face datasets republished in MTEB's exact layout (decision 31), into an organisation if wanted; every
      pinned reference updated; mteb's `RetrievalDatasetLoader` loads each
- [ ] Delete `handover/` and its exclusion in `tests/docs/_markdown.py` in one commit
- [ ] Owner's go; merge `rfc-0001` to `main`; tag `v0.0.1`; watch `release.yml` publish core -> rcp-ndcg ->
      rcp-ndcg-vllm (trusted publishing per package); `rcp-ndcg-test` is never published
