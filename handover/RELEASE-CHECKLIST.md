# Release checklist for 0.0.1 (draft at M3; workstream 07 completes it)

Nothing here is done at M3 unless ticked. Order matters: CPU workstreams first, then the GPU waves, then the release.

## 1. CPU workstreams (see 00-MASTER section 8)
- [ ] 05 layout move (with decisions 18-22)
- [ ] 08 vLLM only, recipes for every role, the six judge recipes
- [ ] 09 one ordered processing pipeline
- [ ] 10 data I/O and MTEB interoperability (round trips through mteb's own loader and evaluator green)
- [ ] 06 final docs and the CHANGELOG fold into `## 0.0.1 — <date at tag time>`
- [ ] 07 QA passes (incl. the 00-MASTER section 9 QA items), fixes, versions, local release dry-run, `pip-audit`
- [ ] GitHub CI green on the final `rfc-0001` tip (M1-M3 were gated locally only; see 07 for runner risks)
- [ ] Public surface frozen (07) and the compatibility and versioning policy page published (06)
- [ ] README images use absolute URLs pinned to the tag, so the PyPI pages render them (06)

## 2. GPU waves (owner; per recipe, on the stock `vllm/vllm-openai:v0.31.0`)
For every recipe: T0 smoke, T1 recordings (observation corpus at the release fingerprint), T2 equivalence including
the `/tokenize` check and the media gate for media recipes, T3 quality, and the T4 scenarios once per run shape.
- Retrieval recipes (18): ctxl-rerank-v2-instruct-multilingual-1b, -2b, -6b; jina-embeddings-v5-text-small;
  jina-reranker-v3; octen-embedding-8b; pplx-embed-v2-context-9b-preview; qwen3-embedding-0.6b; qwen3-reranker-0.6b,
  -4b, -8b; qwen3-vl-embedding-2b; qwen3-vl-reranker-2b; topk-embed-v1-small; zembed-1-embedding; zerank-1-reranker;
  zerank-1-small-reranker; zerank-2-reranker.
- Judge recipes (6, after 08): qwen3.5-397b-a17b-nvfp4, gpt-oss-120b, qwen3.6-27b-fp8, qwen3.8-27b-fp8,
  qwen3.8-flash-next-fp8, qwen3.8-flash-next-nvfp4.
- [ ] Re-record the corpora declared stale in `tests/conformance/stale.json` (7) and re-verify every emulator at the
      release fingerprints; `stale.json` empty (the release-flag test enforces it); listwise replay coverage restored.
- [ ] Replace the provisional (shakedown) corpora with release recordings; manifests no longer say provisional.
- [ ] Fill every `pending_gpu` expected value in the reference cases.
- [ ] Media: decide the qwen3-vl-embedding per-clip video pixel budget (engine/client 25,165,824 px vs card
      7,864,320 px); record a page-image observation so the ViDoRe golden's retrieval view needs no waiver.
- [ ] Real infrastructure: one SLURM and one Kubernetes run of the one-container and StatefulSet job shapes.
- [ ] Flip each recipe's `status` to `verified` from its wave evidence only.

## 3. Release (owner)
- [ ] Dependabot PRs #1-#3 closed or superseded with a reason; `pip-audit` alerts reviewed
- [ ] Hugging Face datasets republished in MTEB's exact layout (decision 31), into an organisation if wanted; every
      pinned reference updated; mteb's `RetrievalDatasetLoader` loads each
- [ ] Delete `handover/` and its exclusion in `tests/docs/_markdown.py` in one commit
- [ ] Owner's go; merge `rfc-0001` to `main`; tag `v0.0.1`; watch `release.yml` publish core -> rcp-ndcg -> rcp-ndcg-vllm
      (trusted publishing per package); `rcp-ndcg-test` is never published
