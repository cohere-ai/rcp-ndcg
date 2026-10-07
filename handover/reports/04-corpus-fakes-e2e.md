# Report 04: corpus code, verified fake engines, T4 driver (integrated)

**Status:** DONE. `wip/gpu-quality` and `wip/fake-engines` were reviewed and fixed (each with an independent
verifier), then all three branches landed together on `rfc-0001`; an independent verifier confirmed the integration
lossless (every test the three branches added is present and passes).

## gpu-quality
One over-length padding implementation in the harness; a runtime-bound test for the request generator (counts
tokenizer work); `rc_build.sh` stages `packages/rcp-ndcg-vllm/pairs/`; one behaviour fingerprint; the observation
corpus checks per `observations-spec.md` sections 1-7 (volatile fields, provenance with explicit `unavailable` reasons,
keyed immutable paths, credential scan, completeness, record migrations); the T3 quality stage runs the product's real
CLI; negative controls (a)-(e) are real vLLM v0.31.0 breakages, served by the wave and proven to fail the gates.

## fake-engines
- B1: the replay key is the full behaviour-shaping context (prompt plus `use_activation`, `dimensions`,
  `add_special_tokens`, `task`); unmodelled fields get a marked 400; colliding keys are refused. Field classes match
  vLLM v0.31.0's request models field for field.
- B2: the staleness gate resolves corpora by scanning manifests and names the changed fingerprint inputs.
- M1: only same-request repetitions are measured; the tolerance is declared unmeasured until a corpus has them.
- M2: the retrieval golden moves with the vectors; the ViDoRe retrieval view is waived with a tripwire; goldens are
  labelled regression pins.
- M3: headers and raw bytes are compared; undecodable bodies refused; unobserved routes declared.
- M4: the fingerprint (`rcp-fp/3`) keys exactly the fields that change request bytes or model outputs.

## Landing
Recipe files from `rfc-0001` won everywhere; one fingerprint, one corpus reader (`rcp_ndcg.testing.corpus`, now with
`nondeterminism`, the verification log, `normalise_raw`, wider credential findings and public provenance keys);
pairs regenerated for all 18 recipes in render mode (two render checks recorded as blocked: pplx's 262k-token probe,
qwen3-vl-reranker's whitespace-only query); the T4 supervision re-run drives the verified emulators; the judge pins
re-verified on the Hub.

## Corpus decisions (all corpora are provisional, from a GPU shakedown)
Current: octen-embedding-8b, qwen3-embedding-0.6b, zembed-1-embedding. Re-keyed on metadata only:
qwen3-reranker-8b, qwen3-vl-reranker-2b. Declared stale for re-recording (`tests/conformance/stale.json`; each fails
the staleness gate by name): jina-embeddings-v5-text-small, jina-reranker-v3, qwen3-reranker-0.6b, qwen3-reranker-4b,
qwen3-vl-embedding-2b, zerank-1-small-reranker, zerank-2-reranker. Listwise replay coverage is absent until
jina-reranker-v3 is re-recorded.
