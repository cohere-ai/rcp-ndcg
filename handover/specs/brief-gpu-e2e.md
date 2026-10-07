<!-- Handover copy of the operator's working note `gpu-e2e/BRIEF.md` (sanitized: private names, the private repository, the job platform and storage locations removed; `<operator-notes>/...` paths point to files that are NOT available — the content you need is in handover/). Treat it as the specification it was for its lane. -->

# Lane `gpu-e2e`

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
`packages/rcp-ndcg-vllm/` (recipe schema, equivalence harness, recorder, wave runner, `bootstrap.sh`, `submit.sh`).
file; scripts take their paths in `RCP_GCS_AUTH_FILE` and `RCP_HF_TOKEN_FILE`.


## Lane `gpu-e2e` (T4)
Node-runtime items 2 and 10 are yours to use: install from the staged wheelhouse through the runners' install-source
option (built by fix-review-2), and assert the process boundaries.
Goal: the end-to-end scenarios of GPU-VALIDATION.md as a harness stage driven inside the pod (not pytest). Scope:
`packages/rcp-ndcg-vllm/src/rcp_ndcg_vllm/e2e.py`, scenario configs under `packages/rcp-ndcg-vllm/scenarios/`, the
driver that renders the run's phased job script with rcp-ndcg's SLURM renderer (`container_runtime: none`) and runs it
in the pod.
1. Judges (owner): `Qwen/Qwen3.8-27B-FP8` for the outage and identity scenarios; for the four-phase run
   `nvidia/Qwen3.8-Flash-Next-NVFP4` if vLLM v0.31.0 supports it on B200 (establish from the vLLM source at the
   v0.31.0 tag, `<repo>/.refs/vllm`, and a T0 smoke in the first wave), else `Qwen/Qwen3.8-Flash-Next-FP8`.
   Verify each model id on the Hub (`curl -s https://huggingface.co/api/models/<id>`) and report the commit.
2. Scenarios 1-4 (text four phases; outage with a killed judge engine; identity rerun with new ports; ViDoRe with media).
   Assertions: every step completes; the manifest records each phase's engines; the outage run parks and recovers, and
   the `wait_on_outage_s` expiry path fails with `BackendUnavailableError`; the identity rerun recomputes nothing;
   two identical runs give identical identities and outputs (judgements may differ at temperature > 0: state what is
   compared).
3. Offline counterparts: the rendered phased script as a golden file; the supervision re-run in the root suite with
   the verified fake engines as its engines; the observed outage behaviour as a transport test.
4. CPU tests of the driver with stub engines (the root suite's supervision stubs); shellcheck.
