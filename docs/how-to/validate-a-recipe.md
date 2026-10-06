# Validate a recipe on GPUs

A recipe ships only after it has been validated end to end on GPU against its reference; `status.state` records
the outcome (`unverified`: written, not yet checked; `verified`: the harness passed every gate; `failed`; with
the engine `image`, the `date` and the report beside it). This page is what runs before that flip: the waves,
their tiers and what each one proves. The node-level procedure (the staged wheels, the host checks) is
[release candidates and the GPU waves](release-candidates.md); the recipe format is [add a serving
recipe](add-a-model.md).

## Three environments on one node

The engine, the client and each reference never share an environment:

1. **The engine environment** is the image's own Python (the stock `vllm/vllm-openai` image), untouched except
   the one pure-Python wheel installed with `--no-deps` ([serve a retrieval model](serve-a-model.md)): a
   `pip freeze` before and after the install differs by exactly that wheel.
2. **The client environment** is a fresh venv holding the release's staged wheels constrained by the release's
   constraints file. It runs the equivalence harness, the recorder, the command line and the wave tooling over
   HTTP only -- no torch.
3. **The reference environment** is per reference implementation: its own pinned requirements (each run records
   the versions it saw), installed in its own venv, run as a subprocess.

## The waves

- **T0 -- smoke.** The engine is up from the recipe's own serve argv: `GET /v1/models` answers, one request per
  role route scores, and the engine version and fingerprint are recorded with the report.
- **T1 -- observations.** The recorded corpus: inputs the reference and the engine both see, whose outputs the
  equivalence checks and the later CPU fakes are built from.
- **T2 -- equivalence.** Served outputs against the reference implementation -- token-id equality and the
  insertion checks on the CPU side first, then the scored pairs on GPU, with the engine's token counts checked
  against the client's. A recipe whose reference deliberately drops anchors declares the deviation and is
  compared under the cap only ([the equivalence policy](../reference/recipes.md#equivalence-policy)).
- **T3 -- quality.** The MTEB suites: the served model's rankings scored with the released gains, against the
  thresholds the model card claims.
- **T4 -- end to end.** The served model as a run's step, with a served judge, through the product's clients.

## Fakes and conformance on CPU afterwards

GPU work produces observation corpora; it never runs under pytest. Afterwards, model-level fakes rebuilt from
the recordings verify the CPU side: one conformance suite with two targets -- the live engine and a
recipe-level fake engine -- both driven through the product's role clients (never raw HTTP, never a copy of
the client), plus golden replays and a forward-compatibility run. Those suites live in the unpublished
contributor package (`rcp_ndcg_test`) and run per its README.

## The release-candidate loop

Before the tag: build the wheelhouse and constraints of the candidate (the release-candidates page), run the
fixed waves, fix what fails, re-run on GPU only what a fix changed, and repeat until every recipe's `status:`
reads `verified`. The tag ships none unverified.