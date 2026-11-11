# Validate a recipe on GPUs

A recipe ships only after it is validated end to end on GPU against its reference. `status.state` records the
outcome. `unverified` means written but not yet checked. `verified` means the harness passed every gate. The
state can also be `failed`. The engine `image`, the `date` and the report sit beside the state. This page
describes what runs before that status change: the waves, their tiers, and what each tier proves. A wave is a
set of GPU checks. A tier is one group of checks in a wave. The node-level procedure (the staged wheels, the
host checks) is [release candidates and the GPU waves](release-candidates.md). The recipe format is [add a
serving recipe](add-a-model.md).

## Three environments on one node

The engine, the client and each reference never share an environment:

1. **The engine environment** is the image's own Python. The image is the stock `vllm/vllm-openai` image, or a
   recipe's digest-pinned nightly when the release image lacks its architecture. The environment is untouched
   except for the one pure-Python wheel. Install that wheel with `--no-deps` ([serve a retrieval
   model](serve-a-model.md)). A `pip freeze` before and after the install differs by exactly that wheel.
2. **The client environment** is a fresh venv. It holds the release's staged wheels, constrained by the
   release's constraints file. It runs the equivalence harness, the recorder, the command line and the wave
   tooling over HTTP only. It has no torch.
3. **The reference environment** is one venv **per family**. The bootstrap builds it from the family's
   `reference.lock`. It uses `--system-site-packages` over the image's torch/CUDA, with the family's own pins
   installed into the venv. A family that declares `# own-torch: true` gets a venv of its own. The bootstrap
   builds each needed family once per pod and reuses it across the family's variants. An import check fails
   loudly and names the family. The check verifies that torch sees the image's build. It also verifies that every pinned distribution is installed at its pin and imports. The reference runs as a subprocess of that python. Stage 2
   stores its outputs, so an unchanged output is reused (below).

## The waves

- **T0 -- smoke.** The engine is up from the recipe's own serve argv. `GET /v1/models` answers. One request per
  role route scores. The engine version and fingerprint are recorded with the report.
- **T1 -- observations.** The recorded corpus contains inputs that the reference and the engine both see. The
  equivalence checks and the later CPU fakes are built from its outputs.
- **T2 -- equivalence.** Served outputs against the reference implementation. Token-id equality and the
  insertion checks run on the CPU side first. Then the scored pairs run on GPU, with the engine's token counts
  checked against the client's. A media recipe's image/video rows gate here too. The reference's embed/score mode computes their vectors, per token and with the card's declared keep-rule. It can also use scores from the harness's `media` field. A recipe whose reference deliberately drops anchors declares the deviation. The harness
  compares that recipe under the cap only ([the equivalence policy](../reference/recipes.md#equivalence-policy)).
  A recipe that declares Matryoshka dimensions is gated per declared `k`. The gate runs ex-post from one
  full-width served pass ([gating every declared k](../concepts/matryoshka.md#gating-every-declared-k)). A
  family declares its shared gate bounds in `gates`. A size whose measured precision floor differs overrides
  individual fields in its `overrides.gates`. A set field replaces the family's value. An unset field keeps it.
- **T3 -- quality.** The MTEB suites score the served model's rankings with the released gains. The scores must
  meet the thresholds that the model card claims.
- **T4 -- end to end.** The served model runs as a run's step, with a served judge. The product's clients drive it.

The submitted wave ([release candidates and the GPU waves](release-candidates.md#which-gates-a-submitted-wave-runs))
runs T0, T2 and the recorder. An operator runs T1's observation corpus, T3 and the negative controls against the same RC. The flags are named on that page. T4 is its own `--script e2e` job. A recipe's `status` becomes
`verified` only from a wave whose document records the gates it needs.

## The wave runner: step budgets, the pod log and partial results

A wave serves several recipes on one node at the same time. Each recipe's steps run in a worker of their own.
Thus a stuck request never holds another recipe's steps. The first GPU wave lost 40 minutes behind one stuck
request. Every step has a **declared wall-clock budget**. The runner computes the budget from the recipe's
request count: a base plus one allowance per request. The pairs file's rows are the count. A recipe that
legitimately needs longer raises the budget with `engine.step_budget_s`. This value is a floor: the runner's
formula can only raise the budget, never lower it.

When a step outruns its budget, the runner cancels the request in flight. It stops that recipe's engine. It
fails the step and names the request:

```
step equivalence exceeded 2400s; in flight: POST /rerank (request 3)
```

The other recipes continue. The engine's own death (a crash, an OOM) fails only that recipe's `serve` step. The
engine's last log lines are in `<out>/<recipe>/serve.log`. A clipped tail of those lines rides in the status
document. The harness's own requests (smoke, record, the corpus's bare probes) run with one declared
per-request timeout. That timeout is shorter than every step budget and is reported in the step documents.

The pod log gets one line per step boundary. It has no request bodies and no environment values:

```
run_wave: jina-reranker-v3 smoke start
run_wave: jina-reranker-v3 smoke passed 0.4s
run_wave: jina-reranker-v3 equivalence start
run_wave: jina-reranker-v3 equivalence passed 812.1s
```

Results do not wait for the pod to end. The runner rewrites `<out>/<recipe>/status.json` atomically after every
step. With `--upload`, it copies each finished recipe's directory to the output URI when the recipe ends. Thus a
cancelled or killed pod still leaves the evidence of everything that finished. The runner verifies every upload
against the destination and retries with backoff. It records the outcome in the recipe's status row. The wave
summary (`wave.json`/`WAVE.md`) is written before the last upload. Thus it reaches the URI too. A wave with a
failed upload does not report PASS.

The reference subprocess gets **a GPU of its own** beside the engine's. It never gets the engine's GPU, which
holds most of its memory. The runner reserves the reference GPU when packing. Thus a node packs fewer engines per wave. With 8 GPUs, a node takes at most 4 single-GPU recipes when each needs a reference GPU. The device and the GPU index
are recorded in `equivalence.json` (`device`, `reference_gpu`). A recipe whose reference cannot run on CPU
declares `reference.device: cuda` in its recipe file. A CPU reference run for that recipe is refused with the
way out. A pod that cannot spare the GPU fails that recipe early, never silently on CPU.

Stage 2's reference outputs are **stored**. The store is keyed by the family reference hash, the variant revision and the pairs-file hash. It is also keyed by the environment lock hash (or, when no lock applies, the reference interpreter's real path). It is also keyed by the device and the dtype, the resolved recipe's `serve` block and the client render fields. The
outputs are under `<out>/references` (or `--reference-store <dir>` from a previous wave). A wave computes only
the missing or stale entries. `equivalence.json` records `reference_outputs`: `computed` or `reused`, the fingerprint and the inputs that moved. It also records `reference_environment`: the family, its lock hash and the venv's freeze. A change to the pairs file, revision, lock, dtype, serve pin, client render field or unmanaged
interpreter invalidates the entry. An unchanged entry is reused.

Media rows use the same comparison. The five media families' references read the harness's `media` field in their embed and score modes. They are qwen3-vl-embedding, qwen3-vl-reranker, topk-embed-v1, pplx-embed-v2-late and embeddinggemma-2. An inline image is a loaded PIL image. A container is at the recipe's declared video pin. It goes through
the checkpoint's own video processor where the card's pipeline has one, as qwen3-vl-embedding's
`Qwen3VLVideoProcessor` route does. Otherwise it goes through the card's own loader. The interleaved parts stay
in order. The reference computes the card's own vectors (per token, with the card's declared keep-rule) or
rerank scores for them. Thus stage 2 gates their media rows exactly like the text rows. A recipe whose card's
own pipeline genuinely cannot run an input keeps the declaration. `reference.known_deviations: [media_approximation]` reports its media rows as non-gating with the reason. The media stage still gates
placement, geometry and tokens.

The wave's recipes are grouped by their `engine.image` before submission. There is one job per image. Thus a
digest-pinned nightly recipe runs on its own image without a change to the wave list. The pairs files under
`packages/rcp-test/pairs/` are the fixed request sets.

## Fakes and conformance on CPU afterwards

GPU work produces observation corpora. It never runs under pytest. Afterwards, model-level fakes rebuilt from
the recordings verify the CPU side. There is one conformance suite with two targets: the live engine and a
recipe-level fake engine. Both run through the product's role clients, never raw HTTP and never a copy of the
client. The verification also runs golden replays and a forward-compatibility run. Those suites live in the
unpublished contributor package (`rcp_test`) and run per its README.

## The release-candidate loop

Before the tag, build the wheelhouse and constraints of the candidate (the release-candidates page). Run the
fixed waves. Fix what fails. Re-run on GPU only what a fix changed. Repeat until every recipe's `status:` reads
`verified`. The tag ships no `unverified` recipe. A variant whose gate failed ships only when the owner's
explicit disposition list names it. For 0.0.1, that list is exactly the two `topk-embed-v1` variants.
