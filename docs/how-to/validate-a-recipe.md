# Validate a recipe on GPUs

A recipe ships only after it has been validated end to end on GPU against its reference; `status.state` records
the outcome (`unverified`: written, not yet checked; `verified`: the harness passed every gate; `failed`; with
the engine `image`, the `date` and the report beside it). This page is what runs before that flip: the waves,
their tiers and what each one proves. The node-level procedure (the staged wheels, the host checks) is
[release candidates and the GPU waves](release-candidates.md); the recipe format is [add a serving
recipe](add-a-model.md).

## Three environments on one node

The engine, the client and each reference never share an environment:

1. **The engine environment** is the image's own Python (the stock `vllm/vllm-openai` image, or a recipe's
digest-pinned nightly when the release image lacks its architecture), untouched except
   the one pure-Python wheel installed with `--no-deps` ([serve a retrieval model](serve-a-model.md)): a
   `pip freeze` before and after the install differs by exactly that wheel.
2. **The client environment** is a fresh venv holding the release's staged wheels constrained by the release's
   constraints file. It runs the equivalence harness, the recorder, the command line and the wave tooling over
   HTTP only -- no torch.
3. **The reference environment** is one venv **per family** (owner decision 35), built from the family's
   `reference.lock`: `--system-site-packages` over the image's torch/CUDA with the family's own pins
   installed into the venv, or a venv of its own for a family declaring `# own-torch: true`. The bootstrap
   builds each needed family once per pod and reuses it across the family's variants; an import check
   (torch sees the image's build, every pinned distribution is installed at its pin and imports) fails
   loudly with the family named. The reference runs as a subprocess of that python, and stage 2 stores its
   outputs so an unchanged one is reused (below).

## The waves

- **T0 -- smoke.** The engine is up from the recipe's own serve argv: `GET /v1/models` answers, one request per
  role route scores, and the engine version and fingerprint are recorded with the report.
- **T1 -- observations.** The recorded corpus: inputs the reference and the engine both see, whose outputs the
  equivalence checks and the later CPU fakes are built from.
- **T2 -- equivalence.** Served outputs against the reference implementation -- token-id equality and the
  insertion checks on the CPU side first, then the scored pairs on GPU, with the engine's token counts checked
  against the client's. A media recipe's image/video rows gate here too: the reference's embed/score mode
  computes their vectors (per token, with the card's declared keep-rule) or scores from the harness's `media`
  field. A recipe whose reference deliberately drops anchors declares the deviation and is
  compared under the cap only ([the equivalence policy](../reference/recipes.md#equivalence-policy)); a recipe
  that declares Matryoshka dimensions is gated per declared `k` ex-post from one full-width served pass
  ([gating every declared k](../concepts/matryoshka.md#gating-every-declared-k)).
- **T3 -- quality.** The MTEB suites: the served model's rankings scored with the released gains, against the
  thresholds the model card claims.
- **T4 -- end to end.** The served model as a run's step, with a served judge, through the product's clients.

The submitted wave ([release candidates and the GPU waves](release-candidates.md#which-gates-a-submitted-wave-runs))
runs T0, T2 and the recorder; T1's observation corpus, T3 and the negative controls are operator-run against
the same RC with the flags named on that page, and T4 is its own `--script e2e` job. A recipe's `status`
flips to `verified` only from a wave whose document records the gates it needs.

## The wave runner: step budgets, the pod log and partial results

A wave serves several recipes on one node at once; each recipe's steps run in a worker of their own, so a
stuck request never holds another recipe's steps (the first GPU wave lost 40 minutes behind one stuck
request). Every step has a **declared wall-clock budget**, computed by the runner from the recipe's
request count (a base plus one allowance per request; the pairs file's rows are the count). A recipe that
legitimately needs longer raises it with `engine.step_budget_s` (a floor: the runner's formula can only
raise the budget, never lower it).

When a step outruns its budget, the runner cancels the request in flight, stops that recipe's engine, and
fails the step with the request named:

```
step equivalence exceeded 2400s; in flight: POST /rerank (request 3)
```

The other recipes continue. The engine's own death (a crash, an OOM) fails only that recipe's `serve`
step: the engine's last log lines are in `<out>/<recipe>/serve.log`, and a clipped tail of them rides in
the status document. The harness's own requests (smoke, record, the corpus's bare probes) run with one
declared per-request timeout, shorter than every step budget and reported in the step documents.

The pod log gets one line per step boundary -- no request bodies, no environment values:

```
run_wave: jina-reranker-v3 smoke start
run_wave: jina-reranker-v3 smoke passed 0.4s
run_wave: jina-reranker-v3 equivalence start
run_wave: jina-reranker-v3 equivalence passed 812.1s
```

Results do not wait for the pod to end: `<out>/<recipe>/status.json` is rewritten atomically after every
step, and with `--upload` each finished recipe's directory is copied to the output URI the moment the
recipe ends, so a cancelled or killed pod still leaves the evidence of everything that finished. Every
upload is verified against the destination and retried with backoff, and its outcome is recorded in the
recipe's status row; the wave summary (`wave.json`/`WAVE.md`) is written before the last upload, so it
reaches the URI too, and a wave with a failed upload does not report PASS.

The reference subprocess gets **a GPU of its own** beside the engine's (never the engine's GPU, which
holds most of its memory); the runner reserves it when packing, so a node packs fewer engines per wave
(8 GPUs: at most 4 single-GPU recipes when each needs a reference GPU). The device and the GPU index are
recorded in `equivalence.json` (`device`, `reference_gpu`); a recipe whose reference cannot run on CPU
declares `reference.device: cuda` in its recipe file, and a CPU reference run for it is refused with the
way out (a pod that cannot spare the GPU fails that recipe early, never silently on CPU).

Stage 2's reference outputs are **stored** (owner decision 35): keyed by the family reference hash, the
variant revision, the pairs-file hash, the environment lock hash, the device and the dtype, under
`<out>/references` (or `--reference-store <dir>` from a previous wave). A wave computes only the missing
or stale entries; `equivalence.json` records `reference_outputs` (`computed` or `reused`, the fingerprint
and the inputs that moved) and `reference_environment` (the family, its lock hash and the venv's freeze).
A changed pairs file, revision, lock or dtype invalidates; an unchanged one reuses.  Media rows ride the
same comparison: the five media families' references (qwen3-vl-embedding, qwen3-vl-reranker,
topk-embed-v1, pplx-embed-v2-late, embeddinggemma-2) read the harness's `media` field in their embed and
score modes -- an inline image as a loaded PIL image, a container through the card's own loader at the
recipe's declared video pin, the interleaved parts in order -- and compute the card's own vectors (per
token, with the card's declared keep-rule) or rerank scores for them, so stage 2 gates their media rows
exactly like the text rows.  A recipe whose card's own pipeline genuinely cannot run an input keeps the
declaration: `reference.known_deviations: [media_approximation]` reports its media rows non-gating with the
reason (the media stage still gates placement, geometry and tokens).

The wave's recipes are grouped by their `engine.image` before submission (owner decisions 38/35): one job
per image, so a digest-pinned nightly recipe runs on its own image without changing the wave list. The
pairs files under `rcp-ndcg-test/pairs/` are the fixed request sets.

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
